"""Astra DB vector store integration."""

from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from operator import itemgetter
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    Sequence,
    TypeVar,
)

import numpy as np
from astrapy.exceptions import InsertManyException
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from langchain_core.runnables.utils import gather_with_concurrency
from langchain_core.vectorstores import VectorStore
from typing_extensions import override

from langchain_astradb.utils.astradb import (
    COMPONENT_NAME_VECTORSTORE,
    DEFAULT_DOCUMENT_CHUNK_SIZE,
    MAX_CONCURRENT_DOCUMENT_DELETIONS,
    MAX_CONCURRENT_DOCUMENT_INSERTIONS,
    MAX_CONCURRENT_DOCUMENT_REPLACEMENTS,
    SetupMode,
    _AstraDBCollectionEnvironment,
    _survey_collection,
)
from langchain_astradb.utils.vector_store_autodetect import (
    _detect_document_codec,
)
from langchain_astradb.utils.vector_store_codecs import (
    _AstraDBVectorStoreDocumentCodec,
    _DefaultVectorizeVSDocumentCodec,
    _DefaultVSDocumentCodec,
)

if TYPE_CHECKING:
    from astrapy.authentication import EmbeddingHeadersProvider, TokenProvider
    from astrapy.db import (
        AstraDB as AstraDBClient,
    )
    from astrapy.db import (
        AsyncAstraDB as AsyncAstraDBClient,
    )
    from astrapy.info import CollectionVectorServiceOptions
    from astrapy.results import UpdateResult
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings

T = TypeVar("T")
U = TypeVar("U")
DocDict = Dict[str, Any]  # dicts expressing entries to insert

# indexing options when creating a collection
DEFAULT_INDEXING_OPTIONS = {"allow": ["metadata"]}
# error code to check for during bulk insertions
DOCUMENT_ALREADY_EXISTS_API_ERROR_CODE = "DOCUMENT_ALREADY_EXISTS"

logger = logging.getLogger(__name__)


def _unique_list(lst: list[T], key: Callable[[T], U]) -> list[T]:
    visited_keys: set[U] = set()
    new_lst = []
    for item in lst:
        item_key = key(item)
        if item_key not in visited_keys:
            visited_keys.add(item_key)
            new_lst.append(item)
    return new_lst


def _normalize_content_field(
    content_field: str | None,
    *,
    is_autodetect: bool,
    has_vectorize: bool,
) -> str:
    if has_vectorize:
        if content_field is not None:
            msg = "content_field is not configurable for vectorize collections."
            raise ValueError(msg)
        return "$vectorize"

    if content_field is None:
        return "*" if is_autodetect else "content"

    if content_field == "*":
        if not is_autodetect:
            msg = "content_field='*' illegal if autodetect_collection is False."
            raise ValueError(msg)
        return content_field

    return content_field


def _validate_autodetect_init_params(
    *,
    metric: str | None = None,
    setup_mode: SetupMode | None,
    pre_delete_collection: bool,
    metadata_indexing_include: Iterable[str] | None,
    metadata_indexing_exclude: Iterable[str] | None,
    collection_indexing_policy: dict[str, Any] | None,
    collection_vector_service_options: CollectionVectorServiceOptions | None,
) -> None:
    """Check that the passed parameters do not violate the autodetect constraints."""
    forbidden_parameters = [
        p_name
        for p_name, p_value in (
            ("metric", metric),
            ("metadata_indexing_include", metadata_indexing_include),
            ("metadata_indexing_exclude", metadata_indexing_exclude),
            ("collection_indexing_policy", collection_indexing_policy),
            ("collection_vector_service_options", collection_vector_service_options),
        )
        if p_value is not None
    ]
    fp_error: str | None = None
    if forbidden_parameters:
        fp_error = (
            f"Parameter(s) {', '.join(forbidden_parameters)}. were provided "
            "but cannot be passed."
        )
    pd_error: str | None = None
    if pre_delete_collection:
        pd_error = "Parameter `pre_delete_collection` cannot be True."
    sm_error: str | None = None
    if setup_mode is not None:
        sm_error = "Parameter `setup_mode` not allowed."
    am_errors = [err_s for err_s in (fp_error, pd_error, sm_error) if err_s is not None]
    if am_errors:
        msg = f"Invalid parameters for autodetect mode: {'; '.join(am_errors)}"
        raise ValueError(msg)


class AstraDBVectorStore(VectorStore):
    """AstraDB vector store integration.

    Setup:
        Install the ``langchain-astradb`` package and head to the
        `AstraDB website <https://astra.datastax.com>`_, create an account, create a
        new database and `create an application token <https://docs.datastax.com/en/astra-db-serverless/administration/manage-application-tokens.html>`_.

        .. code-block:: bash

            pip install -qU langchain-astradb

    Key init args — indexing params:
        collection_name: str
            Name of the collection.
        embedding: Embeddings
            Embedding function to use.

    Key init args — client params:
        api_endpoint: str
            AstraDB API endpoint.
        token: str
            API token for Astra DB usage.
        namespace: Optional[str]
            Namespace (aka keyspace) where the collection is created

    Instantiate:
        Get your API endpoint and application token from the dashboard of your database.

        .. code-block:: python

            import getpass
            from langchain_astradb import AstraDBVectorStore
            from langchain_openai import OpenAIEmbeddings

            ASTRA_DB_API_ENDPOINT = getpass.getpass("ASTRA_DB_API_ENDPOINT = ")
            ASTRA_DB_APPLICATION_TOKEN = getpass.getpass("ASTRA_DB_APPLICATION_TOKEN = ")

            vector_store = AstraDBVectorStore(
                collection_name="astra_vector_langchain",
                embedding=OpenAIEmbeddings(),
                api_endpoint=ASTRA_DB_API_ENDPOINT,
                token=ASTRA_DB_APPLICATION_TOKEN,
            )

        Have the vector store figure out its configuration (documents scheme on DB)
        from an existing collection, in the case of `server-side-embeddings <https://docs.datastax.com/en/astra-db-serverless/databases/embedding-generation.html>`_:

        .. code-block:: python

            import getpass
            from langchain_astradb import AstraDBVectorStore

            ASTRA_DB_API_ENDPOINT = getpass.getpass("ASTRA_DB_API_ENDPOINT = ")
            ASTRA_DB_APPLICATION_TOKEN = getpass.getpass("ASTRA_DB_APPLICATION_TOKEN = ")

            vector_store = AstraDBVectorStore(
                collection_name="astra_vector_langchain",
                api_endpoint=ASTRA_DB_API_ENDPOINT,
                token=ASTRA_DB_APPLICATION_TOKEN,
                autodetect_collection=True,
            )

    Add Documents:
        .. code-block:: python

            from langchain_core.documents import Document

            document_1 = Document(page_content="foo", metadata={"baz": "bar"})
            document_2 = Document(page_content="thud", metadata={"bar": "baz"})
            document_3 = Document(page_content="i will be deleted :(")

            documents = [document_1, document_2, document_3]
            ids = ["1", "2", "3"]
            vector_store.add_documents(documents=documents, ids=ids)

    Delete Documents:
        .. code-block:: python

            vector_store.delete(ids=["3"])

    Search:
        .. code-block:: python

            results = vector_store.similarity_search(query="thud",k=1)
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: none

            thud [{'bar': 'baz'}]

    Search with filter:
        .. code-block:: python

            results = vector_store.similarity_search(query="thud",k=1,filter={"bar": "baz"})
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: none

            thud [{'bar': 'baz'}]

    Search with score:
        .. code-block:: python

            results = vector_store.similarity_search_with_score(query="qux",k=1)
            for doc, score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

        .. code-block:: none

            [SIM=0.916135] foo [{'baz': 'bar'}]

    Async:
        .. code-block:: python

            # add documents
            await vector_store.aadd_documents(documents=documents, ids=ids)

            # delete documents
            await vector_store.adelete(ids=["3"])

            # search
            results = vector_store.asimilarity_search(query="thud",k=1)

            # search with score
            results = await vector_store.asimilarity_search_with_score(query="qux",k=1)
            for doc,score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

        .. code-block:: none

            [SIM=0.916135] foo [{'baz': 'bar'}]

    Use as Retriever:
        .. code-block:: python

            retriever = vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 1, "score_threshold": 0.5},
            )
            retriever.invoke("thud")

        .. code-block:: none

            [Document(metadata={'bar': 'baz'}, page_content='thud')]

    """  # noqa: E501

    def filter_to_query(self, filter_dict: dict[str, Any] | None) -> dict[str, Any]:
        """Prepare a query for use on DB based on metadata filter.

        Encode an "abstract" filter clause on metadata into a query filter
        condition aware of the collection schema choice.

        Args:
            filter_dict: a metadata condition in the form {"field": "value"}
                or related.

        Returns:
            the corresponding mapping ready for use in queries,
            aware of the details of the schema used to encode the document on DB.
        """
        if filter_dict is None:
            return {}

        return self.document_codec.encode_filter(filter_dict)

    @staticmethod
    def _normalize_metadata_indexing_policy(
        metadata_indexing_include: Iterable[str] | None,
        metadata_indexing_exclude: Iterable[str] | None,
        collection_indexing_policy: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Normalize the constructor indexing parameters.

        Validate the constructor indexing parameters and normalize them
        into a ready-to-use dict for the 'options' when creating a collection.
        """
        params = [
            metadata_indexing_include,
            metadata_indexing_exclude,
            collection_indexing_policy,
        ]
        if params.count(None) < len(params) - 1:
            msg = (
                "At most one of the parameters `metadata_indexing_include`,"
                " `metadata_indexing_exclude` and `collection_indexing_policy`"
                " can be specified as non null."
            )
            raise ValueError(msg)

        if metadata_indexing_include is not None:
            return {
                "allow": [
                    f"metadata.{md_field}" for md_field in metadata_indexing_include
                ]
            }
        if metadata_indexing_exclude is not None:
            return {
                "deny": [
                    f"metadata.{md_field}" for md_field in metadata_indexing_exclude
                ]
            }
        return (
            collection_indexing_policy
            if collection_indexing_policy is not None
            else DEFAULT_INDEXING_OPTIONS
        )

    def __init__(
        self,
        *,
        collection_name: str,
        embedding: Embeddings | None = None,
        token: str | TokenProvider | None = None,
        api_endpoint: str | None = None,
        environment: str | None = None,
        namespace: str | None = None,
        metric: str | None = None,
        batch_size: int | None = None,
        bulk_insert_batch_concurrency: int | None = None,
        bulk_insert_overwrite_concurrency: int | None = None,
        bulk_delete_concurrency: int | None = None,
        setup_mode: SetupMode | None = None,
        pre_delete_collection: bool = False,
        metadata_indexing_include: Iterable[str] | None = None,
        metadata_indexing_exclude: Iterable[str] | None = None,
        collection_indexing_policy: dict[str, Any] | None = None,
        collection_vector_service_options: CollectionVectorServiceOptions | None = None,
        collection_embedding_api_key: str | EmbeddingHeadersProvider | None = None,
        content_field: str | None = None,
        ignore_invalid_documents: bool = False,
        autodetect_collection: bool = False,
        ext_callers: list[tuple[str | None, str | None] | str | None] | None = None,
        component_name: str = COMPONENT_NAME_VECTORSTORE,
        astra_db_client: AstraDBClient | None = None,
        async_astra_db_client: AsyncAstraDBClient | None = None,
    ) -> None:
        """Wrapper around DataStax Astra DB for vector-store workloads.

        For quickstart and details, visit
        https://docs.datastax.com/en/astra-db-serverless/index.html

        Args:
            embedding: the embeddings function or service to use.
                This enables client-side embedding functions or calls to external
                embedding providers. If ``embedding`` is provided, arguments
                ``collection_vector_service_options`` and
                ``collection_embedding_api_key`` cannot be provided.
            collection_name: name of the Astra DB collection to create/use.
            token: API token for Astra DB usage, either in the form of a string
                or a subclass of ``astrapy.authentication.TokenProvider``.
                If not provided, the environment variable
                ASTRA_DB_APPLICATION_TOKEN is inspected.
            api_endpoint: full URL to the API endpoint, such as
                ``https://<DB-ID>-us-east1.apps.astra.datastax.com``. If not provided,
                the environment variable ASTRA_DB_API_ENDPOINT is inspected.
            environment: a string specifying the environment of the target Data API.
                If omitted, defaults to "prod" (Astra DB production).
                Other values are in ``astrapy.constants.Environment`` enum class.
            namespace: namespace (aka keyspace) where the collection is created.
                If not provided, the environment variable ASTRA_DB_KEYSPACE is
                inspected. Defaults to the database's "default namespace".
            metric: similarity function to use out of those available in Astra DB.
                If left out, it will use Astra DB API's defaults (i.e. "cosine" - but,
                for performance reasons, "dot_product" is suggested if embeddings are
                normalized to one).
            batch_size: Size of document chunks for each individual insertion
                API request. If not provided, astrapy defaults are applied.
            bulk_insert_batch_concurrency: Number of threads or coroutines to insert
                batches concurrently.
            bulk_insert_overwrite_concurrency: Number of threads or coroutines in a
                batch to insert pre-existing entries.
            bulk_delete_concurrency: Number of threads or coroutines for
                multiple-entry deletes.
            setup_mode: mode used to create the collection (SYNC, ASYNC or OFF).
            pre_delete_collection: whether to delete the collection before creating it.
                If False and the collection already exists, the collection will be used
                as is.
            metadata_indexing_include: an allowlist of the specific metadata subfields
                that should be indexed for later filtering in searches.
            metadata_indexing_exclude: a denylist of the specific metadata subfields
                that should not be indexed for later filtering in searches.
            collection_indexing_policy: a full "indexing" specification for
                what fields should be indexed for later filtering in searches.
                This dict must conform to to the API specifications
                (see https://docs.datastax.com/en/astra-db-serverless/api-reference/collections.html#the-indexing-option)
            collection_vector_service_options: specifies the use of server-side
                embeddings within Astra DB. If passing this parameter, ``embedding``
                cannot be provided.
            collection_embedding_api_key: for usage of server-side embeddings
                within Astra DB. With this parameter one can supply an API Key
                that will be passed to Astra DB with each data request.
                This parameter can be either a string or a subclass of
                ``astrapy.authentication.EmbeddingHeadersProvider``.
                This is useful when the service is configured for the collection,
                but no corresponding secret is stored within
                Astra's key management system.
            content_field: name of the field containing the textual content
                in the documents when saved on Astra DB. For vectorize collections,
                this cannot be specified; for non-vectorize collection, defaults
                to "content".
                The special value "*" can be passed only if autodetect_collection=True.
                In this case, the actual name of the key for the textual content is
                guessed by inspection of a few documents from the collection, under the
                assumption that the longer strings are the most likely candidates.
                Please understand the limitations of this method and get some
                understanding of your data before passing ``"*"`` for this parameter.
            ignore_invalid_documents: if False (default), exceptions are raised
                when a document is found on the Astra DB collection that does
                not have the expected shape. If set to True, such results
                from the database are ignored and a warning is issued. Note
                that in this case a similarity search may end up returning fewer
                results than the required ``k``.
            autodetect_collection: if True, turns on autodetect behavior.
                The store will look for an existing collection of the provided name
                and infer the store settings from it. Default is False.
                In autodetect mode, ``content_field`` can be given as ``"*"``, meaning
                that an attempt will be made to determine it by inspection (unless
                vectorize is enabled, in which case ``content_field`` is ignored).
                In autodetect mode, the store not only determines whether embeddings
                are client- or server-side, but - most importantly - switches
                automatically between "nested" and "flat" representations of documents
                on DB (i.e. having the metadata key-value pairs grouped in a
                ``metadata`` field or spread at the documents' top-level). The former
                scheme is the native mode of the AstraDBVectorStore; the store resorts
                to the latter in case of vector collections populated with external
                means (such as a third-party data import tool) before applying
                an AstraDBVectorStore to them.
                Note that the following parameters cannot be used if this is True:
                ``metric``, ``setup_mode``, ``metadata_indexing_include``,
                ``metadata_indexing_exclude``, ``collection_indexing_policy``,
                ``collection_vector_service_options``.
            ext_callers: one or more caller identities to identify Data API calls
                in the User-Agent header. This is a list of (name, version) pairs,
                or just strings if no version info is provided, which, if supplied,
                becomes the leading part of the User-Agent string in all API requests
                related to this component.
            component_name: the string identifying this specific component in the
                stack of usage info passed as the User-Agent string to the Data API.
                Defaults to "langchain_vectorstore", but can be overridden if this
                component actually serves as the building block for another component
                (such as a Graph Vector Store).
            astra_db_client:
                *DEPRECATED starting from version 0.3.5.*
                *Please use 'token', 'api_endpoint' and optionally 'environment'.*
                you can pass an already-created 'astrapy.db.AstraDB' instance
                (alternatively to 'token', 'api_endpoint' and 'environment').
            async_astra_db_client:
                *DEPRECATED starting from version 0.3.5.*
                *Please use 'token', 'api_endpoint' and optionally 'environment'.*
                you can pass an already-created 'astrapy.db.AsyncAstraDB' instance
                (alternatively to 'token', 'api_endpoint' and 'environment').

        Note:
            For concurrency in synchronous :meth:`~add_texts`:, as a rule of thumb,
            on a typical client machine it is suggested to keep the quantity
            bulk_insert_batch_concurrency * bulk_insert_overwrite_concurrency
            much below 1000 to avoid exhausting the client multithreading/networking
            resources. The hardcoded defaults are somewhat conservative to meet
            most machines' specs, but a sensible choice to test may be:

            - bulk_insert_batch_concurrency = 80
            - bulk_insert_overwrite_concurrency = 10

            A bit of experimentation is required to nail the best results here,
            depending on both the machine/network specs and the expected workload
            (specifically, how often a write is an update of an existing id).
            Remember you can pass concurrency settings to individual calls to
            :meth:`~add_texts` and :meth:`~add_documents` as well.
        """
        # general collection settings
        self.collection_name = collection_name
        self.token = token
        self.api_endpoint = api_endpoint
        self.environment = environment
        self.namespace = namespace
        self.indexing_policy: dict[str, Any]
        self.autodetect_collection = autodetect_collection
        # vector-related settings
        self.embedding_dimension: int | None = None
        self.embedding = embedding
        self.metric = metric
        self.collection_embedding_api_key = collection_embedding_api_key
        self.collection_vector_service_options = collection_vector_service_options
        # DB-encoding settings:
        self.document_codec: _AstraDBVectorStoreDocumentCodec
        # concurrency settings
        self.batch_size: int | None = batch_size or DEFAULT_DOCUMENT_CHUNK_SIZE
        self.bulk_insert_batch_concurrency: int = (
            bulk_insert_batch_concurrency or MAX_CONCURRENT_DOCUMENT_INSERTIONS
        )
        self.bulk_insert_overwrite_concurrency: int = (
            bulk_insert_overwrite_concurrency or MAX_CONCURRENT_DOCUMENT_REPLACEMENTS
        )
        self.bulk_delete_concurrency: int = (
            bulk_delete_concurrency or MAX_CONCURRENT_DOCUMENT_DELETIONS
        )

        _setup_mode: SetupMode
        _embedding_dimension: int | Awaitable[int] | None

        if not self.autodetect_collection:
            logger.info(
                "vector store default init, collection '%s'", self.collection_name
            )
            _setup_mode = SetupMode.SYNC if setup_mode is None else setup_mode
            _embedding_dimension = self._prepare_embedding_dimension(_setup_mode)
            # determine vectorize/nonvectorize
            has_vectorize = self.collection_vector_service_options is not None
            _content_field = _normalize_content_field(
                content_field,
                is_autodetect=False,
                has_vectorize=has_vectorize,
            )

            if self.collection_vector_service_options is not None:
                self.document_codec = _DefaultVectorizeVSDocumentCodec(
                    ignore_invalid_documents=ignore_invalid_documents,
                )
            else:
                self.document_codec = _DefaultVSDocumentCodec(
                    content_field=_content_field,
                    ignore_invalid_documents=ignore_invalid_documents,
                )
            # indexing policy setting
            self.indexing_policy = self._normalize_metadata_indexing_policy(
                metadata_indexing_include=metadata_indexing_include,
                metadata_indexing_exclude=metadata_indexing_exclude,
                collection_indexing_policy=collection_indexing_policy,
            )
        else:
            logger.info(
                "vector store autodetect init, collection '%s'", self.collection_name
            )
            # specific checks for autodetect logic
            _validate_autodetect_init_params(
                metric=self.metric,
                setup_mode=setup_mode,
                pre_delete_collection=pre_delete_collection,
                metadata_indexing_include=metadata_indexing_include,
                metadata_indexing_exclude=metadata_indexing_exclude,
                collection_indexing_policy=collection_indexing_policy,
                collection_vector_service_options=self.collection_vector_service_options,
            )
            _setup_mode = SetupMode.OFF

            # fetch collection intelligence
            c_descriptor, c_documents = _survey_collection(
                collection_name=self.collection_name,
                token=self.token,
                api_endpoint=self.api_endpoint,
                keyspace=self.namespace,
                environment=self.environment,
                ext_callers=ext_callers,
                component_name=component_name,
                astra_db_client=astra_db_client,
                async_astra_db_client=async_astra_db_client,
            )
            if c_descriptor is None:
                msg = f"Collection '{self.collection_name}' not found."
                raise ValueError(msg)
            # use the collection info to set the store properties
            self.indexing_policy = self._normalize_metadata_indexing_policy(
                metadata_indexing_include=None,
                metadata_indexing_exclude=None,
                collection_indexing_policy=c_descriptor.options.indexing,
            )
            if c_descriptor.options.vector is None:
                msg = "Non-vector collection detected."
                raise ValueError(msg)
            _embedding_dimension = c_descriptor.options.vector.dimension
            self.collection_vector_service_options = c_descriptor.options.vector.service
            has_vectorize = self.collection_vector_service_options is not None
            logger.info("vector store autodetect: has_vectorize = %s", has_vectorize)
            norm_content_field = _normalize_content_field(
                content_field,
                is_autodetect=True,
                has_vectorize=has_vectorize,
            )
            self.document_codec = _detect_document_codec(
                c_documents,
                has_vectorize=has_vectorize,
                ignore_invalid_documents=ignore_invalid_documents,
                norm_content_field=norm_content_field,
            )

        # validate embedding/vectorize compatibility and such.
        # Embedding and the server-side embeddings are mutually exclusive,
        # as both specify how to produce embeddings.
        # Also API key makes no sense unless vectorize.
        if self.embedding is None and not self.document_codec.server_side_embeddings:
            msg = "Embedding is required for non-vectorize collections."
            raise ValueError(msg)

        if self.embedding is not None and self.document_codec.server_side_embeddings:
            msg = "Embedding cannot be provided for vectorize collections."
            raise ValueError(msg)

        if (
            not self.document_codec.server_side_embeddings
            and self.collection_embedding_api_key is not None
        ):
            msg = "Embedding API Key cannot be provided for non-vectorize collections."
            raise ValueError(msg)

        self.astra_env = _AstraDBCollectionEnvironment(
            collection_name=collection_name,
            token=self.token,
            api_endpoint=self.api_endpoint,
            keyspace=self.namespace,
            environment=self.environment,
            setup_mode=_setup_mode,
            pre_delete_collection=pre_delete_collection,
            embedding_dimension=_embedding_dimension,
            metric=self.metric,
            requested_indexing_policy=self.indexing_policy,
            default_indexing_policy=DEFAULT_INDEXING_OPTIONS,
            collection_vector_service_options=self.collection_vector_service_options,
            collection_embedding_api_key=self.collection_embedding_api_key,
            ext_callers=ext_callers,
            component_name=component_name,
            astra_db_client=astra_db_client,
            async_astra_db_client=async_astra_db_client,
        )

    def _get_safe_embedding(self) -> Embeddings:
        if not self.embedding:
            msg = "Missing embedding"
            raise ValueError(msg)
        return self.embedding

    def _prepare_embedding_dimension(
        self, setup_mode: SetupMode
    ) -> int | Awaitable[int] | None:
        """Return the right kind of object for the astra_env to use."""
        if self.embedding is None:
            return None
        if setup_mode == SetupMode.ASYNC:
            # in this case, we wrap the computation as an awaitable
            async def _aget_embedding_dimension() -> int:
                if self.embedding_dimension is None:
                    self.embedding_dimension = len(
                        await self._get_safe_embedding().aembed_query(
                            text="This is a sample sentence."
                        )
                    )
                return self.embedding_dimension

            return _aget_embedding_dimension()
        # case of setup_mode = SetupMode.SYNC, SetupMode.OFF
        if self.embedding_dimension is None:
            self.embedding_dimension = len(
                self._get_safe_embedding().embed_query(
                    text="This is a sample sentence."
                )
            )
        return self.embedding_dimension

    @property
    @override
    def embeddings(self) -> Embeddings | None:
        """Accesses the supplied embeddings object.

        If using server-side embeddings, this will return None.
        """
        return self.embedding

    @override
    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        # The underlying API calls already returns a "score proper",
        # i.e. one in [0, 1] where higher means more *similar*,
        # so here the final score transformation is not reversing the interval.
        return lambda score: score

    def clear(self) -> None:
        """Empty the collection of all its stored entries."""
        self.astra_env.ensure_db_setup()
        self.astra_env.collection.delete_many({})

    async def aclear(self) -> None:
        """Empty the collection of all its stored entries."""
        await self.astra_env.aensure_db_setup()
        await self.astra_env.async_collection.delete_many({})

    def delete_by_document_id(self, document_id: str) -> bool:
        """Remove a single document from the store, given its document ID.

        Args:
            document_id: The document ID

        Returns:
            True if a document has indeed been deleted, False if ID not found.
        """
        self.astra_env.ensure_db_setup()
        # self.collection is not None (by _ensure_astra_db_client)
        deletion_response = self.astra_env.collection.delete_one({"_id": document_id})
        return deletion_response.deleted_count == 1

    async def adelete_by_document_id(self, document_id: str) -> bool:
        """Remove a single document from the store, given its document ID.

        Args:
            document_id: The document ID

        Returns:
            True if a document has indeed been deleted, False if ID not found.
        """
        await self.astra_env.aensure_db_setup()
        deletion_response = await self.astra_env.async_collection.delete_one(
            {"_id": document_id},
        )
        return deletion_response.deleted_count == 1

    @override
    def delete(
        self,
        ids: list[str] | None = None,
        concurrency: int | None = None,
        **kwargs: Any,
    ) -> bool | None:
        """Delete by vector ids.

        Args:
            ids: List of ids to delete.
            concurrency: max number of threads issuing single-doc delete requests.
                Defaults to vector-store overall setting.
            **kwargs: Additional arguments are ignored.

        Returns:
            True if deletion is (entirely) successful, False otherwise.
        """
        if kwargs:
            warnings.warn(
                "Method 'delete' of AstraDBVectorStore vector store invoked with "
                f"unsupported arguments ({', '.join(sorted(kwargs.keys()))}), "
                "which will be ignored.",
                stacklevel=2,
            )

        if ids is None:
            msg = "No ids provided to delete."
            raise ValueError(msg)

        _max_workers = concurrency or self.bulk_delete_concurrency
        with ThreadPoolExecutor(max_workers=_max_workers) as tpe:
            _ = list(
                tpe.map(
                    self.delete_by_document_id,
                    ids,
                )
            )
        return True

    @override
    async def adelete(
        self,
        ids: list[str] | None = None,
        concurrency: int | None = None,
        **kwargs: Any,
    ) -> bool | None:
        """Delete by vector ids.

        Args:
            ids: List of ids to delete.
            concurrency: max number of simultaneous coroutines for single-doc
                delete requests. Defaults to vector-store overall setting.
            **kwargs: Additional arguments are ignored.

        Returns:
            True if deletion is (entirely) successful, False otherwise.
        """
        if kwargs:
            warnings.warn(
                "Method 'adelete' of AstraDBVectorStore invoked with "
                f"unsupported arguments ({', '.join(sorted(kwargs.keys()))}), "
                "which will be ignored.",
                stacklevel=2,
            )

        if ids is None:
            msg = "No ids provided to delete."
            raise ValueError(msg)

        _max_workers = concurrency or self.bulk_delete_concurrency
        await gather_with_concurrency(
            _max_workers, *[self.adelete_by_document_id(doc_id) for doc_id in ids]
        )
        return True

    def delete_by_metadata_filter(
        self,
        filter: dict[str, Any],  # noqa: A002
    ) -> int:
        """Delete all documents matching a certain metadata filtering condition.

        This operation does not use the vector embeddings in any way, it simply
        removes all documents whose metadata match the provided condition.

        Args:
            filter: Filter on the metadata to apply. The filter cannot be empty.

        Returns:
            A number expressing the amount of deleted documents.
        """
        if not filter:
            msg = (
                "Method `delete_by_metadata_filter` does not accept an empty "
                "filter. Use the `clear()` method if you really want to empty "
                "the vector store."
            )
            raise ValueError(msg)
        self.astra_env.ensure_db_setup()
        metadata_parameter = self.filter_to_query(filter)
        del_result = self.astra_env.collection.delete_many(
            filter=metadata_parameter,
        )
        return del_result.deleted_count or 0

    async def adelete_by_metadata_filter(
        self,
        filter: dict[str, Any],  # noqa: A002
    ) -> int:
        """Delete all documents matching a certain metadata filtering condition.

        This operation does not use the vector embeddings in any way, it simply
        removes all documents whose metadata match the provided condition.

        Args:
            filter: Filter on the metadata to apply. The filter cannot be empty.

        Returns:
            A number expressing the amount of deleted documents.
        """
        if not filter:
            msg = (
                "Method `adelete_by_metadata_filter` does not accept an empty "
                "filter. Use the `aclear()` method if you really want to empty "
                "the vector store."
            )
            raise ValueError(msg)
        await self.astra_env.aensure_db_setup()
        metadata_parameter = self.filter_to_query(filter)
        del_result = await self.astra_env.async_collection.delete_many(
            filter=metadata_parameter,
        )
        return del_result.deleted_count or 0

    def delete_collection(self) -> None:
        """Completely delete the collection from the database.

        Completely delete the collection from the database (as opposed
        to :meth:`~clear`, which empties it only).
        Stored data is lost and unrecoverable, resources are freed.
        Use with caution.
        """
        self.astra_env.ensure_db_setup()
        self.astra_env.collection.drop()

    async def adelete_collection(self) -> None:
        """Completely delete the collection from the database.

        Completely delete the collection from the database (as opposed
        to :meth:`~aclear`, which empties it only).
        Stored data is lost and unrecoverable, resources are freed.
        Use with caution.
        """
        await self.astra_env.aensure_db_setup()
        await self.astra_env.async_collection.drop()

    def _get_documents_to_insert(
        self,
        texts: Iterable[str],
        embedding_vectors: Sequence[list[float] | None],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> list[DocDict]:
        if ids is None:
            ids = [uuid.uuid4().hex for _ in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]
        documents_to_insert = [
            self.document_codec.encode(
                content=b_txt,
                document_id=b_id,
                vector=b_emb,
                metadata=b_md,
            )
            for b_txt, b_emb, b_id, b_md in zip(
                texts,
                embedding_vectors,
                ids,
                metadatas,
            )
        ]
        # make unique by id, keeping the last
        return _unique_list(
            documents_to_insert[::-1],
            itemgetter("_id"),
        )[::-1]

    @staticmethod
    def _get_missing_from_batch(
        document_batch: list[DocDict], insert_result: dict[str, Any]
    ) -> tuple[list[str], list[DocDict]]:
        if "status" not in insert_result:
            msg = f"API Exception while running bulk insertion: {insert_result}"
            raise ValueError(msg)
        batch_inserted = insert_result["status"]["insertedIds"]
        # estimation of the preexisting documents that failed
        missed_inserted_ids = {document["_id"] for document in document_batch} - set(
            batch_inserted
        )
        errors = insert_result.get("errors", [])
        # careful for other sources of error other than "doc already exists"
        num_errors = len(errors)
        unexpected_errors = any(
            error.get("errorCode") != "DOCUMENT_ALREADY_EXISTS" for error in errors
        )
        if num_errors != len(missed_inserted_ids) or unexpected_errors:
            msg = f"API Exception while running bulk insertion: {errors}"
            raise ValueError(msg)
        # deal with the missing insertions as upserts
        missing_from_batch = [
            document
            for document in document_batch
            if document["_id"] in missed_inserted_ids
        ]
        return batch_inserted, missing_from_batch

    @override
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        *,
        batch_size: int | None = None,
        batch_concurrency: int | None = None,
        overwrite_concurrency: int | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Run texts through the embeddings and add them to the vectorstore.

        If passing explicit ids, those entries whose id is in the store already
        will be replaced.

        Args:
            texts: Texts to add to the vectorstore.
            metadatas: Optional list of metadatas.
            ids: Optional list of ids.
            batch_size: Size of document chunks for each individual insertion
                API request. If not provided, defaults to the vector-store
                overall defaults (which in turn falls to astrapy defaults).
            batch_concurrency: number of threads to process
                insertion batches concurrently. Defaults to the vector-store
                overall setting if not provided.
            overwrite_concurrency: number of threads to process
                pre-existing documents in each batch. Defaults to the vector-store
                overall setting if not provided.
            **kwargs: Additional arguments are ignored.

        Note:
            There are constraints on the allowed field names
            in the metadata dictionaries, coming from the underlying Astra DB API.
            For instance, the ``$`` (dollar sign) cannot be used in the dict keys.
            See this document for details:
            https://docs.datastax.com/en/astra-db-serverless/api-reference/overview.html#limits

        Returns:
            The list of ids of the added texts.
        """
        if kwargs:
            warnings.warn(
                "Method 'add_texts' of AstraDBVectorStore vector store invoked with "
                f"unsupported arguments ({', '.join(sorted(kwargs.keys()))}), "
                "which will be ignored.",
                stacklevel=2,
            )
        self.astra_env.ensure_db_setup()

        embedding_vectors: Sequence[list[float] | None]
        if self.document_codec.server_side_embeddings:
            embedding_vectors = [None for _ in list(texts)]
        else:
            embedding_vectors = self._get_safe_embedding().embed_documents(list(texts))
        documents_to_insert = self._get_documents_to_insert(
            texts, embedding_vectors, metadatas, ids
        )

        # perform an AstraPy insert_many, catching exceptions for overwriting docs
        ids_to_replace: list[int]
        inserted_ids: list[str] = []
        try:
            insert_many_result = self.astra_env.collection.insert_many(
                documents_to_insert,
                ordered=False,
                concurrency=batch_concurrency or self.bulk_insert_batch_concurrency,
                chunk_size=batch_size or self.batch_size,
            )
            ids_to_replace = []
            inserted_ids = insert_many_result.inserted_ids
        except InsertManyException as err:
            # check that the error is solely due to already-existing documents
            error_codes = {err_desc.error_code for err_desc in err.error_descriptors}
            if error_codes == {DOCUMENT_ALREADY_EXISTS_API_ERROR_CODE}:
                inserted_ids = err.partial_result.inserted_ids
                inserted_ids_set = set(inserted_ids)
                ids_to_replace = [
                    document["_id"]
                    for document in documents_to_insert
                    if document["_id"] not in inserted_ids_set
                ]
            else:
                raise

        # if necessary, replace docs for the non-inserted ids
        if ids_to_replace:
            documents_to_replace = [
                document
                for document in documents_to_insert
                if document["_id"] in ids_to_replace
            ]

            _max_workers = (
                overwrite_concurrency or self.bulk_insert_overwrite_concurrency
            )
            with ThreadPoolExecutor(
                max_workers=_max_workers,
            ) as executor:

                def _replace_document(
                    document: dict[str, Any],
                ) -> tuple[UpdateResult, str]:
                    return self.astra_env.collection.replace_one(
                        {"_id": document["_id"]},
                        document,
                    ), document["_id"]

                replace_results = list(
                    executor.map(
                        _replace_document,
                        documents_to_replace,
                    )
                )

            replaced_count = sum(r_res.update_info["n"] for r_res, _ in replace_results)
            inserted_ids += [replaced_id for _, replaced_id in replace_results]
            if replaced_count != len(ids_to_replace):
                missing = len(ids_to_replace) - replaced_count
                msg = (
                    "AstraDBVectorStore.add_texts could not insert all requested "
                    f"documents ({missing} failed replace_one calls)"
                )
                raise ValueError(msg)
        return inserted_ids

    @override
    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        *,
        batch_size: int | None = None,
        batch_concurrency: int | None = None,
        overwrite_concurrency: int | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Run texts through the embeddings and add them to the vectorstore.

        If passing explicit ids, those entries whose id is in the store already
        will be replaced.

        Args:
            texts: Texts to add to the vectorstore.
            metadatas: Optional list of metadatas.
            ids: Optional list of ids.
            batch_size: Size of document chunks for each individual insertion
                API request. If not provided, defaults to the vector-store
                overall defaults (which in turn falls to astrapy defaults).
            batch_concurrency: number of simultaneous coroutines to process
                insertion batches concurrently. Defaults to the vector-store
                overall setting if not provided.
            overwrite_concurrency: number of simultaneous coroutines to process
                pre-existing documents in each batch. Defaults to the vector-store
                overall setting if not provided.
            **kwargs: Additional arguments are ignored.

        Note:
            There are constraints on the allowed field names
            in the metadata dictionaries, coming from the underlying Astra DB API.
            For instance, the ``$`` (dollar sign) cannot be used in the dict keys.
            See this document for details:
            https://docs.datastax.com/en/astra-db-serverless/api-reference/overview.html#limits

        Returns:
            The list of ids of the added texts.
        """
        if kwargs:
            warnings.warn(
                "Method 'aadd_texts' of AstraDBVectorStore invoked with "
                f"unsupported arguments ({', '.join(sorted(kwargs.keys()))}), "
                "which will be ignored.",
                stacklevel=2,
            )
        await self.astra_env.aensure_db_setup()

        embedding_vectors: Sequence[list[float] | None]
        if self.document_codec.server_side_embeddings:
            embedding_vectors = [None for _ in list(texts)]
        else:
            embedding_vectors = await self._get_safe_embedding().aembed_documents(
                list(texts)
            )
        documents_to_insert = self._get_documents_to_insert(
            texts, embedding_vectors, metadatas, ids
        )

        # perform an AstraPy insert_many, catching exceptions for overwriting docs
        ids_to_replace: list[int]
        inserted_ids: list[str] = []
        try:
            insert_many_result = await self.astra_env.async_collection.insert_many(
                documents_to_insert,
                ordered=False,
                concurrency=batch_concurrency or self.bulk_insert_batch_concurrency,
                chunk_size=batch_size or self.batch_size,
            )
            ids_to_replace = []
            inserted_ids = insert_many_result.inserted_ids
        except InsertManyException as err:
            # check that the error is solely due to already-existing documents
            error_codes = {err_desc.error_code for err_desc in err.error_descriptors}
            if error_codes == {DOCUMENT_ALREADY_EXISTS_API_ERROR_CODE}:
                inserted_ids = err.partial_result.inserted_ids
                inserted_ids_set = set(inserted_ids)
                ids_to_replace = [
                    document["_id"]
                    for document in documents_to_insert
                    if document["_id"] not in inserted_ids_set
                ]
            else:
                raise

        # if necessary, replace docs for the non-inserted ids
        if ids_to_replace:
            documents_to_replace = [
                document
                for document in documents_to_insert
                if document["_id"] in ids_to_replace
            ]

            sem = asyncio.Semaphore(
                overwrite_concurrency or self.bulk_insert_overwrite_concurrency,
            )

            _async_collection = self.astra_env.async_collection

            async def _replace_document(
                document: dict[str, Any],
            ) -> tuple[UpdateResult, str]:
                async with sem:
                    return await _async_collection.replace_one(
                        {"_id": document["_id"]},
                        document,
                    ), document["_id"]

            tasks = [
                asyncio.create_task(_replace_document(document))
                for document in documents_to_replace
            ]

            replace_results = await asyncio.gather(*tasks, return_exceptions=False)

            replaced_count = sum(r_res.update_info["n"] for r_res, _ in replace_results)
            inserted_ids += [replaced_id for _, replaced_id in replace_results]

            if replaced_count != len(ids_to_replace):
                missing = len(ids_to_replace) - replaced_count
                msg = (
                    "AstraDBVectorStore.add_texts could not insert all requested "
                    f"documents ({missing} failed replace_one calls)"
                )
                raise ValueError(msg)
        return inserted_ids

    def update_metadata(
        self,
        id_to_metadata: dict[str, dict],
        *,
        overwrite_concurrency: int | None = None,
    ) -> int:
        """Add/overwrite the metadata of existing documents.

        For each document to update, the new metadata dictionary is appended
        to the existing metadata, overwriting individual keys that existed already.

        Args:
            id_to_metadata: map from the Document IDs to modify to the
                new metadata for updating. Keys in this dictionary that
                do not correspond to an existing document will be silently ignored.
                The values of this map are metadata dictionaries for updating
                the documents. Any pre-existing metadata will be merged with
                these entries, which take precedence on a key-by-key basis.
            overwrite_concurrency: number of threads to process the updates.
                Defaults to the vector-store overall setting if not provided.

        Returns:
            the number of documents successfully updated (i.e. found to exist,
            since even an update with `{}` as the new metadata counts as successful.)
        """
        self.astra_env.ensure_db_setup()

        _max_workers = overwrite_concurrency or self.bulk_insert_overwrite_concurrency
        with ThreadPoolExecutor(
            max_workers=_max_workers,
        ) as executor:

            def _update_document(
                id_md_pair: tuple[str, dict],
            ) -> UpdateResult:
                document_id, update_metadata = id_md_pair
                encoded_metadata = self.filter_to_query(update_metadata)
                return self.astra_env.collection.update_one(
                    {"_id": document_id},
                    {"$set": encoded_metadata},
                )

            update_results = list(
                executor.map(
                    _update_document,
                    id_to_metadata.items(),
                )
            )

        return sum(u_res.update_info["n"] for u_res in update_results)

    async def aupdate_metadata(
        self,
        id_to_metadata: dict[str, dict],
        *,
        overwrite_concurrency: int | None = None,
    ) -> int:
        """Add/overwrite the metadata of existing documents.

        For each document to update, the new metadata dictionary is appended
        to the existing metadata, overwriting individual keys that existed already.

        Args:
            id_to_metadata: map from the Document IDs to modify to the
                new metadata for updating. Keys in this dictionary that
                do not correspond to an existing document will be silently ignored.
                The values of this map are metadata dictionaries for updating
                the documents. Any pre-existing metadata will be merged with
                these entries, which take precedence on a key-by-key basis.
            overwrite_concurrency: number of asynchronous tasks to process the updates.
                Defaults to the vector-store overall setting if not provided.

        Returns:
            the number of documents successfully updated (i.e. found to exist,
            since even an update with `{}` as the new metadata counts as successful.)
        """
        await self.astra_env.aensure_db_setup()

        sem = asyncio.Semaphore(
            overwrite_concurrency or self.bulk_insert_overwrite_concurrency,
        )

        _async_collection = self.astra_env.async_collection

        async def _update_document(
            id_md_pair: tuple[str, dict],
        ) -> UpdateResult:
            document_id, update_metadata = id_md_pair
            encoded_metadata = self.filter_to_query(update_metadata)
            async with sem:
                return await _async_collection.update_one(
                    {"_id": document_id},
                    {"$set": encoded_metadata},
                )

        tasks = [
            asyncio.create_task(_update_document(id_md_pair))
            for id_md_pair in id_to_metadata.items()
        ]

        update_results = await asyncio.gather(*tasks, return_exceptions=False)

        return sum(u_res.update_info["n"] for u_res in update_results)

    def metadata_search(
        self,
        filter: dict[str, Any] | None = None,  # noqa: A002
        n: int = 5,
    ) -> list[Document]:
        """Get documents via a metadata search.

        Args:
            filter: the metadata to query for.
            n: the maximum number of documents to return.
        """
        self.astra_env.ensure_db_setup()
        metadata_parameter = self.filter_to_query(filter)
        hits_ite = self.astra_env.collection.find(
            filter=metadata_parameter,
            projection=self.document_codec.base_projection,
            limit=n,
        )
        docs = [self.document_codec.decode(hit) for hit in hits_ite]
        return [doc for doc in docs if doc is not None]

    async def ametadata_search(
        self,
        filter: dict[str, Any] | None = None,  # noqa: A002
        n: int = 5,
    ) -> Iterable[Document]:
        """Get documents via a metadata search.

        Args:
            filter: the metadata to query for.
            n: the maximum number of documents to return.
        """
        await self.astra_env.aensure_db_setup()
        metadata_parameter = self.filter_to_query(filter)
        return [
            doc
            async for doc in (
                self.document_codec.decode(hit)
                async for hit in self.astra_env.async_collection.find(
                    filter=metadata_parameter,
                    projection=self.document_codec.base_projection,
                    limit=n,
                )
            )
            if doc is not None
        ]

    def get_by_document_id(self, document_id: str) -> Document | None:
        """Retrieve a single document from the store, given its document ID.

        Args:
            document_id: The document ID

        Returns:
            The the document if it exists. Otherwise None.
        """
        self.astra_env.ensure_db_setup()
        # self.collection is not None (by _ensure_astra_db_client)
        hit = self.astra_env.collection.find_one(
            {"_id": document_id},
            projection=self.document_codec.base_projection,
        )
        if hit is None:
            return None
        return self.document_codec.decode(hit)

    async def aget_by_document_id(self, document_id: str) -> Document | None:
        """Retrieve a single document from the store, given its document ID.

        Args:
            document_id: The document ID

        Returns:
            The the document if it exists. Otherwise None.
        """
        await self.astra_env.aensure_db_setup()
        # self.collection is not None (by _ensure_astra_db_client)
        hit = await self.astra_env.async_collection.find_one(
            {"_id": document_id},
            projection=self.document_codec.base_projection,
        )
        if hit is None:
            return None
        return self.document_codec.decode(hit)

    @override
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to query.

        Args:
            query: Query to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            **kwargs: Additional arguments are ignored.

        Returns:
            The list of Documents most similar to the query.
        """
        return [
            doc
            for (doc, _, _) in self.similarity_search_with_score_id(
                query=query,
                k=k,
                filter=filter,
            )
        ]

    @override
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """Return docs most similar to query with score.

        Args:
            query: Query to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.

        Returns:
            The list of (Document, score), the most similar to the query vector.
        """
        return [
            (doc, score)
            for (doc, score, _) in self.similarity_search_with_score_id(
                query=query,
                k=k,
                filter=filter,
            )
        ]

    def similarity_search_with_score_id(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> list[tuple[Document, float, str]]:
        """Return docs most similar to the query with score and id.

        Args:
            query: Query to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.

        Returns:
            The list of (Document, score, id), the most similar to the query.
        """
        if self.document_codec.server_side_embeddings:
            sort = {"$vectorize": query}
            return self._similarity_search_with_score_id_by_sort(
                sort=sort,
                k=k,
                filter=filter,
            )

        embedding_vector = self._get_safe_embedding().embed_query(query)
        return self.similarity_search_with_score_id_by_vector(
            embedding=embedding_vector,
            k=k,
            filter=filter,
        )

    @override
    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            **kwargs: Additional arguments are ignored.

        Returns:
            The list of Documents most similar to the query vector.
        """
        return [
            doc
            for (doc, _, _) in self.similarity_search_with_score_id_by_vector(
                embedding=embedding,
                k=k,
                filter=filter,
            )
        ]

    def similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> list[tuple[Document, float]]:
        """Return docs most similar to embedding vector with score.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.

        Returns:
            The list of (Document, score), the most similar to the query vector.
        """
        return [
            (doc, score)
            for (doc, score, _) in self.similarity_search_with_score_id_by_vector(
                embedding=embedding,
                k=k,
                filter=filter,
            )
        ]

    def similarity_search_with_score_id_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> list[tuple[Document, float, str]]:
        """Return docs most similar to embedding vector with score and id.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.

        Returns:
            The list of (Document, score, id), the most similar to the query vector.
        """
        if self.document_codec.server_side_embeddings:
            msg = (
                "Searching by vector on a Vector Store that uses server-side "
                "embeddings is not allowed."
            )
            raise ValueError(msg)
        sort = {"$vector": embedding}
        return self._similarity_search_with_score_id_by_sort(
            sort=sort,
            k=k,
            filter=filter,
        )

    def _similarity_search_with_score_id_by_sort(
        self,
        sort: dict[str, Any],
        k: int = 4,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> list[tuple[Document, float, str]]:
        """Run ANN search with a provided sort clause."""
        self.astra_env.ensure_db_setup()
        metadata_parameter = self.filter_to_query(filter)
        hits_ite = self.astra_env.collection.find(
            filter=metadata_parameter,
            projection=self.document_codec.base_projection,
            limit=k,
            include_similarity=True,
            sort=sort,
        )
        return [
            (doc, sim, did)
            for (doc, sim, did) in (
                (
                    self.document_codec.decode(hit),
                    hit["$similarity"],
                    hit["_id"],
                )
                for hit in hits_ite
            )
            if doc is not None
        ]

    @override
    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to query.

        Args:
            query: Query to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            **kwargs: Additional arguments are ignored.

        Returns:
            The list of Documents most similar to the query.
        """
        return [
            doc
            for (doc, _, _) in await self.asimilarity_search_with_score_id(
                query=query,
                k=k,
                filter=filter,
            )
        ]

    @override
    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """Return docs most similar to query with score.

        Args:
            query: Query to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.

        Returns:
            The list of (Document, score), the most similar to the query vector.
        """
        return [
            (doc, score)
            for (doc, score, _) in await self.asimilarity_search_with_score_id(
                query=query,
                k=k,
                filter=filter,
            )
        ]

    async def asimilarity_search_with_score_id(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> list[tuple[Document, float, str]]:
        """Return docs most similar to the query with score and id.

        Args:
            query: Query to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.

        Returns:
            The list of (Document, score, id), the most similar to the query.
        """
        if self.document_codec.server_side_embeddings:
            sort = {"$vectorize": query}
            return await self._asimilarity_search_with_score_id_by_sort(
                sort=sort,
                k=k,
                filter=filter,
            )

        embedding_vector = await self._get_safe_embedding().aembed_query(query)
        return await self.asimilarity_search_with_score_id_by_vector(
            embedding=embedding_vector,
            k=k,
            filter=filter,
        )

    @override
    async def asimilarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            **kwargs: Additional arguments are ignored.

        Returns:
            The list of Documents most similar to the query vector.
        """
        return [
            doc
            for (doc, _, _) in await self.asimilarity_search_with_score_id_by_vector(
                embedding=embedding,
                k=k,
                filter=filter,
            )
        ]

    async def asimilarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> list[tuple[Document, float]]:
        """Return docs most similar to embedding vector with score.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.

        Returns:
            The list of (Document, score), the most similar to the query vector.
        """
        return [
            (doc, scr)
            for (doc, scr, _) in await self.asimilarity_search_with_score_id_by_vector(
                embedding=embedding,
                k=k,
                filter=filter,
            )
        ]

    async def asimilarity_search_with_score_id_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> list[tuple[Document, float, str]]:
        """Return docs most similar to embedding vector with score and id.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.

        Returns:
            The list of (Document, score, id), the most similar to the query vector.
        """
        if self.document_codec.server_side_embeddings:
            msg = (
                "Searching by vector on a Vector Store that uses server-side "
                "embeddings is not allowed."
            )
            raise ValueError(msg)
        sort = {"$vector": embedding}
        return await self._asimilarity_search_with_score_id_by_sort(
            sort=sort,
            k=k,
            filter=filter,
        )

    def similarity_search_with_embedding_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> list[tuple[Document, list[float]]]:
        """Return docs most similar to embedding vector with embedding.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.

        Returns:
            (The query embedding vector, The list of (Document, embedding),
            the most similar to the query vector.).
        """
        sort = self.document_codec.encode_vector_sort(vector=embedding)
        _, doc_emb_list = self._similarity_search_with_embedding_by_sort(
            sort=sort, k=k, filter=filter
        )
        return doc_emb_list

    async def asimilarity_search_with_embedding_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> list[tuple[Document, list[float]]]:
        """Return docs most similar to embedding vector with embedding.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.

        Returns:
            (The query embedding vector, The list of (Document, embedding),
            the most similar to the query vector.).
        """
        sort = self.document_codec.encode_vector_sort(vector=embedding)
        _, doc_emb_list = await self._asimilarity_search_with_embedding_by_sort(
            sort=sort, k=k, filter=filter
        )
        return doc_emb_list

    def similarity_search_with_embedding(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> tuple[list[float], list[tuple[Document, list[float]]]]:
        """Return docs most similar to the query with embedding.

        Also includes the query embedding vector.

        Args:
            query: Query to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.

        Returns:
            (The query embedding vector, The list of (Document, embedding),
            the most similar to the query vector.).
        """
        if self.document_codec.server_side_embeddings:
            sort = {"$vectorize": query}
        else:
            query_embedding = self._get_safe_embedding().embed_query(text=query)
            # shortcut return if query isn't needed.
            if k == 0:
                return (query_embedding, [])
            sort = self.document_codec.encode_vector_sort(vector=query_embedding)

        return self._similarity_search_with_embedding_by_sort(
            sort=sort, k=k, filter=filter
        )

    async def asimilarity_search_with_embedding(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> tuple[list[float], list[tuple[Document, list[float]]]]:
        """Return docs most similar to the query with embedding.

        Also includes the query embedding vector.

        Args:
            query: Query to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.

        Returns:
            (The query embedding vector, The list of (Document, embedding),
            the most similar to the query vector.).
        """
        if self.document_codec.server_side_embeddings:
            sort = {"$vectorize": query}
        else:
            query_embedding = self._get_safe_embedding().embed_query(text=query)
            # shortcut return if query isn't needed.
            if k == 0:
                return (query_embedding, [])
            sort = self.document_codec.encode_vector_sort(vector=query_embedding)

        return await self._asimilarity_search_with_embedding_by_sort(
            sort=sort, k=k, filter=filter
        )

    async def _asimilarity_search_with_embedding_by_sort(
        self,
        sort: dict[str, Any],
        k: int = 4,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> tuple[list[float], list[tuple[Document, list[float]]]]:
        """Run ANN search with a provided sort clause.

        Returns:
            (query_embedding, List of (Document, embedding) most similar to the query).
        """
        await self.astra_env.aensure_db_setup()
        async_cursor = self.astra_env.async_collection.find(
            filter=self.filter_to_query(filter),
            projection=self.document_codec.full_projection,
            limit=k,
            include_sort_vector=True,
            sort=sort,
        )
        sort_vector = await async_cursor.get_sort_vector()
        if sort_vector is None:
            msg = "Unable to retrieve the server-side embedding of the query."
            raise ValueError(msg)
        query_embedding = sort_vector

        return (
            query_embedding,
            [
                (doc, emb)
                async for (doc, emb) in (
                    (
                        self.document_codec.decode(hit),
                        self.document_codec.decode_vector(hit),
                    )
                    async for hit in async_cursor
                )
                if doc is not None and emb is not None
            ],
        )

    def _similarity_search_with_embedding_by_sort(
        self,
        sort: dict[str, Any],
        k: int = 4,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> tuple[list[float], list[tuple[Document, list[float]]]]:
        """Run ANN search with a provided sort clause.

        Returns:
            (query_embedding, List of (Document, embedding) most similar to the query).
        """
        self.astra_env.ensure_db_setup()
        cursor = self.astra_env.collection.find(
            filter=self.filter_to_query(filter),
            projection=self.document_codec.full_projection,
            limit=k,
            include_sort_vector=True,
            sort=sort,
        )
        sort_vector = cursor.get_sort_vector()
        if sort_vector is None:
            msg = "Unable to retrieve the server-side embedding of the query."
            raise ValueError(msg)
        query_embedding = sort_vector

        return (
            query_embedding,
            [
                (doc, emb)
                for (doc, emb) in (
                    (
                        self.document_codec.decode(hit),
                        self.document_codec.decode_vector(hit),
                    )
                    for hit in cursor
                )
                if doc is not None and emb is not None
            ],
        )

    async def _asimilarity_search_with_score_id_by_sort(
        self,
        sort: dict[str, Any],
        k: int = 4,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> list[tuple[Document, float, str]]:
        """Run ANN search with a provided sort clause."""
        await self.astra_env.aensure_db_setup()
        metadata_parameter = self.filter_to_query(filter)
        return [
            (doc, sim, did)
            async for (doc, sim, did) in (
                (
                    self.document_codec.decode(hit),
                    hit["$similarity"],
                    hit["_id"],
                )
                async for hit in self.astra_env.async_collection.find(
                    filter=metadata_parameter,
                    projection=self.document_codec.base_projection,
                    limit=k,
                    include_similarity=True,
                    sort=sort,
                )
            )
            if doc is not None
        ]

    def _run_mmr_query_by_sort(
        self,
        sort: dict[str, Any],
        k: int,
        fetch_k: int,
        lambda_mult: float,
        metadata_parameter: dict[str, Any],
    ) -> list[Document]:
        prefetch_cursor = self.astra_env.collection.find(
            filter=metadata_parameter,
            projection=self.document_codec.full_projection,
            limit=fetch_k,
            include_similarity=True,
            include_sort_vector=True,
            sort=sort,
        )
        prefetch_hits = list(prefetch_cursor)
        query_vector = prefetch_cursor.get_sort_vector()
        return self._get_mmr_hits(
            embedding=query_vector,  # type: ignore[arg-type]
            k=k,
            lambda_mult=lambda_mult,
            prefetch_hits=prefetch_hits,
        )

    async def _arun_mmr_query_by_sort(
        self,
        sort: dict[str, Any],
        k: int,
        fetch_k: int,
        lambda_mult: float,
        metadata_parameter: dict[str, Any],
    ) -> list[Document]:
        prefetch_cursor = self.astra_env.async_collection.find(
            filter=metadata_parameter,
            projection=self.document_codec.full_projection,
            limit=fetch_k,
            include_similarity=True,
            include_sort_vector=True,
            sort=sort,
        )
        prefetch_hits = [hit async for hit in prefetch_cursor]
        query_vector = await prefetch_cursor.get_sort_vector()
        return self._get_mmr_hits(
            embedding=query_vector,  # type: ignore[arg-type]
            k=k,
            lambda_mult=lambda_mult,
            prefetch_hits=prefetch_hits,
        )

    def _get_mmr_hits(
        self,
        embedding: list[float],
        k: int,
        lambda_mult: float,
        prefetch_hits: list[DocDict],
    ) -> list[Document]:
        mmr_chosen_indices = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            [prefetch_hit["$vector"] for prefetch_hit in prefetch_hits],
            k=k,
            lambda_mult=lambda_mult,
        )
        mmr_hits = [
            prefetch_hit
            for prefetch_index, prefetch_hit in enumerate(prefetch_hits)
            if prefetch_index in mmr_chosen_indices
        ]
        return [
            doc
            for doc in (self.document_codec.decode(hit) for hit in mmr_hits)
            if doc is not None
        ]

    @override
    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
            filter: Filter on the metadata to apply.
            **kwargs: Additional arguments are ignored.

        Returns:
            The list of Documents selected by maximal marginal relevance.
        """
        self.astra_env.ensure_db_setup()
        metadata_parameter = self.filter_to_query(filter)

        return self._run_mmr_query_by_sort(
            sort={"$vector": embedding},
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            metadata_parameter=metadata_parameter,
        )

    @override
    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
            filter: Filter on the metadata to apply.
            **kwargs: Additional arguments are ignored.

        Returns:
            The list of Documents selected by maximal marginal relevance.
        """
        await self.astra_env.aensure_db_setup()
        metadata_parameter = self.filter_to_query(filter)

        return await self._arun_mmr_query_by_sort(
            sort={"$vector": embedding},
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            metadata_parameter=metadata_parameter,
        )

    @override
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Query to look up documents similar to.
            k: Number of Documents to return.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
            filter: Filter on the metadata to apply.
            **kwargs: Additional arguments are ignored.

        Returns:
            The list of Documents selected by maximal marginal relevance.
        """
        if self.document_codec.server_side_embeddings:
            # this case goes directly to the "_by_sort" method
            # (and does its own filter normalization, as it cannot
            #  use the path for the with-embedding mmr querying)
            metadata_parameter = self.filter_to_query(filter)
            return self._run_mmr_query_by_sort(
                sort={"$vectorize": query},
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                metadata_parameter=metadata_parameter,
            )

        embedding_vector = self._get_safe_embedding().embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding_vector,
            k,
            fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )

    @override
    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Query to look up documents similar to.
            k: Number of Documents to return.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
            filter: Filter on the metadata to apply.
            **kwargs: Additional arguments are ignored.

        Returns:
            The list of Documents selected by maximal marginal relevance.
        """
        if self.document_codec.server_side_embeddings:
            # this case goes directly to the "_by_sort" method
            # (and does its own filter normalization, as it cannot
            #  use the path for the with-embedding mmr querying)
            metadata_parameter = self.filter_to_query(filter)
            return await self._arun_mmr_query_by_sort(
                sort={"$vectorize": query},
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                metadata_parameter=metadata_parameter,
            )

        embedding_vector = await self._get_safe_embedding().aembed_query(query)
        return await self.amax_marginal_relevance_search_by_vector(
            embedding_vector,
            k,
            fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )

    @classmethod
    def _from_kwargs(
        cls: type[AstraDBVectorStore],
        **kwargs: Any,
    ) -> AstraDBVectorStore:
        _args = inspect.getfullargspec(AstraDBVectorStore.__init__).args
        _kwargs = inspect.getfullargspec(AstraDBVectorStore.__init__).kwonlyargs
        known_kwarg_keys = (set(_args) | set(_kwargs)) - {"self"}
        if kwargs:
            unknown_kwarg_keys = set(kwargs.keys()) - known_kwarg_keys
            if unknown_kwarg_keys:
                warnings.warn(
                    (
                        "Method 'from_texts/afrom_texts' of AstraDBVectorStore "
                        "vector store invoked with unsupported arguments "
                        f"({', '.join(sorted(unknown_kwarg_keys))}), "
                        "which will be ignored."
                    ),
                    UserWarning,
                    stacklevel=3,
                )

        known_kwargs = {k: v for k, v in kwargs.items() if k in known_kwarg_keys}
        return cls(**known_kwargs)

    @classmethod
    @override
    def from_texts(
        cls: type[AstraDBVectorStore],
        texts: list[str],
        embedding: Embeddings | None = None,
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> AstraDBVectorStore:
        """Create an Astra DB vectorstore from raw texts.

        Args:
            texts: the texts to insert.
            embedding: the embedding function to use in the store.
            metadatas: metadata dicts for the texts.
            ids: ids to associate to the texts.
            **kwargs: you can pass any argument that you would
                to :meth:`~add_texts` and/or to the
                ``AstraDBVectorStore`` constructor (see these methods for
                details). These arguments will be
                routed to the respective methods as they are.

        Returns:
            an ``AstraDBVectorStore`` vectorstore.
        """
        _add_texts_inspection = inspect.getfullargspec(AstraDBVectorStore.add_texts)
        _method_args = (
            set(_add_texts_inspection.kwonlyargs)
            | set(_add_texts_inspection.kwonlyargs)
        ) - {"self", "texts", "metadatas", "ids"}
        _init_kwargs = {k: v for k, v in kwargs.items() if k not in _method_args}
        _method_kwargs = {k: v for k, v in kwargs.items() if k in _method_args}
        astra_db_store = AstraDBVectorStore._from_kwargs(
            embedding=embedding,
            **_init_kwargs,
        )
        astra_db_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            **_method_kwargs,
        )
        return astra_db_store

    @override
    @classmethod
    async def afrom_texts(
        cls: type[AstraDBVectorStore],
        texts: list[str],
        embedding: Embeddings | None = None,
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> AstraDBVectorStore:
        """Create an Astra DB vectorstore from raw texts.

        Args:
            texts: the texts to insert.
            embedding: embedding function to use.
            metadatas: metadata dicts for the texts.
            ids: ids to associate to the texts.
            **kwargs: you can pass any argument that you would
                to :meth:`~aadd_texts` and/or to the ``AstraDBVectorStore``
                constructor (see these methods for details). These arguments
                will be routed to the respective methods as they are.

        Returns:
            an ``AstraDBVectorStore`` vectorstore.
        """
        _aadd_texts_inspection = inspect.getfullargspec(AstraDBVectorStore.aadd_texts)
        _method_args = (
            set(_aadd_texts_inspection.kwonlyargs)
            | set(_aadd_texts_inspection.kwonlyargs)
        ) - {"self", "texts", "metadatas", "ids"}
        _init_kwargs = {k: v for k, v in kwargs.items() if k not in _method_args}
        _method_kwargs = {k: v for k, v in kwargs.items() if k in _method_args}
        astra_db_store = AstraDBVectorStore._from_kwargs(
            embedding=embedding,
            **_init_kwargs,
        )
        await astra_db_store.aadd_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            **_method_kwargs,
        )
        return astra_db_store

    @classmethod
    @override
    def from_documents(
        cls: type[AstraDBVectorStore],
        documents: list[Document],
        embedding: Embeddings | None = None,
        **kwargs: Any,
    ) -> AstraDBVectorStore:
        """Create an Astra DB vectorstore from a document list.

        Utility method that defers to :meth:`from_texts` (see that one).

        Args:
            texts: the texts to insert.
            documents: a list of `Document` objects for insertion in the store.
            embedding: the embedding function to use in the store.
            **kwargs: you can pass any argument that you would
                to :meth:`~add_texts` and/or to the
                ``AstraDBVectorStore`` constructor (see these methods for
                details). These arguments will be
                routed to the respective methods as they are.

        Returns:
            an ``AstraDBVectorStore`` vectorstore.
        """
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]

        if "ids" in kwargs:
            warnings.warn(
                (
                    "Parameter `ids` to AstraDBVectorStore's `from_documents` "
                    "method is deprecated. Please set the supplied documents' "
                    "`.id` attribute instead. The id attribute of Document "
                    "is ignored as long as the `ids` parameter is passed."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
            ids = kwargs.pop("ids")
        else:
            _ids = [doc.id for doc in documents]
            ids = _ids if any(the_id is not None for the_id in _ids) else None
        return cls.from_texts(
            texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )

    @classmethod
    async def afrom_documents(
        cls: type[AstraDBVectorStore],
        documents: list[Document],
        embedding: Embeddings | None = None,
        **kwargs: Any,
    ) -> AstraDBVectorStore:
        """Create an Astra DB vectorstore from a document list.

        Utility method that defers to :meth:`afrom_texts` (see that one).

        Args: see :meth:`afrom_texts`, except here you have to supply ``documents``
            in place of ``texts`` and ``metadatas``.

        Returns:
            an ``AstraDBVectorStore`` vectorstore.
        """
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]

        if "ids" in kwargs:
            warnings.warn(
                (
                    "Parameter `ids` to AstraDBVectorStore's `from_documents` "
                    "method is deprecated. Please set the supplied documents' "
                    "`.id` attribute instead. The id attribute of Document "
                    "is ignored as long as the `ids` parameter is passed."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
            ids = kwargs.pop("ids")
        else:
            _ids = [doc.id for doc in documents]
            ids = _ids if any(the_id is not None for the_id in _ids) else None
        return await cls.afrom_texts(
            texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )
