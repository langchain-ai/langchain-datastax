"""Astra DB vector store integration."""

from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterable,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    Literal,
    NamedTuple,
    Sequence,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
from astrapy.constants import Environment
from astrapy.exceptions import CollectionInsertManyException, DataAPIResponseException
from astrapy.info import (
    CollectionLexicalOptions,
    CollectionRerankOptions,
    VectorServiceOptions,
)
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from langchain_core.documents import Document
from langchain_core.runnables.utils import gather_with_concurrency
from langchain_core.vectorstores import VectorStore
from typing_extensions import override

from langchain_astradb.utils.astradb import (
    COMPONENT_NAME_VECTORSTORE,
    DEFAULT_DOCUMENT_CHUNK_SIZE,
    MAX_CONCURRENT_DOCUMENT_DELETIONS,
    MAX_CONCURRENT_DOCUMENT_INSERTIONS,
    MAX_CONCURRENT_DOCUMENT_REPLACEMENTS,
    HybridSearchMode,
    SetupMode,
    _AstraDBCollectionEnvironment,
    _survey_collection,
)
from langchain_astradb.utils.vector_store_autodetect import (
    _detect_document_codec,
)
from langchain_astradb.utils.vector_store_codecs import (
    LEXICAL_FIELD_NAME,
    VECTOR_FIELD_NAME,
    VECTORIZE_FIELD_NAME,
    _AstraDBVectorStoreDocumentCodec,
    _DefaultVectorizeVSDocumentCodec,
    _DefaultVSDocumentCodec,
)

if TYPE_CHECKING:
    from astrapy.authentication import (
        EmbeddingHeadersProvider,
        RerankingHeadersProvider,
        TokenProvider,
    )
    from astrapy.cursors import RerankedResult
    from astrapy.info import RerankServiceOptions
    from astrapy.results import CollectionUpdateResult
    from langchain_core.embeddings import Embeddings

T = TypeVar("T")
U = TypeVar("U")
DocDict = Dict[str, Any]  # dicts expressing entries to insert

# error code to check for during bulk insertions
DOCUMENT_ALREADY_EXISTS_API_ERROR_CODE = "DOCUMENT_ALREADY_EXISTS"
# max number of errors shown in full insertion error messages
MAX_SHOWN_INSERTION_ERRORS = 8
# key for the 'rerank' score within the find_and_rerank scores
RERANK_SCORE_KEY = "$rerank"
# Error message for receiving a lexical_query for a non-hybrid search
ERROR_LEXICAL_QUERY_ON_NONHYBRID_SEARCH = (
    "Parameter 'lexical_query' cannot be passed for a non-hybrid search"
)

logger = logging.getLogger(__name__)


class AstraDBQueryResult(NamedTuple):
    """The complete information contained in a vector store entry.

    This class represents all that can be returned from the collection when running
    a query, which goes beyond just the corresponding Document.

    Atributes:
        document: a ``langchain.schema.Document`` object representing the query result.
        id: the ID of the returned document.
        embedding: the embedding vector associated to the document. This may be None,
            depending on whether the embeddings were requested in the query or not.
        similarity: the numeric similarity score of the document in the query. In case
            this quantity was not requested by the query, it will be set to None.
    """

    document: Document
    id: str
    embedding: list[float] | None
    similarity: float | None


class HybridLimitFactorPrescription(NamedTuple):
    """A per-subsearch setting for the hybrid-search 'limit' factors.

    This structure is to be used to set different values for
    the vector and the lexical portions of the hybrid search.

    Each of the attributes is a floating-point number, representing the multiplicative
    factor applied to a search final 'k' to calculate the "limit" value for
    the associated sub-search. For instance, if vector=1.5 and lexical=3.0,
    a hybrid search called by asking a final set of k=4 results will be executed
    with limits of 6 for vector and 12 for lexical. (The results are approximated
    to an integer.)

    Attributes:
        vector: the multiplicative factor for the "vector" part of the hybrid search.
        lexical: the multiplicative factor for the "lexical" part of the hybrid search.
    """

    vector: float
    lexical: float


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
        return VECTORIZE_FIELD_NAME

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
    collection_vector_service_options: VectorServiceOptions | None,
    collection_rerank: CollectionRerankOptions | RerankServiceOptions | None,
    collection_lexical: str | dict[str, Any] | CollectionLexicalOptions | None,
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
            ("collection_rerank", collection_rerank),
            ("collection_lexical", collection_lexical),
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


def _decide_hybrid_search_setting(
    *,
    required_hybrid_search: HybridSearchMode | None,
    has_hybrid: bool,
) -> bool:
    """Determine whether searches must be hybrid.

    Args:
        required_hybrid_search: the hybrid_search parameter required in the constructor.
        has_hybrid: whether the collection actually is hybrid-capable.
    """
    if required_hybrid_search == HybridSearchMode.OFF:
        return False
    if required_hybrid_search == HybridSearchMode.ON:
        return True
    return has_hybrid


def _make_hybrid_limits(
    hlf: None | float | dict[str, float],
    k: int,
) -> None | int | dict[str, int]:
    if hlf is None:
        return None
    if isinstance(hlf, float):
        return max(int(hlf * k), 1)
    # hlf is a dict:
    return {hlk: max(int(hlf * k), 1) for hlk, hlf in hlf.items()}


def _normalize_hybrid_limit_factor(
    hybrid_limit_factor: float
    | None
    | dict[str, float]
    | HybridLimitFactorPrescription,
) -> float | dict[str, float] | None:
    """Bring `hybrid_limit_factor` to a normal form."""
    if hybrid_limit_factor is None:
        return None
    if isinstance(hybrid_limit_factor, float):
        return hybrid_limit_factor

    if isinstance(hybrid_limit_factor, HybridLimitFactorPrescription):
        return {
            VECTOR_FIELD_NAME: hybrid_limit_factor.vector,
            LEXICAL_FIELD_NAME: hybrid_limit_factor.lexical,
        }

    # already a dict:
    return hybrid_limit_factor


def _insertmany_error_message(err: CollectionInsertManyException) -> str:
    """Format an astrapy insert exception into an error message.

    This utility prepares a detailed message from an astrapy
    CollectionInsertManyException, to be used in raising an exception within a
    vectorstore multiple insertion.

    This operation must filter out duplicate-id specific errors
    (which the vector store could actually handle, if they were the only ones).
    """
    err_msg = "Cannot insert documents. The Data API returned the following error(s): "

    def _describe_error(_errd: Exception) -> list[str]:
        if isinstance(_errd, DataAPIResponseException):
            return [
                edesc.message or ""
                for edesc in _errd.error_descriptors
                if edesc.error_code != DOCUMENT_ALREADY_EXISTS_API_ERROR_CODE
            ]
        return [str(_errd)]

    filtered_error_descs = [
        edesc
        for insmany_err in err.exceptions
        for edesc in _describe_error(insmany_err)
    ]
    err_msg += "; ".join(
        edesc or "" for edesc in filtered_error_descs[:MAX_SHOWN_INSERTION_ERRORS]
    )

    if (num_residual := len(filtered_error_descs) - MAX_SHOWN_INSERTION_ERRORS) > 0:
        err_msg += f". (Note: {num_residual} further errors omitted.)"

    err_msg += (
        " (Full API error in '<this-exception>.__cause__.error_descriptors'"
        f": ignore '{DOCUMENT_ALREADY_EXISTS_API_ERROR_CODE}'.)"
    )
    return err_msg


class AstraDBVectorStoreError(Exception):
    """An exception during vector-store activities.

    This exception represents any operational exception occurring while
    performing an action within an AstraDBVectorStore.
    """


class AstraDBVectorStore(VectorStore):
    """A vector store which uses DataStax Astra DB as backend.

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
            Astra DB API endpoint.
        token: str
            API token for Astra DB usage.
        namespace: Optional[str]
            Namespace (aka keyspace) where the collection is created

    Instantiate:
        Get your API endpoint and application token from the dashboard of your database.

        Create a vector store and provide a LangChain embedding object for working with it:

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

        (Vectorize) Create a vector store where the embedding vector computation happens entirely
        on the server-side, using the `vectorize <https://docs.datastax.com/en/astra-db-serverless/databases/embedding-generation.html>`_ feature:

        .. code-block:: python

            import getpass
            from astrapy.info import VectorServiceOptions

            from langchain_astradb import AstraDBVectorStore

            ASTRA_DB_API_ENDPOINT = getpass.getpass("ASTRA_DB_API_ENDPOINT = ")
            ASTRA_DB_APPLICATION_TOKEN = getpass.getpass("ASTRA_DB_APPLICATION_TOKEN = ")

            vector_store = AstraDBVectorStore(
                collection_name="astra_vectorize_langchain",
                api_endpoint=ASTRA_DB_API_ENDPOINT,
                token=ASTRA_DB_APPLICATION_TOKEN,
                collection_vector_service_options=VectorServiceOptions(
                    provider="nvidia",
                    model_name="NV-Embed-QA",
                    # authentication=...,  # needed by some providers/models
                ),
            )

        (Hybrid) The underlying Astra DB typically supports hybrid search
        (i.e. lexical + vector ANN) to boost the results' accuracy.
        This is provisioned and used automatically when available. For manual control,
        use the ``collection_rerank`` and ``collection_lexical`` constructor parameters:

        .. code-block:: python

            import getpass
            from astrapy.info import (
                CollectionLexicalOptions,
                CollectionRerankOptions,
                RerankServiceOptions,
                VectorServiceOptions,
            )

            from langchain_astradb import AstraDBVectorStore

            ASTRA_DB_API_ENDPOINT = getpass.getpass("ASTRA_DB_API_ENDPOINT = ")
            ASTRA_DB_APPLICATION_TOKEN = getpass.getpass("ASTRA_DB_APPLICATION_TOKEN = ")

            vector_store = AstraDBVectorStore(
                collection_name="astra_vectorize_langchain",
                # embedding=...,  # needed unless using 'vectorize'
                api_endpoint=ASTRA_DB_API_ENDPOINT,
                token=ASTRA_DB_APPLICATION_TOKEN,
                collection_vector_service_options=VectorServiceOptions(...),  # see above
                collection_lexical=CollectionLexicalOptions(analyzer="standard"),
                collection_rerank=CollectionRerankOptions(
                    service=RerankServiceOptions(
                        provider="nvidia",
                        model_name="nvidia/llama-3.2-nv-rerankqa-1b-v2",
                    ),
                ),
                collection_reranking_api_key=...,  # if needed by the model/setup
            )

        Hybrid-related server upgrades may introduce a mismatch between the store
        defaults and a pre-existing collection: in case one such mismatch is
        reported (as a Data API "EXISTING_COLLECTION_DIFFERENT_SETTINGS" error),
        the options to resolve are:
        (1) use autodetect mode, (2) switch to ``setup_mode`` "OFF", or
        (3) explicitly specify lexical and/or rerank settings in the vector
        store constructor, to match the existing collection configuration.
        See `here <https://github.com/langchain-ai/langchain-datastax/blob/main/libs/astradb/README.md#collection-defaults-mismatch>`_ for more details.

        (Autodetect) Let the vector store figure out the configuration (including vectorize
        and document encoding scheme on DB), by inspection of an existing collection:

        .. code-block:: python

            import getpass

            from langchain_astradb import AstraDBVectorStore

            ASTRA_DB_API_ENDPOINT = getpass.getpass("ASTRA_DB_API_ENDPOINT = ")
            ASTRA_DB_APPLICATION_TOKEN = getpass.getpass("ASTRA_DB_APPLICATION_TOKEN = ")

            vector_store = AstraDBVectorStore(
                collection_name="astra_existing_collection",
                # embedding=...,  # needed unless using 'vectorize'
                api_endpoint=ASTRA_DB_API_ENDPOINT,
                token=ASTRA_DB_APPLICATION_TOKEN,
                autodetect_collection=True,
            )

        (Non-Astra DB) This class can also target a non-Astra DB database, such as a
        self-deployed HCD, through the Data API:

        .. code-block:: python

            import getpass

            from astrapy.authentication import UsernamePasswordTokenProvider

            from langchain_astradb import AstraDBVectorStore

            vector_store = AstraDBVectorStore(
                collection_name="astra_existing_collection",
                # embedding=...,  # needed unless using 'vectorize'
                api_endpoint="http://localhost:8181",
                token=UsernamePasswordTokenProvider(
                    username="user",
                    password="pwd",
                ),
                collection_vector_service_options=...,  # if 'vectorize'
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
        document_codec: _AstraDBVectorStoreDocumentCodec,
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
                    document_codec.metadata_key_to_field_identifier(md_field)
                    for md_field in metadata_indexing_include
                ]
            }
        if metadata_indexing_exclude is not None:
            return {
                "deny": [
                    document_codec.metadata_key_to_field_identifier(md_field)
                    for md_field in metadata_indexing_exclude
                ]
            }
        return (
            collection_indexing_policy
            if collection_indexing_policy is not None
            else document_codec.default_collection_indexing_policy
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
        collection_vector_service_options: VectorServiceOptions | None = None,
        collection_embedding_api_key: str | EmbeddingHeadersProvider | None = None,
        content_field: str | None = None,
        ignore_invalid_documents: bool = False,
        autodetect_collection: bool = False,
        ext_callers: list[tuple[str | None, str | None] | str | None] | None = None,
        component_name: str = COMPONENT_NAME_VECTORSTORE,
        collection_rerank: CollectionRerankOptions | RerankServiceOptions | None = None,
        collection_reranking_api_key: str | RerankingHeadersProvider | None = None,
        collection_lexical: str
        | dict[str, Any]
        | CollectionLexicalOptions
        | None = None,
        hybrid_search: HybridSearchMode | None = None,
        hybrid_limit_factor: float
        | None
        | dict[str, float]
        | HybridLimitFactorPrescription = None,
    ) -> None:
        """A vector store wich uses DataStax Astra DB as backend.

        For more on Astra DB, visit
        https://docs.datastax.com/en/astra-db-serverless/index.html

        Args:
            embedding: the embeddings function or service to use.
                This enables client-side embedding functions or calls to external
                embedding providers. If ``embedding`` is passed, then
                ``collection_vector_service_options`` can not be provided.
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
            collection_rerank: providing reranking settings is necessary to run
                hybrid searches for similarity. This parameter can be an instance
                of the astrapy classes `CollectionRerankOptions` or
                ``RerankServiceOptions``.
            collection_reranking_api_key: for usage of server-side reranking services
                within Astra DB. With this parameter one can supply an API Key
                that will be passed to Astra DB with each data request.
                This parameter can be either a string or a subclass of
                ``astrapy.authentication.RerankingHeadersProvider``.
                This is useful when the service is configured for the collection,
                but no corresponding secret is stored within
                Astra's key management system.
            collection_lexical: configuring a lexical analyzer is necessary to run
                lexical and hybrid searches. This parameter can be a string or dict,
                which is then passed as-is for the "analyzer" field of a
                createCollection's "$lexical.analyzer" value, or a ready-made
                astrapy `CollectionLexicalOptions` object.
            hybrid_search: whether similarity searches should be run as Hybrid searches
                or not. Values are DEFAULT, ON or OFF. In case of DEFAULT, searches
                are performed as permitted by the collection configuration, with a
                preference for hybrid search. Forcing this setting to ON for a
                non-hybrid-enabled collection would result in a server error when
                running searches.
            hybrid_limit_factor: subsearch "limit" specification for hybrid searches.
                If omitted, hybrid searches do not specify it and leave the Data API
                to use its defaults.
                If a floating-point positive number is provided: each subsearch
                participating in the hybrid search (i.e. both the vector-based ANN
                and the lexical-based) will be requested to fecth up to
                `int(k*hybrid_limit_factor)` items, where `k` is the desired result
                count from the whole search.
                If a `HybridLimitFactorPrescription` is provided (see the class
                docstring for details), separate factors are applied to the vector
                and the lexical subsearches. Alternatively, a simple dictionary
                with keys "$lexical" and "$vector" achieves the same effect.

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
        self.has_lexical: bool
        self.has_hybrid: bool
        self.hybrid_search: bool  # affecting the actual behaviour when running searches
        self.hybrid_limit_factor: None | float | dict[str, float]
        self.collection_reranking_api_key = collection_reranking_api_key

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

            self.has_lexical = (
                collection_lexical.enabled
                if isinstance(collection_lexical, CollectionLexicalOptions)
                else (collection_lexical is not None)
            )
            _has_reranking = (
                collection_rerank.enabled
                if isinstance(collection_rerank, CollectionRerankOptions)
                else (collection_rerank is not None)
            )
            self.has_hybrid = self.has_lexical and _has_reranking

            self.hybrid_search = _decide_hybrid_search_setting(
                required_hybrid_search=hybrid_search,
                has_hybrid=self.has_hybrid,
            )

            if self.collection_vector_service_options is not None:
                self.document_codec = _DefaultVectorizeVSDocumentCodec(
                    ignore_invalid_documents=ignore_invalid_documents,
                    has_lexical=self.has_lexical,
                )
            else:
                self.document_codec = _DefaultVSDocumentCodec(
                    content_field=_content_field,
                    ignore_invalid_documents=ignore_invalid_documents,
                    has_lexical=self.has_lexical,
                )
            # indexing policy setting
            self.indexing_policy = self._normalize_metadata_indexing_policy(
                metadata_indexing_include=metadata_indexing_include,
                metadata_indexing_exclude=metadata_indexing_exclude,
                collection_indexing_policy=collection_indexing_policy,
                document_codec=self.document_codec,
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
                collection_lexical=collection_lexical,
                collection_rerank=collection_rerank,
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
            )
            if c_descriptor is None:
                msg = f"Collection '{self.collection_name}' not found."
                raise ValueError(msg)
            # use the collection info to set the store properties
            c_vector_options = c_descriptor.definition.as_dict().get("vector") or {}
            if not c_vector_options:
                msg = "Non-vector collection detected."
                raise ValueError(msg)
            _embedding_dimension = c_vector_options.get("dimension")
            self.collection_vector_service_options = c_vector_options.get("service")
            has_vectorize = self.collection_vector_service_options is not None
            logger.info("vector store autodetect: has_vectorize = %s", has_vectorize)
            norm_content_field = _normalize_content_field(
                content_field,
                is_autodetect=True,
                has_vectorize=has_vectorize,
            )

            self.has_lexical = (
                c_descriptor.definition.lexical is not None
                and c_descriptor.definition.lexical.enabled
            )

            self.document_codec = _detect_document_codec(
                c_documents,
                has_vectorize=has_vectorize,
                has_lexical=self.has_lexical,
                ignore_invalid_documents=ignore_invalid_documents,
                norm_content_field=norm_content_field,
            )
            self.indexing_policy = self._normalize_metadata_indexing_policy(
                metadata_indexing_include=None,
                metadata_indexing_exclude=None,
                collection_indexing_policy=c_descriptor.definition.indexing,
                document_codec=self.document_codec,
            )

            _has_reranking = (
                c_descriptor.definition.rerank is not None
                and c_descriptor.definition.rerank.enabled
            )
            self.has_hybrid = self.has_lexical and _has_reranking

            self.hybrid_search = _decide_hybrid_search_setting(
                required_hybrid_search=hybrid_search,
                has_hybrid=self.has_hybrid,
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

        self.hybrid_limit_factor = _normalize_hybrid_limit_factor(hybrid_limit_factor)

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
            default_indexing_policy=(
                self.document_codec.default_collection_indexing_policy
            ),
            collection_vector_service_options=self.collection_vector_service_options,
            collection_embedding_api_key=self.collection_embedding_api_key,
            ext_callers=ext_callers,
            component_name=component_name,
            collection_rerank=collection_rerank,
            collection_reranking_api_key=self.collection_reranking_api_key,
            collection_lexical=collection_lexical,
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

    def copy(
        self,
        *,
        token: str | TokenProvider | None = None,
        ext_callers: list[tuple[str | None, str | None] | str | None] | None = None,
        component_name: str | None = None,
        collection_embedding_api_key: str | EmbeddingHeadersProvider | None = None,
        collection_reranking_api_key: str | RerankingHeadersProvider | None = None,
    ) -> AstraDBVectorStore:
        """Create a copy, possibly with changed attributes.

        This method creates a shallow copy of this environment. If a parameter
        is passed and differs from None, it will replace the corresponding value
        in the copy.

        The method allows changing only the parameters that ensure the copy is
        functional and does not trigger side-effects:
        for example, one cannot create a copy acting on a new collection.
        In those cases, one should create a new instance of ``AstraDBVectorStore``
        from scratch.

        Attributes:
            token: API token for Astra DB usage, either in the form of a string
                or a subclass of ``astrapy.authentication.TokenProvider``.
                In order to suppress token usage in the copy, explicitly pass
                ``astrapy.authentication.StaticTokenProvider(None)``.
            ext_callers: additional custom (caller_name, caller_version) pairs
                to attach to the User-Agent header when issuing Data API requests.
            component_name: a value for the LangChain component name to use when
                identifying the originator of the Data API requests.
            collection_embedding_api_key: the API Key to supply in each Data API
                request if necessary. This is necessary if using the Vectorize
                feature and no secret is stored with the database.
                In order to suppress the API Key in the copy, explicitly pass
                ``astrapy.authentication.EmbeddingAPIKeyHeaderProvider(None)``.
            collection_reranking_api_key: for usage of server-side reranking services
                within Astra DB. With this parameter one can supply an API Key
                that will be passed to Astra DB with each data request.
                This parameter can be either a string or a subclass of
                ``astrapy.authentication.RerankingHeadersProvider``.
                This is useful when the service is configured for the collection,
                but no corresponding secret is stored within
                Astra's key management system.
        """
        copy = AstraDBVectorStore(
            collection_name="moot",
            api_endpoint="http://moot",
            environment=Environment.OTHER,
            namespace="moot",
            setup_mode=SetupMode.OFF,
            collection_vector_service_options=VectorServiceOptions(
                provider="moot",
                model_name="moot",
            ),
        )
        copy.collection_name = self.collection_name
        copy.token = self.token if token is None else token
        copy.api_endpoint = self.api_endpoint
        copy.environment = self.environment
        copy.namespace = self.namespace
        copy.indexing_policy = self.indexing_policy
        copy.autodetect_collection = self.autodetect_collection
        copy.embedding_dimension = self.embedding_dimension
        copy.embedding = self.embedding
        copy.metric = self.metric
        copy.collection_embedding_api_key = (
            self.collection_embedding_api_key
            if collection_embedding_api_key is None
            else collection_embedding_api_key
        )
        copy.collection_reranking_api_key = (
            self.collection_reranking_api_key
            if collection_reranking_api_key is None
            else collection_reranking_api_key
        )
        copy.collection_vector_service_options = self.collection_vector_service_options
        copy.document_codec = self.document_codec
        copy.has_lexical = self.has_lexical
        copy.has_hybrid = self.hybrid_search
        copy.hybrid_limit_factor = self.hybrid_limit_factor
        copy.batch_size = self.batch_size
        copy.bulk_insert_batch_concurrency = self.bulk_insert_batch_concurrency
        copy.bulk_insert_overwrite_concurrency = self.bulk_insert_overwrite_concurrency
        copy.bulk_delete_concurrency = self.bulk_delete_concurrency
        # Now the .astra_env attribute:
        copy.astra_env = self.astra_env.copy(
            token=token,
            ext_callers=ext_callers,
            component_name=component_name,
            collection_embedding_api_key=collection_embedding_api_key,
            collection_reranking_api_key=collection_reranking_api_key,
        )

        return copy

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
        deletion_response = self.astra_env.collection.delete_one(
            self.document_codec.encode_query(ids=[document_id]),
        )
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
            self.document_codec.encode_query(ids=[document_id]),
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
            self.document_codec.get_id,
        )[::-1]

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
            The allowed field names for the metadata document attributes must
            obey certain rules (such as: keys cannot start with a dollar sign
            and cannot be empty).
            See `Naming Conventions <https://docs.datastax.com/en/astra-db-serverless/api-reference/dataapiclient.html#naming-conventions>`_
            for details.

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
        ids_to_replace: list[str]
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
        except CollectionInsertManyException as err:
            # check that the error is solely due to already-existing documents
            if any(
                not isinstance(in_err, DataAPIResponseException)
                for in_err in err.exceptions
            ):
                full_err_message = _insertmany_error_message(err)
                raise AstraDBVectorStoreError(full_err_message) from err
            # here, assume all in err.exceptions is a DataAPIResponseException:
            error_codes = {
                err_desc.error_code
                for in_err in cast(list[DataAPIResponseException], err.exceptions)
                for err_desc in in_err.error_descriptors
            }
            if error_codes == {DOCUMENT_ALREADY_EXISTS_API_ERROR_CODE}:
                inserted_ids = err.inserted_ids
                inserted_ids_set = set(inserted_ids)
                ids_to_replace = [
                    doc_id
                    for document in documents_to_insert
                    if (doc_id := self.document_codec.get_id(document))
                    not in inserted_ids_set
                ]
            else:
                full_err_message = _insertmany_error_message(err)
                raise AstraDBVectorStoreError(full_err_message) from err

        # if necessary, replace docs for the non-inserted ids
        if ids_to_replace:
            documents_to_replace = [
                document
                for document in documents_to_insert
                if self.document_codec.get_id(document) in ids_to_replace
            ]

            _max_workers = (
                overwrite_concurrency or self.bulk_insert_overwrite_concurrency
            )
            with ThreadPoolExecutor(
                max_workers=_max_workers,
            ) as executor:

                def _replace_document(
                    document: DocDict,
                ) -> tuple[CollectionUpdateResult, str]:
                    doc_id = self.document_codec.get_id(document)
                    return self.astra_env.collection.replace_one(
                        self.document_codec.encode_query(ids=[doc_id]),
                        document,
                    ), doc_id

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
                raise AstraDBVectorStoreError(msg)
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
            The allowed field names for the metadata document attributes must
            obey certain rules (such as: keys cannot start with a dollar sign
            and cannot be empty).
            See `Naming Conventions <https://docs.datastax.com/en/astra-db-serverless/api-reference/dataapiclient.html#naming-conventions>`_
            for details.

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
        ids_to_replace: list[str]
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
        except CollectionInsertManyException as err:
            # check that the error is solely due to already-existing documents
            if any(
                not isinstance(in_err, DataAPIResponseException)
                for in_err in err.exceptions
            ):
                full_err_message = _insertmany_error_message(err)
                raise AstraDBVectorStoreError(full_err_message) from err
            # here, assume all in err.exceptions is a DataAPIResponseException:
            error_codes = {
                err_desc.error_code
                for in_err in cast(list[DataAPIResponseException], err.exceptions)
                for err_desc in in_err.error_descriptors
            }
            if error_codes == {DOCUMENT_ALREADY_EXISTS_API_ERROR_CODE}:
                inserted_ids = err.inserted_ids
                inserted_ids_set = set(inserted_ids)
                ids_to_replace = [
                    doc_id
                    for document in documents_to_insert
                    if (doc_id := self.document_codec.get_id(document))
                    not in inserted_ids_set
                ]
            else:
                full_err_message = _insertmany_error_message(err)
                raise AstraDBVectorStoreError(full_err_message) from err

        # if necessary, replace docs for the non-inserted ids
        if ids_to_replace:
            documents_to_replace = [
                document
                for document in documents_to_insert
                if self.document_codec.get_id(document) in ids_to_replace
            ]

            sem = asyncio.Semaphore(
                overwrite_concurrency or self.bulk_insert_overwrite_concurrency,
            )

            _async_collection = self.astra_env.async_collection

            async def _replace_document(
                document: DocDict,
            ) -> tuple[CollectionUpdateResult, str]:
                async with sem:
                    doc_id = self.document_codec.get_id(document)
                    return await _async_collection.replace_one(
                        self.document_codec.encode_query(ids=[doc_id]),
                        document,
                    ), doc_id

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
                raise AstraDBVectorStoreError(msg)
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
            ) -> CollectionUpdateResult:
                document_id, update_metadata = id_md_pair
                encoded_metadata = self.filter_to_query(update_metadata)
                return self.astra_env.collection.update_one(
                    self.document_codec.encode_query(ids=[document_id]),
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
        ) -> CollectionUpdateResult:
            document_id, update_metadata = id_md_pair
            encoded_metadata = self.filter_to_query(update_metadata)
            async with sem:
                return await _async_collection.update_one(
                    self.document_codec.encode_query(ids=[document_id]),
                    {"$set": encoded_metadata},
                )

        tasks = [
            asyncio.create_task(_update_document(id_md_pair))
            for id_md_pair in id_to_metadata.items()
        ]

        update_results = await asyncio.gather(*tasks, return_exceptions=False)

        return sum(u_res.update_info["n"] for u_res in update_results)

    def full_decode_astra_db_found_document(
        self,
        astra_db_document: DocDict,
    ) -> AstraDBQueryResult | None:
        """Decode an Astra DB document in full, i.e. into Document+embedding/similarity.

        This operation returns a representation that is independent of the codec
        being used in the collection (whereas the input, a 'raw' Astra DB document,
        is codec-dependent).

        The input raw document can carry information on embedding and similarity,
        depending on details of the query used to retrieve it. These can be set
        to None in the resulf if not found.

        The whole method can return a None, to signal that the codec has refused
        the conversion (e.g. because the input document is deemed faulty).

        Args:
            astra_db_document: a dictionary obtained through `run_query_raw` from
                the collection.

        Returns:
            a AstraDBQueryResult named tuple with Document, id, embedding
                (where applicable) and similarity (where applicable),
                or an overall None if the decoding is refused by the codec.
        """
        decoded = self.document_codec.decode(astra_db_document)
        if decoded is not None:
            doc_id = self.document_codec.get_id(astra_db_document)
            doc_embedding = self.document_codec.decode_vector(astra_db_document)
            doc_similarity = self.document_codec.get_similarity(astra_db_document)
            return AstraDBQueryResult(
                document=decoded,
                id=doc_id,
                embedding=doc_embedding,
                similarity=doc_similarity,
            )
        return None

    def full_decode_astra_db_reranked_result(
        self,
        astra_db_reranked_result: RerankedResult[DocDict],
    ) -> AstraDBQueryResult | None:
        """Full-decode an Astra DB find-and-rerank hit (Document+embedding/similarity).

        This operation returns a representation that is independent of the codec
        being used in the collection (whereas the 'document' part of the input,
        a 'raw' Astra DB response from a find-and-rerank hybrid search, is
        codec-dependent).

        The input raw document is what the find_and_rerank Astrapy method returns,
        i.e. an iterable over RerankedResult objects. Missing entries (such as
        the embedding) are  set to None in the resulf if not found.

        The whole method can return a None, to signal that the codec has refused
        the conversion (e.g. because the input document is deemed faulty).

        Args:
            astra_db_reranked_result: a RerankedResult obtained by a `find_and_rerank`
                method call on the collection.

        Returns:
            a AstraDBQueryResult named tuple with Document, id, embedding
                (where applicable) and similarity (where applicable),
                or an overall None if the decoding is refused by the codec.
        """
        astra_db_document = astra_db_reranked_result.document
        astra_db_scores = astra_db_reranked_result.scores
        decoded = self.document_codec.decode(astra_db_document)
        if decoded is not None:
            doc_id = self.document_codec.get_id(astra_db_document)
            doc_embedding = self.document_codec.decode_vector(astra_db_document)
            doc_similarity = astra_db_scores.get(RERANK_SCORE_KEY)
            return AstraDBQueryResult(
                document=decoded,
                id=doc_id,
                embedding=doc_embedding,
                similarity=doc_similarity,
            )
        return None

    @overload
    def run_query_raw(
        self,
        *,
        n: int,
        include_sort_vector: Literal[False] = False,
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,
        sort: dict[str, Any] | None = None,
        include_similarity: bool | None = None,
        include_embeddings: bool = False,
    ) -> Iterable[DocDict]: ...

    @overload
    def run_query_raw(
        self,
        *,
        n: int,
        include_sort_vector: Literal[True],
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,
        sort: dict[str, Any] | None = None,
        include_similarity: bool | None = None,
        include_embeddings: bool = False,
    ) -> tuple[list[float] | None, Iterable[DocDict]]: ...

    def run_query_raw(
        self,
        *,
        n: int,
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,  # noqa: A002
        sort: dict[str, Any] | None = None,
        include_similarity: bool | None = None,
        include_sort_vector: bool = False,
        include_embeddings: bool = False,
    ) -> tuple[list[float] | None, Iterable[DocDict]] | Iterable[DocDict]:
        """Execute a generic query on stored documents, returning Astra DB documents.

        The return value has a variable format, depending on whether the 'sort vector'
        is requested back from the server.

        Only the `n` parameter is required. Omitting all other parameters results
        in a query that matches each and every document found on the collection.

        The method does not expose a projection directly, which is instead
        automatically determined based on the invocation options.

        The returned documents are exactly as they come back from Astra DB (taking
        into account the projection as well). A further step, namely subsequent
        invocation of the `convert_astra_db_document` method, is required to reconstruct
        codec-independent Document objects.
        The reason for keeping the retrieval and the decoding steps separate is that
        a caller may want to first deduplicate/discard items, in order to convert only
        the items actually needed.

        Args:
            n: amount of items to return. Fewer items than `n` may be returned  if
                the collection has not enough matches.
            ids: a list of document IDs to restrict the query to. If this is supplied,
                only document with an ID among the provided one will match. If further
                query filters are provided (i.e. metadata), matches must satisfy both
                requirements.
            filter: a metadata filtering part. If provided, it must refer to
                metadata keys by their bare name (such as `{"key": 123}`).
                This filter can combine nested conditions with "$or"/"$and" connectors,
                for example:
                - `{"tag": "a"}`
                - `{"$or": [{"tag": "a"}, "label": "b"]}`
                - `{"$and": [{"tag": {"$in": ["a", "z"]}}, "label": "b"]}`
            sort: a 'sort' clause for the query, such as `{"$vector": [...]}`,
                `{"$vectorize": "..."}` or `{"mdkey": 1}`. Metadata sort conditions
                must be expressed by their 'bare' name.
            include_similarity: whether to return similarity scores with each match.
                Requires vector sort.
            include_sort_vector: whether to return the very query vector used for the
                ANN search alongside the iterable of results. Requires vector sort.
                Note that the shape of the return value depends on this parameter.
            include_embeddings: whether to retrieve the matches' own embedding vectors.

        Returns:
            The shape of the return value depends on the value of `include_sort_vector`:
            * if `include_sort_vector = False`, the return value is an iterable over
                Astra DB documents (dictionaries);
            * if `include_sort_vector = True`, the return value is a 2-item tuple
                `(sort_v, astra_db_ite)` tuple, where:
                - `sort_v` is the sort vector, if requested, or None if not available.
                - `astra_db_ite` is an iterable over Astra DB documents (dictionaries).
        """
        self.astra_env.ensure_db_setup()

        find_query = self.document_codec.encode_query(
            ids=ids,
            filter_dict=filter,
        )
        find_sort = self.document_codec.encode_filter(sort or {})
        find_projection = (
            self.document_codec.full_projection
            if include_embeddings
            else self.document_codec.base_projection
        )

        find_raw_iterator = self.astra_env.collection.find(
            filter=find_query,
            projection=find_projection,
            limit=n,
            include_similarity=include_similarity,
            include_sort_vector=include_sort_vector,
            sort=find_sort,
        )
        # stripping down the Astra DB cursor details into a plain iterator:
        final_doc_iterator = (doc for doc in find_raw_iterator)
        if include_sort_vector:
            # the codec option in the AstraDBEnv class disables DataAPIVectors here:
            sort_vector = cast(
                Union[list[float], None],
                (find_raw_iterator.get_sort_vector() if include_sort_vector else None),
            )
            return sort_vector, final_doc_iterator
        return final_doc_iterator

    @overload
    def run_query(
        self,
        *,
        n: int,
        include_sort_vector: Literal[False] = False,
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,
        sort: dict[str, Any] | None = None,
        include_similarity: bool | None = None,
        include_embeddings: bool = False,
    ) -> Iterable[AstraDBQueryResult]: ...

    @overload
    def run_query(
        self,
        *,
        n: int,
        include_sort_vector: Literal[True],
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,
        sort: dict[str, Any] | None = None,
        include_similarity: bool | None = None,
        include_embeddings: bool = False,
    ) -> tuple[list[float] | None, Iterable[AstraDBQueryResult]]: ...

    def run_query(
        self,
        *,
        n: int,
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,  # noqa: A002
        sort: dict[str, Any] | None = None,
        include_similarity: bool | None = None,
        include_sort_vector: bool = False,
        include_embeddings: bool = False,
    ) -> (
        tuple[list[float] | None, Iterable[AstraDBQueryResult]]
        | Iterable[AstraDBQueryResult]
    ):
        """Execute a generic query on stored documents, returning Documents+other info.

        The return value has a variable format, depending on whether the 'sort vector'
        is requested back from the server.

        Only the `n` parameter is required. Omitting all other parameters results
        in a query that matches each and every document found on the collection.

        The method does not expose a projection directly, which is instead
        automatically determined based on the invocation options.

        The returned Document objects are codec-independent.

        Args:
            n: amount of items to return. Fewer items than `n` may be returned in the
                following cases: (a) the decoding skips some raw entries from
                the server; (b) the collection has not enough matches.
            ids: a list of document IDs to restrict the query to. If this is supplied,
                only document with an ID among the provided one will match. If further
                query filters are provided (i.e. metadata), matches must satisfy both
                requirements.
            filter: a metadata filtering part. If provided, it must refer to
                metadata keys by their bare name (such as `{"key": 123}`).
                This filter can combine nested conditions with "$or"/"$and" connectors,
                for example:
                - `{"tag": "a"}`
                - `{"$or": [{"tag": "a"}, "label": "b"]}`
                - `{"$and": [{"tag": {"$in": ["a", "z"]}}, "label": "b"]}`
            sort: a 'sort' clause for the query, such as `{"$vector": [...]}`,
                `{"$vectorize": "..."}` or `{"mdkey": 1}`. Metadata sort conditions
                must be expressed by their 'bare' name.
            include_similarity: whether to return similarity scores with each match.
                Requires vector sort.
            include_sort_vector: whether to return the very query vector used for the
                ANN search alongside the iterable of results. Requires vector sort.
                Note that the shape of the return value depends on this parameter.
            include_embeddings: whether to retrieve the matches' own embedding vectors.

        Returns:
            The shape of the return value depends on the value of `include_sort_vector`:
            * if `include_sort_vector = False`, the return value is an iterable over
                the AstraDBQueryResult items returned by the query. Entries that fail
                the decoding step, if any, are discarded after the query, which may
                lead to fewer items being returned than the required `n`.
            * if `include_sort_vector = True`, the return value is a 2-item tuple
                `(sort_v, results_ite)` tuple, where:
                - `sort_v` is the sort vector, if requested, or None if not available.
                - `results_ite` is an iterable over AstraDBQueryResult items as above.
        """
        if include_sort_vector:
            query_v, astra_docs_ite = self.run_query_raw(
                n=n,
                ids=ids,
                filter=filter,
                sort=sort,
                include_similarity=include_similarity,
                include_sort_vector=True,
                include_embeddings=include_embeddings,
            )
            return (
                query_v,
                (
                    decoded_tuple
                    for astra_db_doc in astra_docs_ite
                    if (
                        decoded_tuple := self.full_decode_astra_db_found_document(
                            astra_db_doc,
                        )
                    )
                    is not None
                ),
            )
        astra_docs_ite = self.run_query_raw(
            n=n,
            ids=ids,
            filter=filter,
            sort=sort,
            include_similarity=include_similarity,
            include_sort_vector=False,
            include_embeddings=include_embeddings,
        )
        return (
            decoded_tuple
            for astra_db_doc in astra_docs_ite
            if (
                decoded_tuple := self.full_decode_astra_db_found_document(
                    astra_db_doc,
                )
            )
            is not None
        )

    @overload
    async def arun_query_raw(
        self,
        *,
        n: int,
        include_sort_vector: Literal[False] = False,
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,
        sort: dict[str, Any] | None = None,
        include_similarity: bool | None = None,
        include_embeddings: bool = False,
    ) -> AsyncIterable[DocDict]: ...

    @overload
    async def arun_query_raw(
        self,
        *,
        n: int,
        include_sort_vector: Literal[True],
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,
        sort: dict[str, Any] | None = None,
        include_similarity: bool | None = None,
        include_embeddings: bool = False,
    ) -> tuple[list[float] | None, AsyncIterable[DocDict]]: ...

    async def arun_query_raw(
        self,
        *,
        n: int,
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,  # noqa: A002
        sort: dict[str, Any] | None = None,
        include_similarity: bool | None = None,
        include_sort_vector: bool = False,
        include_embeddings: bool = False,
    ) -> tuple[list[float] | None, AsyncIterable[DocDict]] | AsyncIterable[DocDict]:
        """Execute a generic query on stored documents, returning Astra DB documents.

        The return value has a variable format, depending on whether the 'sort vector'
        is requested back from the server.

        Only the `n` parameter is required. Omitting all other parameters results
        in a query that matches each and every document found on the collection.

        The method does not expose a projection directly, which is instead
        automatically determined based on the invocation options.

        The returned documents are exactly as they come back from Astra DB (taking
        into account the projection as well). A further step, namely subsequent
        invocation of the `convert_astra_db_document` method, is required to reconstruct
        codec-independent Document objects.
        The reason for keeping the retrieval and the decoding steps separate is that
        a caller may want to first deduplicate/discard items, in order to convert only
        the items actually needed.

        Args:
            n: amount of items to return. Fewer items than `n` may be returned in the
                following cases: (a) the decoding skips some raw entries from
                the server; (b) the collection has not enough matches.
            ids: a list of document IDs to restrict the query to. If this is supplied,
                only document with an ID among the provided one will match. If further
                query filters are provided (i.e. metadata), matches must satisfy both
                requirements.
            filter: a metadata filtering part. If provided, it must refer to
                metadata keys by their bare name (such as `{"key": 123}`).
                This filter can combine nested conditions with "$or"/"$and" connectors,
                for example:
                - `{"tag": "a"}`
                - `{"$or": [{"tag": "a"}, "label": "b"]}`
                - `{"$and": [{"tag": {"$in": ["a", "z"]}}, "label": "b"]}`
            sort: a 'sort' clause for the query, such as `{"$vector": [...]}`,
                `{"$vectorize": "..."}` or `{"mdkey": 1}`. Metadata sort conditions
                must be expressed by their 'bare' name.
            include_similarity: whether to return similarity scores with each match.
                Requires vector sort.
            include_sort_vector: whether to return the very query vector used for the
                ANN search alongside the iterable of results. Requires vector sort.
                Note that the shape of the return value depends on this parameter.
            include_embeddings: whether to retrieve the matches' own embedding vectors.

        Returns:
            The shape of the return value depends on the value of `include_sort_vector`:
            * if `include_sort_vector = False`, the return value is an iterable over
                Astra DB documents (dictionaries);
            * if `include_sort_vector = True`, the return value is a 2-item tuple
                `(sort_v, astra_db_ite)` tuple, where:
                - `sort_v` is the sort vector, if requested, or None if not available.
                - `astra_db_ite` is an iterable over Astra DB documents (dictionaries).
        """
        await self.astra_env.aensure_db_setup()

        find_query = self.document_codec.encode_query(
            ids=ids,
            filter_dict=filter,
        )
        find_sort = self.document_codec.encode_filter(sort or {})
        find_projection = (
            self.document_codec.full_projection
            if include_embeddings
            else self.document_codec.base_projection
        )

        find_raw_iterator = self.astra_env.async_collection.find(
            filter=find_query,
            projection=find_projection,
            limit=n,
            include_similarity=include_similarity,
            include_sort_vector=include_sort_vector,
            sort=find_sort,
        )
        # stripping down the Astra DB cursor details into a plain iterator:
        final_doc_iterator = (doc async for doc in find_raw_iterator)
        if include_sort_vector:
            # the codec option in the AstraDBEnv class disables DataAPIVectors here:
            sort_vector = cast(
                Union[list[float], None],
                (
                    await find_raw_iterator.get_sort_vector()
                    if include_sort_vector
                    else None
                ),
            )
            return sort_vector, final_doc_iterator
        return final_doc_iterator

    @overload
    async def arun_query(
        self,
        *,
        n: int,
        include_sort_vector: Literal[False] = False,
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,
        sort: dict[str, Any] | None = None,
        include_similarity: bool | None = None,
        include_embeddings: bool = False,
    ) -> AsyncIterable[AstraDBQueryResult]: ...

    @overload
    async def arun_query(
        self,
        *,
        n: int,
        include_sort_vector: Literal[True],
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,
        sort: dict[str, Any] | None = None,
        include_similarity: bool | None = None,
        include_embeddings: bool = False,
    ) -> tuple[list[float] | None, AsyncIterable[AstraDBQueryResult]]: ...

    async def arun_query(
        self,
        *,
        n: int,
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,  # noqa: A002
        sort: dict[str, Any] | None = None,
        include_similarity: bool | None = None,
        include_sort_vector: bool = False,
        include_embeddings: bool = False,
    ) -> (
        tuple[list[float] | None, AsyncIterable[AstraDBQueryResult]]
        | AsyncIterable[AstraDBQueryResult]
    ):
        """Execute a generic query on stored documents, returning Documents+other info.

        The return value has a variable format, depending on whether the 'sort vector'
        is requested back from the server.

        Only the `n` parameter is required. Omitting all other parameters results
        in a query that matches each and every document found on the collection.

        The method does not expose a projection directly, which is instead
        automatically determined based on the invocation options.

        The returned Document objects are codec-independent.

        Args:
            n: amount of items to return. Fewer items than `n` may be returned  if
                the collection has not enough matches.
            ids: a list of document IDs to restrict the query to. If this is supplied,
                only document with an ID among the provided one will match. If further
                query filters are provided (i.e. metadata), matches must satisfy both
                requirements.
            filter: a metadata filtering part. If provided, it must refer to
                metadata keys by their bare name (such as `{"key": 123}`).
                This filter can combine nested conditions with "$or"/"$and" connectors,
                for example:
                - `{"tag": "a"}`
                - `{"$or": [{"tag": "a"}, "label": "b"]}`
                - `{"$and": [{"tag": {"$in": ["a", "z"]}}, "label": "b"]}`
            sort: a 'sort' clause for the query, such as `{"$vector": [...]}`,
                `{"$vectorize": "..."}` or `{"mdkey": 1}`. Metadata sort conditions
                must be expressed by their 'bare' name.
            include_similarity: whether to return similarity scores with each match.
                Requires vector sort.
            include_sort_vector: whether to return the very query vector used for the
                ANN search alongside the iterable of results. Requires vector sort.
                Note that the shape of the return value depends on this parameter.
            include_embeddings: whether to retrieve the matches' own embedding vectors.

        Returns:
            The shape of the return value depends on the value of `include_sort_vector`:
            * if `include_sort_vector = False`, the return value is an iterable over
                the AstraDBQueryResult items returned by the query. Entries that fail
                the decoding step, if any, are discarded after the query, which may
                lead to fewer items being returned than the required `n`.
            * if `include_sort_vector = True`, the return value is a 2-item tuple
                `(sort_v, results_ite)` tuple, where:
                - `sort_v` is the sort vector, if requested, or None if not available.
                - `results_ite` is an iterable over AstraDBQueryResult items as above.
        """
        if include_sort_vector:
            query_v, astra_docs_ite = await self.arun_query_raw(
                n=n,
                ids=ids,
                filter=filter,
                sort=sort,
                include_similarity=include_similarity,
                include_sort_vector=True,
                include_embeddings=include_embeddings,
            )
            return (
                query_v,
                (
                    decoded_tuple
                    async for astra_db_doc in astra_docs_ite
                    if (
                        decoded_tuple := self.full_decode_astra_db_found_document(
                            astra_db_doc,
                        )
                    )
                    is not None
                ),
            )
        astra_docs_ite = await self.arun_query_raw(
            n=n,
            ids=ids,
            filter=filter,
            sort=sort,
            include_similarity=include_similarity,
            include_sort_vector=False,
            include_embeddings=include_embeddings,
        )
        return (
            decoded_tuple
            async for astra_db_doc in astra_docs_ite
            if (
                decoded_tuple := self.full_decode_astra_db_found_document(
                    astra_db_doc,
                )
            )
            is not None
        )

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
        docs_ite = self.run_query(n=n, filter=filter)
        return [doc for doc, _, _, _ in docs_ite]

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
        docs_ite = await self.arun_query(n=n, filter=filter)
        return [doc async for doc, _, _, _ in docs_ite]

    def get_by_document_id(self, document_id: str) -> Document | None:
        """Retrieve a single document from the store, given its document ID.

        Args:
            document_id: The document ID

        Returns:
            The the document if it exists. Otherwise None.
        """
        hits_ite = self.run_query(
            n=1,
            ids=[document_id],
        )
        hits = [doc for doc, _, _, _ in hits_ite]
        if hits:
            return hits[0]

        return None

    async def aget_by_document_id(self, document_id: str) -> Document | None:
        """Retrieve a single document from the store, given its document ID.

        Args:
            document_id: The document ID

        Returns:
            The the document if it exists. Otherwise None.
        """
        hits_ite = await self.arun_query(
            n=1,
            ids=[document_id],
        )
        hits = [doc async for doc, _, _, _ in hits_ite]
        if hits:
            return hits[0]

        return None

    @override
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        lexical_query: str | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to query.

        Args:
            query: Query to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            lexical_query: for hybrid search, a specific query for the lexical
                portion of the retrieval. If omitted or empty, defaults to the same
                as 'query'. If passed on a non-hybrid search, an error is raised.
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
                lexical_query=lexical_query,
            )
        ]

    @override
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        lexical_query: str | None = None,
    ) -> list[tuple[Document, float]]:
        """Return docs most similar to query with score.

        Args:
            query: Query to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            lexical_query: for hybrid search, a specific query for the lexical
                portion of the retrieval. If omitted or empty, defaults to the same
                as 'query'. If passed on a non-hybrid search, an error is raised.

        Returns:
            The list of (Document, score), the most similar to the query vector.
        """
        return [
            (doc, score)
            for (doc, score, _) in self.similarity_search_with_score_id(
                query=query,
                k=k,
                filter=filter,
                lexical_query=lexical_query,
            )
        ]

    def similarity_search_with_score_id(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,  # noqa: A002
        lexical_query: str | None = None,
    ) -> list[tuple[Document, float, str]]:
        """Return docs most similar to the query with score and id.

        Args:
            query: Query to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            lexical_query: for hybrid search, a specific query for the lexical
                portion of the retrieval. If omitted or empty, defaults to the same
                as 'query'. If passed on a non-hybrid search, an error is raised.

        Returns:
            The list of (Document, score, id), the most similar to the query.
        """
        sort: dict[str, Any]

        if self.hybrid_search:
            rerank_on = self.document_codec.rerank_on
            rerank_query: str | None
            if self.document_codec.server_side_embeddings:
                sort = self.document_codec.encode_hybrid_sort(
                    vector=None,
                    vectorize=query,
                    lexical=lexical_query or query,
                )
                rerank_query = None
            else:
                embedding_vector = self._get_safe_embedding().embed_query(query)
                sort = self.document_codec.encode_hybrid_sort(
                    vector=embedding_vector,
                    vectorize=None,
                    lexical=lexical_query or query,
                )
                rerank_query = query

            return self._hybrid_search_with_score_id_by_sort(
                sort=sort,
                k=k,
                filter_dict=filter,
                rerank_on=rerank_on,
                rerank_query=rerank_query,
            )

        if lexical_query is not None:
            raise ValueError(ERROR_LEXICAL_QUERY_ON_NONHYBRID_SEARCH)
        if self.document_codec.server_side_embeddings:
            sort = self.document_codec.encode_vectorize_sort(query)
        else:
            embedding_vector = self._get_safe_embedding().embed_query(query)
            sort = self.document_codec.encode_vector_sort(embedding_vector)

        return self._similarity_find_with_score_id_by_sort(
            sort=sort,
            k=k,
            filter_dict=filter,
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
        sort = self.document_codec.encode_vector_sort(embedding)
        return self._similarity_find_with_score_id_by_sort(
            sort=sort,
            k=k,
            filter_dict=filter,
        )

    def _similarity_find_with_score_id_by_sort(
        self,
        sort: dict[str, Any],
        k: int,
        filter_dict: dict[str, Any] | None,
    ) -> list[tuple[Document, float, str]]:
        """Run ANN search with a provided sort clause."""
        hits_ite = self.run_query(
            n=k,
            filter=filter_dict,
            sort=sort,
            include_similarity=True,
        )
        # doc is a Document and sim is a float:
        return [
            cast(tuple[Document, float, str], (doc, sim, did))
            for doc, did, _, sim in hits_ite
        ]

    def _hybrid_search_with_score_id_by_sort(
        self,
        sort: dict[str, Any],
        k: int,
        filter_dict: dict[str, Any] | None,
        rerank_on: str | None,
        rerank_query: str | None,
    ) -> list[tuple[Document, float, str]]:
        """Run a hybrid search with a provided sort clause."""
        self.astra_env.ensure_db_setup()
        encoded_filter = self.document_codec.encode_query(filter_dict=filter_dict)
        hybrid_limits = _make_hybrid_limits(self.hybrid_limit_factor, k)
        hybrid_reranked_results = self.astra_env.collection.find_and_rerank(
            filter=encoded_filter,
            sort=sort,
            projection=self.document_codec.base_projection,
            limit=k,
            hybrid_limits=hybrid_limits,
            include_scores=True,
            rerank_on=rerank_on,
            rerank_query=rerank_query,
        )
        return [
            cast(
                tuple[Document, float, str],
                (
                    decoded_tuple.document,
                    decoded_tuple.similarity,
                    decoded_tuple.id,
                ),
            )
            for rrk_result in hybrid_reranked_results
            if (
                decoded_tuple := self.full_decode_astra_db_reranked_result(
                    rrk_result,
                )
            )
            is not None
        ]

    @override
    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        lexical_query: str | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to query.

        Args:
            query: Query to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            lexical_query: for hybrid search, a specific query for the lexical
                portion of the retrieval. If omitted or empty, defaults to the same
                as 'query'. If passed on a non-hybrid search, an error is raised.
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
                lexical_query=lexical_query,
            )
        ]

    @override
    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        lexical_query: str | None = None,
    ) -> list[tuple[Document, float]]:
        """Return docs most similar to query with score.

        Args:
            query: Query to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            lexical_query: for hybrid search, a specific query for the lexical
                portion of the retrieval. If omitted or empty, defaults to the same
                as 'query'. If passed on a non-hybrid search, an error is raised.

        Returns:
            The list of (Document, score), the most similar to the query vector.
        """
        return [
            (doc, score)
            for (doc, score, _) in await self.asimilarity_search_with_score_id(
                query=query,
                k=k,
                filter=filter,
                lexical_query=lexical_query,
            )
        ]

    async def asimilarity_search_with_score_id(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,  # noqa: A002
        lexical_query: str | None = None,
    ) -> list[tuple[Document, float, str]]:
        """Return docs most similar to the query with score and id.

        Args:
            query: Query to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            lexical_query: for hybrid search, a specific query for the lexical
                portion of the retrieval. If omitted or empty, defaults to the same
                as 'query'. If passed on a non-hybrid search, an error is raised.

        Returns:
            The list of (Document, score, id), the most similar to the query.
        """
        sort: dict[str, Any]

        if self.hybrid_search:
            rerank_on = self.document_codec.rerank_on
            rerank_query: str | None
            if self.document_codec.server_side_embeddings:
                sort = self.document_codec.encode_hybrid_sort(
                    vector=None,
                    vectorize=query,
                    lexical=lexical_query or query,
                )
                rerank_query = None
            else:
                embedding_vector = await self._get_safe_embedding().aembed_query(query)
                sort = self.document_codec.encode_hybrid_sort(
                    vector=embedding_vector,
                    vectorize=None,
                    lexical=lexical_query or query,
                )
                rerank_query = query

            return await self._ahybrid_search_with_score_id_by_sort(
                sort=sort,
                k=k,
                filter_dict=filter,
                rerank_on=rerank_on,
                rerank_query=rerank_query,
            )

        if lexical_query is not None:
            raise ValueError(ERROR_LEXICAL_QUERY_ON_NONHYBRID_SEARCH)
        if self.document_codec.server_side_embeddings:
            sort = self.document_codec.encode_vectorize_sort(query)
        else:
            embedding_vector = await self._get_safe_embedding().aembed_query(query)
            sort = self.document_codec.encode_vector_sort(embedding_vector)

        return await self._asimilarity_find_with_score_id_by_sort(
            sort=sort,
            k=k,
            filter_dict=filter,
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
        sort = self.document_codec.encode_vector_sort(embedding)
        return await self._asimilarity_find_with_score_id_by_sort(
            sort=sort,
            k=k,
            filter_dict=filter,
        )

    async def _asimilarity_find_with_score_id_by_sort(
        self,
        sort: dict[str, Any],
        k: int,
        filter_dict: dict[str, Any] | None,
    ) -> list[tuple[Document, float, str]]:
        """Run ANN search with a provided sort clause."""
        hits_ite = await self.arun_query(
            n=k,
            filter=filter_dict,
            sort=sort,
            include_similarity=True,
        )
        # doc is a Document and sim is a float:
        return [
            cast(tuple[Document, float, str], (doc, sim, did))
            async for doc, did, _, sim in hits_ite
        ]

    async def _ahybrid_search_with_score_id_by_sort(
        self,
        sort: dict[str, Any],
        k: int,
        filter_dict: dict[str, Any] | None,
        rerank_on: str | None,
        rerank_query: str | None,
    ) -> list[tuple[Document, float, str]]:
        """Run a hybrid search with a provided sort clause."""
        await self.astra_env.aensure_db_setup()
        encoded_filter = self.document_codec.encode_query(filter_dict=filter_dict)
        hybrid_limits = _make_hybrid_limits(self.hybrid_limit_factor, k)
        hybrid_reranked_results = self.astra_env.async_collection.find_and_rerank(
            filter=encoded_filter,
            sort=sort,
            projection=self.document_codec.base_projection,
            limit=k,
            hybrid_limits=hybrid_limits,
            include_scores=True,
            rerank_on=rerank_on,
            rerank_query=rerank_query,
        )
        return [
            cast(
                tuple[Document, float, str],
                (
                    decoded_tuple.document,
                    decoded_tuple.similarity,
                    decoded_tuple.id,
                ),
            )
            async for rrk_result in hybrid_reranked_results
            if (
                decoded_tuple := self.full_decode_astra_db_reranked_result(
                    rrk_result,
                )
            )
            is not None
        ]

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
        _, doc_emb_list = self._similarity_find_with_embedding_by_sort(
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
        _, doc_emb_list = await self._asimilarity_find_with_embedding_by_sort(
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
        if self.hybrid_search:
            warnings.warn(
                (
                    "Method `similarity_search_with_embedding` was called on a vector "
                    "store equipped with Hybrid capabilities. Since this method cannot "
                    "make use of Hybrid search, the vector store will fall back to "
                    "regular vector ANN similarity search."
                ),
                UserWarning,
                stacklevel=2,
            )

        if self.document_codec.server_side_embeddings:
            sort = self.document_codec.encode_vectorize_sort(query)
        else:
            query_embedding = self._get_safe_embedding().embed_query(text=query)
            # shortcut return if query isn't needed.
            if k == 0:
                return (query_embedding, [])
            sort = self.document_codec.encode_vector_sort(vector=query_embedding)

        return self._similarity_find_with_embedding_by_sort(
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
        if self.hybrid_search:
            warnings.warn(
                (
                    "Method `asimilarity_search_with_embedding` was called on a vector "
                    "store equipped with Hybrid capabilities. Since this method cannot "
                    "make use of Hybrid search, the vector store will fall back to "
                    "regular vector ANN similarity search."
                ),
                UserWarning,
                stacklevel=2,
            )

        if self.document_codec.server_side_embeddings:
            sort = self.document_codec.encode_vectorize_sort(query)
        else:
            query_embedding = self._get_safe_embedding().embed_query(text=query)
            # shortcut return if query isn't needed.
            if k == 0:
                return (query_embedding, [])
            sort = self.document_codec.encode_vector_sort(vector=query_embedding)

        return await self._asimilarity_find_with_embedding_by_sort(
            sort=sort, k=k, filter=filter
        )

    def _similarity_find_with_embedding_by_sort(
        self,
        sort: dict[str, Any],
        k: int = 4,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> tuple[list[float], list[tuple[Document, list[float]]]]:
        """Run ANN search with a provided sort clause.

        Returns:
            (query_embedding, List of (Document, embedding) most similar to the query).
        """
        sort_vec, hits_ite = self.run_query(
            n=k,
            filter=filter,
            sort=sort,
            include_sort_vector=True,
            include_embeddings=True,
        )
        if sort_vec is None:
            msg = "Unable to retrieve the server-side embedding of the query."
            raise AstraDBVectorStoreError(msg)
        # doc is a Document and emb is a list[float]:
        return (
            sort_vec,
            [
                cast(tuple[Document, list[float]], (doc, emb))
                for doc, _, emb, _ in hits_ite
            ],
        )

    async def _asimilarity_find_with_embedding_by_sort(
        self,
        sort: dict[str, Any],
        k: int = 4,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> tuple[list[float], list[tuple[Document, list[float]]]]:
        """Run ANN search with a provided sort clause.

        Returns:
            (query_embedding, List of (Document, embedding) most similar to the query).
        """
        sort_vec, hits_ite = await self.arun_query(
            n=k,
            filter=filter,
            sort=sort,
            include_sort_vector=True,
            include_embeddings=True,
        )
        if sort_vec is None:
            msg = "Unable to retrieve the server-side embedding of the query."
            raise AstraDBVectorStoreError(msg)
        # doc is a Document and emb is a list[float]:
        return (
            sort_vec,
            [
                cast(tuple[Document, list[float]], (doc, emb))
                async for doc, _, emb, _ in hits_ite
            ],
        )

    def _run_mmr_find_by_sort(
        self,
        sort: dict[str, Any],
        k: int,
        fetch_k: int,
        lambda_mult: float,
        filter: dict[str, Any] | None,  # noqa: A002
    ) -> list[Document]:
        sort_vec, hits_ite = self.run_query(
            n=fetch_k,
            filter=filter,
            sort=sort,
            include_sort_vector=True,
            include_embeddings=True,
        )
        # this is list[tuple[Document, list[float]]]:
        prefetch_hit_pairs = cast(
            list[tuple[Document, list[float]]],
            [(doc, emb) for doc, _, emb, _ in hits_ite],
        )
        if sort_vec is None:
            msg = "Unable to retrieve the server-side embedding of the query."
            raise AstraDBVectorStoreError(msg)
        return self._get_mmr_hits(
            embedding=sort_vec,
            k=k,
            lambda_mult=lambda_mult,
            prefetch_hit_pairs=prefetch_hit_pairs,
        )

    async def _arun_mmr_find_by_sort(
        self,
        sort: dict[str, Any],
        k: int,
        fetch_k: int,
        lambda_mult: float,
        filter: dict[str, Any] | None,  # noqa: A002
    ) -> list[Document]:
        sort_vec, hits_ite = await self.arun_query(
            n=fetch_k,
            filter=filter,
            sort=sort,
            include_sort_vector=True,
            include_embeddings=True,
        )
        # this is list[tuple[Document, list[float]]]:
        prefetch_hit_pairs = cast(
            list[tuple[Document, list[float]]],
            [(doc, emb) async for doc, _, emb, _ in hits_ite],
        )
        if sort_vec is None:
            msg = "Unable to retrieve the server-side embedding of the query."
            raise AstraDBVectorStoreError(msg)
        return self._get_mmr_hits(
            embedding=sort_vec,
            k=k,
            lambda_mult=lambda_mult,
            prefetch_hit_pairs=prefetch_hit_pairs,
        )

    def _get_mmr_hits(
        self,
        embedding: list[float],
        k: int,
        lambda_mult: float,
        prefetch_hit_pairs: list[tuple[Document, list[float]]],
    ) -> list[Document]:
        mmr_chosen_indices = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            [hit_pair[1] for hit_pair in prefetch_hit_pairs],
            k=k,
            lambda_mult=lambda_mult,
        )
        return [
            hit_pair[0]
            for pf_hit_index, hit_pair in enumerate(prefetch_hit_pairs)
            if pf_hit_index in mmr_chosen_indices
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
        return self._run_mmr_find_by_sort(
            sort=self.document_codec.encode_vector_sort(embedding),
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
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
        return await self._arun_mmr_find_by_sort(
            sort=self.document_codec.encode_vector_sort(embedding),
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
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
        if self.hybrid_search:
            warnings.warn(
                (
                    "Method `max_marginal_relevance_search` was called on a vector "
                    "store equipped with Hybrid capabilities. Since this method cannot "
                    "make use of Hybrid search, the vector store will fall back to "
                    "regular vector ANN similarity search."
                ),
                UserWarning,
                stacklevel=2,
            )

        if self.document_codec.server_side_embeddings:
            # this case goes directly to the "_by_sort" method
            # (and does its own filter normalization, as it cannot
            #  use the path for the with-embedding mmr querying)
            return self._run_mmr_find_by_sort(
                sort=self.document_codec.encode_vectorize_sort(query),
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter,
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
        if self.hybrid_search:
            warnings.warn(
                (
                    "Method `amax_marginal_relevance_search` was called on a vector "
                    "store equipped with Hybrid capabilities. Since this method cannot "
                    "make use of Hybrid search, the vector store will fall back to "
                    "regular vector ANN similarity search."
                ),
                UserWarning,
                stacklevel=2,
            )

        if self.document_codec.server_side_embeddings:
            # this case goes directly to the "_by_sort" method
            # (and does its own filter normalization, as it cannot
            #  use the path for the with-embedding mmr querying)
            return await self._arun_mmr_find_by_sort(
                sort=self.document_codec.encode_vectorize_sort(query),
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter,
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
