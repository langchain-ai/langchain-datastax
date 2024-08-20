from __future__ import annotations

import asyncio
import inspect
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
from astrapy.authentication import EmbeddingHeadersProvider, TokenProvider
from astrapy.db import (
    AstraDB as AstraDBClient,
)
from astrapy.db import (
    AsyncAstraDB as AsyncAstraDBClient,
)
from astrapy.exceptions import InsertManyException
from astrapy.info import CollectionVectorServiceOptions
from astrapy.results import UpdateResult
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.utils import gather_with_concurrency
from langchain_core.vectorstores import VectorStore

from langchain_astradb.utils.astradb import (
    DEFAULT_DOCUMENT_CHUNK_SIZE,
    MAX_CONCURRENT_DOCUMENT_DELETIONS,
    MAX_CONCURRENT_DOCUMENT_INSERTIONS,
    MAX_CONCURRENT_DOCUMENT_REPLACEMENTS,
    SetupMode,
    _AstraDBCollectionEnvironment,
)
from langchain_astradb.utils.encoders import (
    DefaultVectorizeVSDocumentEncoder,
    DefaultVSDocumentEncoder,
    VSDocumentEncoder,
)
from langchain_astradb.utils.mmr import maximal_marginal_relevance

T = TypeVar("T")
U = TypeVar("U")
DocDict = Dict[str, Any]  # dicts expressing entries to insert

# indexing options when creating a collection
DEFAULT_INDEXING_OPTIONS = {"allow": ["metadata"]}


def _unique_list(lst: List[T], key: Callable[[T], U]) -> List[T]:
    visited_keys: Set[U] = set()
    new_lst = []
    for item in lst:
        item_key = key(item)
        if item_key not in visited_keys:
            visited_keys.add(item_key)
            new_lst.append(item)
    return new_lst


class AstraDBVectorStore(VectorStore):
    """AstraDB vector store integration.

    Setup:
        Install ``langchain-astradb`` and head to the [AstraDB website](https://astra.datastax.com), create an account, create a new database and [create an application token](https://docs.datastax.com/en/astra-db-serverless/administration/manage-application-tokens.html#generate-application-token).

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

    # TODO: Replace with relevant init params.
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

        .. code-block:: python

            * thud [{'bar': 'baz'}]

    Search with filter:
        .. code-block:: python

            results = vector_store.similarity_search(query="thud",k=1,filter={"bar": "baz"})
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * thud [{'bar': 'baz'}]

    Search with score:
        .. code-block:: python

            results = vector_store.similarity_search_with_score(query="qux",k=1)
            for doc, score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * [SIM=0.916135] foo [{'baz': 'bar'}]

    Async:
        .. code-block:: python

            # add documents
            # await vector_store.aadd_documents(documents=documents, ids=ids)

            # delete documents
            # await vector_store.adelete(ids=["3"])

            # search
            # results = vector_store.asimilarity_search(query="thud",k=1)

            # search with score
            results = await vector_store.asimilarity_search_with_score(query="qux",k=1)
            for doc,score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * [SIM=0.916135] foo [{'baz': 'bar'}]

    Use as Retriever:
        .. code-block:: python

            retriever = vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 1, "score_threshold": 0.5},
            )
            retriever.invoke("thud")

        .. code-block:: python

            [Document(metadata={'bar': 'baz'}, page_content='thud')]

    """  # noqa: E501

    def _filter_to_metadata(
        self, filter_dict: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if filter_dict is None:
            return {}
        else:
            return self.document_encoder.encode_filter(filter_dict)

    @staticmethod
    def _normalize_metadata_indexing_policy(
        metadata_indexing_include: Optional[Iterable[str]],
        metadata_indexing_exclude: Optional[Iterable[str]],
        collection_indexing_policy: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Validate the constructor indexing parameters and normalize them
        into a ready-to-use dict for the 'options' when creating a collection.
        """
        none_count = sum(
            [
                1 if var is None else 0
                for var in [
                    metadata_indexing_include,
                    metadata_indexing_exclude,
                    collection_indexing_policy,
                ]
            ]
        )
        if none_count >= 2:
            if metadata_indexing_include is not None:
                return {
                    "allow": [
                        f"metadata.{md_field}" for md_field in metadata_indexing_include
                    ]
                }
            elif metadata_indexing_exclude is not None:
                return {
                    "deny": [
                        f"metadata.{md_field}" for md_field in metadata_indexing_exclude
                    ]
                }
            elif collection_indexing_policy is not None:
                return collection_indexing_policy
            else:
                return DEFAULT_INDEXING_OPTIONS
        else:
            raise ValueError(
                "At most one of the parameters `metadata_indexing_include`,"
                " `metadata_indexing_exclude` and `collection_indexing_policy`"
                " can be specified as non null."
            )

    def __init__(
        self,
        *,
        collection_name: str,
        embedding: Optional[Embeddings] = None,
        token: Optional[Union[str, TokenProvider]] = None,
        api_endpoint: Optional[str] = None,
        environment: Optional[str] = None,
        astra_db_client: Optional[AstraDBClient] = None,
        async_astra_db_client: Optional[AsyncAstraDBClient] = None,
        namespace: Optional[str] = None,
        metric: Optional[str] = None,
        batch_size: Optional[int] = None,
        bulk_insert_batch_concurrency: Optional[int] = None,
        bulk_insert_overwrite_concurrency: Optional[int] = None,
        bulk_delete_concurrency: Optional[int] = None,
        setup_mode: SetupMode = SetupMode.SYNC,
        pre_delete_collection: bool = False,
        metadata_indexing_include: Optional[Iterable[str]] = None,
        metadata_indexing_exclude: Optional[Iterable[str]] = None,
        collection_indexing_policy: Optional[Dict[str, Any]] = None,
        collection_vector_service_options: Optional[
            CollectionVectorServiceOptions
        ] = None,
        collection_embedding_api_key: Optional[
            Union[str, EmbeddingHeadersProvider]
        ] = None,
    ) -> None:
        """Wrapper around DataStax Astra DB for vector-store workloads.

        For quickstart and details, visit
        https://docs.datastax.com/en/astra/astra-db-vector/

        Args:
            embedding: the embeddings function or service to use.
                This enables client-side embedding functions or calls to external
                embedding providers. If `embedding` is provided, arguments
                `collection_vector_service_options` and
                `collection_embedding_api_key` cannot be provided.
            collection_name: name of the Astra DB collection to create/use.
            token: API token for Astra DB usage, either in the form of a string
                or a subclass of `astrapy.authentication.TokenProvider`.
                If not provided, the environment variable
                ASTRA_DB_APPLICATION_TOKEN is inspected.
            api_endpoint: full URL to the API endpoint, such as
                `https://<DB-ID>-us-east1.apps.astra.datastax.com`. If not provided,
                the environment variable ASTRA_DB_API_ENDPOINT is inspected.
            environment: a string specifying the environment of the target Data API.
                If omitted, defaults to "prod" (Astra DB production).
                Other values are in `astrapy.constants.Environment` enum class.
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
                (see docs.datastax.com/en/astra/astra-db-vector/api-reference/
                data-api-commands.html#advanced-feature-indexing-clause-on-createcollection)
            collection_vector_service_options: specifies the use of server-side
                embeddings within Astra DB. If passing this parameter, `embedding`
                cannot be provided.
            collection_embedding_api_key: for usage of server-side embeddings
                within Astra DB. With this parameter one can supply an API Key
                that will be passed to Astra DB with each data request.
                This parameter can be either a string or a subclass of
                `astrapy.authentication.EmbeddingHeadersProvider`.
                This is useful when the service is configured for the collection,
                but no corresponding secret is stored within
                Astra's key management system.
                This parameter cannot be provided without
                specifying `collection_vector_service_options`.

        Note:
            For concurrency in synchronous :meth:`~add_texts`:, as a rule of thumb, on a
            typical client machine it is suggested to keep the quantity
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
        # Embedding and the server-side embeddings are mutually exclusive,
        # as both specify how to produce embeddings
        if embedding is None and collection_vector_service_options is None:
            raise ValueError(
                "Either an `embedding` or a `collection_vector_service_options`\
                    must be provided."
            )

        if embedding is not None and collection_vector_service_options is not None:
            raise ValueError(
                "Only one of `embedding` or `collection_vector_service_options`\
                    can be provided."
            )

        if (
            collection_vector_service_options is None
            and collection_embedding_api_key is not None
        ):
            raise ValueError(
                "`collection_embedding_api_key` cannot be provided unless"
                " `collection_vector_service_options` is also passed."
            )

        self.embedding_dimension: Optional[int] = None
        self.embedding = embedding
        self.collection_name = collection_name
        self.token = token
        self.api_endpoint = api_endpoint
        self.environment = environment
        self.namespace = namespace
        self.collection_vector_service_options = collection_vector_service_options
        self.document_encoder: VSDocumentEncoder
        if self.collection_vector_service_options is not None:
            self.document_encoder = DefaultVectorizeVSDocumentEncoder()
        else:
            self.document_encoder = DefaultVSDocumentEncoder()
        self.collection_embedding_api_key = collection_embedding_api_key
        # Concurrency settings
        self.batch_size: Optional[int] = batch_size or DEFAULT_DOCUMENT_CHUNK_SIZE
        self.bulk_insert_batch_concurrency: int = (
            bulk_insert_batch_concurrency or MAX_CONCURRENT_DOCUMENT_INSERTIONS
        )
        self.bulk_insert_overwrite_concurrency: int = (
            bulk_insert_overwrite_concurrency or MAX_CONCURRENT_DOCUMENT_REPLACEMENTS
        )
        self.bulk_delete_concurrency: int = (
            bulk_delete_concurrency or MAX_CONCURRENT_DOCUMENT_DELETIONS
        )
        # "vector-related" settings
        self.metric = metric
        embedding_dimension_m: Union[int, Awaitable[int], None] = None
        if self.embedding is not None:
            if setup_mode == SetupMode.ASYNC:
                embedding_dimension_m = self._aget_embedding_dimension()
            elif setup_mode == SetupMode.SYNC or setup_mode == SetupMode.OFF:
                embedding_dimension_m = self._get_embedding_dimension()

        # indexing policy setting
        self.indexing_policy: Dict[str, Any] = self._normalize_metadata_indexing_policy(
            metadata_indexing_include=metadata_indexing_include,
            metadata_indexing_exclude=metadata_indexing_exclude,
            collection_indexing_policy=collection_indexing_policy,
        )

        self.astra_env = _AstraDBCollectionEnvironment(
            collection_name=collection_name,
            token=self.token,
            api_endpoint=self.api_endpoint,
            environment=self.environment,
            astra_db_client=astra_db_client,
            async_astra_db_client=async_astra_db_client,
            namespace=self.namespace,
            setup_mode=setup_mode,
            pre_delete_collection=pre_delete_collection,
            embedding_dimension=embedding_dimension_m,
            metric=self.metric,
            requested_indexing_policy=self.indexing_policy,
            default_indexing_policy=DEFAULT_INDEXING_OPTIONS,
            collection_vector_service_options=self.collection_vector_service_options,
            collection_embedding_api_key=self.collection_embedding_api_key,
        )

    def _get_embedding_dimension(self) -> int:
        assert self.embedding is not None

        if self.embedding_dimension is None:
            self.embedding_dimension = len(
                self.embedding.embed_query(text="This is a sample sentence.")
            )
        return self.embedding_dimension

    async def _aget_embedding_dimension(self) -> int:
        assert self.embedding is not None

        if self.embedding_dimension is None:
            self.embedding_dimension = len(
                await self.embedding.aembed_query(text="This is a sample sentence.")
            )
        return self.embedding_dimension

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """
        Accesses the supplied embeddings object. If using server-side embeddings,
        this will return None.
        """
        return self.embedding

    def _using_vectorize(self) -> bool:
        """Indicates whether server-side embeddings are being used."""
        return self.document_encoder.server_side_embeddings

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The underlying API calls already returns a "score proper",
        i.e. one in [0, 1] where higher means more *similar*,
        so here the final score transformation is not reversing the interval:
        """
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
        """
        Remove a single document from the store, given its document ID.

        Args:
            document_id: The document ID

        Returns
            True if a document has indeed been deleted, False if ID not found.
        """
        self.astra_env.ensure_db_setup()
        # self.collection is not None (by _ensure_astra_db_client)
        deletion_response = self.astra_env.collection.delete_one({"_id": document_id})
        return deletion_response.deleted_count == 1

    async def adelete_by_document_id(self, document_id: str) -> bool:
        """
        Remove a single document from the store, given its document ID.

        Args:
            document_id: The document ID

        Returns
            True if a document has indeed been deleted, False if ID not found.
        """
        await self.astra_env.aensure_db_setup()
        deletion_response = await self.astra_env.async_collection.delete_one(
            {"_id": document_id},
        )
        return deletion_response.deleted_count == 1

    def delete(
        self,
        ids: Optional[List[str]] = None,
        concurrency: Optional[int] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete by vector ids.

        Args:
            ids: List of ids to delete.
            concurrency: max number of threads issuing single-doc delete requests.
                Defaults to vector-store overall setting.

        Returns:
            True if deletion is (entirely) successful, False otherwise.
        """

        if kwargs:
            warnings.warn(
                "Method 'delete' of AstraDBVectorStore vector store invoked with "
                f"unsupported arguments ({', '.join(sorted(kwargs.keys()))}), "
                "which will be ignored."
            )

        if ids is None:
            raise ValueError("No ids provided to delete.")

        _max_workers = concurrency or self.bulk_delete_concurrency
        with ThreadPoolExecutor(max_workers=_max_workers) as tpe:
            _ = list(
                tpe.map(
                    self.delete_by_document_id,
                    ids,
                )
            )
        return True

    async def adelete(
        self,
        ids: Optional[List[str]] = None,
        concurrency: Optional[int] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete by vector ids.

        Args:
            ids: List of ids to delete.
            concurrency: max number of simultaneous coroutines for single-doc
                delete requests. Defaults to vector-store overall setting.

        Returns:
            True if deletion is (entirely) successful, False otherwise.
        """
        if kwargs:
            warnings.warn(
                "Method 'adelete' of AstraDBVectorStore invoked with "
                f"unsupported arguments ({', '.join(sorted(kwargs.keys()))}), "
                "which will be ignored."
            )

        if ids is None:
            raise ValueError("No ids provided to delete.")

        _max_workers = concurrency or self.bulk_delete_concurrency
        return all(
            await gather_with_concurrency(
                _max_workers, *[self.adelete_by_document_id(doc_id) for doc_id in ids]
            )
        )

    def delete_collection(self) -> None:
        """
        Completely delete the collection from the database (as opposed
        to :meth:`~clear`, which empties it only).
        Stored data is lost and unrecoverable, resources are freed.
        Use with caution.
        """
        self.astra_env.ensure_db_setup()
        self.astra_env.collection.drop()

    async def adelete_collection(self) -> None:
        """
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
        embedding_vectors: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[DocDict]:
        if ids is None:
            ids = [uuid.uuid4().hex for _ in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]
        #
        documents_to_insert = [
            self.document_encoder.encode(
                content=b_txt,
                id=b_id,
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
        uniqued_documents_to_insert = _unique_list(
            documents_to_insert[::-1],
            lambda document: document["_id"],
        )[::-1]
        return uniqued_documents_to_insert

    def _get_vectorize_documents_to_insert(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[DocDict]:
        if ids is None:
            ids = [uuid.uuid4().hex for _ in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]
        #
        documents_to_insert = [
            self.document_encoder.encode(
                content=b_txt,
                id=b_id,
                vector=None,
                metadata=b_md,
            )
            for b_txt, b_id, b_md in zip(
                texts,
                ids,
                metadatas,
            )
        ]
        # make unique by id, keeping the last
        uniqued_documents_to_insert = _unique_list(
            documents_to_insert[::-1],
            lambda document: document["_id"],
        )[::-1]
        return uniqued_documents_to_insert

    @staticmethod
    def _get_missing_from_batch(
        document_batch: List[DocDict], insert_result: Dict[str, Any]
    ) -> Tuple[List[str], List[DocDict]]:
        if "status" not in insert_result:
            raise ValueError(
                f"API Exception while running bulk insertion: {str(insert_result)}"
            )
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
            raise ValueError(
                f"API Exception while running bulk insertion: {str(errors)}"
            )
        # deal with the missing insertions as upserts
        missing_from_batch = [
            document
            for document in document_batch
            if document["_id"] in missed_inserted_ids
        ]
        return batch_inserted, missing_from_batch

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        *,
        batch_size: Optional[int] = None,
        batch_concurrency: Optional[int] = None,
        overwrite_concurrency: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
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

        Note:
            There are constraints on the allowed field names
            in the metadata dictionaries, coming from the underlying Astra DB API.
            For instance, the `$` (dollar sign) cannot be used in the dict keys.
            See this document for details:
            https://docs.datastax.com/en/astra/astra-db-vector/api-reference/data-api.html

        Returns:
            The list of ids of the added texts.
        """

        if kwargs:
            warnings.warn(
                "Method 'add_texts' of AstraDBVectorStore vector store invoked with "
                f"unsupported arguments ({', '.join(sorted(kwargs.keys()))}), "
                "which will be ignored."
            )
        self.astra_env.ensure_db_setup()

        if self._using_vectorize():
            documents_to_insert = self._get_vectorize_documents_to_insert(
                texts, metadatas, ids
            )
        else:
            assert self.embedding is not None
            embedding_vectors = self.embedding.embed_documents(list(texts))
            documents_to_insert = self._get_documents_to_insert(
                texts, embedding_vectors, metadatas, ids
            )

        # perform an AstraPy insert_many, catching exceptions for overwriting docs
        ids_to_replace: List[int]
        inserted_ids: List[str] = []
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
            inserted_ids = err.partial_result.inserted_ids
            inserted_ids_set = set(inserted_ids)
            ids_to_replace = [
                document["_id"]
                for document in documents_to_insert
                if document["_id"] not in inserted_ids_set
            ]

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
                    document: Dict[str, Any],
                ) -> Tuple[UpdateResult, str]:
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
                raise ValueError(
                    "AstraDBVectorStore.add_texts could not insert all requested "
                    f"documents ({missing} failed replace_one calls)"
                )
        return inserted_ids

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        *,
        batch_size: Optional[int] = None,
        batch_concurrency: Optional[int] = None,
        overwrite_concurrency: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
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

        Note:
            There are constraints on the allowed field names
            in the metadata dictionaries, coming from the underlying Astra DB API.
            For instance, the `$` (dollar sign) cannot be used in the dict keys.
            See this document for details:
            https://docs.datastax.com/en/astra/astra-db-vector/api-reference/data-api.html

        Returns:
            The list of ids of the added texts.
        """
        if kwargs:
            warnings.warn(
                "Method 'aadd_texts' of AstraDBVectorStore invoked with "
                f"unsupported arguments ({', '.join(sorted(kwargs.keys()))}), "
                "which will be ignored."
            )
        await self.astra_env.aensure_db_setup()

        if self._using_vectorize():
            # using server-side embeddings
            documents_to_insert = self._get_vectorize_documents_to_insert(
                texts, metadatas, ids
            )
        else:
            assert self.embedding is not None
            embedding_vectors = await self.embedding.aembed_documents(list(texts))
            documents_to_insert = self._get_documents_to_insert(
                texts, embedding_vectors, metadatas, ids
            )

        # perform an AstraPy insert_many, catching exceptions for overwriting docs
        ids_to_replace: List[int]
        inserted_ids: List[str] = []
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
            inserted_ids = err.partial_result.inserted_ids
            inserted_ids_set = set(inserted_ids)
            ids_to_replace = [
                document["_id"]
                for document in documents_to_insert
                if document["_id"] not in inserted_ids_set
            ]

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
                document: Dict[str, Any],
            ) -> Tuple[UpdateResult, str]:
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
                raise ValueError(
                    "AstraDBVectorStore.add_texts could not insert all requested "
                    f"documents ({missing} failed replace_one calls)"
                )
        return inserted_ids

    def similarity_search_with_score_id_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float, str]]:
        """Return docs most similar to embedding vector with score and id.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.

        Returns:
            The list of (Document, score, id), the most similar to the query vector.
        """
        self.astra_env.ensure_db_setup()
        metadata_parameter = self._filter_to_metadata(filter)
        #
        hits = list(
            self.astra_env.collection.find(
                filter=metadata_parameter,
                projection=self.document_encoder.base_projection,
                limit=k,
                include_similarity=True,
                sort={"$vector": embedding},
            )
        )
        #
        return [
            (
                self.document_encoder.decode(hit),
                hit["$similarity"],
                hit["_id"],
            )
            for hit in hits
        ]

    async def asimilarity_search_with_score_id_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float, str]]:
        """Return docs most similar to embedding vector with score and id.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.

        Returns:
            The list of (Document, score, id), the most similar to the query vector.
        """
        await self.astra_env.aensure_db_setup()
        metadata_parameter = self._filter_to_metadata(filter)
        #
        return [
            (
                self.document_encoder.decode(hit),
                hit["$similarity"],
                hit["_id"],
            )
            async for hit in self.astra_env.async_collection.find(
                filter=metadata_parameter,
                projection=self.document_encoder.base_projection,
                limit=k,
                include_similarity=True,
                sort={"$vector": embedding},
            )
        ]

    def _similarity_search_with_score_id_with_vectorize(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float, str]]:
        """Return docs most similar to the query with score and id using $vectorize.

        This is only available when using server-side embeddings.
        """
        self.astra_env.ensure_db_setup()
        metadata_parameter = self._filter_to_metadata(filter)
        #
        hits = list(
            self.astra_env.collection.find(
                filter=metadata_parameter,
                projection=self.document_encoder.base_projection,
                limit=k,
                include_similarity=True,
                sort={"$vectorize": query},
            )
        )
        #
        return [
            (
                self.document_encoder.decode(hit),
                hit["$similarity"],
                hit["_id"],
            )
            for hit in hits
        ]

    async def _asimilarity_search_with_score_id_with_vectorize(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float, str]]:
        """Return docs most similar to the query with score and id using $vectorize.

        This is only available when using server-side embeddings.
        """
        await self.astra_env.aensure_db_setup()
        metadata_parameter = self._filter_to_metadata(filter)
        #
        return [
            (
                self.document_encoder.decode(hit),
                hit["$similarity"],
                hit["_id"],
            )
            async for hit in self.astra_env.async_collection.find(
                filter=metadata_parameter,
                projection=self.document_encoder.base_projection,
                limit=k,
                include_similarity=True,
                sort={"$vectorize": query},
            )
        ]

    def similarity_search_with_score_id(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float, str]]:
        """Return docs most similar to the query with score and id.

        Args:
            query: Query to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.

        Returns:
            The list of (Document, score, id), the most similar to the query.
        """

        if self._using_vectorize():
            return self._similarity_search_with_score_id_with_vectorize(
                query=query,
                k=k,
                filter=filter,
            )
        else:
            assert self.embedding is not None
            embedding_vector = self.embedding.embed_query(query)
            return self.similarity_search_with_score_id_by_vector(
                embedding=embedding_vector,
                k=k,
                filter=filter,
            )

    async def asimilarity_search_with_score_id(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float, str]]:
        """Return docs most similar to the query with score and id.

        Args:
            query: Query to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.

        Returns:
            The list of (Document, score, id), the most similar to the query.
        """
        if self._using_vectorize():
            return await self._asimilarity_search_with_score_id_with_vectorize(
                query=query,
                k=k,
                filter=filter,
            )
        else:
            assert self.embedding is not None
            embedding_vector = await self.embedding.aembed_query(query)
            return await self.asimilarity_search_with_score_id_by_vector(
                embedding=embedding_vector,
                k=k,
                filter=filter,
            )

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
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
            for (doc, score, doc_id) in self.similarity_search_with_score_id_by_vector(
                embedding=embedding,
                k=k,
                filter=filter,
            )
        ]

    async def asimilarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
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
            for (
                doc,
                score,
                doc_id,
            ) in await self.asimilarity_search_with_score_id_by_vector(
                embedding=embedding,
                k=k,
                filter=filter,
            )
        ]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Query to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.

        Returns:
            The list of Documents most similar to the query.
        """
        if self._using_vectorize():
            return [
                doc
                for (doc, _, _) in self._similarity_search_with_score_id_with_vectorize(
                    query,
                    k,
                    filter=filter,
                )
            ]
        else:
            assert self.embedding is not None
            embedding_vector = self.embedding.embed_query(query)
            return self.similarity_search_by_vector(
                embedding_vector,
                k,
                filter=filter,
            )

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Query to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.

        Returns:
            The list of Documents most similar to the query.
        """
        if self._using_vectorize():
            return [
                doc
                for (
                    doc,
                    _,
                    _,
                ) in await self._asimilarity_search_with_score_id_with_vectorize(
                    query,
                    k,
                    filter=filter,
                )
            ]
        else:
            assert self.embedding is not None
            embedding_vector = await self.embedding.aembed_query(query)
            return await self.asimilarity_search_by_vector(
                embedding_vector,
                k,
                filter=filter,
            )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.

        Returns:
            The list of Documents most similar to the query vector.
        """
        return [
            doc
            for doc, _ in self.similarity_search_with_score_by_vector(
                embedding,
                k,
                filter=filter,
            )
        ]

    async def asimilarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.

        Returns:
            The list of Documents most similar to the query vector.
        """
        return [
            doc
            for doc, _ in await self.asimilarity_search_with_score_by_vector(
                embedding,
                k,
                filter=filter,
            )
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query with score.

        Args:
            query: Query to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.

        Returns:
            The list of (Document, score), the most similar to the query vector.
        """
        if self._using_vectorize():
            return [
                (doc, score)
                for (
                    doc,
                    score,
                    doc_id,
                ) in self._similarity_search_with_score_id_with_vectorize(
                    query=query,
                    k=k,
                    filter=filter,
                )
            ]
        else:
            assert self.embedding is not None
            embedding_vector = self.embedding.embed_query(query)
            return self.similarity_search_with_score_by_vector(
                embedding_vector,
                k,
                filter=filter,
            )

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query with score.

        Args:
            query: Query to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.

        Returns:
            The list of (Document, score), the most similar to the query vector.
        """
        if self._using_vectorize():
            return [
                (doc, score)
                for (
                    doc,
                    score,
                    doc_id,
                ) in await self._asimilarity_search_with_score_id_with_vectorize(
                    query=query,
                    k=k,
                    filter=filter,
                )
            ]
        else:
            assert self.embedding is not None
            embedding_vector = await self.embedding.aembed_query(query)
            return await self.asimilarity_search_with_score_by_vector(
                embedding_vector,
                k,
                filter=filter,
            )

    def _run_mmr_query_by_sort(
        self,
        sort: Dict[str, Any],
        k: int,
        fetch_k: int,
        lambda_mult: float,
        metadata_parameter: Dict[str, Any],
        **kwargs: Any,
    ) -> List[Document]:
        prefetch_cursor = self.astra_env.collection.find(
            filter=metadata_parameter,
            projection=self.document_encoder.full_projection,
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
        sort: Dict[str, Any],
        k: int,
        fetch_k: int,
        lambda_mult: float,
        metadata_parameter: Dict[str, Any],
        **kwargs: Any,
    ) -> List[Document]:
        prefetch_cursor = self.astra_env.async_collection.find(
            filter=metadata_parameter,
            projection=self.document_encoder.full_projection,
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
        embedding: List[float],
        k: int,
        lambda_mult: float,
        prefetch_hits: List[DocDict],
    ) -> List[Document]:
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
        return [self.document_encoder.decode(hit) for hit in mmr_hits]

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
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

        Returns:
            The list of Documents selected by maximal marginal relevance.
        """
        self.astra_env.ensure_db_setup()
        metadata_parameter = self._filter_to_metadata(filter)

        return self._run_mmr_query_by_sort(
            sort={"$vector": embedding},
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            metadata_parameter=metadata_parameter,
        )

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
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

        Returns:
            The list of Documents selected by maximal marginal relevance.
        """
        await self.astra_env.aensure_db_setup()
        metadata_parameter = self._filter_to_metadata(filter)

        return await self._arun_mmr_query_by_sort(
            sort={"$vector": embedding},
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            metadata_parameter=metadata_parameter,
        )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
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

        Returns:
            The list of Documents selected by maximal marginal relevance.
        """
        if self._using_vectorize():
            # this case goes directly to the "_by_sort" method
            # (and does its own filter normalization, as it cannot
            #  use the path for the with-embedding mmr querying)
            metadata_parameter = self._filter_to_metadata(filter)
            return self._run_mmr_query_by_sort(
                sort={"$vectorize": query},
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                metadata_parameter=metadata_parameter,
            )
        else:
            assert self.embedding is not None
            embedding_vector = self.embedding.embed_query(query)
            return self.max_marginal_relevance_search_by_vector(
                embedding_vector,
                k,
                fetch_k,
                lambda_mult=lambda_mult,
                filter=filter,
            )

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
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

        Returns:
            The list of Documents selected by maximal marginal relevance.
        """
        if self._using_vectorize():
            # this case goes directly to the "_by_sort" method
            # (and does its own filter normalization, as it cannot
            #  use the path for the with-embedding mmr querying)
            metadata_parameter = self._filter_to_metadata(filter)
            return await self._arun_mmr_query_by_sort(
                sort={"$vectorize": query},
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                metadata_parameter=metadata_parameter,
            )
        else:
            assert self.embedding is not None
            embedding_vector = await self.embedding.aembed_query(query)
            return await self.amax_marginal_relevance_search_by_vector(
                embedding_vector,
                k,
                fetch_k,
                lambda_mult=lambda_mult,
                filter=filter,
            )

    @classmethod
    def _from_kwargs(
        cls: Type[AstraDBVectorStore],
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
                )

        known_kwargs = {k: v for k, v in kwargs.items() if k in known_kwarg_keys}
        return cls(**known_kwargs)

    @classmethod
    def from_texts(
        cls: Type[AstraDBVectorStore],
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AstraDBVectorStore:
        """Create an Astra DB vectorstore from raw texts.

        Args:
            texts: the texts to insert.
            embedding: the embedding function to use in the store.
            metadatas: metadata dicts for the texts.
            ids: ids to associate to the texts.
            **kwargs: you can pass any argument that you would
                to :meth:`~add_texts` and/or to the 'AstraDBVectorStore' constructor
                (see these methods for details). These arguments will be
                routed to the respective methods as they are.

        Returns:
            an `AstraDBVectorStore` vectorstore.
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

    @classmethod
    async def afrom_texts(
        cls: Type[AstraDBVectorStore],
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AstraDBVectorStore:
        """Create an Astra DB vectorstore from raw texts.

        Args:
            texts: the texts to insert.
            metadatas: metadata dicts for the texts.
            ids: ids to associate to the texts.
            **kwargs: you can pass any argument that you would
                to :meth:`~aadd_texts` and/or to the 'AstraDBVectorStore' constructor
                (see these methods for details). These arguments will be
                routed to the respective methods as they are.

        Returns:
            an `AstraDBVectorStore` vectorstore.
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
    def from_documents(
        cls: Type[AstraDBVectorStore],
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        **kwargs: Any,
    ) -> AstraDBVectorStore:
        """Create an Astra DB vectorstore from a document list.

        Utility method that defers to 'from_texts' (see that one).

        Args: see 'from_texts', except here you have to supply 'documents'
            in place of 'texts' and 'metadatas'.

        Returns:
            an `AstraDBVectorStore` vectorstore.
        """
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        return cls.from_texts(
            texts,
            embedding=embedding,
            metadatas=metadatas,
            **kwargs,
        )

    @classmethod
    async def afrom_documents(
        cls: Type[AstraDBVectorStore],
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        **kwargs: Any,
    ) -> AstraDBVectorStore:
        """Create an Astra DB vectorstore from a document list.

        Utility method that defers to 'afrom_texts' (see that one).

        Args: see 'afrom_texts', except here you have to supply 'documents'
            in place of 'texts' and 'metadatas'.

        Returns:
            an `AstraDBVectorStore` vectorstore.
        """
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        return await cls.afrom_texts(
            texts,
            embedding=embedding,
            metadatas=metadatas,
            **kwargs,
        )
