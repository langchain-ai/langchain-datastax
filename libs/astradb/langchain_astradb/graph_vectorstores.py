"""Astra DB graph vector store integration."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    cast,
)

from langchain_community.graph_vectorstores.cassandra_base import (
    CassandraGraphVectorStoreBase,
)
from langchain_community.graph_vectorstores.links import (
    METADATA_LINKS_KEY,
    Link,
    get_links,
    incoming_links,
)
from langchain_core._api import beta
from typing_extensions import override

from langchain_astradb.utils.astradb import COMPONENT_NAME_GRAPHVECTORSTORE, SetupMode
from langchain_astradb.vectorstores import AstraDBVectorStore

if TYPE_CHECKING:
    from astrapy.authentication import EmbeddingHeadersProvider, TokenProvider
    from astrapy.db import AstraDB as AstraDBClient
    from astrapy.db import AsyncAstraDB as AsyncAstraDBClient
    from astrapy.info import CollectionVectorServiceOptions
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings

DEFAULT_INDEXING_OPTIONS = {"allow": ["metadata"]}


logger = logging.getLogger(__name__)


def _serialize_links(links: list[Link]) -> str:
    class SetAndLinkEncoder(json.JSONEncoder):
        def default(self, obj: Any) -> Any:  # noqa: ANN401
            if not isinstance(obj, type) and is_dataclass(obj):
                return asdict(obj)

            if isinstance(obj, Iterable):
                return list(obj)

            # Let the base class default method raise the TypeError
            return super().default(obj)

    return json.dumps(links, cls=SetAndLinkEncoder)


def _deserialize_links(json_blob: str | None) -> set[Link]:
    return {
        Link(kind=link["kind"], direction=link["direction"], tag=link["tag"])
        for link in cast(list[dict[str, Any]], json.loads(json_blob or "[]"))
    }


def _metadata_link_key(link: Link) -> str:
    return f"link:{link.kind}:{link.tag}"


@beta()
class AstraDBGraphVectorStore(CassandraGraphVectorStoreBase):
    def __init__(
        self,
        *,
        collection_name: str,
        embedding: Embeddings | None = None,
        metadata_incoming_links_key: str = "incoming_links",
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
        component_name: str = COMPONENT_NAME_GRAPHVECTORSTORE,
        astra_db_client: AstraDBClient | None = None,
        async_astra_db_client: AsyncAstraDBClient | None = None,
    ):
        """Graph Vector Store backed by AstraDB.

        Args:
            embedding: the embeddings function or service to use.
                This enables client-side embedding functions or calls to external
                embedding providers. If ``embedding`` is provided, arguments
                ``collection_vector_service_options`` and
                ``collection_embedding_api_key`` cannot be provided.
            collection_name: name of the Astra DB collection to create/use.
            metadata_incoming_links_key: document metadata key where the incoming
                links are stored (and indexed).
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
                Defaults to "langchain_graphvectorstore", but can be overridden if this
                component actually serves as the building block for another component.
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
        self.metadata_incoming_links_key = metadata_incoming_links_key

        # update indexing policy to ensure incoming_links are indexed
        if metadata_indexing_include is not None:
            metadata_indexing_include = set(metadata_indexing_include)
            metadata_indexing_include.add(self.metadata_incoming_links_key)
        elif collection_indexing_policy is not None:
            allow_list = collection_indexing_policy.get("allow")
            if allow_list is not None:
                allow_list = set(allow_list)
                allow_list.add(self.metadata_incoming_links_key)
                collection_indexing_policy["allow"] = list(allow_list)

        try:
            super().__init__(
                vector_store=AstraDBVectorStore(
                    collection_name=collection_name,
                    embedding=embedding,
                    token=token,
                    api_endpoint=api_endpoint,
                    environment=environment,
                    namespace=namespace,
                    metric=metric,
                    batch_size=batch_size,
                    bulk_insert_batch_concurrency=bulk_insert_batch_concurrency,
                    bulk_insert_overwrite_concurrency=bulk_insert_overwrite_concurrency,
                    bulk_delete_concurrency=bulk_delete_concurrency,
                    setup_mode=setup_mode,
                    pre_delete_collection=pre_delete_collection,
                    metadata_indexing_include=metadata_indexing_include,
                    metadata_indexing_exclude=metadata_indexing_exclude,
                    collection_indexing_policy=collection_indexing_policy,
                    collection_vector_service_options=collection_vector_service_options,
                    collection_embedding_api_key=collection_embedding_api_key,
                    content_field=content_field,
                    ignore_invalid_documents=ignore_invalid_documents,
                    autodetect_collection=autodetect_collection,
                    ext_callers=ext_callers,
                    component_name=component_name,
                    astra_db_client=astra_db_client,
                    async_astra_db_client=async_astra_db_client,
                )
            )

            # for the test search, if setup_mode is ASYNC,
            # create a temp store with SYNC
            if setup_mode == SetupMode.ASYNC:
                test_vs = AstraDBVectorStore(
                    collection_name=collection_name,
                    embedding=embedding,
                    token=token,
                    api_endpoint=api_endpoint,
                    environment=environment,
                    namespace=namespace,
                    metric=metric,
                    batch_size=batch_size,
                    bulk_insert_batch_concurrency=bulk_insert_batch_concurrency,
                    bulk_insert_overwrite_concurrency=bulk_insert_overwrite_concurrency,
                    bulk_delete_concurrency=bulk_delete_concurrency,
                    setup_mode=SetupMode.SYNC,
                    pre_delete_collection=pre_delete_collection,
                    metadata_indexing_include=metadata_indexing_include,
                    metadata_indexing_exclude=metadata_indexing_exclude,
                    collection_indexing_policy=collection_indexing_policy,
                    collection_vector_service_options=collection_vector_service_options,
                    collection_embedding_api_key=collection_embedding_api_key,
                    content_field=content_field,
                    ignore_invalid_documents=ignore_invalid_documents,
                    autodetect_collection=autodetect_collection,
                    ext_callers=ext_callers,
                    component_name=component_name,
                    astra_db_client=astra_db_client,
                    async_astra_db_client=async_astra_db_client,
                )
            else:
                test_vs = self.vector_store

            # try a simple search to ensure that the indexes are setup properly
            test_vs.metadata_search(
                filter={self.metadata_incoming_links_key: "test"}, n=1
            )
        except ValueError as exp:
            # determine if error is because of a un-indexed column. Ref:
            # https://docs.datastax.com/en/astra-db-serverless/api-reference/collections.html#considerations-for-selective-indexing
            error_message = str(exp).lower()
            if ("unindexed filter path" in error_message) or (
                "incompatible with the requested indexing policy" in error_message
            ):
                msg = (
                    "The collection configuration is incompatible with vector graph "
                    "store. Please create a new collection and make sure the metadata "
                    "path is not excluded by indexing."
                )

                raise ValueError(msg) from exp
            raise exp  # noqa: TRY201

        self.astra_env = self.vector_store.astra_env

    def _get_metadata_for_insertion(self, doc: Document) -> dict[str, Any]:
        """Prepares the links in a document by serializing them to metadata.

        Args:
            doc: Document to prepare

        Returns:
            The document metadata ready for insertion into the database.
        """
        links = get_links(doc=doc)
        metadata = doc.metadata.copy()
        metadata[METADATA_LINKS_KEY] = _serialize_links(links=links)
        metadata[self.metadata_incoming_links_key] = [
            _metadata_link_key(link=link) for link in incoming_links(links=links)
        ]
        return metadata

    def _restore_links(self, doc: Document) -> Document:
        """Restores links in a document by deserializing them from metadata.

        Args:
            doc: Document to restore

        Returns:
            The document ready for use in the graph vector store.
        """
        links = _deserialize_links(doc.metadata.get(METADATA_LINKS_KEY))
        doc.metadata[METADATA_LINKS_KEY] = links
        doc.metadata.pop(self.metadata_incoming_links_key)
        return doc

    def _get_metadata_filter(
        self,
        metadata: dict[str, Any] | None = None,
        outgoing_link: Link | None = None,
    ) -> dict[str, Any]:
        """Builds a metadata filter to search for document.

        Args:
            metadata: Any metadata that should be used for hybrid search
            outgoing_link: An optional outgoing link to add to the search

        Returns:
            The document metadata ready for insertion into the database.
        """
        if outgoing_link is None:
            return metadata or {}

        metadata_filter = {} if metadata is None else metadata.copy()
        metadata_filter[self.metadata_incoming_links_key] = _metadata_link_key(
            link=outgoing_link
        )
        return metadata_filter

    @classmethod
    @override
    def from_texts(
        cls: type[AstraDBGraphVectorStore],
        texts: Iterable[str],
        embedding: Embeddings | None = None,
        metadatas: list[dict] | None = None,
        ids: Iterable[str] | None = None,
        collection_vector_service_options: CollectionVectorServiceOptions | None = None,
        collection_embedding_api_key: str | EmbeddingHeadersProvider | None = None,
        **kwargs: Any,
    ) -> AstraDBGraphVectorStore:
        """Return AstraDBGraphVectorStore initialized from texts and embeddings."""
        store = cls(
            embedding=embedding,
            collection_vector_service_options=collection_vector_service_options,
            collection_embedding_api_key=collection_embedding_api_key,
            **kwargs,
        )
        store.add_texts(texts, metadatas, ids=ids)
        return store

    @classmethod
    @override
    async def afrom_texts(
        cls: type[AstraDBGraphVectorStore],
        texts: Iterable[str],
        embedding: Embeddings | None = None,
        metadatas: list[dict] | None = None,
        ids: Iterable[str] | None = None,
        collection_vector_service_options: CollectionVectorServiceOptions | None = None,
        collection_embedding_api_key: str | EmbeddingHeadersProvider | None = None,
        **kwargs: Any,
    ) -> AstraDBGraphVectorStore:
        """Return AstraDBGraphVectorStore initialized from texts and embeddings."""
        store = cls(
            embedding=embedding,
            collection_vector_service_options=collection_vector_service_options,
            collection_embedding_api_key=collection_embedding_api_key,
            setup_mode=SetupMode.ASYNC,
            **kwargs,
        )
        await store.aadd_texts(texts, metadatas, ids=ids)
        return store

    @classmethod
    @override
    def from_documents(
        cls: type[AstraDBGraphVectorStore],
        documents: Iterable[Document],
        embedding: Embeddings | None = None,
        ids: Iterable[str] | None = None,
        collection_vector_service_options: CollectionVectorServiceOptions | None = None,
        collection_embedding_api_key: str | EmbeddingHeadersProvider | None = None,
        **kwargs: Any,
    ) -> AstraDBGraphVectorStore:
        """Return AstraDBGraphVectorStore initialized from docs and embeddings."""
        store = cls(
            embedding=embedding,
            collection_vector_service_options=collection_vector_service_options,
            collection_embedding_api_key=collection_embedding_api_key,
            **kwargs,
        )
        if ids is None:
            store.add_documents(documents)
        else:
            store.add_documents(documents, ids=ids)
        return store

    @classmethod
    @override
    async def afrom_documents(
        cls: type[AstraDBGraphVectorStore],
        documents: Iterable[Document],
        embedding: Embeddings | None = None,
        ids: Iterable[str] | None = None,
        collection_vector_service_options: CollectionVectorServiceOptions | None = None,
        collection_embedding_api_key: str | EmbeddingHeadersProvider | None = None,
        **kwargs: Any,
    ) -> AstraDBGraphVectorStore:
        """Return AstraDBGraphVectorStore initialized from docs and embeddings."""
        store = cls(
            embedding=embedding,
            collection_vector_service_options=collection_vector_service_options,
            collection_embedding_api_key=collection_embedding_api_key,
            setup_mode=SetupMode.ASYNC,
            **kwargs,
        )
        if ids is None:
            await store.aadd_documents(documents)
        else:
            await store.aadd_documents(documents, ids=ids)
        return store
