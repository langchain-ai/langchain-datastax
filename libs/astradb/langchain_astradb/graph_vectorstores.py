"""Astra DB graph vector store integration."""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from dataclasses import asdict, is_dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterable,
    Iterable,
    Sequence,
    cast,
)

from langchain_community.graph_vectorstores.base import GraphVectorStore, Node
from langchain_community.graph_vectorstores.links import METADATA_LINKS_KEY, Link
from langchain_core.documents import Document
from typing_extensions import override

from langchain_astradb.utils.astradb import (
    COMPONENT_NAME_GRAPHVECTORSTORE,
    HybridSearchMode,
    SetupMode,
)
from langchain_astradb.utils.mmr_helper import MmrHelper
from langchain_astradb.vectorstores import AstraDBVectorStore

if TYPE_CHECKING:
    from astrapy.authentication import (
        EmbeddingHeadersProvider,
        RerankingHeadersProvider,
        TokenProvider,
    )
    from astrapy.info import (
        CollectionLexicalOptions,
        CollectionRerankOptions,
        RerankServiceOptions,
        VectorServiceOptions,
    )
    from langchain_core.embeddings import Embeddings

DEFAULT_INDEXING_OPTIONS = {"allow": ["metadata"]}


logger = logging.getLogger(__name__)


class EmbeddedNode:
    id: str
    links: list[Link]
    embedding: list[float]

    def __init__(self, doc: Document, embedding: list[float]) -> None:
        """Create an Embedded Node."""
        node = _doc_to_node(doc=doc)
        self.id = node.id or ""
        self.links = node.links
        self.embedding = embedding


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


def _doc_to_node(doc: Document) -> Node:
    metadata = doc.metadata.copy()
    links = _deserialize_links(metadata.get(METADATA_LINKS_KEY))
    metadata[METADATA_LINKS_KEY] = links

    return Node(
        id=doc.id,
        text=doc.page_content,
        metadata=metadata,
        links=list(links),
    )


def _incoming_links(node: Node | EmbeddedNode) -> set[Link]:
    return {link for link in node.links if link.direction in ["in", "bidir"]}


def _outgoing_links(node: Node | EmbeddedNode) -> set[Link]:
    return {link for link in node.links if link.direction in ["out", "bidir"]}


class AstraDBGraphVectorStore(GraphVectorStore):
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
        collection_vector_service_options: VectorServiceOptions | None = None,
        collection_embedding_api_key: str | EmbeddingHeadersProvider | None = None,
        content_field: str | None = None,
        ignore_invalid_documents: bool = False,
        autodetect_collection: bool = False,
        ext_callers: list[tuple[str | None, str | None] | str | None] | None = None,
        component_name: str = COMPONENT_NAME_GRAPHVECTORSTORE,
        collection_rerank: CollectionRerankOptions | RerankServiceOptions | None = None,
        collection_reranking_api_key: str | RerankingHeadersProvider | None = None,
        collection_lexical: str
        | dict[str, Any]
        | CollectionLexicalOptions
        | None = None,
        hybrid_search: HybridSearchMode | None = None,
        hybrid_limit_factor: float | None = None,
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
            collection_rerank: providing reranking settings is necessary to run
                hybrid searches for similarity. This parameter can be an instance
                of the astrapy classes `CollectionRerankOptions` or
                `RerankServiceOptions`.
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
            self.vector_store = AstraDBVectorStore(
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
                collection_rerank=collection_rerank,
                collection_reranking_api_key=collection_reranking_api_key,
                collection_lexical=collection_lexical,
                hybrid_search=hybrid_search,
                hybrid_limit_factor=hybrid_limit_factor,
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
                    collection_rerank=collection_rerank,
                    collection_reranking_api_key=collection_reranking_api_key,
                    collection_lexical=collection_lexical,
                    hybrid_search=hybrid_search,
                    hybrid_limit_factor=hybrid_limit_factor,
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

    @property
    @override
    def embeddings(self) -> Embeddings | None:
        return self.vector_store.embedding

    def _get_metadata_filter(
        self,
        metadata: dict[str, Any] | None = None,
        outgoing_link: Link | None = None,
    ) -> dict[str, Any]:
        if outgoing_link is None:
            return metadata or {}

        metadata_filter = {} if metadata is None else metadata.copy()
        metadata_filter[self.metadata_incoming_links_key] = _metadata_link_key(
            link=outgoing_link
        )
        return metadata_filter

    def _restore_links(self, doc: Document) -> Document:
        """Restores the links in the document by deserializing them from metadata.

        Args:
            doc: A single Document

        Returns:
            The same Document with restored links.
        """
        links = _deserialize_links(doc.metadata.get(METADATA_LINKS_KEY))
        doc.metadata[METADATA_LINKS_KEY] = links
        if self.metadata_incoming_links_key in doc.metadata:
            del doc.metadata[self.metadata_incoming_links_key]
        return doc

    def _get_node_metadata_for_insertion(self, node: Node) -> dict[str, Any]:
        metadata = node.metadata.copy()
        metadata[METADATA_LINKS_KEY] = _serialize_links(node.links)
        metadata[self.metadata_incoming_links_key] = [
            _metadata_link_key(link=link) for link in _incoming_links(node=node)
        ]
        return metadata

    def _get_docs_for_insertion(
        self, nodes: Iterable[Node]
    ) -> tuple[list[Document], list[str]]:
        docs = []
        ids = []
        for node in nodes:
            node_id = secrets.token_hex(8) if not node.id else node.id

            doc = Document(
                page_content=node.text,
                metadata=self._get_node_metadata_for_insertion(node=node),
                id=node_id,
            )
            docs.append(doc)
            ids.append(node_id)
        return (docs, ids)

    @override
    def add_nodes(
        self,
        nodes: Iterable[Node],
        **kwargs: Any,
    ) -> Iterable[str]:
        """Add nodes to the graph store.

        Args:
            nodes: the nodes to add.
            **kwargs: Additional keyword arguments.
        """
        (docs, ids) = self._get_docs_for_insertion(nodes=nodes)
        return self.vector_store.add_documents(docs, ids=ids)

    @override
    async def aadd_nodes(
        self,
        nodes: Iterable[Node],
        **kwargs: Any,
    ) -> AsyncIterable[str]:
        """Add nodes to the graph store.

        Args:
            nodes: the nodes to add.
            **kwargs: Additional keyword arguments.
        """
        (docs, ids) = self._get_docs_for_insertion(nodes=nodes)
        for inserted_id in await self.vector_store.aadd_documents(docs, ids=ids):
            yield inserted_id

    @classmethod
    @override
    def from_texts(
        cls: type[AstraDBGraphVectorStore],
        texts: Iterable[str],
        embedding: Embeddings | None = None,
        metadatas: list[dict] | None = None,
        ids: Iterable[str] | None = None,
        collection_vector_service_options: VectorServiceOptions | None = None,
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
        collection_vector_service_options: VectorServiceOptions | None = None,
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
        collection_vector_service_options: VectorServiceOptions | None = None,
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
        store.add_documents(documents, ids=ids)
        return store

    @classmethod
    @override
    async def afrom_documents(
        cls: type[AstraDBGraphVectorStore],
        documents: Iterable[Document],
        embedding: Embeddings | None = None,
        ids: Iterable[str] | None = None,
        collection_vector_service_options: VectorServiceOptions | None = None,
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
        await store.aadd_documents(documents, ids=ids)
        return store

    @override
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Retrieve documents from this graph store.

        Args:
            query: The query string.
            k: The number of Documents to return. Defaults to 4.
            filter: Optional metadata to filter the results.
            **kwargs: Additional keyword arguments.

        Returns:
            Collection of retrieved documents.
        """
        return [
            self._restore_links(doc)
            for doc in self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter,
                **kwargs,
            )
        ]

    @override
    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Retrieve documents from this graph store.

        Args:
            query: The query string.
            k: The number of Documents to return. Defaults to 4.
            filter: Optional metadata to filter the results.
            **kwargs: Additional keyword arguments.

        Returns:
            Collection of retrieved documents.
        """
        return [
            self._restore_links(doc)
            for doc in await self.vector_store.asimilarity_search(
                query=query,
                k=k,
                filter=filter,
                **kwargs,
            )
        ]

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
            self._restore_links(doc)
            for doc in self.vector_store.similarity_search_by_vector(
                embedding,
                k=k,
                filter=filter,
                **kwargs,
            )
        ]

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
            self._restore_links(doc)
            for doc in await self.vector_store.asimilarity_search_by_vector(
                embedding,
                k=k,
                filter=filter,
                **kwargs,
            )
        ]

    def metadata_search(
        self,
        filter: dict[str, Any] | None = None,  # noqa: A002
        n: int = 5,
    ) -> Iterable[Document]:
        """Get documents via a metadata search.

        Args:
            filter: the metadata to query for.
            n: the maximum number of documents to return.
        """
        return [
            self._restore_links(doc)
            for doc in self.vector_store.metadata_search(
                filter=filter or {},
                n=n,
            )
        ]

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
        return [
            self._restore_links(doc)
            for doc in await self.vector_store.ametadata_search(
                filter=filter or {},
                n=n,
            )
        ]

    def get_by_document_id(self, document_id: str) -> Document | None:
        """Retrieve a single document from the store, given its document ID.

        Args:
            document_id: The document ID

        Returns:
            The the document if it exists. Otherwise None.
        """
        doc = self.vector_store.get_by_document_id(document_id=document_id)
        return self._restore_links(doc) if doc is not None else None

    async def aget_by_document_id(self, document_id: str) -> Document | None:
        """Retrieve a single document from the store, given its document ID.

        Args:
            document_id: The document ID

        Returns:
            The the document if it exists. Otherwise None.
        """
        doc = await self.vector_store.aget_by_document_id(document_id=document_id)
        return self._restore_links(doc) if doc is not None else None

    def get_node(self, node_id: str) -> Node | None:
        """Retrieve a single node from the store, given its ID.

        Args:
            node_id: The node ID

        Returns:
            The the node if it exists. Otherwise None.
        """
        doc = self.vector_store.get_by_document_id(document_id=node_id)
        if doc is None:
            return None
        return _doc_to_node(doc=doc)

    async def aget_node(self, node_id: str) -> Node | None:
        """Retrieve a single node from the store, given its ID.

        Args:
            node_id: The node ID

        Returns:
            The the node if it exists. Otherwise None.
        """
        doc = await self.vector_store.aget_by_document_id(document_id=node_id)
        if doc is None:
            return None
        return _doc_to_node(doc=doc)

    @override
    async def ammr_traversal_search(  # noqa: C901
        self,
        query: str,
        *,
        initial_roots: Sequence[str] = (),
        k: int = 4,
        depth: int = 2,
        fetch_k: int = 100,
        adjacent_k: int = 10,
        lambda_mult: float = 0.5,
        score_threshold: float = float("-inf"),
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[Document]:
        """Retrieve documents from this graph store using MMR-traversal.

        This strategy first retrieves the top `fetch_k` results by similarity to
        the question. It then selects the top `k` results based on
        maximum-marginal relevance using the given `lambda_mult`.

        At each step, it considers the (remaining) documents from `fetch_k` as
        well as any documents connected by edges to a selected document
        retrieved based on similarity (a "root").

        Args:
            query: The query string to search for.
            initial_roots: Optional list of document IDs to use for initializing search.
                The top `adjacent_k` nodes adjacent to each initial root will be
                included in the set of initial candidates. To fetch only in the
                neighborhood of these nodes, set `fetch_k = 0`.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of initial Documents to fetch via similarity.
                Will be added to the nodes adjacent to `initial_roots`.
                Defaults to 100.
            adjacent_k: Number of adjacent Documents to fetch.
                Defaults to 10.
            depth: Maximum depth of a node (number of edges) from a node
                retrieved via similarity. Defaults to 2.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding to maximum
                diversity and 1 to minimum diversity. Defaults to 0.5.
            score_threshold: Only documents with a score greater than or equal
                this threshold will be chosen. Defaults to -infinity.
            filter: Optional metadata to filter the results.
            **kwargs: Additional keyword arguments.
        """
        # For each unselected node, stores the outgoing links.
        outgoing_links_map: dict[str, set[Link]] = {}
        visited_links: set[Link] = set()
        # Map from id to Document, used as a cache
        retrieved_docs: dict[str, Document] = {}

        def get_candidates(nodes: Iterable[EmbeddedNode]) -> dict[str, list[float]]:
            nonlocal outgoing_links_map

            candidates: dict[str, list[float]] = {}
            for node in nodes:
                if node.id not in outgoing_links_map:
                    outgoing_links_map[node.id] = _outgoing_links(node=node)
                    candidates[node.id] = node.embedding
            return candidates

        async def fetch_initial_candidates() -> (
            tuple[list[float], dict[str, list[float]]]
        ):
            """Gets the embedded query and the set of initial candidates.

            If fetch_k is zero, there will be no initial candidates.
            """
            nonlocal retrieved_docs

            query_embedding, initial_nodes = await self._aget_initial(
                query=query,
                retrieved_docs=retrieved_docs,
                fetch_k=fetch_k,
                filter=filter,
            )

            return query_embedding, get_candidates(nodes=initial_nodes)

        async def fetch_neighborhood_candidates(
            neighborhood: Sequence[str],
        ) -> dict[str, list[float]]:
            nonlocal outgoing_links_map, visited_links, retrieved_docs

            # Put the neighborhood into the outgoing links, to avoid adding it
            # to the candidate set in the future.
            outgoing_links_map.update(
                {content_id: set() for content_id in neighborhood}
            )

            # Initialize the visited_links with the set of outgoing links from the
            # neighborhood. This prevents re-visiting them.
            visited_links = await self._aget_outgoing_links(neighborhood)

            # Call `self._aget_adjacent` to fetch the candidates.
            adjacent_nodes = await self._aget_adjacent(
                links=visited_links,
                query_embedding=query_embedding,
                k_per_link=adjacent_k,
                filter=filter,
                retrieved_docs=retrieved_docs,
            )

            return get_candidates(nodes=adjacent_nodes)

        query_embedding, initial_candidates = await fetch_initial_candidates()
        helper = MmrHelper(
            k=k,
            query_embedding=query_embedding,
            lambda_mult=lambda_mult,
            score_threshold=score_threshold,
        )
        helper.add_candidates(candidates=initial_candidates)

        if initial_roots:
            neighborhood_candidates = await fetch_neighborhood_candidates(initial_roots)
            helper.add_candidates(candidates=neighborhood_candidates)

        # Tracks the depth of each candidate.
        depths = {candidate_id: 0 for candidate_id in helper.candidate_ids()}

        # Select the best item, K times.
        for _ in range(k):
            selected_id = helper.pop_best()

            if selected_id is None:
                break

            next_depth = depths[selected_id] + 1
            if next_depth < depth:
                # If the next nodes would not exceed the depth limit, find the
                # adjacent nodes.

                # Find the links linked to from the selected ID.
                selected_outgoing_links = outgoing_links_map.pop(selected_id)

                # Don't re-visit already visited links.
                selected_outgoing_links.difference_update(visited_links)

                # Find the nodes with incoming links from those links.
                adjacent_nodes = await self._aget_adjacent(
                    links=selected_outgoing_links,
                    query_embedding=query_embedding,
                    k_per_link=adjacent_k,
                    filter=filter,
                    retrieved_docs=retrieved_docs,
                )

                # Record the selected_outgoing_links as visited.
                visited_links.update(selected_outgoing_links)

                new_candidates = {}
                for adjacent_node in adjacent_nodes:
                    if adjacent_node.id not in outgoing_links_map:
                        outgoing_links_map[adjacent_node.id] = _outgoing_links(
                            node=adjacent_node
                        )
                        new_candidates[adjacent_node.id] = adjacent_node.embedding
                        if next_depth < depths.get(adjacent_node.id, depth + 1):
                            # If this is a new shortest depth, or there was no
                            # previous depth, update the depths. This ensures that
                            # when we discover a node we will have the shortest
                            # depth available.
                            #
                            # NOTE: No effort is made to traverse from nodes that
                            # were previously selected if they become reachable via
                            # a shorter path via nodes selected later. This is
                            # currently "intended", but may be worth experimenting
                            # with.
                            depths[adjacent_node.id] = next_depth
                helper.add_candidates(new_candidates)

        for doc_id, similarity_score, mmr_score in zip(
            helper.selected_ids,
            helper.selected_similarity_scores,
            helper.selected_mmr_scores,
        ):
            if doc_id in retrieved_docs:
                doc = self._restore_links(retrieved_docs[doc_id])
                doc.metadata["similarity_score"] = similarity_score
                doc.metadata["mmr_score"] = mmr_score
                yield doc
            else:
                msg = f"retrieved_docs should contain id: {doc_id}"
                raise RuntimeError(msg)

    @override
    def mmr_traversal_search(  # noqa: C901
        self,
        query: str,
        *,
        initial_roots: Sequence[str] = (),
        k: int = 4,
        depth: int = 2,
        fetch_k: int = 100,
        adjacent_k: int = 10,
        lambda_mult: float = 0.5,
        score_threshold: float = float("-inf"),
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterable[Document]:
        """Retrieve documents from this graph store using MMR-traversal.

        This strategy first retrieves the top `fetch_k` results by similarity to
        the question. It then selects the top `k` results based on
        maximum-marginal relevance using the given `lambda_mult`.

        At each step, it considers the (remaining) documents from `fetch_k` as
        well as any documents connected by edges to a selected document
        retrieved based on similarity (a "root").

        Args:
            query: The query string to search for.
            initial_roots: Optional list of document IDs to use for initializing search.
                The top `adjacent_k` nodes adjacent to each initial root will be
                included in the set of initial candidates. To fetch only in the
                neighborhood of these nodes, set `fetch_k = 0`.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of initial Documents to fetch via similarity.
                Will be added to the nodes adjacent to `initial_roots`.
                Defaults to 100.
            adjacent_k: Number of adjacent Documents to fetch.
                Defaults to 10.
            depth: Maximum depth of a node (number of edges) from a node
                retrieved via similarity. Defaults to 2.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding to maximum
                diversity and 1 to minimum diversity. Defaults to 0.5.
            score_threshold: Only documents with a score greater than or equal
                this threshold will be chosen. Defaults to -infinity.
            filter: Optional metadata to filter the results.
            **kwargs: Additional keyword arguments.
        """
        # For each unselected node, stores the outgoing links.
        outgoing_links_map: dict[str, set[Link]] = {}
        visited_links: set[Link] = set()
        # Map from id to Document, used as a cache
        retrieved_docs: dict[str, Document] = {}

        def get_candidates(nodes: Iterable[EmbeddedNode]) -> dict[str, list[float]]:
            nonlocal outgoing_links_map

            candidates: dict[str, list[float]] = {}
            for node in nodes:
                if node.id not in outgoing_links_map:
                    outgoing_links_map[node.id] = _outgoing_links(node=node)
                    candidates[node.id] = node.embedding
            return candidates

        def fetch_initial_candidates() -> tuple[list[float], dict[str, list[float]]]:
            """Gets the embedded query and the set of initial candidates.

            If fetch_k is zero, there will be no initial candidates.
            """
            nonlocal retrieved_docs

            query_embedding, initial_nodes = self._get_initial(
                query=query,
                retrieved_docs=retrieved_docs,
                fetch_k=fetch_k,
                filter=filter,
            )

            return query_embedding, get_candidates(nodes=initial_nodes)

        def fetch_neighborhood_candidates(
            neighborhood: Sequence[str],
        ) -> dict[str, list[float]]:
            nonlocal outgoing_links_map, visited_links, retrieved_docs

            # Put the neighborhood into the outgoing links, to avoid adding it
            # to the candidate set in the future.
            outgoing_links_map.update(
                {content_id: set() for content_id in neighborhood}
            )

            # Initialize the visited_links with the set of outgoing links from the
            # neighborhood. This prevents re-visiting them.
            visited_links = self._get_outgoing_links(neighborhood)

            # Call `self._get_adjacent` to fetch the candidates.
            adjacent_nodes = self._get_adjacent(
                links=visited_links,
                query_embedding=query_embedding,
                k_per_link=adjacent_k,
                filter=filter,
                retrieved_docs=retrieved_docs,
            )

            return get_candidates(nodes=adjacent_nodes)

        query_embedding, initial_candidates = fetch_initial_candidates()
        helper = MmrHelper(
            k=k,
            query_embedding=query_embedding,
            lambda_mult=lambda_mult,
            score_threshold=score_threshold,
        )
        helper.add_candidates(candidates=initial_candidates)

        if initial_roots:
            neighborhood_candidates = fetch_neighborhood_candidates(initial_roots)
            helper.add_candidates(candidates=neighborhood_candidates)

        # Tracks the depth of each candidate.
        depths = {candidate_id: 0 for candidate_id in helper.candidate_ids()}

        # Select the best item, K times.
        for _ in range(k):
            selected_id = helper.pop_best()

            if selected_id is None:
                break

            next_depth = depths[selected_id] + 1
            if next_depth < depth:
                # If the next nodes would not exceed the depth limit, find the
                # adjacent nodes.

                # Find the links linked to from the selected ID.
                selected_outgoing_links = outgoing_links_map.pop(selected_id)

                # Don't re-visit already visited links.
                selected_outgoing_links.difference_update(visited_links)

                # Find the nodes with incoming links from those links.
                adjacent_nodes = self._get_adjacent(
                    links=selected_outgoing_links,
                    query_embedding=query_embedding,
                    k_per_link=adjacent_k,
                    filter=filter,
                    retrieved_docs=retrieved_docs,
                )

                # Record the selected_outgoing_links as visited.
                visited_links.update(selected_outgoing_links)

                new_candidates = {}
                for adjacent_node in adjacent_nodes:
                    if adjacent_node.id not in outgoing_links_map:
                        outgoing_links_map[adjacent_node.id] = _outgoing_links(
                            node=adjacent_node
                        )
                        new_candidates[adjacent_node.id] = adjacent_node.embedding
                        if next_depth < depths.get(adjacent_node.id, depth + 1):
                            # If this is a new shortest depth, or there was no
                            # previous depth, update the depths. This ensures that
                            # when we discover a node we will have the shortest
                            # depth available.
                            #
                            # NOTE: No effort is made to traverse from nodes that
                            # were previously selected if they become reachable via
                            # a shorter path via nodes selected later. This is
                            # currently "intended", but may be worth experimenting
                            # with.
                            depths[adjacent_node.id] = next_depth
                helper.add_candidates(new_candidates)

        for doc_id, similarity_score, mmr_score in zip(
            helper.selected_ids,
            helper.selected_similarity_scores,
            helper.selected_mmr_scores,
        ):
            if doc_id in retrieved_docs:
                doc = self._restore_links(retrieved_docs[doc_id])
                doc.metadata["similarity_score"] = similarity_score
                doc.metadata["mmr_score"] = mmr_score
                yield doc
            else:
                msg = f"retrieved_docs should contain id: {doc_id}"
                raise RuntimeError(msg)

    @override
    async def atraversal_search(  # noqa: C901
        self,
        query: str,
        *,
        k: int = 4,
        depth: int = 1,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[Document]:
        """Retrieve documents from this knowledge store.

        First, `k` nodes are retrieved using a vector search for the `query` string.
        Then, additional nodes are discovered up to the given `depth` from those
        starting nodes.

        Args:
            query: The query string.
            k: The number of Documents to return from the initial vector search.
                Defaults to 4.
            depth: The maximum depth of edges to traverse. Defaults to 1.
            filter: Optional metadata to filter the results.
            **kwargs: Additional keyword arguments.

        Returns:
            Collection of retrieved documents.
        """
        # Depth 0:
        #   Query for `k` nodes similar to the question.
        #   Retrieve `content_id` and `outgoing_links()`.
        #
        # Depth 1:
        #   Query for nodes that have an incoming link in the `outgoing_links()` set.
        #   Combine node IDs.
        #   Query for `outgoing_links()` of those "new" node IDs.
        #
        # ...

        # Map from visited ID to depth
        visited_ids: dict[str, int] = {}

        # Map from visited link to depth
        visited_links: dict[Link, int] = {}

        # Map from id to Document
        retrieved_docs: dict[str, Document] = {}

        async def visit_nodes(d: int, docs: Iterable[Document]) -> None:
            """Recursively visit nodes and their outgoing links."""
            nonlocal visited_ids, visited_links, retrieved_docs

            # Iterate over nodes, tracking the *new* outgoing links for this
            # depth. These are links that are either new, or newly discovered at a
            # lower depth.
            outgoing_links: set[Link] = set()
            for doc in docs:
                if doc.id is not None:
                    if doc.id not in retrieved_docs:
                        retrieved_docs[doc.id] = doc

                    # If this node is at a closer depth, update visited_ids
                    if d <= visited_ids.get(doc.id, depth):
                        visited_ids[doc.id] = d

                        # If we can continue traversing from this node,
                        if d < depth:
                            node = _doc_to_node(doc=doc)
                            # Record any new (or newly discovered at a lower depth)
                            # links to the set to traverse.
                            for link in _outgoing_links(node=node):
                                if d <= visited_links.get(link, depth):
                                    # Record that we'll query this link at the
                                    # given depth, so we don't fetch it again
                                    # (unless we find it an earlier depth)
                                    visited_links[link] = d
                                    outgoing_links.add(link)

            if outgoing_links:
                metadata_search_tasks = []
                for outgoing_link in outgoing_links:
                    metadata_filter = self._get_metadata_filter(
                        metadata=filter,
                        outgoing_link=outgoing_link,
                    )
                    metadata_search_tasks.append(
                        asyncio.create_task(
                            self.vector_store.ametadata_search(
                                filter=metadata_filter, n=1000
                            )
                        )
                    )
                results = await asyncio.gather(*metadata_search_tasks)

                # Visit targets concurrently
                visit_target_tasks = [
                    visit_targets(d=d + 1, docs=docs) for docs in results
                ]
                await asyncio.gather(*visit_target_tasks)

        async def visit_targets(d: int, docs: Iterable[Document]) -> None:
            """Visit target nodes retrieved from outgoing links."""
            nonlocal visited_ids, retrieved_docs

            new_ids_at_next_depth = set()
            for doc in docs:
                if doc.id is not None:
                    if doc.id not in retrieved_docs:
                        retrieved_docs[doc.id] = doc

                    if d <= visited_ids.get(doc.id, depth):
                        new_ids_at_next_depth.add(doc.id)

            if new_ids_at_next_depth:
                visit_node_tasks = [
                    visit_nodes(d=d, docs=[retrieved_docs[doc_id]])
                    for doc_id in new_ids_at_next_depth
                    if doc_id in retrieved_docs
                ]

                fetch_tasks = [
                    asyncio.create_task(
                        self.vector_store.aget_by_document_id(document_id=doc_id)
                    )
                    for doc_id in new_ids_at_next_depth
                    if doc_id not in retrieved_docs
                ]

                new_docs: list[Document | None] = await asyncio.gather(*fetch_tasks)

                visit_node_tasks.extend(
                    visit_nodes(d=d, docs=[new_doc])
                    for new_doc in new_docs
                    if new_doc is not None
                )

                await asyncio.gather(*visit_node_tasks)

        # Start the traversal
        initial_docs = await self.vector_store.asimilarity_search(
            query=query,
            k=k,
            filter=filter,
        )
        await visit_nodes(d=0, docs=initial_docs)

        for doc_id in visited_ids:
            if doc_id in retrieved_docs:
                yield self._restore_links(retrieved_docs[doc_id])
            else:
                msg = f"retrieved_docs should contain id: {doc_id}"
                raise RuntimeError(msg)

    @override
    def traversal_search(  # noqa: C901
        self,
        query: str,
        *,
        k: int = 4,
        depth: int = 1,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterable[Document]:
        """Retrieve documents from this knowledge store.

        First, `k` nodes are retrieved using a vector search for the `query` string.
        Then, additional nodes are discovered up to the given `depth` from those
        starting nodes.

        Args:
            query: The query string.
            k: The number of Documents to return from the initial vector search.
                Defaults to 4.
            depth: The maximum depth of edges to traverse. Defaults to 1.
            filter: Optional metadata to filter the results.
            **kwargs: Additional keyword arguments.

        Returns:
            Collection of retrieved documents.
        """
        # Depth 0:
        #   Query for `k` nodes similar to the question.
        #   Retrieve `content_id` and `outgoing_links()`.
        #
        # Depth 1:
        #   Query for nodes that have an incoming link in the `outgoing_links()` set.
        #   Combine node IDs.
        #   Query for `outgoing_links()` of those "new" node IDs.
        #
        # ...

        # Map from visited ID to depth
        visited_ids: dict[str, int] = {}

        # Map from visited link to depth
        visited_links: dict[Link, int] = {}

        # Map from id to Document
        retrieved_docs: dict[str, Document] = {}

        def visit_nodes(d: int, docs: Iterable[Document]) -> None:
            """Recursively visit nodes and their outgoing links."""
            nonlocal visited_ids, visited_links, retrieved_docs

            # Iterate over nodes, tracking the *new* outgoing links for this
            # depth. These are links that are either new, or newly discovered at a
            # lower depth.
            outgoing_links: set[Link] = set()
            for doc in docs:
                if doc.id is not None:
                    if doc.id not in retrieved_docs:
                        retrieved_docs[doc.id] = doc

                    # If this node is at a closer depth, update visited_ids
                    if d <= visited_ids.get(doc.id, depth):
                        visited_ids[doc.id] = d

                        # If we can continue traversing from this node,
                        if d < depth:
                            node = _doc_to_node(doc=doc)
                            # Record any new (or newly discovered at a lower depth)
                            # links to the set to traverse.
                            for link in _outgoing_links(node=node):
                                if d <= visited_links.get(link, depth):
                                    # Record that we'll query this link at the
                                    # given depth, so we don't fetch it again
                                    # (unless we find it an earlier depth)
                                    visited_links[link] = d
                                    outgoing_links.add(link)

            if outgoing_links:
                for outgoing_link in outgoing_links:
                    metadata_filter = self._get_metadata_filter(
                        metadata=filter,
                        outgoing_link=outgoing_link,
                    )

                    docs = self.vector_store.metadata_search(
                        filter=metadata_filter, n=1000
                    )

                    visit_targets(d=d + 1, docs=docs)

        def visit_targets(d: int, docs: Iterable[Document]) -> None:
            """Visit target nodes retrieved from outgoing links."""
            nonlocal visited_ids, retrieved_docs

            new_ids_at_next_depth = set()
            for doc in docs:
                if doc.id is not None:
                    if doc.id not in retrieved_docs:
                        retrieved_docs[doc.id] = doc

                    if d <= visited_ids.get(doc.id, depth):
                        new_ids_at_next_depth.add(doc.id)

            if new_ids_at_next_depth:
                for doc_id in new_ids_at_next_depth:
                    if doc_id in retrieved_docs:
                        visit_nodes(d=d, docs=[retrieved_docs[doc_id]])
                    else:
                        new_doc = self.vector_store.get_by_document_id(
                            document_id=doc_id
                        )
                        if new_doc is not None:
                            visit_nodes(d=d, docs=[new_doc])

        # Start the traversal
        initial_docs = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter,
        )
        visit_nodes(d=0, docs=initial_docs)

        for doc_id in visited_ids:
            if doc_id in retrieved_docs:
                yield self._restore_links(retrieved_docs[doc_id])
            else:
                msg = f"retrieved_docs should contain id: {doc_id}"
                raise RuntimeError(msg)

    def _get_outgoing_links(self, source_ids: Iterable[str]) -> set[Link]:
        """Return the set of outgoing links for the given source IDs synchronously.

        Args:
            source_ids: The IDs of the source nodes to retrieve outgoing links for.

        Returns:
            A set of `Link` objects representing the outgoing links from the source
            nodes.
        """
        links = set()

        for source_id in source_ids:
            doc = self.vector_store.get_by_document_id(document_id=source_id)
            if doc is not None:
                node = _doc_to_node(doc=doc)
                links.update(_outgoing_links(node=node))

        return links

    async def _aget_outgoing_links(self, source_ids: Iterable[str]) -> set[Link]:
        """Return the set of outgoing links for the given source IDs asynchronously.

        Args:
            source_ids: The IDs of the source nodes to retrieve outgoing links for.

        Returns:
            A set of `Link` objects representing the outgoing links from the source
            nodes.
        """
        links = set()

        # Create coroutine objects without scheduling them yet
        coroutines = [
            self.vector_store.aget_by_document_id(document_id=source_id)
            for source_id in source_ids
        ]

        # Schedule and await all coroutines
        docs = await asyncio.gather(*coroutines)

        for doc in docs:
            if doc is not None:
                node = _doc_to_node(doc=doc)
                links.update(_outgoing_links(node=node))

        return links

    def _get_initial(
        self,
        query: str,
        retrieved_docs: dict[str, Document],
        fetch_k: int,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> tuple[list[float], list[EmbeddedNode]]:
        (
            query_embedding,
            result,
        ) = self.vector_store.similarity_search_with_embedding(
            query=query,
            k=fetch_k,
            filter=filter,
        )

        initial_nodes: list[EmbeddedNode] = []
        for doc, embedding in result:
            if doc.id is not None:
                retrieved_docs[doc.id] = doc
            initial_nodes.append(EmbeddedNode(doc=doc, embedding=embedding))

        return query_embedding, initial_nodes

    async def _aget_initial(
        self,
        query: str,
        retrieved_docs: dict[str, Document],
        fetch_k: int,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> tuple[list[float], list[EmbeddedNode]]:
        (
            query_embedding,
            result,
        ) = await self.vector_store.asimilarity_search_with_embedding(
            query=query,
            k=fetch_k,
            filter=filter,
        )

        initial_nodes: list[EmbeddedNode] = []
        for doc, embedding in result:
            if doc.id is not None:
                retrieved_docs[doc.id] = doc
            initial_nodes.append(EmbeddedNode(doc=doc, embedding=embedding))

        return query_embedding, initial_nodes

    def _get_adjacent(
        self,
        links: set[Link],
        query_embedding: list[float],
        retrieved_docs: dict[str, Document],
        k_per_link: int | None = None,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> Iterable[EmbeddedNode]:
        """Return the target nodes with incoming links from any of the given links.

        Args:
            links: The links to look for.
            query_embedding: The query embedding. Used to rank target nodes.
            retrieved_docs: A cache of retrieved docs. This will be added to.
            k_per_link: The number of target nodes to fetch for each link.
            filter: Optional metadata to filter the results.

        Returns:
            Iterable of adjacent edges.
        """
        targets: dict[str, EmbeddedNode] = {}

        for link in links:
            metadata_filter = self._get_metadata_filter(
                metadata=filter,
                outgoing_link=link,
            )

            result = self.vector_store.similarity_search_with_embedding_by_vector(
                embedding=query_embedding,
                k=k_per_link or 10,
                filter=metadata_filter,
            )

            for doc, embedding in result:
                if doc.id is not None:
                    retrieved_docs[doc.id] = doc
                    if doc.id not in targets:
                        targets[doc.id] = EmbeddedNode(doc=doc, embedding=embedding)

        # TODO: Consider a combined limit based on the similarity and/or
        # predicated MMR score?
        return targets.values()

    async def _aget_adjacent(
        self,
        links: set[Link],
        query_embedding: list[float],
        retrieved_docs: dict[str, Document],
        k_per_link: int | None = None,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> Iterable[EmbeddedNode]:
        """Return the target nodes with incoming links from any of the given links.

        Args:
            links: The links to look for.
            query_embedding: The query embedding. Used to rank target nodes.
            retrieved_docs: A cache of retrieved docs. This will be added to.
            k_per_link: The number of target nodes to fetch for each link.
            filter: Optional metadata to filter the results.

        Returns:
            Iterable of adjacent edges.
        """
        targets: dict[str, EmbeddedNode] = {}

        tasks = []
        for link in links:
            metadata_filter = self._get_metadata_filter(
                metadata=filter,
                outgoing_link=link,
            )

            tasks.append(
                self.vector_store.asimilarity_search_with_embedding_by_vector(
                    embedding=query_embedding,
                    k=k_per_link or 10,
                    filter=metadata_filter,
                )
            )

        results: list[list[tuple[Document, list[float]]]] = await asyncio.gather(*tasks)

        for result in results:
            for doc, embedding in result:
                if doc.id is not None:
                    retrieved_docs[doc.id] = doc
                    if doc.id not in targets:
                        targets[doc.id] = EmbeddedNode(doc=doc, embedding=embedding)

        # TODO: Consider a combined limit based on the similarity and/or
        # predicated MMR score?
        return targets.values()
