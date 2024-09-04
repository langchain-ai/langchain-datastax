"""Astra DB graph vector store integration."""

# ruff: noqa: FIX002 TD002 TD003 C901 D102 EM101 TRY003 SLF001 ANN003 UP007 ARG002

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Iterable,
    Sequence,
    dict_values,
)

from _mmr_helper import MmrHelper
from langchain_core.documents import Document
from langchain_core.graph_vectorstores.base import (
    GraphVectorStore,
    Node,
)
from typing_extensions import override

from langchain_astradb import AstraDBVectorStore
from langchain_astradb.utils.astradb import (
    SetupMode,
    _AstraDBCollectionEnvironment,
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
    from langchain_core.embeddings import Embeddings

DEFAULT_INDEXING_OPTIONS = {"allow": ["metadata"]}


@dataclass
class _Edge:
    target_content_id: str
    target_text_embedding: list[float]
    target_link_to_tags: set[str]
    target_doc: Document


# NOTE: Conversion to string is necessary
# becasue AstraDB doesn't support matching on arrays of tuples
def _tag_to_str(kind: str, tag: str) -> str:
    return f"{kind}:{tag}"


class AstraDBGraphVectorStore(GraphVectorStore):
    def __init__(
        self,
        embedding: Embeddings,
        collection_name: str,
        link_to_metadata_key: str = "links_to",
        link_from_metadata_key: str = "links_from",
        content_id_key: str = "content_id",
        token: str | TokenProvider | None = None,
        api_endpoint: str | None = None,
        astra_db_client: AstraDBClient | None = None,
        async_astra_db_client: AsyncAstraDBClient | None = None,
        namespace: str | None = None,
        setup_mode: SetupMode = SetupMode.SYNC,
        pre_delete_collection: bool = False,  # noqa: FBT001 FBT002
        metric: str | None = None,
        metadata_indexing_include: Iterable[str] | None = None,
        metadata_indexing_exclude: Iterable[str] | None = None,
        collection_indexing_policy: dict[str, Any] | None = None,
        collection_vector_service_options: CollectionVectorServiceOptions | None = None,
        collection_embedding_api_key: str | EmbeddingHeadersProvider | None = None,
    ):
        """Create a new Graph Vector Store backed by AstraDB."""
        self.link_to_metadata_key = link_to_metadata_key
        self.link_from_metadata_key = link_from_metadata_key
        self.content_id_key = content_id_key
        self.session = None
        self.embedding = embedding

        # TODO: Use the _AstraDBCollectionEnvironment everywhere
        # or nowhere - this can cause incompatibilities
        self.vectorstore = AstraDBVectorStore(
            collection_name=collection_name,
            embedding=embedding,
            metadata_indexing_include=metadata_indexing_include,
            metadata_indexing_exclude=metadata_indexing_exclude,
            collection_indexing_policy=collection_indexing_policy,
        )

        self.embedding_dimension: int | None = None
        embedding_dimension: int | Awaitable[int] | None = None
        if self.embedding is not None:
            if setup_mode == SetupMode.ASYNC:
                embedding_dimension = self._aget_embedding_dimension()
            elif setup_mode == SetupMode.SYNC or setup_mode == SetupMode.OFF:  # noqa: PLR1714
                embedding_dimension = self._get_embedding_dimension()

        # indexing policy setting
        indexing_policy: dict[str, Any] = (
            AstraDBVectorStore._normalize_metadata_indexing_policy(
                metadata_indexing_include=metadata_indexing_include,
                metadata_indexing_exclude=metadata_indexing_exclude,
                collection_indexing_policy=collection_indexing_policy,
            )
        )

        self.astra_env = _AstraDBCollectionEnvironment(
            collection_name=collection_name,
            token=token,
            api_endpoint=api_endpoint,
            astra_db_client=astra_db_client,
            async_astra_db_client=async_astra_db_client,
            namespace=namespace,
            setup_mode=setup_mode,
            pre_delete_collection=pre_delete_collection,
            embedding_dimension=embedding_dimension,
            metric=metric,
            requested_indexing_policy=indexing_policy,
            default_indexing_policy=DEFAULT_INDEXING_OPTIONS,
            collection_vector_service_options=collection_vector_service_options,
            collection_embedding_api_key=collection_embedding_api_key,
        )

    def _get_safe_embedding(self) -> Embeddings:
        if not self.embedding:
            msg = "Missing embedding"
            raise ValueError(msg)
        return self.embedding

    def _get_embedding_dimension(self) -> int:
        if self.embedding_dimension is None:
            self.embedding_dimension = len(
                self._get_safe_embedding().embed_query(
                    text="This is a sample sentence."
                )
            )
        return self.embedding_dimension

    async def _aget_embedding_dimension(self) -> int:
        if self.embedding_dimension is None:
            self.embedding_dimension = len(
                await self._get_safe_embedding().aembed_query(
                    text="This is a sample sentence."
                )
            )
        return self.embedding_dimension

    @property
    @override
    def embeddings(self) -> Embeddings | None:
        return self.embedding

    def add_nodes(
        self,
        nodes: Iterable[Node],
        **kwargs,
    ) -> Iterable[str]:
        """Add nodes to the graph store."""
        docs = []
        ids = []
        for node in nodes:
            node_id = secrets.token_hex(8) if not node.id else node.id

            link_to_tags = set()  # link to these tags
            link_from_tags = set()  # link from these tags

            for tag in node.links:
                if tag.direction in {"in", "bidir"}:
                    # An incoming link should be linked *from* nodes with the given
                    # tag.
                    link_from_tags.add(_tag_to_str(tag.kind, tag.tag))
                if tag.direction in {"out", "bidir"}:
                    link_to_tags.add(_tag_to_str(tag.kind, tag.tag))

            metadata = node.metadata
            metadata[self.content_id_key] = node_id
            metadata[self.link_to_metadata_key] = list(link_to_tags)
            metadata[self.link_from_metadata_key] = list(link_from_tags)

            doc = Document(
                page_content=node.text,
                metadata=metadata,
                id=node_id,
            )
            docs.append(doc)
            ids.append(node_id)

        return self.vectorstore.add_documents(docs, ids=ids)

    @classmethod
    def from_texts(
        cls: type[AstraDBGraphVectorStore],
        texts: Iterable[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        ids: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> AstraDBGraphVectorStore:
        store = cls(embedding, **kwargs)
        store.add_texts(texts, metadatas, ids=ids)
        return store

    @classmethod
    def from_documents(
        cls: type[AstraDBGraphVectorStore],
        documents: Iterable[Document],
        embedding: Embeddings,
        ids: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> AstraDBGraphVectorStore:
        store = cls(embedding, **kwargs)
        store.add_documents(documents, ids=ids)
        return store

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        metadata_filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        return self.vectorstore.similarity_search(query, k, metadata_filter, **kwargs)

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        metadata_filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        return self.vectorstore.similarity_search_by_vector(
            embedding, k, metadata_filter, **kwargs
        )

    def traversal_search(
        self,
        query: str,
        *,
        k: int = 4,
        depth: int = 1,
        adjacent_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterable[Document]:
        # Map from visited ID to depth
        visited_ids: dict[str, int] = {}
        visited_docs: list[Document] = []

        # Map from visited tag `(kind, tag)` to depth. Allows skipping queries
        # for tags that we've already traversed.
        visited_tags: dict[str, int] = {}

        def visit_documents(d: int, docs: dict_values[Any, Document]) -> None:
            nonlocal visited_ids
            nonlocal visited_docs
            nonlocal visited_tags

            # Visit documents at the given depth.
            # Each document has `id`, `link_from_tags` and `link_to_tags`.

            # Iterate over documents, tracking the *new* outgoing kind tags for this
            # depth. This is tags that are either new, or newly discovered at a
            # lower depth.
            outgoing_tags = set()
            for doc in docs:
                content_id = doc.metadata[self.content_id_key]

                # Add visited ID. If it is closer it is a new document at this depth:
                if d <= visited_ids.get(content_id, depth):
                    visited_ids[content_id] = d
                    visited_docs.append(doc)

                    # If we can continue traversing from this document,
                    if d < depth and doc.metadata[self.link_to_metadata_key]:
                        # Record any new (or newly discovered at a lower depth)
                        # tags to the set to traverse.
                        for tag in doc.metadata[self.link_to_metadata_key]:
                            if d <= visited_tags.get(tag, depth):
                                # Record that we'll query this tag at the
                                # given depth, so we don't fetch it again
                                # (unless we find it an earlier depth)
                                visited_tags[tag] = d
                                outgoing_tags.add(tag)

            if outgoing_tags:
                # If there are new tags to visit at the next depth, query for the
                # doc IDs.
                for tag in outgoing_tags:
                    m_filter = (metadata_filter or {}).copy()
                    m_filter[self.link_from_metadata_key] = tag

                    rows = self.vectorstore.similarity_search(
                        query=query, k=adjacent_k, filter=m_filter, **kwargs
                    )
                    visit_targets(d, rows)

        def visit_targets(d: int, targets: Sequence[Document]) -> None:
            nonlocal visited_ids

            new_docs_at_next_depth = {}
            for target in targets:
                content_id = target.metadata[self.content_id_key]

                if d < visited_ids.get(content_id, depth):
                    new_docs_at_next_depth[content_id] = target

            if new_docs_at_next_depth:
                visit_documents(d + 1, new_docs_at_next_depth.values())

        docs = self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=metadata_filter,
            **kwargs,
        )
        visit_documents(0, docs)

        return visited_docs

    def mmr_traversal_search(
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
        metadata_filter: dict[str, Any] | None = None,
    ) -> Iterable[Document]:
        query_embedding = self.embedding.embed_query(query)
        helper = MmrHelper(
            k=k,
            query_embedding=query_embedding,
            lambda_mult=lambda_mult,
            score_threshold=score_threshold,
        )

        # For each unselected node, stores the outgoing tags.
        outgoing_tags: dict[str, set[str]] = {}

        visited_tags: set[str] = set()

        def get_adjacent(tags: set[str]) -> Iterable[_Edge]:
            targets: dict[str, _Edge] = {}

            # TODO: Parralelize
            for tag in tags:
                metadata_parameter = self.vectorstore._filter_to_metadata(
                    metadata_filter
                )
                metadata_parameter[f"metadata.{self.link_from_metadata_key}"] = tag
                hits = list(
                    self.astra_env.collection.paginated_find(
                        filter=metadata_parameter,
                        sort={"$vector": query_embedding},
                        options={
                            "limit": adjacent_k,
                            "includeSimilarity": True,
                            "includeSortVector": True,
                        },
                        projection={
                            "_id": True,
                            "content": True,
                            "metadata": True,
                            "$vector": True,
                        },
                    )
                )

                for hit in hits:
                    vector = hit["$vector"]
                    content_id = hit["metadata"][self.content_id_key]

                    if content_id not in targets:
                        targets[content_id] = _Edge(
                            target_content_id=content_id,
                            target_text_embedding=vector,
                            target_link_to_tags=set(
                                hit.get(self.link_to_metadata_key, [])
                            ),
                            target_doc=Document(
                                page_content=hit["content"],
                                metadata=hit["metadata"],
                            ),
                        )

            # TODO: Consider a combined limit based on the similarity and/or
            # predicated MMR score?
            return targets.values()

        def fetch_neighborhood(neighborhood: Sequence[str]) -> None:
            # Put the neighborhood into the outgoing tags, to avoid adding it
            # to the candidate set in the future.
            outgoing_tags.update({content_id: set() for content_id in neighborhood})

            # Initialize the visited_tags with the set of outgoing from the
            # neighborhood. This prevents re-visiting them.
            visited_tags = self._get_outgoing_tags(neighborhood)

            # Call `self._get_adjacent` to fetch the candidates.
            adjacents = get_adjacent(visited_tags)

            new_candidates = {}
            for adjacent in adjacents:
                if adjacent.target_content_id not in outgoing_tags:
                    outgoing_tags[adjacent.target_content_id] = (
                        adjacent.target_link_to_tags
                    )

                    new_candidates[adjacent.target_content_id] = (
                        adjacent.target_doc,
                        adjacent.target_text_embedding,
                    )
            helper.add_candidates(new_candidates)

        def fetch_initial_candidates() -> None:
            metadata_parameter = self.vectorstore._filter_to_metadata(metadata_filter)
            hits = list(
                self.astra_env.collection.paginated_find(
                    filter=metadata_parameter,
                    sort={"$vector": query_embedding},
                    options={
                        "limit": fetch_k,
                        "includeSimilarity": True,
                        "includeSortVector": True,
                    },
                    projection={
                        "_id": True,
                        "content": True,
                        "metadata": True,
                        "$vector": True,
                    },
                )
            )

            candidates = {}
            for hit in hits:
                vector = hit["$vector"]
                content_id = hit["metadata"][self.content_id_key]
                doc = Document(page_content=hit["content"], metadata=hit["metadata"])

                candidates[content_id] = (doc, vector)
                tags = set(hit["metadata"].get(self.link_to_metadata_key, []))
                outgoing_tags[content_id] = tags

            helper.add_candidates(candidates)

        if initial_roots:
            fetch_neighborhood(initial_roots)
        if fetch_k > 0:
            fetch_initial_candidates()

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
                #
                # TODO: For a big performance win, we should track which tags we've
                # already incorporated. We don't need to issue adjacent queries for
                # those.

                # Find the tags linked to from the selected ID.
                link_to_tags = outgoing_tags.pop(selected_id)

                # Don't re-visit already visited tags.
                link_to_tags.difference_update(visited_tags)

                # Find the nodes with incoming links from those tags.
                adjacents = get_adjacent(link_to_tags)

                # Record the link_to_tags as visited.
                visited_tags.update(link_to_tags)

                new_candidates = {}
                for adjacent in adjacents:
                    if adjacent.target_content_id not in outgoing_tags:
                        outgoing_tags[adjacent.target_content_id] = (
                            adjacent.target_link_to_tags
                        )
                        new_candidates[adjacent.target_content_id] = (
                            adjacent.target_doc,
                            adjacent.target_text_embedding,
                        )
                        if next_depth < depths.get(
                            adjacent.target_content_id, depth + 1
                        ):
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
                            depths[adjacent.target_content_id] = next_depth
                helper.add_candidates(new_candidates)

        return [helper.candidate_docs[sid] for sid in helper.selected_ids]
