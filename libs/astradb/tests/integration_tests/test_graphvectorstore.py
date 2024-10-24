"""Test of Astra DB graph vector store class `AstraDBGraphVectorStore`

Refer to `test_vectorstores.py` for the requirements to run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from astrapy.authentication import StaticTokenProvider
from langchain_community.graph_vectorstores.base import Node
from langchain_community.graph_vectorstores.links import Link, add_links
from langchain_core.documents import Document

from langchain_astradb.graph_vectorstores import AstraDBGraphVectorStore
from langchain_astradb.utils.astradb import SetupMode

from .conftest import (
    CUSTOM_CONTENT_KEY,
    LONG_TEXT,
    OPENAI_VECTORIZE_OPTIONS_HEADER,
    astra_db_env_vars_available,
)

if TYPE_CHECKING:
    from astrapy import Collection, Database
    from langchain_core.embeddings import Embeddings

    from .conftest import AstraDBCredentials


@pytest.fixture
def graph_vector_store_docs() -> list[Document]:
    """
    This is a set of Documents to pre-populate a graph vector store,
    with entries placed in a certain way.

    Space of the entries (under Euclidean similarity):

                      A0    (*)
        ....        AL   AR       <....
        :              |              :
        :              |  ^           :
        v              |  .           v
                       |   :
       TR              |   :          BL
    T0   --------------x--------------   B0
       TL              |   :          BR
                       |   :
                       |  .
                       | .
                       |
                    FL   FR
                      F0

    the query point is meant to be at (*).
    the A nodes are linked bidirectionally with B
    the A nodes are linked outgoing to T
    the A nodes are linked incoming from F
    The links are like: L with L, 0 with 0 and R with R.
    """

    docs_a = [
        Document(id="AL", page_content="[-1, 9]", metadata={"label": "AL"}),
        Document(id="A0", page_content="[0, 10]", metadata={"label": "A0"}),
        Document(id="AR", page_content="[1, 9]", metadata={"label": "AR"}),
    ]
    docs_b = [
        Document(id="BL", page_content="[9, 1]", metadata={"label": "BL"}),
        Document(id="B0", page_content="[10, 0]", metadata={"label": "B0"}),
        Document(id="BL", page_content="[9, -1]", metadata={"label": "BR"}),
    ]
    docs_f = [
        Document(id="FL", page_content="[1, -9]", metadata={"label": "FL"}),
        Document(id="F0", page_content="[0, -10]", metadata={"label": "F0"}),
        Document(id="FR", page_content="[-1, -9]", metadata={"label": "FR"}),
    ]
    docs_t = [
        Document(id="TL", page_content="[-9, -1]", metadata={"label": "TL"}),
        Document(id="T0", page_content="[-10, 0]", metadata={"label": "T0"}),
        Document(id="TR", page_content="[-9, 1]", metadata={"label": "TR"}),
    ]
    for doc_a, suffix in zip(docs_a, ["l", "0", "r"]):
        add_links(doc_a, Link.bidir(kind="ab_example", tag=f"tag_{suffix}"))
        add_links(doc_a, Link.outgoing(kind="at_example", tag=f"tag_{suffix}"))
        add_links(doc_a, Link.incoming(kind="af_example", tag=f"tag_{suffix}"))
    for doc_b, suffix in zip(docs_b, ["l", "0", "r"]):
        add_links(doc_b, Link.bidir(kind="ab_example", tag=f"tag_{suffix}"))
    for doc_t, suffix in zip(docs_t, ["l", "0", "r"]):
        add_links(doc_t, Link.incoming(kind="at_example", tag=f"tag_{suffix}"))
    for doc_f, suffix in zip(docs_f, ["l", "0", "r"]):
        add_links(doc_f, Link.outgoing(kind="af_example", tag=f"tag_{suffix}"))
    return docs_a + docs_b + docs_f + docs_t


@pytest.fixture
def graph_vector_store_docs_vz() -> list[Document]:
    """
    This is a set of Documents to pre-populate a graph vector store,
    with entries placed in a certain way.

    the A nodes are linked bidirectionally with B
    the A nodes are linked outgoing to T
    the A nodes are linked incoming from F
    The links are like: L with L, 0 with 0 and R with R.
    """

    docs_a = [  # docs related to space and the universe
        Document(
            id="AL", page_content="planets orbit quietly", metadata={"label": "AL"}
        ),
        Document(id="A0", page_content="distant stars shine", metadata={"label": "A0"}),
        Document(
            id="AR", page_content="nebulae swirl in space", metadata={"label": "AR"}
        ),
    ]
    docs_b = [  # docs related to emotions and relationships
        Document(id="BL", page_content="hearts intertwined", metadata={"label": "BL"}),
        Document(id="B0", page_content="a gentle embrace", metadata={"label": "B0"}),
        Document(id="BL", page_content="love conquers all", metadata={"label": "BR"}),
    ]
    docs_f = [  # docs related to technology and programming
        Document(
            id="FL", page_content="code compiles efficiently", metadata={"label": "FL"}
        ),
        Document(
            id="F0", page_content="a neural network learns", metadata={"label": "F0"}
        ),
        Document(
            id="FR", page_content="data structures organize", metadata={"label": "FR"}
        ),
    ]
    docs_t = [  # docs related to nature and wildlife
        Document(
            id="TL", page_content="trees sway in the wind", metadata={"label": "TL"}
        ),
        Document(id="T0", page_content="a river runs deep", metadata={"label": "T0"}),
        Document(
            id="TR", page_content="birds chirping at dawn", metadata={"label": "TR"}
        ),
    ]
    for doc_a, suffix in zip(docs_a, ["l", "0", "r"]):
        add_links(doc_a, Link.bidir(kind="ab_example", tag=f"tag_{suffix}"))
        add_links(doc_a, Link.outgoing(kind="at_example", tag=f"tag_{suffix}"))
        add_links(doc_a, Link.incoming(kind="af_example", tag=f"tag_{suffix}"))
    for doc_b, suffix in zip(docs_b, ["l", "0", "r"]):
        add_links(doc_b, Link.bidir(kind="ab_example", tag=f"tag_{suffix}"))
    for doc_t, suffix in zip(docs_t, ["l", "0", "r"]):
        add_links(doc_t, Link.incoming(kind="at_example", tag=f"tag_{suffix}"))
    for doc_f, suffix in zip(docs_f, ["l", "0", "r"]):
        add_links(doc_f, Link.outgoing(kind="af_example", tag=f"tag_{suffix}"))
    return docs_a + docs_b + docs_f + docs_t


@pytest.fixture
def auth_kwargs(
    astra_db_credentials: AstraDBCredentials,
) -> dict[str, Any]:
    return {
        "token": StaticTokenProvider(astra_db_credentials["token"]),
        "api_endpoint": astra_db_credentials["api_endpoint"],
        "namespace": astra_db_credentials["namespace"],
        "environment": astra_db_credentials["environment"],
    }


@pytest.fixture
def graph_vector_store_d2(
    auth_kwargs: dict[str, Any],
    empty_collection_d2: Collection,
    embedding_d2: Embeddings,
) -> AstraDBGraphVectorStore:
    return AstraDBGraphVectorStore(
        embedding=embedding_d2,
        collection_name=empty_collection_d2.name,
        setup_mode=SetupMode.OFF,
        **auth_kwargs,
    )


@pytest.fixture
def graph_vector_store_vz(
    auth_kwargs: dict[str, Any],
    openai_api_key: str,
    empty_collection_vz: Collection,
) -> AstraDBGraphVectorStore:
    return AstraDBGraphVectorStore(
        collection_vector_service_options=OPENAI_VECTORIZE_OPTIONS_HEADER,
        collection_embedding_api_key=openai_api_key,
        collection_name=empty_collection_vz.name,
        setup_mode=SetupMode.OFF,
        **auth_kwargs,
    )


@pytest.fixture
def populated_graph_vector_store_d2(
    graph_vector_store_d2: AstraDBGraphVectorStore,
    graph_vector_store_docs: list[Document],
) -> AstraDBGraphVectorStore:
    graph_vector_store_d2.add_documents(graph_vector_store_docs)
    return graph_vector_store_d2


@pytest.fixture
def populated_graph_vector_store_vz(
    graph_vector_store_vz: AstraDBGraphVectorStore,
    graph_vector_store_docs_vz: list[Document],
) -> AstraDBGraphVectorStore:
    graph_vector_store_vz.add_documents(graph_vector_store_docs_vz)
    return graph_vector_store_vz


@pytest.fixture
def autodetect_populated_graph_vector_store_d2(
    auth_kwargs: dict[str, Any],
    database: Database,
    embedding_d2: Embeddings,
    graph_vector_store_docs: list[Document],
    ephemeral_collection_cleaner_idxall_d2: str,
) -> AstraDBGraphVectorStore:
    """
    Pre-populate the collection and have (VectorStore)autodetect work on it,
    then create and return a GraphVectorStore, additionally filled with
    the same (graph-)entries as for `populated_graph_vector_store_d2`.
    """
    empty_collection_d2_idxall = database.create_collection(
        ephemeral_collection_cleaner_idxall_d2,
        dimension=2,
        check_exists=False,
        metric="euclidean",
    )
    empty_collection_d2_idxall.insert_many(
        [
            {
                CUSTOM_CONTENT_KEY: LONG_TEXT,
                "$vector": [100, 0],
                "mds": "S",
                "mdi": 100,
            },
            {
                CUSTOM_CONTENT_KEY: LONG_TEXT,
                "$vector": [100, 1],
                "mds": "T",
                "mdi": 101,
            },
            {
                CUSTOM_CONTENT_KEY: LONG_TEXT,
                "$vector": [100, 2],
                "mds": "U",
                "mdi": 102,
            },
        ]
    )
    g_store = AstraDBGraphVectorStore(
        embedding=embedding_d2,
        collection_name=ephemeral_collection_cleaner_idxall_d2,
        metadata_incoming_links_key="x_link_to_x",
        content_field="*",
        autodetect_collection=True,
        **auth_kwargs,
    )
    g_store.add_documents(graph_vector_store_docs)
    return g_store


@pytest.fixture
def autodetect_populated_graph_vector_store_vz(
    auth_kwargs: dict[str, Any],
    openai_api_key: str,
    graph_vector_store_docs_vz: list[Document],
    empty_collection_idxall_vz: Collection,
) -> AstraDBGraphVectorStore:
    """
    Pre-populate the collection and have (VectorStore)autodetect work on it,
    then create and return a GraphVectorStore, additionally filled with
    the same (graph-)entries as for `populated_graph_vector_store_vz`.
    """
    empty_collection_idxall_vz.insert_many(
        [
            {
                "_id": "1",
                "$vectorize": "Cont1",
                "mds": "S",
                "mdi": 100,
            },
            {
                "_id": "2",
                "$vectorize": "Cont2",
                "mds": "T",
                "mdi": 101,
            },
            {
                "_id": "3",
                "$vectorize": "Cont3",
                "mds": "U",
                "mdi": 102,
            },
        ]
    )
    g_store = AstraDBGraphVectorStore(
        collection_embedding_api_key=openai_api_key,
        collection_name=empty_collection_idxall_vz.name,
        metadata_incoming_links_key="x_link_to_x",
        autodetect_collection=True,
        **auth_kwargs,
    )
    g_store.add_documents(graph_vector_store_docs_vz)
    return g_store


def assert_all_flat_docs(collection: Collection, is_vectorize: bool) -> None:  # noqa: FBT001
    """
    Check that all docs in the store obey the underlying (flat) autodetected
    doc schema on DB.
    Useful for checking the store after graph-store-driven insertions.
    """
    for doc in collection.find({}, projection={"*": True}):
        assert all(not isinstance(v, dict) for v in doc.values())
        content_key = "$vectorize" if is_vectorize else CUSTOM_CONTENT_KEY
        assert content_key in doc
        assert isinstance(doc["$vector"], list)


@pytest.mark.skipif(
    not astra_db_env_vars_available(), reason="Missing Astra DB env. vars"
)
class TestAstraDBGraphVectorStore:
    @pytest.mark.parametrize(
        ("store_name", "is_autodetected", "is_vectorize"),
        [
            ("populated_graph_vector_store_d2", False, False),
            ("autodetect_populated_graph_vector_store_d2", True, False),
            ("populated_graph_vector_store_vz", False, True),
            ("autodetect_populated_graph_vector_store_vz", True, True),
        ],
        ids=[
            "native_store_d2",
            "autodetected_store_d2",
            "native_store_vz",
            "autodetected_store_vz",
        ],
    )
    def test_gvs_similarity_search_sync(
        self,
        *,
        store_name: str,
        is_autodetected: bool,
        is_vectorize: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """Simple (non-graph) similarity search on a graph vector store."""
        g_store: AstraDBGraphVectorStore = request.getfixturevalue(store_name)
        query = "universe" if is_vectorize else "[2, 10]"
        embedding = [2.0, 10.0]

        ss_response = g_store.similarity_search(query=query, k=2)
        ss_labels = [doc.metadata["label"] for doc in ss_response]
        assert ss_labels == ["AR", "A0"]

        if is_vectorize:
            with pytest.raises(
                ValueError, match=r"Searching by vector .* embeddings is not allowed"
            ):
                g_store.similarity_search_by_vector(embedding=embedding, k=2)
        else:
            ss_by_v_response = g_store.similarity_search_by_vector(
                embedding=embedding, k=2
            )
            ss_by_v_labels = [doc.metadata["label"] for doc in ss_by_v_response]
            assert ss_by_v_labels == ["AR", "A0"]

        if is_autodetected:
            assert_all_flat_docs(
                g_store.vector_store.astra_env.collection, is_vectorize=is_vectorize
            )

    @pytest.mark.parametrize(
        ("store_name", "is_autodetected", "is_vectorize"),
        [
            ("populated_graph_vector_store_d2", False, False),
            ("autodetect_populated_graph_vector_store_d2", True, False),
            ("populated_graph_vector_store_vz", False, True),
            ("autodetect_populated_graph_vector_store_vz", True, True),
        ],
        ids=[
            "native_store_d2",
            "autodetected_store_d2",
            "native_store_vz",
            "autodetected_store_vz",
        ],
    )
    async def test_gvs_similarity_search_async(
        self,
        *,
        store_name: str,
        is_autodetected: bool,
        is_vectorize: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """Simple (non-graph) similarity search on a graph vector store."""
        g_store: AstraDBGraphVectorStore = request.getfixturevalue(store_name)
        query = "universe" if is_vectorize else "[2, 10]"
        embedding = [2.0, 10.0]

        ss_response = await g_store.asimilarity_search(query=query, k=2)
        ss_labels = [doc.metadata["label"] for doc in ss_response]
        assert ss_labels == ["AR", "A0"]

        if is_vectorize:
            with pytest.raises(
                ValueError, match=r"Searching by vector .* embeddings is not allowed"
            ):
                await g_store.asimilarity_search_by_vector(embedding=embedding, k=2)
        else:
            ss_by_v_response = await g_store.asimilarity_search_by_vector(
                embedding=embedding, k=2
            )
            ss_by_v_labels = [doc.metadata["label"] for doc in ss_by_v_response]
            assert ss_by_v_labels == ["AR", "A0"]

        if is_autodetected:
            assert_all_flat_docs(
                g_store.vector_store.astra_env.collection, is_vectorize=is_vectorize
            )

    @pytest.mark.parametrize(
        ("store_name", "is_autodetected", "is_vectorize"),
        [
            ("populated_graph_vector_store_d2", False, False),
            ("autodetect_populated_graph_vector_store_d2", True, False),
            ("populated_graph_vector_store_vz", False, True),
            ("autodetect_populated_graph_vector_store_vz", True, True),
        ],
        ids=[
            "native_store_d2",
            "autodetected_store_d2",
            "native_store_vz",
            "autodetected_store_vz",
        ],
    )
    def test_gvs_traversal_search_sync(
        self,
        *,
        store_name: str,
        is_autodetected: bool,
        is_vectorize: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """Graph traversal search on a graph vector store."""
        g_store: AstraDBGraphVectorStore = request.getfixturevalue(store_name)
        query = "universe" if is_vectorize else "[2, 10]"

        # this is a set, as some of the internals of trav.search are set-driven
        # so ordering is not deterministic:
        ts_labels = {
            doc.metadata["label"]
            for doc in g_store.traversal_search(query=query, k=2, depth=2)
        }
        assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}
        if is_autodetected:
            assert_all_flat_docs(
                g_store.vector_store.astra_env.collection, is_vectorize=is_vectorize
            )

    @pytest.mark.parametrize(
        ("store_name", "is_autodetected", "is_vectorize"),
        [
            ("populated_graph_vector_store_d2", False, False),
            ("autodetect_populated_graph_vector_store_d2", True, False),
            ("populated_graph_vector_store_vz", False, True),
            ("autodetect_populated_graph_vector_store_vz", True, True),
        ],
        ids=[
            "native_store_d2",
            "autodetected_store_d2",
            "native_store_vz",
            "autodetected_store_vz",
        ],
    )
    async def test_gvs_traversal_search_async(
        self,
        *,
        store_name: str,
        is_autodetected: bool,
        is_vectorize: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """Graph traversal search on a graph vector store."""
        g_store: AstraDBGraphVectorStore = request.getfixturevalue(store_name)
        query = "universe" if is_vectorize else "[2, 10]"

        # this is a set, as some of the internals of trav.search are set-driven
        # so ordering is not deterministic:
        ts_labels = {
            doc.metadata["label"]
            async for doc in g_store.atraversal_search(query=query, k=2, depth=2)
        }
        assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}
        if is_autodetected:
            assert_all_flat_docs(
                g_store.vector_store.astra_env.collection, is_vectorize=is_vectorize
            )

    @pytest.mark.parametrize(
        ("store_name", "is_autodetected", "is_vectorize"),
        [
            ("populated_graph_vector_store_d2", False, False),
            ("autodetect_populated_graph_vector_store_d2", True, False),
            ("populated_graph_vector_store_vz", False, True),
            ("autodetect_populated_graph_vector_store_vz", True, True),
        ],
        ids=[
            "native_store_d2",
            "autodetected_store_d2",
            "native_store_vz",
            "autodetected_store_vz",
        ],
    )
    def test_gvs_mmr_traversal_search_sync(
        self,
        *,
        store_name: str,
        is_autodetected: bool,
        is_vectorize: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """MMR Graph traversal search on a graph vector store."""
        g_store: AstraDBGraphVectorStore = request.getfixturevalue(store_name)
        query = "universe" if is_vectorize else "[2, 10]"

        mt_labels = [
            doc.metadata["label"]
            for doc in g_store.mmr_traversal_search(
                query=query,
                k=2,
                depth=2,
                fetch_k=1,
                adjacent_k=2,
                lambda_mult=0.1,
            )
        ]

        assert mt_labels == ["AR", "BR"]
        if is_autodetected:
            assert_all_flat_docs(
                g_store.vector_store.astra_env.collection, is_vectorize=is_vectorize
            )

    @pytest.mark.parametrize(
        ("store_name", "is_autodetected", "is_vectorize"),
        [
            ("populated_graph_vector_store_d2", False, False),
            ("autodetect_populated_graph_vector_store_d2", True, False),
            ("populated_graph_vector_store_vz", False, True),
            ("autodetect_populated_graph_vector_store_vz", True, True),
        ],
        ids=[
            "native_store_d2",
            "autodetected_store_d2",
            "native_store_vz",
            "autodetected_store_vz",
        ],
    )
    async def test_gvs_mmr_traversal_search_async(
        self,
        *,
        store_name: str,
        is_autodetected: bool,
        is_vectorize: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """MMR Graph traversal search on a graph vector store."""
        g_store: AstraDBGraphVectorStore = request.getfixturevalue(store_name)
        query = "universe" if is_vectorize else "[2, 10]"

        mt_labels = [
            doc.metadata["label"]
            async for doc in g_store.ammr_traversal_search(
                query=query,
                k=2,
                depth=2,
                fetch_k=1,
                adjacent_k=2,
                lambda_mult=0.1,
            )
        ]

        assert mt_labels == ["AR", "BR"]
        if is_autodetected:
            assert_all_flat_docs(
                g_store.vector_store.astra_env.collection, is_vectorize=is_vectorize
            )

    @pytest.mark.parametrize(
        "store_name",
        [
            "populated_graph_vector_store_d2",
            "autodetect_populated_graph_vector_store_d2",
            "populated_graph_vector_store_vz",
            "autodetect_populated_graph_vector_store_vz",
        ],
        ids=[
            "native_store_d2",
            "autodetected_store_d2",
            "native_store_vz",
            "autodetected_store_vz",
        ],
    )
    def test_gvs_metadata_search_sync(
        self,
        *,
        store_name: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """Metadata search on a graph vector store."""
        g_store: AstraDBGraphVectorStore = request.getfixturevalue(store_name)
        mt_response = g_store.metadata_search(
            filter={"label": "T0"},
            n=2,
        )
        doc: Document = next(iter(mt_response))
        assert doc.id == "T0"
        links = doc.metadata["links"]
        assert len(links) == 1
        link: Link = links.pop()
        assert isinstance(link, Link)
        assert link.direction == "in"
        assert link.kind == "at_example"
        assert link.tag == "tag_0"

    @pytest.mark.parametrize(
        "store_name",
        [
            "populated_graph_vector_store_d2",
            "autodetect_populated_graph_vector_store_d2",
            "populated_graph_vector_store_vz",
            "autodetect_populated_graph_vector_store_vz",
        ],
        ids=[
            "native_store_d2",
            "autodetected_store_d2",
            "native_store_vz",
            "autodetected_store_vz",
        ],
    )
    async def test_gvs_metadata_search_async(
        self,
        *,
        store_name: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """Metadata search on a graph vector store."""
        g_store: AstraDBGraphVectorStore = request.getfixturevalue(store_name)
        mt_response = await g_store.ametadata_search(
            filter={"label": "T0"},
            n=2,
        )
        doc: Document = next(iter(mt_response))
        assert doc.id == "T0"
        links: set[Link] = doc.metadata["links"]
        assert len(links) == 1
        link: Link = links.pop()
        assert isinstance(link, Link)
        assert link.direction == "in"
        assert link.kind == "at_example"
        assert link.tag == "tag_0"

    @pytest.mark.parametrize(
        "store_name",
        [
            "populated_graph_vector_store_d2",
            "autodetect_populated_graph_vector_store_d2",
            "populated_graph_vector_store_vz",
            "autodetect_populated_graph_vector_store_vz",
        ],
        ids=[
            "native_store_d2",
            "autodetected_store_d2",
            "native_store_vz",
            "autodetected_store_vz",
        ],
    )
    def test_gvs_get_by_document_id_sync(
        self,
        *,
        store_name: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """Get by document_id on a graph vector store."""
        g_store: AstraDBGraphVectorStore = request.getfixturevalue(store_name)
        doc = g_store.get_by_document_id(document_id="FL")
        assert doc is not None
        assert doc.metadata["label"] == "FL"
        links = doc.metadata["links"]
        assert len(links) == 1
        link: Link = links.pop()
        assert isinstance(link, Link)
        assert link.direction == "out"
        assert link.kind == "af_example"
        assert link.tag == "tag_l"

        invalid_doc = g_store.get_by_document_id(document_id="invalid")
        assert invalid_doc is None

    @pytest.mark.parametrize(
        "store_name",
        [
            "populated_graph_vector_store_d2",
            "autodetect_populated_graph_vector_store_d2",
            "populated_graph_vector_store_vz",
            "autodetect_populated_graph_vector_store_vz",
        ],
        ids=[
            "native_store_d2",
            "autodetected_store_d2",
            "native_store_vz",
            "autodetected_store_vz",
        ],
    )
    async def test_gvs_get_by_document_id_async(
        self,
        *,
        store_name: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """Get by document_id on a graph vector store."""
        g_store: AstraDBGraphVectorStore = request.getfixturevalue(store_name)
        doc = await g_store.aget_by_document_id(document_id="FL")
        assert doc is not None
        assert doc.metadata["label"] == "FL"
        links = doc.metadata["links"]
        assert len(links) == 1
        link: Link = links.pop()
        assert isinstance(link, Link)
        assert link.direction == "out"
        assert link.kind == "af_example"
        assert link.tag == "tag_l"

        invalid_doc = await g_store.aget_by_document_id(document_id="invalid")
        assert invalid_doc is None

    @pytest.mark.parametrize(
        ("is_vectorize", "page_contents", "collection_fixture_name"),
        [
            (False, ["[1, 2]"], "empty_collection_d2"),
            (True, ["varenyky, holubtsi, and deruny"], "empty_collection_vz"),
        ],
        ids=["nonvectorize_store", "vectorize_store"],
    )
    def test_gvs_from_texts(
        self,
        *,
        auth_kwargs: dict[str, Any],
        openai_api_key: str,
        embedding_d2: Embeddings,
        is_vectorize: bool,
        page_contents: list[str],
        collection_fixture_name: str,
        request: pytest.FixtureRequest,
    ) -> None:
        collection: Collection = request.getfixturevalue(collection_fixture_name)
        init_kwargs: dict[str, Any]
        if is_vectorize:
            init_kwargs = {
                "collection_vector_service_options": OPENAI_VECTORIZE_OPTIONS_HEADER,
                "collection_embedding_api_key": openai_api_key,
            }
        else:
            init_kwargs = {"embedding": embedding_d2}

        content_field = CUSTOM_CONTENT_KEY if not is_vectorize else None

        g_store = AstraDBGraphVectorStore.from_texts(
            texts=page_contents,
            metadatas=[{"md": 1}],
            ids=["x_id"],
            collection_name=collection.name,
            content_field=content_field,
            setup_mode=SetupMode.OFF,
            **auth_kwargs,
            **init_kwargs,
        )

        query = "ukrainian food" if is_vectorize else "[2, 1]"
        hits = g_store.similarity_search(query=query, k=2)
        assert len(hits) == 1
        assert hits[0].page_content == page_contents[0]
        assert hits[0].id == "x_id"
        assert hits[0].metadata["md"] == 1

    @pytest.mark.parametrize(
        ("is_vectorize", "page_contents", "collection_fixture_name"),
        [
            (False, ["[1, 2]"], "empty_collection_d2"),
            (True, ["varenyky, holubtsi, and deruny"], "empty_collection_vz"),
        ],
        ids=["nonvectorize_store", "vectorize_store"],
    )
    async def test_gvs_from_texts_async(
        self,
        *,
        auth_kwargs: dict[str, Any],
        openai_api_key: str,
        embedding_d2: Embeddings,
        is_vectorize: bool,
        page_contents: list[str],
        collection_fixture_name: str,
        request: pytest.FixtureRequest,
    ) -> None:
        collection: Collection = request.getfixturevalue(collection_fixture_name)
        init_kwargs: dict[str, Any]
        if is_vectorize:
            init_kwargs = {
                "collection_vector_service_options": OPENAI_VECTORIZE_OPTIONS_HEADER,
                "collection_embedding_api_key": openai_api_key,
            }
        else:
            init_kwargs = {"embedding": embedding_d2}

        content_field = CUSTOM_CONTENT_KEY if not is_vectorize else None

        g_store = await AstraDBGraphVectorStore.afrom_texts(
            texts=page_contents,
            metadatas=[{"md": 1}],
            ids=["x_id"],
            collection_name=collection.name,
            content_field=content_field,
            **auth_kwargs,
            **init_kwargs,
        )

        query = "ukrainian food" if is_vectorize else "[2, 1]"
        hits = g_store.similarity_search(query=query, k=2)
        assert len(hits) == 1
        assert hits[0].page_content == page_contents[0]
        assert hits[0].id == "x_id"
        assert hits[0].metadata["md"] == 1

    @pytest.mark.parametrize(
        ("is_vectorize", "page_contents", "collection_fixture_name"),
        [
            (False, ["[1, 2]"], "empty_collection_d2"),
            (True, ["tacos, tamales, and mole"], "empty_collection_vz"),
        ],
        ids=["nonvectorize_store", "vectorize_store"],
    )
    def test_gvs_from_documents_containing_ids(
        self,
        *,
        auth_kwargs: dict[str, Any],
        openai_api_key: str,
        embedding_d2: Embeddings,
        is_vectorize: bool,
        page_contents: list[str],
        collection_fixture_name: str,
        request: pytest.FixtureRequest,
    ) -> None:
        collection: Collection = request.getfixturevalue(collection_fixture_name)
        init_kwargs: dict[str, Any]
        if is_vectorize:
            init_kwargs = {
                "collection_vector_service_options": OPENAI_VECTORIZE_OPTIONS_HEADER,
                "collection_embedding_api_key": openai_api_key,
            }
        else:
            init_kwargs = {"embedding": embedding_d2}

        content_field = CUSTOM_CONTENT_KEY if not is_vectorize else None

        the_document = Document(
            page_content=page_contents[0],
            metadata={"md": 1},
            id="x_id",
        )
        g_store = AstraDBGraphVectorStore.from_documents(
            documents=[the_document],
            collection_name=collection.name,
            content_field=content_field,
            setup_mode=SetupMode.OFF,
            **auth_kwargs,
            **init_kwargs,
        )

        query = "mexican food" if is_vectorize else "[2, 1]"
        hits = g_store.similarity_search(query=query, k=2)
        assert len(hits) == 1
        assert hits[0].page_content == page_contents[0]
        assert hits[0].id == "x_id"
        assert hits[0].metadata["md"] == 1

    @pytest.mark.parametrize(
        ("is_vectorize", "page_contents", "collection_fixture_name"),
        [
            (False, ["[1, 2]"], "empty_collection_d2"),
            (True, ["tacos, tamales, and mole"], "empty_collection_vz"),
        ],
        ids=["nonvectorize_store", "vectorize_store"],
    )
    async def test_gvs_from_documents_containing_ids_async(
        self,
        *,
        auth_kwargs: dict[str, Any],
        openai_api_key: str,
        embedding_d2: Embeddings,
        is_vectorize: bool,
        page_contents: list[str],
        collection_fixture_name: str,
        request: pytest.FixtureRequest,
    ) -> None:
        collection: Collection = request.getfixturevalue(collection_fixture_name)
        init_kwargs: dict[str, Any]
        if is_vectorize:
            init_kwargs = {
                "collection_vector_service_options": OPENAI_VECTORIZE_OPTIONS_HEADER,
                "collection_embedding_api_key": openai_api_key,
            }
        else:
            init_kwargs = {"embedding": embedding_d2}

        content_field = CUSTOM_CONTENT_KEY if not is_vectorize else None

        the_document = Document(
            page_content=page_contents[0],
            metadata={"md": 1},
            id="x_id",
        )
        g_store = await AstraDBGraphVectorStore.afrom_documents(
            documents=[the_document],
            collection_name=collection.name,
            content_field=content_field,
            **auth_kwargs,
            **init_kwargs,
        )

        query = "mexican food" if is_vectorize else "[2, 1]"
        hits = g_store.similarity_search(query=query, k=2)
        assert len(hits) == 1
        assert hits[0].page_content == page_contents[0]
        assert hits[0].id == "x_id"
        assert hits[0].metadata["md"] == 1

    @pytest.mark.parametrize(
        ("is_vectorize", "page_contents", "store_name"),
        [
            (False, ["[0, 2]", "[0, 1]"], "graph_vector_store_d2"),
            (True, ["lasagna", "hamburger"], "graph_vector_store_vz"),
        ],
        ids=["nonvectorize_store", "vectorize_store"],
    )
    def test_gvs_add_nodes_sync(
        self,
        *,
        is_vectorize: bool,
        page_contents: list[str],
        store_name: str,
        request: pytest.FixtureRequest,
    ) -> None:
        g_store: AstraDBGraphVectorStore = request.getfixturevalue(store_name)
        links0 = [
            Link(kind="kA", direction="out", tag="tA"),
            Link(kind="kB", direction="bidir", tag="tB"),
        ]
        links1 = [
            Link(kind="kC", direction="in", tag="tC"),
        ]
        nodes = [
            Node(id="id0", text=page_contents[0], metadata={"m": 0}, links=links0),
            Node(text=page_contents[1], metadata={"m": 1}, links=links1),
        ]
        g_store.add_nodes(nodes)

        query = "italian food" if is_vectorize else "[0, 3]"
        hits = g_store.similarity_search(query=query)
        assert len(hits) == 2
        assert hits[0].id == "id0"
        md0 = hits[0].metadata
        assert md0["m"] == 0
        assert any(isinstance(v, set) for k, v in md0.items() if k != "m")
        assert hits[1].id != "id0"
        md1 = hits[1].metadata
        assert md1["m"] == 1
        assert any(isinstance(v, set) for k, v in md1.items() if k != "m")

    @pytest.mark.parametrize(
        ("is_vectorize", "page_contents", "store_name"),
        [
            (False, ["[0, 2]", "[0, 1]"], "graph_vector_store_d2"),
            (True, ["lasagna", "hamburger"], "graph_vector_store_vz"),
        ],
        ids=["nonvectorize_store", "vectorize_store"],
    )
    async def test_gvs_add_nodes_async(
        self,
        *,
        is_vectorize: bool,
        page_contents: list[str],
        store_name: str,
        request: pytest.FixtureRequest,
    ) -> None:
        g_store: AstraDBGraphVectorStore = request.getfixturevalue(store_name)
        links0 = [
            Link(kind="kA", direction="out", tag="tA"),
            Link(kind="kB", direction="bidir", tag="tB"),
        ]
        links1 = [
            Link(kind="kC", direction="in", tag="tC"),
        ]
        nodes = [
            Node(id="id0", text=page_contents[0], metadata={"m": 0}, links=links0),
            Node(text=page_contents[1], metadata={"m": 1}, links=links1),
        ]
        async for _ in g_store.aadd_nodes(nodes):
            pass

        query = "italian food" if is_vectorize else "[0, 3]"
        hits = await g_store.asimilarity_search(query=query)
        assert len(hits) == 2
        assert hits[0].id == "id0"
        md0 = hits[0].metadata
        assert md0["m"] == 0
        assert any(isinstance(v, set) for k, v in md0.items() if k != "m")
        assert hits[1].id != "id0"
        md1 = hits[1].metadata
        assert md1["m"] == 1
        assert any(isinstance(v, set) for k, v in md1.items() if k != "m")
