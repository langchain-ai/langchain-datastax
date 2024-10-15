"""Test of Astra DB graph vector store class `AstraDBGraphVectorStore`

Refer to `test_vectorstores.py` for the requirements to run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    the A are bidirectionally with B
    the A are outgoing to T
    the A are incoming from F
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
def graph_vector_store_d2(
    astra_db_credentials: AstraDBCredentials,
    empty_collection_d2: Collection,
    embedding_d2: Embeddings,
) -> AstraDBGraphVectorStore:
    return AstraDBGraphVectorStore(
        embedding=embedding_d2,
        collection_name=empty_collection_d2.name,
        token=StaticTokenProvider(astra_db_credentials["token"]),
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
        setup_mode=SetupMode.OFF,
    )


@pytest.fixture
def populated_graph_vector_store_d2(
    graph_vector_store_d2: AstraDBGraphVectorStore,
    graph_vector_store_docs: list[Document],
) -> AstraDBGraphVectorStore:
    graph_vector_store_d2.add_documents(graph_vector_store_docs)
    return graph_vector_store_d2


@pytest.fixture
def autodetect_populated_graph_vector_store_d2(
    astra_db_credentials: AstraDBCredentials,
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
    gstore = AstraDBGraphVectorStore(
        embedding=embedding_d2,
        collection_name=ephemeral_collection_cleaner_idxall_d2,
        metadata_incoming_links_key="x_link_to_x",
        token=StaticTokenProvider(astra_db_credentials["token"]),
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
        content_field="*",
        autodetect_collection=True,
    )
    gstore.add_documents(graph_vector_store_docs)
    return gstore


def assert_all_flat_docs(collection: Collection) -> None:
    """
    Check that all docs in the store obey the underlying (flat) autodetected
    doc schema on DB.
    Useful for checking the store after graph-store-driven insertions.
    """
    for doc in collection.find({}, projection={"*": True}):
        assert all(not isinstance(v, dict) for v in doc.values())
        assert CUSTOM_CONTENT_KEY in doc
        assert isinstance(doc["$vector"], list)


@pytest.mark.skipif(
    not astra_db_env_vars_available(), reason="Missing Astra DB env. vars"
)
class TestAstraDBGraphVectorStore:
    @pytest.mark.parametrize(
        ("store_name", "is_autodetected"),
        [
            ("populated_graph_vector_store_d2", False),
            ("autodetect_populated_graph_vector_store_d2", True),
        ],
        ids=["native_store", "autodetected_store"],
    )
    def test_gvs_similarity_search_sync(
        self,
        *,
        store_name: str,
        is_autodetected: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """Simple (non-graph) similarity search on a graph vector store."""
        store: AstraDBGraphVectorStore = request.getfixturevalue(store_name)
        ss_response = store.similarity_search(query="[2, 10]", k=2)
        ss_labels = [doc.metadata["label"] for doc in ss_response]
        assert ss_labels == ["AR", "A0"]
        ss_by_v_response = store.similarity_search_by_vector(embedding=[2, 10], k=2)
        ss_by_v_labels = [doc.metadata["label"] for doc in ss_by_v_response]
        assert ss_by_v_labels == ["AR", "A0"]
        if is_autodetected:
            assert_all_flat_docs(store.vector_store.astra_env.collection)

    @pytest.mark.parametrize(
        ("store_name", "is_autodetected"),
        [
            ("populated_graph_vector_store_d2", False),
            ("autodetect_populated_graph_vector_store_d2", True),
        ],
        ids=["native_store", "autodetected_store"],
    )
    async def test_gvs_similarity_search_async(
        self,
        *,
        store_name: str,
        is_autodetected: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """Simple (non-graph) similarity search on a graph vector store."""
        store: AstraDBGraphVectorStore = request.getfixturevalue(store_name)
        ss_response = await store.asimilarity_search(query="[2, 10]", k=2)
        ss_labels = [doc.metadata["label"] for doc in ss_response]
        assert ss_labels == ["AR", "A0"]
        ss_by_v_response = await store.asimilarity_search_by_vector(
            embedding=[2, 10], k=2
        )
        ss_by_v_labels = [doc.metadata["label"] for doc in ss_by_v_response]
        assert ss_by_v_labels == ["AR", "A0"]
        if is_autodetected:
            assert_all_flat_docs(store.vector_store.astra_env.collection)

    @pytest.mark.parametrize(
        ("store_name", "is_autodetected"),
        [
            ("populated_graph_vector_store_d2", False),
            ("autodetect_populated_graph_vector_store_d2", True),
        ],
        ids=["native_store", "autodetected_store"],
    )
    def test_gvs_traversal_search_sync(
        self,
        *,
        store_name: str,
        is_autodetected: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """Graph traversal search on a graph vector store."""
        store: AstraDBGraphVectorStore = request.getfixturevalue(store_name)
        ts_response = store.traversal_search(query="[2, 10]", k=2, depth=2)
        # this is a set, as some of the internals of trav.search are set-driven
        # so ordering is not deterministic:
        ts_labels = {doc.metadata["label"] for doc in ts_response}
        assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}
        if is_autodetected:
            assert_all_flat_docs(store.vector_store.astra_env.collection)

    @pytest.mark.parametrize(
        ("store_name", "is_autodetected"),
        [
            ("populated_graph_vector_store_d2", False),
            ("autodetect_populated_graph_vector_store_d2", True),
        ],
        ids=["native_store", "autodetected_store"],
    )
    async def test_gvs_traversal_search_async(
        self,
        *,
        store_name: str,
        is_autodetected: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """Graph traversal search on a graph vector store."""
        store: AstraDBGraphVectorStore = request.getfixturevalue(store_name)
        ts_labels = set()
        async for doc in store.atraversal_search(query="[2, 10]", k=2, depth=2):
            ts_labels.add(doc.metadata["label"])
        # this is a set, as some of the internals of trav.search are set-driven
        # so ordering is not deterministic:
        assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}
        if is_autodetected:
            assert_all_flat_docs(store.vector_store.astra_env.collection)

    @pytest.mark.parametrize(
        ("store_name", "is_autodetected"),
        [
            ("populated_graph_vector_store_d2", False),
            ("autodetect_populated_graph_vector_store_d2", True),
        ],
        ids=["native_store", "autodetected_store"],
    )
    def test_gvs_mmr_traversal_search_sync(
        self,
        *,
        store_name: str,
        is_autodetected: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """MMR Graph traversal search on a graph vector store."""
        store: AstraDBGraphVectorStore = request.getfixturevalue(store_name)
        mt_response = store.mmr_traversal_search(
            query="[2, 10]",
            k=2,
            depth=2,
            fetch_k=1,
            adjacent_k=2,
            lambda_mult=0.1,
        )
        # TODO: can this rightfully be a list (or must it be a set)?
        mt_labels = {doc.metadata["label"] for doc in mt_response}
        assert mt_labels == {"AR", "BR"}
        if is_autodetected:
            assert_all_flat_docs(store.vector_store.astra_env.collection)

    @pytest.mark.parametrize(
        ("store_name", "is_autodetected"),
        [
            ("populated_graph_vector_store_d2", False),
            ("autodetect_populated_graph_vector_store_d2", True),
        ],
        ids=["native_store", "autodetected_store"],
    )
    async def test_gvs_mmr_traversal_search_async(
        self,
        *,
        store_name: str,
        is_autodetected: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """MMR Graph traversal search on a graph vector store."""
        store: AstraDBGraphVectorStore = request.getfixturevalue(store_name)
        mt_labels = set()
        async for doc in store.ammr_traversal_search(
            query="[2, 10]",
            k=2,
            depth=2,
            fetch_k=1,
            adjacent_k=2,
            lambda_mult=0.1,
        ):
            mt_labels.add(doc.metadata["label"])
        # TODO: can this rightfully be a list (or must it be a set)?
        assert mt_labels == {"AR", "BR"}
        if is_autodetected:
            assert_all_flat_docs(store.vector_store.astra_env.collection)

    @pytest.mark.parametrize(
        ("store_name"),
        [
            ("populated_graph_vector_store_d2"),
            ("autodetect_populated_graph_vector_store_d2"),
        ],
        ids=["native_store", "autodetected_store"],
    )
    def test_gvs_metadata_search_sync(
        self,
        *,
        store_name: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """Metadata search on a graph vector store."""
        store: AstraDBGraphVectorStore = request.getfixturevalue(store_name)
        mt_response = store.metadata_search(
            filter={"label": "T0"},
            n=2,
        )
        doc: Document = next(iter(mt_response))
        assert doc.page_content == "[-10, 0]"
        links = doc.metadata["links"]
        assert len(links) == 1
        link: Link = links.pop()
        assert isinstance(link, Link)
        assert link.direction == "in"
        assert link.kind == "at_example"
        assert link.tag == "tag_0"

    @pytest.mark.parametrize(
        ("store_name"),
        [
            ("populated_graph_vector_store_d2"),
            ("autodetect_populated_graph_vector_store_d2"),
        ],
        ids=["native_store", "autodetected_store"],
    )
    async def test_gvs_metadata_search_async(
        self,
        *,
        store_name: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """Metadata search on a graph vector store."""
        store: AstraDBGraphVectorStore = request.getfixturevalue(store_name)
        mt_response = await store.ametadata_search(
            filter={"label": "T0"},
            n=2,
        )
        doc: Document = next(iter(mt_response))
        assert doc.page_content == "[-10, 0]"
        links: set[Link] = doc.metadata["links"]
        assert len(links) == 1
        link: Link = links.pop()
        assert isinstance(link, Link)
        assert link.direction == "in"
        assert link.kind == "at_example"
        assert link.tag == "tag_0"

    @pytest.mark.parametrize(
        ("store_name"),
        [
            ("populated_graph_vector_store_d2"),
            ("autodetect_populated_graph_vector_store_d2"),
        ],
        ids=["native_store", "autodetected_store"],
    )
    def test_gvs_get_by_document_id_sync(
        self,
        *,
        store_name: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """Get by document_id on a graph vector store."""
        store: AstraDBGraphVectorStore = request.getfixturevalue(store_name)
        doc = store.get_by_document_id(document_id="FL")
        assert doc is not None
        assert doc.page_content == "[1, -9]"
        links = doc.metadata["links"]
        assert len(links) == 1
        link: Link = links.pop()
        assert isinstance(link, Link)
        assert link.direction == "out"
        assert link.kind == "af_example"
        assert link.tag == "tag_l"

        invalid_doc = store.get_by_document_id(document_id="invalid")
        assert invalid_doc is None

    @pytest.mark.parametrize(
        ("store_name"),
        [
            ("populated_graph_vector_store_d2"),
            ("autodetect_populated_graph_vector_store_d2"),
        ],
        ids=["native_store", "autodetected_store"],
    )
    async def test_gvs_get_by_document_id_async(
        self,
        *,
        store_name: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """Get by document_id on a graph vector store."""
        store: AstraDBGraphVectorStore = request.getfixturevalue(store_name)
        doc = await store.aget_by_document_id(document_id="FL")
        assert doc is not None
        assert doc.page_content == "[1, -9]"
        links = doc.metadata["links"]
        assert len(links) == 1
        link: Link = links.pop()
        assert isinstance(link, Link)
        assert link.direction == "out"
        assert link.kind == "af_example"
        assert link.tag == "tag_l"

        invalid_doc = await store.aget_by_document_id(document_id="invalid")
        assert invalid_doc is None

    def test_gvs_from_texts(
        self,
        *,
        astra_db_credentials: AstraDBCredentials,
        empty_collection_d2: Collection,
        embedding_d2: Embeddings,
    ) -> None:
        g_store = AstraDBGraphVectorStore.from_texts(
            texts=["[1, 2]"],
            embedding=embedding_d2,
            metadatas=[{"md": 1}],
            ids=["x_id"],
            collection_name=empty_collection_d2.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            content_field=CUSTOM_CONTENT_KEY,
            setup_mode=SetupMode.OFF,
        )
        hits = g_store.similarity_search("[2, 1]", k=2)
        assert len(hits) == 1
        assert hits[0].page_content == "[1, 2]"
        assert hits[0].id == "x_id"
        # there may be more re:graph structure.
        assert hits[0].metadata["md"] == 1

    def test_gvs_from_documents_containing_ids(
        self,
        *,
        astra_db_credentials: AstraDBCredentials,
        empty_collection_d2: Collection,
        embedding_d2: Embeddings,
    ) -> None:
        the_document = Document(
            page_content="[1, 2]",
            metadata={"md": 1},
            id="x_id",
        )
        g_store = AstraDBGraphVectorStore.from_documents(
            documents=[the_document],
            embedding=embedding_d2,
            collection_name=empty_collection_d2.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            content_field=CUSTOM_CONTENT_KEY,
            setup_mode=SetupMode.OFF,
        )
        hits = g_store.similarity_search("[2, 1]", k=2)
        assert len(hits) == 1
        assert hits[0].page_content == "[1, 2]"
        assert hits[0].id == "x_id"
        # there may be more re:graph structure.
        assert hits[0].metadata["md"] == 1

    def test_gvs_add_nodes_sync(
        self,
        *,
        graph_vector_store_d2: AstraDBGraphVectorStore,
    ) -> None:
        links0 = [
            Link(kind="kA", direction="out", tag="tA"),
            Link(kind="kB", direction="bidir", tag="tB"),
        ]
        links1 = [
            Link(kind="kC", direction="in", tag="tC"),
        ]
        nodes = [
            Node(id="id0", text="[0, 2]", metadata={"m": 0}, links=links0),
            Node(text="[0, 1]", metadata={"m": 1}, links=links1),
        ]
        graph_vector_store_d2.add_nodes(nodes)
        hits = graph_vector_store_d2.similarity_search_by_vector([0, 3])
        assert len(hits) == 2
        assert hits[0].id == "id0"
        assert hits[0].page_content == "[0, 2]"
        md0 = hits[0].metadata
        assert md0["m"] == 0
        assert any(isinstance(v, set) for k, v in md0.items() if k != "m")
        assert hits[1].id != "id0"
        assert hits[1].page_content == "[0, 1]"
        md1 = hits[1].metadata
        assert md1["m"] == 1
        assert any(isinstance(v, set) for k, v in md1.items() if k != "m")

    async def test_gvs_add_nodes_async(
        self,
        *,
        graph_vector_store_d2: AstraDBGraphVectorStore,
    ) -> None:
        links0 = [
            Link(kind="kA", direction="out", tag="tA"),
            Link(kind="kB", direction="bidir", tag="tB"),
        ]
        links1 = [
            Link(kind="kC", direction="in", tag="tC"),
        ]
        nodes = [
            Node(id="id0", text="[0, 2]", metadata={"m": 0}, links=links0),
            Node(text="[0, 1]", metadata={"m": 1}, links=links1),
        ]
        async for _ in graph_vector_store_d2.aadd_nodes(nodes):
            pass

        hits = await graph_vector_store_d2.asimilarity_search_by_vector([0, 3])
        assert len(hits) == 2
        assert hits[0].id == "id0"
        assert hits[0].page_content == "[0, 2]"
        md0 = hits[0].metadata
        assert md0["m"] == 0
        assert any(isinstance(v, set) for k, v in md0.items() if k != "m")
        assert hits[1].id != "id0"
        assert hits[1].page_content == "[0, 1]"
        md1 = hits[1].metadata
        assert md1["m"] == 1
        assert any(isinstance(v, set) for k, v in md1.items() if k != "m")
