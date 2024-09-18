"""Test of Astra DB graph vector store class `AstraDBGraphVectorStore`

Refer to `test_vectorstores.py` for the requirements to run.
"""

# ruff: noqa: FIX002 TD002 TD003

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Iterable

import pytest
from astrapy import DataAPIClient
from astrapy.authentication import StaticTokenProvider
from langchain_core.documents import Document
from langchain_core.graph_vectorstores.base import Node
from langchain_core.graph_vectorstores.links import Link, add_links

from langchain_astradb.graph_vectorstores import AstraDBGraphVectorStore
from langchain_astradb.utils.astradb import SetupMode
from tests.conftest import ParserEmbeddings

from .conftest import AstraDBCredentials, _has_env_vars

if TYPE_CHECKING:
    from astrapy import Collection
    from langchain_core.embeddings import Embeddings

# Faster testing (no actual collection deletions). Off by default (=full tests)
SKIP_COLLECTION_DELETE = (
    int(os.environ.get("ASTRA_DB_SKIP_COLLECTION_DELETIONS", "0")) != 0
)

GVS_NOVECTORIZE_COLLECTION = "lc_gvs_novectorize"
# for testing with autodetect
CUSTOM_CONTENT_KEY = "xcontent"
LONG_TEXT = "This is the textual content field in the doc."


@pytest.fixture(scope="session")
def provisioned_novectorize_collection(
    astra_db_credentials: AstraDBCredentials,
) -> Iterable[Collection]:
    """Provision a general-purpose collection for the no-vectorize tests."""
    client = DataAPIClient(environment=astra_db_credentials["environment"])
    database = client.get_database(
        astra_db_credentials["api_endpoint"],
        token=StaticTokenProvider(astra_db_credentials["token"]),
        namespace=astra_db_credentials["namespace"],
    )
    collection = database.create_collection(
        GVS_NOVECTORIZE_COLLECTION,
        dimension=2,
        check_exists=False,
        metric="euclidean",
    )
    yield collection

    if not SKIP_COLLECTION_DELETE:
        collection.drop()


@pytest.fixture
def novectorize_empty_collection(
    provisioned_novectorize_collection: Collection,
) -> Iterable[Collection]:
    provisioned_novectorize_collection.delete_many({})
    yield provisioned_novectorize_collection

    provisioned_novectorize_collection.delete_many({})


@pytest.fixture
def embedding() -> Embeddings:
    return ParserEmbeddings(dimension=2)


@pytest.fixture
def novectorize_empty_graph_store(
    novectorize_empty_collection: Collection,  # noqa: ARG001
    astra_db_credentials: AstraDBCredentials,
    embedding: Embeddings,
) -> AstraDBGraphVectorStore:
    return AstraDBGraphVectorStore(
        embedding=embedding,
        collection_name=GVS_NOVECTORIZE_COLLECTION,
        token=StaticTokenProvider(astra_db_credentials["token"]),
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
        setup_mode=SetupMode.OFF,
    )


@pytest.fixture
def novectorize_autodetect_full_graph_store(
    astra_db_credentials: AstraDBCredentials,
    novectorize_empty_collection: Collection,
    embedding: Embeddings,
    graph_docs: list[Document],
) -> AstraDBGraphVectorStore:
    """
    Pre-populate the collection and have (VectorStore)autodetect work on it,
    then create and return a GraphVectorStore, additionally filled with
    the same (graph-)entries as for `novectorize_full_graph_store`.
    """
    novectorize_empty_collection.insert_many(
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
        embedding=embedding,
        collection_name=GVS_NOVECTORIZE_COLLECTION,
        link_to_metadata_key="x_link_to_x",
        link_from_metadata_key="x_link_from_x",
        token=StaticTokenProvider(astra_db_credentials["token"]),
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
        content_field="*",
        autodetect_collection=True,
    )
    gstore.add_documents(graph_docs)
    return gstore


def assert_all_flat_docs(collection: Collection) -> None:
    """
    Check that after graph-insertions all docs in the store
    still obey the underlying autodetected doc schema on DB.
    """
    for doc in collection.find({}, projection={"*": True}):
        assert all(not isinstance(v, dict) for v in doc.values())
        assert CUSTOM_CONTENT_KEY in doc
        assert isinstance(doc["$vector"], list)


@pytest.fixture
def graph_docs() -> list[Document]:
    """
    This is a pre-populated graph vector store,
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

    the query point is at (*).
    the A are bidirectionally with B
    the A are outgoing to T
    the A are incoming from F
    The links are like: L with L, 0 with 0 and R with R.
    """

    docs_a = [
        Document(page_content="[-1, 9]", metadata={"label": "AL"}),
        Document(page_content="[0, 10]", metadata={"label": "A0"}),
        Document(page_content="[1, 9]", metadata={"label": "AR"}),
    ]
    docs_b = [
        Document(page_content="[9, 1]", metadata={"label": "BL"}),
        Document(page_content="[10, 0]", metadata={"label": "B0"}),
        Document(page_content="[9, -1]", metadata={"label": "BR"}),
    ]
    docs_f = [
        Document(page_content="[1, -9]", metadata={"label": "BL"}),
        Document(page_content="[0, -10]", metadata={"label": "B0"}),
        Document(page_content="[-1, -9]", metadata={"label": "BR"}),
    ]
    docs_t = [
        Document(page_content="[-9, -1]", metadata={"label": "TL"}),
        Document(page_content="[-10, 0]", metadata={"label": "T0"}),
        Document(page_content="[-9, 1]", metadata={"label": "TR"}),
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
def novectorize_full_graph_store(
    novectorize_empty_graph_store: AstraDBGraphVectorStore,
    graph_docs: list[Document],
) -> AstraDBGraphVectorStore:
    novectorize_empty_graph_store.add_documents(graph_docs)
    return novectorize_empty_graph_store


@pytest.mark.skipif(not _has_env_vars(), reason="Missing Astra DB env. vars")
class TestAstraDBGraphVectorStore:
    @pytest.mark.parametrize(
        ("store_name", "is_autodetected"),
        [
            ("novectorize_full_graph_store", False),
            ("novectorize_autodetect_full_graph_store", True),
        ],
        ids=["native_store", "autodetected_store"],
    )
    def test_gvs_similarity_search(
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
            assert_all_flat_docs(store.vectorstore.astra_env.collection)

    @pytest.mark.parametrize(
        ("store_name", "is_autodetected"),
        [
            ("novectorize_full_graph_store", False),
            ("novectorize_autodetect_full_graph_store", True),
        ],
        ids=["native_store", "autodetected_store"],
    )
    def test_gvs_traversal_search(
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
            assert_all_flat_docs(store.vectorstore.astra_env.collection)

    @pytest.mark.parametrize(
        ("store_name", "is_autodetected"),
        [
            ("novectorize_full_graph_store", False),
            ("novectorize_autodetect_full_graph_store", True),
        ],
        ids=["native_store", "autodetected_store"],
    )
    def test_gvs_mmr_traversal_search(
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
            assert_all_flat_docs(store.vectorstore.astra_env.collection)

    def test_gvs_from_texts(
        self,
        *,
        astra_db_credentials: AstraDBCredentials,
        novectorize_empty_collection: Collection,  # noqa: ARG002
        embedding: Embeddings,
    ) -> None:
        g_store = AstraDBGraphVectorStore.from_texts(
            texts=["[1, 2]"],
            embedding=embedding,
            metadatas=[{"md": 1}],
            ids=["x_id"],
            collection_name=GVS_NOVECTORIZE_COLLECTION,
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

    def test_gvs_from_documents_separate_ids(
        self,
        *,
        astra_db_credentials: AstraDBCredentials,
        novectorize_empty_collection: Collection,  # noqa: ARG002
        embedding: Embeddings,
    ) -> None:
        the_document = Document(
            page_content="[1, 2]",
            metadata={"md": 1},
        )
        g_store = AstraDBGraphVectorStore.from_documents(
            documents=[the_document],
            embedding=embedding,
            ids=["x_id"],
            collection_name=GVS_NOVECTORIZE_COLLECTION,
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
        novectorize_empty_collection: Collection,  # noqa: ARG002
        embedding: Embeddings,
    ) -> None:
        the_document = Document(
            page_content="[1, 2]",
            metadata={"md": 1},
            id="x_id",
        )
        g_store = AstraDBGraphVectorStore.from_documents(
            documents=[the_document],
            embedding=embedding,
            collection_name=GVS_NOVECTORIZE_COLLECTION,
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

    def test_gvs_from_documents_pass_ids_twice(
        self,
        *,
        astra_db_credentials: AstraDBCredentials,
        novectorize_empty_collection: Collection,  # noqa: ARG002
        embedding: Embeddings,
    ) -> None:
        """The explicitly-provided `ids` parameter should take precedence."""
        the_document = Document(
            page_content="[1, 2]",
            metadata={"md": 1},
            id="y_id",
        )
        g_store = AstraDBGraphVectorStore.from_documents(
            documents=[the_document],
            embedding=embedding,
            ids=["x_id"],
            collection_name=GVS_NOVECTORIZE_COLLECTION,
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

    def test_gvs_add_nodes(
        self,
        *,
        novectorize_empty_graph_store: AstraDBGraphVectorStore,
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
        novectorize_empty_graph_store.add_nodes(nodes)
        hits = novectorize_empty_graph_store.similarity_search_by_vector([0, 3])
        assert len(hits) == 2
        assert hits[0].id == "id0"
        assert hits[0].page_content == "[0, 2]"
        md0 = hits[0].metadata
        assert md0["m"] == 0
        assert any(isinstance(v, list) for k, v in md0.items() if k != "m")
        assert hits[1].id != "id0"
        assert hits[1].page_content == "[0, 1]"
        md1 = hits[1].metadata
        assert md1["m"] == 1
        assert any(isinstance(v, list) for k, v in md1.items() if k != "m")
