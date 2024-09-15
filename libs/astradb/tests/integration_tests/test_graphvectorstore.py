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
from langchain_core.graph_vectorstores import Link
from langchain_core.graph_vectorstores.links import add_links

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
def novectorize_collection(
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
    novectorize_collection: Collection,  # noqa: ARG001
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
def novectorize_full_graph_store(
        novectorize_empty_graph_store: AstraDBGraphVectorStore,
) -> AstraDBGraphVectorStore:
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
    for doc_a, suffix in zip(docs_a, ["l","0","r"]):
        add_links(doc_a, Link.bidir(kind="ab_example", tag=f"tag_{suffix}"))
        add_links(doc_a, Link.outgoing(kind="at_example", tag=f"tag_{suffix}"))
        add_links(doc_a, Link.incoming(kind="af_example", tag=f"tag_{suffix}"))
    for doc_b, suffix in zip(docs_b, ["l","0","r"]):
        add_links(doc_b, Link.bidir(kind="ab_example", tag=f"tag_{suffix}"))
    for doc_t, suffix in zip(docs_t, ["l","0","r"]):
        add_links(doc_t, Link.incoming(kind="at_example", tag=f"tag_{suffix}"))
    for doc_f, suffix in zip(docs_f, ["l","0","r"]):
        add_links(doc_f, Link.outgoing(kind="af_example", tag=f"tag_{suffix}"))
    novectorize_empty_graph_store.add_documents(
        docs_a + docs_b + docs_f + docs_t
    )
    return novectorize_empty_graph_store


@pytest.mark.skipif(not _has_env_vars(), reason="Missing Astra DB env. vars")
class TestAstraDBGraphVectorStore:
    def test_similarity_search(
        self, novectorize_full_graph_store: AstraDBGraphVectorStore
    ) -> None:
        """Simple (non-graph) similarity search on a graph vector store."""
        store = novectorize_full_graph_store
        ss_response = store.similarity_search(query="[2, 10]", k=2)
        ss_labels = [doc.metadata["label"] for doc in ss_response]
        assert ss_labels == ["AR", "A0"]

    def test_traversal_search(
        self, novectorize_full_graph_store: AstraDBGraphVectorStore
    ) -> None:
        """Graph traversal search on a graph vector store."""
        store = novectorize_full_graph_store
        ts_response = store.traversal_search(query="[2, 10]", k=2, depth=2)
        # this is a set, as some of the internals of trav.search are set-driven
        # so ordering is not deterministic:
        ts_labels = {doc.metadata["label"] for doc in ts_response}
        assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}

    def test_mmr_traversal_search(
        self, novectorize_full_graph_store: AstraDBGraphVectorStore
    ) -> None:
        """MMR Graph traversal search on a graph vector store."""
        store = novectorize_full_graph_store
        mt_response = store.mmr_traversal_search(
            query="[2, 10]",
            k=2,
            depth=2,
            fetch_k=1,
            adjacent_k=2,
            lambda_mult=0.1,
        )
        # TODO: can this rightfully be a list (or must it be a set?)
        mt_labels = {doc.metadata["label"] for doc in mt_response}
        assert mt_labels == {"AR", "BR"}
