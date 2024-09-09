"""Test of Astra DB vector store class `AstraDBVectorStore`

Refer to `test_vectorstores.py` for the requirements to run.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Iterable

import pytest
from astrapy import DataAPIClient
from astrapy.authentication import StaticTokenProvider
from langchain_core.documents import Document

from langchain_astradb.utils.astradb import SetupMode
from langchain_astradb.vectorstores import AstraDBVectorStore
from tests.conftest import SomeEmbeddings

from .conftest import OPENAI_VECTORIZE_OPTIONS, AstraDBCredentials, _has_env_vars

if TYPE_CHECKING:
    from astrapy import Collection
    from langchain_core.embeddings import Embeddings

# Faster testing (no actual collection deletions). Off by default (=full tests)
SKIP_COLLECTION_DELETE = (
    int(os.environ.get("ASTRA_DB_SKIP_COLLECTION_DELETIONS", "0")) != 0
)

AD_NOVECTORIZE_COLLECTION = "lc_ad_novectorize"
AD_VECTORIZE_COLLECTION = "lc_ad_vectorize"


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
        AD_NOVECTORIZE_COLLECTION,
        dimension=2,
        check_exists=False,
        metric="cosine",
    )
    yield collection

    if not SKIP_COLLECTION_DELETE:
        collection.drop()


@pytest.fixture(scope="session")
def provisioned_vectorize_collection(
    astra_db_credentials: AstraDBCredentials,
) -> Iterable[Collection]:
    """Provision a general-purpose collection for the vectorize tests."""
    client = DataAPIClient(environment=astra_db_credentials["environment"])
    database = client.get_database(
        astra_db_credentials["api_endpoint"],
        token=StaticTokenProvider(astra_db_credentials["token"]),
        namespace=astra_db_credentials["namespace"],
    )
    collection = database.create_collection(
        AD_VECTORIZE_COLLECTION,
        dimension=2,
        check_exists=False,
        metric="cosine",
        service=OPENAI_VECTORIZE_OPTIONS,
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
def vectorize_collection(
    provisioned_vectorize_collection: Collection,
) -> Iterable[Collection]:
    provisioned_vectorize_collection.delete_many({})
    yield provisioned_vectorize_collection

    provisioned_vectorize_collection.delete_many({})


@pytest.fixture(scope="session")
def embedding() -> Embeddings:
    return SomeEmbeddings(dimension=2)


@pytest.fixture
def novectorize_store(
    novectorize_collection: Collection,  # noqa: ARG001
    astra_db_credentials: AstraDBCredentials,
    embedding: Embeddings,
) -> AstraDBVectorStore:
    return AstraDBVectorStore(
        embedding=embedding,
        collection_name=AD_NOVECTORIZE_COLLECTION,
        token=StaticTokenProvider(astra_db_credentials["token"]),
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
        setup_mode=SetupMode.OFF,
    )


@pytest.fixture
def vectorize_store(
    vectorize_collection: Collection,  # noqa: ARG001
    astra_db_credentials: AstraDBCredentials,
) -> AstraDBVectorStore:
    return AstraDBVectorStore(
        collection_name=AD_VECTORIZE_COLLECTION,
        token=StaticTokenProvider(astra_db_credentials["token"]),
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
        setup_mode=SetupMode.OFF,
        collection_vector_service_options=OPENAI_VECTORIZE_OPTIONS,
    )


@pytest.mark.skipif(not _has_env_vars(), reason="Missing Astra DB env. vars")
class TestVectorStoreAutodetect:
    def test_autodetect_flat_novectorize_crud(
        self,
        novectorize_collection: Collection,
        astra_db_credentials: AstraDBCredentials,
        embedding: Embeddings,
    ) -> None:
        """Test autodetect on a populated flat collection, checking all codecs."""
        novectorize_collection.insert_many(
            [
                {
                    "_id": "1",
                    "$vector": [0.1, 0.2],
                    "cont": "Cont1",
                    "m1": "a",
                    "m2": "x",
                },
                {
                    "_id": "2",
                    "$vector": [0.3, 0.4],
                    "cont": "Cont2",
                    "m1": "b",
                    "m2": "y",
                },
                {
                    "_id": "3",
                    "$vector": [0.5, 0.6],
                    "cont": "Cont3",
                    "m1": "c",
                    "m2": "z",
                },
            ]
        )
        ad_store = AstraDBVectorStore(
            embedding=embedding,
            collection_name=AD_NOVECTORIZE_COLLECTION,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            autodetect_collection=True,
        )

        # ANN and the metadata
        results = ad_store.similarity_search("query", k=3)
        assert {res.page_content for res in results} == {"Cont1", "Cont2", "Cont3"}
        assert "m1" in results[0].metadata
        assert "m2" in results[0].metadata

        # inserting
        pc4 = "Cont4"
        md4 = {"q1": "Q1", "q2": "Q2"}
        inserted_ids = ad_store.add_texts(
            texts=[pc4],
            metadatas=[md4],
            ids=["4"],
        )
        assert inserted_ids == ["4"]

        # reading with filtering
        results2 = ad_store.similarity_search("query", k=3, filter={"q2": "Q2"})
        assert results2 == [Document(page_content=pc4, metadata=md4)]

    def test_autodetect_default_novectorize_crud(
        self,
        novectorize_collection: Collection,  # noqa: ARG002
        astra_db_credentials: AstraDBCredentials,
        embedding: Embeddings,
        novectorize_store: AstraDBVectorStore,
    ) -> None:
        """Test autodetect on a VS-made collection, checking all codecs."""
        novectorize_store.add_texts(
            texts=[
                "Cont1",
                "Cont2",
                "Cont3",
            ],
            metadatas=[
                {"m1": "a", "m2": "x"},
                {"m1": "b", "m2": "y"},
                {"m1": "c", "m2": "z"},
            ],
            ids=[
                "1",
                "2",
                "3",
            ],
        )
        # now with the autodetect
        ad_store = AstraDBVectorStore(
            embedding=embedding,
            collection_name=AD_NOVECTORIZE_COLLECTION,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            autodetect_collection=True,
        )

        # ANN and the metadata
        results = ad_store.similarity_search("query", k=3)
        assert {res.page_content for res in results} == {"Cont1", "Cont2", "Cont3"}
        assert "m1" in results[0].metadata
        assert "m2" in results[0].metadata

        # inserting
        pc4 = "Cont4"
        md4 = {"q1": "Q1", "q2": "Q2"}
        inserted_ids = ad_store.add_texts(
            texts=[pc4],
            metadatas=[md4],
            ids=["4"],
        )
        assert inserted_ids == ["4"]

        # reading with filtering
        results2 = ad_store.similarity_search("query", k=3, filter={"q2": "Q2"})
        assert results2 == [Document(page_content=pc4, metadata=md4)]

    def test_autodetect_flat_vectorize_crud(
        self,
        vectorize_collection: Collection,
        astra_db_credentials: AstraDBCredentials,
    ) -> None:
        """Test autodetect on a populated flat collection, checking all codecs."""
        vectorize_collection.insert_many(
            [
                {
                    "_id": "1",
                    "$vectorize": "Cont1",
                    "m1": "a",
                    "m2": "x",
                },
                {
                    "_id": "2",
                    "$vectorize": "Cont2",
                    "m1": "b",
                    "m2": "y",
                },
                {
                    "_id": "3",
                    "$vectorize": "Cont3",
                    "m1": "c",
                    "m2": "z",
                },
            ]
        )
        ad_store = AstraDBVectorStore(
            collection_name=AD_VECTORIZE_COLLECTION,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            autodetect_collection=True,
        )

        # ANN and the metadata
        results = ad_store.similarity_search("query", k=3)
        assert {res.page_content for res in results} == {"Cont1", "Cont2", "Cont3"}
        assert "m1" in results[0].metadata
        assert "m2" in results[0].metadata

        # inserting
        pc4 = "Cont4"
        md4 = {"q1": "Q1", "q2": "Q2"}
        inserted_ids = ad_store.add_texts(
            texts=[pc4],
            metadatas=[md4],
            ids=["4"],
        )
        assert inserted_ids == ["4"]

        # reading with filtering
        results2 = ad_store.similarity_search("query", k=3, filter={"q2": "Q2"})
        assert results2 == [Document(page_content=pc4, metadata=md4)]

    def test_autodetect_default_vectorize_crud(
        self,
        vectorize_collection: Collection,  # noqa: ARG002
        astra_db_credentials: AstraDBCredentials,
        vectorize_store: AstraDBVectorStore,
    ) -> None:
        """Test autodetect on a VS-made collection, checking all codecs."""
        vectorize_store.add_texts(
            texts=[
                "Cont1",
                "Cont2",
                "Cont3",
            ],
            metadatas=[
                {"m1": "a", "m2": "x"},
                {"m1": "b", "m2": "y"},
                {"m1": "c", "m2": "z"},
            ],
            ids=[
                "1",
                "2",
                "3",
            ],
        )
        # now with the autodetect
        ad_store = AstraDBVectorStore(
            collection_name=AD_VECTORIZE_COLLECTION,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            autodetect_collection=True,
        )

        # ANN and the metadata
        results = ad_store.similarity_search("query", k=3)
        assert {res.page_content for res in results} == {"Cont1", "Cont2", "Cont3"}
        assert "m1" in results[0].metadata
        assert "m2" in results[0].metadata

        # inserting
        pc4 = "Cont4"
        md4 = {"q1": "Q1", "q2": "Q2"}
        inserted_ids = ad_store.add_texts(
            texts=[pc4],
            metadatas=[md4],
            ids=["4"],
        )
        assert inserted_ids == ["4"]

        # reading with filtering
        results2 = ad_store.similarity_search("query", k=3, filter={"q2": "Q2"})
        assert results2 == [Document(page_content=pc4, metadata=md4)]

    def test_failed_docs_autodetect_flat_novectorize_crud(
        self,
        novectorize_collection: Collection,
        astra_db_credentials: AstraDBCredentials,
        embedding: Embeddings,
    ) -> None:
        """Test autodetect + skipping failing documents."""
        novectorize_collection.insert_many(
            [
                {
                    "_id": "1",
                    "$vector": [0.1, 0.2],
                    "cont": "Cont1",
                    "m1": "a",
                    "m2": "x",
                },
            ]
        )
        ad_store_e = AstraDBVectorStore(
            collection_name=AD_NOVECTORIZE_COLLECTION,
            embedding=embedding,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            autodetect_collection=True,
            ignore_invalid_documents=False,
        )
        ad_store_w = AstraDBVectorStore(
            collection_name=AD_NOVECTORIZE_COLLECTION,
            embedding=embedding,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            autodetect_collection=True,
            ignore_invalid_documents=True,
        )

        results_e = ad_store_e.similarity_search("query", k=3)
        assert len(results_e) == 1

        results_w = ad_store_w.similarity_search("query", k=3)
        assert len(results_w) == 1

        novectorize_collection.insert_one(
            {
                "_id": "2",
                "$vector": [0.1, 0.2],
                "m1": "invalid:",
                "m2": "no $vector!",
            }
        )

        with pytest.raises(KeyError):
            ad_store_e.similarity_search("query", k=3)

        results_w_post = ad_store_w.similarity_search("query", k=3)
        assert len(results_w_post) == 1
