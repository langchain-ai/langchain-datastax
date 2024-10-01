"""Test of Astra DB vector store class `AstraDBVectorStore`

Refer to `test_vectorstores.py` for the requirements to run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from astrapy.authentication import StaticTokenProvider
from langchain_core.documents import Document

from langchain_astradb.vectorstores import AstraDBVectorStore

from .conftest import (
    CUSTOM_CONTENT_KEY,
    astra_db_env_vars_available,
)

if TYPE_CHECKING:
    from astrapy import Collection
    from langchain_core.embeddings import Embeddings

    from .conftest import AstraDBCredentials


@pytest.mark.skipif(
    not astra_db_env_vars_available(), reason="Missing Astra DB env. vars"
)
class TestAstraDBVectorStoreAutodetect:
    def test_autodetect_flat_novectorize_crud(
        self,
        astra_db_credentials: AstraDBCredentials,
        empty_collection_idxall_d2: Collection,
        embedding_d2: Embeddings,
    ) -> None:
        """Test autodetect on a populated flat collection, checking all codecs."""
        empty_collection_idxall_d2.insert_many(
            [
                {
                    "_id": "1",
                    "$vector": [1, 2],
                    CUSTOM_CONTENT_KEY: "[1,2]",
                    "m1": "a",
                    "m2": "x",
                },
                {
                    "_id": "2",
                    "$vector": [3, 4],
                    CUSTOM_CONTENT_KEY: "[3,4]",
                    "m1": "b",
                    "m2": "y",
                },
                {
                    "_id": "3",
                    "$vector": [5, 6],
                    CUSTOM_CONTENT_KEY: "[5,6]",
                    "m1": "c",
                    "m2": "z",
                },
            ]
        )
        ad_store = AstraDBVectorStore(
            embedding=embedding_d2,
            collection_name=empty_collection_idxall_d2.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            autodetect_collection=True,
        )

        # ANN and the metadata
        results = ad_store.similarity_search("[-1,-1]", k=3)
        assert {res.page_content for res in results} == {"[1,2]", "[3,4]", "[5,6]"}
        assert "m1" in results[0].metadata
        assert "m2" in results[0].metadata

        # inserting
        id4 = "4"
        pc4 = "[7,8]"
        md4 = {"q1": "Q1", "q2": "Q2"}
        inserted_ids = ad_store.add_texts(
            texts=[pc4],
            metadatas=[md4],
            ids=[id4],
        )
        assert inserted_ids == [id4]

        # reading with filtering
        results2 = ad_store.similarity_search("[-1,-1]", k=3, filter={"q2": "Q2"})
        assert results2 == [Document(id=id4, page_content=pc4, metadata=md4)]

        # delete by metadata
        del_by_md = ad_store.delete_by_metadata_filter(filter={"q2": "Q2"})
        assert del_by_md is not None
        assert del_by_md == 1
        results2n = ad_store.similarity_search("[-1,-1]", k=3, filter={"q2": "Q2"})
        assert results2n == []

    def test_autodetect_default_novectorize_crud(
        self,
        astra_db_credentials: AstraDBCredentials,
        embedding_d2: Embeddings,
        vector_store_idxall_d2: AstraDBVectorStore,
    ) -> None:
        """Test autodetect on a VS-made collection, checking all codecs."""
        vector_store_idxall_d2.add_texts(
            texts=[
                "[1,2]",
                "[3,4]",
                "[5,6]",
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
            embedding=embedding_d2,
            collection_name=vector_store_idxall_d2.collection_name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            autodetect_collection=True,
        )

        # ANN and the metadata
        results = ad_store.similarity_search("[-1,-1]", k=3)
        assert {res.page_content for res in results} == {"[1,2]", "[3,4]", "[5,6]"}
        assert "m1" in results[0].metadata
        assert "m2" in results[0].metadata

        # inserting
        id4 = "4"
        pc4 = "[7,8]"
        md4 = {"q1": "Q1", "q2": "Q2"}
        inserted_ids = ad_store.add_texts(
            texts=[pc4],
            metadatas=[md4],
            ids=[id4],
        )
        assert inserted_ids == [id4]

        # reading with filtering
        results2 = ad_store.similarity_search("[9,10]", k=3, filter={"q2": "Q2"})
        assert results2 == [Document(id=id4, page_content=pc4, metadata=md4)]

        # delete by metadata
        del_by_md = ad_store.delete_by_metadata_filter(filter={"q2": "Q2"})
        assert del_by_md is not None
        assert del_by_md == 1
        results2n = ad_store.similarity_search("[-1,-1]", k=3, filter={"q2": "Q2"})
        assert results2n == []

    def test_autodetect_flat_vectorize_crud(
        self,
        astra_db_credentials: AstraDBCredentials,
        openai_api_key: str,
        empty_collection_idxall_vz: Collection,
    ) -> None:
        """Test autodetect on a populated flat collection, checking all codecs."""
        empty_collection_idxall_vz.insert_many(
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
            collection_name=empty_collection_idxall_vz.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            autodetect_collection=True,
            collection_embedding_api_key=openai_api_key,
        )

        # ANN and the metadata
        results = ad_store.similarity_search("query", k=3)
        assert {res.page_content for res in results} == {"Cont1", "Cont2", "Cont3"}
        assert "m1" in results[0].metadata
        assert "m2" in results[0].metadata

        # inserting
        id4 = "4"
        pc4 = "Cont4"
        md4 = {"q1": "Q1", "q2": "Q2"}
        inserted_ids = ad_store.add_texts(
            texts=[pc4],
            metadatas=[md4],
            ids=["4"],
        )
        assert inserted_ids == [id4]

        # reading with filtering
        results2 = ad_store.similarity_search("query", k=3, filter={"q2": "Q2"})
        assert results2 == [Document(id=id4, page_content=pc4, metadata=md4)]

        # delete by metadata
        del_by_md = ad_store.delete_by_metadata_filter(filter={"q2": "Q2"})
        assert del_by_md is not None
        assert del_by_md == 1
        results2n = ad_store.similarity_search("[-1,-1]", k=3, filter={"q2": "Q2"})
        assert results2n == []

    def test_autodetect_default_vectorize_crud(
        self,
        *,
        astra_db_credentials: AstraDBCredentials,
        openai_api_key: str,
        empty_collection_idxall_vz: Collection,
        vector_store_idxall_vz: AstraDBVectorStore,
    ) -> None:
        """Test autodetect on a VS-made collection, checking all codecs."""
        vector_store_idxall_vz.add_texts(
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
            collection_name=empty_collection_idxall_vz.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            autodetect_collection=True,
            collection_embedding_api_key=openai_api_key,
        )

        # ANN and the metadata
        results = ad_store.similarity_search("query", k=3)
        assert {res.page_content for res in results} == {"Cont1", "Cont2", "Cont3"}
        assert "m1" in results[0].metadata
        assert "m2" in results[0].metadata

        # inserting
        id4 = "4"
        pc4 = "Cont4"
        md4 = {"q1": "Q1", "q2": "Q2"}
        inserted_ids = ad_store.add_texts(
            texts=[pc4],
            metadatas=[md4],
            ids=[id4],
        )
        assert inserted_ids == [id4]

        # reading with filtering
        results2 = ad_store.similarity_search("query", k=3, filter={"q2": "Q2"})
        assert results2 == [Document(id=id4, page_content=pc4, metadata=md4)]

        # delete by metadata
        del_by_md = ad_store.delete_by_metadata_filter(filter={"q2": "Q2"})
        assert del_by_md is not None
        assert del_by_md == 1
        results2n = ad_store.similarity_search("[-1,-1]", k=3, filter={"q2": "Q2"})
        assert results2n == []

    def test_failed_docs_autodetect_flat_novectorize_crud(
        self,
        astra_db_credentials: AstraDBCredentials,
        empty_collection_idxall_d2: Collection,
        embedding_d2: Embeddings,
    ) -> None:
        """Test autodetect + skipping failing documents."""
        empty_collection_idxall_d2.insert_many(
            [
                {
                    "_id": "1",
                    "$vector": [1, 2],
                    CUSTOM_CONTENT_KEY: "[1,2]",
                    "m1": "a",
                    "m2": "x",
                },
            ]
        )
        ad_store_e = AstraDBVectorStore(
            collection_name=empty_collection_idxall_d2.name,
            embedding=embedding_d2,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            autodetect_collection=True,
            ignore_invalid_documents=False,
        )
        ad_store_w = AstraDBVectorStore(
            collection_name=empty_collection_idxall_d2.name,
            embedding=embedding_d2,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            autodetect_collection=True,
            ignore_invalid_documents=True,
        )

        results_e = ad_store_e.similarity_search("[-1,-1]", k=3)
        assert len(results_e) == 1

        results_w = ad_store_w.similarity_search("[-1,-1]", k=3)
        assert len(results_w) == 1

        empty_collection_idxall_d2.insert_one(
            {
                "_id": "2",
                "$vector": [3, 4],
                "m1": "invalid:",
                "m2": "no 'cont'",
            }
        )

        with pytest.raises(KeyError):
            ad_store_e.similarity_search("[7,8]", k=3)

        # one case should result in just a warning:
        with pytest.warns(UserWarning) as rec_warnings:
            results_w_post = ad_store_w.similarity_search("[7,8]", k=3)
            f_rec_warnings = [
                wrn for wrn in rec_warnings if issubclass(wrn.category, UserWarning)
            ]
            assert len(f_rec_warnings) == 1
        assert len(results_w_post) == 1
