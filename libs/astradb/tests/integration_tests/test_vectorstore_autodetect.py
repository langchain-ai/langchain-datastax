"""Test of Astra DB vector store class `AstraDBVectorStore`, autodetect features.

Refer to `test_vectorstores.py` for the requirements to run.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Iterable

import pytest
from astrapy.authentication import StaticTokenProvider
from astrapy.info import (
    CollectionDefinition,
    CollectionLexicalOptions,
    CollectionRerankOptions,
)
from langchain_core.documents import Document

from langchain_astradb.utils.vector_store_codecs import (
    _DefaultVectorizeVSDocumentCodec,
    _FlatVectorizeVSDocumentCodec,
)
from langchain_astradb.vectorstores import AstraDBVectorStore

from .conftest import (
    CUSTOM_CONTENT_KEY,
    LEXICAL_OPTIONS,
    NVIDIA_RERANKING_OPTIONS_HEADER,
    OPENAI_VECTORIZE_OPTIONS_HEADER,
    astra_db_env_vars_available,
)

if TYPE_CHECKING:
    from astrapy import Collection, Database
    from langchain_core.embeddings import Embeddings

    from .conftest import AstraDBCredentials


COLLECTION_NAME_FORCEHYBRID_VECTORIZE = "lc_test_coll_hyb_vze"
COLLECTION_NAME_FORCENOHYBRID_VECTORIZE = "lc_test_coll_nohyb_vze"


@pytest.fixture(scope="module")
def collection_forcehybrid_vectorize(
    openai_api_key: str,
    nvidia_reranking_api_key: str,
    database: Database,
) -> Iterable[Collection]:
    """A general-purpose D=2(Euclidean) collection for per-test reuse."""
    collection = database.create_collection(
        COLLECTION_NAME_FORCEHYBRID_VECTORIZE,
        definition=(
            CollectionDefinition.builder()
            .set_vector_service(OPENAI_VECTORIZE_OPTIONS_HEADER)
            .set_lexical(LEXICAL_OPTIONS)
            .set_rerank(NVIDIA_RERANKING_OPTIONS_HEADER)
            .build()
        ),
        embedding_api_key=openai_api_key,
        reranking_api_key=nvidia_reranking_api_key,
    )
    yield collection

    collection.drop()


@pytest.fixture
def empty_collection_forcehybrid_vectorize(
    collection_forcehybrid_vectorize: Collection,
) -> Collection:
    collection_forcehybrid_vectorize.delete_many({})
    return collection_forcehybrid_vectorize


@pytest.fixture(scope="module")
def collection_forcenohybrid_vectorize(
    openai_api_key: str,
    database: Database,
) -> Iterable[Collection]:
    """A general-purpose D=2(Euclidean) collection for per-test reuse."""
    collection = database.create_collection(
        COLLECTION_NAME_FORCENOHYBRID_VECTORIZE,
        definition=(
            CollectionDefinition.builder()
            .set_vector_service(OPENAI_VECTORIZE_OPTIONS_HEADER)
            .set_lexical(CollectionLexicalOptions(enabled=False))
            .set_rerank(CollectionRerankOptions(enabled=False))
            .build()
        ),
        embedding_api_key=openai_api_key,
    )
    yield collection

    collection.drop()


@pytest.fixture
def empty_collection_forcenohybrid_vectorize(
    collection_forcenohybrid_vectorize: Collection,
) -> Collection:
    collection_forcenohybrid_vectorize.delete_many({})
    return collection_forcenohybrid_vectorize


@pytest.mark.skipif(
    not astra_db_env_vars_available(), reason="Missing Astra DB env. vars"
)
class TestAstraDBVectorStoreAutodetect:
    def test_autodetect_flat_novectorize_crud(
        self,
        astra_db_credentials: AstraDBCredentials,
        nvidia_reranking_api_key: str,
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
            collection_reranking_api_key=nvidia_reranking_api_key,
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

        def doc_sorter(doc: Document) -> str:
            return doc.id or ""

        # update metadata
        ad_store.update_metadata(
            {
                "1": {"m1": "A", "mZ": "Z"},
                "2": {"m1": "B", "mZ": "Z"},
            }
        )
        matches_z = ad_store.similarity_search("[-1,-1]", k=3, filter={"mZ": "Z"})
        assert len(matches_z) == 2
        s_matches_z = sorted(matches_z, key=doc_sorter)
        assert s_matches_z[0].metadata == {"m1": "A", "m2": "x", "mZ": "Z"}
        assert s_matches_z[1].metadata == {"m1": "B", "m2": "y", "mZ": "Z"}

    def test_autodetect_default_novectorize_crud(
        self,
        astra_db_credentials: AstraDBCredentials,
        embedding_d2: Embeddings,
        nvidia_reranking_api_key: str,
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
            collection_reranking_api_key=nvidia_reranking_api_key,
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

        def doc_sorter(doc: Document) -> str:
            return doc.id or ""

        # update metadata
        ad_store.update_metadata(
            {
                "1": {"m1": "A", "mZ": "Z"},
                "2": {"m1": "B", "mZ": "Z"},
            }
        )
        matches_z = ad_store.similarity_search("[-1,-1]", k=3, filter={"mZ": "Z"})
        assert len(matches_z) == 2
        s_matches_z = sorted(matches_z, key=doc_sorter)
        assert s_matches_z[0].metadata == {"m1": "A", "m2": "x", "mZ": "Z"}
        assert s_matches_z[1].metadata == {"m1": "B", "m2": "y", "mZ": "Z"}

    def test_autodetect_flat_vectorize_crud(
        self,
        astra_db_credentials: AstraDBCredentials,
        openai_api_key: str,
        nvidia_reranking_api_key: str,
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
            collection_reranking_api_key=nvidia_reranking_api_key,
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
        # TODO: reinstate once empty results are ok from API
        # results2n = ad_store.similarity_search("[-1,-1]", k=3, filter={"q2": "Q2"})
        # assert results2n == []

        def doc_sorter(doc: Document) -> str:
            return doc.id or ""

        # update metadata
        ad_store.update_metadata(
            {
                "1": {"m1": "A", "mZ": "Z"},
                "2": {"m1": "B", "mZ": "Z"},
            }
        )
        matches_z = ad_store.similarity_search("[-1,-1]", k=3, filter={"mZ": "Z"})
        assert len(matches_z) == 2
        s_matches_z = sorted(matches_z, key=doc_sorter)
        assert s_matches_z[0].metadata == {"m1": "A", "m2": "x", "mZ": "Z"}
        assert s_matches_z[1].metadata == {"m1": "B", "m2": "y", "mZ": "Z"}

    def test_autodetect_default_vectorize_crud(
        self,
        *,
        astra_db_credentials: AstraDBCredentials,
        openai_api_key: str,
        nvidia_reranking_api_key: str,
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
            collection_reranking_api_key=nvidia_reranking_api_key,
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
        # TODO: reinstate once empty results are ok from API
        # results2n = ad_store.similarity_search("[-1,-1]", k=3, filter={"q2": "Q2"})
        # assert results2n == []

        def doc_sorter(doc: Document) -> str:
            return doc.id or ""

        # update metadata
        ad_store.update_metadata(
            {
                "1": {"m1": "A", "mZ": "Z"},
                "2": {"m1": "B", "mZ": "Z"},
            }
        )
        matches_z = ad_store.similarity_search("[-1,-1]", k=3, filter={"mZ": "Z"})
        assert len(matches_z) == 2
        s_matches_z = sorted(matches_z, key=doc_sorter)
        assert s_matches_z[0].metadata == {"m1": "A", "m2": "x", "mZ": "Z"}
        assert s_matches_z[1].metadata == {"m1": "B", "m2": "y", "mZ": "Z"}

    def test_failed_docs_autodetect_flat_novectorize_crud(
        self,
        astra_db_credentials: AstraDBCredentials,
        nvidia_reranking_api_key: str,
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
            collection_reranking_api_key=nvidia_reranking_api_key,
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
            collection_reranking_api_key=nvidia_reranking_api_key,
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

    @pytest.mark.skipif(
        "LANGCHAIN_TEST_HYBRID" not in os.environ,
        reason="Hybrid tests not manually requested",
    )
    def test_vectorstore_autodetect_hybrid_prepopulated_vectorize(
        self,
        astra_db_credentials: AstraDBCredentials,
        openai_api_key: str,
        nvidia_reranking_api_key: str,
        empty_collection_forcehybrid_vectorize: Collection,
    ) -> None:
        # populate (with/out $lexical, flat/nested) to then check autodetection result
        for lexical_in_docs in [True, False]:
            for flat_md in [True, False]:
                empty_collection_forcehybrid_vectorize.delete_many({})
                # populate a collection
                _md_part: dict[str, Any] = (
                    {"k0": "v0"} if flat_md else {"metadata": {"k0": "v0"}}
                )
                empty_collection_forcehybrid_vectorize.insert_one(
                    {
                        "_id": "doc0",
                        "$vectorize": "the content",
                        **_md_part,
                        **({"$lexical": "the lex content"} if lexical_in_docs else {}),
                    }
                )
                # instantiate store with autodetect
                store_ad = AstraDBVectorStore(
                    collection_name=empty_collection_forcehybrid_vectorize.name,
                    token=StaticTokenProvider(astra_db_credentials["token"]),
                    api_endpoint=astra_db_credentials["api_endpoint"],
                    namespace=astra_db_credentials["namespace"],
                    environment=astra_db_credentials["environment"],
                    autodetect_collection=True,
                    collection_embedding_api_key=openai_api_key,
                    collection_reranking_api_key=nvidia_reranking_api_key,
                )
                # inspect store relevant attributes
                assert store_ad.has_lexical  # regardless of documents found
                # inspect codec
                assert store_ad.document_codec.server_side_embeddings
                assert store_ad.document_codec.has_lexical
                assert store_ad.document_codec.content_field == "$vectorize"
                if flat_md:
                    assert isinstance(
                        store_ad.document_codec,
                        _FlatVectorizeVSDocumentCodec,
                    )
                else:
                    assert isinstance(
                        store_ad.document_codec,
                        _DefaultVectorizeVSDocumentCodec,
                    )
                # insert one more doc and check it
                store_ad.add_documents(
                    [
                        Document(
                            id="doc1",
                            page_content="inserted",
                            metadata={"k1": "v1"},
                        ),
                    ]
                )
                doc1 = empty_collection_forcehybrid_vectorize.find_one(
                    {"_id": "doc1"},
                    projection={"*": True},
                )
                assert doc1 is not None
                assert "$lexical" in doc1
                assert "$vectorize" in doc1
                assert doc1["$lexical"] == "inserted"
                assert doc1["$vectorize"] == "inserted"
                if flat_md:
                    assert doc1["k1"] == "v1"
                else:
                    assert doc1["metadata"] == {"k1": "v1"}
                # run perfunctory search + inspect results
                hits = store_ad.similarity_search("query", k=2)
                assert len(hits) == 2
                assert {doc.id for doc in hits} == {"doc0", "doc1"}

    @pytest.mark.skipif(
        "LANGCHAIN_TEST_HYBRID" not in os.environ,
        reason="Hybrid tests not manually requested",
    )
    def test_vectorstore_autodetect_nohybrid_prepopulated_vectorize(
        self,
        astra_db_credentials: AstraDBCredentials,
        openai_api_key: str,
        empty_collection_forcenohybrid_vectorize: Collection,
    ) -> None:
        # populate (flat/nested) to then check autodetection result
        for flat_md in [True, False]:
            empty_collection_forcenohybrid_vectorize.delete_many({})
            # populate a collection
            _md_part: dict[str, Any] = (
                {"k0": "v0"} if flat_md else {"metadata": {"k0": "v0"}}
            )
            empty_collection_forcenohybrid_vectorize.insert_one(
                {
                    "_id": "doc0",
                    "$vectorize": "the content",
                    **_md_part,
                }
            )
            # instantiate store with autodetect
            store_ad = AstraDBVectorStore(
                collection_name=empty_collection_forcenohybrid_vectorize.name,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                autodetect_collection=True,
                collection_embedding_api_key=openai_api_key,
            )
            # inspect store relevant attributes
            assert not store_ad.has_lexical
            # inspect codec
            assert store_ad.document_codec.server_side_embeddings
            assert not store_ad.document_codec.has_lexical
            assert store_ad.document_codec.content_field == "$vectorize"
            if flat_md:
                assert isinstance(
                    store_ad.document_codec,
                    _FlatVectorizeVSDocumentCodec,
                )
            else:
                assert isinstance(
                    store_ad.document_codec,
                    _DefaultVectorizeVSDocumentCodec,
                )
            # insert one more doc and check it
            store_ad.add_documents(
                [
                    Document(
                        id="doc1",
                        page_content="inserted",
                        metadata={"k1": "v1"},
                    ),
                ]
            )
            doc1 = empty_collection_forcenohybrid_vectorize.find_one(
                {"_id": "doc1"},
                projection={"*": True},
            )
            assert doc1 is not None
            assert "$lexical" not in doc1
            assert "$vectorize" in doc1
            assert doc1["$vectorize"] == "inserted"
            if flat_md:
                assert doc1["k1"] == "v1"
            else:
                assert doc1["metadata"] == {"k1": "v1"}
            # run perfunctory search + inspect results
            hits = store_ad.similarity_search("query", k=2)
            assert len(hits) == 2
            assert {doc.id for doc in hits} == {"doc0", "doc1"}
