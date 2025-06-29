"""Test of Astra DB vector store class `AstraDBVectorStore`."""

from __future__ import annotations

import asyncio
import json
import math
import random
from typing import TYPE_CHECKING, Any

import pytest
from astrapy.api_options import APIOptions, TimeoutOptions
from astrapy.authentication import (
    EmbeddingAPIKeyHeaderProvider,
    RerankingAPIKeyHeaderProvider,
    StaticTokenProvider,
)
from astrapy.constants import SortMode
from astrapy.exceptions import DataAPITimeoutException
from langchain_core.documents import Document

from langchain_astradb.utils.astradb import COMPONENT_NAME_VECTORSTORE, SetupMode
from langchain_astradb.vectorstores import AstraDBVectorStore

from .conftest import (
    EUCLIDEAN_MIN_SIM_UNIT_VECTORS,
    MATCH_EPSILON,
    OPENAI_VECTORIZE_OPTIONS_HEADER,
    astra_db_env_vars_available,
)

if TYPE_CHECKING:
    from astrapy import Collection
    from langchain_core.embeddings import Embeddings

    from .conftest import AstraDBCredentials


def assert_list_of_numeric(value: list[float]) -> None:
    assert isinstance(value, list)
    assert all(isinstance(item, (float, int)) for item in value)


@pytest.fixture
def metadata_documents() -> list[Document]:
    """Documents for metadata and id tests"""
    return [
        Document(
            id="q",
            page_content="[1,2]",
            metadata={"ord": ord("q"), "group": "consonant", "letter": "q"},
        ),
        Document(
            id="w",
            page_content="[3,4]",
            metadata={"ord": ord("w"), "group": "consonant", "letter": "w"},
        ),
        Document(
            id="r",
            page_content="[5,6]",
            metadata={"ord": ord("r"), "group": "consonant", "letter": "r"},
        ),
        Document(
            id="e",
            page_content="[-1,2]",
            metadata={"ord": ord("e"), "group": "vowel", "letter": "e"},
        ),
        Document(
            id="i",
            page_content="[-3,4]",
            metadata={"ord": ord("i"), "group": "vowel", "letter": "i"},
        ),
        Document(
            id="o",
            page_content="[-5,6]",
            metadata={"ord": ord("o"), "group": "vowel", "letter": "o"},
        ),
    ]


@pytest.mark.skipif(
    not astra_db_env_vars_available(), reason="Missing Astra DB env. vars"
)
class TestAstraDBVectorStore:
    @pytest.mark.parametrize(
        ("is_vectorize", "page_contents", "collection_fixture_name"),
        [
            (
                False,
                [
                    "[1,2]",
                    "[3,4]",
                    "[5,6]",
                    "[7,8]",
                    "[9,10]",
                    "[11,12]",
                ],
                "empty_collection_d2",
            ),
            (
                True,
                [
                    "Dogs 1",
                    "Cats 3",
                    "Giraffes 5",
                    "Spiders 7",
                    "Pycnogonids 9",
                    "Rabbits 11",
                ],
                "empty_collection_vz",
            ),
        ],
        ids=["nonvectorize_store", "vectorize_store"],
    )
    def test_astradb_vectorstore_from_texts_sync(
        self,
        *,
        astra_db_credentials: AstraDBCredentials,
        openai_api_key: str,
        embedding_d2: Embeddings,
        is_vectorize: bool,
        page_contents: list[str],
        collection_fixture_name: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """from_texts methods and the associated warnings."""
        collection: Collection = request.getfixturevalue(collection_fixture_name)
        init_kwargs: dict[str, Any]
        if is_vectorize:
            init_kwargs = {
                "collection_vector_service_options": OPENAI_VECTORIZE_OPTIONS_HEADER,
                "collection_embedding_api_key": openai_api_key,
            }
        else:
            init_kwargs = {"embedding": embedding_d2}

        v_store = AstraDBVectorStore.from_texts(
            texts=page_contents[0:2],
            metadatas=[{"m": 1}, {"m": 3}],
            ids=["ft1", "ft3"],
            collection_name=collection.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            setup_mode=SetupMode.OFF,
            **init_kwargs,
        )
        search_results_triples_0 = v_store.similarity_search_with_score_id(
            page_contents[1],
            k=1,
        )
        assert len(search_results_triples_0) == 1
        res_doc_0, _, res_id_0 = search_results_triples_0[0]
        assert res_doc_0.page_content == page_contents[1]
        assert res_doc_0.metadata == {"m": 3}
        assert res_id_0 == "ft3"

        # testing additional kwargs & from_text-specific kwargs
        with pytest.warns(UserWarning):
            # unknown kwargs going to the constructor through _from_kwargs
            AstraDBVectorStore.from_texts(
                texts=page_contents[2:4],
                metadatas=[{"m": 5}, {"m": 7}],
                ids=["ft5", "ft7"],
                collection_name=collection.name,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                setup_mode=SetupMode.OFF,
                number_of_wizards=123,
                name_of_river="Thames",
                **init_kwargs,
            )
        search_results_triples_1 = v_store.similarity_search_with_score_id(
            page_contents[3],
            k=1,
        )
        assert len(search_results_triples_1) == 1
        res_doc_1, _, res_id_1 = search_results_triples_1[0]
        assert res_doc_1.page_content == page_contents[3]
        assert res_doc_1.metadata == {"m": 7}
        assert res_id_1 == "ft7"
        # routing of 'add_texts' keyword arguments
        v_store_2 = AstraDBVectorStore.from_texts(
            texts=page_contents[4:6],
            metadatas=[{"m": 9}, {"m": 11}],
            ids=["ft9", "ft11"],
            collection_name=collection.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            setup_mode=SetupMode.OFF,
            batch_size=19,
            batch_concurrency=23,
            overwrite_concurrency=29,
            **init_kwargs,
        )
        assert v_store_2.batch_size != 19
        assert v_store_2.bulk_insert_batch_concurrency != 23
        assert v_store_2.bulk_insert_overwrite_concurrency != 29
        search_results_triples_2 = v_store_2.similarity_search_with_score_id(
            page_contents[5],
            k=1,
        )
        assert len(search_results_triples_2) == 1
        res_doc_2, _, res_id_2 = search_results_triples_2[0]
        assert res_doc_2.page_content == page_contents[5]
        assert res_doc_2.metadata == {"m": 11}
        assert res_id_2 == "ft11"

    @pytest.mark.parametrize(
        ("is_vectorize", "page_contents", "collection_fixture_name"),
        [
            (False, ["[1,2]", "[3,4]"], "empty_collection_d2"),
            (True, ["Whales 1", "Tomatoes 3"], "empty_collection_vz"),
        ],
        ids=["nonvectorize_store", "vectorize_store"],
    )
    def test_astradb_vectorstore_from_documents_sync(
        self,
        *,
        astra_db_credentials: AstraDBCredentials,
        openai_api_key: str,
        embedding_d2: Embeddings,
        is_vectorize: bool,
        page_contents: list[str],
        collection_fixture_name: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """from_documents, esp. the various handling of ID-in-doc vs external."""
        collection: Collection = request.getfixturevalue(collection_fixture_name)
        pc1, pc2 = page_contents
        init_kwargs: dict[str, Any]
        if is_vectorize:
            init_kwargs = {
                "collection_vector_service_options": OPENAI_VECTORIZE_OPTIONS_HEADER,
                "collection_embedding_api_key": openai_api_key,
            }
        else:
            init_kwargs = {"embedding": embedding_d2}
        # no IDs.
        v_store = AstraDBVectorStore.from_documents(
            [
                Document(page_content=pc1, metadata={"m": 1}),
                Document(page_content=pc2, metadata={"m": 3}),
            ],
            collection_name=collection.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            setup_mode=SetupMode.OFF,
            **init_kwargs,
        )
        hits = v_store.similarity_search(pc2, k=1)
        assert len(hits) == 1
        assert hits[0].page_content == pc2
        assert hits[0].metadata == {"m": 3}
        v_store.clear()

        # IDs passed separately.
        with pytest.warns(DeprecationWarning) as rec_warnings:
            v_store_2 = AstraDBVectorStore.from_documents(
                [
                    Document(page_content=pc1, metadata={"m": 1}),
                    Document(page_content=pc2, metadata={"m": 3}),
                ],
                ids=["idx1", "idx3"],
                collection_name=collection.name,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                setup_mode=SetupMode.OFF,
                **init_kwargs,
            )
        f_rec_warnings = [
            wrn for wrn in rec_warnings if issubclass(wrn.category, DeprecationWarning)
        ]
        assert len(f_rec_warnings) == 1
        hits = v_store_2.similarity_search(pc2, k=1)
        assert len(hits) == 1
        assert hits[0].page_content == pc2
        assert hits[0].metadata == {"m": 3}
        assert hits[0].id == "idx3"
        v_store_2.clear()

        # IDs in documents.
        v_store_3 = AstraDBVectorStore.from_documents(
            [
                Document(page_content=pc1, metadata={"m": 1}, id="idx1"),
                Document(page_content=pc2, metadata={"m": 3}, id="idx3"),
            ],
            collection_name=collection.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            setup_mode=SetupMode.OFF,
            **init_kwargs,
        )
        hits = v_store_3.similarity_search(pc2, k=1)
        assert len(hits) == 1
        assert hits[0].page_content == pc2
        assert hits[0].metadata == {"m": 3}
        assert hits[0].id == "idx3"
        v_store_3.clear()

        # IDs both in documents and aside.
        with pytest.warns(DeprecationWarning) as rec_warnings:
            v_store_4 = AstraDBVectorStore.from_documents(
                [
                    Document(page_content=pc1, metadata={"m": 1}),
                    Document(page_content=pc2, metadata={"m": 3}, id="idy3"),
                ],
                ids=["idx1", "idx3"],
                collection_name=collection.name,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                setup_mode=SetupMode.OFF,
                **init_kwargs,
            )
        f_rec_warnings = [
            wrn for wrn in rec_warnings if issubclass(wrn.category, DeprecationWarning)
        ]
        assert len(f_rec_warnings) == 1
        hits = v_store_4.similarity_search(pc2, k=1)
        assert len(hits) == 1
        assert hits[0].page_content == pc2
        assert hits[0].metadata == {"m": 3}
        assert hits[0].id == "idx3"

    @pytest.mark.parametrize(
        ("is_vectorize", "page_contents", "collection_fixture_name"),
        [
            (
                False,
                [
                    "[1,2]",
                    "[3,4]",
                    "[5,6]",
                    "[7,8]",
                    "[9,10]",
                    "[11,12]",
                ],
                "empty_collection_d2",
            ),
            (
                True,
                [
                    "Dogs 1",
                    "Cats 3",
                    "Giraffes 5",
                    "Spiders 7",
                    "Pycnogonids 9",
                    "Rabbits 11",
                ],
                "empty_collection_vz",
            ),
        ],
        ids=["nonvectorize_store", "vectorize_store"],
    )
    async def test_astradb_vectorstore_from_texts_async(
        self,
        *,
        astra_db_credentials: AstraDBCredentials,
        openai_api_key: str,
        embedding_d2: Embeddings,
        is_vectorize: bool,
        page_contents: list[str],
        collection_fixture_name: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """from_texts methods and the associated warnings, async version."""
        collection: Collection = await asyncio.to_thread(
            request.getfixturevalue, collection_fixture_name
        )
        init_kwargs: dict[str, Any]
        if is_vectorize:
            init_kwargs = {
                "collection_vector_service_options": OPENAI_VECTORIZE_OPTIONS_HEADER,
                "collection_embedding_api_key": openai_api_key,
            }
        else:
            init_kwargs = {"embedding": embedding_d2}

        v_store = await AstraDBVectorStore.afrom_texts(
            texts=page_contents[0:2],
            metadatas=[{"m": 1}, {"m": 3}],
            ids=["ft1", "ft3"],
            collection_name=collection.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            setup_mode=SetupMode.OFF,
            **init_kwargs,
        )
        search_results_triples_0 = await v_store.asimilarity_search_with_score_id(
            page_contents[1],
            k=1,
        )
        assert len(search_results_triples_0) == 1
        res_doc_0, _, res_id_0 = search_results_triples_0[0]
        assert res_doc_0.page_content == page_contents[1]
        assert res_doc_0.metadata == {"m": 3}
        assert res_id_0 == "ft3"

        # testing additional kwargs & from_text-specific kwargs
        with pytest.warns(UserWarning):
            # unknown kwargs going to the constructor through _from_kwargs
            await AstraDBVectorStore.afrom_texts(
                texts=page_contents[2:4],
                metadatas=[{"m": 5}, {"m": 7}],
                ids=["ft5", "ft7"],
                collection_name=collection.name,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                setup_mode=SetupMode.OFF,
                number_of_wizards=123,
                name_of_river="Thames",
                **init_kwargs,
            )
        search_results_triples_1 = await v_store.asimilarity_search_with_score_id(
            page_contents[3],
            k=1,
        )
        assert len(search_results_triples_1) == 1
        res_doc_1, _, res_id_1 = search_results_triples_1[0]
        assert res_doc_1.page_content == page_contents[3]
        assert res_doc_1.metadata == {"m": 7}
        assert res_id_1 == "ft7"
        # routing of 'add_texts' keyword arguments
        v_store_2 = await AstraDBVectorStore.afrom_texts(
            texts=page_contents[4:6],
            metadatas=[{"m": 9}, {"m": 11}],
            ids=["ft9", "ft11"],
            collection_name=collection.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            setup_mode=SetupMode.OFF,
            batch_size=19,
            batch_concurrency=23,
            overwrite_concurrency=29,
            **init_kwargs,
        )
        assert v_store_2.batch_size != 19
        assert v_store_2.bulk_insert_batch_concurrency != 23
        assert v_store_2.bulk_insert_overwrite_concurrency != 29
        search_results_triples_2 = await v_store_2.asimilarity_search_with_score_id(
            page_contents[5],
            k=1,
        )
        assert len(search_results_triples_2) == 1
        res_doc_2, _, res_id_2 = search_results_triples_2[0]
        assert res_doc_2.page_content == page_contents[5]
        assert res_doc_2.metadata == {"m": 11}
        assert res_id_2 == "ft11"

    @pytest.mark.parametrize(
        ("is_vectorize", "page_contents", "collection_fixture_name"),
        [
            (False, ["[1,2]", "[3,4]"], "empty_collection_d2"),
            (True, ["Whales 1", "Tomatoes 3"], "empty_collection_vz"),
        ],
        ids=["nonvectorize_store", "vectorize_store"],
    )
    async def test_astradb_vectorstore_from_documents_async(
        self,
        *,
        astra_db_credentials: AstraDBCredentials,
        openai_api_key: str,
        embedding_d2: Embeddings,
        is_vectorize: bool,
        page_contents: list[str],
        collection_fixture_name: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """
        from_documents, esp. the various handling of ID-in-doc vs external.
        Async version.
        """
        collection: Collection = await asyncio.to_thread(
            request.getfixturevalue, collection_fixture_name
        )
        pc1, pc2 = page_contents
        init_kwargs: dict[str, Any]
        if is_vectorize:
            init_kwargs = {
                "collection_vector_service_options": OPENAI_VECTORIZE_OPTIONS_HEADER,
                "collection_embedding_api_key": openai_api_key,
            }
        else:
            init_kwargs = {"embedding": embedding_d2}
        # no IDs.
        v_store = await AstraDBVectorStore.afrom_documents(
            [
                Document(page_content=pc1, metadata={"m": 1}),
                Document(page_content=pc2, metadata={"m": 3}),
            ],
            collection_name=collection.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            setup_mode=SetupMode.OFF,
            **init_kwargs,
        )
        hits = await v_store.asimilarity_search(pc2, k=1)
        assert len(hits) == 1
        assert hits[0].page_content == pc2
        assert hits[0].metadata == {"m": 3}
        await v_store.aclear()

        # IDs passed separately.
        with pytest.warns(DeprecationWarning) as rec_warnings:
            v_store_2 = await AstraDBVectorStore.afrom_documents(
                [
                    Document(page_content=pc1, metadata={"m": 1}),
                    Document(page_content=pc2, metadata={"m": 3}),
                ],
                ids=["idx1", "idx3"],
                collection_name=collection.name,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                setup_mode=SetupMode.OFF,
                **init_kwargs,
            )
        f_rec_warnings = [
            wrn for wrn in rec_warnings if issubclass(wrn.category, DeprecationWarning)
        ]
        assert len(f_rec_warnings) == 1
        hits = await v_store_2.asimilarity_search(pc2, k=1)
        assert len(hits) == 1
        assert hits[0].page_content == pc2
        assert hits[0].metadata == {"m": 3}
        assert hits[0].id == "idx3"
        await v_store_2.aclear()

        # IDs in documents.
        v_store_3 = await AstraDBVectorStore.afrom_documents(
            [
                Document(page_content=pc1, metadata={"m": 1}, id="idx1"),
                Document(page_content=pc2, metadata={"m": 3}, id="idx3"),
            ],
            collection_name=collection.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            setup_mode=SetupMode.OFF,
            **init_kwargs,
        )
        hits = await v_store_3.asimilarity_search(pc2, k=1)
        assert len(hits) == 1
        assert hits[0].page_content == pc2
        assert hits[0].metadata == {"m": 3}
        assert hits[0].id == "idx3"
        await v_store_3.aclear()

        # IDs both in documents and aside.
        with pytest.warns(DeprecationWarning) as rec_warnings:
            v_store_4 = await AstraDBVectorStore.afrom_documents(
                [
                    Document(page_content=pc1, metadata={"m": 1}),
                    Document(page_content=pc2, metadata={"m": 3}, id="idy3"),
                ],
                ids=["idx1", "idx3"],
                collection_name=collection.name,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                setup_mode=SetupMode.OFF,
                **init_kwargs,
            )
        f_rec_warnings = [
            wrn for wrn in rec_warnings if issubclass(wrn.category, DeprecationWarning)
        ]
        assert len(f_rec_warnings) == 1
        hits = await v_store_4.asimilarity_search(pc2, k=1)
        assert len(hits) == 1
        assert hits[0].page_content == pc2
        assert hits[0].metadata == {"m": 3}
        assert hits[0].id == "idx3"

    @pytest.mark.parametrize(
        "vector_store",
        [
            "vector_store_d2",
            "vector_store_d2_stringtoken",
            "vector_store_vz",
        ],
    )
    def test_astradb_vectorstore_crud_sync(
        self,
        vector_store: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """Add/delete/update behaviour."""
        vstore: AstraDBVectorStore = request.getfixturevalue(vector_store)

        res0 = vstore.similarity_search("[-1,-1]", k=2)
        assert res0 == []
        # write and check again
        added_ids = vstore.add_texts(
            texts=["[1,2]", "[3,4]", "[5,6]"],
            metadatas=[
                {"k": "a", "ord": 0},
                {"k": "b", "ord": 1},
                {"k": "c", "ord": 2},
            ],
            ids=["a", "b", "c"],
        )
        # not requiring ordered match (elsewhere it may be overwriting some)
        assert set(added_ids) == {"a", "b", "c"}
        res1 = vstore.similarity_search("[-1,-1]", k=5)
        assert {doc.page_content for doc in res1} == {"[1,2]", "[3,4]", "[5,6]"}
        res2 = vstore.similarity_search("[3,4]", k=1)
        assert len(res2) == 1
        assert res2[0].page_content == "[3,4]"
        assert res2[0].metadata == {"k": "b", "ord": 1}
        assert res2[0].id == "b"
        # partial overwrite and count total entries
        added_ids_1 = vstore.add_texts(
            texts=["[5,6]", "[7,8]"],
            metadatas=[
                {"k": "c_new", "ord": 102},
                {"k": "d_new", "ord": 103},
            ],
            ids=["c", "d"],
        )
        # not requiring ordered match (elsewhere it may be overwriting some)
        assert set(added_ids_1) == {"c", "d"}
        res2 = vstore.similarity_search("[-1,-1]", k=10)
        assert len(res2) == 4
        # pick one that was just updated and check its metadata
        res3 = vstore.similarity_search_with_score_id(
            query="[5,6]", k=1, filter={"k": "c_new"}
        )
        doc3, _, id3 = res3[0]
        assert doc3.page_content == "[5,6]"
        assert doc3.metadata == {"k": "c_new", "ord": 102}
        assert id3 == "c"
        # delete and count again
        del1_res = vstore.delete(["b"])
        assert del1_res is True
        del2_res = vstore.delete(["a", "c", "Z!"])
        assert del2_res is True  # a non-existing ID was supplied
        assert len(vstore.similarity_search("[-1,-1]", k=10)) == 1
        # clear store
        vstore.clear()
        assert vstore.similarity_search("[-1,-1]", k=2) == []
        # add_documents with "ids" arg passthrough
        vstore.add_documents(
            [
                Document(page_content="[9,10]", metadata={"k": "v", "ord": 204}),
                Document(page_content="[11,12]", metadata={"k": "w", "ord": 205}),
            ],
            ids=["v", "w"],
        )
        assert len(vstore.similarity_search("[-1,-1]", k=10)) == 2
        res4 = vstore.similarity_search("[11,12]", k=1, filter={"k": "w"})
        assert res4[0].metadata["ord"] == 205
        assert res4[0].id == "w"
        # add_texts with "ids" arg passthrough
        vstore.add_texts(
            texts=["[13,14]", "[15,16]"],
            metadatas=[{"k": "r", "ord": 306}, {"k": "s", "ord": 307}],
            ids=["r", "s"],
        )
        assert len(vstore.similarity_search("[-1,-1]", k=10)) == 4
        res4 = vstore.similarity_search("[-1,-1]", k=1, filter={"k": "s"})
        assert res4[0].metadata["ord"] == 307
        assert res4[0].id == "s"
        # delete_by_document_id
        vstore.delete_by_document_id("s")
        assert len(vstore.similarity_search("[-1,-1]", k=10)) == 3

    @pytest.mark.parametrize(
        "vector_store",
        [
            "vector_store_d2",
            "vector_store_d2_stringtoken",
            "vector_store_vz",
        ],
    )
    async def test_astradb_vectorstore_crud_async(
        self,
        vector_store: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """Add/delete/update behaviour, async version."""
        vstore: AstraDBVectorStore = await asyncio.to_thread(
            request.getfixturevalue, vector_store
        )

        res0 = await vstore.asimilarity_search("[-1,-1]", k=2)
        assert res0 == []
        # write and check again
        added_ids = await vstore.aadd_texts(
            texts=["[1,2]", "[3,4]", "[5,6]"],
            metadatas=[
                {"k": "a", "ord": 0},
                {"k": "b", "ord": 1},
                {"k": "c", "ord": 2},
            ],
            ids=["a", "b", "c"],
        )
        # not requiring ordered match (elsewhere it may be overwriting some)
        assert set(added_ids) == {"a", "b", "c"}
        res1 = await vstore.asimilarity_search("[-1,-1]", k=5)
        assert {doc.page_content for doc in res1} == {"[1,2]", "[3,4]", "[5,6]"}
        res2 = await vstore.asimilarity_search("[3,4]", k=1)
        assert len(res2) == 1
        assert res2[0].page_content == "[3,4]"
        assert res2[0].metadata == {"k": "b", "ord": 1}
        assert res2[0].id == "b"
        # partial overwrite and count total entries
        added_ids_1 = await vstore.aadd_texts(
            texts=["[5,6]", "[7,8]"],
            metadatas=[
                {"k": "c_new", "ord": 102},
                {"k": "d_new", "ord": 103},
            ],
            ids=["c", "d"],
        )
        # not requiring ordered match (elsewhere it may be overwriting some)
        assert set(added_ids_1) == {"c", "d"}
        res2 = await vstore.asimilarity_search("[-1,-1]", k=10)
        assert len(res2) == 4
        # pick one that was just updated and check its metadata
        res3 = await vstore.asimilarity_search_with_score_id(
            query="[5,6]", k=1, filter={"k": "c_new"}
        )
        doc3, _, id3 = res3[0]
        assert doc3.page_content == "[5,6]"
        assert doc3.metadata == {"k": "c_new", "ord": 102}
        assert id3 == "c"
        # delete and count again
        del1_res = await vstore.adelete(["b"])
        assert del1_res is True
        del2_res = await vstore.adelete(["a", "c", "Z!"])
        assert del2_res is True  # a non-existing ID was supplied
        assert len(await vstore.asimilarity_search("[-1,-1]", k=10)) == 1
        # clear store
        await vstore.aclear()
        assert await vstore.asimilarity_search("[-1,-1]", k=2) == []
        # add_documents with "ids" arg passthrough
        await vstore.aadd_documents(
            [
                Document(page_content="[9,10]", metadata={"k": "v", "ord": 204}),
                Document(page_content="[11,12]", metadata={"k": "w", "ord": 205}),
            ],
            ids=["v", "w"],
        )
        assert len(await vstore.asimilarity_search("[-1,-1]", k=10)) == 2
        res4 = await vstore.asimilarity_search("[11,12]", k=1, filter={"k": "w"})
        assert res4[0].metadata["ord"] == 205
        assert res4[0].id == "w"
        # add_texts with "ids" arg passthrough
        await vstore.aadd_texts(
            texts=["[13,14]", "[15,16]"],
            metadatas=[{"k": "r", "ord": 306}, {"k": "s", "ord": 307}],
            ids=["r", "s"],
        )
        assert len(await vstore.asimilarity_search("[-1,-1]", k=10)) == 4
        res4 = await vstore.asimilarity_search("[-1,-1]", k=1, filter={"k": "s"})
        assert res4[0].metadata["ord"] == 307
        assert res4[0].id == "s"
        # delete_by_document_id
        await vstore.adelete_by_document_id("s")
        assert len(await vstore.asimilarity_search("[-1,-1]", k=10)) == 3

    def test_astradb_vectorstore_massive_insert_replace_sync(
        self,
        vector_store_d2: AstraDBVectorStore,
    ) -> None:
        """Testing the insert-many-and-replace-some patterns thoroughly."""
        full_size = 300
        first_group_size = 150
        second_group_slicer = [30, 100, 2]

        all_ids = [f"doc_{idx}" for idx in range(full_size)]
        all_texts = [f"[0,{idx + 1}]" for idx in range(full_size)]

        # massive insertion on empty
        group0_ids = all_ids[0:first_group_size]
        group0_texts = all_texts[0:first_group_size]
        inserted_ids0 = vector_store_d2.add_texts(
            texts=group0_texts,
            ids=group0_ids,
        )
        assert set(inserted_ids0) == set(group0_ids)
        # massive insertion with many overwrites scattered through
        # (we change the text to later check on DB for successful update)
        _s, _e, _st = second_group_slicer
        group1_ids = all_ids[_s:_e:_st] + all_ids[first_group_size:full_size]
        group1_texts = [
            txt.upper()
            for txt in (all_texts[_s:_e:_st] + all_texts[first_group_size:full_size])
        ]
        inserted_ids1 = vector_store_d2.add_texts(
            texts=group1_texts,
            ids=group1_ids,
        )
        assert set(inserted_ids1) == set(group1_ids)
        # final read (we want the IDs to do a full check)
        expected_text_by_id = {
            **dict(zip(group0_ids, group0_texts)),
            **dict(zip(group1_ids, group1_texts)),
        }
        full_results = vector_store_d2.similarity_search_with_score_id_by_vector(
            embedding=[1.0, 1.0],
            k=full_size,
        )
        for doc, _, doc_id in full_results:
            assert doc.page_content == expected_text_by_id[doc_id]

    async def test_astradb_vectorstore_massive_insert_replace_async(
        self,
        vector_store_d2: AstraDBVectorStore,
    ) -> None:
        """
        Testing the insert-many-and-replace-some patterns thoroughly.
        Async version.
        """
        full_size = 300
        first_group_size = 150
        second_group_slicer = [30, 100, 2]

        all_ids = [f"doc_{idx}" for idx in range(full_size)]
        all_texts = [f"[0,{idx + 1}]" for idx in range(full_size)]

        # massive insertion on empty
        group0_ids = all_ids[0:first_group_size]
        group0_texts = all_texts[0:first_group_size]

        inserted_ids0 = await vector_store_d2.aadd_texts(
            texts=group0_texts,
            ids=group0_ids,
        )
        assert set(inserted_ids0) == set(group0_ids)
        # massive insertion with many overwrites scattered through
        # (we change the text to later check on DB for successful update)
        _s, _e, _st = second_group_slicer
        group1_ids = all_ids[_s:_e:_st] + all_ids[first_group_size:full_size]
        group1_texts = [
            txt.upper()
            for txt in (all_texts[_s:_e:_st] + all_texts[first_group_size:full_size])
        ]
        inserted_ids1 = await vector_store_d2.aadd_texts(
            texts=group1_texts,
            ids=group1_ids,
        )
        assert set(inserted_ids1) == set(group1_ids)
        # final read (we want the IDs to do a full check)
        expected_text_by_id = dict(zip(all_ids, all_texts))
        full_results = await vector_store_d2.asimilarity_search_with_score_id_by_vector(
            embedding=[1.0, 1.0],
            k=full_size,
        )
        for doc, _, doc_id in full_results:
            assert doc.page_content == expected_text_by_id[doc_id]

    def test_astradb_vectorstore_delete_by_metadata_sync(
        self,
        vector_store_d2: AstraDBVectorStore,
    ) -> None:
        """Testing delete_by_metadata_filter."""
        full_size = 400
        # one in ... will be deleted
        deletee_ratio = 3

        documents = [
            Document(
                page_content="[1,1]", metadata={"deletee": doc_i % deletee_ratio == 0}
            )
            for doc_i in range(full_size)
        ]
        num_deletees = len([doc for doc in documents if doc.metadata["deletee"]])

        inserted_ids0 = vector_store_d2.add_documents(documents)
        assert len(inserted_ids0) == len(documents)

        d_result0 = vector_store_d2.delete_by_metadata_filter({"deletee": True})
        assert d_result0 == num_deletees
        count_on_store0 = len(
            vector_store_d2.similarity_search("[1,1]", k=full_size + 1)
        )
        assert count_on_store0 == full_size - num_deletees

        with pytest.raises(ValueError, match="does not accept an empty"):
            vector_store_d2.delete_by_metadata_filter({})
        count_on_store1 = len(
            vector_store_d2.similarity_search("[1,1]", k=full_size + 1)
        )
        assert count_on_store1 == full_size - num_deletees

    async def test_astradb_vectorstore_delete_by_metadata_async(
        self,
        vector_store_d2: AstraDBVectorStore,
    ) -> None:
        """Testing delete_by_metadata_filter, async version."""
        full_size = 400
        # one in ... will be deleted
        deletee_ratio = 3

        documents = [
            Document(
                page_content="[1,1]", metadata={"deletee": doc_i % deletee_ratio == 0}
            )
            for doc_i in range(full_size)
        ]
        num_deletees = len([doc for doc in documents if doc.metadata["deletee"]])

        inserted_ids0 = await vector_store_d2.aadd_documents(documents)
        assert len(inserted_ids0) == len(documents)

        d_result0 = await vector_store_d2.adelete_by_metadata_filter({"deletee": True})
        assert d_result0 == num_deletees
        count_on_store0 = len(
            await vector_store_d2.asimilarity_search("[1,1]", k=full_size + 1)
        )
        assert count_on_store0 == full_size - num_deletees

        with pytest.raises(ValueError, match="does not accept an empty"):
            await vector_store_d2.adelete_by_metadata_filter({})
        count_on_store1 = len(
            await vector_store_d2.asimilarity_search("[1,1]", k=full_size + 1)
        )
        assert count_on_store1 == full_size - num_deletees

    def test_astradb_vectorstore_update_metadata_sync(
        self,
        vector_store_d2: AstraDBVectorStore,
    ) -> None:
        """Testing update_metadata."""
        # this should not exceed the max number of hits from ANN search
        full_size = 20
        # one in ... will be updated
        updatee_ratio = 2
        # set this to lower than full_size // updatee_ratio to test everything.
        update_concurrency = 7

        def doc_sorter(doc: Document) -> str:
            return doc.id or ""

        orig_documents0 = [
            Document(
                page_content="[1,1]",
                metadata={
                    "to_update": doc_i % updatee_ratio == 0,
                    "inert_field": "I",
                    "updatee_field": "0",
                },
                id=f"um_doc_{doc_i}",
            )
            for doc_i in range(full_size)
        ]
        orig_documents = sorted(orig_documents0, key=doc_sorter)

        inserted_ids0 = vector_store_d2.add_documents(orig_documents)
        assert len(inserted_ids0) == len(orig_documents)

        update_map = {
            f"um_doc_{doc_i}": {"updatee_field": "1", "to_update": False}
            for doc_i in range(full_size)
            if doc_i % updatee_ratio == 0
        }
        u_result0 = vector_store_d2.update_metadata(
            update_map,
            overwrite_concurrency=update_concurrency,
        )
        assert u_result0 == len(update_map)

        all_documents = sorted(
            vector_store_d2.similarity_search("[1,1]", k=full_size),
            key=doc_sorter,
        )
        assert len(all_documents) == len(orig_documents)
        for doc, orig_doc in zip(all_documents, orig_documents):
            assert doc.id == orig_doc.id
            if doc.id in update_map:
                assert doc.metadata == orig_doc.metadata | update_map[doc.id]

    async def test_astradb_vectorstore_update_metadata_async(
        self,
        vector_store_d2: AstraDBVectorStore,
    ) -> None:
        """Testing update_metadata, async version."""
        # this should not exceed the max number of hits from ANN search
        full_size = 20
        # one in ... will be updated
        updatee_ratio = 2
        # set this to lower than full_size // updatee_ratio to test everything.
        update_concurrency = 7

        def doc_sorter(doc: Document) -> str:
            return doc.id or ""

        orig_documents0 = [
            Document(
                page_content="[1,1]",
                metadata={
                    "to_update": doc_i % updatee_ratio == 0,
                    "inert_field": "I",
                    "updatee_field": "0",
                },
                id=f"um_doc_{doc_i}",
            )
            for doc_i in range(full_size)
        ]
        orig_documents = sorted(orig_documents0, key=doc_sorter)

        inserted_ids0 = await vector_store_d2.aadd_documents(orig_documents)
        assert len(inserted_ids0) == len(orig_documents)

        update_map = {
            f"um_doc_{doc_i}": {"updatee_field": "1", "to_update": False}
            for doc_i in range(full_size)
            if doc_i % updatee_ratio == 0
        }
        u_result0 = await vector_store_d2.aupdate_metadata(
            update_map,
            overwrite_concurrency=update_concurrency,
        )
        assert u_result0 == len(update_map)

        all_documents = sorted(
            await vector_store_d2.asimilarity_search("[1,1]", k=full_size),
            key=doc_sorter,
        )
        assert len(all_documents) == len(orig_documents)
        for doc, orig_doc in zip(all_documents, orig_documents):
            assert doc.id == orig_doc.id
            if doc.id in update_map:
                assert doc.metadata == orig_doc.metadata | update_map[doc.id]

    def test_astradb_vectorstore_mmr_sync(
        self,
        vector_store_d2: AstraDBVectorStore,
    ) -> None:
        """MMR testing. We work on the unit circle with angle multiples
        of 2*pi/20 and prepare a store with known vectors for a controlled
        MMR outcome.
        """

        def _v_from_i(i: int, n: int) -> str:
            angle = 2 * math.pi * i / n
            vector = [math.cos(angle), math.sin(angle)]
            return json.dumps(vector)

        i_vals = [0, 4, 5, 13]
        n_val = 20
        vector_store_d2.add_texts(
            [_v_from_i(i, n_val) for i in i_vals], metadatas=[{"i": i} for i in i_vals]
        )
        res1 = vector_store_d2.max_marginal_relevance_search(
            _v_from_i(3, n_val),
            k=2,
            fetch_k=3,
        )
        res_i_vals = {doc.metadata["i"] for doc in res1}
        assert res_i_vals == {0, 4}

    async def test_astradb_vectorstore_mmr_async(
        self,
        vector_store_d2: AstraDBVectorStore,
    ) -> None:
        """MMR testing. We work on the unit circle with angle multiples
        of 2*pi/20 and prepare a store with known vectors for a controlled
        MMR outcome.
        Async version.
        """

        def _v_from_i(i: int, n: int) -> str:
            angle = 2 * math.pi * i / n
            vector = [math.cos(angle), math.sin(angle)]
            return json.dumps(vector)

        i_vals = [0, 4, 5, 13]
        n_val = 20
        await vector_store_d2.aadd_texts(
            [_v_from_i(i, n_val) for i in i_vals],
            metadatas=[{"i": i} for i in i_vals],
        )
        res1 = await vector_store_d2.amax_marginal_relevance_search(
            _v_from_i(3, n_val),
            k=2,
            fetch_k=3,
        )
        res_i_vals = {doc.metadata["i"] for doc in res1}
        assert res_i_vals == {0, 4}

    def test_astradb_vectorstore_mmr_vectorize_sync(
        self,
        vector_store_vz: AstraDBVectorStore,
    ) -> None:
        """MMR testing with vectorize, sync."""
        vector_store_vz.add_texts(
            [
                "Some dogs bark",
                "My dog growls",
                "Your cat meows",
                "Please do the dishes after you're done",
            ],
            ids=["db", "dg", "c", "z"],
        )

        hits = vector_store_vz.max_marginal_relevance_search(
            "The dogs say woof",
            k=2,
            fetch_k=3,
        )
        assert {doc.id for doc in hits} == {"db", "c"}

    async def test_astradb_vectorstore_mmr_vectorize_async(
        self,
        vector_store_vz: AstraDBVectorStore,
    ) -> None:
        """MMR async testing with vectorize, async."""
        await vector_store_vz.aadd_texts(
            [
                "Some dogs bark",
                "My dog growls",
                "Your cat meows",
                "Please do the dishes after you're done",
            ],
            ids=["db", "dg", "c", "z"],
        )

        hits = await vector_store_vz.amax_marginal_relevance_search(
            "The dogs say woof",
            k=2,
            fetch_k=3,
        )
        assert {doc.id for doc in hits} == {"db", "c"}

    @pytest.mark.parametrize(
        "vector_store",
        [
            "vector_store_d2",
            "vector_store_vz",
        ],
    )
    def test_astradb_vectorstore_metadata_filter(
        self,
        vector_store: str,
        request: pytest.FixtureRequest,
        metadata_documents: list[Document],
    ) -> None:
        """Metadata filtering."""
        vstore: AstraDBVectorStore = request.getfixturevalue(vector_store)
        vstore.add_documents(metadata_documents)
        # no filters
        res0 = vstore.similarity_search("[-1,-1]", k=10)
        assert {doc.metadata["letter"] for doc in res0} == set("qwreio")
        # single filter
        res1 = vstore.similarity_search(
            "[-1,-1]",
            k=10,
            filter={"group": "vowel"},
        )
        assert {doc.metadata["letter"] for doc in res1} == set("eio")
        # multiple filters
        res2 = vstore.similarity_search(
            "[-1,-1]",
            k=10,
            filter={"group": "consonant", "ord": ord("q")},
        )
        assert {doc.metadata["letter"] for doc in res2} == set("q")
        # excessive filters
        res3 = vstore.similarity_search(
            "[-1,-1]",
            k=10,
            filter={"group": "consonant", "ord": ord("q"), "case": "upper"},
        )
        assert res3 == []
        # filter with logical operator
        res4 = vstore.similarity_search(
            "[-1,-1]",
            k=10,
            filter={"$or": [{"ord": ord("q")}, {"ord": ord("r")}]},
        )
        assert {doc.metadata["letter"] for doc in res4} == {"q", "r"}

    @pytest.mark.parametrize(
        "vector_store",
        [
            "vector_store_d2",
            "vector_store_vz",
        ],
    )
    def test_astradb_vectorstore_metadata_search_sync(
        self,
        vector_store: str,
        request: pytest.FixtureRequest,
        metadata_documents: list[Document],
    ) -> None:
        """Metadata Search"""
        vstore: AstraDBVectorStore = request.getfixturevalue(vector_store)
        vstore.add_documents(metadata_documents)
        # no filters
        res0 = vstore.metadata_search(filter={}, n=10)
        assert {doc.metadata["letter"] for doc in res0} == set("qwreio")
        # single filter
        res1 = vstore.metadata_search(
            n=10,
            filter={"group": "vowel"},
        )
        assert {doc.metadata["letter"] for doc in res1} == set("eio")
        # multiple filters
        res2 = vstore.metadata_search(
            n=10,
            filter={"group": "consonant", "ord": ord("q")},
        )
        assert {doc.metadata["letter"] for doc in res2} == set("q")
        # excessive filters
        res3 = vstore.metadata_search(
            n=10,
            filter={"group": "consonant", "ord": ord("q"), "case": "upper"},
        )
        assert res3 == []
        # filter with logical operator
        res4 = vstore.metadata_search(
            n=10,
            filter={"$or": [{"ord": ord("q")}, {"ord": ord("r")}]},
        )
        assert {doc.metadata["letter"] for doc in res4} == {"q", "r"}

    @pytest.mark.parametrize(
        "vector_store",
        [
            "vector_store_d2",
            "vector_store_vz",
        ],
    )
    async def test_astradb_vectorstore_metadata_search_async(
        self,
        vector_store: str,
        request: pytest.FixtureRequest,
        metadata_documents: list[Document],
    ) -> None:
        """Metadata Search"""
        vstore: AstraDBVectorStore = await asyncio.to_thread(
            request.getfixturevalue, vector_store
        )
        await vstore.aadd_documents(metadata_documents)
        # no filters
        res0 = await vstore.ametadata_search(filter={}, n=10)
        assert {doc.metadata["letter"] for doc in res0} == set("qwreio")
        # single filter
        res1 = await vstore.ametadata_search(
            n=10,
            filter={"group": "vowel"},
        )
        assert {doc.metadata["letter"] for doc in res1} == set("eio")
        # multiple filters
        res2 = await vstore.ametadata_search(
            n=10,
            filter={"group": "consonant", "ord": ord("q")},
        )
        assert {doc.metadata["letter"] for doc in res2} == set("q")
        # excessive filters
        res3 = await vstore.ametadata_search(
            n=10,
            filter={"group": "consonant", "ord": ord("q"), "case": "upper"},
        )
        assert res3 == []
        # filter with logical operator
        res4 = await vstore.ametadata_search(
            n=10,
            filter={"$or": [{"ord": ord("q")}, {"ord": ord("r")}]},
        )
        assert {doc.metadata["letter"] for doc in res4} == {"q", "r"}

    @pytest.mark.parametrize(
        "vector_store",
        [
            "vector_store_d2",
            "vector_store_vz",
        ],
    )
    def test_astradb_vectorstore_get_by_document_id_sync(
        self,
        vector_store: str,
        request: pytest.FixtureRequest,
        metadata_documents: list[Document],
    ) -> None:
        """Get by document_id"""
        vstore: AstraDBVectorStore = request.getfixturevalue(vector_store)
        vstore.add_documents(metadata_documents)
        # invalid id
        invalid = vstore.get_by_document_id(document_id="z")
        assert invalid is None
        # valid id
        valid = vstore.get_by_document_id(document_id="q")
        assert isinstance(valid, Document)
        assert valid.id == "q"
        assert valid.page_content == "[1,2]"
        assert valid.metadata["group"] == "consonant"
        assert valid.metadata["letter"] == "q"

    @pytest.mark.parametrize(
        "vector_store",
        [
            "vector_store_d2",
            "vector_store_vz",
        ],
    )
    async def test_astradb_vectorstore_get_by_document_id_async(
        self,
        vector_store: str,
        request: pytest.FixtureRequest,
        metadata_documents: list[Document],
    ) -> None:
        """Get by document_id"""
        vstore: AstraDBVectorStore = await asyncio.to_thread(
            request.getfixturevalue, vector_store
        )
        await vstore.aadd_documents(metadata_documents)
        # invalid id
        invalid = await vstore.aget_by_document_id(document_id="z")
        assert invalid is None
        # valid id
        valid = await vstore.aget_by_document_id(document_id="q")
        assert isinstance(valid, Document)
        assert valid.id == "q"
        assert valid.page_content == "[1,2]"
        assert valid.metadata["group"] == "consonant"
        assert valid.metadata["letter"] == "q"

    @pytest.mark.parametrize(
        ("is_vectorize", "vector_store", "texts", "query"),
        [
            (
                False,
                "vector_store_d2",
                ["[1,1]", "[-1,-1]"],
                "[0.99999,1.00001]",
            ),
            (
                True,
                "vector_store_vz",
                ["the boat is in the sea", "perhaps triangles are blue"],
                "there's a ship in the ocean",
            ),
        ],
        ids=["nonvectorize_store", "vectorize_store"],
    )
    def test_astradb_vectorstore_similarity_scale_sync(
        self,
        *,
        is_vectorize: bool,
        vector_store: str,
        texts: list[str],
        query: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """Scale of the similarity scores."""
        vstore: AstraDBVectorStore = request.getfixturevalue(vector_store)
        vstore.add_texts(
            texts=texts,
            ids=["near", "far"],
        )
        res1 = vstore.similarity_search_with_score(
            query,
            k=2,
        )
        scores = [sco for _, sco in res1]
        sco_near, sco_far = scores
        assert sco_far >= 0
        if not is_vectorize:
            assert abs(1 - sco_near) < MATCH_EPSILON
            assert sco_far < EUCLIDEAN_MIN_SIM_UNIT_VECTORS + MATCH_EPSILON

    @pytest.mark.parametrize(
        ("is_vectorize", "vector_store", "texts", "query"),
        [
            (
                False,
                "vector_store_d2",
                ["[1,1]", "[-1,-1]"],
                "[0.99999,1.00001]",
            ),
            (
                True,
                "vector_store_vz",
                ["the boat is in the sea", "perhaps triangles are blue"],
                "there's a ship in the ocean",
            ),
        ],
        ids=["nonvectorize_store", "vectorize_store"],
    )
    async def test_astradb_vectorstore_similarity_scale_async(
        self,
        *,
        is_vectorize: bool,
        vector_store: str,
        texts: list[str],
        query: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """Scale of the similarity scores, async version."""
        vstore: AstraDBVectorStore = await asyncio.to_thread(
            request.getfixturevalue, vector_store
        )
        await vstore.aadd_texts(
            texts=texts,
            ids=["near", "far"],
        )
        res1 = await vstore.asimilarity_search_with_score(
            query,
            k=2,
        )
        scores = [sco for _, sco in res1]
        sco_near, sco_far = scores
        assert sco_far >= 0
        if not is_vectorize:
            assert abs(1 - sco_near) < MATCH_EPSILON
            assert sco_far < EUCLIDEAN_MIN_SIM_UNIT_VECTORS + MATCH_EPSILON

    @pytest.mark.parametrize(
        "vector_store",
        [
            "vector_store_d2",
            "vector_store_vz",
        ],
        ids=["nonvectorize_store", "vectorize_store"],
    )
    async def test_astradb_vectorstore_asimilarity_search_with_embedding(
        self,
        *,
        vector_store: str,
        metadata_documents: list[Document],
        request: pytest.FixtureRequest,
    ) -> None:
        """asimilarity_search_with_embedding can be used as building-block
        primitive by other components.
        """
        vstore: AstraDBVectorStore = await asyncio.to_thread(
            request.getfixturevalue, vector_store
        )
        await vstore.aadd_documents(metadata_documents)

        query_embedding, results = await vstore.asimilarity_search_with_embedding(
            query="[-1,2]"
        )

        assert_list_of_numeric(query_embedding)
        assert isinstance(results, list)
        assert len(results) > 0
        (doc, embedding) = results[0]
        assert isinstance(doc, Document)
        assert_list_of_numeric(embedding)

    @pytest.mark.parametrize(
        ("is_vectorize", "vector_store"),
        [
            (False, "vector_store_d2"),
            (True, "vector_store_vz"),
        ],
        ids=["nonvectorize_store", "vectorize_store"],
    )
    async def test_astradb_vectorstore_asimilarity_search_with_embedding_by_vector(
        self,
        *,
        is_vectorize: bool,
        vector_store: str,
        metadata_documents: list[Document],
        request: pytest.FixtureRequest,
    ) -> None:
        """asimilarity_search_with_embedding_by_vector can be used as
        building-block primitive by other components.
        """
        vstore: AstraDBVectorStore = await asyncio.to_thread(
            request.getfixturevalue, vector_store
        )
        await vstore.aadd_documents(metadata_documents)

        vector_dimensions = 1536 if is_vectorize else 2
        results = await vstore.asimilarity_search_with_embedding_by_vector(
            embedding=[
                random.uniform(0.0, 1.0)  # noqa: S311
                for _ in range(vector_dimensions)
            ]
        )

        assert isinstance(results, list)
        assert len(results) > 0
        (doc, embedding) = results[0]
        assert isinstance(doc, Document)
        assert_list_of_numeric(embedding)

    @pytest.mark.parametrize(
        "vector_store",
        [
            "vector_store_d2",
            "vector_store_vz",
        ],
        ids=["nonvectorize_store", "vectorize_store"],
    )
    def test_astradb_vectorstore_similarity_search_with_embedding(
        self,
        *,
        vector_store: str,
        metadata_documents: list[Document],
        request: pytest.FixtureRequest,
    ) -> None:
        """similarity_search_with_embedding can be used as building-block
        primitive by other components.
        """
        vstore: AstraDBVectorStore = request.getfixturevalue(vector_store)
        vstore.add_documents(metadata_documents)

        query_embedding, results = vstore.similarity_search_with_embedding(
            query="[-1,2]"
        )

        assert_list_of_numeric(query_embedding)
        assert isinstance(results, list)
        assert len(results) > 0
        (doc, embedding) = results[0]
        assert isinstance(doc, Document)
        assert_list_of_numeric(embedding)

    @pytest.mark.parametrize(
        ("is_vectorize", "vector_store"),
        [
            (False, "vector_store_d2"),
            (True, "vector_store_vz"),
        ],
        ids=["nonvectorize_store", "vectorize_store"],
    )
    def test_astradb_vectorstore_similarity_search_with_embedding_by_vector(
        self,
        *,
        is_vectorize: bool,
        vector_store: str,
        metadata_documents: list[Document],
        request: pytest.FixtureRequest,
    ) -> None:
        """similarity_search_with_embedding_by_vector can be used as
        building-block primitive by other components.
        """
        vstore: AstraDBVectorStore = request.getfixturevalue(vector_store)
        vstore.add_documents(metadata_documents)

        vector_dimensions = 1536 if is_vectorize else 2
        results = vstore.similarity_search_with_embedding_by_vector(
            embedding=[
                random.uniform(0.0, 1.0)  # noqa: S311
                for _ in range(vector_dimensions)
            ]
        )

        assert isinstance(results, list)
        assert len(results) > 0
        (doc, embedding) = results[0]
        assert isinstance(doc, Document)
        assert_list_of_numeric(embedding)

    @pytest.mark.parametrize(
        "vector_store",
        [
            "vector_store_d2",
            "vector_store_vz",
        ],
    )
    def test_astradb_vectorstore_massive_delete(
        self,
        vector_store: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """Larger-scale bulk deletes."""
        vstore: AstraDBVectorStore = request.getfixturevalue(vector_store)
        m = 150
        texts = [f"[0,{i + 1 / 7.0}]" for i in range(2 * m)]
        ids0 = [f"doc_{i}" for i in range(m)]
        ids1 = [f"doc_{i + m}" for i in range(m)]
        ids = ids0 + ids1
        vstore.add_texts(texts=texts, ids=ids)
        # deleting a bunch of these
        del_res0 = vstore.delete(ids0)
        assert del_res0 is True
        # deleting the rest plus a fake one
        del_res1 = vstore.delete([*ids1, "ghost!"])
        assert del_res1 is True  # ensure no error
        # nothing left
        assert vstore.similarity_search("[-1,-1]", k=2 * m) == []

    def test_astradb_vectorstore_custom_params_sync(
        self,
        astra_db_credentials: AstraDBCredentials,
        empty_collection_d2: Collection,
        embedding_d2: Embeddings,
    ) -> None:
        """Custom batch size and concurrency params."""
        v_store = AstraDBVectorStore(
            embedding=embedding_d2,
            collection_name=empty_collection_d2.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            setup_mode=SetupMode.OFF,
            batch_size=17,
            bulk_insert_batch_concurrency=13,
            bulk_insert_overwrite_concurrency=7,
            bulk_delete_concurrency=19,
        )
        # add_texts and delete some
        n = 120
        texts = [f"[0,{i + 1 / 7.0}]" for i in range(n)]
        ids = [f"doc_{i}" for i in range(n)]
        v_store.add_texts(texts=texts, ids=ids)
        v_store.add_texts(
            texts=texts,
            ids=ids,
            batch_size=19,
            batch_concurrency=7,
            overwrite_concurrency=13,
        )
        v_store.delete(ids[: n // 2])
        v_store.delete(ids[n // 2 :], concurrency=23)

    async def test_astradb_vectorstore_custom_params_async(
        self,
        astra_db_credentials: AstraDBCredentials,
        empty_collection_d2: Collection,
        embedding_d2: Embeddings,
    ) -> None:
        """Custom batch size and concurrency params, async version"""
        v_store = AstraDBVectorStore(
            embedding=embedding_d2,
            collection_name=empty_collection_d2.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            setup_mode=SetupMode.OFF,
            batch_size=17,
            bulk_insert_batch_concurrency=13,
            bulk_insert_overwrite_concurrency=7,
            bulk_delete_concurrency=19,
        )
        # add_texts and delete some
        n = 120
        texts = [f"[0,{i + 1 / 7.0}]" for i in range(n)]
        ids = [f"doc_{i}" for i in range(n)]
        await v_store.aadd_texts(texts=texts, ids=ids)
        await v_store.aadd_texts(
            texts=texts,
            ids=ids,
            batch_size=19,
            batch_concurrency=7,
            overwrite_concurrency=13,
        )
        await v_store.adelete(ids[: n // 2])
        await v_store.adelete(ids[n // 2 :], concurrency=23)

    def test_astradb_vectorstore_metrics(
        self,
        astra_db_credentials: AstraDBCredentials,
        embedding_d2: Embeddings,
        vector_store_d2: AstraDBVectorStore,
        ephemeral_collection_cleaner_d2: str,
    ) -> None:
        """Different choices of similarity metric.
        Both stores (with "cosine" and "euclidea" metrics) contain these two:
            - a vector slightly rotated w.r.t query vector
            - a vector which is a long multiple of query vector
        so, which one is "the closest one" depends on the metric.
        """
        euclidean_store = vector_store_d2

        isq2 = 0.5**0.5
        isa = 0.7
        isb = (1.0 - isa * isa) ** 0.5
        texts = [
            json.dumps([isa, isb]),
            json.dumps([10 * isq2, 10 * isq2]),
        ]
        ids = ["rotated", "scaled"]
        query_text = json.dumps([isq2, isq2])

        # prepare empty collections
        cosine_store = AstraDBVectorStore(
            embedding=embedding_d2,
            collection_name=ephemeral_collection_cleaner_d2,
            metric="cosine",
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        )

        cosine_store.add_texts(texts=texts, ids=ids)
        euclidean_store.add_texts(texts=texts, ids=ids)

        cosine_triples = cosine_store.similarity_search_with_score_id(
            query_text,
            k=1,
        )
        euclidean_triples = euclidean_store.similarity_search_with_score_id(
            query_text,
            k=1,
        )
        assert len(cosine_triples) == 1
        assert len(euclidean_triples) == 1
        assert cosine_triples[0][2] == "scaled"
        assert euclidean_triples[0][2] == "rotated"

    @pytest.mark.parametrize(
        "vector_store",
        [
            "vector_store_d2",
            "vector_store_vz",
        ],
        ids=["nonvectorize_store", "vectorize_store"],
    )
    def test_astradb_vectorstore_copy(
        self,
        *,
        vector_store: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """Verify changed attributes in 'copy', down in the astra_env of the store."""
        vstore0: AstraDBVectorStore = request.getfixturevalue(vector_store)

        # component_name, override test
        caller_names0 = {caller[0] for caller in vstore0.astra_env.full_callers}
        assert COMPONENT_NAME_VECTORSTORE in caller_names0

        vstore1 = vstore0.copy(component_name="xyz_component")

        caller_names1 = {caller[0] for caller in vstore1.astra_env.full_callers}
        assert COMPONENT_NAME_VECTORSTORE not in caller_names1
        assert "xyz_component" in caller_names1

        # basic copy (no changes)
        vstore1b = vstore0.copy()

        assert vstore1b.astra_env.token == vstore0.astra_env.token
        assert vstore1b.astra_env.ext_callers == vstore0.astra_env.ext_callers
        assert vstore1b.astra_env.component_name == vstore0.astra_env.component_name
        assert (
            vstore1b.astra_env.collection_embedding_api_key
            == vstore0.astra_env.collection_embedding_api_key
        )
        assert (
            vstore1b.astra_env.collection_reranking_api_key
            == vstore0.astra_env.collection_reranking_api_key
        )

        # other changeable attributes
        token2 = StaticTokenProvider("xyz")
        apikey2 = EmbeddingAPIKeyHeaderProvider("another_api_key")
        rrkkey2 = RerankingAPIKeyHeaderProvider("a fancy reranking key")
        vstore2 = vstore0.copy(
            token=token2,
            ext_callers=[("cnx", "cvx")],
            component_name="component_name2",
            collection_embedding_api_key=apikey2,
            collection_reranking_api_key=rrkkey2,
        )

        assert vstore2.astra_env.token == token2
        assert vstore2.astra_env.ext_callers == [("cnx", "cvx")]
        assert vstore2.astra_env.component_name == "component_name2"
        assert vstore2.astra_env.collection_embedding_api_key == apikey2
        assert vstore2.astra_env.collection_reranking_api_key == rrkkey2

    @pytest.mark.parametrize(
        ("vector_store", "is_vectorize"),
        [
            ("vector_store_d2", False),
            ("vector_store_vz", True),
        ],
        ids=[
            "nonvectorize_store",
            "vectorize_store",
        ],
    )
    def test_astradb_vectorstore_run_query(
        self,
        *,
        vector_store: str,
        is_vectorize: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """Structure of the metadata/vector for the 'run_query' test:
        ID      a       b       c       vec(when 2d)
        --------------------------------------------
         '1'    'a'                     [1,1]
         '2'    'a'                     [1,2]
         '3'    'a'                     ...
         '4'    'a'     'b'             ...
         '5'    'a'     'b'
         '6'            'b'     'c'
         '7'            'b'     'c'
         '8'                    'c'
         '9'                    'c'
        '10'                    'c'     [1,10]
        """
        vstore: AstraDBVectorStore = request.getfixturevalue(vector_store)

        documents_to_insert = [
            Document(
                page_content=f"This is number {i+1}" if is_vectorize else f"[1,{i+1}]",
                metadata={
                    **({"a": "a"} if i < 5 else {}),
                    **({"b": "b"} if i < 7 and i >= 3 else {}),
                    **({"c": "c"} if i >= 5 else {}),
                    "int_index": i + 1,
                },
                id=f"{i+1}",
            )
            for i in range(10)
        ]
        vstore.add_documents(documents_to_insert)
        sort_clause: dict[str, list[float] | str] = (
            {"$vectorize": "This is number 6"} if is_vectorize else {"$vector": [1, 6]}
        )

        # baseline
        hits0 = vstore.run_query(n=5)
        hits0_l = list(hits0)
        assert len(hits0_l) == 5
        assert isinstance(hits0_l[0][0], Document)
        assert isinstance(hits0_l[0][1], str)
        assert hits0_l[0][2] is None
        assert hits0_l[0][3] is None

        # just id
        hits1 = vstore.run_query(
            n=2,
            ids=["5", "6", "7", "8"],
        )
        hits1_l = list(hits1)
        assert len(hits1_l) >= 2  # sometimes _id-based queries return more.
        assert len(hits1_l) <= 4
        assert isinstance(hits1_l[0][0], Document)
        assert hits1_l[0][1] in {"5", "6", "7", "8"}
        assert hits1_l[0][2] is None
        assert hits1_l[0][3] is None

        # just simple filters
        hits2 = vstore.run_query(
            n=20,
            filter={"a": "a"},
        )
        hits2_l = list(hits2)
        assert len(hits2_l) == 5
        assert isinstance(hits2_l[0][0], Document)
        assert hits2_l[0][1] in {"1", "2", "3", "4", "5"}
        assert hits2_l[0][2] is None
        assert hits2_l[0][3] is None

        # just elaborate filters
        hits3 = vstore.run_query(
            n=30,
            filter={"$or": [{"a": "a"}, {"b": "b"}]},
        )
        hits3_l = list(hits3)
        assert len(hits3_l) == 7
        assert isinstance(hits3_l[0][0], Document)
        assert hits3_l[0][1] in {"1", "2", "3", "4", "5", "6", "7"}
        assert all(hit[1] in {"1", "2", "3", "4", "5", "6", "7"} for hit in hits3_l)
        assert hits3_l[0][2] is None
        assert hits3_l[0][3] is None

        # id + filters
        hits4 = vstore.run_query(
            n=2,
            ids=["1", "2", "6", "7", "9", "10"],
            filter={"$or": [{"a": "a"}, {"b": "b"}]},
        )
        hits4_l = list(hits4)
        assert len(hits4_l) >= 2  # sometimes _id-based queries return more.
        assert len(hits4_l) <= 4
        assert isinstance(hits4_l[0][0], Document)
        assert hits4_l[0][1] in {"1", "2", "6", "7"}
        assert all(hit[1] in {"1", "2", "6", "7"} for hit in hits4_l)
        assert hits4_l[0][2] is None
        assert hits4_l[0][3] is None

        # get similarity
        hits5 = vstore.run_query(
            n=2,
            ids=["1", "2", "6", "7", "9", "10"],
            filter={"$or": [{"a": "a"}, {"b": "b"}]},
            sort=sort_clause,
            include_similarity=True,
            include_sort_vector=False,
            include_embeddings=False,
        )
        hits5_l = list(hits5)
        assert len(hits5_l) >= 2  # sometimes _id-based queries return more.
        assert len(hits5_l) <= 4
        assert isinstance(hits5_l[0][0], Document)
        assert hits5_l[0][1] in {"1", "2", "6", "7"}
        assert hits5_l[0][2] is None
        assert isinstance(hits5_l[0][3], float)

        # get sortvector
        search_vector6, hits6 = vstore.run_query(
            n=2,
            ids=["1", "2", "6", "7", "9", "10"],
            filter={"$or": [{"a": "a"}, {"b": "b"}]},
            sort=sort_clause,
            include_similarity=False,
            include_sort_vector=True,
            include_embeddings=False,
        )
        hits6_l = list(hits6)
        assert isinstance(search_vector6, list)
        assert isinstance(search_vector6[0], (int, float))
        assert len(hits6_l) >= 2  # sometimes _id-based queries return more.
        assert len(hits6_l) <= 4
        assert isinstance(hits6_l[0][0], Document)
        assert hits6_l[0][1] in {"1", "2", "6", "7"}
        assert hits6_l[0][2] is None
        assert hits6_l[0][3] is None

        # get embeddings
        hits7 = vstore.run_query(
            n=2,
            ids=["1", "2", "6", "7", "9", "10"],
            filter={"$or": [{"a": "a"}, {"b": "b"}]},
            sort=sort_clause,
            include_similarity=False,
            include_sort_vector=False,
            include_embeddings=True,
        )
        hits7_l = list(hits7)
        assert len(hits7_l) >= 2  # sometimes _id-based queries return more.
        assert len(hits7_l) <= 4
        assert isinstance(hits7_l[0][0], Document)
        assert hits7_l[0][1] in {"1", "2", "6", "7"}
        assert isinstance(hits7_l[0][2], list)
        assert isinstance(hits7_l[0][2][0], (int, float))
        assert hits7_l[0][3] is None

        # nonvector sort
        hits9a = vstore.run_query(
            n=3,
            sort={"int_index": SortMode.ASCENDING},
        )
        hits9a_l = list(hits9a)
        assert [doc_id for _, doc_id, _, _ in hits9a_l] == ["1", "2", "3"]
        hits9d = vstore.run_query(
            n=3,
            sort={"int_index": SortMode.DESCENDING},
        )
        hits9d_l = list(hits9d)
        assert [doc_id for _, doc_id, _, _ in hits9d_l] == ["10", "9", "8"]

    @pytest.mark.parametrize(
        ("vector_store", "is_vectorize"),
        [
            ("vector_store_d2", False),
            ("vector_store_vz", True),
        ],
        ids=[
            "nonvectorize_store",
            "vectorize_store",
        ],
    )
    async def test_astradb_vectorstore_arun_query(
        self,
        *,
        vector_store: str,
        is_vectorize: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """Structure of the metadata/vector for the 'run_query' test:
        ID      a       b       c       vec(when 2d)
        --------------------------------------------
         '1'    'a'                     [1,1]
         '2'    'a'                     [1,2]
         '3'    'a'                     ...
         '4'    'a'     'b'             ...
         '5'    'a'     'b'
         '6'            'b'     'c'
         '7'            'b'     'c'
         '8'                    'c'
         '9'                    'c'
        '10'                    'c'     [1,10]
        """
        vstore: AstraDBVectorStore = await asyncio.to_thread(
            request.getfixturevalue, vector_store
        )

        documents_to_insert = [
            Document(
                page_content=f"This is number {i+1}" if is_vectorize else f"[1,{i+1}]",
                metadata={
                    **({"a": "a"} if i < 5 else {}),
                    **({"b": "b"} if i < 7 and i >= 3 else {}),
                    **({"c": "c"} if i >= 5 else {}),
                    "int_index": i + 1,
                },
                id=f"{i+1}",
            )
            for i in range(10)
        ]
        await vstore.aadd_documents(documents_to_insert)
        sort_clause: dict[str, list[float] | str] = (
            {"$vectorize": "This is number 6"} if is_vectorize else {"$vector": [1, 6]}
        )

        # baseline
        hits0 = await vstore.arun_query(n=5)
        hits0_l = [tpl async for tpl in hits0]
        assert len(hits0_l) == 5
        assert isinstance(hits0_l[0][0], Document)
        assert isinstance(hits0_l[0][1], str)
        assert hits0_l[0][2] is None
        assert hits0_l[0][3] is None

        # just id
        hits1 = await vstore.arun_query(
            n=2,
            ids=["5", "6", "7", "8"],
        )
        hits1_l = [tpl async for tpl in hits1]
        assert len(hits1_l) >= 2  # sometimes _id-based queries return more.
        assert len(hits1_l) <= 4
        assert isinstance(hits1_l[0][0], Document)
        assert hits1_l[0][1] in {"5", "6", "7", "8"}
        assert hits1_l[0][2] is None
        assert hits1_l[0][3] is None

        # just simple filters
        hits2 = await vstore.arun_query(
            n=20,
            filter={"a": "a"},
        )
        hits2_l = [tpl async for tpl in hits2]
        assert len(hits2_l) == 5
        assert isinstance(hits2_l[0][0], Document)
        assert hits2_l[0][1] in {"1", "2", "3", "4", "5"}
        assert hits2_l[0][2] is None
        assert hits2_l[0][3] is None

        # just elaborate filters
        hits3 = await vstore.arun_query(
            n=30,
            filter={"$or": [{"a": "a"}, {"b": "b"}]},
        )
        hits3_l = [tpl async for tpl in hits3]
        assert len(hits3_l) == 7
        assert isinstance(hits3_l[0][0], Document)
        assert hits3_l[0][1] in {"1", "2", "3", "4", "5", "6", "7"}
        assert all(hit[1] in {"1", "2", "3", "4", "5", "6", "7"} for hit in hits3_l)
        assert hits3_l[0][2] is None
        assert hits3_l[0][3] is None

        # id + filters
        hits4 = await vstore.arun_query(
            n=2,
            ids=["1", "2", "6", "7", "9", "10"],
            filter={"$or": [{"a": "a"}, {"b": "b"}]},
        )
        hits4_l = [tpl async for tpl in hits4]
        assert len(hits4_l) >= 2  # sometimes _id-based queries return more.
        assert len(hits4_l) <= 4
        assert isinstance(hits4_l[0][0], Document)
        assert hits4_l[0][1] in {"1", "2", "6", "7"}
        assert all(hit[1] in {"1", "2", "6", "7"} for hit in hits4_l)
        assert hits4_l[0][2] is None
        assert hits4_l[0][3] is None

        # get similarity
        hits5 = await vstore.arun_query(
            n=2,
            ids=["1", "2", "6", "7", "9", "10"],
            filter={"$or": [{"a": "a"}, {"b": "b"}]},
            sort=sort_clause,
            include_similarity=True,
            include_sort_vector=False,
            include_embeddings=False,
        )
        hits5_l = [tpl async for tpl in hits5]
        assert len(hits5_l) >= 2  # sometimes _id-based queries return more.
        assert len(hits5_l) <= 4
        assert isinstance(hits5_l[0][0], Document)
        assert hits5_l[0][1] in {"1", "2", "6", "7"}
        assert hits5_l[0][2] is None
        assert isinstance(hits5_l[0][3], float)

        # get sortvector
        search_vector6, hits6 = await vstore.arun_query(
            n=2,
            ids=["1", "2", "6", "7", "9", "10"],
            filter={"$or": [{"a": "a"}, {"b": "b"}]},
            sort=sort_clause,
            include_similarity=False,
            include_sort_vector=True,
            include_embeddings=False,
        )
        hits6_l = [tpl async for tpl in hits6]
        assert isinstance(search_vector6, list)
        assert isinstance(search_vector6[0], (int, float))
        assert len(hits6_l) >= 2  # sometimes _id-based queries return more.
        assert len(hits6_l) <= 4
        assert isinstance(hits6_l[0][0], Document)
        assert hits6_l[0][1] in {"1", "2", "6", "7"}
        assert hits6_l[0][2] is None
        assert hits6_l[0][3] is None

        # get embeddings
        hits7 = await vstore.arun_query(
            n=2,
            ids=["1", "2", "6", "7", "9", "10"],
            filter={"$or": [{"a": "a"}, {"b": "b"}]},
            sort=sort_clause,
            include_similarity=False,
            include_sort_vector=False,
            include_embeddings=True,
        )
        hits7_l = [tpl async for tpl in hits7]
        assert len(hits7_l) >= 2  # sometimes _id-based queries return more.
        assert len(hits7_l) <= 4
        assert isinstance(hits7_l[0][0], Document)
        assert hits7_l[0][1] in {"1", "2", "6", "7"}
        assert isinstance(hits7_l[0][2], list)
        assert isinstance(hits7_l[0][2][0], (int, float))
        assert hits7_l[0][3] is None

        # nonvector sort
        hits9a = await vstore.arun_query(
            n=3,
            sort={"int_index": SortMode.ASCENDING},
        )
        hits9a_l = [tpl async for tpl in hits9a]
        assert [doc_id for _, doc_id, _, _ in hits9a_l] == ["1", "2", "3"]
        hits9d = await vstore.arun_query(
            n=3,
            sort={"int_index": SortMode.DESCENDING},
        )
        hits9d_l = [tpl async for tpl in hits9d]
        assert [doc_id for _, doc_id, _, _ in hits9d_l] == ["10", "9", "8"]

    def test_astradb_vectorstore_custom_api_options(
        self,
        astra_db_credentials: AstraDBCredentials,
        empty_collection_d2: Collection,
        embedding_d2: Embeddings,
    ) -> None:
        """Craft a custom APIOptions (very low timeout), expect a timeout to occur."""
        baseline_v_store = AstraDBVectorStore(
            embedding=embedding_d2,
            collection_name=empty_collection_d2.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            setup_mode=SetupMode.OFF,
        )
        baseline_v_store.similarity_search("[0,1]")

        impatient_ao = APIOptions(timeout_options=TimeoutOptions(request_timeout_ms=1))
        impatient_v_store = AstraDBVectorStore(
            embedding=embedding_d2,
            collection_name=empty_collection_d2.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            setup_mode=SetupMode.OFF,
            api_options=impatient_ao,
        )
        with pytest.raises(DataAPITimeoutException):
            impatient_v_store.similarity_search("[0,1]")
