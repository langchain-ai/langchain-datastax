"""Test of Astra DB vector store class `AstraDBVectorStore`

Required to run this test:
    - a recent `astrapy` Python package available
    - an Astra DB instance;
    - the two environment variables set:
        export ASTRA_DB_API_ENDPOINT="https://<DB-ID>-us-east1.apps.astra.datastax.com"
        export ASTRA_DB_APPLICATION_TOKEN="AstraCS:........."
    - optionally this as well (otherwise defaults are used):
        export ASTRA_DB_KEYSPACE="my_keyspace"
    - an openai secret for SHARED_SECRET mode, associated to the DB, with name on KMS:
        export SHARED_SECRET_NAME_OPENAI="the_api_key_name_in_Astra_KMS"
    - an OpenAI key for the vectorize test (in HEADER mode):
        export OPENAI_API_KEY="..."
    - optionally:
        export ASTRA_DB_SKIP_COLLECTION_DELETIONS="0" ("1" = no deletions, default)
"""

from __future__ import annotations

import json
import math
import os
import warnings
from typing import TYPE_CHECKING, Iterable

import pytest
from astrapy.authentication import EmbeddingAPIKeyHeaderProvider, StaticTokenProvider
from astrapy.exceptions import InsertManyException
from langchain_core.documents import Document

from langchain_astradb.utils.astradb import SetupMode
from langchain_astradb.vectorstores import AstraDBVectorStore
from tests.conftest import ParserEmbeddings

from .conftest import (
    COLLECTION_NAME_D2,
    COLLECTION_NAME_VZ,
    EPHEMERAL_COLLECTION_NAME_D2,
    EPHEMERAL_COLLECTION_NAME_VZ,
    INCOMPATIBLE_INDEXING_MSG,
    OPENAI_VECTORIZE_OPTIONS,
    _has_env_vars,
)

if TYPE_CHECKING:
    from astrapy import Database
    from astrapy.db import AstraDB

    from .conftest import AstraDBCredentials


MATCH_EPSILON = 0.0001


@pytest.mark.skipif(not _has_env_vars(), reason="Missing Astra DB env. vars")
class TestAstraDBVectorStore:
    def test_astradb_vectorstore_create_delete_sync(
        self,
        database: Database,
        collection_d2: Collection,
        astra_db_credentials: AstraDBCredentials,
        embedding_d2: Embeddings,
        ephemeral_collection_cleaner_vd2: Collection,
    ) -> None:
        """Create and delete."""
        v_store = AstraDBVectorStore(
            embedding=embedding_d2,
            collection_name=EPHEMERAL_COLLECTION_NAME_D2,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        )
        v_store.add_texts(["[1,2]"])
        v_store.delete_collection()
        assert EPHEMERAL_COLLECTION_NAME_D2 not in database.list_collection_names()

    def test_astradb_vectorstore_create_delete_vectorize_sync(
        self,
        database: Database,
        collection_vz: Collection,
        astra_db_credentials: AstraDBCredentials,
        ephemeral_collection_cleaner_vz: Collection,
    ) -> None:
        """Create and delete with vectorize option."""
        v_store = AstraDBVectorStore(
            collection_name=EPHEMERAL_COLLECTION_NAME_VZ,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            collection_vector_service_options=OPENAI_VECTORIZE_OPTIONS_HEADER,
            collection_embedding_api_key=OPENAI_API_KEY,
        )
        v_store.add_texts(["This is text"])
        v_store.delete_collection()
        assert EPHEMERAL_COLLECTION_NAME_VZ not in database.list_collection_names()

    async def test_astradb_vectorstore_create_delete_async(
        self,
        database: Database,
        collection_d2: Collection,
        astra_db_credentials: AstraDBCredentials,
        embedding_d2: Embeddings,
        ephemeral_collection_cleaner_d2: Collection,
    ) -> None:
        """Create and delete, async."""
        v_store = AstraDBVectorStore(
            embedding=embedding_d2,
            collection_name=EPHEMERAL_COLLECTION_NAME_D2,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        )
        await v_store.aadd_texts(["[1,2]"])
        await v_store.adelete_collection()
        assert EPHEMERAL_COLLECTION_NAME_D2 not in database.list_collection_names()

    async def test_astradb_vectorstore_create_delete_vectorize_async(
        self,
        database: Database,
        collection_vz: Collection,
        astra_db_credentials: AstraDBCredentials,
        ephemeral_collection_cleaner_vz: Collection,
    ) -> None:
        """Create and delete with vectorize option, async."""
        v_store = AstraDBVectorStore(
            collection_name=EPHEMERAL_COLLECTION_NAME_VZ,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            collection_vector_service_options=OPENAI_VECTORIZE_OPTIONS_HEADER,
            collection_embedding_api_key=OPENAI_API_KEY,
        )
        await v_store.aadd_texts(["[1,2]"])
        await v_store.adelete_collection()
        assert EPHEMERAL_COLLECTION_NAME_VZ not in database.list_collection_names()

    def test_astradb_vectorstore_pre_delete_collection_sync(
        self,
        embedding_d2: Embeddings,
        astra_db_credentials: AstraDBCredentials,
        ephemeral_collection_cleaner_d2: Collection,
    ) -> None:
        """Use of the pre_delete_collection flag."""
        v_store = AstraDBVectorStore(
            embedding=embedding_d2,
            collection_name=EPHEMERAL_COLLECTION_NAME_D2,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        )
        v_store.add_texts(texts=["[1,2]"])
        res1 = v_store.similarity_search("[-1,-1]", k=5)
        assert len(res1) == 1
        v_store = AstraDBVectorStore(
            embedding=embedding_d2,
            collection_name=EPHEMERAL_COLLECTION_NAME_D2,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            pre_delete_collection=True,
        )
        res1 = v_store.similarity_search("[-1,-1]", k=5)
        assert len(res1) == 0

    async def test_astradb_vectorstore_pre_delete_collection_async(
        self,
        embedding_d2: Embeddings,
        astra_db_credentials: AstraDBCredentials,
        ephemeral_collection_cleaner_d2: Collection,
    ) -> None:
        v_store = AstraDBVectorStore(
            embedding=embedding_d2,
            collection_name=EPHEMERAL_COLLECTION_NAME_D2,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            setup_mode=SetupMode.ASYNC,
        )
        v_store.add_texts(
            texts=["[1,2]"],
        )
        await v_store.aadd_texts(texts=["[1,2]"])
        res1 = await v_store.asimilarity_search("[-1,-1]", k=5)
        assert len(res1) == 1
        v_store = AstraDBVectorStore(
            embedding=embedding_d2,
            collection_name=EPHEMERAL_COLLECTION_NAME_D2,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            setup_mode=SetupMode.ASYNC,
            pre_delete_collection=True,
        )
        res1 = await v_store.asimilarity_search("[-1,-1]", k=5)
        assert len(res1) == 0

    ### END OF THE FUNCTIONS THAT CREATE EPHEMERAL COLLECTIONS

    def test_astradb_vectorstore_vectorize_headers_precedence_stringheader(
        self,
        collection_vz: Collection,
        astra_db_credentials: AstraDBCredentials,
    ) -> None:
        """
        Test that header, if passed, takes precedence over vectorize setting.
        To do so, a faulty header is passed, expecting the call to fail.
        """
        v_store = AstraDBVectorStore(
            collection_name=COLLECTION_NAME_VZ,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            setup_mode=SetupMode.OFF,
            collection_vector_service_options=OPENAI_VECTORIZE_OPTIONS,
            collection_embedding_api_key="verywrong",
        )
        with pytest.raises(InsertManyException):
            v_store.add_texts(["Failing"])

    def test_astradb_vectorstore_vectorize_headers_precedence_headerprovider(
        self,
        collection_vz: Collection,
        astra_db_credentials: AstraDBCredentials,
    ) -> None:
        """
        Test that header, if passed, takes precedence over vectorize setting.
        To do so, a faulty header is passed, expecting the call to fail.
        This version passes the header through an EmbeddingHeaderProvider
        """
        v_store = AstraDBVectorStore(
            collection_name=COLLECTION_NAME_VZ,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            setup_mode=SetupMode.OFF,
            collection_vector_service_options=OPENAI_VECTORIZE_OPTIONS,
            collection_embedding_api_key=EmbeddingAPIKeyHeaderProvider("verywrong"),
        )
        with pytest.raises(InsertManyException):
            v_store.add_texts(["Failing"])

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
                "empty_collection_d2"
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
        page_contents: list[str],
        is_vectorize: bool,
        collection_fixture_name: str,
        embedding_d2: Embeddings,
        astra_db_credentials: AstraDBCredentials,
        request: pytest.FixtureRequest,
    ) -> None:
        """from_texts methods and the associated warnings."""
        collection: Collection = request.getfixturevalue(collection_fixture_name)
        init_kwargs: dict[str, Any]
        if is_vectorize:
            init_kwargs = {"collection_vector_service_options": OPENAI_VECTORIZE_OPTIONS}
        else:
            init_kwargs = {"embedding": embedding_d2}

        v_store = AstraDBVectorStore.from_texts(
            texts=["[1,2]", "[3,4]"],
            metadatas=[{"m": 1}, {"m": 3}],
            ids=["ft1", "ft3"],
            collection_name=collection.name,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            setup_mode=SetupMode.OFF,
            **init_kwargs,
        )
        search_results_triples_0 = v_store.similarity_search_with_score_id(
            "[3,4]",
            k=1,
        )
        assert len(search_results_triples_0) == 1
        res_doc_0, _, res_id_0 = search_results_triples_0[0]
        assert res_doc_0.page_content == "[3,4]"
        assert res_doc_0.metadata == {"m": 3}
        assert res_id_0 == "ft3"

        # testing additional kwargs & from_text-specific kwargs
        with pytest.warns(UserWarning):
            # unknown kwargs going to the constructor through _from_kwargs
            AstraDBVectorStore.from_texts(
                texts=["[5,6]", "[7,8]"],
                metadatas=[{"m": 5}, {"m": 7}],
                ids=["ft5", "ft7"],
                collection_name=collection.name,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                setup_mode=SetupMode.OFF,
                number_of_wizards=123,
                name_of_river="Thames",
                **init_kwargs,
            )
        search_results_triples_1 = v_store.similarity_search_with_score_id(
            "[7,8]",
            k=1,
        )
        assert len(search_results_triples_1) == 1
        res_doc_1, _, res_id_1 = search_results_triples_1[0]
        assert res_doc_1.page_content == "[7,8]"
        assert res_doc_1.metadata == {"m": 7}
        assert res_id_1 == "ft7"
        # routing of 'add_texts' keyword arguments
        v_store_2 = AstraDBVectorStore.from_texts(
            texts=["[9,10]", "[11,12]"],
            metadatas=[{"m": 9}, {"m": 11}],
            ids=["ft9", "ft11"],
            collection_name=collection.name,
            token=astra_db_credentials["token"],
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
            "[11,12]",
            k=1,
        )
        assert len(search_results_triples_2) == 1
        res_doc_2, _, res_id_2 = search_results_triples_2[0]
        assert res_doc_2.page_content == "[11,12]"
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
        page_contents: list[str],
        is_vectorize: bool,
        collection_fixture_name: str,
        embedding_d2: Embeddings,
        astra_db_credentials: AstraDBCredentials,
        request: pytest.FixtureRequest,
    ) -> None:
        """from_documents, esp. the various handling of ID-in-doc vs external."""
        collection: Collection = request.getfixturevalue(collection_fixture_name)
        pc1, pc2 = page_contents
        init_kwargs: dict[str, Any]
        if is_vectorize:
            init_kwargs = {"collection_vector_service_options": OPENAI_VECTORIZE_OPTIONS}
        else:
            init_kwargs = {"embedding": embedding_d2}
        # no IDs.
        v_store = AstraDBVectorStore.from_documents(
            [
                Document(page_content=pc1, metadata={"m": 1}),
                Document(page_content=pc2, metadata={"m": 3}),
            ],
            collection_name=collection.name,
            token=astra_db_credentials["token"],
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
                token=astra_db_credentials["token"],
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
            token=astra_db_credentials["token"],
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
                token=astra_db_credentials["token"],
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
                "empty_collection_d2"
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
        page_contents: list[str],
        is_vectorize: bool,
        collection_fixture_name: str,
        embedding_d2: Embeddings,
        astra_db_credentials: AstraDBCredentials,
        request: pytest.FixtureRequest,
    ) -> None:
        """from_texts methods and the associated warnings, async version."""
        collection: Collection = request.getfixturevalue(collection_fixture_name)
        init_kwargs: dict[str, Any]
        if is_vectorize:
            init_kwargs = {"collection_vector_service_options": OPENAI_VECTORIZE_OPTIONS}
        else:
            init_kwargs = {"embedding": embedding_d2}

        v_store = await AstraDBVectorStore.afrom_texts(
            texts=["[1,2]", "[3,4]"],
            metadatas=[{"m": 1}, {"m": 3}],
            ids=["ft1", "ft3"],
            collection_name=collection.name,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            setup_mode=SetupMode.OFF,
            **init_kwargs,
        )
        search_results_triples_0 = await v_store.asimilarity_search_with_score_id(
            "[3,4]",
            k=1,
        )
        assert len(search_results_triples_0) == 1
        res_doc_0, _, res_id_0 = search_results_triples_0[0]
        assert res_doc_0.page_content == "[3,4]"
        assert res_doc_0.metadata == {"m": 3}
        assert res_id_0 == "ft3"

        # testing additional kwargs & from_text-specific kwargs
        with pytest.warns(UserWarning):
            # unknown kwargs going to the constructor through _from_kwargs
            await AstraDBVectorStore.afrom_texts(
                texts=["[5,6]", "[7,8]"],
                metadatas=[{"m": 5}, {"m": 7}],
                ids=["ft5", "ft7"],
                collection_name=collection.name,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                setup_mode=SetupMode.OFF,
                number_of_wizards=123,
                name_of_river="Thames",
                **init_kwargs,
            )
        search_results_triples_1 = await v_store.asimilarity_search_with_score_id(
            "[7,8]",
            k=1,
        )
        assert len(search_results_triples_1) == 1
        res_doc_1, _, res_id_1 = search_results_triples_1[0]
        assert res_doc_1.page_content == "[7,8]"
        assert res_doc_1.metadata == {"m": 7}
        assert res_id_1 == "ft7"
        # routing of 'add_texts' keyword arguments
        v_store_2 = await AstraDBVectorStore.afrom_texts(
            texts=["[9,10]", "[11,12]"],
            metadatas=[{"m": 9}, {"m": 11}],
            ids=["ft9", "ft11"],
            collection_name=collection.name,
            token=astra_db_credentials["token"],
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
            "[11,12]",
            k=1,
        )
        assert len(search_results_triples_2) == 1
        res_doc_2, _, res_id_2 = search_results_triples_2[0]
        assert res_doc_2.page_content == "[11,12]"
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
        page_contents: list[str],
        is_vectorize: bool,
        collection_fixture_name: str,
        embedding_d2: Embeddings,
        astra_db_credentials: AstraDBCredentials,
        request: pytest.FixtureRequest,
    ) -> None:
        """
        from_documents, esp. the various handling of ID-in-doc vs external.
        Async version.
        """
        collection: Collection = request.getfixturevalue(collection_fixture_name)
        pc1, pc2 = page_contents
        init_kwargs: dict[str, Any]
        if is_vectorize:
            init_kwargs = {"collection_vector_service_options": OPENAI_VECTORIZE_OPTIONS}
        else:
            init_kwargs = {"embedding": embedding_d2}
        # no IDs.
        v_store = await AstraDBVectorStore.afrom_documents(
            [
                Document(page_content=pc1, metadata={"m": 1}),
                Document(page_content=pc2, metadata={"m": 3}),
            ],
            collection_name=collection.name,
            token=astra_db_credentials["token"],
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
        v_store.clear()

        # IDs passed separately.
        with pytest.warns(DeprecationWarning) as rec_warnings:
            v_store_2 = await AstraDBVectorStore.afrom_documents(
                [
                    Document(page_content=pc1, metadata={"m": 1}),
                    Document(page_content=pc2, metadata={"m": 3}),
                ],
                ids=["idx1", "idx3"],
                collection_name=collection.name,
                token=astra_db_credentials["token"],
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
        v_store_2.clear()

        # IDs in documents.
        v_store_3 = await AstraDBVectorStore.afrom_documents(
            [
                Document(page_content=pc1, metadata={"m": 1}, id="idx1"),
                Document(page_content=pc2, metadata={"m": 3}, id="idx3"),
            ],
            collection_name=collection.name,
            token=astra_db_credentials["token"],
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
        v_store_3.clear()

        # IDs both in documents and aside.
        with pytest.warns(DeprecationWarning) as rec_warnings:
            v_store_4 = await AstraDBVectorStore.afrom_documents(
                [
                    Document(page_content=pc1, metadata={"m": 1}),
                    Document(page_content=pc2, metadata={"m": 3}, id="idy3"),
                ],
                ids=["idx1", "idx3"],
                collection_name=collection.name,
                token=astra_db_credentials["token"],
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

# HERE a different series of vs tests start

    @pytest.mark.parametrize(
        "vector_store",
        [
            "vector_store_d2",
            "vector_store_d2_stringtoken",
            "vector_store_vz",
        ],
    )
    def test_astradb_vectorstore_crud_sync(
        self, vector_store: str, request: pytest.FixtureRequest
    ) -> None:
        """Add/delete/update behaviour."""
        vstore: AstraDBVectorStore = request.getfixturevalue(vector_store)

        res0 = vstore.similarity_search("[-1,-1]", k=2)
        assert res0 == []
        # write and check again
        vstore.add_texts(
            texts=["[1,2]", "[3,4]", "[5,6]"],
            metadatas=[
                {"k": "a", "ord": 0},
                {"k": "b", "ord": 1},
                {"k": "c", "ord": 2},
            ],
            ids=["a", "b", "c"],
        )
        res1 = vstore.similarity_search("[-1,-1]", k=5)
        assert {doc.page_content for doc in res1} == {"[1,2]", "[3,4]", "[5,6]"}
        # partial overwrite and count total entries
        vstore.add_texts(
            texts=["[5,6]", "[7,8]"],
            metadatas=[
                {"k": "c_new", "ord": 102},
                {"k": "d_new", "ord": 103},
            ],
            ids=["c", "d"],
        )
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

    @pytest.mark.parametrize(
        "vector_store",
        [
            "vector_store_d2",
            "vector_store_vz",
        ],
    )
    async def test_astradb_vectorstore_crud_async(
        self, vector_store: str, request: pytest.FixtureRequest
    ) -> None:
        """Basic add/delete/update behaviour."""
        vstore: AstraDBVectorStore = request.getfixturevalue(vector_store)

        res0 = await vstore.asimilarity_search("[-1,-1]", k=2)
        assert res0 == []
        # write and check again
        await vstore.aadd_texts(
            texts=["[1,2]", "[3,4]", "[5,6]"],
            metadatas=[
                {"k": "a", "ord": 0},
                {"k": "b", "ord": 1},
                {"k": "c", "ord": 2},
            ],
            ids=["a", "b", "c"],
        )
        res1 = await vstore.asimilarity_search("[-1,-1]", k=5)
        assert {doc.page_content for doc in res1} == {"[1,2]", "[3,4]", "[5,6]"}
        # partial overwrite and count total entries
        await vstore.aadd_texts(
            texts=["[7,8]", "[9,10]"],
            metadatas=[
                {"k": "c_new", "ord": 102},
                {"k": "d_new", "ord": 103},
            ],
            ids=["c", "d"],
        )
        res2 = await vstore.asimilarity_search("[-1,-1]", k=10)
        assert len(res2) == 4
        # pick one that was just updated and check its metadata
        res3 = await vstore.asimilarity_search_with_score_id(
            query="[7,8]", k=1, filter={"k": "c_new"}
        )
        doc3, _, id3 = res3[0]
        assert doc3.page_content == "[7,8]"
        assert doc3.metadata == {"k": "c_new", "ord": 102}
        assert id3 == "c"
        # delete and count again
        del1_res = await vstore.adelete(["b"])
        assert del1_res is True
        del2_res = await vstore.adelete(["a", "c", "Z!"])
        assert del2_res is False  # a non-existing ID was supplied
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

    def test_astradb_vectorstore_massive_insert_replace_sync(
        self,
        vector_store_d2: AstraDBVectorStore,
    ) -> None:
        """Testing the insert-many-and-replace-some patterns thoroughly."""
        full_size = 300
        first_group_size = 150
        second_group_slicer = [30, 100, 2]

        all_ids = [f"doc_{idx}" for idx in range(full_size)]
        all_texts = [f"[0,{idx+1}]" for idx in range(full_size)]

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
        """Testing the insert-many-and-replace-some patterns thoroughly."""
        full_size = 300
        first_group_size = 150
        second_group_slicer = [30, 100, 2]

        all_ids = [f"doc_{idx}" for idx in range(full_size)]
        all_texts = [f"[0,{idx+1}]" for idx in range(full_size)]

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

    def test_astradb_vectorstore_mmr_sync(
        self, vector_store_d2: AstraDBVectorStore
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
        self, vector_store_d2: AstraDBVectorStore
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
        self, vector_store_vz: AstraDBVectorStore
    ) -> None:
        """MMR testing with vectorize, sync."""
        vector_store_vz.add_texts(
            [
                "Dog",
                "Wolf",
                "Ant",
                "Sunshine and piadina",
            ],
            ids=["d", "w", "a", "s"],
        )

        hits = vector_store_vz.max_marginal_relevance_search("Dingo", k=2, fetch_k=3)
        assert {doc.page_content for doc in hits} == {"Dog", "Ant"}

    async def test_astradb_vectorstore_mmr_vectorize_async(
        self, vector_store_vz: AstraDBVectorStore
    ) -> None:
        """MMR async testing with vectorize, async."""
        await vector_store_vz.aadd_texts(
            [
                "Dog",
                "Wolf",
                "Ant",
                "Sunshine and piadina",
            ],
            ids=["d", "w", "a", "s"],
        )

        hits = await vector_store_vz.amax_marginal_relevance_search(
            "Dingo",
            k=2,
            fetch_k=3,
        )
        assert {doc.page_content for doc in hits} == {"Dog", "Ant"}

    @pytest.mark.parametrize(
        "vector_store",
        [
            "vector_store_d2",
            "vector_store_vz",
        ],
    )
    def test_astradb_vectorstore_metadata(
        self, vector_store: str, request: pytest.FixtureRequest
    ) -> None:
        """Metadata filtering."""
        vstore: AstraDBVectorStore = request.getfixturevalue(vector_store)
        vstore.add_documents(
            [
                Document(
                    page_content="[1,2]",
                    metadata={"ord": ord("q"), "group": "consonant", "letter": "q"},
                ),
                Document(
                    page_content="[3,4]",
                    metadata={"ord": ord("w"), "group": "consonant", "letter": "w"},
                ),
                Document(
                    page_content="[5,6]",
                    metadata={"ord": ord("r"), "group": "consonant", "letter": "r"},
                ),
                Document(
                    page_content="[-1,2]",
                    metadata={"ord": ord("e"), "group": "vowel", "letter": "e"},
                ),
                Document(
                    page_content="[-3,4]",
                    metadata={"ord": ord("i"), "group": "vowel", "letter": "i"},
                ),
                Document(
                    page_content="[-5,6]",
                    metadata={"ord": ord("o"), "group": "vowel", "letter": "o"},
                ),
            ]
        )
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

    @pytest.mark.parametrize("vector_store", ["vector_store_d2"])
    def test_astradb_vectorstore_similarity_scale_sync(
        self, vector_store: str, request: pytest.FixtureRequest
    ) -> None:
        """Scale of the similarity scores."""
        vstore: AstraDBVectorStore = request.getfixturevalue(vector_store)
        vstore.add_texts(
            texts=[
                "[1,1]",
                "[-1,-1]",
            ],
            ids=["near", "far"],
        )
        res1 = vstore.similarity_search_with_score(
            "[0.99999,1.00001]",
            k=2,
        )
        scores = [sco for _, sco in res1]
        sco_near, sco_far = scores
        assert abs(1 - sco_near) < MATCH_EPSILON
        assert abs(sco_far) < 0.21
        assert abs(sco_far) >= 0

    @pytest.mark.parametrize("vector_store", ["vector_store_d2"])
    async def test_astradb_vectorstore_similarity_scale_async(
        self, vector_store: str, request: pytest.FixtureRequest
    ) -> None:
        """Scale of the similarity scores."""
        vstore: AstraDBVectorStore = request.getfixturevalue(vector_store)
        await vstore.aadd_texts(
            texts=[
                "[1,1]",
                "[-1,-1]",
            ],
            ids=["near", "far"],
        )
        res1 = await vstore.asimilarity_search_with_score(
            "[0.99999,1.00001]",
            k=2,
        )
        scores = [sco for _, sco in res1]
        sco_near, sco_far = scores
        assert abs(1 - sco_near) < MATCH_EPSILON
        assert abs(sco_far) < 0.21
        assert abs(sco_far) >= 0

    @pytest.mark.parametrize(
        "vector_store",
        [
            "vector_store_d2",
            "vector_store_vz",
        ],
    )
    def test_astradb_vectorstore_massive_delete(
        self, vector_store: str, request: pytest.FixtureRequest
    ) -> None:
        """Larger-scale bulk deletes."""
        vstore: AstraDBVectorStore = request.getfixturevalue(vector_store)
        m = 150
        texts = [f"[0,{i + 1 / 7.0}]" for i in range(2 * m)]
        ids0 = ["doc_%i" % i for i in range(m)]
        ids1 = ["doc_%i" % (i + m) for i in range(m)]
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

    def test_astradb_vectorstore_delete_collection(
        self, astra_db_credentials: AstraDBCredentials
    ) -> None:
        """Behaviour of 'delete_collection'."""
        collection_name = COLLECTION_NAME_DIM2
        emb = ParserEmbeddings(dimension=2)
        v_store = AstraDBVectorStore(
            embedding=emb,
            collection_name=collection_name,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        )
        v_store.add_texts(["[1,2]"])
        assert len(v_store.similarity_search("[-1,-1]", k=10)) == 1
        # another instance pointing to the same collection on DB
        v_store_kenny = AstraDBVectorStore(
            embedding=emb,
            collection_name=collection_name,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        )
        v_store_kenny.delete_collection()
        # dropped on DB, but 'v_store' should have no clue:
        with pytest.raises(ValueError, match="Collection does not exist"):
            _ = v_store.similarity_search("[-1,-1]", k=10)

    def test_astradb_vectorstore_custom_params_sync(
        self, astra_db_credentials: AstraDBCredentials
    ) -> None:
        """Custom batch size and concurrency params."""
        emb = ParserEmbeddings(dimension=2)
        # prepare empty collection
        AstraDBVectorStore(
            embedding=emb,
            collection_name=COLLECTION_NAME_DIM2,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        ).clear()
        v_store = AstraDBVectorStore(
            embedding=emb,
            collection_name=COLLECTION_NAME_DIM2,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            batch_size=17,
            bulk_insert_batch_concurrency=13,
            bulk_insert_overwrite_concurrency=7,
            bulk_delete_concurrency=19,
        )
        try:
            # add_texts
            n = 120
            texts = [f"[0,{i + 1 / 7.0}]" for i in range(n)]
            ids = ["doc_%i" % i for i in range(n)]
            v_store.add_texts(texts=texts, ids=ids)
            v_store.add_texts(
                texts=texts,
                ids=ids,
                batch_size=19,
                batch_concurrency=7,
                overwrite_concurrency=13,
            )
            _ = v_store.delete(ids[: n // 2])
            _ = v_store.delete(ids[n // 2 :], concurrency=23)
        finally:
            if not SKIP_COLLECTION_DELETE:
                v_store.delete_collection()
            else:
                v_store.clear()

    async def test_astradb_vectorstore_custom_params_async(
        self, astra_db_credentials: AstraDBCredentials
    ) -> None:
        """Custom batch size and concurrency params."""
        emb = ParserEmbeddings(dimension=2)
        v_store = AstraDBVectorStore(
            embedding=emb,
            collection_name="lc_test_c_async",
            batch_size=17,
            bulk_insert_batch_concurrency=13,
            bulk_insert_overwrite_concurrency=7,
            bulk_delete_concurrency=19,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        )
        try:
            # add_texts
            n = 120
            texts = [f"[0,{i + 1 / 7.0}]" for i in range(n)]
            ids = ["doc_%i" % i for i in range(n)]
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
        finally:
            if not SKIP_COLLECTION_DELETE:
                await v_store.adelete_collection()
            else:
                await v_store.aclear()

    def test_astradb_vectorstore_metrics(
        self, astra_db_credentials: AstraDBCredentials
    ) -> None:
        """Different choices of similarity metric.
        Both stores (with "cosine" and "euclidea" metrics) contain these two:
            - a vector slightly rotated w.r.t query vector
            - a vector which is a long multiple of query vector
        so, which one is "the closest one" depends on the metric.
        """
        emb = ParserEmbeddings(dimension=2)
        isq2 = 0.5**0.5
        isa = 0.7
        isb = (1.0 - isa * isa) ** 0.5
        texts = [
            json.dumps([isa, isb]),
            json.dumps([10 * isq2, 10 * isq2]),
        ]
        ids = [
            "rotated",
            "scaled",
        ]
        query_text = json.dumps([isq2, isq2])

        # prepare empty collections
        AstraDBVectorStore(
            embedding=emb,
            collection_name=COLLECTION_NAME_DIM2,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        ).clear()
        AstraDBVectorStore(
            embedding=emb,
            collection_name=COLLECTION_NAME_DIM2_EUCLIDEAN,
            metric="euclidean",
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        ).clear()

        # creation, population, query - cosine
        vstore_cos = AstraDBVectorStore(
            embedding=emb,
            collection_name=COLLECTION_NAME_DIM2,
            metric="cosine",
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        )
        try:
            vstore_cos.add_texts(
                texts=texts,
                ids=ids,
            )
            _, _, id_from_cos = vstore_cos.similarity_search_with_score_id(
                query_text,
                k=1,
            )[0]
            assert id_from_cos == "scaled"
        finally:
            if not SKIP_COLLECTION_DELETE:
                vstore_cos.delete_collection()
            else:
                vstore_cos.clear()
        # creation, population, query - euclidean

        vstore_euc = AstraDBVectorStore(
            embedding=emb,
            collection_name=COLLECTION_NAME_DIM2_EUCLIDEAN,
            metric="euclidean",
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        )
        try:
            vstore_euc.add_texts(
                texts=texts,
                ids=ids,
            )
            _, _, id_from_euc = vstore_euc.similarity_search_with_score_id(
                query_text,
                k=1,
            )[0]
            assert id_from_euc == "rotated"
        finally:
            if not SKIP_COLLECTION_DELETE:
                vstore_euc.delete_collection()
            else:
                vstore_euc.clear()

    def test_astradb_vectorstore_indexing_sync(
        self,
        astra_db_credentials: dict[str, str | None],
        database: Database,
    ) -> None:
        """Test that the right errors/warnings are issued depending
        on the compatibility of on-DB indexing settings and the requested ones.

        We do NOT check for substrings in the warning messages: that would
        be too brittle a test.
        """
        embe = ParserEmbeddings(dimension=2)

        # creation of three collections to test warnings against
        database.create_collection("lc_legacy_coll", dimension=2, metric=None)
        AstraDBVectorStore(
            collection_name="lc_default_idx",
            embedding=embe,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        )
        AstraDBVectorStore(
            collection_name="lc_custom_idx",
            embedding=embe,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            metadata_indexing_exclude={"long_summary", "the_divine_comedy"},
        )

        # these invocations should just work without warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            AstraDBVectorStore(
                collection_name="lc_default_idx",
                embedding=embe,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
            )
            AstraDBVectorStore(
                collection_name="lc_custom_idx",
                embedding=embe,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                metadata_indexing_exclude={"long_summary", "the_divine_comedy"},
            )

        # some are to throw an error:
        with pytest.raises(ValueError, match=INCOMPATIBLE_INDEXING_MSG):
            AstraDBVectorStore(
                collection_name="lc_default_idx",
                embedding=embe,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                metadata_indexing_exclude={"long_summary", "the_divine_comedy"},
            )

        with pytest.raises(ValueError, match=INCOMPATIBLE_INDEXING_MSG):
            AstraDBVectorStore(
                collection_name="lc_custom_idx",
                embedding=embe,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                metadata_indexing_exclude={"changed_fields"},
            )

        with pytest.raises(ValueError, match=INCOMPATIBLE_INDEXING_MSG):
            AstraDBVectorStore(
                collection_name="lc_custom_idx",
                embedding=embe,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
            )

        with pytest.raises(
            ValueError,
            match="Astra DB collection 'lc_legacy_coll' is detected as having "
            "indexing turned on for all fields",
        ):
            AstraDBVectorStore(
                collection_name="lc_legacy_coll",
                embedding=embe,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                metadata_indexing_exclude={"long_summary", "the_divine_comedy"},
            )

        # one case should result in just a warning:
        with pytest.warns(UserWarning) as rec_warnings:
            AstraDBVectorStore(
                collection_name="lc_legacy_coll",
                embedding=embe,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
            )
            f_rec_warnings = [
                wrn for wrn in rec_warnings if issubclass(wrn.category, UserWarning)
            ]
            assert len(f_rec_warnings) == 1

        # cleanup
        database.drop_collection("lc_legacy_coll")
        database.drop_collection("lc_default_idx")
        database.drop_collection("lc_custom_idx")

    async def test_astradb_vectorstore_indexing_async(
        self,
        astra_db_credentials: dict[str, str | None],
        database: Database,
    ) -> None:
        """Async version of the same test on warnings/errors related
        to incompatible indexing choices.
        """
        embe = ParserEmbeddings(dimension=2)

        # creation of three collections to test warnings against
        database.create_collection("lc_legacy_coll", dimension=2, metric=None)
        await AstraDBVectorStore(
            collection_name="lc_default_idx",
            embedding=embe,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            setup_mode=SetupMode.ASYNC,
        ).asimilarity_search("[-1,-1]")
        await AstraDBVectorStore(
            collection_name="lc_custom_idx",
            embedding=embe,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            metadata_indexing_exclude={"long_summary", "the_divine_comedy"},
            setup_mode=SetupMode.ASYNC,
        ).asimilarity_search("[-1,-1]")

        # these invocations should just work without warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            def_store = AstraDBVectorStore(
                collection_name="lc_default_idx",
                embedding=embe,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                setup_mode=SetupMode.ASYNC,
            )
            await def_store.aadd_texts(["[1,2]"])
            cus_store = AstraDBVectorStore(
                collection_name="lc_custom_idx",
                embedding=embe,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                metadata_indexing_exclude={"long_summary", "the_divine_comedy"},
                setup_mode=SetupMode.ASYNC,
            )
            await cus_store.aadd_texts(["[1,2]"])

        # some are to throw an error:
        def_store = AstraDBVectorStore(
            collection_name="lc_default_idx",
            embedding=embe,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            metadata_indexing_exclude={"long_summary", "the_divine_comedy"},
            setup_mode=SetupMode.ASYNC,
        )
        with pytest.raises(ValueError, match=INCOMPATIBLE_INDEXING_MSG):
            await def_store.aadd_texts(["[9,8]"])

        cus_store = AstraDBVectorStore(
            collection_name="lc_custom_idx",
            embedding=embe,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            metadata_indexing_exclude={"changed_fields"},
            setup_mode=SetupMode.ASYNC,
        )
        with pytest.raises(ValueError, match=INCOMPATIBLE_INDEXING_MSG):
            await cus_store.aadd_texts(["[9,8]"])

        cus_store = AstraDBVectorStore(
            collection_name="lc_custom_idx",
            embedding=embe,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            setup_mode=SetupMode.ASYNC,
        )
        with pytest.raises(ValueError, match=INCOMPATIBLE_INDEXING_MSG):
            await cus_store.aadd_texts(["[9,8]"])

        leg_store = AstraDBVectorStore(
            collection_name="lc_legacy_coll",
            embedding=embe,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            metadata_indexing_exclude={"long_summary", "the_divine_comedy"},
            setup_mode=SetupMode.ASYNC,
        )
        with pytest.raises(
            ValueError,
            match="Astra DB collection 'lc_legacy_coll' is detected as having "
            "indexing turned on for all fields",
        ):
            await leg_store.aadd_texts(["[9,8]"])

        # one case should result in just a warning:
        with pytest.warns(UserWarning) as rec_warnings:
            leg_store = AstraDBVectorStore(
                collection_name="lc_legacy_coll",
                embedding=embe,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                setup_mode=SetupMode.ASYNC,
            )
            await leg_store.aadd_texts(["[6,6]"])
            # cleaning out 'spurious' "unclosed socket/transport..." warnings
            f_rec_warnings = [
                wrn for wrn in rec_warnings if issubclass(wrn.category, UserWarning)
            ]
            assert len(f_rec_warnings) == 1

        await database.to_async().drop_collection("lc_legacy_coll")
        await database.to_async().drop_collection("lc_default_idx")
        await database.to_async().drop_collection("lc_custom_idx")

    @pytest.mark.skipif(
        os.environ.get("ASTRA_DB_ENVIRONMENT", "prod").upper() != "PROD",
        reason="Can run on Astra DB prod only",
    )
    def test_astradb_vectorstore_coreclients_init_sync(
        self,
        astra_db_credentials: dict[str, str | None],
        core_astra_db: AstraDB,
    ) -> None:
        """A deprecation warning from passing a (core) AstraDB, but it works."""
        collection_name = "lc_test_vstore_coreclsync"
        emb = ParserEmbeddings(dimension=2)

        try:
            v_store_init_ok = AstraDBVectorStore(
                embedding=emb,
                collection_name=collection_name,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
            )
            v_store_init_ok.add_texts(["[1,2]"])

            with pytest.warns(DeprecationWarning) as rec_warnings:
                v_store_init_core = AstraDBVectorStore(
                    embedding=emb,
                    collection_name=collection_name,
                    astra_db_client=core_astra_db,
                )

            results = v_store_init_core.similarity_search("[-1,-1]", k=1)
            # cleaning out 'spurious' "unclosed socket/transport..." warnings
            f_rec_warnings = [
                wrn
                for wrn in rec_warnings
                if issubclass(wrn.category, DeprecationWarning)
            ]
            assert len(f_rec_warnings) == 1
            assert len(results) == 1
            assert results[0].page_content == "[1,2]"
        finally:
            if not SKIP_COLLECTION_DELETE:
                v_store_init_ok.delete_collection()
            else:
                v_store_init_ok.clear()

    @pytest.mark.skipif(
        os.environ.get("ASTRA_DB_ENVIRONMENT", "prod").upper() != "PROD",
        reason="Can run on Astra DB prod only",
    )
    async def test_astradb_vectorstore_coreclients_init_async(
        self,
        astra_db_credentials: dict[str, str | None],
        core_astra_db: AstraDB,
    ) -> None:
        """A deprecation warning from passing a (core) AstraDB, but it works."""
        collection_name = "lc_test_vstore_coreclasync"
        emb = ParserEmbeddings(dimension=2)

        try:
            v_store_init_ok = AstraDBVectorStore(
                embedding=emb,
                collection_name=collection_name,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                setup_mode=SetupMode.ASYNC,
            )
            await v_store_init_ok.aadd_texts(["[1,2]"])

            with pytest.warns(DeprecationWarning) as rec_warnings:
                v_store_init_core = AstraDBVectorStore(
                    embedding=emb,
                    collection_name=collection_name,
                    astra_db_client=core_astra_db,
                    setup_mode=SetupMode.ASYNC,
                )

            results = await v_store_init_core.asimilarity_search("[-1,-1]", k=1)
            f_rec_warnings = [
                wrn
                for wrn in rec_warnings
                if issubclass(wrn.category, DeprecationWarning)
            ]
            assert len(f_rec_warnings) == 1
            assert len(results) == 1
            assert results[0].page_content == "[1,2]"
        finally:
            if not SKIP_COLLECTION_DELETE:
                await v_store_init_ok.adelete_collection()
            else:
                await v_store_init_ok.aclear()
