"""
Test of Astra DB vector store class `AstraDBVectorStore`

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
    - optionally to test vectorize with nvidia as well (besides openai):
        export NVIDIA_VECTORIZE_AVAILABLE="1"
    - optionally:
        export ASTRA_DB_SKIP_COLLECTION_DELETIONS="0" ("1" = no deletions, default)
"""

import json
import math
import os
import warnings
from typing import Dict, Iterable, Optional

import pytest
from astrapy import Database
from astrapy.db import AstraDB
from astrapy.info import CollectionVectorServiceOptions
from langchain_core.documents import Document

from langchain_astradb.utils.astradb import SetupMode
from langchain_astradb.vectorstores import AstraDBVectorStore

from ..conftest import ParserEmbeddings, SomeEmbeddings
from .conftest import AstraDBCredentials, _has_env_vars

# Faster testing (no actual collection deletions). Off by default (=full tests)
SKIP_COLLECTION_DELETE = (
    int(os.environ.get("ASTRA_DB_SKIP_COLLECTION_DELETIONS", "0")) != 0
)

COLLECTION_NAME_DIM2 = "lc_test_d2"
COLLECTION_NAME_DIM2_EUCLIDEAN = "lc_test_d2_eucl"
COLLECTION_NAME_VECTORIZE_OPENAI = "lc_test_vec_openai"
COLLECTION_NAME_VECTORIZE_OPENAI_HEADER = "lc_test_vec_openai_h"
COLLECTION_NAME_VECTORIZE_NVIDIA = "lc_test_nvidia"

MATCH_EPSILON = 0.0001


openai_vectorize_options = CollectionVectorServiceOptions(
    provider="openai",
    model_name="text-embedding-3-small",
    authentication={
        "providerKey": f"{os.environ.get('SHARED_SECRET_NAME_OPENAI', '')}",
    },
)
openai_vectorize_options_header = CollectionVectorServiceOptions(
    provider="openai",
    model_name="text-embedding-3-small",
)
nvidia_vectorize_options = CollectionVectorServiceOptions(
    provider="nvidia",
    model_name="NV-Embed-QA",
)


def is_nvidia_vector_service_available() -> bool:
    # For the time being, this is manually controlled
    if os.environ.get("NVIDIA_VECTORIZE_AVAILABLE"):
        try:
            return int(os.environ["NVIDIA_VECTORIZE_AVAILABLE"]) != 0
        except Exception:
            return False
    else:
        return False


@pytest.fixture(scope="function")
def store_someemb(
    astra_db_credentials: AstraDBCredentials,
) -> Iterable[AstraDBVectorStore]:
    emb = SomeEmbeddings(dimension=2)
    v_store = AstraDBVectorStore(
        embedding=emb,
        collection_name=COLLECTION_NAME_DIM2,
        token=astra_db_credentials["token"],
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
    )
    v_store.clear()

    yield v_store

    if not SKIP_COLLECTION_DELETE:
        v_store.delete_collection()
    else:
        v_store.clear()


@pytest.fixture(scope="function")
def store_parseremb(
    astra_db_credentials: AstraDBCredentials,
) -> Iterable[AstraDBVectorStore]:
    emb = ParserEmbeddings(dimension=2)
    v_store = AstraDBVectorStore(
        embedding=emb,
        collection_name=COLLECTION_NAME_DIM2,
        token=astra_db_credentials["token"],
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
    )
    v_store.clear()

    yield v_store

    if not SKIP_COLLECTION_DELETE:
        v_store.delete_collection()
    else:
        v_store.clear()


@pytest.fixture(scope="function")
def vectorize_store(
    astra_db_credentials: AstraDBCredentials,
) -> Iterable[AstraDBVectorStore]:
    """
    astra db vector store with server-side embeddings using openai + shared_secret
    """
    v_store = AstraDBVectorStore(
        collection_vector_service_options=openai_vectorize_options,
        collection_name=COLLECTION_NAME_VECTORIZE_OPENAI,
        token=astra_db_credentials["token"],
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
    )
    v_store.clear()

    yield v_store

    # explicitly delete the collection to avoid max collection limit
    v_store.delete_collection()


@pytest.fixture(scope="function")
def vectorize_store_w_header(
    astra_db_credentials: AstraDBCredentials,
) -> Iterable[AstraDBVectorStore]:
    """
    astra db vector store with server-side embeddings using openai + header
    """
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OpenAI key not available")

    v_store = AstraDBVectorStore(
        collection_vector_service_options=openai_vectorize_options_header,
        collection_name=COLLECTION_NAME_VECTORIZE_OPENAI_HEADER,
        collection_embedding_api_key=os.environ["OPENAI_API_KEY"],
        token=astra_db_credentials["token"],
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
    )
    v_store.clear()

    yield v_store

    # explicitly delete the collection to avoid max collection limit
    v_store.delete_collection()


@pytest.fixture(scope="function")
def vectorize_store_nvidia(
    astra_db_credentials: AstraDBCredentials,
) -> Iterable[AstraDBVectorStore]:
    """
    astra db vector store with server-side embeddings using the nvidia model
    """
    if not is_nvidia_vector_service_available():
        pytest.skip("vectorize/nvidia unavailable")

    v_store = AstraDBVectorStore(
        collection_vector_service_options=nvidia_vectorize_options,
        collection_name=COLLECTION_NAME_VECTORIZE_NVIDIA,
        token=astra_db_credentials["token"],
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
    )
    v_store.clear()

    yield v_store

    # explicitly delete the collection to avoid max collection limit
    v_store.delete_collection()


@pytest.mark.skipif(not _has_env_vars(), reason="Missing Astra DB env. vars")
class TestAstraDBVectorStore:
    def test_astradb_vectorstore_create_delete_sync(
        self, astra_db_credentials: AstraDBCredentials
    ) -> None:
        """Create and delete."""

        emb = SomeEmbeddings(dimension=2)

        v_store = AstraDBVectorStore(
            embedding=emb,
            collection_name=COLLECTION_NAME_DIM2,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        )
        v_store.add_texts("Sample 1")
        if not SKIP_COLLECTION_DELETE:
            v_store.delete_collection()
        else:
            v_store.clear()

    def test_astradb_vectorstore_create_delete_vectorize_sync(
        self, astra_db_credentials: AstraDBCredentials
    ) -> None:
        """Create and delete with vectorize option."""
        v_store = AstraDBVectorStore(
            collection_vector_service_options=openai_vectorize_options,
            collection_name=COLLECTION_NAME_VECTORIZE_OPENAI,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        )
        v_store.add_texts(["Sample 1"])
        v_store.delete_collection()

    async def test_astradb_vectorstore_create_delete_async(
        self, astra_db_credentials: AstraDBCredentials
    ) -> None:
        """Create and delete."""
        emb = SomeEmbeddings(dimension=2)
        # Creation by passing the connection secrets
        v_store = AstraDBVectorStore(
            embedding=emb,
            collection_name=COLLECTION_NAME_DIM2,
            **astra_db_credentials,
        )
        await v_store.adelete_collection()

    async def test_astradb_vectorstore_create_delete_vectorize_async(
        self, astra_db_credentials: AstraDBCredentials
    ) -> None:
        """Create and delete with vectorize option."""
        v_store = AstraDBVectorStore(
            collection_vector_service_options=openai_vectorize_options,
            collection_name=COLLECTION_NAME_VECTORIZE_OPENAI,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        )
        await v_store.adelete_collection()

    @pytest.mark.skipif(
        SKIP_COLLECTION_DELETE,
        reason="Collection-deletion tests are suppressed",
    )
    def test_astradb_vectorstore_pre_delete_collection_sync(
        self, astra_db_credentials: AstraDBCredentials
    ) -> None:
        """Use of the pre_delete_collection flag."""
        emb = SomeEmbeddings(dimension=2)
        v_store = AstraDBVectorStore(
            embedding=emb,
            collection_name=COLLECTION_NAME_DIM2,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        )
        v_store.clear()
        try:
            v_store.add_texts(
                texts=["aa"],
                metadatas=[
                    {"k": "a", "ord": 0},
                ],
                ids=["a"],
            )
            res1 = v_store.similarity_search("aa", k=5)
            assert len(res1) == 1
            v_store = AstraDBVectorStore(
                embedding=emb,
                pre_delete_collection=True,
                collection_name=COLLECTION_NAME_DIM2,
                **astra_db_credentials,
            )
            res1 = v_store.similarity_search("aa", k=5)
            assert len(res1) == 0
        finally:
            v_store.delete_collection()

    @pytest.mark.skipif(
        SKIP_COLLECTION_DELETE,
        reason="Collection-deletion tests are suppressed",
    )
    async def test_astradb_vectorstore_pre_delete_collection_async(
        self, astra_db_credentials: AstraDBCredentials
    ) -> None:
        """Use of the pre_delete_collection flag."""
        emb = SomeEmbeddings(dimension=2)
        # creation by passing the connection secrets

        v_store = AstraDBVectorStore(
            embedding=emb,
            collection_name=COLLECTION_NAME_DIM2,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        )
        try:
            await v_store.aadd_texts(
                texts=["aa"],
                metadatas=[
                    {"k": "a", "ord": 0},
                ],
                ids=["a"],
            )
            res1 = await v_store.asimilarity_search("aa", k=5)
            assert len(res1) == 1
            v_store = AstraDBVectorStore(
                embedding=emb,
                pre_delete_collection=True,
                collection_name=COLLECTION_NAME_DIM2,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
            )
            res1 = await v_store.asimilarity_search("aa", k=5)
            assert len(res1) == 0
        finally:
            await v_store.adelete_collection()

    def test_astradb_vectorstore_from_x_sync(
        self, astra_db_credentials: AstraDBCredentials
    ) -> None:
        """from_texts and from_documents methods."""
        emb = SomeEmbeddings(dimension=2)
        # prepare empty collection
        AstraDBVectorStore(
            embedding=emb,
            collection_name=COLLECTION_NAME_DIM2,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        ).clear()
        # from_texts
        v_store = AstraDBVectorStore.from_texts(
            texts=["Hi", "Ho"],
            embedding=emb,
            collection_name=COLLECTION_NAME_DIM2,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        )
        try:
            assert v_store.similarity_search("Ho", k=1)[0].page_content == "Ho"
        finally:
            if not SKIP_COLLECTION_DELETE:
                v_store.delete_collection()
            else:
                v_store.clear()

        # from_documents
        v_store_2 = AstraDBVectorStore.from_documents(
            [
                Document(page_content="Hee"),
                Document(page_content="Hoi"),
            ],
            embedding=emb,
            collection_name=COLLECTION_NAME_DIM2,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        )
        try:
            assert v_store_2.similarity_search("Hoi", k=1)[0].page_content == "Hoi"
        finally:
            if not SKIP_COLLECTION_DELETE:
                v_store_2.delete_collection()
            else:
                v_store_2.clear()

    def test_astradb_vectorstore_from_x_vectorize_sync(
        self, astra_db_credentials: AstraDBCredentials
    ) -> None:
        """from_texts and from_documents methods with vectorize."""
        AstraDBVectorStore(
            collection_vector_service_options=openai_vectorize_options,
            collection_name=COLLECTION_NAME_VECTORIZE_OPENAI,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        ).clear()

        # from_texts
        v_store = AstraDBVectorStore.from_texts(
            texts=["Hi", "Ho"],
            collection_vector_service_options=openai_vectorize_options,
            collection_name=COLLECTION_NAME_VECTORIZE_OPENAI,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        )
        try:
            assert v_store.similarity_search("Ho", k=1)[0].page_content == "Ho"
        finally:
            v_store.delete_collection()

        # from_documents
        v_store_2 = AstraDBVectorStore.from_documents(
            [
                Document(page_content="Hee"),
                Document(page_content="Hoi"),
            ],
            collection_vector_service_options=openai_vectorize_options,
            collection_name=COLLECTION_NAME_VECTORIZE_OPENAI,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        )
        try:
            assert v_store_2.similarity_search("Hoi", k=1)[0].page_content == "Hoi"
        finally:
            v_store_2.delete_collection()

    async def test_astradb_vectorstore_from_x_async(
        self, astra_db_credentials: AstraDBCredentials
    ) -> None:
        """from_texts and from_documents methods."""
        emb = SomeEmbeddings(dimension=2)
        # prepare empty collection
        await AstraDBVectorStore(
            embedding=emb,
            collection_name=COLLECTION_NAME_DIM2,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        ).aclear()
        # from_texts
        v_store = await AstraDBVectorStore.afrom_texts(
            texts=["Hi", "Ho"],
            embedding=emb,
            collection_name=COLLECTION_NAME_DIM2,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        )
        try:
            assert (await v_store.asimilarity_search("Ho", k=1))[0].page_content == "Ho"
        finally:
            if not SKIP_COLLECTION_DELETE:
                await v_store.adelete_collection()
            else:
                await v_store.aclear()

        # from_documents
        v_store_2 = await AstraDBVectorStore.afrom_documents(
            [
                Document(page_content="Hee"),
                Document(page_content="Hoi"),
            ],
            embedding=emb,
            collection_name=COLLECTION_NAME_DIM2,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        )
        try:
            assert (await v_store_2.asimilarity_search("Hoi", k=1))[
                0
            ].page_content == "Hoi"
        finally:
            if not SKIP_COLLECTION_DELETE:
                await v_store_2.adelete_collection()
            else:
                await v_store_2.aclear()

    async def test_astradb_vectorstore_from_x_vectorize_async(
        self, astra_db_credentials: AstraDBCredentials
    ) -> None:
        """from_texts and from_documents methods with vectorize."""
        # from_text with vectorize
        v_store = await AstraDBVectorStore.afrom_texts(
            texts=["Haa", "Huu"],
            collection_vector_service_options=openai_vectorize_options,
            collection_name=COLLECTION_NAME_VECTORIZE_OPENAI,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        )
        try:
            assert (await v_store.asimilarity_search("Haa", k=1))[
                0
            ].page_content == "Haa"
        finally:
            await v_store.adelete_collection()

        # from_documents with vectorize
        v_store_2 = await AstraDBVectorStore.afrom_documents(
            [
                Document(page_content="HeeH"),
                Document(page_content="HooH"),
            ],
            collection_vector_service_options=openai_vectorize_options,
            collection_name=COLLECTION_NAME_VECTORIZE_OPENAI,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        )
        try:
            assert (await v_store_2.asimilarity_search("HeeH", k=1))[
                0
            ].page_content == "HeeH"
        finally:
            await v_store_2.adelete_collection()

    @pytest.mark.parametrize(
        "vector_store",
        [
            "store_someemb",
            "vectorize_store",
            "vectorize_store_w_header",
            "vectorize_store_nvidia",
        ],
    )
    def test_astradb_vectorstore_crud_sync(
        self, vector_store: str, request: pytest.FixtureRequest
    ) -> None:
        """Basic add/delete/update behaviour."""
        vstore: AstraDBVectorStore = request.getfixturevalue(vector_store)

        res0 = vstore.similarity_search("Abc", k=2)
        assert res0 == []
        # write and check again
        vstore.add_texts(
            texts=["aa", "bb", "cc"],
            metadatas=[
                {"k": "a", "ord": 0},
                {"k": "b", "ord": 1},
                {"k": "c", "ord": 2},
            ],
            ids=["a", "b", "c"],
        )
        res1 = vstore.similarity_search("Abc", k=5)
        assert {doc.page_content for doc in res1} == {"aa", "bb", "cc"}
        # partial overwrite and count total entries
        vstore.add_texts(
            texts=["cc", "dd"],
            metadatas=[
                {"k": "c_new", "ord": 102},
                {"k": "d_new", "ord": 103},
            ],
            ids=["c", "d"],
        )
        res2 = vstore.similarity_search("Abc", k=10)
        assert len(res2) == 4
        # pick one that was just updated and check its metadata
        res3 = vstore.similarity_search_with_score_id(
            query="cc", k=1, filter={"k": "c_new"}
        )
        doc3, _, id3 = res3[0]
        assert doc3.page_content == "cc"
        assert doc3.metadata == {"k": "c_new", "ord": 102}
        assert id3 == "c"
        # delete and count again
        del1_res = vstore.delete(["b"])
        assert del1_res is True
        del2_res = vstore.delete(["a", "c", "Z!"])
        assert del2_res is True  # a non-existing ID was supplied
        assert len(vstore.similarity_search("xy", k=10)) == 1
        # clear store
        vstore.clear()
        assert vstore.similarity_search("Abc", k=2) == []
        # add_documents with "ids" arg passthrough
        vstore.add_documents(
            [
                Document(page_content="vv", metadata={"k": "v", "ord": 204}),
                Document(page_content="ww", metadata={"k": "w", "ord": 205}),
            ],
            ids=["v", "w"],
        )
        assert len(vstore.similarity_search("xy", k=10)) == 2
        res4 = vstore.similarity_search("ww", k=1, filter={"k": "w"})
        assert res4[0].metadata["ord"] == 205

    @pytest.mark.parametrize(
        "vector_store",
        [
            "store_someemb",
            "vectorize_store",
            "vectorize_store_w_header",
            "vectorize_store_nvidia",
        ],
    )
    async def test_astradb_vectorstore_crud_async(
        self, vector_store: str, request: pytest.FixtureRequest
    ) -> None:
        """Basic add/delete/update behaviour."""
        vstore: AstraDBVectorStore = request.getfixturevalue(vector_store)

        res0 = await vstore.asimilarity_search("Abc", k=2)
        assert res0 == []
        # write and check again
        await vstore.aadd_texts(
            texts=["aa", "bb", "cc"],
            metadatas=[
                {"k": "a", "ord": 0},
                {"k": "b", "ord": 1},
                {"k": "c", "ord": 2},
            ],
            ids=["a", "b", "c"],
        )
        res1 = await vstore.asimilarity_search("Abc", k=5)
        assert {doc.page_content for doc in res1} == {"aa", "bb", "cc"}
        # partial overwrite and count total entries
        await vstore.aadd_texts(
            texts=["cc", "dd"],
            metadatas=[
                {"k": "c_new", "ord": 102},
                {"k": "d_new", "ord": 103},
            ],
            ids=["c", "d"],
        )
        res2 = await vstore.asimilarity_search("Abc", k=10)
        assert len(res2) == 4
        # pick one that was just updated and check its metadata
        res3 = await vstore.asimilarity_search_with_score_id(
            query="cc", k=1, filter={"k": "c_new"}
        )
        doc3, _, id3 = res3[0]
        assert doc3.page_content == "cc"
        assert doc3.metadata == {"k": "c_new", "ord": 102}
        assert id3 == "c"
        # delete and count again
        del1_res = await vstore.adelete(["b"])
        assert del1_res is True
        del2_res = await vstore.adelete(["a", "c", "Z!"])
        assert del2_res is False  # a non-existing ID was supplied
        assert len(await vstore.asimilarity_search("xy", k=10)) == 1
        # clear store
        await vstore.aclear()
        assert await vstore.asimilarity_search("Abc", k=2) == []
        # add_documents with "ids" arg passthrough
        await vstore.aadd_documents(
            [
                Document(page_content="vv", metadata={"k": "v", "ord": 204}),
                Document(page_content="ww", metadata={"k": "w", "ord": 205}),
            ],
            ids=["v", "w"],
        )
        assert len(await vstore.asimilarity_search("xy", k=10)) == 2
        res4 = await vstore.asimilarity_search("ww", k=1, filter={"k": "w"})
        assert res4[0].metadata["ord"] == 205

    def test_astradb_vectorstore_massive_insert_replace_sync(
        self,
        store_someemb: AstraDBVectorStore,
    ) -> None:
        """Testing the insert-many-and-replace-some patterns thoroughly."""
        FULL_SIZE = 300
        FIRST_GROUP_SIZE = 150
        SECOND_GROUP_SLICER = [30, 100, 2]

        all_ids = [f"doc_{idx}" for idx in range(FULL_SIZE)]
        all_texts = [f"document number {idx}" for idx in range(FULL_SIZE)]

        # massive insertion on empty
        group0_ids = all_ids[0:FIRST_GROUP_SIZE]
        group0_texts = all_texts[0:FIRST_GROUP_SIZE]
        inserted_ids0 = store_someemb.add_texts(
            texts=group0_texts,
            ids=group0_ids,
        )
        assert set(inserted_ids0) == set(group0_ids)
        # massive insertion with many overwrites scattered through
        # (we change the text to later check on DB for successful update)
        _s, _e, _st = SECOND_GROUP_SLICER
        group1_ids = all_ids[_s:_e:_st] + all_ids[FIRST_GROUP_SIZE:FULL_SIZE]
        group1_texts = [
            txt.upper()
            for txt in (all_texts[_s:_e:_st] + all_texts[FIRST_GROUP_SIZE:FULL_SIZE])
        ]
        inserted_ids1 = store_someemb.add_texts(
            texts=group1_texts,
            ids=group1_ids,
        )
        assert set(inserted_ids1) == set(group1_ids)
        # final read (we want the IDs to do a full check)
        expected_text_by_id = {
            **{d_id: d_tx for d_id, d_tx in zip(group0_ids, group0_texts)},
            **{d_id: d_tx for d_id, d_tx in zip(group1_ids, group1_texts)},
        }
        full_results = store_someemb.similarity_search_with_score_id_by_vector(
            embedding=[1.0, 1.0],
            k=FULL_SIZE,
        )
        for doc, _, doc_id in full_results:
            assert doc.page_content == expected_text_by_id[doc_id]

    async def test_astradb_vectorstore_massive_insert_replace_async(
        self,
        store_someemb: AstraDBVectorStore,
    ) -> None:
        """Testing the insert-many-and-replace-some patterns thoroughly."""
        FULL_SIZE = 300
        FIRST_GROUP_SIZE = 150
        SECOND_GROUP_SLICER = [30, 100, 2]

        all_ids = [f"doc_{idx}" for idx in range(FULL_SIZE)]
        all_texts = [f"document number {idx}" for idx in range(FULL_SIZE)]

        # massive insertion on empty
        group0_ids = all_ids[0:FIRST_GROUP_SIZE]
        group0_texts = all_texts[0:FIRST_GROUP_SIZE]

        inserted_ids0 = await store_someemb.aadd_texts(
            texts=group0_texts,
            ids=group0_ids,
        )
        assert set(inserted_ids0) == set(group0_ids)
        # massive insertion with many overwrites scattered through
        # (we change the text to later check on DB for successful update)
        _s, _e, _st = SECOND_GROUP_SLICER
        group1_ids = all_ids[_s:_e:_st] + all_ids[FIRST_GROUP_SIZE:FULL_SIZE]
        group1_texts = [
            txt.upper()
            for txt in (all_texts[_s:_e:_st] + all_texts[FIRST_GROUP_SIZE:FULL_SIZE])
        ]
        inserted_ids1 = await store_someemb.aadd_texts(
            texts=group1_texts,
            ids=group1_ids,
        )
        assert set(inserted_ids1) == set(group1_ids)
        # final read (we want the IDs to do a full check)
        expected_text_by_id = {
            **{d_id: d_tx for d_id, d_tx in zip(group0_ids, group0_texts)},
            **{d_id: d_tx for d_id, d_tx in zip(group1_ids, group1_texts)},
        }
        full_results = await store_someemb.asimilarity_search_with_score_id_by_vector(
            embedding=[1.0, 1.0],
            k=FULL_SIZE,
        )
        for doc, _, doc_id in full_results:
            assert doc.page_content == expected_text_by_id[doc_id]

    def test_astradb_vectorstore_mmr_sync(
        self, store_parseremb: AstraDBVectorStore
    ) -> None:
        """
        MMR testing. We work on the unit circle with angle multiples
        of 2*pi/20 and prepare a store with known vectors for a controlled
        MMR outcome.
        """

        def _v_from_i(i: int, N: int) -> str:
            angle = 2 * math.pi * i / N
            vector = [math.cos(angle), math.sin(angle)]
            return json.dumps(vector)

        i_vals = [0, 4, 5, 13]
        N_val = 20
        store_parseremb.add_texts(
            [_v_from_i(i, N_val) for i in i_vals], metadatas=[{"i": i} for i in i_vals]
        )
        res1 = store_parseremb.max_marginal_relevance_search(
            _v_from_i(3, N_val),
            k=2,
            fetch_k=3,
        )
        res_i_vals = {doc.metadata["i"] for doc in res1}
        assert res_i_vals == {0, 4}

    async def test_astradb_vectorstore_mmr_async(
        self, store_parseremb: AstraDBVectorStore
    ) -> None:
        """
        MMR testing. We work on the unit circle with angle multiples
        of 2*pi/20 and prepare a store with known vectors for a controlled
        MMR outcome.
        """

        def _v_from_i(i: int, N: int) -> str:
            angle = 2 * math.pi * i / N
            vector = [math.cos(angle), math.sin(angle)]
            return json.dumps(vector)

        i_vals = [0, 4, 5, 13]
        N_val = 20
        await store_parseremb.aadd_texts(
            [_v_from_i(i, N_val) for i in i_vals],
            metadatas=[{"i": i} for i in i_vals],
        )
        res1 = await store_parseremb.amax_marginal_relevance_search(
            _v_from_i(3, N_val),
            k=2,
            fetch_k=3,
        )
        res_i_vals = {doc.metadata["i"] for doc in res1}
        assert res_i_vals == {0, 4}

    def test_astradb_vectorstore_mmr_vectorize_sync(
        self, vectorize_store: AstraDBVectorStore
    ) -> None:
        """
        MMR testing with vectorize, sync.
        """
        vectorize_store.add_texts(
            [
                "Dog",
                "Wolf",
                "Ant",
                "Sunshine and piadina",
            ],
            ids=["d", "w", "a", "s"],
        )

        hits = vectorize_store.max_marginal_relevance_search("Dingo", k=2, fetch_k=3)
        assert {doc.page_content for doc in hits} == {"Dog", "Ant"}

    async def test_astradb_vectorstore_mmr_vectorize_async(
        self, vectorize_store: AstraDBVectorStore
    ) -> None:
        """
        MMR async testing with vectorize, async.
        """
        await vectorize_store.aadd_texts(
            [
                "Dog",
                "Wolf",
                "Ant",
                "Sunshine and piadina",
            ],
            ids=["d", "w", "a", "s"],
        )

        hits = await vectorize_store.amax_marginal_relevance_search(
            "Dingo",
            k=2,
            fetch_k=3,
        )
        assert {doc.page_content for doc in hits} == {"Dog", "Ant"}

    @pytest.mark.parametrize(
        "vector_store",
        [
            "store_someemb",
            "vectorize_store",
            "vectorize_store_w_header",
            "vectorize_store_nvidia",
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
                    page_content="q",
                    metadata={"ord": ord("q"), "group": "consonant"},
                ),
                Document(
                    page_content="w",
                    metadata={"ord": ord("w"), "group": "consonant"},
                ),
                Document(
                    page_content="r",
                    metadata={"ord": ord("r"), "group": "consonant"},
                ),
                Document(
                    page_content="e",
                    metadata={"ord": ord("e"), "group": "vowel"},
                ),
                Document(
                    page_content="i",
                    metadata={"ord": ord("i"), "group": "vowel"},
                ),
                Document(
                    page_content="o",
                    metadata={"ord": ord("o"), "group": "vowel"},
                ),
            ]
        )
        # no filters
        res0 = vstore.similarity_search("x", k=10)
        assert {doc.page_content for doc in res0} == set("qwreio")
        # single filter
        res1 = vstore.similarity_search(
            "x",
            k=10,
            filter={"group": "vowel"},
        )
        assert {doc.page_content for doc in res1} == set("eio")
        # multiple filters
        res2 = vstore.similarity_search(
            "x",
            k=10,
            filter={"group": "consonant", "ord": ord("q")},
        )
        assert {doc.page_content for doc in res2} == set("q")
        # excessive filters
        res3 = vstore.similarity_search(
            "x",
            k=10,
            filter={"group": "consonant", "ord": ord("q"), "case": "upper"},
        )
        assert res3 == []
        # filter with logical operator
        res4 = vstore.similarity_search(
            "x",
            k=10,
            filter={"$or": [{"ord": ord("q")}, {"ord": ord("r")}]},
        )
        assert {doc.page_content for doc in res4} == {"q", "r"}

    @pytest.mark.parametrize("vector_store", ["store_parseremb"])
    def test_astradb_vectorstore_similarity_scale_sync(
        self, vector_store: str, request: pytest.FixtureRequest
    ) -> None:
        """Scale of the similarity scores."""
        vstore: AstraDBVectorStore = request.getfixturevalue(vector_store)
        vstore.add_texts(
            texts=[
                json.dumps([1, 1]),
                json.dumps([-1, -1]),
            ],
            ids=["near", "far"],
        )
        res1 = vstore.similarity_search_with_score(
            json.dumps([0.5, 0.5]),
            k=2,
        )
        scores = [sco for _, sco in res1]
        sco_near, sco_far = scores
        assert abs(1 - sco_near) < MATCH_EPSILON and abs(sco_far) < MATCH_EPSILON

    @pytest.mark.parametrize("vector_store", ["store_parseremb"])
    async def test_astradb_vectorstore_similarity_scale_async(
        self, vector_store: str, request: pytest.FixtureRequest
    ) -> None:
        """Scale of the similarity scores."""
        vstore: AstraDBVectorStore = request.getfixturevalue(vector_store)
        await vstore.aadd_texts(
            texts=[
                json.dumps([1, 1]),
                json.dumps([-1, -1]),
            ],
            ids=["near", "far"],
        )
        res1 = await vstore.asimilarity_search_with_score(
            json.dumps([0.5, 0.5]),
            k=2,
        )
        scores = [sco for _, sco in res1]
        sco_near, sco_far = scores
        assert abs(1 - sco_near) < MATCH_EPSILON and abs(sco_far) < MATCH_EPSILON

    @pytest.mark.parametrize(
        "vector_store",
        [
            "store_someemb",
            "vectorize_store",
            "vectorize_store_w_header",
            "vectorize_store_nvidia",
        ],
    )
    def test_astradb_vectorstore_massive_delete(
        self, vector_store: str, request: pytest.FixtureRequest
    ) -> None:
        """Larger-scale bulk deletes."""
        vstore: AstraDBVectorStore = request.getfixturevalue(vector_store)
        M = 150
        texts = [str(i + 1 / 7.0) for i in range(2 * M)]
        ids0 = ["doc_%i" % i for i in range(M)]
        ids1 = ["doc_%i" % (i + M) for i in range(M)]
        ids = ids0 + ids1
        vstore.add_texts(texts=texts, ids=ids)
        # deleting a bunch of these
        del_res0 = vstore.delete(ids0)
        assert del_res0 is True
        # deleting the rest plus a fake one
        del_res1 = vstore.delete(ids1 + ["ghost!"])
        assert del_res1 is True  # ensure no error
        # nothing left
        assert vstore.similarity_search("x", k=2 * M) == []

    @pytest.mark.skipif(
        SKIP_COLLECTION_DELETE,
        reason="Collection-deletion tests are suppressed",
    )
    def test_astradb_vectorstore_delete_collection(
        self, astra_db_credentials: AstraDBCredentials
    ) -> None:
        """behaviour of 'delete_collection'."""
        collection_name = COLLECTION_NAME_DIM2
        emb = SomeEmbeddings(dimension=2)
        v_store = AstraDBVectorStore(
            embedding=emb,
            collection_name=collection_name,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        )
        v_store.add_texts(["huh"])
        assert len(v_store.similarity_search("hah", k=10)) == 1
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
        with pytest.raises(ValueError):
            _ = v_store.similarity_search("hah", k=10)

    def test_astradb_vectorstore_custom_params_sync(
        self, astra_db_credentials: AstraDBCredentials
    ) -> None:
        """Custom batch size and concurrency params."""
        emb = SomeEmbeddings(dimension=2)
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
            N = 50
            texts = [str(i + 1 / 7.0) for i in range(N)]
            ids = ["doc_%i" % i for i in range(N)]
            v_store.add_texts(texts=texts, ids=ids)
            v_store.add_texts(
                texts=texts,
                ids=ids,
                batch_size=19,
                batch_concurrency=7,
                overwrite_concurrency=13,
            )
            #
            _ = v_store.delete(ids[: N // 2])
            _ = v_store.delete(ids[N // 2 :], concurrency=23)
            #
        finally:
            if not SKIP_COLLECTION_DELETE:
                v_store.delete_collection()
            else:
                v_store.clear()

    async def test_astradb_vectorstore_custom_params_async(
        self, astra_db_credentials: AstraDBCredentials
    ) -> None:
        """Custom batch size and concurrency params."""
        emb = SomeEmbeddings(dimension=2)
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
            N = 50
            texts = [str(i + 1 / 7.0) for i in range(N)]
            ids = ["doc_%i" % i for i in range(N)]
            await v_store.aadd_texts(texts=texts, ids=ids)
            await v_store.aadd_texts(
                texts=texts,
                ids=ids,
                batch_size=19,
                batch_concurrency=7,
                overwrite_concurrency=13,
            )
            #
            await v_store.adelete(ids[: N // 2])
            await v_store.adelete(ids[N // 2 :], concurrency=23)
            #
        finally:
            if not SKIP_COLLECTION_DELETE:
                await v_store.adelete_collection()
            else:
                await v_store.aclear()

    def test_astradb_vectorstore_metrics(
        self, astra_db_credentials: AstraDBCredentials
    ) -> None:
        """
        Different choices of similarity metric.
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
        astra_db_credentials: Dict[str, Optional[str]],
        database: Database,
    ) -> None:
        """
        Test that the right errors/warnings are issued depending
        on the compatibility of on-DB indexing settings and the requested ones.

        We do NOT check for substrings in the warning messages: that would
        be too brittle a test.
        """
        embe = SomeEmbeddings(dimension=2)

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
        with pytest.raises(ValueError):
            AstraDBVectorStore(
                collection_name="lc_default_idx",
                embedding=embe,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                metadata_indexing_exclude={"long_summary", "the_divine_comedy"},
            )

        with pytest.raises(ValueError):
            AstraDBVectorStore(
                collection_name="lc_custom_idx",
                embedding=embe,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                metadata_indexing_exclude={"changed_fields"},
            )

        with pytest.raises(ValueError):
            AstraDBVectorStore(
                collection_name="lc_custom_idx",
                embedding=embe,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
            )

        with pytest.raises(ValueError):
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
            assert len(rec_warnings) == 1

        # cleanup
        database.drop_collection("lc_legacy_coll")
        database.drop_collection("lc_default_idx")
        database.drop_collection("lc_custom_idx")

    async def test_astradb_vectorstore_indexing_async(
        self,
        astra_db_credentials: Dict[str, Optional[str]],
        database: Database,
    ) -> None:
        """
        Async version of the same test on warnings/errors related
        to incompatible indexing choices.
        """
        embe = SomeEmbeddings(dimension=2)

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
        ).asimilarity_search("boo")
        await AstraDBVectorStore(
            collection_name="lc_custom_idx",
            embedding=embe,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            metadata_indexing_exclude={"long_summary", "the_divine_comedy"},
            setup_mode=SetupMode.ASYNC,
        ).asimilarity_search("boo")

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
            await def_store.aadd_texts(["All good."])
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
            await cus_store.aadd_texts(["All good."])

        # some are to throw an error:
        with pytest.raises(ValueError):
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
            await def_store.aadd_texts(["Not working."])

        with pytest.raises(ValueError):
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
            await cus_store.aadd_texts(["Not working."])

        with pytest.raises(ValueError):
            cus_store = AstraDBVectorStore(
                collection_name="lc_custom_idx",
                embedding=embe,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                setup_mode=SetupMode.ASYNC,
            )
            await cus_store.aadd_texts(["Not working."])

        with pytest.raises(ValueError):
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
            await leg_store.aadd_texts(["Not working."])

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
            await leg_store.aadd_texts(["Triggering warning."])
        assert len(rec_warnings) == 1

        await database.to_async().drop_collection("lc_legacy_coll")
        await database.to_async().drop_collection("lc_default_idx")
        await database.to_async().drop_collection("lc_custom_idx")

    @pytest.mark.skipif(
        os.environ.get("ASTRA_DB_ENVIRONMENT", "prod").upper() != "PROD",
        reason="Can run on Astra DB prod only",
    )
    def test_astradb_vectorstore_coreclients_init_sync(
        self,
        astra_db_credentials: Dict[str, Optional[str]],
        core_astra_db: AstraDB,
    ) -> None:
        """A deprecation warning from passing a (core) AstraDB, but it works."""

        collection_name = "lc_test_vstore_coreclsync"
        emb = SomeEmbeddings(dimension=2)

        try:
            v_store_init_ok = AstraDBVectorStore(
                embedding=emb,
                collection_name=collection_name,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
            )
            v_store_init_ok.add_texts(["One text"])

            with pytest.warns(DeprecationWarning) as rec_warnings:
                v_store_init_core = AstraDBVectorStore(
                    embedding=emb,
                    collection_name=collection_name,
                    astra_db_client=core_astra_db,
                )

            results = v_store_init_core.similarity_search("another", k=1)
            assert len(rec_warnings) == 1
            assert len(results) == 1
            assert results[0].page_content == "One text"
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
        astra_db_credentials: Dict[str, Optional[str]],
        core_astra_db: AstraDB,
    ) -> None:
        """A deprecation warning from passing a (core) AstraDB, but it works."""

        collection_name = "lc_test_vstore_coreclasync"
        emb = SomeEmbeddings(dimension=2)

        try:
            v_store_init_ok = AstraDBVectorStore(
                embedding=emb,
                collection_name=collection_name,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                setup_mode=SetupMode.ASYNC,
            )
            await v_store_init_ok.aadd_texts(["One text"])

            with pytest.warns(DeprecationWarning) as rec_warnings:
                v_store_init_core = AstraDBVectorStore(
                    embedding=emb,
                    collection_name=collection_name,
                    astra_db_client=core_astra_db,
                    setup_mode=SetupMode.ASYNC,
                )

            results = await v_store_init_core.asimilarity_search("another", k=1)
            assert len(rec_warnings) == 1
            assert len(results) == 1
            assert results[0].page_content == "One text"
        finally:
            if not SKIP_COLLECTION_DELETE:
                await v_store_init_ok.adelete_collection()
            else:
                await v_store_init_ok.aclear()
