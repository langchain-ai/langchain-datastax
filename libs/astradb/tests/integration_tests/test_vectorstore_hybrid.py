"""Test of Astra DB vector store class `AstraDBVectorStore`, hybrid-workflows."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
from astrapy.exceptions import DataAPIResponseException
from astrapy.info import CollectionLexicalOptions, CollectionRerankOptions
from langchain_core.documents import Document

from langchain_astradb.utils.astradb import HybridSearchMode, SetupMode
from langchain_astradb.utils.vector_store_codecs import (
    _DefaultVectorizeVSDocumentCodec,
    _DefaultVSDocumentCodec,
)
from langchain_astradb.vectorstores import (
    AstraDBVectorStore,
    HybridLimitFactorPrescription,
)

from .conftest import (
    LEXICAL_OPTIONS,
    NVIDIA_RERANKING_OPTIONS_HEADER,
    OPENAI_VECTORIZE_OPTIONS_HEADER,
    astra_db_env_vars_available,
)

if TYPE_CHECKING:
    from astrapy import Database
    from langchain_core.embeddings import Embeddings

    from .conftest import AstraDBCredentials


COLLECTION_NAME_VECTORIZE = "lc_vstore_hybrid_vectorize"
COLLECTION_NAME_NOVECTORIZE = "lc_vstore_hybrid_novectorize"
COLLECTION_NAME_NOHYBRID_NOVECTORIZE = "lc_vstore_nohybrid_novectorize"
QUERY_TEXT = "need a number?"
LEXICAL_QUERY_TEXT = "number counting units"
QUERY_TEXT_NOVECTORIZE = "[1,0]"


@pytest.fixture
def documents() -> list[Document]:
    """Documents for nominal insertions"""
    return [
        Document(
            id="doc_00",
            page_content="number zero",
            metadata={"tag": "00", "is_document": True},
        ),
        Document(
            id="doc_01",
            page_content="number one",
            metadata={"tag": "01", "is_document": True},
        ),
        Document(
            id="doc_02",
            page_content="number two",
            metadata={"tag": "02", "is_document": True},
        ),
    ]


@pytest.fixture
def documents2() -> list[Document]:
    """Documents for another nominal insertions"""
    return [
        Document(
            id="doc_03",
            page_content="number three",
            metadata={"tag": "03", "is_document": True, "first_batch": "no"},
        ),
    ]


@pytest.fixture
def documents_novectorize() -> list[Document]:
    """Documents for nominal insertions"""
    return [
        Document(
            id="doc_00",
            page_content="[0,1]",
            metadata={"tag": "00", "is_document": True},
        ),
        Document(
            id="doc_01",
            page_content="[1,1]",
            metadata={"tag": "01", "is_document": True},
        ),
        Document(
            id="doc_02",
            page_content="[2,1]",
            metadata={"tag": "02", "is_document": True},
        ),
    ]


@pytest.fixture
def documents2_novectorize() -> list[Document]:
    """Documents for another nominal insertions"""
    return [
        Document(
            id="doc_03",
            page_content="[3,1]",
            metadata={"tag": "03", "is_document": True, "first_batch": "no"},
        ),
    ]


@pytest.mark.skipif(
    not astra_db_env_vars_available(), reason="Missing Astra DB env. vars"
)
class TestAstraDBVectorStoreHybrid:
    def test_astradb_vectorstore_explicit_hybrid_lifecycle_vectorize_sync(
        self,
        *,
        astra_db_credentials: AstraDBCredentials,
        database: Database,
        openai_api_key: str,
        nvidia_reranking_api_key: str | None,
        documents: list[Document],
        documents2: list[Document],
    ) -> None:
        # Hybrid search coll.config is explicit ==> run hyb search is automatic.
        try:
            # create vstore ( => actual collection creation)
            store0 = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_VECTORIZE,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                collection_vector_service_options=OPENAI_VECTORIZE_OPTIONS_HEADER,
                collection_embedding_api_key=openai_api_key,
                collection_rerank=NVIDIA_RERANKING_OPTIONS_HEADER,
                collection_reranking_api_key=nvidia_reranking_api_key,
                collection_lexical=LEXICAL_OPTIONS,
            )
            # verify it would run hybrid
            assert store0.hybrid_search
            # insert items, check they get $lexical on DB
            store0.add_documents(documents)
            assert all(
                "$lexical" in doc
                for doc in store0.astra_env.collection.find(
                    limit=10, projection={"*": True}
                )
            )
            # run a 'search' (trusting it to be hybrid), some checks on the results
            hits_triples = store0.similarity_search_with_score_id(QUERY_TEXT, k=2)
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("number")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)

            # another search with a metadata filter on top
            hits_triples_b = store0.similarity_search_with_score_id(
                QUERY_TEXT,
                k=2,
                filter={"tag": "01"},
            )
            assert len(hits_triples_b) == 1
            assert hits_triples_b[0][0].page_content == "number one"

            # another search with a different lexical_query
            hits_triples_lq = store0.similarity_search_with_score_id(
                QUERY_TEXT,
                k=2,
                lexical_query=LEXICAL_QUERY_TEXT,
            )
            assert len(hits_triples_lq) == 2
            rdoc_lq, rscore_lq, rid_lq = hits_triples_lq[0]
            assert rdoc_lq.page_content.startswith("number")
            assert isinstance(rdoc_lq.page_content, str)
            assert rscore_lq > -100
            assert rscore_lq < 100
            assert isinstance(rid_lq, str)

            # re-instantiate just like above, re-check
            store1 = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_VECTORIZE,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                collection_vector_service_options=OPENAI_VECTORIZE_OPTIONS_HEADER,
                collection_embedding_api_key=openai_api_key,
                collection_rerank=NVIDIA_RERANKING_OPTIONS_HEADER,
                collection_reranking_api_key=nvidia_reranking_api_key,
                collection_lexical=LEXICAL_OPTIONS,
            )
            assert store1.hybrid_search
            hits_triples = store1.similarity_search_with_score_id(QUERY_TEXT, k=2)
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("number")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)

            # autodetect instantiation (no other changes)
            store2_ad = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_VECTORIZE,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                collection_embedding_api_key=openai_api_key,
                collection_reranking_api_key=nvidia_reranking_api_key,
                autodetect_collection=True,
            )
            # check it runs hybrid search
            assert store2_ad.hybrid_search
            # check the right codec is selected
            assert isinstance(
                store2_ad.document_codec,
                _DefaultVectorizeVSDocumentCodec,
            )
            assert not store2_ad.document_codec.ignore_invalid_documents
            assert store2_ad.document_codec.has_lexical
            # run a 'search' (trusting it to be hybrid), some checks on the results
            hits_triples = store2_ad.similarity_search_with_score_id(QUERY_TEXT, k=2)
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("number")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)
            # run other search methods
            with pytest.raises(ValueError, match="not allowed"):
                store2_ad.similarity_search_by_vector([1, 2, 3])
            mmr_hits_docs = store2_ad.max_marginal_relevance_search(
                query=QUERY_TEXT,
                k=1,
            )
            assert len(mmr_hits_docs) == 1
            mmr_doc = mmr_hits_docs[0]
            assert mmr_doc.page_content.startswith("number")
            assert isinstance(mmr_doc.page_content, str)
            assert isinstance(mmr_doc.id, str)

            # autodetect instantiation #2 (nondefault hybrid_limits)
            store3_ad = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_VECTORIZE,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                collection_embedding_api_key=openai_api_key,
                collection_reranking_api_key=nvidia_reranking_api_key,
                autodetect_collection=True,
                hybrid_limit_factor=math.pi,
            )
            # run a 'search' (trusting it to be hybrid), some checks on the results
            hits_triples = store3_ad.similarity_search_with_score_id(QUERY_TEXT, k=2)
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("number")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)

            # autodetect instantiation #4 (split hybrid_limits, struct)
            store3b_ad = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_VECTORIZE,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                collection_embedding_api_key=openai_api_key,
                collection_reranking_api_key=nvidia_reranking_api_key,
                autodetect_collection=True,
                hybrid_limit_factor=HybridLimitFactorPrescription(
                    vector=3.2, lexical=1.8
                ),
            )
            # run a 'search' (trusting it to be hybrid), some checks on the results
            hits_triples = store3b_ad.similarity_search_with_score_id(QUERY_TEXT, k=2)
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("number")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)

            # autodetect instantiation #5 (split hybrid_limits, dict)
            store3c_ad = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_VECTORIZE,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                collection_embedding_api_key=openai_api_key,
                collection_reranking_api_key=nvidia_reranking_api_key,
                autodetect_collection=True,
                hybrid_limit_factor={"$vector": 3.2, "$lexical": 1.8},
            )
            # run a 'search' (trusting it to be hybrid), some checks on the results
            hits_triples = store3c_ad.similarity_search_with_score_id(QUERY_TEXT, k=2)
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("number")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)

            # instantiate explicitly, disabling hybrid in searching
            store4 = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_VECTORIZE,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                collection_vector_service_options=OPENAI_VECTORIZE_OPTIONS_HEADER,
                collection_embedding_api_key=openai_api_key,
                collection_rerank=NVIDIA_RERANKING_OPTIONS_HEADER,
                collection_reranking_api_key=nvidia_reranking_api_key,
                collection_lexical=LEXICAL_OPTIONS,
                hybrid_search=HybridSearchMode.OFF,
            )
            # check it's not doing hybrid
            assert not store4.hybrid_search
            # insert more documents, ensure writes respect $lexical nevertheless
            store4.add_documents(documents2)
            assert all(
                "$lexical" in doc
                for doc in store4.astra_env.collection.find(
                    limit=10, projection={"*": True}
                )
            )
            # run similarity search (expecting regular ANN to be done)
            hits_triples = store4.similarity_search_with_score_id(QUERY_TEXT, k=2)
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("number")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)
            # run other search methods
            with pytest.raises(ValueError, match="not allowed"):
                store4.similarity_search_by_vector([1, 2, 3])
            mmr_hits_docs = store4.max_marginal_relevance_search(
                query=QUERY_TEXT,
                k=1,
            )
            assert len(mmr_hits_docs) == 1
            mmr_doc = mmr_hits_docs[0]
            assert mmr_doc.page_content.startswith("number")
            assert isinstance(mmr_doc.page_content, str)
            assert isinstance(mmr_doc.id, str)

            # regular instantiation, wrong rerank api key -> error on search
            store5 = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_VECTORIZE,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                collection_vector_service_options=OPENAI_VECTORIZE_OPTIONS_HEADER,
                collection_embedding_api_key=openai_api_key,
                collection_rerank=NVIDIA_RERANKING_OPTIONS_HEADER,
                collection_reranking_api_key="bababa-wrong",
                collection_lexical=LEXICAL_OPTIONS,
            )
            with pytest.raises(DataAPIResponseException):
                store5.similarity_search(QUERY_TEXT)

            # autodetect instantiation, wrong rerank api key -> error on search
            store6_ad = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_VECTORIZE,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                collection_embedding_api_key=openai_api_key,
                collection_reranking_api_key="bababa-wrong",
                autodetect_collection=True,
            )
            with pytest.raises(DataAPIResponseException):
                store6_ad.similarity_search(QUERY_TEXT)
        finally:
            database.drop_collection(COLLECTION_NAME_VECTORIZE)

    async def test_astradb_vectorstore_explicit_hybrid_lifecycle_vectorize_async(
        self,
        *,
        astra_db_credentials: AstraDBCredentials,
        database: Database,
        openai_api_key: str,
        nvidia_reranking_api_key: str | None,
        documents: list[Document],
        documents2: list[Document],
    ) -> None:
        # Hybrid search coll.config is explicit ==> run hyb search is automatic.
        try:
            # create vstore ( => actual collection creation)
            store0 = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_VECTORIZE,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                setup_mode=SetupMode.ASYNC,
                environment=astra_db_credentials["environment"],
                collection_vector_service_options=OPENAI_VECTORIZE_OPTIONS_HEADER,
                collection_embedding_api_key=openai_api_key,
                collection_rerank=NVIDIA_RERANKING_OPTIONS_HEADER,
                collection_reranking_api_key=nvidia_reranking_api_key,
                collection_lexical=LEXICAL_OPTIONS,
            )
            # verify it would run hybrid
            assert store0.hybrid_search
            # insert items, check they get $lexical on DB
            await store0.aadd_documents(documents)
            assert all(
                "$lexical" in doc
                for doc in (
                    await store0.astra_env.async_collection.find(
                        limit=10, projection={"*": True}
                    ).to_list()
                )
            )
            # run a 'search' (trusting it to be hybrid), some checks on the results
            hits_triples = await store0.asimilarity_search_with_score_id(
                QUERY_TEXT, k=2
            )
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("number")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)

            # another search with a metadata filter on top
            hits_triples_b = await store0.asimilarity_search_with_score_id(
                QUERY_TEXT,
                k=2,
                filter={"tag": "01"},
            )
            assert len(hits_triples_b) == 1
            assert hits_triples_b[0][0].page_content == "number one"

            # another search with a different lexical_query
            hits_triples_lq = await store0.asimilarity_search_with_score_id(
                QUERY_TEXT,
                k=2,
                lexical_query=LEXICAL_QUERY_TEXT,
            )
            assert len(hits_triples_lq) == 2
            rdoc_lq, rscore_lq, rid_lq = hits_triples_lq[0]
            assert rdoc_lq.page_content.startswith("number")
            assert isinstance(rdoc_lq.page_content, str)
            assert rscore_lq > -100
            assert rscore_lq < 100
            assert isinstance(rid_lq, str)

            # re-instantiate just like above, re-check
            store1 = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_VECTORIZE,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                setup_mode=SetupMode.ASYNC,
                environment=astra_db_credentials["environment"],
                collection_vector_service_options=OPENAI_VECTORIZE_OPTIONS_HEADER,
                collection_embedding_api_key=openai_api_key,
                collection_rerank=NVIDIA_RERANKING_OPTIONS_HEADER,
                collection_reranking_api_key=nvidia_reranking_api_key,
                collection_lexical=LEXICAL_OPTIONS,
            )
            assert store1.hybrid_search
            hits_triples = await store1.asimilarity_search_with_score_id(
                QUERY_TEXT, k=2
            )
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("number")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)

            # re-instantiate again with nonstandard hybrid limits
            store1b = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_VECTORIZE,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                setup_mode=SetupMode.ASYNC,
                environment=astra_db_credentials["environment"],
                collection_vector_service_options=OPENAI_VECTORIZE_OPTIONS_HEADER,
                collection_embedding_api_key=openai_api_key,
                collection_rerank=NVIDIA_RERANKING_OPTIONS_HEADER,
                collection_reranking_api_key=nvidia_reranking_api_key,
                collection_lexical=LEXICAL_OPTIONS,
                hybrid_limit_factor=math.pi,
            )
            assert store1b.hybrid_search
            hits_triples = await store1b.asimilarity_search_with_score_id(
                QUERY_TEXT, k=2
            )
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("number")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)

            # re-instantiate again with split hybrid limits (struct)
            store1c = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_VECTORIZE,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                setup_mode=SetupMode.ASYNC,
                environment=astra_db_credentials["environment"],
                collection_vector_service_options=OPENAI_VECTORIZE_OPTIONS_HEADER,
                collection_embedding_api_key=openai_api_key,
                collection_rerank=NVIDIA_RERANKING_OPTIONS_HEADER,
                collection_reranking_api_key=nvidia_reranking_api_key,
                collection_lexical=LEXICAL_OPTIONS,
                hybrid_limit_factor=HybridLimitFactorPrescription(
                    vector=3.2,
                    lexical=1.8,
                ),
            )
            assert store1c.hybrid_search
            hits_triples = await store1c.asimilarity_search_with_score_id(
                QUERY_TEXT, k=2
            )
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("number")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)

            # re-instantiate again with split hybrid limits (dict)
            store1d = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_VECTORIZE,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                setup_mode=SetupMode.ASYNC,
                environment=astra_db_credentials["environment"],
                collection_vector_service_options=OPENAI_VECTORIZE_OPTIONS_HEADER,
                collection_embedding_api_key=openai_api_key,
                collection_rerank=NVIDIA_RERANKING_OPTIONS_HEADER,
                collection_reranking_api_key=nvidia_reranking_api_key,
                collection_lexical=LEXICAL_OPTIONS,
                hybrid_limit_factor={"$vector": 3.2, "$lexical": 1.8},
            )
            assert store1d.hybrid_search
            hits_triples = await store1d.asimilarity_search_with_score_id(
                QUERY_TEXT, k=2
            )
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("number")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)

            # instantiate explicitly, disabling hybrid in searching
            store4 = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_VECTORIZE,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                setup_mode=SetupMode.ASYNC,
                collection_vector_service_options=OPENAI_VECTORIZE_OPTIONS_HEADER,
                collection_embedding_api_key=openai_api_key,
                collection_rerank=NVIDIA_RERANKING_OPTIONS_HEADER,
                collection_reranking_api_key=nvidia_reranking_api_key,
                collection_lexical=LEXICAL_OPTIONS,
                hybrid_search=HybridSearchMode.OFF,
            )
            # check it's not doing hybrid
            assert not store4.hybrid_search
            # insert more documents, ensure writes respect $lexical nevertheless
            await store4.aadd_documents(documents2)
            assert all(
                "$lexical" in doc
                for doc in (
                    await store0.astra_env.async_collection.find(
                        limit=10, projection={"*": True}
                    ).to_list()
                )
            )
            # run similarity search (expecting regular ANN to be done)
            hits_triples = await store4.asimilarity_search_with_score_id(
                QUERY_TEXT, k=2
            )
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("number")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)
            # run other search methods
            with pytest.raises(ValueError, match="not allowed"):
                await store4.asimilarity_search_by_vector([1, 2, 3])
            mmr_hits_docs = await store4.amax_marginal_relevance_search(
                query=QUERY_TEXT,
                k=1,
            )
            assert len(mmr_hits_docs) == 1
            mmr_doc = mmr_hits_docs[0]
            assert mmr_doc.page_content.startswith("number")
            assert isinstance(mmr_doc.page_content, str)
            assert isinstance(mmr_doc.id, str)

            # regular instantiation, wrong rerank api key -> error on search
            store5 = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_VECTORIZE,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                setup_mode=SetupMode.ASYNC,
                collection_vector_service_options=OPENAI_VECTORIZE_OPTIONS_HEADER,
                collection_embedding_api_key=openai_api_key,
                collection_rerank=NVIDIA_RERANKING_OPTIONS_HEADER,
                collection_lexical=LEXICAL_OPTIONS,
                collection_reranking_api_key="bababa-wrong",
            )
            with pytest.raises(DataAPIResponseException):
                await store5.asimilarity_search(QUERY_TEXT)

        finally:
            await database.to_async().drop_collection(COLLECTION_NAME_VECTORIZE)

    def test_astradb_vectorstore_explicit_hybrid_lifecycle_novectorize_sync(
        self,
        *,
        astra_db_credentials: AstraDBCredentials,
        database: Database,
        embedding_d2: Embeddings,
        nvidia_reranking_api_key: str | None,
        documents_novectorize: list[Document],
        documents2_novectorize: list[Document],
    ) -> None:
        # Hybrid search coll.config is explicit ==> run hyb search is automatic.
        try:
            # create vstore ( => actual collection creation)
            store0 = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_NOVECTORIZE,
                embedding=embedding_d2,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                collection_rerank=NVIDIA_RERANKING_OPTIONS_HEADER,
                collection_reranking_api_key=nvidia_reranking_api_key,
                collection_lexical=LEXICAL_OPTIONS,
            )
            # verify it would run hybrid
            assert store0.hybrid_search
            # insert items, check they get $lexical on DB
            store0.add_documents(documents_novectorize)
            assert all(
                "$lexical" in doc
                for doc in store0.astra_env.collection.find(
                    limit=10, projection={"*": True}
                )
            )
            # run a 'search' (trusting it to be hybrid), some checks on the results
            hits_triples = store0.similarity_search_with_score_id(
                QUERY_TEXT_NOVECTORIZE, k=2
            )
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("[")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)

            # another search with a metadata filter on top
            hits_triples_b = store0.similarity_search_with_score_id(
                QUERY_TEXT_NOVECTORIZE,
                k=2,
                filter={"tag": "01"},
            )
            assert len(hits_triples_b) == 1
            assert hits_triples_b[0][0].page_content == "[1,1]"

            # another search with a different lexical_query
            hits_triples_lq = store0.similarity_search_with_score_id(
                QUERY_TEXT_NOVECTORIZE,
                k=2,
                lexical_query=LEXICAL_QUERY_TEXT,
            )
            assert len(hits_triples_lq) == 2
            rdoc_lq, rscore_lq, rid_lq = hits_triples_lq[0]
            assert rdoc_lq.page_content.startswith("[")
            assert isinstance(rdoc_lq.page_content, str)
            assert rscore_lq > -100
            assert rscore_lq < 100
            assert isinstance(rid_lq, str)

            # re-instantiate just like above, re-check
            store1 = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_NOVECTORIZE,
                embedding=embedding_d2,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                collection_rerank=NVIDIA_RERANKING_OPTIONS_HEADER,
                collection_reranking_api_key=nvidia_reranking_api_key,
                collection_lexical=LEXICAL_OPTIONS,
            )
            assert store1.hybrid_search
            hits_triples = store1.similarity_search_with_score_id(
                QUERY_TEXT_NOVECTORIZE, k=2
            )
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("[")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)

            # autodetect instantiation (no other changes)
            store2_ad = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_NOVECTORIZE,
                embedding=embedding_d2,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                collection_reranking_api_key=nvidia_reranking_api_key,
                autodetect_collection=True,
            )
            # check it runs hybrid search
            assert store2_ad.hybrid_search
            # check the right codec is selected
            assert isinstance(
                store2_ad.document_codec,
                _DefaultVSDocumentCodec,
            )
            assert not store2_ad.document_codec.ignore_invalid_documents
            assert store2_ad.document_codec.has_lexical
            # run a 'search' (trusting it to be hybrid), some checks on the results
            hits_triples = store2_ad.similarity_search_with_score_id(
                QUERY_TEXT_NOVECTORIZE, k=2
            )
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("[")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)
            # run other search methods
            store2_ad.similarity_search_by_vector([1, 2])
            mmr_hits_docs = store2_ad.max_marginal_relevance_search(
                query=QUERY_TEXT_NOVECTORIZE,
                k=1,
            )
            assert len(mmr_hits_docs) == 1
            mmr_doc = mmr_hits_docs[0]
            assert mmr_doc.page_content.startswith("[")
            assert isinstance(mmr_doc.page_content, str)
            assert isinstance(mmr_doc.id, str)

            # autodetect instantiation #2 (nondefault hybrid_limits)
            store3_ad = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_NOVECTORIZE,
                embedding=embedding_d2,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                collection_reranking_api_key=nvidia_reranking_api_key,
                autodetect_collection=True,
                hybrid_limit_factor=math.pi,
            )
            # run a 'search' (trusting it to be hybrid), some checks on the results
            hits_triples = store3_ad.similarity_search_with_score_id(
                QUERY_TEXT_NOVECTORIZE, k=2
            )
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("[")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)

            # autodetect instantiation #4 (split hybrid_limits, struct)
            store3b_ad = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_NOVECTORIZE,
                embedding=embedding_d2,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                collection_reranking_api_key=nvidia_reranking_api_key,
                autodetect_collection=True,
                hybrid_limit_factor=HybridLimitFactorPrescription(
                    vector=3.2, lexical=1.8
                ),
            )
            # run a 'search' (trusting it to be hybrid), some checks on the results
            hits_triples = store3b_ad.similarity_search_with_score_id(
                QUERY_TEXT_NOVECTORIZE, k=2
            )
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("[")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)

            # autodetect instantiation #5 (split hybrid_limits, dict)
            store3c_ad = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_NOVECTORIZE,
                embedding=embedding_d2,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                collection_reranking_api_key=nvidia_reranking_api_key,
                autodetect_collection=True,
                hybrid_limit_factor={"$vector": 3.2, "$lexical": 1.8},
            )
            # run a 'search' (trusting it to be hybrid), some checks on the results
            hits_triples = store3c_ad.similarity_search_with_score_id(
                QUERY_TEXT_NOVECTORIZE, k=2
            )
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("[")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)

            # instantiate explicitly, disabling hybrid in searching
            store4 = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_NOVECTORIZE,
                embedding=embedding_d2,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                collection_rerank=NVIDIA_RERANKING_OPTIONS_HEADER,
                collection_reranking_api_key=nvidia_reranking_api_key,
                collection_lexical=LEXICAL_OPTIONS,
                hybrid_search=HybridSearchMode.OFF,
            )
            # check it's not doing hybrid
            assert not store4.hybrid_search
            # insert more documents, ensure writes respect $lexical nevertheless
            store4.add_documents(documents2_novectorize)
            assert all(
                "$lexical" in doc
                for doc in store4.astra_env.collection.find(
                    limit=10, projection={"*": True}
                )
            )
            # run similarity search (expecting regular ANN to be done)
            hits_triples = store4.similarity_search_with_score_id(
                QUERY_TEXT_NOVECTORIZE, k=2
            )
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("[")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)
            # run other search methods
            store4.similarity_search_by_vector([1, 2])
            mmr_hits_docs = store4.max_marginal_relevance_search(
                query=QUERY_TEXT_NOVECTORIZE,
                k=1,
            )
            assert len(mmr_hits_docs) == 1
            mmr_doc = mmr_hits_docs[0]
            assert mmr_doc.page_content.startswith("[")
            assert isinstance(mmr_doc.page_content, str)
            assert isinstance(mmr_doc.id, str)

            # regular instantiation, wrong rerank api key -> error on search
            store5 = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_NOVECTORIZE,
                embedding=embedding_d2,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                collection_rerank=NVIDIA_RERANKING_OPTIONS_HEADER,
                collection_lexical=LEXICAL_OPTIONS,
                collection_reranking_api_key="bababa-wrong",
            )
            with pytest.raises(DataAPIResponseException):
                store5.similarity_search(QUERY_TEXT_NOVECTORIZE)

            # autodetect instantiation, wrong rerank api key -> error on search
            store6_ad = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_NOVECTORIZE,
                embedding=embedding_d2,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                autodetect_collection=True,
                collection_reranking_api_key="bababa-wrong",
            )
            with pytest.raises(DataAPIResponseException):
                store6_ad.similarity_search(QUERY_TEXT_NOVECTORIZE)
        finally:
            database.drop_collection(COLLECTION_NAME_NOVECTORIZE)

    async def test_astradb_vectorstore_explicit_hybrid_lifecycle_novectorize_async(
        self,
        *,
        astra_db_credentials: AstraDBCredentials,
        database: Database,
        embedding_d2: Embeddings,
        nvidia_reranking_api_key: str | None,
        documents_novectorize: list[Document],
        documents2_novectorize: list[Document],
    ) -> None:
        # Hybrid search coll.config is explicit ==> run hyb search is automatic.
        try:
            # create vstore ( => actual collection creation)
            store0 = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_NOVECTORIZE,
                embedding=embedding_d2,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                setup_mode=SetupMode.ASYNC,
                environment=astra_db_credentials["environment"],
                collection_rerank=NVIDIA_RERANKING_OPTIONS_HEADER,
                collection_reranking_api_key=nvidia_reranking_api_key,
                collection_lexical=LEXICAL_OPTIONS,
            )
            # verify it would run hybrid
            assert store0.hybrid_search
            # insert items, check they get $lexical on DB
            await store0.aadd_documents(documents_novectorize)
            assert all(
                "$lexical" in doc
                for doc in (
                    await store0.astra_env.async_collection.find(
                        limit=10, projection={"*": True}
                    ).to_list()
                )
            )
            # run a 'search' (trusting it to be hybrid), some checks on the results
            hits_triples = await store0.asimilarity_search_with_score_id(
                QUERY_TEXT_NOVECTORIZE,
                k=2,
            )
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("[")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)

            # another search with a metadata filter on top
            hits_triples_b = await store0.asimilarity_search_with_score_id(
                QUERY_TEXT_NOVECTORIZE,
                k=2,
                filter={"tag": "01"},
            )
            assert len(hits_triples_b) == 1
            assert hits_triples_b[0][0].page_content == "[1,1]"

            # another search with a different lexical_query
            hits_triples_lq = await store0.asimilarity_search_with_score_id(
                QUERY_TEXT_NOVECTORIZE,
                k=2,
                lexical_query=LEXICAL_QUERY_TEXT,
            )
            assert len(hits_triples_lq) == 2
            rdoc_lq, rscore_lq, rid_lq = hits_triples_lq[0]
            assert rdoc_lq.page_content.startswith("[")
            assert isinstance(rdoc_lq.page_content, str)
            assert rscore_lq > -100
            assert rscore_lq < 100
            assert isinstance(rid_lq, str)

            # re-instantiate just like above, re-check
            store1 = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_NOVECTORIZE,
                embedding=embedding_d2,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                setup_mode=SetupMode.ASYNC,
                environment=astra_db_credentials["environment"],
                collection_rerank=NVIDIA_RERANKING_OPTIONS_HEADER,
                collection_reranking_api_key=nvidia_reranking_api_key,
                collection_lexical=LEXICAL_OPTIONS,
            )
            assert store1.hybrid_search
            hits_triples = await store1.asimilarity_search_with_score_id(
                QUERY_TEXT_NOVECTORIZE,
                k=2,
            )
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("[")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)

            # re-instantiate again, nonstandard hybrid limits
            store1b = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_NOVECTORIZE,
                embedding=embedding_d2,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                setup_mode=SetupMode.ASYNC,
                environment=astra_db_credentials["environment"],
                collection_rerank=NVIDIA_RERANKING_OPTIONS_HEADER,
                collection_reranking_api_key=nvidia_reranking_api_key,
                collection_lexical=LEXICAL_OPTIONS,
                hybrid_limit_factor=math.pi,
            )
            assert store1b.hybrid_search
            hits_triples = await store1b.asimilarity_search_with_score_id(
                QUERY_TEXT_NOVECTORIZE,
                k=2,
            )
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("[")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)

            # re-instantiate again, split hybrid limits (struct)
            store1c = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_NOVECTORIZE,
                embedding=embedding_d2,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                setup_mode=SetupMode.ASYNC,
                environment=astra_db_credentials["environment"],
                collection_rerank=NVIDIA_RERANKING_OPTIONS_HEADER,
                collection_reranking_api_key=nvidia_reranking_api_key,
                collection_lexical=LEXICAL_OPTIONS,
                hybrid_limit_factor=HybridLimitFactorPrescription(
                    vector=3.2,
                    lexical=1.8,
                ),
            )
            assert store1c.hybrid_search
            hits_triples = await store1c.asimilarity_search_with_score_id(
                QUERY_TEXT_NOVECTORIZE,
                k=2,
            )
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("[")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)

            # re-instantiate again, split hybrid limits (dict)
            store1d = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_NOVECTORIZE,
                embedding=embedding_d2,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                setup_mode=SetupMode.ASYNC,
                environment=astra_db_credentials["environment"],
                collection_rerank=NVIDIA_RERANKING_OPTIONS_HEADER,
                collection_reranking_api_key=nvidia_reranking_api_key,
                collection_lexical=LEXICAL_OPTIONS,
                hybrid_limit_factor={"$vector": 3.2, "$lexical": 1.8},
            )
            assert store1d.hybrid_search
            hits_triples = await store1d.asimilarity_search_with_score_id(
                QUERY_TEXT_NOVECTORIZE,
                k=2,
            )
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("[")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)

            # instantiate explicitly, disabling hybrid in searching
            store4 = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_NOVECTORIZE,
                embedding=embedding_d2,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                setup_mode=SetupMode.ASYNC,
                environment=astra_db_credentials["environment"],
                collection_rerank=NVIDIA_RERANKING_OPTIONS_HEADER,
                collection_reranking_api_key=nvidia_reranking_api_key,
                collection_lexical=LEXICAL_OPTIONS,
                hybrid_search=HybridSearchMode.OFF,
            )
            # check it's not doing hybrid
            assert not store4.hybrid_search
            # insert more documents, ensure writes respect $lexical nevertheless
            await store4.aadd_documents(documents2_novectorize)
            assert all(
                "$lexical" in doc
                for doc in (
                    await store4.astra_env.async_collection.find(
                        limit=10, projection={"*": True}
                    ).to_list()
                )
            )
            # run similarity search (expecting regular ANN to be done)
            hits_triples = await store4.asimilarity_search_with_score_id(
                QUERY_TEXT_NOVECTORIZE,
                k=2,
            )
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("[")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)
            # run other search methods
            await store4.asimilarity_search_by_vector([1, 2])
            mmr_hits_docs = await store4.amax_marginal_relevance_search(
                query=QUERY_TEXT_NOVECTORIZE,
                k=1,
            )
            assert len(mmr_hits_docs) == 1
            mmr_doc = mmr_hits_docs[0]
            assert mmr_doc.page_content.startswith("[")
            assert isinstance(mmr_doc.page_content, str)
            assert isinstance(mmr_doc.id, str)

            # regular instantiation, wrong rerank api key -> error on search
            store5 = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_NOVECTORIZE,
                embedding=embedding_d2,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                setup_mode=SetupMode.ASYNC,
                environment=astra_db_credentials["environment"],
                collection_rerank=NVIDIA_RERANKING_OPTIONS_HEADER,
                collection_lexical=LEXICAL_OPTIONS,
                collection_reranking_api_key="bababa-wrong",
            )
            with pytest.raises(DataAPIResponseException):
                await store5.asimilarity_search(QUERY_TEXT_NOVECTORIZE)
        finally:
            await database.to_async().drop_collection(COLLECTION_NAME_NOVECTORIZE)

    def test_astradb_vectorstore_explicit_nohybrid_lifecycle_novectorize_sync(
        self,
        *,
        astra_db_credentials: AstraDBCredentials,
        database: Database,
        embedding_d2: Embeddings,
        documents_novectorize: list[Document],
        documents2_novectorize: list[Document],
    ) -> None:
        # Hybrid search coll.config is explicit ==> run hyb search is automatic.
        try:
            # create vstore ( => actual collection creation)
            store0 = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_NOHYBRID_NOVECTORIZE,
                embedding=embedding_d2,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                collection_rerank=CollectionRerankOptions(enabled=False),
                collection_lexical=CollectionLexicalOptions(enabled=False),
            )
            # verify it would run hybrid
            assert not store0.hybrid_search
            # insert items, check they get $lexical on DB
            store0.add_documents(documents_novectorize)
            assert all(
                "$lexical" not in doc
                for doc in store0.astra_env.collection.find(
                    limit=10, projection={"*": True}
                )
            )
            # run a 'search' (trusting it to be hybrid), some checks on the results
            hits_triples = store0.similarity_search_with_score_id(
                QUERY_TEXT_NOVECTORIZE, k=2
            )
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("[")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)

            # another search with a metadata filter on top
            hits_triples_b = store0.similarity_search_with_score_id(
                QUERY_TEXT_NOVECTORIZE,
                k=2,
                filter={"tag": "01"},
            )
            assert len(hits_triples_b) == 1
            assert hits_triples_b[0][0].page_content == "[1,1]"

            # searches with a different lexical_query should fail
            with pytest.raises(ValueError, match="cannot be passed"):
                store0.similarity_search_with_score_id(
                    QUERY_TEXT,
                    k=2,
                    lexical_query=LEXICAL_QUERY_TEXT,
                )

            # re-instantiate just like above, re-check
            store1 = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_NOHYBRID_NOVECTORIZE,
                embedding=embedding_d2,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                collection_rerank=CollectionRerankOptions(enabled=False),
                collection_lexical=CollectionLexicalOptions(enabled=False),
            )
            assert not store1.hybrid_search
            hits_triples = store1.similarity_search_with_score_id(
                QUERY_TEXT_NOVECTORIZE, k=2
            )
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("[")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)

            # autodetect instantiation (no other changes)
            store2_ad = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_NOHYBRID_NOVECTORIZE,
                embedding=embedding_d2,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                autodetect_collection=True,
            )
            # check it runs hybrid search
            assert not store2_ad.hybrid_search
            # check the right codec is selected
            assert isinstance(
                store2_ad.document_codec,
                _DefaultVSDocumentCodec,
            )
            assert not store2_ad.document_codec.ignore_invalid_documents
            assert not store2_ad.document_codec.has_lexical
            # run a 'search' (trusting it to be hybrid), some checks on the results
            hits_triples = store2_ad.similarity_search_with_score_id(
                QUERY_TEXT_NOVECTORIZE, k=2
            )
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("[")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)
            # run other search methods
            store2_ad.similarity_search_by_vector([1, 2])
            mmr_hits_docs = store2_ad.max_marginal_relevance_search(
                query=QUERY_TEXT_NOVECTORIZE,
                k=1,
            )
            assert len(mmr_hits_docs) == 1
            mmr_doc = mmr_hits_docs[0]
            assert mmr_doc.page_content.startswith("[")
            assert isinstance(mmr_doc.page_content, str)
            assert isinstance(mmr_doc.id, str)

            # autodetect instantiation #2 (nondefault hybrid_limits)
            store3_ad = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_NOHYBRID_NOVECTORIZE,
                embedding=embedding_d2,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                autodetect_collection=True,
                hybrid_limit_factor=math.pi,
            )
            # run a 'search' (trusting it to be hybrid), some checks on the results
            hits_triples = store3_ad.similarity_search_with_score_id(
                QUERY_TEXT_NOVECTORIZE, k=2
            )
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("[")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)

            # instantiate explicitly, disabling hybrid in searching
            store4 = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_NOHYBRID_NOVECTORIZE,
                embedding=embedding_d2,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                collection_rerank=CollectionRerankOptions(enabled=False),
                collection_lexical=CollectionLexicalOptions(enabled=False),
                hybrid_search=HybridSearchMode.OFF,
            )
            # check it's not doing hybrid
            assert not store4.hybrid_search
            # insert more documents, ensure writes respect $lexical nevertheless
            store4.add_documents(documents2_novectorize)
            assert all(
                "$lexical" not in doc
                for doc in store4.astra_env.collection.find(
                    limit=10, projection={"*": True}
                )
            )
            # run similarity search (expecting regular ANN to be done)
            hits_triples = store4.similarity_search_with_score_id(
                QUERY_TEXT_NOVECTORIZE, k=2
            )
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("[")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)
            # run other search methods
            store4.similarity_search_by_vector([1, 2])
            mmr_hits_docs = store4.max_marginal_relevance_search(
                query=QUERY_TEXT_NOVECTORIZE,
                k=1,
            )
            assert len(mmr_hits_docs) == 1
            mmr_doc = mmr_hits_docs[0]
            assert mmr_doc.page_content.startswith("[")
            assert isinstance(mmr_doc.page_content, str)
            assert isinstance(mmr_doc.id, str)

        finally:
            database.drop_collection(COLLECTION_NAME_NOHYBRID_NOVECTORIZE)

    async def test_astradb_vectorstore_explicit_nohybrid_lifecycle_novectorize_async(
        self,
        *,
        astra_db_credentials: AstraDBCredentials,
        database: Database,
        embedding_d2: Embeddings,
        documents_novectorize: list[Document],
        documents2_novectorize: list[Document],
    ) -> None:
        # Hybrid search coll.config is explicit ==> run hyb search is automatic.
        try:
            # create vstore ( => actual collection creation)
            store0 = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_NOHYBRID_NOVECTORIZE,
                embedding=embedding_d2,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                setup_mode=SetupMode.ASYNC,
                environment=astra_db_credentials["environment"],
                collection_rerank=CollectionRerankOptions(enabled=False),
                collection_lexical=CollectionLexicalOptions(enabled=False),
            )
            # verify it would run hybrid
            assert not store0.hybrid_search
            # insert items, check they get $lexical on DB
            await store0.aadd_documents(documents_novectorize)
            assert all(
                "$lexical" not in doc
                for doc in (
                    await store0.astra_env.async_collection.find(
                        limit=10, projection={"*": True}
                    ).to_list()
                )
            )
            # run a 'search' (trusting it to be hybrid), some checks on the results
            hits_triples = await store0.asimilarity_search_with_score_id(
                QUERY_TEXT_NOVECTORIZE,
                k=2,
            )
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("[")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)

            # another search with a metadata filter on top
            hits_triples_b = await store0.asimilarity_search_with_score_id(
                QUERY_TEXT_NOVECTORIZE,
                k=2,
                filter={"tag": "01"},
            )
            assert len(hits_triples_b) == 1
            assert hits_triples_b[0][0].page_content == "[1,1]"

            # searches with a different lexical_query should fail
            with pytest.raises(ValueError, match="cannot be passed"):
                await store0.asimilarity_search_with_score_id(
                    QUERY_TEXT,
                    k=2,
                    lexical_query=LEXICAL_QUERY_TEXT,
                )

            # re-instantiate just like above, re-check
            store1 = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_NOHYBRID_NOVECTORIZE,
                embedding=embedding_d2,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                setup_mode=SetupMode.ASYNC,
                environment=astra_db_credentials["environment"],
                collection_rerank=CollectionRerankOptions(enabled=False),
                collection_lexical=CollectionLexicalOptions(enabled=False),
            )
            assert not store1.hybrid_search
            hits_triples = await store1.asimilarity_search_with_score_id(
                QUERY_TEXT_NOVECTORIZE,
                k=2,
            )
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("[")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)

            # instantiate explicitly, disabling hybrid in searching
            store4 = AstraDBVectorStore(
                collection_name=COLLECTION_NAME_NOHYBRID_NOVECTORIZE,
                embedding=embedding_d2,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                setup_mode=SetupMode.ASYNC,
                environment=astra_db_credentials["environment"],
                collection_rerank=CollectionRerankOptions(enabled=False),
                collection_lexical=CollectionLexicalOptions(enabled=False),
                hybrid_search=HybridSearchMode.OFF,
            )
            # check it's not doing hybrid
            assert not store4.hybrid_search
            # insert more documents, ensure writes respect $lexical nevertheless
            await store4.aadd_documents(documents2_novectorize)
            assert all(
                "$lexical" not in doc
                for doc in (
                    await store4.astra_env.async_collection.find(
                        limit=10, projection={"*": True}
                    ).to_list()
                )
            )
            # run similarity search (expecting regular ANN to be done)
            hits_triples = await store4.asimilarity_search_with_score_id(
                QUERY_TEXT_NOVECTORIZE,
                k=2,
            )
            assert len(hits_triples) == 2
            rdoc, rscore, rid = hits_triples[0]
            assert rdoc.page_content.startswith("[")
            assert isinstance(rdoc.page_content, str)
            assert rscore > -100
            assert rscore < 100
            assert isinstance(rid, str)
            # run other search methods
            await store4.asimilarity_search_by_vector([1, 2])
            mmr_hits_docs = await store4.amax_marginal_relevance_search(
                query=QUERY_TEXT_NOVECTORIZE,
                k=1,
            )
            assert len(mmr_hits_docs) == 1
            mmr_doc = mmr_hits_docs[0]
            assert mmr_doc.page_content.startswith("[")
            assert isinstance(mmr_doc.page_content, str)
            assert isinstance(mmr_doc.id, str)

        finally:
            await database.to_async().drop_collection(
                COLLECTION_NAME_NOHYBRID_NOVECTORIZE
            )
