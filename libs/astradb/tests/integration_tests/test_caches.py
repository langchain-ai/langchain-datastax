"""Test AstraDB caches. Requires an Astra DB vector instance.

Required to run this test:
    - a recent `astrapy` Python package available
    - an Astra DB instance;
    - the two environment variables set:
        export ASTRA_DB_API_ENDPOINT="https://<DB-ID>-us-east1.apps.astra.datastax.com"
        export ASTRA_DB_APPLICATION_TOKEN="AstraCS:........."
    - optionally this as well (otherwise defaults are used):
        export ASTRA_DB_KEYSPACE="my_keyspace"
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, Mapping, Optional, cast

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.globals import get_llm_cache, set_llm_cache
from langchain_core.language_models import LLM
from langchain_core.outputs import Generation, LLMResult
from typing_extensions import override

from langchain_astradb import AstraDBCache, AstraDBSemanticCache
from langchain_astradb.utils.astradb import SetupMode

from .conftest import AstraDBCredentials, _has_env_vars

if TYPE_CHECKING:
    from astrapy.db import AstraDB
    from langchain_core.caches import BaseCache
    from langchain_core.callbacks import CallbackManagerForLLMRun


class FakeEmbeddings(Embeddings):
    """Fake embeddings functionality for testing."""

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return simple embeddings.
        Embeddings encode each text as its index.
        """
        return [[1.0] * 9 + [float(i)] for i in range(len(texts))]

    @override
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    @override
    def embed_query(self, text: str) -> list[float]:
        """Return constant query embeddings.
        Embeddings are identical to embed_documents(texts)[0].
        Distance to each text will be that text's index,
        as it was passed to embed_documents.
        """
        return [1.0] * 9 + [0.0]

    @override
    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


class FakeLLM(LLM):
    """Fake LLM wrapper for testing purposes."""

    queries: Optional[Mapping] = None  # noqa: UP007
    sequential_responses: Optional[bool] = False  # noqa: UP007
    response_index: int = 0

    @override
    def get_num_tokens(self, text: str) -> int:
        """Return number of tokens."""
        return len(text.split())

    @property
    @override
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake"

    @override
    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        if self.sequential_responses:
            return self._get_next_response_in_sequence
        if self.queries is not None:
            return self.queries[prompt]
        return "foo" if stop is None else "bar"

    @property
    @override
    def _identifying_params(self) -> dict[str, Any]:
        return {}

    @property
    def _get_next_response_in_sequence(self) -> str:
        queries = cast(Mapping, self.queries)
        response = queries[list(queries.keys())[self.response_index]]
        self.response_index = self.response_index + 1
        return response


@pytest.fixture(scope="module")
def astradb_cache(astra_db_credentials: AstraDBCredentials) -> Iterator[AstraDBCache]:
    cache = AstraDBCache(
        collection_name="lc_integration_test_cache",
        token=astra_db_credentials["token"],
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
    )
    yield cache
    cache.collection.drop()


@pytest.fixture
async def async_astradb_cache(
    astra_db_credentials: AstraDBCredentials,
) -> AsyncIterator[AstraDBCache]:
    cache = AstraDBCache(
        collection_name="lc_integration_test_cache_async",
        token=astra_db_credentials["token"],
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
        setup_mode=SetupMode.ASYNC,
    )
    yield cache
    await cache.async_collection.drop()


@pytest.fixture(scope="module")
def astradb_semantic_cache(
    astra_db_credentials: AstraDBCredentials,
) -> Iterator[AstraDBSemanticCache]:
    fake_embe = FakeEmbeddings()
    sem_cache = AstraDBSemanticCache(
        collection_name="lc_integration_test_sem_cache",
        token=astra_db_credentials["token"],
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
        embedding=fake_embe,
    )
    yield sem_cache
    sem_cache.collection.drop()


@pytest.fixture
async def async_astradb_semantic_cache(
    astra_db_credentials: AstraDBCredentials,
) -> AsyncIterator[AstraDBSemanticCache]:
    fake_embe = FakeEmbeddings()
    sem_cache = AstraDBSemanticCache(
        collection_name="lc_integration_test_sem_cache_async",
        token=astra_db_credentials["token"],
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
        embedding=fake_embe,
        setup_mode=SetupMode.ASYNC,
    )
    yield sem_cache
    sem_cache.collection.drop()


@pytest.mark.skipif(not _has_env_vars(), reason="Missing Astra DB env. vars")
class TestAstraDBCaches:
    def test_astradb_cache_sync(self, astradb_cache: AstraDBCache) -> None:
        self.do_cache_test(FakeLLM(), astradb_cache, "foo")

    async def test_astradb_cache_async(self, async_astradb_cache: AstraDBCache) -> None:
        await self.ado_cache_test(FakeLLM(), async_astradb_cache, "foo")

    def test_astradb_semantic_cache_sync(
        self, astradb_semantic_cache: AstraDBSemanticCache
    ) -> None:
        llm = FakeLLM()
        self.do_cache_test(llm, astradb_semantic_cache, "bar")
        output = llm.generate(["bar"])  # 'fizz' is erased away now
        assert output != LLMResult(
            generations=[[Generation(text="fizz")]],
            llm_output={},
        )
        astradb_semantic_cache.clear()

    async def test_astradb_semantic_cache_async(
        self, async_astradb_semantic_cache: AstraDBSemanticCache
    ) -> None:
        llm = FakeLLM()
        await self.ado_cache_test(llm, async_astradb_semantic_cache, "bar")
        output = await llm.agenerate(["bar"])  # 'fizz' is erased away now
        assert output != LLMResult(
            generations=[[Generation(text="fizz")]],
            llm_output={},
        )
        await async_astradb_semantic_cache.aclear()

    @staticmethod
    def do_cache_test(llm: LLM, cache: BaseCache, prompt: str) -> None:
        set_llm_cache(cache)
        params = llm.dict()
        params["stop"] = None
        llm_string = str(sorted(params.items()))
        get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])
        output = llm.generate([prompt])
        expected_output = LLMResult(
            generations=[[Generation(text="fizz")]],
            llm_output={},
        )
        assert output == expected_output
        # clear the cache
        cache.clear()

    @staticmethod
    async def ado_cache_test(llm: LLM, cache: BaseCache, prompt: str) -> None:
        set_llm_cache(cache)
        params = llm.dict()
        params["stop"] = None
        llm_string = str(sorted(params.items()))
        await get_llm_cache().aupdate("foo", llm_string, [Generation(text="fizz")])
        output = await llm.agenerate([prompt])
        expected_output = LLMResult(
            generations=[[Generation(text="fizz")]],
            llm_output={},
        )
        assert output == expected_output
        # clear the cache
        await cache.aclear()

    @pytest.mark.skipif(
        os.environ.get("ASTRA_DB_ENVIRONMENT", "prod").upper() != "PROD",
        reason="Can run on Astra DB prod only",
    )
    def test_cache_coreclients_init_sync(
        self,
        astra_db_credentials: AstraDBCredentials,
        core_astra_db: AstraDB,
    ) -> None:
        """A deprecation warning from passing a (core) AstraDB, but it works."""
        collection_name = "lc_test_cache_coreclsync"
        test_gens = [Generation(text="ret_val0123")]
        try:
            cache_init_ok = AstraDBCache(
                collection_name=collection_name,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
            )
            cache_init_ok.update("pr", "llms", test_gens)
            # create an equivalent cache with core AstraDB in init
            with pytest.warns(DeprecationWarning) as rec_warnings:
                cache_init_core = AstraDBCache(
                    collection_name=collection_name,
                    astra_db_client=core_astra_db,
                )
            assert len(rec_warnings) == 1
            assert cache_init_core.lookup("pr", "llms") == test_gens
        finally:
            cache_init_ok.astra_env.database.drop_collection(collection_name)

    @pytest.mark.skipif(
        os.environ.get("ASTRA_DB_ENVIRONMENT", "prod").upper() != "PROD",
        reason="Can run on Astra DB prod only",
    )
    async def test_cache_coreclients_init_async(
        self,
        astra_db_credentials: AstraDBCredentials,
        core_astra_db: AstraDB,
    ) -> None:
        """A deprecation warning from passing a (core) AstraDB, but it works."""
        collection_name = "lc_test_cache_coreclasync"
        test_gens = [Generation(text="ret_val4567")]
        try:
            cache_init_ok = AstraDBCache(
                collection_name=collection_name,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                setup_mode=SetupMode.ASYNC,
            )
            await cache_init_ok.aupdate("pr", "llms", test_gens)
            # create an equivalent cache with core AstraDB in init
            with pytest.warns(DeprecationWarning) as rec_warnings:
                cache_init_core = AstraDBCache(
                    collection_name=collection_name,
                    astra_db_client=core_astra_db,
                    setup_mode=SetupMode.ASYNC,
                )
            assert len(rec_warnings) == 1
            assert await cache_init_core.alookup("pr", "llms") == test_gens
        finally:
            await cache_init_ok.astra_env.async_database.drop_collection(
                collection_name
            )

    @pytest.mark.skipif(
        os.environ.get("ASTRA_DB_ENVIRONMENT", "prod").upper() != "PROD",
        reason="Can run on Astra DB prod only",
    )
    def test_semcache_coreclients_init_sync(
        self,
        astra_db_credentials: AstraDBCredentials,
        core_astra_db: AstraDB,
    ) -> None:
        """A deprecation warning from passing a (core) AstraDB, but it works."""
        fake_embe = FakeEmbeddings()
        collection_name = "lc_test_cache_coreclsync"
        test_gens = [Generation(text="ret_val0123")]
        try:
            cache_init_ok = AstraDBSemanticCache(
                collection_name=collection_name,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                embedding=fake_embe,
            )
            cache_init_ok.update("pr", "llms", test_gens)
            # create an equivalent cache with core AstraDB in init
            with pytest.warns(DeprecationWarning) as rec_warnings:
                cache_init_core = AstraDBSemanticCache(
                    collection_name=collection_name,
                    astra_db_client=core_astra_db,
                    embedding=fake_embe,
                )
            assert len(rec_warnings) == 1
            assert cache_init_core.lookup("pr", "llms") == test_gens
        finally:
            cache_init_ok.astra_env.database.drop_collection(collection_name)

    @pytest.mark.skipif(
        os.environ.get("ASTRA_DB_ENVIRONMENT", "prod").upper() != "PROD",
        reason="Can run on Astra DB prod only",
    )
    async def test_semcache_coreclients_init_async(
        self,
        astra_db_credentials: AstraDBCredentials,
        core_astra_db: AstraDB,
    ) -> None:
        """A deprecation warning from passing a (core) AstraDB, but it works."""
        fake_embe = FakeEmbeddings()
        collection_name = "lc_test_cache_coreclasync"
        test_gens = [Generation(text="ret_val4567")]
        try:
            cache_init_ok = AstraDBSemanticCache(
                collection_name=collection_name,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                setup_mode=SetupMode.ASYNC,
                embedding=fake_embe,
            )
            await cache_init_ok.aupdate("pr", "llms", test_gens)
            # create an equivalent cache with core AstraDB in init
            with pytest.warns(DeprecationWarning) as rec_warnings:
                cache_init_core = AstraDBSemanticCache(
                    collection_name=collection_name,
                    astra_db_client=core_astra_db,
                    setup_mode=SetupMode.ASYNC,
                    embedding=fake_embe,
                )
            assert len(rec_warnings) == 1
            assert await cache_init_core.alookup("pr", "llms") == test_gens
        finally:
            await cache_init_ok.astra_env.async_database.drop_collection(
                collection_name
            )
