from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from astrapy.authentication import StaticTokenProvider
from langchain_core.globals import get_llm_cache, set_llm_cache
from langchain_core.outputs import Generation, LLMResult

from langchain_astradb import AstraDBCache
from langchain_astradb.utils.astradb import SetupMode

from .conftest import (
    AstraDBCredentials,
    astra_db_env_vars_available,
)

if TYPE_CHECKING:
    from astrapy import Collection

    from .conftest import IdentityLLM


@pytest.fixture
def astradb_cache(
    astra_db_credentials: AstraDBCredentials,
    empty_collection_idxall: Collection,
) -> AstraDBCache:
    return AstraDBCache(
        collection_name=empty_collection_idxall.name,
        token=StaticTokenProvider(astra_db_credentials["token"]),
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
    )


@pytest.fixture
async def astradb_cache_async(
    astra_db_credentials: AstraDBCredentials,
    empty_collection_idxall: Collection,
) -> AstraDBCache:
    return AstraDBCache(
        collection_name=empty_collection_idxall.name,
        token=StaticTokenProvider(astra_db_credentials["token"]),
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
        setup_mode=SetupMode.ASYNC,
    )


@pytest.mark.skipif(
    not astra_db_env_vars_available(), reason="Missing Astra DB env. vars"
)
class TestAstraDBCache:
    def test_cache_crud_sync(
        self,
        astradb_cache: AstraDBCache,
    ) -> None:
        """Tests for basic cache CRUD, not involving an LLM."""
        gens0 = [Generation(text="gen0_text")]
        gens1 = [Generation(text="gen1_text")]
        assert astradb_cache.lookup("prompt0", "llm_string") is None

        astradb_cache.update("prompt0", "llm_string", gens0)
        astradb_cache.update("prompt1", "llm_string", gens1)
        assert astradb_cache.lookup("prompt0", "llm_string") == gens0
        assert astradb_cache.lookup("prompt1", "llm_string") == gens1

        astradb_cache.delete("prompt0", "llm_string")
        assert astradb_cache.lookup("prompt0", "llm_string") is None
        assert astradb_cache.lookup("prompt1", "llm_string") == gens1

        astradb_cache.clear()
        assert astradb_cache.lookup("prompt0", "llm_string") is None
        assert astradb_cache.lookup("prompt1", "llm_string") is None

    async def test_cache_crud_async(
        self,
        astradb_cache_async: AstraDBCache,
    ) -> None:
        """
        Tests for basic cache CRUD, not involving an LLM.
        Async version.
        """
        gens0 = [Generation(text="gen0_text")]
        gens1 = [Generation(text="gen1_text")]
        assert await astradb_cache_async.alookup("prompt0", "llm_string") is None

        await astradb_cache_async.aupdate("prompt0", "llm_string", gens0)
        await astradb_cache_async.aupdate("prompt1", "llm_string", gens1)
        assert await astradb_cache_async.alookup("prompt0", "llm_string") == gens0
        assert await astradb_cache_async.alookup("prompt1", "llm_string") == gens1

        await astradb_cache_async.adelete("prompt0", "llm_string")
        assert await astradb_cache_async.alookup("prompt0", "llm_string") is None
        assert await astradb_cache_async.alookup("prompt1", "llm_string") == gens1

        await astradb_cache_async.aclear()
        assert await astradb_cache_async.alookup("prompt0", "llm_string") is None
        assert await astradb_cache_async.alookup("prompt1", "llm_string") is None

    def test_cache_through_llm_sync(
        self,
        test_llm: IdentityLLM,
        astradb_cache: AstraDBCache,
    ) -> None:
        """Tests for cache as used with a (mock) LLM."""
        gens0 = [Generation(text="gen0_text")]
        set_llm_cache(astradb_cache)

        params = {"stop": None, **test_llm.dict()}
        llm_string = str(sorted(params.items()))

        assert test_llm.num_calls == 0

        # inject cache entry, check no LLM call is done
        get_llm_cache().update("prompt0", llm_string, gens0)
        output = test_llm.generate(["prompt0"])
        expected_output = LLMResult(
            generations=[gens0],
            llm_output={},
        )
        assert test_llm.num_calls == 0
        assert output == expected_output

        # check *one* new call for a new prompt, even if 'generate' repeated
        test_llm.generate(["prompt1"])
        test_llm.generate(["prompt1"])
        test_llm.generate(["prompt1"])
        assert test_llm.num_calls == 1

        # remove the cache and check a new LLM call is actually made
        astradb_cache.delete_through_llm("prompt1", test_llm, stop=None)
        test_llm.generate(["prompt1"])
        test_llm.generate(["prompt1"])
        assert test_llm.num_calls == 2

    async def test_cache_through_llm_async(
        self,
        test_llm: IdentityLLM,
        astradb_cache_async: AstraDBCache,
    ) -> None:
        """Tests for cache as used with a (mock) LLM, async version"""
        gens0 = [Generation(text="gen0_text")]
        set_llm_cache(astradb_cache_async)

        params = {"stop": None, **test_llm.dict()}
        llm_string = str(sorted(params.items()))

        assert test_llm.num_calls == 0

        # inject cache entry, check no LLM call is done
        await get_llm_cache().aupdate("prompt0", llm_string, gens0)
        output = await test_llm.agenerate(["prompt0"])
        expected_output = LLMResult(
            generations=[gens0],
            llm_output={},
        )
        assert test_llm.num_calls == 0
        assert output == expected_output

        # check *one* new call for a new prompt, even if 'generate' repeated
        await test_llm.agenerate(["prompt1"])
        await test_llm.agenerate(["prompt1"])
        await test_llm.agenerate(["prompt1"])
        assert test_llm.num_calls == 1

        # remove the cache and check a new LLM call is actually made
        await astradb_cache_async.adelete_through_llm("prompt1", test_llm, stop=None)
        await test_llm.agenerate(["prompt1"])
        await test_llm.agenerate(["prompt1"])
        assert test_llm.num_calls == 2
