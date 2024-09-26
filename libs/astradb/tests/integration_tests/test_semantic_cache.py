from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from astrapy.authentication import StaticTokenProvider
from langchain_core.globals import get_llm_cache, set_llm_cache
from langchain_core.outputs import Generation, LLMResult

from langchain_astradb import AstraDBSemanticCache
from langchain_astradb.utils.astradb import SetupMode

from .conftest import (
    COLLECTION_NAME_IDXALL_D2,
    AstraDBCredentials,
    astra_db_env_vars_available,
)

if TYPE_CHECKING:
    from astrapy import Collection
    from astrapy.db import AstraDB
    from langchain_core.embeddings import Embeddings

    from .conftest import IdentityLLM


@pytest.fixture
def astradb_semantic_cache(
    astra_db_credentials: AstraDBCredentials,
    empty_collection_idxall_d2: Collection,  # noqa: ARG001
    embedding_d2: Embeddings,
) -> AstraDBSemanticCache:
    return AstraDBSemanticCache(
        collection_name=COLLECTION_NAME_IDXALL_D2,
        token=StaticTokenProvider(astra_db_credentials["token"]),
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
        embedding=embedding_d2,
        metric="euclidean",
    )


@pytest.fixture
async def astradb_semantic_cache_async(
    astra_db_credentials: AstraDBCredentials,
    empty_collection_idxall_d2: Collection,  # noqa: ARG001
    embedding_d2: Embeddings,
) -> AstraDBSemanticCache:
    return AstraDBSemanticCache(
        collection_name=COLLECTION_NAME_IDXALL_D2,
        token=StaticTokenProvider(astra_db_credentials["token"]),
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
        embedding=embedding_d2,
        metric="euclidean",
        setup_mode=SetupMode.ASYNC,
    )


@pytest.mark.skipif(
    not astra_db_env_vars_available(), reason="Missing Astra DB env. vars"
)
class TestAstraDBSemanticCache:
    def test_semantic_cache_crud_sync(
        self,
        astradb_semantic_cache: AstraDBSemanticCache,
    ) -> None:
        """Tests for basic cache CRUD, not involving an LLM."""
        gens0 = [Generation(text="gen0_text")]
        gens1 = [Generation(text="gen1_text")]
        assert astradb_semantic_cache.lookup_with_id("[1,2]", "lms") is None

        astradb_semantic_cache.update("[0.999,2.001]", "lms", gens0)
        astradb_semantic_cache.update("[2.999,4.001]", "lms", gens1)

        hit12 = astradb_semantic_cache.lookup_with_id("[1,2]", "lms")
        assert hit12 is not None
        assert hit12[1] == gens0
        hit34 = astradb_semantic_cache.lookup_with_id("[3,4]", "lms")
        assert hit34 is not None
        assert hit34[1] == gens1

        astradb_semantic_cache.delete_by_document_id(hit12[0])
        assert astradb_semantic_cache.lookup_with_id("[1,2]", "lms") is None
        hit34_b = astradb_semantic_cache.lookup_with_id("[3,4]", "lms")
        assert hit34_b is not None
        assert hit34_b[1] == gens1

        astradb_semantic_cache.clear()
        assert astradb_semantic_cache.lookup_with_id("[1,2]", "lms") is None
        assert astradb_semantic_cache.lookup_with_id("[3,4]", "lms") is None

    async def test_semantic_cache_crud_async(
        self,
        astradb_semantic_cache_async: AstraDBSemanticCache,
    ) -> None:
        """Tests for basic cache CRUD, not involving an LLM. Async version"""
        gens0 = [Generation(text="gen0_text")]
        gens1 = [Generation(text="gen1_text")]
        assert (
            await astradb_semantic_cache_async.alookup_with_id("[1,2]", "lms") is None
        )

        await astradb_semantic_cache_async.aupdate("[0.999,2.001]", "lms", gens0)
        await astradb_semantic_cache_async.aupdate("[2.999,4.001]", "lms", gens1)

        hit12 = await astradb_semantic_cache_async.alookup_with_id("[1,2]", "lms")
        assert hit12 is not None
        assert hit12[1] == gens0
        hit34 = await astradb_semantic_cache_async.alookup_with_id("[3,4]", "lms")
        assert hit34 is not None
        assert hit34[1] == gens1

        await astradb_semantic_cache_async.adelete_by_document_id(hit12[0])
        assert (
            await astradb_semantic_cache_async.alookup_with_id("[1,2]", "lms") is None
        )
        hit34_b = await astradb_semantic_cache_async.alookup_with_id("[3,4]", "lms")
        assert hit34_b is not None
        assert hit34_b[1] == gens1

        await astradb_semantic_cache_async.aclear()
        assert (
            await astradb_semantic_cache_async.alookup_with_id("[1,2]", "lms") is None
        )
        assert (
            await astradb_semantic_cache_async.alookup_with_id("[3,4]", "lms") is None
        )

    def test_semantic_cache_through_llm_sync(
        self,
        test_llm: IdentityLLM,
        astradb_semantic_cache: AstraDBSemanticCache,
    ) -> None:
        """Tests for semantic cache as used with a (mock) LLM."""
        gens0 = [Generation(text="gen0_text")]
        set_llm_cache(astradb_semantic_cache)

        params = {"stop": None, **test_llm.dict()}
        llm_string = str(sorted(params.items()))

        assert test_llm.num_calls == 0

        # inject cache entry, check no LLM call is done
        get_llm_cache().update("[1,2]", llm_string, gens0)
        output = test_llm.generate(["[0.999,2.001]"])
        expected_output = LLMResult(
            generations=[gens0],
            llm_output={},
        )
        assert test_llm.num_calls == 0
        assert output == expected_output

        # check *one* new call for a new prompt, even if 'generate' repeated
        test_llm.generate(["[3,4]"])
        test_llm.generate(["[3,4]"])
        test_llm.generate(["[3,4]"])
        assert test_llm.num_calls == 1

        # clear the cache and check a new LLM call is actually made
        astradb_semantic_cache.clear()
        test_llm.generate(["[3,4]"])
        test_llm.generate(["[3,4]"])
        assert test_llm.num_calls == 2

    async def test_semantic_cache_through_llm_async(
        self,
        test_llm: IdentityLLM,
        astradb_semantic_cache: AstraDBSemanticCache,
    ) -> None:
        """Tests for semantic cache as used with a (mock) LLM, async version."""
        gens0 = [Generation(text="gen0_text")]
        set_llm_cache(astradb_semantic_cache)

        params = {"stop": None, **test_llm.dict()}
        llm_string = str(sorted(params.items()))

        assert test_llm.num_calls == 0

        # inject cache entry, check no LLM call is done
        await get_llm_cache().aupdate("[1,2]", llm_string, gens0)
        output = await test_llm.agenerate(["[0.999,2.001]"])
        expected_output = LLMResult(
            generations=[gens0],
            llm_output={},
        )
        assert test_llm.num_calls == 0
        assert output == expected_output

        # check *one* new call for a new prompt, even if 'generate' repeated
        await test_llm.agenerate(["[3,4]"])
        await test_llm.agenerate(["[3,4]"])
        await test_llm.agenerate(["[3,4]"])
        assert test_llm.num_calls == 1

        # clear the cache and check a new LLM call is actually made
        await astradb_semantic_cache.aclear()
        await test_llm.agenerate(["[3,4]"])
        await test_llm.agenerate(["[3,4]"])
        assert test_llm.num_calls == 2

    @pytest.mark.skipif(
        os.environ.get("ASTRA_DB_ENVIRONMENT", "prod").upper() != "PROD",
        reason="Can run on Astra DB production environment only",
    )
    def test_semcache_coreclients_init_sync(
        self,
        core_astra_db: AstraDB,
        embedding_d2: Embeddings,
        astradb_semantic_cache: AstraDBSemanticCache,
    ) -> None:
        """A deprecation warning from passing a (core) AstraDB, but it works."""
        gens0 = [Generation(text="gen0_text")]
        astradb_semantic_cache.update("[0.999,2.001]", "llm_string", gens0)
        # create an equivalent cache with core AstraDB in init
        with pytest.warns(DeprecationWarning) as rec_warnings:
            semantic_cache_init_core = AstraDBSemanticCache(
                collection_name=COLLECTION_NAME_IDXALL_D2,
                astra_db_client=core_astra_db,
                embedding=embedding_d2,
                metric="euclidean",
            )
        f_rec_warnings = [
            wrn for wrn in rec_warnings if issubclass(wrn.category, DeprecationWarning)
        ]
        assert len(f_rec_warnings) == 1
        hit12 = semantic_cache_init_core.lookup_with_id("[1,2]", "llm_string")
        assert hit12 is not None
        assert hit12[1] == gens0

    @pytest.mark.skipif(
        os.environ.get("ASTRA_DB_ENVIRONMENT", "prod").upper() != "PROD",
        reason="Can run on Astra DB production environment only",
    )
    async def test_semcache_coreclients_init_async(
        self,
        core_astra_db: AstraDB,
        embedding_d2: Embeddings,
        astradb_semantic_cache_async: AstraDBSemanticCache,
    ) -> None:
        """
        A deprecation warning from passing a (core) AstraDB, but it works.
        Async version.
        """
        gens0 = [Generation(text="gen0_text")]
        await astradb_semantic_cache_async.aupdate("[0.999,2.001]", "llm_string", gens0)
        # create an equivalent cache with core AstraDB in init
        with pytest.warns(DeprecationWarning) as rec_warnings:
            semantic_cache_init_core = AstraDBSemanticCache(
                collection_name=COLLECTION_NAME_IDXALL_D2,
                astra_db_client=core_astra_db,
                embedding=embedding_d2,
                metric="euclidean",
                setup_mode=SetupMode.ASYNC,
            )
        f_rec_warnings = [
            wrn for wrn in rec_warnings if issubclass(wrn.category, DeprecationWarning)
        ]
        assert len(f_rec_warnings) == 1
        hit12 = await semantic_cache_init_core.alookup_with_id("[1,2]", "llm_string")
        assert hit12 is not None
        assert hit12[1] == gens0
