import pytest
from astrapy import Collection
from astrapy.authentication import StaticTokenProvider
from langchain_tests.integration_tests import AsyncCacheTestSuite, SyncCacheTestSuite

from langchain_astradb import AstraDBCache
from tests.integration_tests.conftest import (
    AstraDBCredentials,
    astra_db_env_vars_available,
)


class _BaseTestAstraDBCache:
    @pytest.fixture(autouse=True)
    def setup(
        self,
        astra_db_credentials: AstraDBCredentials,
        empty_collection_idxall: Collection,
    ) -> None:
        self._cache = AstraDBCache(
            collection_name=empty_collection_idxall.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        )


@pytest.mark.skipif(
    not astra_db_env_vars_available(), reason="Missing Astra DB env. vars"
)
class TestAstraDBCache(_BaseTestAstraDBCache, SyncCacheTestSuite):
    @pytest.fixture
    def cache(self) -> AstraDBCache:
        return self._cache


@pytest.mark.skipif(
    not astra_db_env_vars_available(), reason="Missing Astra DB env. vars"
)
class TestAstraDBCacheAsync(_BaseTestAstraDBCache, AsyncCacheTestSuite):
    @pytest.fixture
    async def cache(self) -> AstraDBCache:
        return self._cache
