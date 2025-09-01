from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from astrapy.authentication import StaticTokenProvider
from langchain_tests.integration_tests import BaseStoreAsyncTests, BaseStoreSyncTests

from langchain_astradb import AstraDBByteStore, AstraDBStore
from langchain_astradb.utils.astradb import SetupMode
from tests.integration_tests.conftest import astra_db_env_vars_available

if TYPE_CHECKING:
    from astrapy import Collection

    from tests.integration_tests.conftest import AstraDBCredentials


class _BaseTestAstraDBStore:
    @pytest.fixture(autouse=True)
    def setup(
        self,
        astra_db_credentials: AstraDBCredentials,
        collection_idxid: Collection,
    ) -> None:
        collection_idxid.delete_many({})
        self._store = AstraDBStore(
            collection_name=collection_idxid.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            setup_mode=SetupMode.OFF,
        )

    @pytest.fixture
    def three_values(self) -> tuple[str, str, str]:
        return "foo", "bar", "buzz"


@pytest.mark.skipif(
    not astra_db_env_vars_available(), reason="Missing Astra DB env. vars"
)
class TestAstraDBStore(_BaseTestAstraDBStore, BaseStoreSyncTests[str]):
    @pytest.fixture
    def kv_store(self) -> AstraDBStore:
        return self._store


@pytest.mark.skipif(
    not astra_db_env_vars_available(), reason="Missing Astra DB env. vars"
)
class TestAstraDBStoreASync(_BaseTestAstraDBStore, BaseStoreAsyncTests[str]):
    @pytest.fixture
    async def kv_store(self) -> AstraDBStore:
        return self._store


class _BaseTestAstraDBByteStore:
    @pytest.fixture(autouse=True)
    def setup(
        self,
        astra_db_credentials: AstraDBCredentials,
        collection_idxid: Collection,
    ) -> None:
        collection_idxid.delete_many({})
        self._store = AstraDBByteStore(
            collection_name=collection_idxid.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            setup_mode=SetupMode.OFF,
        )

    @pytest.fixture
    def three_values(self) -> tuple[bytes, bytes, bytes]:
        return b"foo", b"bar", b"buzz"


@pytest.mark.skipif(
    not astra_db_env_vars_available(), reason="Missing Astra DB env. vars"
)
class TestAstraDBByteStore(_BaseTestAstraDBByteStore, BaseStoreSyncTests[bytes]):
    @pytest.fixture
    def kv_store(self) -> AstraDBByteStore:
        return self._store


@pytest.mark.skipif(
    not astra_db_env_vars_available(), reason="Missing Astra DB env. vars"
)
class TestAstraDBByteStoreASync(_BaseTestAstraDBByteStore, BaseStoreAsyncTests[bytes]):
    @pytest.fixture
    async def kv_store(self) -> AstraDBByteStore:
        return self._store
