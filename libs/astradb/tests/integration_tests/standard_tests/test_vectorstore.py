import pytest
from astrapy import Collection
from astrapy.authentication import StaticTokenProvider
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests

from langchain_astradb import AstraDBVectorStore
from langchain_astradb.utils.astradb import SetupMode
from tests.integration_tests.conftest import (
    AstraDBCredentials,
    astra_db_env_vars_available,
)


@pytest.mark.skipif(
    not astra_db_env_vars_available(), reason="Missing Astra DB env. vars"
)
class TestAstraDBVectorStoreIntegration(VectorStoreIntegrationTests):
    @pytest.fixture(autouse=True)
    def setup(
        self,
        empty_collection_d2: Collection,
        astra_db_credentials: AstraDBCredentials,
    ) -> None:
        self._vectorstore = AstraDBVectorStore(
            embedding=DeterministicFakeEmbedding(size=2),
            collection_name=empty_collection_d2.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            setup_mode=SetupMode.OFF,
        )

    @pytest.fixture
    def vectorstore(self) -> VectorStore:
        return self._vectorstore
