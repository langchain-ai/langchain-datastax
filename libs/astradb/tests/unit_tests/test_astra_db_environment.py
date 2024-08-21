import os

import pytest
from astrapy.db import AstraDB

from langchain_astradb.utils.astradb import (
    API_ENDPOINT_ENV_VAR,
    NAMESPACE_ENV_VAR,
    TOKEN_ENV_VAR,
    _AstraDBEnvironment,
)


class TestAstraDBEnvironment:
    def test_initialization(self) -> None:
        """Test the various ways to initialize the environment."""
        a_e_string = (
            "https://01234567-89ab-cdef-0123-456789abcdef-us-east1"
            ".apps.astra.datastax.com"
        )
        a_e_string_2 = (
            "https://98765432-10fe-dcba-9876-543210fedcba-us-east1"
            ".apps.astra.datastax.com"
        )
        mock_astra_db = AstraDB(
            token="t",
            api_endpoint=a_e_string,
            namespace="n",
        )

        ENV_VARS_TO_RESTORE = {}
        try:
            # clean environment
            if TOKEN_ENV_VAR in os.environ:
                ENV_VARS_TO_RESTORE[TOKEN_ENV_VAR] = os.environ[TOKEN_ENV_VAR]
                del os.environ[TOKEN_ENV_VAR]
            if API_ENDPOINT_ENV_VAR in os.environ:
                ENV_VARS_TO_RESTORE[API_ENDPOINT_ENV_VAR] = os.environ[
                    API_ENDPOINT_ENV_VAR
                ]
                del os.environ[API_ENDPOINT_ENV_VAR]
            if NAMESPACE_ENV_VAR in os.environ:
                ENV_VARS_TO_RESTORE[NAMESPACE_ENV_VAR] = os.environ[NAMESPACE_ENV_VAR]
                del os.environ[NAMESPACE_ENV_VAR]

            # token+endpoint
            env1 = _AstraDBEnvironment(
                token="t",
                api_endpoint=a_e_string,
                namespace="n",
            )

            # through a core AstraDB instance
            with pytest.warns(DeprecationWarning):
                env2 = _AstraDBEnvironment(astra_db_client=mock_astra_db)

            assert env1.data_api_client == env2.data_api_client
            assert env1.database == env2.database
            assert env1.async_database == env2.async_database

            # token+endpoint, but also a ready-made client
            with pytest.raises(ValueError):
                _AstraDBEnvironment(
                    token="t",
                    api_endpoint=a_e_string,
                    astra_db_client=mock_astra_db,
                )
            with pytest.raises(ValueError):
                _AstraDBEnvironment(
                    token="t",
                    api_endpoint=a_e_string,
                    async_astra_db_client=mock_astra_db.to_async(),
                )

            # just tokenn, no endpoint
            with pytest.raises(ValueError):
                _AstraDBEnvironment(
                    token="t",
                )

            # just client(s)
            with pytest.warns(DeprecationWarning):
                env3 = _AstraDBEnvironment(
                    async_astra_db_client=mock_astra_db.to_async(),
                )
            assert env1.data_api_client == env3.data_api_client
            assert env1.database == env3.database
            assert env1.async_database == env3.async_database

            # both sync and async (matching)
            with pytest.warns(DeprecationWarning):
                _AstraDBEnvironment(
                    astra_db_client=mock_astra_db,
                    async_astra_db_client=mock_astra_db.to_async(),
                )

            # both sync and async, but mismatching in various ways
            with pytest.raises(ValueError):
                with pytest.warns(DeprecationWarning):
                    _AstraDBEnvironment(
                        async_astra_db_client=mock_astra_db.to_async(),
                        astra_db_client=AstraDB(
                            token="t",
                            api_endpoint=a_e_string_2,
                            namespace="n",
                        ),
                    )
            with pytest.raises(ValueError):
                with pytest.warns(DeprecationWarning):
                    _AstraDBEnvironment(
                        async_astra_db_client=mock_astra_db.to_async(),
                        astra_db_client=AstraDB(
                            token="t",
                            api_endpoint=a_e_string,
                            namespace="n2",
                        ),
                    )
            with pytest.raises(ValueError):
                with pytest.warns(DeprecationWarning):
                    _AstraDBEnvironment(
                        async_astra_db_client=mock_astra_db.to_async(),
                        astra_db_client=AstraDB(
                            token="t2",
                            api_endpoint=a_e_string,
                            namespace="n",
                        ),
                    )

            # token+client
            with pytest.raises(ValueError):
                _AstraDBEnvironment(
                    token="t",
                    astra_db_client=mock_astra_db,
                )
            # endpoint+client
            with pytest.raises(ValueError):
                _AstraDBEnvironment(
                    api_endpoint=a_e_string,
                    async_astra_db_client=mock_astra_db.to_async(),
                )

            # token via environment variable:
            os.environ[TOKEN_ENV_VAR] = "t"
            env4 = _AstraDBEnvironment(
                api_endpoint=a_e_string,
                namespace="n",
            )
            del os.environ[TOKEN_ENV_VAR]
            assert env1.data_api_client == env4.data_api_client
            assert env1.database == env4.database
            assert env1.async_database == env4.async_database

            # endpoint via environment variable:
            os.environ[API_ENDPOINT_ENV_VAR] = a_e_string
            env5 = _AstraDBEnvironment(
                token="t",
                namespace="n",
            )
            del os.environ[API_ENDPOINT_ENV_VAR]
            assert env1.data_api_client == env5.data_api_client
            assert env1.database == env5.database
            assert env1.async_database == env5.async_database

            # both and also namespace via env vars
            os.environ[TOKEN_ENV_VAR] = "t"
            os.environ[API_ENDPOINT_ENV_VAR] = a_e_string
            os.environ[NAMESPACE_ENV_VAR] = "n"
            env6 = _AstraDBEnvironment()
            assert env1.data_api_client == env6.data_api_client
            assert env1.database == env6.database
            assert env1.async_database == env6.async_database
            del os.environ[TOKEN_ENV_VAR]
            del os.environ[API_ENDPOINT_ENV_VAR]
            del os.environ[NAMESPACE_ENV_VAR]

            # env vars do not interfere if client(s) passed
            os.environ[TOKEN_ENV_VAR] = "NO!"
            os.environ[API_ENDPOINT_ENV_VAR] = "NO!"
            os.environ[NAMESPACE_ENV_VAR] = "NO!"
            with pytest.warns(DeprecationWarning):
                env7a = _AstraDBEnvironment(
                    async_astra_db_client=mock_astra_db.to_async(),
                )
            with pytest.warns(DeprecationWarning):
                env7b = _AstraDBEnvironment(
                    astra_db_client=mock_astra_db,
                )
            with pytest.warns(DeprecationWarning):
                env7c = _AstraDBEnvironment(
                    astra_db_client=mock_astra_db,
                    async_astra_db_client=mock_astra_db.to_async(),
                )
            assert env1.data_api_client == env7a.data_api_client
            assert env1.database == env7a.database
            assert env1.async_database == env7a.async_database
            assert env1.data_api_client == env7b.data_api_client
            assert env1.database == env7b.database
            assert env1.async_database == env7b.async_database
            assert env1.data_api_client == env7c.data_api_client
            assert env1.database == env7c.database
            assert env1.async_database == env7c.async_database

            # env. vars do not interfere if parameters passed
            env8 = _AstraDBEnvironment(
                token="t",
                api_endpoint=a_e_string,
                namespace="n",
            )
            assert env1.data_api_client == env8.data_api_client
            assert env1.database == env8.database
            assert env1.async_database == env8.async_database

        finally:
            # reinstate the env. variables to what they were before this test:
            if TOKEN_ENV_VAR in os.environ:
                del os.environ[TOKEN_ENV_VAR]
            if API_ENDPOINT_ENV_VAR in os.environ:
                del os.environ[API_ENDPOINT_ENV_VAR]
            if NAMESPACE_ENV_VAR in os.environ:
                del os.environ[NAMESPACE_ENV_VAR]
            for ENV_VAR_NAME, ENV_VAR_VALUE in ENV_VARS_TO_RESTORE.items():
                os.environ[ENV_VAR_NAME] = ENV_VAR_VALUE
