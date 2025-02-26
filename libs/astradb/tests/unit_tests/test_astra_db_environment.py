import os

import pytest
from astrapy.constants import Environment

from langchain_astradb.utils.astradb import (
    API_ENDPOINT_ENV_VAR,
    KEYSPACE_ENV_VAR,
    TOKEN_ENV_VAR,
    _AstraDBEnvironment,
)

FAKE_TOKEN = "t"  # noqa: S105


CLIENT_PARAM_CONFLICT_MSG = (
    "You cannot pass 'astra_db_client' or 'async_astra_db_client' to "
    "AstraDBEnvironment if passing 'token', 'api_endpoint' or 'environment'."
)


class TestAstraDBEnvironment:
    def test_initialization(self) -> None:
        """Test the various ways to initialize the environment."""
        a_e_string = (
            "https://01234567-89ab-cdef-0123-456789abcdef-us-east1"
            ".apps.astra.datastax.com"
        )

        env_vars_to_restore = {}
        try:
            # clean environment
            if TOKEN_ENV_VAR in os.environ:
                env_vars_to_restore[TOKEN_ENV_VAR] = os.environ[TOKEN_ENV_VAR]
                del os.environ[TOKEN_ENV_VAR]
            if API_ENDPOINT_ENV_VAR in os.environ:
                env_vars_to_restore[API_ENDPOINT_ENV_VAR] = os.environ[
                    API_ENDPOINT_ENV_VAR
                ]
                del os.environ[API_ENDPOINT_ENV_VAR]
            if KEYSPACE_ENV_VAR in os.environ:
                env_vars_to_restore[KEYSPACE_ENV_VAR] = os.environ[KEYSPACE_ENV_VAR]
                del os.environ[KEYSPACE_ENV_VAR]

            # token+endpoint
            env1 = _AstraDBEnvironment(
                token=FAKE_TOKEN,
                api_endpoint=a_e_string,
                keyspace="n",
            )

            # just tokenn, no endpoint
            with pytest.raises(
                ValueError, match="API endpoint for Data API not provided."
            ):
                _AstraDBEnvironment(
                    token=FAKE_TOKEN,
                )

            # token via environment variable:
            os.environ[TOKEN_ENV_VAR] = "t"
            env4 = _AstraDBEnvironment(
                api_endpoint=a_e_string,
                keyspace="n",
            )
            del os.environ[TOKEN_ENV_VAR]
            assert env1.data_api_client == env4.data_api_client
            assert env1.database == env4.database
            assert env1.async_database == env4.async_database

            # endpoint via environment variable:
            os.environ[API_ENDPOINT_ENV_VAR] = a_e_string
            env5 = _AstraDBEnvironment(
                token=FAKE_TOKEN,
                keyspace="n",
            )
            del os.environ[API_ENDPOINT_ENV_VAR]
            assert env1.data_api_client == env5.data_api_client
            assert env1.database == env5.database
            assert env1.async_database == env5.async_database

            # both and also namespace via env vars
            os.environ[TOKEN_ENV_VAR] = FAKE_TOKEN
            os.environ[API_ENDPOINT_ENV_VAR] = a_e_string
            os.environ[KEYSPACE_ENV_VAR] = "n"
            env6 = _AstraDBEnvironment()
            assert env1.data_api_client == env6.data_api_client
            assert env1.database == env6.database
            assert env1.async_database == env6.async_database
            del os.environ[TOKEN_ENV_VAR]
            del os.environ[API_ENDPOINT_ENV_VAR]
            del os.environ[KEYSPACE_ENV_VAR]

            # env. vars do not interfere if parameters passed
            os.environ[TOKEN_ENV_VAR] = "NO!"
            os.environ[API_ENDPOINT_ENV_VAR] = "NO!"
            os.environ[KEYSPACE_ENV_VAR] = "NO!"
            env8 = _AstraDBEnvironment(
                token=FAKE_TOKEN,
                api_endpoint=a_e_string,
                keyspace="n",
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
            if KEYSPACE_ENV_VAR in os.environ:
                del os.environ[KEYSPACE_ENV_VAR]
            for env_var_name, env_var_value in env_vars_to_restore.items():
                os.environ[env_var_name] = env_var_value

    def test_env_autodetect(self) -> None:
        a_e_string_prod = (
            "https://01234567-89ab-cdef-0123-456789abcdef-us-east1"
            ".apps.astra.datastax.com"
        )
        a_e_string_dev = (
            "https://01234567-89ab-cdef-0123-456789abcdef-us-east1"
            ".apps.astra-dev.datastax.com"
        )
        a_e_string_other = "http://localhost:1234"

        a_env_prod = _AstraDBEnvironment(
            token=FAKE_TOKEN,
            api_endpoint=a_e_string_prod,
            keyspace="n",
        )
        assert a_env_prod.environment == Environment.PROD
        a_env_dev = _AstraDBEnvironment(
            token=FAKE_TOKEN,
            api_endpoint=a_e_string_dev,
            keyspace="n",
        )
        assert a_env_dev.environment == Environment.DEV
        a_env_other = _AstraDBEnvironment(
            token=FAKE_TOKEN,
            api_endpoint=a_e_string_other,
            keyspace="n",
        )
        assert a_env_other.environment == Environment.OTHER

        # a funny case
        with pytest.raises(ValueError, match="mismatch"):
            _AstraDBEnvironment(
                token=FAKE_TOKEN,
                api_endpoint=a_e_string_prod,
                keyspace="n",
                environment=Environment.DEV,
            )
