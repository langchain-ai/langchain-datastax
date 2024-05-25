import os
from unittest.mock import Mock

import pytest

from langchain_astradb.utils.astradb import (
    API_ENDPOINT_ENV_VAR,
    NAMESPACE_ENV_VAR,
    TOKEN_ENV_VAR,
    _AstraDBEnvironment,
)


class TestAstraDBEnvironment:
    def test_initialization(self) -> None:
        """Test the various ways to initialize the environment."""

        # clean environment
        if TOKEN_ENV_VAR in os.environ:
            del os.environ[TOKEN_ENV_VAR]
        if API_ENDPOINT_ENV_VAR in os.environ:
            del os.environ[API_ENDPOINT_ENV_VAR]
        if NAMESPACE_ENV_VAR in os.environ:
            del os.environ[NAMESPACE_ENV_VAR]

        # token+endpoint
        env1 = _AstraDBEnvironment(
            token="t",
            api_endpoint="ae",
        )
        assert env1.async_astra_db.token == "t"
        assert env1.async_astra_db.api_endpoint == "ae"

        # token+endpoint, but also a ready-made client
        with pytest.raises(ValueError):
            _AstraDBEnvironment(
                token="t",
                api_endpoint="ae",
                astra_db_client=Mock(),
            )
        with pytest.raises(ValueError):
            _AstraDBEnvironment(
                token="t",
                api_endpoint="ae",
                async_astra_db_client=Mock(),
            )

        # just token or endpoint
        with pytest.raises(ValueError):
            _AstraDBEnvironment(
                token="t",
            )
        with pytest.raises(ValueError):
            _AstraDBEnvironment(
                api_endpoint="ae",
            )

        # just client(s)
        _AstraDBEnvironment(
            async_astra_db_client=Mock(),
        )
        _AstraDBEnvironment(
            astra_db_client=Mock(),
        )
        _AstraDBEnvironment(
            astra_db_client=Mock(),
            async_astra_db_client=Mock(),
        )

        # token+client
        with pytest.raises(ValueError):
            _AstraDBEnvironment(
                token="t",
                astra_db_client=Mock(),
            )
        # endpoint+client
        with pytest.raises(ValueError):
            _AstraDBEnvironment(
                api_endpoint="ae",
                async_astra_db_client=Mock(),
            )

        # token via environment variable:
        os.environ[TOKEN_ENV_VAR] = "T"
        env3 = _AstraDBEnvironment(
            api_endpoint="ae",
        )
        assert env3.async_astra_db.token == "T"
        assert env3.async_astra_db.api_endpoint == "ae"

        # endpoint via environment variable:
        del os.environ[TOKEN_ENV_VAR]
        os.environ[API_ENDPOINT_ENV_VAR] = "AE"
        env4 = _AstraDBEnvironment(
            token="t",
        )
        assert env4.async_astra_db.token == "t"
        assert env4.async_astra_db.api_endpoint == "AE"

        # both via env vars
        os.environ[TOKEN_ENV_VAR] = "T"
        os.environ[API_ENDPOINT_ENV_VAR] = "AE"
        env5 = _AstraDBEnvironment()
        assert env5.async_astra_db.token == "T"
        assert env5.async_astra_db.api_endpoint == "AE"

        # env vars do not interfere if client(s) passed
        env6a = _AstraDBEnvironment(
            async_astra_db_client=Mock(),
        )
        env6b = _AstraDBEnvironment(
            astra_db_client=Mock(),
        )
        env6c = _AstraDBEnvironment(
            astra_db_client=Mock(),
            async_astra_db_client=Mock(),
        )
        assert env6a.astra_db.token != "T"
        assert env6b.astra_db.token != "T"
        assert env6c.astra_db.token != "T"
        assert env6a.astra_db.api_endpoint != "AE"
        assert env6b.astra_db.api_endpoint != "AE"
        assert env6c.astra_db.api_endpoint != "AE"

        # env. vars do not interfere if parameters passed
        env7a = _AstraDBEnvironment(
            token="t",
            api_endpoint="ae",
        )
        assert env7a.async_astra_db.token == "t"
        assert env7a.async_astra_db.api_endpoint == "ae"
        env7b = _AstraDBEnvironment(
            api_endpoint="ae",
        )
        assert env7b.async_astra_db.token == "T"
        assert env7b.async_astra_db.api_endpoint == "ae"
        env7c = _AstraDBEnvironment(
            token="t",
        )
        assert env7c.async_astra_db.token == "t"
        assert env7c.async_astra_db.api_endpoint == "AE"

        # namespaces through env. vars
        os.environ[NAMESPACE_ENV_VAR] = "NS"
        env8 = _AstraDBEnvironment(
            token="t",
            api_endpoint="ae",
        )
        assert env8.astra_db.namespace == "NS"
