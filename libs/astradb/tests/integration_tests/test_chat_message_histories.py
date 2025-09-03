"""Test of Astra DB chat message history class `AstraDBChatMessageHistory`"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_astradb.chat_message_histories import (
    AstraDBChatMessageHistory,
)
from langchain_astradb.utils.astradb import SetupMode

from .conftest import (
    AstraDBCredentials,
    astra_db_env_vars_available,
)

if TYPE_CHECKING:
    from astrapy import Collection


@pytest.fixture
def history1(
    astra_db_credentials: AstraDBCredentials,
    empty_collection_idxall: Collection,
) -> AstraDBChatMessageHistory:
    return AstraDBChatMessageHistory(
        session_id="session-test-1",
        collection_name=empty_collection_idxall.name,
        token=astra_db_credentials["token"],
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
    )


@pytest.fixture
def history2(
    astra_db_credentials: AstraDBCredentials,
    history1: AstraDBChatMessageHistory,
) -> AstraDBChatMessageHistory:
    return AstraDBChatMessageHistory(
        session_id="session-test-2",
        collection_name=history1.collection_name,
        token=astra_db_credentials["token"],
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
        # this, with the dependency from history1, ensures
        # no two createCollection calls at once are issued:
        setup_mode=SetupMode.OFF,
    )


@pytest.fixture
async def async_history1(
    astra_db_credentials: AstraDBCredentials,
    history1: AstraDBChatMessageHistory,
) -> AstraDBChatMessageHistory:
    return AstraDBChatMessageHistory(
        session_id="async-session-test-1",
        collection_name=history1.collection_name,
        token=astra_db_credentials["token"],
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
        setup_mode=SetupMode.OFF,
    )


@pytest.fixture
async def async_history2(
    astra_db_credentials: AstraDBCredentials,
    history1: AstraDBChatMessageHistory,
) -> AstraDBChatMessageHistory:
    return AstraDBChatMessageHistory(
        session_id="async-session-test-2",
        collection_name=history1.collection_name,
        token=astra_db_credentials["token"],
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
        # this, with the dependency from history1, ensures
        # no two createCollection calls at once are issued:
        setup_mode=SetupMode.OFF,
    )


@pytest.mark.skipif(
    not astra_db_env_vars_available(), reason="Missing Astra DB env. vars"
)
class TestAstraDBChatMessageHistories:
    def test_memory_with_message_store(
        self, history1: AstraDBChatMessageHistory
    ) -> None:
        """Test the memory with a message store."""

        assert history1.messages == []

        # add some messages
        history1.add_messages(
            [
                AIMessage(content="This is me, the AI"),
                HumanMessage(content="This is me, the human"),
            ]
        )

        messages = history1.messages
        expected = [
            AIMessage(content="This is me, the AI"),
            HumanMessage(content="This is me, the human"),
        ]
        assert messages == expected

        # clear the store
        history1.clear()

        assert history1.messages == []

    async def test_memory_with_message_store_async(
        self,
        async_history1: AstraDBChatMessageHistory,
    ) -> None:
        """Test the memory with a message store."""

        assert await async_history1.aget_messages() == []

        # add some messages
        await async_history1.aadd_messages(
            [
                AIMessage(content="This is me, the AI"),
                HumanMessage(content="This is me, the human"),
            ]
        )

        messages = await async_history1.aget_messages()
        expected = [
            AIMessage(content="This is me, the AI"),
            HumanMessage(content="This is me, the human"),
        ]
        assert messages == expected

        # clear the store
        await async_history1.aclear()

        assert await async_history1.aget_messages() == []

    def test_memory_separate_session_ids(
        self, history1: AstraDBChatMessageHistory, history2: AstraDBChatMessageHistory
    ) -> None:
        """Test that separate session IDs do not share entries."""

        history1.add_messages([AIMessage(content="Just saying.")])
        assert history2.messages == []
        history2.clear()
        assert history1.messages != []
        history1.clear()
        assert history1.messages == []

    async def test_memory_separate_session_ids_async(
        self,
        async_history1: AstraDBChatMessageHistory,
        async_history2: AstraDBChatMessageHistory,
    ) -> None:
        """Test that separate session IDs do not share entries."""

        await async_history1.aadd_messages([AIMessage(content="Just saying.")])
        assert await async_history2.aget_messages() == []
        await async_history2.aclear()
        assert await async_history1.aget_messages() != []
        await async_history1.aclear()
        assert await async_history1.aget_messages() == []
