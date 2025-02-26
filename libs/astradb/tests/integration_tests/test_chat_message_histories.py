import os

import pytest
from astrapy import Collection
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage

from langchain_astradb.chat_message_histories import (
    AstraDBChatMessageHistory,
)
from langchain_astradb.utils.astradb import SetupMode

from .conftest import (
    AstraDBCredentials,
    astra_db_env_vars_available,
)


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
        memory = ConversationBufferMemory(
            memory_key="baz",
            chat_memory=history1,
            return_messages=True,
        )

        assert memory.chat_memory.messages == []

        # add some messages
        memory.chat_memory.add_messages(
            [
                AIMessage(content="This is me, the AI"),
                HumanMessage(content="This is me, the human"),
            ]
        )

        messages = memory.chat_memory.messages
        expected = [
            AIMessage(content="This is me, the AI"),
            HumanMessage(content="This is me, the human"),
        ]
        assert messages == expected

        # clear the store
        memory.chat_memory.clear()

        assert memory.chat_memory.messages == []

    async def test_memory_with_message_store_async(
        self,
        async_history1: AstraDBChatMessageHistory,
    ) -> None:
        """Test the memory with a message store."""
        memory = ConversationBufferMemory(
            memory_key="baz",
            chat_memory=async_history1,
            return_messages=True,
        )

        assert await memory.chat_memory.aget_messages() == []

        # add some messages
        await memory.chat_memory.aadd_messages(
            [
                AIMessage(content="This is me, the AI"),
                HumanMessage(content="This is me, the human"),
            ]
        )

        messages = await memory.chat_memory.aget_messages()
        expected = [
            AIMessage(content="This is me, the AI"),
            HumanMessage(content="This is me, the human"),
        ]
        assert messages == expected

        # clear the store
        await memory.chat_memory.aclear()

        assert await memory.chat_memory.aget_messages() == []

    def test_memory_separate_session_ids(
        self, history1: AstraDBChatMessageHistory, history2: AstraDBChatMessageHistory
    ) -> None:
        """Test that separate session IDs do not share entries."""
        memory1 = ConversationBufferMemory(
            memory_key="mk1",
            chat_memory=history1,
            return_messages=True,
        )
        memory2 = ConversationBufferMemory(
            memory_key="mk2",
            chat_memory=history2,
            return_messages=True,
        )

        memory1.chat_memory.add_messages([AIMessage(content="Just saying.")])
        assert memory2.chat_memory.messages == []
        memory2.chat_memory.clear()
        assert memory1.chat_memory.messages != []
        memory1.chat_memory.clear()
        assert memory1.chat_memory.messages == []

    async def test_memory_separate_session_ids_async(
        self,
        async_history1: AstraDBChatMessageHistory,
        async_history2: AstraDBChatMessageHistory,
    ) -> None:
        """Test that separate session IDs do not share entries."""
        memory1 = ConversationBufferMemory(
            memory_key="mk1",
            chat_memory=async_history1,
            return_messages=True,
        )
        memory2 = ConversationBufferMemory(
            memory_key="mk2",
            chat_memory=async_history2,
            return_messages=True,
        )

        await memory1.chat_memory.aadd_messages([AIMessage(content="Just saying.")])
        assert await memory2.chat_memory.aget_messages() == []
        await memory2.chat_memory.aclear()
        assert await memory1.chat_memory.aget_messages() != []
        await memory1.chat_memory.aclear()
        assert await memory1.chat_memory.aget_messages() == []
