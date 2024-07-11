import os
from typing import AsyncIterable, Dict, Iterable, Optional

import pytest
from astrapy.db import AstraDB
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage

from langchain_astradb.chat_message_histories import (
    AstraDBChatMessageHistory,
)
from langchain_astradb.utils.astradb import SetupMode

from .conftest import _has_env_vars


@pytest.fixture(scope="function")
def history1(
    astra_db_credentials: Dict[str, Optional[str]]
) -> Iterable[AstraDBChatMessageHistory]:
    history1 = AstraDBChatMessageHistory(
        session_id="session-test-1",
        collection_name="langchain_cmh_test",
        token=astra_db_credentials["token"],
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
    )
    yield history1
    history1.collection.drop()


@pytest.fixture(scope="function")
def history2(
    history1: AstraDBChatMessageHistory,
    astra_db_credentials: Dict[str, Optional[str]],
) -> Iterable[AstraDBChatMessageHistory]:
    history2 = AstraDBChatMessageHistory(
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
    yield history2
    # no deletion here, this is riding on history1


@pytest.fixture
async def async_history1(
    astra_db_credentials: Dict[str, Optional[str]]
) -> AsyncIterable[AstraDBChatMessageHistory]:
    history1 = AstraDBChatMessageHistory(
        session_id="async-session-test-1",
        collection_name="langchain_cmh_test",
        token=astra_db_credentials["token"],
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
        setup_mode=SetupMode.ASYNC,
    )
    yield history1
    await history1.async_collection.drop()


@pytest.fixture(scope="function")
async def async_history2(
    history1: AstraDBChatMessageHistory,
    astra_db_credentials: Dict[str, Optional[str]],
) -> AsyncIterable[AstraDBChatMessageHistory]:
    history2 = AstraDBChatMessageHistory(
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
    yield history2
    # no deletion here, this is riding on history1


@pytest.mark.skipif(not _has_env_vars(), reason="Missing Astra DB env. vars")
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

    @pytest.mark.skipif(
        os.environ.get("ASTRA_DB_ENVIRONMENT", "prod").upper() != "PROD",
        reason="Can run on Astra DB prod only",
    )
    def test_chatms_coreclients_init_sync(
        self,
        astra_db_credentials: Dict[str, Optional[str]],
        core_astra_db: AstraDB,
    ) -> None:
        """A deprecation warning from passing a (core) AstraDB, but it works."""
        collection_name = "lc_test_cmh_coreclsync"
        test_messages=[AIMessage(content="Meow.")]
        try:
            chatmh_init_ok = AstraDBChatMessageHistory(
                session_id="gattini",
                collection_name=collection_name,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
            )
            chatmh_init_ok.add_messages(test_messages)
            # create an equivalent cache with core AstraDB in init
            with pytest.warns(DeprecationWarning) as rec_warnings:
                chatmh_init_core = AstraDBChatMessageHistory(
                    collection_name=collection_name,
                    session_id="gattini",
                    astra_db_client=core_astra_db,
                )
            assert len(rec_warnings) == 1
            assert chatmh_init_core.messages == test_messages
        finally:
            chatmh_init_ok.astra_env.collection.drop()

    @pytest.mark.skipif(
        os.environ.get("ASTRA_DB_ENVIRONMENT", "prod").upper() != "PROD",
        reason="Can run on Astra DB prod only",
    )
    async def test_chatms_coreclients_init_async(
        self,
        astra_db_credentials: Dict[str, Optional[str]],
        core_astra_db: AstraDB,
    ) -> None:
        """A deprecation warning from passing a (core) AstraDB, but it works."""
        collection_name = "lc_test_cmh_coreclasync"
        test_messages=[AIMessage(content="Ameow.")]
        try:
            chatmh_init_ok = AstraDBChatMessageHistory(
                session_id="gattini",
                collection_name=collection_name,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                setup_mode=SetupMode.ASYNC,
            )
            await chatmh_init_ok.aadd_messages(test_messages)
            # create an equivalent cache with core AstraDB in init
            with pytest.warns(DeprecationWarning) as rec_warnings:
                chatmh_init_core = AstraDBChatMessageHistory(
                    collection_name=collection_name,
                    session_id="gattini",
                    astra_db_client=core_astra_db,
                    setup_mode=SetupMode.ASYNC,
                )
            assert len(rec_warnings) == 1
            assert await chatmh_init_core.aget_messages() == test_messages
        finally:
            await chatmh_init_ok.astra_env.async_collection.drop()
