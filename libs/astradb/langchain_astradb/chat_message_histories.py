"""Astra DB - based chat message history, based on astrapy."""

from __future__ import annotations

import json
import time
from operator import itemgetter
from typing import TYPE_CHECKING, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)
from typing_extensions import override

from langchain_astradb.utils.astradb import (
    COMPONENT_NAME_CHATMESSAGEHISTORY,
    SetupMode,
    _AstraDBCollectionEnvironment,
)

if TYPE_CHECKING:
    from astrapy.authentication import TokenProvider

DEFAULT_COLLECTION_NAME = "langchain_message_store"


class AstraDBChatMessageHistory(BaseChatMessageHistory):
    def __init__(
        self,
        *,
        session_id: str,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        token: str | TokenProvider | None = None,
        api_endpoint: str | None = None,
        namespace: str | None = None,
        environment: str | None = None,
        setup_mode: SetupMode = SetupMode.SYNC,
        pre_delete_collection: bool = False,
        ext_callers: list[tuple[str | None, str | None] | str | None] | None = None,
    ) -> None:
        """Chat message history that stores history in Astra DB.

        Args:
            session_id: arbitrary key that is used to store the messages
                of a single chat session.
            collection_name: name of the Astra DB collection to create/use.
            token: API token for Astra DB usage, either in the form of a string
                or a subclass of `astrapy.authentication.TokenProvider`.
                If not provided, the environment variable
                ASTRA_DB_APPLICATION_TOKEN is inspected.
            api_endpoint: full URL to the API endpoint, such as
                `https://<DB-ID>-us-east1.apps.astra.datastax.com`. If not provided,
                the environment variable ASTRA_DB_API_ENDPOINT is inspected.
            namespace: namespace (aka keyspace) where the collection is created.
                If not provided, the environment variable ASTRA_DB_KEYSPACE is
                inspected. Defaults to the database's "default namespace".
            environment: a string specifying the environment of the target Data API.
                If omitted, defaults to "prod" (Astra DB production).
                Other values are in `astrapy.constants.Environment` enum class.
            setup_mode: mode used to create the Astra DB collection (SYNC, ASYNC or
                OFF).
            pre_delete_collection: whether to delete the collection
                before creating it. If False and the collection already exists,
                the collection will be used as is.
            ext_callers: one or more caller identities to identify Data API calls
                in the User-Agent header. This is a list of (name, version) pairs,
                or just strings if no version info is provided, which, if supplied,
                becomes the leading part of the User-Agent string in all API requests
                related to this component.
        """
        self.astra_env = _AstraDBCollectionEnvironment(
            collection_name=collection_name,
            token=token,
            api_endpoint=api_endpoint,
            keyspace=namespace,
            environment=environment,
            setup_mode=setup_mode,
            pre_delete_collection=pre_delete_collection,
            ext_callers=ext_callers,
            component_name=COMPONENT_NAME_CHATMESSAGEHISTORY,
        )

        self.collection = self.astra_env.collection
        self.async_collection = self.astra_env.async_collection

        self.session_id = session_id
        self.collection_name = collection_name

    @property
    def messages(self) -> list[BaseMessage]:
        """Retrieve all session messages from DB."""
        self.astra_env.ensure_db_setup()
        message_blobs = [
            doc["body_blob"]
            for doc in sorted(
                self.collection.find(
                    filter={
                        "session_id": self.session_id,
                    },
                    projection={
                        "timestamp": True,
                        "body_blob": True,
                    },
                ),
                key=itemgetter("timestamp"),
            )
        ]
        items = [json.loads(message_blob) for message_blob in message_blobs]
        return messages_from_dict(items)

    @messages.setter
    def messages(self, _: list[BaseMessage]) -> None:
        msg = "Use add_messages instead"
        raise NotImplementedError(msg)

    @override
    async def aget_messages(self) -> list[BaseMessage]:
        await self.astra_env.aensure_db_setup()
        docs = self.async_collection.find(
            filter={
                "session_id": self.session_id,
            },
            projection={
                "timestamp": True,
                "body_blob": True,
            },
        )
        sorted_docs = sorted(
            [doc async for doc in docs],
            key=itemgetter("timestamp"),
        )
        message_blobs = [doc["body_blob"] for doc in sorted_docs]
        items = [json.loads(message_blob) for message_blob in message_blobs]
        return messages_from_dict(items)

    @override
    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        self.astra_env.ensure_db_setup()
        docs = [
            {
                "timestamp": time.time(),
                "session_id": self.session_id,
                "body_blob": json.dumps(message_to_dict(message)),
            }
            for message in messages
        ]
        self.collection.insert_many(docs)

    @override
    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        await self.astra_env.aensure_db_setup()
        docs = [
            {
                "timestamp": time.time(),
                "session_id": self.session_id,
                "body_blob": json.dumps(message_to_dict(message)),
            }
            for message in messages
        ]
        await self.async_collection.insert_many(docs)

    @override
    def clear(self) -> None:
        self.astra_env.ensure_db_setup()
        self.collection.delete_many(filter={"session_id": self.session_id})

    @override
    async def aclear(self) -> None:
        await self.astra_env.aensure_db_setup()
        await self.async_collection.delete_many(filter={"session_id": self.session_id})
