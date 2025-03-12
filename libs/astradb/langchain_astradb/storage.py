"""Astra DB - based storages."""

from __future__ import annotations

import asyncio
import base64
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Generic,
    Iterator,
    Sequence,
    TypeVar,
)

from astrapy.exceptions import CollectionInsertManyException
from langchain_core.stores import BaseStore, ByteStore
from typing_extensions import override

from langchain_astradb.utils.astradb import (
    COMPONENT_NAME_BYTESTORE,
    COMPONENT_NAME_STORE,
    MAX_CONCURRENT_DOCUMENT_INSERTIONS,
    MAX_CONCURRENT_DOCUMENT_REPLACEMENTS,
    SetupMode,
    _AstraDBCollectionEnvironment,
)

if TYPE_CHECKING:
    from astrapy.authentication import TokenProvider
    from astrapy.results import CollectionUpdateResult

V = TypeVar("V")


class AstraDBBaseStore(BaseStore[str, V], Generic[V]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Base class for the DataStax Astra DB data store."""
        if "requested_indexing_policy" in kwargs:
            msg = "Do not pass 'requested_indexing_policy' to AstraDBBaseStore init"
            raise ValueError(msg)
        if "default_indexing_policy" in kwargs:
            msg = "Do not pass 'default_indexing_policy' to AstraDBBaseStore init"
            raise ValueError(msg)
        kwargs["requested_indexing_policy"] = {"allow": ["_id"]}
        kwargs["default_indexing_policy"] = {"allow": ["_id"]}

        if "namespace" in kwargs:
            kwargs["keyspace"] = kwargs.pop("namespace")

        self.astra_env = _AstraDBCollectionEnvironment(
            *args,
            **kwargs,
        )
        self.collection = self.astra_env.collection
        self.async_collection = self.astra_env.async_collection

    @abstractmethod
    def decode_value(self, value: Any) -> V | None:  # noqa: ANN401
        """Decodes value from Astra DB."""

    @abstractmethod
    def encode_value(self, value: V | None) -> Any:  # noqa: ANN401
        """Encodes value for Astra DB."""

    @override
    def mget(self, keys: Sequence[str]) -> list[V | None]:
        self.astra_env.ensure_db_setup()
        docs_dict = {}
        for doc in self.collection.find(
            filter={"_id": {"$in": list(keys)}},
            projection={"*": True},
        ):
            docs_dict[doc["_id"]] = doc.get("value")
        return [self.decode_value(docs_dict.get(key)) for key in keys]

    @override
    async def amget(self, keys: Sequence[str]) -> list[V | None]:
        await self.astra_env.aensure_db_setup()
        docs_dict = {}
        async for doc in self.async_collection.find(
            filter={"_id": {"$in": list(keys)}},
            projection={"*": True},
        ):
            docs_dict[doc["_id"]] = doc.get("value")
        return [self.decode_value(docs_dict.get(key)) for key in keys]

    @override
    def mset(self, key_value_pairs: Sequence[tuple[str, V]]) -> None:
        self.astra_env.ensure_db_setup()
        documents_to_insert = [
            {"_id": k, "value": self.encode_value(v)} for k, v in key_value_pairs
        ]
        # perform an AstraPy insert_many, catching exceptions for overwriting docs
        ids_to_replace: list[int]
        try:
            self.collection.insert_many(
                documents_to_insert,
                ordered=False,
                concurrency=MAX_CONCURRENT_DOCUMENT_INSERTIONS,
            )
            ids_to_replace = []
        except CollectionInsertManyException as err:
            inserted_ids_set = set(err.inserted_ids)
            ids_to_replace = [
                document["_id"]
                for document in documents_to_insert
                if document["_id"] not in inserted_ids_set
            ]

        # if necessary, replace docs for the non-inserted ids
        if ids_to_replace:
            documents_to_replace = [
                document
                for document in documents_to_insert
                if document["_id"] in ids_to_replace
            ]

            with ThreadPoolExecutor(
                max_workers=MAX_CONCURRENT_DOCUMENT_REPLACEMENTS
            ) as executor:

                def _replace_document(
                    document: dict[str, Any],
                ) -> CollectionUpdateResult:
                    return self.collection.replace_one(
                        {"_id": document["_id"]},
                        document,
                    )

                replace_results = list(
                    executor.map(
                        _replace_document,
                        documents_to_replace,
                    )
                )

            replaced_count = sum(r_res.update_info["n"] for r_res in replace_results)
            if replaced_count != len(ids_to_replace):
                missing = len(ids_to_replace) - replaced_count
                msg = (
                    "AstraDBBaseStore.mset could not insert all requested "
                    f"documents ({missing} failed replace_one calls)"
                )
                raise ValueError(msg)

    @override
    async def amset(self, key_value_pairs: Sequence[tuple[str, V]]) -> None:
        await self.astra_env.aensure_db_setup()
        documents_to_insert = [
            {"_id": k, "value": self.encode_value(v)} for k, v in key_value_pairs
        ]
        # perform an AstraPy insert_many, catching exceptions for overwriting docs
        ids_to_replace: list[int]
        try:
            await self.async_collection.insert_many(
                documents_to_insert,
                ordered=False,
            )
            ids_to_replace = []
        except CollectionInsertManyException as err:
            inserted_ids_set = set(err.inserted_ids)
            ids_to_replace = [
                document["_id"]
                for document in documents_to_insert
                if document["_id"] not in inserted_ids_set
            ]

        # if necessary, replace docs for the non-inserted ids
        if ids_to_replace:
            documents_to_replace = [
                document
                for document in documents_to_insert
                if document["_id"] in ids_to_replace
            ]

            sem = asyncio.Semaphore(MAX_CONCURRENT_DOCUMENT_REPLACEMENTS)

            _async_collection = self.async_collection

            async def _replace_document(
                document: dict[str, Any],
            ) -> CollectionUpdateResult:
                async with sem:
                    return await _async_collection.replace_one(
                        {"_id": document["_id"]},
                        document,
                    )

            tasks = [
                asyncio.create_task(_replace_document(document))
                for document in documents_to_replace
            ]

            replace_results = await asyncio.gather(*tasks, return_exceptions=False)

            replaced_count = sum(r_res.update_info["n"] for r_res in replace_results)
            if replaced_count != len(ids_to_replace):
                missing = len(ids_to_replace) - replaced_count
                msg = (
                    "AstraDBBaseStore.mset could not insert all requested "
                    f"documents ({missing} failed replace_one calls)"
                )
                raise ValueError(msg)

    @override
    def mdelete(self, keys: Sequence[str]) -> None:
        self.astra_env.ensure_db_setup()
        self.collection.delete_many(filter={"_id": {"$in": list(keys)}})

    @override
    async def amdelete(self, keys: Sequence[str]) -> None:
        await self.astra_env.aensure_db_setup()
        await self.async_collection.delete_many(filter={"_id": {"$in": list(keys)}})

    @override
    def yield_keys(self, *, prefix: str | None = None) -> Iterator[str]:
        self.astra_env.ensure_db_setup()
        docs = self.collection.find()
        for doc in docs:
            key = doc["_id"]
            if not prefix or key.startswith(prefix):
                yield key

    @override
    async def ayield_keys(self, *, prefix: str | None = None) -> AsyncIterator[str]:
        await self.astra_env.aensure_db_setup()
        async for doc in self.async_collection.find():
            key = doc["_id"]
            if not prefix or key.startswith(prefix):
                yield key


class AstraDBStore(AstraDBBaseStore[Any]):
    def __init__(
        self,
        collection_name: str,
        *,
        token: str | TokenProvider | None = None,
        api_endpoint: str | None = None,
        namespace: str | None = None,
        environment: str | None = None,
        pre_delete_collection: bool = False,
        setup_mode: SetupMode = SetupMode.SYNC,
        ext_callers: list[tuple[str | None, str | None] | str | None] | None = None,
    ) -> None:
        """BaseStore implementation using DataStax AstraDB as the underlying store.

        The value type can be any type serializable by json.dumps.
        Can be used to store embeddings with the CacheBackedEmbeddings.

        Documents in the AstraDB collection will have the format

        .. code-block:: json

            {
              "_id": "<key>",
              "value": <value>
            }

        Args:
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
        super().__init__(
            collection_name=collection_name,
            token=token,
            api_endpoint=api_endpoint,
            namespace=namespace,
            environment=environment,
            setup_mode=setup_mode,
            pre_delete_collection=pre_delete_collection,
            ext_callers=ext_callers,
            component_name=COMPONENT_NAME_STORE,
        )

    @override
    def decode_value(self, value: Any) -> Any:
        return value

    @override
    def encode_value(self, value: Any) -> Any:
        return value


class AstraDBByteStore(AstraDBBaseStore[bytes], ByteStore):
    def __init__(
        self,
        *,
        collection_name: str,
        token: str | TokenProvider | None = None,
        api_endpoint: str | None = None,
        namespace: str | None = None,
        environment: str | None = None,
        pre_delete_collection: bool = False,
        setup_mode: SetupMode = SetupMode.SYNC,
        ext_callers: list[tuple[str | None, str | None] | str | None] | None = None,
    ) -> None:
        """ByteStore implementation using DataStax AstraDB as the underlying store.

        The bytes values are converted to base64 encoded strings
        Documents in the AstraDB collection will have the format

        .. code-block:: json
            {
              "_id": "<key>",
              "value": "<byte64 string value>"
            }

        Args:
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
        super().__init__(
            collection_name=collection_name,
            token=token,
            api_endpoint=api_endpoint,
            namespace=namespace,
            environment=environment,
            setup_mode=setup_mode,
            pre_delete_collection=pre_delete_collection,
            ext_callers=ext_callers,
            component_name=COMPONENT_NAME_BYTESTORE,
        )

    @override
    def decode_value(self, value: Any) -> bytes | None:
        if value is None:
            return None
        return base64.b64decode(value)

    @override
    def encode_value(self, value: bytes | None) -> Any:
        if value is None:
            return None
        return base64.b64encode(value).decode("ascii")
