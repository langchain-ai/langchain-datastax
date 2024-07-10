from __future__ import annotations

import asyncio
import base64
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

from astrapy.db import AstraDB, AsyncAstraDB
from astrapy.exceptions import InsertManyException
from astrapy.results import UpdateResult
from langchain_core.stores import BaseStore, ByteStore

from langchain_astradb.utils.astradb import (
    SetupMode,
    _AstraDBCollectionEnvironment,
    REPLACE_DOCUMENTS_MAX_THREADS,
)

V = TypeVar("V")


class AstraDBBaseStore(Generic[V], BaseStore[str, V], ABC):
    """Base class for the DataStax Astra DB data store."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "requested_indexing_policy" in kwargs:
            raise ValueError(
                "Do not pass 'requested_indexing_policy' to AstraDBBaseStore init"
            )
        if "default_indexing_policy" in kwargs:
            raise ValueError(
                "Do not pass 'default_indexing_policy' to AstraDBBaseStore init"
            )
        kwargs["requested_indexing_policy"] = {"allow": ["_id"]}
        kwargs["default_indexing_policy"] = {"allow": ["_id"]}
        self.astra_env = _AstraDBCollectionEnvironment(
            *args,
            **kwargs,
        )
        self.collection = self.astra_env.collection
        self.async_collection = self.astra_env.async_collection

    @abstractmethod
    def decode_value(self, value: Any) -> Optional[V]:
        """Decodes value from Astra DB"""

    @abstractmethod
    def encode_value(self, value: Optional[V]) -> Any:
        """Encodes value for Astra DB"""

    def mget(self, keys: Sequence[str]) -> List[Optional[V]]:
        self.astra_env.ensure_db_setup()
        docs_dict = {}
        for doc in self.collection.find(
            filter={"_id": {"$in": list(keys)}},
            projection={"*": True},
        ):
            docs_dict[doc["_id"]] = doc.get("value")
        return [self.decode_value(docs_dict.get(key)) for key in keys]

    async def amget(self, keys: Sequence[str]) -> List[Optional[V]]:
        await self.astra_env.aensure_db_setup()
        docs_dict = {}
        async for doc in self.async_collection.find(
            filter={"_id": {"$in": list(keys)}},
            projection={"*": True},
        ):
            docs_dict[doc["_id"]] = doc.get("value")
        return [self.decode_value(docs_dict.get(key)) for key in keys]

    def mset(self, key_value_pairs: Sequence[Tuple[str, V]]) -> None:
        self.astra_env.ensure_db_setup()
        documents_to_insert = [
            {"_id": k, "value": self.encode_value(v)}
            for k, v in key_value_pairs
        ]
        # perform an AstraPy insert_many, catching exceptions for overwriting docs
        ids_to_replace: List[int]
        try:
            self.collection.insert_many(
                documents_to_insert,
                ordered=False,
            )
            ids_to_replace = []
        except InsertManyException as err:
            inserted_ids_set = set(err.partial_result.inserted_ids)
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
                max_workers=REPLACE_DOCUMENTS_MAX_THREADS
            ) as executor:

                def _replace_document(document: Dict[str, Any]) -> UpdateResult:
                    return self.collection.replace_one(
                        {"_id": document["_id"]},
                        document,
                    )

                replace_results = executor.map(
                    _replace_document,
                    documents_to_replace,
                )

            replaced_count = sum(r_res.update_info["n"] for r_res in replace_results)
            if replaced_count != len(ids_to_replace):
                missing = len(ids_to_replace) - replaced_count
                raise ValueError(
                    "AstraDBBaseStore.mset could not insert all requested "
                    f"documents ({missing} failed replace_one calls)"
                )

    async def amset(self, key_value_pairs: Sequence[Tuple[str, V]]) -> None:
        await self.astra_env.aensure_db_setup()
        documents_to_insert = [
            {"_id": k, "value": self.encode_value(v)}
            for k, v in key_value_pairs
        ]
        # perform an AstraPy insert_many, catching exceptions for overwriting docs
        ids_to_replace: List[int]
        try:
            await self.async_collection.insert_many(
                documents_to_insert,
                ordered=False,
            )
            ids_to_replace = []
        except InsertManyException as err:
            inserted_ids_set = set(err.partial_result.inserted_ids)
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

            sem = asyncio.Semaphore(REPLACE_DOCUMENTS_MAX_THREADS)

            _async_collection = self.async_collection
            async def _replace_document(document: Dict[str, Any]) -> None:
                async with sem:
                    await _async_collection.replace_one(
                        {"_id": document["_id"]},
                        document,
                    )

            tasks = [
                asyncio.create_task(
                    _replace_document(document)
                )
                for document in documents_to_replace
            ]

            replace_results = await asyncio.gather(*tasks, return_exceptions=False)

            replaced_count = sum(r_res.update_info["n"] for r_res in replace_results)
            if replaced_count != len(ids_to_replace):
                missing = len(ids_to_replace) - replaced_count
                raise ValueError(
                    "AstraDBBaseStore.mset could not insert all requested "
                    f"documents ({missing} failed replace_one calls)"
                )

    def mdelete(self, keys: Sequence[str]) -> None:
        self.astra_env.ensure_db_setup()
        self.collection.delete_many(filter={"_id": {"$in": list(keys)}})

    async def amdelete(self, keys: Sequence[str]) -> None:
        await self.astra_env.aensure_db_setup()
        await self.async_collection.delete_many(filter={"_id": {"$in": list(keys)}})

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        self.astra_env.ensure_db_setup()
        docs = self.collection.find()
        for doc in docs:
            key = doc["_id"]
            if not prefix or key.startswith(prefix):
                yield key

    async def ayield_keys(self, *, prefix: Optional[str] = None) -> AsyncIterator[str]:
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
        token: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        environment: Optional[str] = None,
        astra_db_client: Optional[AstraDB] = None,
        namespace: Optional[str] = None,
        async_astra_db_client: Optional[AsyncAstraDB] = None,
        pre_delete_collection: bool = False,
        setup_mode: SetupMode = SetupMode.SYNC,
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
            token: API token for Astra DB usage. If not provided, the environment
                variable ASTRA_DB_APPLICATION_TOKEN is inspected.
            api_endpoint: full URL to the API endpoint, such as
                `https://<DB-ID>-us-east1.apps.astra.datastax.com`. If not provided,
                the environment variable ASTRA_DB_API_ENDPOINT is inspected.
            environment: a string specifying the environment of the target Data API.
                If omitted, defaults to "prod" (Astra DB production).
                Other values are in `astrapy.constants.Environment` enum class.
            astra_db_client:
                *DEPRECATED starting from version 0.3.5.*
                *Please use 'token', 'api_endpoint' and optionally 'environment'.*
                you can pass an already-created 'astrapy.db.AstraDB' instance
                (alternatively to 'token', 'api_endpoint' and 'environment').
            async_astra_db_client:
                *DEPRECATED starting from version 0.3.5.*
                *Please use 'token', 'api_endpoint' and optionally 'environment'.*
                you can pass an already-created 'astrapy.db.AsyncAstraDB' instance
                (alternatively to 'token', 'api_endpoint' and 'environment').
            namespace: namespace (aka keyspace) where the collection is created.
                If not provided, the environment variable ASTRA_DB_KEYSPACE is
                inspected. Defaults to the database's "default namespace".
            setup_mode: mode used to create the Astra DB collection (SYNC, ASYNC or
                OFF).
            pre_delete_collection: whether to delete the collection
                before creating it. If False and the collection already exists,
                the collection will be used as is.
        """
        super().__init__(
            collection_name=collection_name,
            token=token,
            api_endpoint=api_endpoint,
            environment=environment,
            astra_db_client=astra_db_client,
            async_astra_db_client=async_astra_db_client,
            namespace=namespace,
            setup_mode=setup_mode,
            pre_delete_collection=pre_delete_collection,
        )

    def decode_value(self, value: Any) -> Any:
        return value

    def encode_value(self, value: Any) -> Any:
        return value


class AstraDBByteStore(AstraDBBaseStore[bytes], ByteStore):
    def __init__(
        self,
        *,
        collection_name: str,
        token: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        environment: Optional[str] = None,
        astra_db_client: Optional[AstraDB] = None,
        namespace: Optional[str] = None,
        async_astra_db_client: Optional[AsyncAstraDB] = None,
        pre_delete_collection: bool = False,
        setup_mode: SetupMode = SetupMode.SYNC,
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
            token: API token for Astra DB usage. If not provided, the environment
                variable ASTRA_DB_APPLICATION_TOKEN is inspected.
            api_endpoint: full URL to the API endpoint, such as
                `https://<DB-ID>-us-east1.apps.astra.datastax.com`. If not provided,
                the environment variable ASTRA_DB_API_ENDPOINT is inspected.
            environment: a string specifying the environment of the target Data API.
                If omitted, defaults to "prod" (Astra DB production).
                Other values are in `astrapy.constants.Environment` enum class.
            astra_db_client:
                *DEPRECATED starting from version 0.3.5.*
                *Please use 'token', 'api_endpoint' and optionally 'environment'.*
                you can pass an already-created 'astrapy.db.AstraDB' instance
                (alternatively to 'token', 'api_endpoint' and 'environment').
            async_astra_db_client:
                *DEPRECATED starting from version 0.3.5.*
                *Please use 'token', 'api_endpoint' and optionally 'environment'.*
                you can pass an already-created 'astrapy.db.AsyncAstraDB' instance
                (alternatively to 'token', 'api_endpoint' and 'environment').
            namespace: namespace (aka keyspace) where the collection is created.
                If not provided, the environment variable ASTRA_DB_KEYSPACE is
                inspected. Defaults to the database's "default namespace".
            setup_mode: mode used to create the Astra DB collection (SYNC, ASYNC or
                OFF).
            pre_delete_collection: whether to delete the collection
                before creating it. If False and the collection already exists,
                the collection will be used as is.
        """
        super().__init__(
            collection_name=collection_name,
            token=token,
            api_endpoint=api_endpoint,
            environment=environment,
            astra_db_client=astra_db_client,
            async_astra_db_client=async_astra_db_client,
            namespace=namespace,
            setup_mode=setup_mode,
            pre_delete_collection=pre_delete_collection,
        )

    def decode_value(self, value: Any) -> Optional[bytes]:
        if value is None:
            return None
        return base64.b64decode(value)

    def encode_value(self, value: Optional[bytes]) -> Any:
        if value is None:
            return None
        return base64.b64encode(value).decode("ascii")
