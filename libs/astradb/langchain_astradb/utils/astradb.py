"""Utilities for AstraDB setup and management."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import warnings
from asyncio import InvalidStateError, Task
from enum import Enum
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, Awaitable

import langchain_core
from astrapy import AsyncDatabase, DataAPIClient, Database
from astrapy.admin import parse_api_endpoint
from astrapy.constants import Environment
from astrapy.exceptions import DataAPIException

if TYPE_CHECKING:
    from astrapy.authentication import EmbeddingHeadersProvider, TokenProvider
    from astrapy.db import AstraDB, AsyncAstraDB
    from astrapy.info import CollectionDescriptor, CollectionVectorServiceOptions

TOKEN_ENV_VAR = "ASTRA_DB_APPLICATION_TOKEN"  # noqa: S105
API_ENDPOINT_ENV_VAR = "ASTRA_DB_API_ENDPOINT"
KEYSPACE_ENV_VAR = "ASTRA_DB_KEYSPACE"

# Caller-related constants
LC_CORE_CALLER_NAME = "langchain"
LC_CORE_CALLER_VERSION = getattr(langchain_core, "__version__", None)
LC_CORE_CALLER = (LC_CORE_CALLER_NAME, LC_CORE_CALLER_VERSION)

LC_ASTRADB_VERSION: str | None
try:
    LC_ASTRADB_VERSION = version("langchain_astradb")
except TypeError:
    LC_ASTRADB_VERSION = None

# component names for the 'callers' parameter
COMPONENT_NAME_CACHE = "langchain_cache"
COMPONENT_NAME_SEMANTICCACHE = "langchain_semanticcache"
COMPONENT_NAME_CHATMESSAGEHISTORY = "langchain_chatmessagehistory"
COMPONENT_NAME_LOADER = "langchain_loader"
COMPONENT_NAME_GRAPHVECTORSTORE = "langchain_graphvectorstore"
COMPONENT_NAME_STORE = "langchain_store"
COMPONENT_NAME_BYTESTORE = "langchain_bytestore"
COMPONENT_NAME_VECTORSTORE = "langchain_vectorstore"


# Default settings for API data operations (concurrency & similar):
# Chunk size for many-document insertions (None meaning defer to astrapy):
DEFAULT_DOCUMENT_CHUNK_SIZE = None
# thread/coroutine count for bulk inserts
MAX_CONCURRENT_DOCUMENT_INSERTIONS = 20
# Thread/coroutine count for one-doc-at-a-time overwrites
MAX_CONCURRENT_DOCUMENT_REPLACEMENTS = 20
# Thread/coroutine count for one-doc-at-a-time deletes:
MAX_CONCURRENT_DOCUMENT_DELETIONS = 20

# Amount of (max) number of documents for surveying a collection
SURVEY_NUMBER_OF_DOCUMENTS = 15

logger = logging.getLogger()


class SetupMode(Enum):
    """Setup mode for the Astra DB collection."""

    SYNC = 1
    ASYNC = 2
    OFF = 3


def _survey_collection(
    collection_name: str,
    *,
    token: str | TokenProvider | None = None,
    api_endpoint: str | None = None,
    keyspace: str | None = None,
    environment: str | None = None,
    ext_callers: list[tuple[str | None, str | None] | str | None] | None = None,
    component_name: str | None = None,
    astra_db_client: AstraDB | None = None,
    async_astra_db_client: AsyncAstraDB | None = None,
) -> tuple[CollectionDescriptor | None, list[dict[str, Any]]]:
    """Return the collection descriptor (if found) and a sample of documents."""
    _astra_db_env = _AstraDBEnvironment(
        token=token,
        api_endpoint=api_endpoint,
        keyspace=keyspace,
        environment=environment,
        ext_callers=ext_callers,
        component_name=component_name,
        astra_db_client=astra_db_client,
        async_astra_db_client=async_astra_db_client,
    )
    descriptors = [
        coll_d
        for coll_d in _astra_db_env.database.list_collections()
        if coll_d.name == collection_name
    ]
    if not descriptors:
        return None, []
    descriptor = descriptors[0]
    # fetch some documents
    document_ite = _astra_db_env.database.get_collection(collection_name).find(
        filter={},
        projection={"*": True},
        limit=SURVEY_NUMBER_OF_DOCUMENTS,
    )
    return (descriptor, list(document_ite))


def _normalize_data_api_environment(
    arg_environment: str | None,
    api_endpoint: str,
) -> str:
    _environment: str
    if arg_environment is not None:
        return arg_environment
    parsed_endpoint = parse_api_endpoint(api_endpoint)
    if parsed_endpoint is None:
        logger.info(
            "Detecting API environment '%s' from supplied endpoint",
            Environment.OTHER,
        )
        return Environment.OTHER

    logger.info(
        "Detecting API environment '%s' from supplied endpoint",
        parsed_endpoint.environment,
    )
    return parsed_endpoint.environment


class _AstraDBEnvironment:
    def __init__(
        self,
        *,
        token: str | TokenProvider | None = None,
        api_endpoint: str | None = None,
        keyspace: str | None = None,
        environment: str | None = None,
        ext_callers: list[tuple[str | None, str | None] | str | None] | None = None,
        component_name: str | None = None,
        astra_db_client: AstraDB | None = None,
        async_astra_db_client: AsyncAstraDB | None = None,
    ) -> None:
        self.token: str | TokenProvider | None
        self.api_endpoint: str | None
        self.keyspace: str | None
        self.environment: str | None

        self.data_api_client: DataAPIClient
        self.database: Database
        self.async_database: AsyncDatabase

        if astra_db_client is not None or async_astra_db_client is not None:
            if token is not None or api_endpoint is not None or environment is not None:
                msg = (
                    "You cannot pass 'astra_db_client' or 'async_astra_db_client' "
                    "to AstraDBEnvironment if passing 'token', 'api_endpoint' or "
                    "'environment'."
                )
                raise ValueError(msg)
            _astra_db = astra_db_client.copy() if astra_db_client is not None else None
            _async_astra_db = (
                async_astra_db_client.copy()
                if async_astra_db_client is not None
                else None
            )

            # deprecation of the 'core classes' in constructor and conversion
            # to token/endpoint(-environment) based init, with checks
            # at least one of the two (core) clients is not None:
            warnings.warn(
                (
                    "Initializing Astra DB LangChain classes by passing "
                    "AstraDB/AsyncAstraDB ready clients is deprecated starting "
                    "with langchain-astradb==0.3.5. Please switch to passing "
                    "'token', 'api_endpoint' (and optionally 'environment') "
                    "instead."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
            _tokens = list(
                {
                    klient.token
                    for klient in [astra_db_client, async_astra_db_client]
                    if klient is not None
                }
            )
            _api_endpoints = list(
                {
                    klient.api_endpoint
                    for klient in [astra_db_client, async_astra_db_client]
                    if klient is not None
                }
            )
            _keyspaces = list(
                {
                    klient.namespace
                    for klient in [astra_db_client, async_astra_db_client]
                    if klient is not None
                }
            )
            if len(_tokens) != 1:
                msg = (
                    "Conflicting tokens found in the sync and async AstraDB "
                    "constructor parameters. Please check the tokens and "
                    "ensure they match."
                )
                raise ValueError(msg)
            if len(_api_endpoints) != 1:
                msg = (
                    "Conflicting API endpoints found in the sync and async "
                    "AstraDB constructor parameters. Please check the tokens "
                    "and ensure they match."
                )
                raise ValueError(msg)
            if len(_keyspaces) != 1:
                msg = (
                    "Conflicting keyspaces found in the sync and async "
                    "AstraDB constructor parameters' 'namespace' attributes. "
                    "Please check the keyspaces and ensure they match."
                )
                raise ValueError(msg)
            # all good: these are 1-element lists here
            self.token = _tokens[0]
            self.api_endpoint = _api_endpoints[0]
            self.keyspace = _keyspaces[0]
        else:
            _token: str | TokenProvider | None
            # secrets-based initialization
            if token is None:
                logger.info(
                    "Attempting to fetch token from environment " "variable '%s'",
                    TOKEN_ENV_VAR,
                )
                _token = os.environ.get(TOKEN_ENV_VAR)
            else:
                _token = token
            if api_endpoint is None:
                logger.info(
                    "Attempting to fetch API endpoint from environment "
                    "variable '%s'",
                    API_ENDPOINT_ENV_VAR,
                )
                _api_endpoint = os.environ.get(API_ENDPOINT_ENV_VAR)
            else:
                _api_endpoint = api_endpoint
            if keyspace is None:
                _keyspace = os.environ.get(KEYSPACE_ENV_VAR)
            else:
                _keyspace = keyspace

            self.token = _token
            self.api_endpoint = _api_endpoint
            self.keyspace = _keyspace

        # init parameters are normalized to self.{token, api_endpoint, keyspace}.
        # Proceed. Keyspace and token can be None (resp. on Astra DB and non-Astra)
        if self.api_endpoint is None:
            msg = (
                "API endpoint for Data API not provided. "
                "Either pass it explicitly to the object constructor "
                f"or set the {API_ENDPOINT_ENV_VAR} environment variable."
            )
            raise ValueError(msg)

        self.environment = _normalize_data_api_environment(
            environment,
            self.api_endpoint,
        )

        # prepare the "callers" list to create the clients.
        # The callers, passed to astrapy, are made of these Caller pairs in this order:
        # - zero, one or more are the "ext_callers" passed to this environment
        # - a single ("langchain", <version of langchain_core>)
        # - if such is provided, a (component_name, <version of langchain_astradb>)
        #   (note: if component_name is None, astrapy strips it out automatically)
        norm_ext_callers = [
            cpair
            for cpair in (
                _raw_caller if isinstance(_raw_caller, tuple) else (_raw_caller, None)
                for _raw_caller in (ext_callers or [])
            )
            if cpair[0] is not None or cpair[1] is not None
        ]
        full_callers = [
            *norm_ext_callers,
            LC_CORE_CALLER,
            (component_name, LC_ASTRADB_VERSION),
        ]

        # create the callers
        self.data_api_client = DataAPIClient(
            environment=self.environment,
            callers=full_callers,
        )
        self.database = self.data_api_client.get_database(
            api_endpoint=self.api_endpoint,
            token=self.token,
            keyspace=self.keyspace,
        )
        self.async_database = self.database.to_async()


class _AstraDBCollectionEnvironment(_AstraDBEnvironment):
    def __init__(
        self,
        collection_name: str,
        *,
        token: str | TokenProvider | None = None,
        api_endpoint: str | None = None,
        keyspace: str | None = None,
        environment: str | None = None,
        ext_callers: list[tuple[str | None, str | None] | str | None] | None = None,
        component_name: str | None = None,
        setup_mode: SetupMode = SetupMode.SYNC,
        pre_delete_collection: bool = False,
        embedding_dimension: int | Awaitable[int] | None = None,
        metric: str | None = None,
        requested_indexing_policy: dict[str, Any] | None = None,
        default_indexing_policy: dict[str, Any] | None = None,
        collection_vector_service_options: CollectionVectorServiceOptions | None = None,
        collection_embedding_api_key: str | EmbeddingHeadersProvider | None = None,
        astra_db_client: AstraDB | None = None,
        async_astra_db_client: AsyncAstraDB | None = None,
    ) -> None:
        super().__init__(
            token=token,
            api_endpoint=api_endpoint,
            keyspace=keyspace,
            environment=environment,
            ext_callers=ext_callers,
            component_name=component_name,
            astra_db_client=astra_db_client,
            async_astra_db_client=async_astra_db_client,
        )
        self.collection_name = collection_name
        self.collection = self.database.get_collection(
            name=self.collection_name,
            embedding_api_key=collection_embedding_api_key,
        )
        self.async_collection = self.collection.to_async()

        self.async_setup_db_task: Task | None = None
        if setup_mode == SetupMode.ASYNC:
            self.async_setup_db_task = asyncio.create_task(
                self._asetup_db(
                    pre_delete_collection=pre_delete_collection,
                    embedding_dimension=embedding_dimension,
                    metric=metric,
                    default_indexing_policy=default_indexing_policy,
                    requested_indexing_policy=requested_indexing_policy,
                    collection_vector_service_options=collection_vector_service_options,
                )
            )
        elif setup_mode == SetupMode.SYNC:
            if pre_delete_collection:
                self.database.drop_collection(collection_name)
            if inspect.isawaitable(embedding_dimension):
                msg = (
                    "Cannot use an awaitable embedding_dimension with async_setup "
                    "set to False"
                )
                raise ValueError(msg)
            try:
                self.database.create_collection(
                    name=collection_name,
                    dimension=embedding_dimension,  # type: ignore[arg-type]
                    metric=metric,
                    indexing=requested_indexing_policy,
                    # Used for enabling $vectorize on the collection
                    service=collection_vector_service_options,
                    check_exists=False,
                )
            except DataAPIException as data_api_exception:
                # possibly the collection is preexisting and may have legacy,
                # or custom, indexing settings: verify
                collection_descriptors = list(self.database.list_collections())
                try:
                    if not self._validate_indexing_policy(
                        collection_descriptors=collection_descriptors,
                        collection_name=self.collection_name,
                        requested_indexing_policy=requested_indexing_policy,
                        default_indexing_policy=default_indexing_policy,
                    ):
                        raise data_api_exception  # noqa: TRY201
                except ValueError as validation_error:
                    raise validation_error from data_api_exception

    async def _asetup_db(
        self,
        *,
        pre_delete_collection: bool,
        embedding_dimension: int | Awaitable[int] | None,
        metric: str | None,
        requested_indexing_policy: dict[str, Any] | None,
        default_indexing_policy: dict[str, Any] | None,
        collection_vector_service_options: CollectionVectorServiceOptions | None,
    ) -> None:
        if pre_delete_collection:
            await self.async_database.drop_collection(self.collection_name)
        if inspect.isawaitable(embedding_dimension):
            dimension = await embedding_dimension
        else:
            dimension = embedding_dimension

        try:
            await self.async_database.create_collection(
                name=self.collection_name,
                dimension=dimension,
                metric=metric,
                indexing=requested_indexing_policy,
                # Used for enabling $vectorize on the collection
                service=collection_vector_service_options,
                check_exists=False,
            )
        except DataAPIException as data_api_exception:
            # possibly the collection is preexisting and may have legacy,
            # or custom, indexing settings: verify
            collection_descriptors = [
                coll_desc async for coll_desc in self.async_database.list_collections()
            ]
            try:
                if not self._validate_indexing_policy(
                    collection_descriptors=collection_descriptors,
                    collection_name=self.collection_name,
                    requested_indexing_policy=requested_indexing_policy,
                    default_indexing_policy=default_indexing_policy,
                ):
                    # other reasons for the exception
                    raise data_api_exception  # noqa: TRY201
            except ValueError as validation_error:
                raise validation_error from data_api_exception

    @staticmethod
    def _validate_indexing_policy(
        collection_descriptors: list[CollectionDescriptor],
        collection_name: str,
        requested_indexing_policy: dict[str, Any] | None,
        default_indexing_policy: dict[str, Any] | None,
    ) -> bool:
        """Validate indexing policy.

        This is a validation helper, to be called when the collection-creation
        call has failed.

        Args:
            collection_descriptors: collection descriptors for the database.
            collection_name: the name of the collection whose attempted
                creation failed
            requested_indexing_policy: the 'indexing' part of the collection
                options, e.g. `{"deny": ["field1", "field2"]}`.
                Leave to its default of None if no options required.
            default_indexing_policy: an optional 'default value' for the
                above, used to issue just a gentle warning in the special
                case that no policy is detected on a preexisting collection
                on DB and the default is requested. This is to enable
                a warning-only transition to new code using indexing without
                disrupting usage of a legacy collection, i.e. one created
                before adopting the usage of indexing policies altogether.
                You cannot pass this one without requested_indexing_policy.

        This function may raise an error (indexing mismatches), issue a warning
        (about legacy collections), or do nothing.
        In any case, when the function returns, it returns either
            - True: the exception was handled here as part of the indexing
              management
            - False: the exception is unrelated to indexing and the caller
              has to reraise it.
        """
        if requested_indexing_policy is None and default_indexing_policy is not None:
            msg = (
                "Cannot specify a default indexing policy "
                "when no indexing policy is requested for this collection "
                "(requested_indexing_policy is None, "
                "default_indexing_policy is not None)."
            )
            raise ValueError(msg)

        preexisting = [
            collection
            for collection in collection_descriptors
            if collection.name == collection_name
        ]

        if not preexisting:
            # foreign-origin for the original exception
            return False

        pre_collection = preexisting[0]
        # if it has no "indexing", it is a legacy collection
        pre_col_options = pre_collection.options
        if not pre_col_options.indexing:
            # legacy collection on DB
            if requested_indexing_policy != default_indexing_policy:
                msg = (
                    f"Astra DB collection '{collection_name}' is "
                    "detected as having indexing turned on for all "
                    "fields (either created manually or by older "
                    "versions of this plugin). This is incompatible with "
                    "the requested indexing policy for this object. "
                    "Consider indexing anew on a fresh "
                    "collection with the requested indexing "
                    "policy, or alternatively leave the indexing "
                    "settings for this object to their defaults "
                    "to keep using this collection."
                )
                raise ValueError(msg)
            warnings.warn(
                (
                    f"Astra DB collection '{collection_name}' is "
                    "detected as having indexing turned on for all "
                    "fields (either created manually or by older "
                    "versions of this plugin). This implies stricter "
                    "limitations on the amount of text each string in a "
                    "document can store. Consider indexing anew on a "
                    "fresh collection to be able to store longer texts. "
                    "See https://github.com/langchain-ai/langchain-"
                    "datastax/blob/main/libs/astradb/README.md#"
                    "warnings-about-indexing for more details."
                ),
                UserWarning,
                stacklevel=2,
            )
            # the original exception, related to indexing, was handled here
            return True

        if pre_col_options.indexing != requested_indexing_policy:
            # collection on DB has indexing settings, but different
            options_json = json.dumps(pre_col_options.indexing)
            default_desc = (
                " (default setting)"
                if pre_col_options.indexing == default_indexing_policy
                else ""
            )
            msg = (
                f"Astra DB collection '{collection_name}' is "
                "detected as having the following indexing policy: "
                f"{options_json}{default_desc}. This is incompatible "
                "with the requested indexing policy for this object. "
                "Consider indexing anew on a fresh "
                "collection with the requested indexing "
                "policy, or alternatively align the requested "
                "indexing settings to the collection to keep using it."
            )
            raise ValueError(msg)

        # the discrepancies have to do with options other than indexing
        return False

    def ensure_db_setup(self) -> None:
        if self.async_setup_db_task:
            try:
                self.async_setup_db_task.result()
            except InvalidStateError as e:
                msg = (
                    "Asynchronous setup of the DB not finished. "
                    "NB: Astra DB components sync methods shouldn't be called from the "
                    "event loop. Consider using their async equivalents."
                )
                raise ValueError(msg) from e

    async def aensure_db_setup(self) -> None:
        if self.async_setup_db_task:
            await self.async_setup_db_task
