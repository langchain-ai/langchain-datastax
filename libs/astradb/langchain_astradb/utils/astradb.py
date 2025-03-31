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
from astrapy.api_options import APIOptions, SerdesOptions, TimeoutOptions
from astrapy.authentication import (
    EmbeddingAPIKeyHeaderProvider,
    EmbeddingHeadersProvider,
    RerankingAPIKeyHeaderProvider,
    RerankingHeadersProvider,
    StaticTokenProvider,
    TokenProvider,
)
from astrapy.constants import Environment
from astrapy.exceptions import DataAPIException, DataAPIResponseException
from astrapy.info import (
    CollectionDefinition,
    CollectionLexicalOptions,
    CollectionRerankOptions,
    RerankServiceOptions,
)

if TYPE_CHECKING:
    from astrapy.info import CollectionDescriptor, VectorServiceOptions

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

# Hardcoded here for the time being
ASTRA_DB_REQUEST_TIMEOUT_MS = 30000

# Amount of (max) number of documents for surveying a collection
SURVEY_NUMBER_OF_DOCUMENTS = 15

# Data API error code for 'collection exists and it's different'
EXISTING_COLLECTION_ERROR_CODE = "EXISTING_COLLECTION_DIFFERENT_SETTINGS"

COLLECTION_DEFAULTS_MISMATCH_ERROR_MESSAGE = (
    "Astra DB collection '{collection_name}' was "
    "found to be configured differently than requested "
    "by the vector store creation. This is resulting "
    "in a hard exception from the Data API (accessible as "
    "`<this-exception>.__cause__`). Please see "
    "https://github.com/langchain-ai/langchain-datastax"
    "/blob/main/libs/astradb/README.md#collection-"
    "defaults-mismatch for more context about this "
    "issue and possible mitigations."
)

logger = logging.getLogger()


class SetupMode(Enum):
    """Setup mode for the Astra DB collection."""

    SYNC = 1
    ASYNC = 2
    OFF = 3


class HybridSearchMode(Enum):
    """Hybrid Search mode for a Vector Store collection."""

    DEFAULT = 1
    ON = 2
    OFF = 3


def _unpack_indexing_policy(
    indexing_dict: dict[str, list[str]] | None,
) -> tuple[str | None, list[str] | None]:
    """{} or None => (None, None); {"a": "b"} => ("a", "b"); multikey => error."""
    if indexing_dict:
        if len(indexing_dict) != 1:
            msg = "Unexpected indexing policy provided: " f"{indexing_dict}"
            raise ValueError(msg)
        return next(iter(indexing_dict.items()))
    return None, None


def _api_exception_error_codes(exc: DataAPIException) -> set[str | None]:
    if isinstance(exc, DataAPIResponseException):
        return {ed.error_code for ed in exc.error_descriptors}
    return set()


def _survey_collection(
    collection_name: str,
    *,
    token: str | TokenProvider | None = None,
    api_endpoint: str | None = None,
    keyspace: str | None = None,
    environment: str | None = None,
    ext_callers: list[tuple[str | None, str | None] | str | None] | None = None,
    component_name: str | None = None,
) -> tuple[CollectionDescriptor | None, list[dict[str, Any]]]:
    """Return the collection descriptor (if found) and a sample of documents."""
    _astra_db_env = _AstraDBEnvironment(
        token=token,
        api_endpoint=api_endpoint,
        keyspace=keyspace,
        environment=environment,
        ext_callers=ext_callers,
        component_name=component_name,
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


class AstraDBError(Exception):
    """An exception during Astra DB- (Data API-) related operations.

    This exception represents any operational exception occurring while
    working with the generic set-up and/or provisioning of components backed
    by Astra DB (in particular, collection creation and inspection).
    """


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
    ) -> None:
        self.token: TokenProvider
        self.api_endpoint: str | None
        self.keyspace: str | None
        self.environment: str | None

        self.data_api_client: DataAPIClient
        self.database: Database
        self.async_database: AsyncDatabase

        if token is None:
            logger.info(
                "Attempting to fetch token from environment " "variable '%s'",
                TOKEN_ENV_VAR,
            )
            self.token = StaticTokenProvider(os.getenv(TOKEN_ENV_VAR))
        elif isinstance(token, TokenProvider):
            self.token = token
        else:
            self.token = StaticTokenProvider(token)

        if api_endpoint is None:
            logger.info(
                "Attempting to fetch API endpoint from environment " "variable '%s'",
                API_ENDPOINT_ENV_VAR,
            )
            self.api_endpoint = os.getenv(API_ENDPOINT_ENV_VAR)
        else:
            self.api_endpoint = api_endpoint

        if keyspace is None:
            logger.info(
                "Attempting to fetch keyspace from environment " "variable '%s'",
                KEYSPACE_ENV_VAR,
            )
            self.keyspace = os.getenv(KEYSPACE_ENV_VAR)
        else:
            self.keyspace = keyspace

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
        self.ext_callers = ext_callers
        self.component_name = component_name
        norm_ext_callers = [
            cpair
            for cpair in (
                _raw_caller if isinstance(_raw_caller, tuple) else (_raw_caller, None)
                for _raw_caller in (self.ext_callers or [])
            )
            if cpair[0] is not None or cpair[1] is not None
        ]
        self.full_callers = [
            *norm_ext_callers,
            LC_CORE_CALLER,
            (self.component_name, LC_ASTRADB_VERSION),
        ]
        # create the client (set to return plain lists for vectors)
        self.data_api_client = DataAPIClient(
            environment=self.environment,
            api_options=APIOptions(
                callers=self.full_callers,
                serdes_options=SerdesOptions(custom_datatypes_in_reading=False),
                timeout_options=TimeoutOptions(
                    request_timeout_ms=ASTRA_DB_REQUEST_TIMEOUT_MS
                ),
            ),
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
        collection_vector_service_options: VectorServiceOptions | None = None,
        collection_embedding_api_key: str | EmbeddingHeadersProvider | None = None,
        collection_rerank: str
        | CollectionRerankOptions
        | RerankServiceOptions
        | None = None,
        collection_reranking_api_key: str | RerankingHeadersProvider | None = None,
        collection_lexical: str
        | dict[str, Any]
        | CollectionLexicalOptions
        | None = None,
    ) -> None:
        super().__init__(
            token=token,
            api_endpoint=api_endpoint,
            keyspace=keyspace,
            environment=environment,
            ext_callers=ext_callers,
            component_name=component_name,
        )
        self.collection_name = collection_name
        self.collection_embedding_api_key = (
            collection_embedding_api_key
            if isinstance(collection_embedding_api_key, EmbeddingHeadersProvider)
            else EmbeddingAPIKeyHeaderProvider(collection_embedding_api_key)
        )
        self.collection_reranking_api_key = (
            collection_reranking_api_key
            if isinstance(collection_reranking_api_key, RerankingHeadersProvider)
            else RerankingAPIKeyHeaderProvider(collection_reranking_api_key)
        )
        self.collection = self.database.get_collection(
            name=self.collection_name,
            embedding_api_key=self.collection_embedding_api_key,
            reranking_api_key=self.collection_reranking_api_key,
        )
        self.async_collection = self.collection.to_async()

        self.collection_rerank = collection_rerank
        self.collection_lexical = collection_lexical

        self.embedding_dimension = embedding_dimension
        self.metric = metric
        self.requested_indexing_policy = requested_indexing_policy
        self.default_indexing_policy = default_indexing_policy
        self.collection_vector_service_options = collection_vector_service_options

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
                _idx_mode, _idx_target = _unpack_indexing_policy(
                    requested_indexing_policy,
                )

                collection_definition = (
                    CollectionDefinition.builder()
                    .set_vector_dimension(embedding_dimension)  # type: ignore[arg-type]
                    .set_vector_metric(metric)
                    .set_indexing(
                        indexing_mode=_idx_mode,
                        indexing_target=_idx_target,
                    )
                    .set_vector_service(collection_vector_service_options)
                    .set_lexical(self.collection_lexical)
                    .set_rerank(self.collection_rerank)
                    .build()
                )
                self.database.create_collection(
                    name=collection_name,
                    definition=collection_definition,
                )
            except DataAPIException as data_api_exception:
                # possibly the collection is preexisting and may have legacy,
                # or custom, indexing settings: verify if it's that error,
                # and if so check for index mismatches - to raise the right error.
                data_api_error_codes = _api_exception_error_codes(data_api_exception)
                if EXISTING_COLLECTION_ERROR_CODE in data_api_error_codes:
                    collection_descriptors = list(self.database.list_collections())
                    try:
                        if not self._validate_indexing_policy(
                            collection_descriptors=collection_descriptors,
                            collection_name=self.collection_name,
                            requested_indexing_policy=requested_indexing_policy,
                            default_indexing_policy=default_indexing_policy,
                        ):
                            # mismatch is not due to indexing
                            msg = COLLECTION_DEFAULTS_MISMATCH_ERROR_MESSAGE.format(
                                collection_name=self.collection_name,
                            )
                            raise AstraDBError(msg) from data_api_exception
                    except ValueError as validation_error:
                        raise validation_error from data_api_exception
                else:
                    raise

    def copy(
        self,
        *,
        token: str | TokenProvider | None = None,
        ext_callers: list[tuple[str | None, str | None] | str | None] | None = None,
        component_name: str | None = None,
        collection_embedding_api_key: str | EmbeddingHeadersProvider | None = None,
        collection_reranking_api_key: str | RerankingHeadersProvider | None = None,
    ) -> _AstraDBCollectionEnvironment:
        """Create a copy, possibly with changed attributes.

        This method creates a shallow copy of this environment. If a parameter
        is passed and differs from None, it will replace the corresponding value
        in the copy.

        The method allows changing only the parameters that ensure the copy is
        functional and does not trigger side-effects:
        for example, one cannot create a copy acting on a new collection.
        In those cases, one should create a new instance
        of ``_AstraDBCollectionEnvironment`` from scratch.

        Attributes:
            token: API token for Astra DB usage, either in the form of a string
                or a subclass of ``astrapy.authentication.TokenProvider``.
                In order to suppress token usage in the copy, explicitly pass
                ``astrapy.authentication.StaticTokenProvider(None)``.
            ext_callers: additional custom (caller_name, caller_version) pairs
                to attach to the User-Agent header when issuing Data API requests.
            component_name: a value for the LangChain component name to use when
                identifying the originator of the Data API requests.
            collection_embedding_api_key: the API Key to supply in each Data API
                request if necessary. This is necessary if using the Vectorize
                feature and a required secret is not stored with the database.
                In order to suppress the API Key in the copy, explicitly pass
                ``astrapy.authentication.EmbeddingAPIKeyHeaderProvider(None)``.
            collection_reranking_api_key: the API Key to supply in each Data API
                request if necessary. This is necessary if using the Rerank
                feature and a required secret is not stored with the database.
                In order to suppress the API Key in the copy, explicitly pass
                ``astrapy.authentication.RerankingAPIKeyHeaderProvider(None)``.
        """
        return _AstraDBCollectionEnvironment(
            collection_name=self.collection_name,
            token=self.token if token is None else token,
            api_endpoint=self.api_endpoint,
            keyspace=self.keyspace,
            environment=self.environment,
            ext_callers=self.ext_callers if ext_callers is None else ext_callers,
            component_name=self.component_name
            if component_name is None
            else component_name,
            setup_mode=SetupMode.OFF,
            collection_embedding_api_key=self.collection_embedding_api_key
            if collection_embedding_api_key is None
            else collection_embedding_api_key,
            collection_rerank=self.collection_rerank,
            collection_reranking_api_key=self.collection_reranking_api_key
            if collection_reranking_api_key is None
            else collection_reranking_api_key,
            collection_lexical=self.collection_lexical,
            embedding_dimension=self.embedding_dimension,
            metric=self.metric,
            requested_indexing_policy=self.requested_indexing_policy,
            default_indexing_policy=self.default_indexing_policy,
            collection_vector_service_options=self.collection_vector_service_options,
        )

    async def _asetup_db(
        self,
        *,
        pre_delete_collection: bool,
        embedding_dimension: int | Awaitable[int] | None,
        metric: str | None,
        requested_indexing_policy: dict[str, Any] | None,
        default_indexing_policy: dict[str, Any] | None,
        collection_vector_service_options: VectorServiceOptions | None,
    ) -> None:
        if pre_delete_collection:
            await self.async_database.drop_collection(self.collection_name)
        if inspect.isawaitable(embedding_dimension):
            dimension = await embedding_dimension
        else:
            dimension = embedding_dimension

        try:
            _idx_mode, _idx_target = _unpack_indexing_policy(requested_indexing_policy)
            collection_definition = (
                CollectionDefinition.builder()
                .set_vector_dimension(dimension)
                .set_vector_metric(metric)
                .set_indexing(
                    indexing_mode=_idx_mode,
                    indexing_target=_idx_target,
                )
                .set_vector_service(collection_vector_service_options)
                .set_lexical(self.collection_lexical)
                .set_rerank(self.collection_rerank)
                .build()
            )
            await self.async_database.create_collection(
                name=self.collection_name,
                definition=collection_definition,
            )
        except DataAPIException as data_api_exception:
            # possibly the collection is preexisting and may have legacy,
            # or custom, indexing settings: verify if it's that error,
            # and if so check for index mismatches - to raise the right error.
            data_api_error_codes = _api_exception_error_codes(data_api_exception)
            if EXISTING_COLLECTION_ERROR_CODE in data_api_error_codes:
                collection_descriptors = list(
                    await asyncio.to_thread(self.database.list_collections)
                )
                try:
                    if not self._validate_indexing_policy(
                        collection_descriptors=collection_descriptors,
                        collection_name=self.collection_name,
                        requested_indexing_policy=requested_indexing_policy,
                        default_indexing_policy=default_indexing_policy,
                    ):
                        # mismatch is not due to indexing
                        msg = COLLECTION_DEFAULTS_MISMATCH_ERROR_MESSAGE.format(
                            collection_name=self.collection_name,
                        )
                        raise AstraDBError(msg) from data_api_exception
                except ValueError as validation_error:
                    raise validation_error from data_api_exception
            else:
                raise

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
        pre_col_definition = pre_collection.definition
        if not pre_col_definition.indexing:
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

        if pre_col_definition.indexing != requested_indexing_policy:
            # collection on DB has indexing settings, but different
            options_json = json.dumps(pre_col_definition.indexing)
            default_desc = (
                " (default setting)"
                if pre_col_definition.indexing == default_indexing_policy
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
