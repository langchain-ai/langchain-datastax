"""Astra DB - based caches."""

from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache, wraps
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Generator

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.language_models.llms import aget_prompts, get_prompts
from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
from langchain_core.outputs import Generation
from typing_extensions import override

from langchain_astradb.utils.astradb import (
    COMPONENT_NAME_CACHE,
    COMPONENT_NAME_SEMANTICCACHE,
    SetupMode,
    _AstraDBCollectionEnvironment,
)

if TYPE_CHECKING:
    from astrapy.authentication import TokenProvider
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import LLM

ASTRA_DB_CACHE_DEFAULT_COLLECTION_NAME = "langchain_astradb_cache"
ASTRA_DB_SEMANTIC_CACHE_DEFAULT_COLLECTION_NAME = "langchain_astradb_semantic_cache"
ASTRA_DB_SEMANTIC_CACHE_DEFAULT_THRESHOLD = 0.85
ASTRA_DB_SEMANTIC_CACHE_EMBEDDING_CACHE_SIZE = 16

logger = logging.getLogger(__name__)


def _hash(_input: str) -> str:
    """Use a deterministic hashing approach."""
    return hashlib.md5(_input.encode()).hexdigest()  # noqa: S324


def _dumps_generations(generations: RETURN_VAL_TYPE) -> str:
    """Serialization for generic RETURN_VAL_TYPE, i.e. sequence of `Generation`.

    Args:
        generations (RETURN_VAL_TYPE): A list of language model generations.

    Returns:
        str: a single string representing a list of generations.

    This function (+ its counterpart `_loads_generations`) rely on
    the dumps/loads pair with Reviver, so are able to deal
    with all subclasses of Generation.

    Each item in the list can be `dumps`ed to a string,
    then we make the whole list of strings into a json-dumped.
    """
    return json.dumps([dumps(_item) for _item in generations])


def _loads_generations(generations_str: str) -> RETURN_VAL_TYPE | None:
    """Get Generations from a string.

    Deserialization of a string into a generic RETURN_VAL_TYPE
    (i.e. a sequence of `Generation`).

    See `_dumps_generations`, the inverse of this function.

    Args:
        generations_str (str): A string representing a list of generations.

    Compatible with the legacy cache-blob format
    Does not raise exceptions for malformed entries, just logs a warning
    and returns none: the caller should be prepared for such a cache miss.

    Returns:
        RETURN_VAL_TYPE: A list of generations.
    """
    try:
        return [loads(_item_str) for _item_str in json.loads(generations_str)]
    except (json.JSONDecodeError, TypeError):
        # deferring the (soft) handling to after the legacy-format attempt
        pass

    try:
        gen_dicts = json.loads(generations_str)
        # not relying on `_load_generations_from_json` (which could disappear):
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            "Malformed/unparsable cached blob encountered: '%s'",
            generations_str,
        )
        return None
    else:
        generations = [Generation(**generation_dict) for generation_dict in gen_dicts]
        logger.warning(
            "Legacy 'Generation' cached blob encountered: '%s'",
            generations_str,
        )
        return generations


class AstraDBCache(BaseCache):
    @staticmethod
    def _make_id(prompt: str, llm_string: str) -> str:
        return f"{_hash(prompt)}#{_hash(llm_string)}"

    def __init__(
        self,
        *,
        collection_name: str = ASTRA_DB_CACHE_DEFAULT_COLLECTION_NAME,
        token: str | TokenProvider | None = None,
        api_endpoint: str | None = None,
        namespace: str | None = None,
        environment: str | None = None,
        pre_delete_collection: bool = False,
        setup_mode: SetupMode = SetupMode.SYNC,
        ext_callers: list[tuple[str | None, str | None] | str | None] | None = None,
    ):
        """Cache that uses Astra DB as a backend.

        It uses a single collection as a kv store
        The lookup keys, combined in the _id of the documents, are:
            - prompt, a string
            - llm_string, a deterministic str representation of the model parameters.
              (needed to prevent same-prompt-different-model collisions)

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
        self.astra_env = _AstraDBCollectionEnvironment(
            collection_name=collection_name,
            token=token,
            api_endpoint=api_endpoint,
            keyspace=namespace,
            environment=environment,
            setup_mode=setup_mode,
            pre_delete_collection=pre_delete_collection,
            ext_callers=ext_callers,
            component_name=COMPONENT_NAME_CACHE,
        )
        self.collection = self.astra_env.collection
        self.async_collection = self.astra_env.async_collection

    @override
    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        self.astra_env.ensure_db_setup()
        doc_id = self._make_id(prompt, llm_string)
        item = self.collection.find_one(
            filter={
                "_id": doc_id,
            },
            projection={
                "body_blob": 1,
            },
        )
        return _loads_generations(item["body_blob"]) if item is not None else None

    @override
    async def alookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        await self.astra_env.aensure_db_setup()
        doc_id = self._make_id(prompt, llm_string)
        item = await self.async_collection.find_one(
            filter={
                "_id": doc_id,
            },
            projection={
                "body_blob": 1,
            },
        )
        return _loads_generations(item["body_blob"]) if item is not None else None

    @override
    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        self.astra_env.ensure_db_setup()
        doc_id = self._make_id(prompt, llm_string)
        blob = _dumps_generations(return_val)
        self.collection.find_one_and_replace(
            {"_id": doc_id},
            {
                "_id": doc_id,
                "body_blob": blob,
            },
            upsert=True,
        )

    @override
    async def aupdate(
        self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE
    ) -> None:
        await self.astra_env.aensure_db_setup()
        doc_id = self._make_id(prompt, llm_string)
        blob = _dumps_generations(return_val)
        await self.async_collection.find_one_and_replace(
            {"_id": doc_id},
            {
                "_id": doc_id,
                "body_blob": blob,
            },
            upsert=True,
        )

    def delete_through_llm(
        self, prompt: str, llm: LLM, stop: list[str] | None = None
    ) -> None:
        """A wrapper around `delete` with the LLM being passed.

        In case the llm(prompt) calls have a `stop` param, you should pass it here.
        """
        llm_string = get_prompts(
            {**llm.dict(), "stop": stop},
            [],
        )[1]
        return self.delete(prompt, llm_string=llm_string)

    async def adelete_through_llm(
        self, prompt: str, llm: LLM, stop: list[str] | None = None
    ) -> None:
        """A wrapper around `adelete` with the LLM being passed.

        In case the llm(prompt) calls have a `stop` param, you should pass it here.
        """
        llm_string = (
            await aget_prompts(
                {**llm.dict(), "stop": stop},
                [],
            )
        )[1]
        return await self.adelete(prompt, llm_string=llm_string)

    def delete(self, prompt: str, llm_string: str) -> None:
        """Evict from cache if there's an entry."""
        self.astra_env.ensure_db_setup()
        doc_id = self._make_id(prompt, llm_string)
        self.collection.delete_one({"_id": doc_id})

    async def adelete(self, prompt: str, llm_string: str) -> None:
        """Evict from cache if there's an entry."""
        await self.astra_env.aensure_db_setup()
        doc_id = self._make_id(prompt, llm_string)
        await self.async_collection.delete_one({"_id": doc_id})

    @override
    def clear(self, **kwargs: Any) -> None:
        self.astra_env.ensure_db_setup()
        self.collection.delete_many({})

    @override
    async def aclear(self, **kwargs: Any) -> None:
        await self.astra_env.aensure_db_setup()
        await self.async_collection.delete_many({})


_unset = ["unset"]


class _CachedAwaitable:
    """Cache the result of an awaitable so it can be awaited multiple times."""

    def __init__(self, awaitable: Awaitable[Any]):
        self.awaitable = awaitable
        self.result = _unset

    def __await__(self) -> Generator:
        if self.result is _unset:
            self.result = yield from self.awaitable.__await__()
        return self.result


def _reawaitable(func: Callable) -> Callable:
    """Make an async function result awaitable multiple times."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> _CachedAwaitable:
        return _CachedAwaitable(func(*args, **kwargs))

    return wrapper


def _async_lru_cache(maxsize: int = 128) -> Callable:
    """Least-recently-used async cache decorator.

    Equivalent to functools.lru_cache for async functions.
    """

    def decorating_function(user_function: Callable) -> Callable:
        return lru_cache(maxsize)(_reawaitable(user_function))

    return decorating_function


class AstraDBSemanticCache(BaseCache):
    def __init__(
        self,
        *,
        collection_name: str = ASTRA_DB_SEMANTIC_CACHE_DEFAULT_COLLECTION_NAME,
        token: str | TokenProvider | None = None,
        api_endpoint: str | None = None,
        namespace: str | None = None,
        environment: str | None = None,
        setup_mode: SetupMode = SetupMode.SYNC,
        pre_delete_collection: bool = False,
        embedding: Embeddings,
        metric: str | None = None,
        similarity_threshold: float = ASTRA_DB_SEMANTIC_CACHE_DEFAULT_THRESHOLD,
        ext_callers: list[tuple[str | None, str | None] | str | None] | None = None,
    ):
        """Astra DB semantic cache.

        Cache that uses Astra DB as a vector-store backend for semantic
        (i.e. similarity-based) lookup.

        It uses a single (vector) collection and can store
        cached values from several LLMs, so the LLM's 'llm_string' is stored
        in the document metadata.

        You can choose the preferred similarity (or use the API default).
        The default score threshold is tuned to the default metric.
        Tune it carefully yourself if switching to another distance metric.

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
            embedding: Embedding provider for semantic encoding and search.
            metric: the function to use for evaluating similarity of text embeddings.
                Defaults to 'cosine' (alternatives: 'euclidean', 'dot_product')
            similarity_threshold: the minimum similarity for accepting a
                (semantic-search) match.
            ext_callers: one or more caller identities to identify Data API calls
                in the User-Agent header. This is a list of (name, version) pairs,
                or just strings if no version info is provided, which, if supplied,
                becomes the leading part of the User-Agent string in all API requests
                related to this component.
        """
        self.embedding = embedding
        self.metric = metric
        self.similarity_threshold = similarity_threshold
        self.collection_name = collection_name

        # The contract for this class has separate lookup and update:
        # in order to spare some embedding calculations we cache them between
        # the two calls.
        # Note: each instance of this class has its own `_get_embedding` with
        # its own lru.
        @lru_cache(maxsize=ASTRA_DB_SEMANTIC_CACHE_EMBEDDING_CACHE_SIZE)
        def _cache_embedding(text: str) -> list[float]:
            return self.embedding.embed_query(text=text)

        self._get_embedding = _cache_embedding

        @_async_lru_cache(maxsize=ASTRA_DB_SEMANTIC_CACHE_EMBEDDING_CACHE_SIZE)
        async def _acache_embedding(text: str) -> list[float]:
            return await self.embedding.aembed_query(text=text)

        self._aget_embedding = _acache_embedding

        embedding_dimension: int | Awaitable[int] | None = None
        if setup_mode == SetupMode.ASYNC:
            embedding_dimension = self._aget_embedding_dimension()
        elif setup_mode == SetupMode.SYNC:
            embedding_dimension = self._get_embedding_dimension()

        self.astra_env = _AstraDBCollectionEnvironment(
            collection_name=collection_name,
            token=token,
            api_endpoint=api_endpoint,
            keyspace=namespace,
            environment=environment,
            setup_mode=setup_mode,
            pre_delete_collection=pre_delete_collection,
            embedding_dimension=embedding_dimension,
            metric=metric,
            ext_callers=ext_callers,
            component_name=COMPONENT_NAME_SEMANTICCACHE,
        )
        self.collection = self.astra_env.collection
        self.async_collection = self.astra_env.async_collection

    def _get_embedding_dimension(self) -> int:
        return len(self._get_embedding(text="This is a sample sentence."))

    async def _aget_embedding_dimension(self) -> int:
        return len(await self._aget_embedding(text="This is a sample sentence."))

    @staticmethod
    def _make_id(prompt: str, llm_string: str) -> str:
        return f"{_hash(prompt)}#{_hash(llm_string)}"

    @override
    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        self.astra_env.ensure_db_setup()
        doc_id = self._make_id(prompt, llm_string)
        llm_string_hash = _hash(llm_string)
        embedding_vector = self._get_embedding(text=prompt)
        body = _dumps_generations(return_val)
        self.collection.find_one_and_replace(
            {"_id": doc_id},
            {
                "_id": doc_id,
                "body_blob": body,
                "llm_string_hash": llm_string_hash,
                "$vector": embedding_vector,
            },
            upsert=True,
        )

    @override
    async def aupdate(
        self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE
    ) -> None:
        await self.astra_env.aensure_db_setup()
        doc_id = self._make_id(prompt, llm_string)
        llm_string_hash = _hash(llm_string)
        embedding_vector = await self._aget_embedding(text=prompt)
        body = _dumps_generations(return_val)
        await self.async_collection.find_one_and_replace(
            {"_id": doc_id},
            {
                "_id": doc_id,
                "body_blob": body,
                "llm_string_hash": llm_string_hash,
                "$vector": embedding_vector,
            },
            upsert=True,
        )

    @override
    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        hit_with_id = self.lookup_with_id(prompt, llm_string)
        return hit_with_id[1] if hit_with_id is not None else None

    @override
    async def alookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        hit_with_id = await self.alookup_with_id(prompt, llm_string)
        return hit_with_id[1] if hit_with_id is not None else None

    def lookup_with_id(
        self, prompt: str, llm_string: str
    ) -> tuple[str, RETURN_VAL_TYPE] | None:
        """Look up based on prompt and llm_string.

        If there are hits, return (document_id, cached_entry) for the top hit
        """
        self.astra_env.ensure_db_setup()
        prompt_embedding: list[float] = self._get_embedding(text=prompt)
        llm_string_hash = _hash(llm_string)

        hit = self.collection.find_one(
            filter={
                "llm_string_hash": llm_string_hash,
            },
            sort={"$vector": prompt_embedding},
            projection={"body_blob": True, "_id": True},
            include_similarity=True,
        )

        if hit is None or hit["$similarity"] < self.similarity_threshold:
            return None

        generations = _loads_generations(hit["body_blob"])
        if generations is None:
            return None

        # this protects against malformed cached items:
        return hit["_id"], generations

    async def alookup_with_id(
        self, prompt: str, llm_string: str
    ) -> tuple[str, RETURN_VAL_TYPE] | None:
        """Look up based on prompt and llm_string.

        If there are hits, return (document_id, cached_entry) for the top hit
        """
        await self.astra_env.aensure_db_setup()
        prompt_embedding: list[float] = await self._aget_embedding(text=prompt)
        llm_string_hash = _hash(llm_string)

        hit = await self.async_collection.find_one(
            filter={
                "llm_string_hash": llm_string_hash,
            },
            sort={"$vector": prompt_embedding},
            projection={"body_blob": True, "_id": True},
            include_similarity=True,
        )

        if hit is None or hit["$similarity"] < self.similarity_threshold:
            return None

        generations = _loads_generations(hit["body_blob"])
        if generations is None:
            return None

        # this protects against malformed cached items:
        return hit["_id"], generations

    def lookup_with_id_through_llm(
        self, prompt: str, llm: LLM, stop: list[str] | None = None
    ) -> tuple[str, RETURN_VAL_TYPE] | None:
        """Look up based on prompt and LLM.

        If there are hits, return (document_id, cached_entry) for the top hit
        """
        llm_string = get_prompts(
            {**llm.dict(), "stop": stop},
            [],
        )[1]
        return self.lookup_with_id(prompt, llm_string=llm_string)

    async def alookup_with_id_through_llm(
        self, prompt: str, llm: LLM, stop: list[str] | None = None
    ) -> tuple[str, RETURN_VAL_TYPE] | None:
        """Look up based on prompt and LLM.

        If there are hits, return (document_id, cached_entry) for the top hit
        """
        llm_string = (
            await aget_prompts(
                {**llm.dict(), "stop": stop},
                [],
            )
        )[1]
        return await self.alookup_with_id(prompt, llm_string=llm_string)

    def delete_by_document_id(self, document_id: str) -> None:
        """Delete by document ID.

        Given this is a "similarity search" cache, an invalidation pattern
        that makes sense is first a lookup to get an ID, and then deleting
        with that ID. This is for the second step.
        """
        self.astra_env.ensure_db_setup()
        self.collection.delete_one({"_id": document_id})

    async def adelete_by_document_id(self, document_id: str) -> None:
        """Delete by document ID.

        Given this is a "similarity search" cache, an invalidation pattern
        that makes sense is first a lookup to get an ID, and then deleting
        with that ID. This is for the second step.
        """
        await self.astra_env.aensure_db_setup()
        await self.async_collection.delete_one({"_id": document_id})

    @override
    def clear(self, **kwargs: Any) -> None:
        self.astra_env.ensure_db_setup()
        self.collection.delete_many({})

    @override
    async def aclear(self, **kwargs: Any) -> None:
        await self.astra_env.aensure_db_setup()
        await self.async_collection.delete_many({})
