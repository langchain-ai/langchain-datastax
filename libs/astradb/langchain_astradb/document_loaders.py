from __future__ import annotations

import json
import logging
import warnings
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
)

from astrapy.db import AstraDB, AsyncAstraDB
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from langchain_astradb.utils.astradb import (
    SetupMode,
    _AstraDBCollectionEnvironment,
)

logger = logging.getLogger(__name__)

_NOT_SET = object()


class AstraDBLoader(BaseLoader):
    def __init__(
        self,
        collection_name: str,
        *,
        token: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        environment: Optional[str] = None,
        astra_db_client: Optional[AstraDB] = None,
        async_astra_db_client: Optional[AsyncAstraDB] = None,
        namespace: Optional[str] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = _NOT_SET,  # type: ignore[assignment]
        find_options: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        nb_prefetched: int = 1000,
        page_content_mapper: Callable[[Dict], str] = json.dumps,
        metadata_mapper: Optional[Callable[[Dict], Dict[str, Any]]] = None,
    ) -> None:
        """Load DataStax Astra DB documents.

        Args:
            collection_name: name of the Astra DB collection to use.
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
            namespace: namespace (aka keyspace) where the collection resides.
                If not provided, the environment variable ASTRA_DB_KEYSPACE is
                inspected. Defaults to the database's "default namespace".
            filter_criteria: Criteria to filter documents.
            projection: Specifies the fields to return. If not provided, reads
                fall back to the Data API default projection.
            find_options: Additional options for the query.
                *DEPRECATED starting from version 0.3.5.*
                *For limiting, please use `limit`. Other options are ignored.*
            limit: a maximum number of documents to return in the read query.
            nb_prefetched: Max number of documents to pre-fetch. Defaults to 1000.
            page_content_mapper: Function applied to collection documents to create
                the `page_content` of the LangChain Document. Defaults to `json.dumps`.
        """
        astra_db_env = _AstraDBCollectionEnvironment(
            collection_name=collection_name,
            token=token,
            api_endpoint=api_endpoint,
            environment=environment,
            astra_db_client=astra_db_client,
            async_astra_db_client=async_astra_db_client,
            namespace=namespace,
            setup_mode=SetupMode.OFF,
        )
        self.astra_db_env = astra_db_env
        self.filter = filter_criteria
        self._projection: Optional[Dict[str, Any]] = (
            projection if projection is not _NOT_SET else {"*": True}
        )
        # normalizing limit and options and deprecations
        _limit: Optional[int]
        if "limit" in (find_options or {}):
            if limit is not None:
                raise ValueError(
                    "Duplicate 'limit' directive supplied. Please remove it "
                    "from the 'find_options' map parameter."
                )
            else:
                warnings.warn(
                    (
                        "Passing 'limit' as part of the 'find_options' "
                        "dictionary is deprecated starting from version 0.3.5. "
                        "Please switch to passing 'limit=<number>' "
                        "directly in the constructor."
                    ),
                    DeprecationWarning,
                )
                _limit = (find_options or {})["limit"]
        else:
            _limit = limit
        self.limit = _limit
        _other_option_keys = set((find_options or {}).keys()) - {"limit"}
        if _other_option_keys:
            warnings.warn(
                (
                    "Unknown keys passed in the 'find_options' dictionary. "
                    "This parameter is deprecated starting from version 0.3.5."
                ),
                DeprecationWarning,
            )
        #
        self.nb_prefetched = nb_prefetched
        self.page_content_mapper = page_content_mapper
        self.metadata_mapper = metadata_mapper or (
            lambda _: {
                "namespace": self.astra_db_env.database.namespace,
                "api_endpoint": self.astra_db_env.database.api_endpoint,
                "collection": collection_name,
            }
        )

    def _to_langchain_doc(self, doc: Dict[str, Any]) -> Document:
        return Document(
            page_content=self.page_content_mapper(doc),
            metadata=self.metadata_mapper(doc),
        )

    def lazy_load(self) -> Iterator[Document]:
        for doc in self.astra_db_env.collection.find(
            filter=self.filter,
            projection=self._projection,
            limit=self.limit,
            # prefetch: not available at the moment (silently ignored)
            # prefetched=self.nb_prefetched,
        ):
            yield self._to_langchain_doc(doc)

    async def aload(self) -> List[Document]:
        """Load data into Document objects."""
        return [doc async for doc in self.alazy_load()]

    async def alazy_load(self) -> AsyncIterator[Document]:
        async for doc in self.astra_db_env.async_collection.find(
            filter=self.filter,
            projection=self._projection,
            limit=self.limit,
            # prefetch: not available at the moment (silently ignored):
            # prefetched=self.nb_prefetched,
        ):
            yield self._to_langchain_doc(doc)
