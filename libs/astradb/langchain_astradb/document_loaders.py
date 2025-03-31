"""Loader for loading documents from DataStax Astra DB."""

from __future__ import annotations

import json
import logging
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Iterator,
)

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from typing_extensions import override

from langchain_astradb.utils.astradb import (
    COMPONENT_NAME_LOADER,
    SetupMode,
    _AstraDBCollectionEnvironment,
)

if TYPE_CHECKING:
    from astrapy.authentication import TokenProvider

logger = logging.getLogger(__name__)

_NOT_SET = object()


class AstraDBLoader(BaseLoader):
    def __init__(
        self,
        collection_name: str,
        *,
        token: str | TokenProvider | None = None,
        api_endpoint: str | None = None,
        environment: str | None = None,
        namespace: str | None = None,
        filter_criteria: dict[str, Any] | None = None,
        projection: dict[str, Any] | None = _NOT_SET,  # type: ignore[assignment]
        limit: int | None = None,
        nb_prefetched: int = _NOT_SET,  # type: ignore[assignment]
        page_content_mapper: Callable[[dict], str] = json.dumps,
        metadata_mapper: Callable[[dict], dict[str, Any]] | None = None,
        ext_callers: list[tuple[str | None, str | None] | str | None] | None = None,
    ) -> None:
        """Load DataStax Astra DB documents.

        Args:
            collection_name: name of the Astra DB collection to use.
            token: API token for Astra DB usage, either in the form of a string
                or a subclass of `astrapy.authentication.TokenProvider`.
                If not provided, the environment variable
                ASTRA_DB_APPLICATION_TOKEN is inspected.
            api_endpoint: full URL to the API endpoint, such as
                `https://<DB-ID>-us-east1.apps.astra.datastax.com`. If not provided,
                the environment variable ASTRA_DB_API_ENDPOINT is inspected.
            environment: a string specifying the environment of the target Data API.
                If omitted, defaults to "prod" (Astra DB production).
                Other values are in `astrapy.constants.Environment` enum class.
            namespace: namespace (aka keyspace) where the collection resides.
                If not provided, the environment variable ASTRA_DB_KEYSPACE is
                inspected. Defaults to the database's "default namespace".
            filter_criteria: Criteria to filter documents.
            projection: Specifies the fields to return. If not provided, reads
                fall back to the Data API default projection.
            limit: a maximum number of documents to return in the read query.
            nb_prefetched: Max number of documents to pre-fetch.
                *IGNORED starting from v. 0.3.5: astrapy v1.0+ does not support it.*
            page_content_mapper: Function applied to collection documents to create
                the `page_content` of the LangChain Document. Defaults to `json.dumps`.
            metadata_mapper: Function applied to collection documents to create the
                `metadata` of the LangChain Document. Defaults to returning the
                 namespace, API endpoint and collection name.
            ext_callers: one or more caller identities to identify Data API calls
                in the User-Agent header. This is a list of (name, version) pairs,
                or just strings if no version info is provided, which, if supplied,
                becomes the leading part of the User-Agent string in all API requests
                related to this component.
        """
        astra_db_env = _AstraDBCollectionEnvironment(
            collection_name=collection_name,
            token=token,
            api_endpoint=api_endpoint,
            keyspace=namespace,
            environment=environment,
            setup_mode=SetupMode.OFF,
            ext_callers=ext_callers,
            component_name=COMPONENT_NAME_LOADER,
        )
        self.astra_db_env = astra_db_env
        self.filter = filter_criteria
        self._projection: dict[str, Any] | None = (
            projection if projection is not _NOT_SET else {"*": True}
        )
        # warning if 'prefetched' passed
        if nb_prefetched is not _NOT_SET:
            warnings.warn(
                (
                    "Parameter 'nb_prefetched' is not supported by the Data API "
                    "client and will be ignored in reading document."
                ),
                UserWarning,
                stacklevel=2,
            )

        self.limit = limit
        self.nb_prefetched = nb_prefetched
        self.page_content_mapper = page_content_mapper
        self.metadata_mapper = metadata_mapper or (
            lambda _: {
                "namespace": self.astra_db_env.database.keyspace,
                "api_endpoint": self.astra_db_env.database.api_endpoint,
                "collection": collection_name,
            }
        )

    def _to_langchain_doc(self, doc: dict[str, Any]) -> Document:
        return Document(
            page_content=self.page_content_mapper(doc),
            metadata=self.metadata_mapper(doc),
        )

    @override
    def lazy_load(self) -> Iterator[Document]:
        for doc in self.astra_db_env.collection.find(
            filter=self.filter,
            projection=self._projection,
            limit=self.limit,
            # prefetch: not available at the moment (silently ignored)
            # prefetched=self.nb_prefetched,
        ):
            yield self._to_langchain_doc(doc)

    async def aload(self) -> list[Document]:
        """Load data into Document objects."""
        return [doc async for doc in self.alazy_load()]

    @override
    async def alazy_load(self) -> AsyncIterator[Document]:
        async for doc in self.astra_db_env.async_collection.find(
            filter=self.filter,
            projection=self._projection,
            limit=self.limit,
            # prefetch: not available at the moment (silently ignored):
            # prefetched=self.nb_prefetched,
        ):
            yield self._to_langchain_doc(doc)
