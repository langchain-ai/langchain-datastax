"""Test of Astra DB document loader class `AstraDBLoader`"""

from __future__ import annotations

import json
from operator import itemgetter
from typing import TYPE_CHECKING

import pytest
from astrapy.authentication import StaticTokenProvider

from langchain_astradb import AstraDBLoader

from .conftest import (
    AstraDBCredentials,
    astra_db_env_vars_available,
)

if TYPE_CHECKING:
    from astrapy import AsyncCollection, Collection, Database


@pytest.fixture(scope="module")
def document_loader_collection(
    collection_idxall: Collection,
) -> Collection:
    collection_idxall.delete_many({})
    collection_idxall.insert_many(
        [{"foo": "bar", "baz": "qux"}] * 24 + [{"foo": "bar2", "baz": "qux"}] * 4
    )
    return collection_idxall


@pytest.fixture
def async_document_loader_collection(
    collection_idxall: Collection,
) -> AsyncCollection:
    return collection_idxall.to_async()


@pytest.mark.skipif(
    not astra_db_env_vars_available(), reason="Missing Astra DB env. vars"
)
class TestAstraDB:
    def test_astradb_loader_prefetched_sync(
        self,
        astra_db_credentials: AstraDBCredentials,
        document_loader_collection: Collection,
    ) -> None:
        """Using 'prefetched' should give a warning but work nonetheless."""
        with pytest.warns(
            UserWarning, match="Parameter 'nb_prefetched' is not supported"
        ) as rec_warnings:
            loader = AstraDBLoader(
                document_loader_collection.name,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                nb_prefetched=123,
                projection={"foo": 1},
                limit=22,
                filter_criteria={"foo": "bar"},
            )
        f_rec_warnings = [
            wrn for wrn in rec_warnings if issubclass(wrn.category, UserWarning)
        ]
        assert len(f_rec_warnings) == 1

        docs = loader.load()
        assert len(docs) == 22

    def test_astradb_loader_base_sync(
        self,
        astra_db_credentials: AstraDBCredentials,
        database: Database,
        document_loader_collection: Collection,
    ) -> None:
        loader = AstraDBLoader(
            document_loader_collection.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            projection={"foo": 1},
            limit=22,
            filter_criteria={"foo": "bar"},
        )
        docs = loader.load()
        assert len(docs) == 22

        ids = set()
        for doc in docs:
            content = json.loads(doc.page_content)
            assert content["foo"] == "bar"
            assert "baz" not in content
            assert content["_id"] not in ids
            ids.add(content["_id"])
            assert doc.metadata == {
                "namespace": database.keyspace,
                "api_endpoint": astra_db_credentials["api_endpoint"],
                "collection": document_loader_collection.name,
            }

        loader2 = AstraDBLoader(
            document_loader_collection.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            projection={"foo": 1},
            limit=22,
            filter_criteria={"foo": "bar2"},
        )
        docs2 = loader2.load()
        assert len(docs2) == 4

    def test_page_content_mapper_sync(
        self,
        astra_db_credentials: AstraDBCredentials,
        document_loader_collection: Collection,
    ) -> None:
        loader = AstraDBLoader(
            document_loader_collection.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            limit=30,
            page_content_mapper=itemgetter("foo"),
            filter_criteria={"foo": "bar"},
        )
        docs = loader.lazy_load()
        doc = next(docs)

        assert doc.page_content == "bar"

    def test_metadata_mapper_sync(
        self,
        astra_db_credentials: AstraDBCredentials,
        document_loader_collection: Collection,
    ) -> None:
        loader = AstraDBLoader(
            document_loader_collection.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            limit=30,
            metadata_mapper=lambda x: {"a": x["foo"]},
            filter_criteria={"foo": "bar"},
        )
        docs = loader.lazy_load()
        doc = next(docs)

        assert doc.metadata == {"a": "bar"}

    async def test_astradb_loader_prefetched_async(
        self,
        astra_db_credentials: AstraDBCredentials,
        database: Database,
        async_document_loader_collection: AsyncCollection,
    ) -> None:
        """Using 'prefetched' should give a warning but work nonetheless."""
        with pytest.warns(
            UserWarning, match="Parameter 'nb_prefetched' is not supported"
        ) as rec_warnings:
            loader = AstraDBLoader(
                async_document_loader_collection.name,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                nb_prefetched=1,
                projection={"foo": 1},
                limit=22,
                filter_criteria={"foo": "bar"},
            )
        # cleaning out 'spurious' "unclosed socket/transport..." warnings
        f_rec_warnings = [
            wrn for wrn in rec_warnings if issubclass(wrn.category, UserWarning)
        ]
        assert len(f_rec_warnings) == 1
        docs = await loader.aload()
        assert len(docs) == 22

        ids = set()
        for doc in docs:
            content = json.loads(doc.page_content)
            assert content["foo"] == "bar"
            assert "baz" not in content
            assert content["_id"] not in ids
            ids.add(content["_id"])
            assert doc.metadata == {
                "namespace": database.keyspace,
                "api_endpoint": astra_db_credentials["api_endpoint"],
                "collection": async_document_loader_collection.name,
            }

        loader2 = AstraDBLoader(
            async_document_loader_collection.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            projection={"foo": 1},
            limit=22,
            filter_criteria={"foo": "bar2"},
        )
        docs2 = await loader2.aload()
        assert len(docs2) == 4

    async def test_astradb_loader_base_async(
        self,
        astra_db_credentials: AstraDBCredentials,
        database: Database,
        async_document_loader_collection: AsyncCollection,
    ) -> None:
        loader = AstraDBLoader(
            async_document_loader_collection.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            projection={"foo": 1},
            limit=22,
            filter_criteria={"foo": "bar"},
        )
        docs = await loader.aload()

        assert len(docs) == 22
        ids = set()
        for doc in docs:
            content = json.loads(doc.page_content)
            assert content["foo"] == "bar"
            assert "baz" not in content
            assert content["_id"] not in ids
            ids.add(content["_id"])
            assert doc.metadata == {
                "namespace": database.keyspace,
                "api_endpoint": astra_db_credentials["api_endpoint"],
                "collection": async_document_loader_collection.name,
            }

    async def test_page_content_mapper_async(
        self,
        astra_db_credentials: AstraDBCredentials,
        async_document_loader_collection: AsyncCollection,
    ) -> None:
        loader = AstraDBLoader(
            async_document_loader_collection.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            limit=30,
            page_content_mapper=itemgetter("foo"),
            filter_criteria={"foo": "bar"},
        )
        doc = await anext(loader.alazy_load())
        assert doc.page_content == "bar"

    async def test_metadata_mapper_async(
        self,
        astra_db_credentials: AstraDBCredentials,
        async_document_loader_collection: AsyncCollection,
    ) -> None:
        loader = AstraDBLoader(
            async_document_loader_collection.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            limit=30,
            metadata_mapper=lambda x: {"a": x["foo"]},
            filter_criteria={"foo": "bar"},
        )
        doc = await anext(loader.alazy_load())
        assert doc.metadata == {"a": "bar"}
