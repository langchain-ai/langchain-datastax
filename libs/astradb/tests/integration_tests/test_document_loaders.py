"""Test of Astra DB document loader class `AstraDBLoader`

Required to run this test:
    - a recent `astrapy` Python package available
    - an Astra DB instance;
    - the two environment variables set:
        export ASTRA_DB_API_ENDPOINT="https://<DB-ID>-us-east1.apps.astra.datastax.com"
        export ASTRA_DB_APPLICATION_TOKEN="AstraCS:........."
    - optionally this as well (otherwise defaults are used):
        export ASTRA_DB_KEYSPACE="my_keyspace"
"""

from __future__ import annotations

import json
import os
import uuid
from typing import TYPE_CHECKING, AsyncIterator, Iterator

import pytest

from langchain_astradb import AstraDBLoader

from .conftest import AstraDBCredentials, astra_db_env_vars_available

if TYPE_CHECKING:
    from astrapy import AsyncCollection, Collection, Database
    from astrapy.db import AstraDB


@pytest.fixture
def collection(database: Database) -> Iterator[Collection]:
    collection_name = f"lc_test_loader_{str(uuid.uuid4()).split('-')[0]}"
    collection = database.create_collection(collection_name)
    collection.insert_many([{"foo": "bar", "baz": "qux"}] * 20)
    collection.insert_many(
        [{"foo": "bar2", "baz": "qux"}] * 4 + [{"foo": "bar", "baz": "qux"}] * 4
    )

    yield collection

    collection.drop()


@pytest.fixture
async def async_collection(database: Database) -> AsyncIterator[AsyncCollection]:
    adatabase = database.to_async()
    collection_name = f"lc_test_loader_{str(uuid.uuid4()).split('-')[0]}"
    collection = await adatabase.create_collection(collection_name)
    await collection.insert_many([{"foo": "bar", "baz": "qux"}] * 20)
    await collection.insert_many(
        [{"foo": "bar2", "baz": "qux"}] * 4 + [{"foo": "bar", "baz": "qux"}] * 4
    )

    yield collection

    await collection.drop()


@pytest.mark.requires("astrapy")
@pytest.mark.skipif(
    not astra_db_env_vars_available(), reason="Missing Astra DB env. vars"
)
class TestAstraDB:
    def test_astradb_loader_prefetched_sync(
        self,
        collection: Collection,
        astra_db_credentials: AstraDBCredentials,
    ) -> None:
        """Using 'prefetched' should give a warning but work nonetheless."""
        with pytest.warns(UserWarning) as rec_warnings:
            loader = AstraDBLoader(
                collection.name,
                token=astra_db_credentials["token"],
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

    def test_astradb_loader_sync(
        self,
        collection: Collection,
        astra_db_credentials: AstraDBCredentials,
    ) -> None:
        loader = AstraDBLoader(
            collection.name,
            token=astra_db_credentials["token"],
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
                "namespace": collection.namespace,
                "api_endpoint": collection.database.api_endpoint,
                "collection": collection.name,
            }

    def test_page_content_mapper_sync(
        self,
        collection: Collection,
        astra_db_credentials: AstraDBCredentials,
    ) -> None:
        loader = AstraDBLoader(
            collection.name,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            limit=30,
            page_content_mapper=lambda x: x["foo"],
            filter_criteria={"foo": "bar"},
        )
        docs = loader.lazy_load()
        doc = next(docs)

        assert doc.page_content == "bar"

    def test_metadata_mapper_sync(
        self,
        collection: Collection,
        astra_db_credentials: AstraDBCredentials,
    ) -> None:
        loader = AstraDBLoader(
            collection.name,
            token=astra_db_credentials["token"],
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
        async_collection: AsyncCollection,
        astra_db_credentials: AstraDBCredentials,
    ) -> None:
        """Using 'prefetched' should give a warning but work nonetheless."""
        with pytest.warns(UserWarning) as rec_warnings:
            loader = AstraDBLoader(
                async_collection.name,
                token=astra_db_credentials["token"],
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

    async def test_astradb_loader_async(
        self,
        async_collection: AsyncCollection,
        astra_db_credentials: AstraDBCredentials,
    ) -> None:
        loader = AstraDBLoader(
            async_collection.name,
            token=astra_db_credentials["token"],
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
                "namespace": async_collection.namespace,
                "api_endpoint": async_collection.database.api_endpoint,
                "collection": async_collection.name,
            }

    async def test_page_content_mapper_async(
        self,
        async_collection: AsyncCollection,
        astra_db_credentials: AstraDBCredentials,
    ) -> None:
        loader = AstraDBLoader(
            async_collection.name,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            limit=30,
            page_content_mapper=lambda x: x["foo"],
            filter_criteria={"foo": "bar"},
        )
        doc = await loader.alazy_load().__anext__()
        assert doc.page_content == "bar"

    async def test_metadata_mapper_async(
        self,
        async_collection: AsyncCollection,
        astra_db_credentials: AstraDBCredentials,
    ) -> None:
        loader = AstraDBLoader(
            async_collection.name,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            limit=30,
            metadata_mapper=lambda x: {"a": x["foo"]},
            filter_criteria={"foo": "bar"},
        )
        doc = await loader.alazy_load().__anext__()
        assert doc.metadata == {"a": "bar"}

    @pytest.mark.skipif(
        os.environ.get("ASTRA_DB_ENVIRONMENT", "prod").upper() != "PROD",
        reason="Can run on Astra DB prod only",
    )
    def test_astradb_loader_coreclients_init(
        self,
        astra_db_credentials: AstraDBCredentials,
        collection: Collection,
        core_astra_db: AstraDB,
    ) -> None:
        """A deprecation warning from passing a (core) AstraDB, but it works."""
        loader_init_ok = AstraDBLoader(
            collection_name=collection.name,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            limit=1,
        )
        docs = loader_init_ok.load()
        # create an equivalent loader with core AstraDB in init
        with pytest.warns(DeprecationWarning) as rec_warnings:
            loader_init_core = AstraDBLoader(
                collection_name=collection.name,
                astra_db_client=core_astra_db,
                limit=1,
            )
        f_rec_warnings = [
            wrn for wrn in rec_warnings if issubclass(wrn.category, DeprecationWarning)
        ]
        assert len(f_rec_warnings) == 1
        assert loader_init_core.load() == docs

    def test_astradb_loader_findoptions_deprecation(
        self,
        astra_db_credentials: AstraDBCredentials,
        collection: Collection,
    ) -> None:
        """Test deprecation of 'find_options' and related warnings/errors."""
        loader0 = AstraDBLoader(
            collection_name=collection.name,
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            limit=1,
        )
        docs0 = loader0.load()

        with pytest.warns(DeprecationWarning) as rec_warnings:
            loader_lo = AstraDBLoader(
                collection_name=collection.name,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                find_options={"limit": 1},
            )
        f_rec_warnings = [
            wrn for wrn in rec_warnings if issubclass(wrn.category, DeprecationWarning)
        ]
        assert len(f_rec_warnings) == 1
        assert loader_lo.load() == docs0

        with pytest.raises(ValueError, match="Duplicate 'limit' directive supplied."):
            AstraDBLoader(
                collection_name=collection.name,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                limit=1,
                find_options={"limit": 1},
            )

        with pytest.warns(DeprecationWarning) as rec_warnings:
            loader_uo = AstraDBLoader(
                collection_name=collection.name,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                find_options={"planets": 8, "spiders": 40000},
                limit=1,
            )
        f_rec_warnings = [
            wrn for wrn in rec_warnings if issubclass(wrn.category, DeprecationWarning)
        ]
        assert len(f_rec_warnings) == 1
        assert loader_uo.load() == docs0
