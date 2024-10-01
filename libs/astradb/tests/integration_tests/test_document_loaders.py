"""Test of Astra DB document loader class `AstraDBLoader`"""

from __future__ import annotations

import json
import os
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
    from astrapy.db import AstraDB


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
async def async_document_loader_collection(
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
        with pytest.warns(UserWarning) as rec_warnings:
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
            page_content_mapper=lambda x: x["foo"],
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
        with pytest.warns(UserWarning) as rec_warnings:
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
            page_content_mapper=lambda x: x["foo"],
            filter_criteria={"foo": "bar"},
        )
        doc = await loader.alazy_load().__anext__()
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
        doc = await loader.alazy_load().__anext__()
        assert doc.metadata == {"a": "bar"}

    @pytest.mark.skipif(
        os.environ.get("ASTRA_DB_ENVIRONMENT", "prod").upper() != "PROD",
        reason="Can run on Astra DB prod only",
    )
    def test_astradb_loader_coreclients_init(
        self,
        astra_db_credentials: AstraDBCredentials,
        core_astra_db: AstraDB,
        document_loader_collection: Collection,
    ) -> None:
        """
        A deprecation warning from passing a (core) AstraDB, but it works.
        Note there is no sync/async here: this class always has SetupMode.OFF.
        """
        loader_init_ok = AstraDBLoader(
            collection_name=document_loader_collection.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            limit=1,
        )
        docs = loader_init_ok.load()
        # create an equivalent loader with core AstraDB in init
        with pytest.warns(DeprecationWarning) as rec_warnings:
            loader_init_core = AstraDBLoader(
                collection_name=document_loader_collection.name,
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
        document_loader_collection: Collection,
    ) -> None:
        """Test deprecation of 'find_options' and related warnings/errors."""
        loader0 = AstraDBLoader(
            collection_name=document_loader_collection.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            limit=1,
        )
        docs0 = loader0.load()

        with pytest.warns(DeprecationWarning) as rec_warnings:
            loader_lo = AstraDBLoader(
                collection_name=document_loader_collection.name,
                token=StaticTokenProvider(astra_db_credentials["token"]),
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
                collection_name=document_loader_collection.name,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                limit=1,
                find_options={"limit": 1},
            )

        with pytest.warns(DeprecationWarning) as rec_warnings:
            loader_uo = AstraDBLoader(
                collection_name=document_loader_collection.name,
                token=StaticTokenProvider(astra_db_credentials["token"]),
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
