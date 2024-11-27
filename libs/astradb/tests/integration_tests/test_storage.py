"""Implement integration tests for AstraDB storage."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from astrapy.authentication import StaticTokenProvider

from langchain_astradb.storage import AstraDBByteStore, AstraDBStore
from langchain_astradb.utils.astradb import SetupMode

from .conftest import (
    EPHEMERAL_CUSTOM_IDX_NAME,
    EPHEMERAL_LEGACY_IDX_NAME,
    AstraDBCredentials,
    astra_db_env_vars_available,
)

if TYPE_CHECKING:
    from astrapy import Collection, Database
    from astrapy.db import AstraDB


@pytest.fixture
def astra_db_empty_store(
    astra_db_credentials: AstraDBCredentials,
    collection_idxid: Collection,
) -> AstraDBStore:
    collection_idxid.delete_many({})
    return AstraDBStore(
        collection_name=collection_idxid.name,
        token=StaticTokenProvider(astra_db_credentials["token"]),
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
        setup_mode=SetupMode.OFF,
    )


@pytest.fixture
async def astra_db_empty_store_async(
    astra_db_credentials: AstraDBCredentials,
    collection_idxid: Collection,
) -> AstraDBStore:
    await collection_idxid.to_async().delete_many({})
    return AstraDBStore(
        collection_name=collection_idxid.name,
        token=StaticTokenProvider(astra_db_credentials["token"]),
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
        setup_mode=SetupMode.ASYNC,
    )


@pytest.fixture
def astra_db_empty_byte_store(
    astra_db_credentials: AstraDBCredentials,
    collection_idxid: Collection,
) -> AstraDBByteStore:
    collection_idxid.delete_many({})
    return AstraDBByteStore(
        collection_name=collection_idxid.name,
        token=StaticTokenProvider(astra_db_credentials["token"]),
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
        setup_mode=SetupMode.OFF,
    )


@pytest.mark.skipif(
    not astra_db_env_vars_available(), reason="Missing Astra DB env. vars"
)
class TestAstraDBStore:
    def test_store_crud_sync(
        self,
        astra_db_empty_store: AstraDBStore,
    ) -> None:
        """Test AstraDBStore mget/mset/mdelete method."""
        astra_db_empty_store.mset([("key1", [0.1, 0.2]), ("key2", "value2")])
        assert astra_db_empty_store.mget(["key1", "key2"]) == [[0.1, 0.2], "value2"]
        astra_db_empty_store.mdelete(["key1", "key2"])
        assert astra_db_empty_store.mget(["key1", "key2"]) == [None, None]

    async def test_store_crud_async(
        self,
        astra_db_empty_store_async: AstraDBStore,
    ) -> None:
        """Test AstraDBStore amget/amset/amdelete method. Async version."""
        await astra_db_empty_store_async.amset(
            [
                ("key1", [0.1, 0.2]),
                ("key2", "value2"),
            ]
        )
        assert await astra_db_empty_store_async.amget(["key1", "key2"]) == [
            [0.1, 0.2],
            "value2",
        ]
        await astra_db_empty_store_async.amdelete(["key1", "key2"])
        assert await astra_db_empty_store_async.amget(["key1", "key2"]) == [None, None]

    def test_store_massive_write_with_replace_sync(
        self,
        astra_db_empty_store: AstraDBStore,
    ) -> None:
        """Testing the insert-many-and-replace-some patterns thoroughly."""
        full_size = 300
        first_group_size = 150
        second_group_slicer = [30, 100, 2]
        max_values_in_in = 100
        ids_and_texts = [
            (
                f"doc_{idx}",
                f"document number {idx}",
            )
            for idx in range(full_size)
        ]

        # massive insertion on empty (zip and rezip for uniformity with later)
        group0_ids, group0_texts = list(zip(*ids_and_texts[0:first_group_size]))
        astra_db_empty_store.mset(list(zip(group0_ids, group0_texts)))

        # massive insertion with many overwrites scattered through
        # (we change the text to later check on DB for successful update)
        _s, _e, _st = second_group_slicer
        group1_ids, group1_texts_pre = list(
            zip(*(ids_and_texts[_s:_e:_st] + ids_and_texts[first_group_size:full_size]))
        )
        group1_texts = [txt.upper() for txt in group1_texts_pre]
        astra_db_empty_store.mset(list(zip(group1_ids, group1_texts)))

        # final read (we want the IDs to do a full check)
        expected_text_by_id = {
            **dict(zip(group0_ids, group0_texts)),
            **dict(zip(group1_ids, group1_texts)),
        }
        all_ids = [doc_id for doc_id, _ in ids_and_texts]
        # The Data API can handle at most max_values_in_in entries, let's chunk
        all_vals = [
            val
            for chunk_start in range(0, full_size, max_values_in_in)
            for val in astra_db_empty_store.mget(
                all_ids[chunk_start : chunk_start + max_values_in_in]
            )
        ]
        for val, doc_id in zip(all_vals, all_ids):
            assert val == expected_text_by_id[doc_id]

    async def test_store_massive_write_with_replace_async(
        self,
        astra_db_empty_store_async: AstraDBStore,
    ) -> None:
        """
        Testing the insert-many-and-replace-some patterns thoroughly.
        Async version.
        """
        full_size = 300
        first_group_size = 150
        second_group_slicer = [30, 100, 2]
        max_values_in_in = 100
        ids_and_texts = [
            (
                f"doc_{idx}",
                f"document number {idx}",
            )
            for idx in range(full_size)
        ]

        # massive insertion on empty (zip and rezip for uniformity with later)
        group0_ids, group0_texts = list(zip(*ids_and_texts[0:first_group_size]))
        await astra_db_empty_store_async.amset(list(zip(group0_ids, group0_texts)))

        # massive insertion with many overwrites scattered through
        # (we change the text to later check on DB for successful update)
        _s, _e, _st = second_group_slicer
        group1_ids, group1_texts_pre = list(
            zip(*(ids_and_texts[_s:_e:_st] + ids_and_texts[first_group_size:full_size]))
        )
        group1_texts = [txt.upper() for txt in group1_texts_pre]
        await astra_db_empty_store_async.amset(list(zip(group1_ids, group1_texts)))

        # final read (we want the IDs to do a full check)
        expected_text_by_id = {
            **dict(zip(group0_ids, group0_texts)),
            **dict(zip(group1_ids, group1_texts)),
        }
        all_ids = [doc_id for doc_id, _ in ids_and_texts]
        # The Data API can handle at most max_values_in_in entries, let's chunk
        all_vals = [
            val
            for chunk_start in range(0, full_size, max_values_in_in)
            for val in await astra_db_empty_store_async.amget(
                all_ids[chunk_start : chunk_start + max_values_in_in]
            )
        ]
        for val, doc_id in zip(all_vals, all_ids):
            assert val == expected_text_by_id[doc_id]

    def test_store_yield_keys_sync(
        self,
        astra_db_empty_store: AstraDBStore,
    ) -> None:
        """Test of yield_keys."""
        astra_db_empty_store.mset([("key1", [0.1, 0.2]), ("key2", "value2")])
        assert set(astra_db_empty_store.yield_keys()) == {"key1", "key2"}
        assert set(astra_db_empty_store.yield_keys(prefix="key")) == {"key1", "key2"}
        assert set(astra_db_empty_store.yield_keys(prefix="lang")) == set()

    async def test_store_yield_keys_async(
        self,
        astra_db_empty_store_async: AstraDBStore,
    ) -> None:
        """Test of yield_keys, async version"""
        await astra_db_empty_store_async.amset(
            [
                ("key1", [0.1, 0.2]),
                ("key2", "value2"),
            ]
        )
        assert {k async for k in astra_db_empty_store_async.ayield_keys()} == {
            "key1",
            "key2",
        }
        assert {
            k async for k in astra_db_empty_store_async.ayield_keys(prefix="key")
        } == {"key1", "key2"}
        assert {
            k async for k in astra_db_empty_store_async.ayield_keys(prefix="lang")
        } == set()

    def test_bytestore_crud_sync(
        self,
        astra_db_empty_byte_store: AstraDBByteStore,
    ) -> None:
        """
        Test AstraDBByteStore mget/mset/mdelete method.

        Since this class shares most of its logic with AstraDBStore,
        there's no need to test async nor the other methods/pathways.
        """
        astra_db_empty_byte_store.mset([("key1", b"value1"), ("key2", b"value2")])
        assert astra_db_empty_byte_store.mget(["key1", "key2"]) == [
            b"value1",
            b"value2",
        ]
        astra_db_empty_byte_store.mdelete(["key1", "key2"])
        assert astra_db_empty_byte_store.mget(["key1", "key2"]) == [None, None]

    @pytest.mark.skipif(
        os.environ.get("ASTRA_DB_ENVIRONMENT", "prod").upper() != "PROD",
        reason="Can run on Astra DB production environment only",
    )
    def test_store_coreclients_init_sync(
        self,
        core_astra_db: AstraDB,
        astra_db_empty_store: AstraDBStore,
    ) -> None:
        """A deprecation warning from passing a (core) AstraDB, but it works."""
        astra_db_empty_store.mset([("key", "val123")])

        # create an equivalent store with core AstraDB in init
        with pytest.warns(DeprecationWarning) as rec_warnings:
            store_init_core = AstraDBStore(
                collection_name=astra_db_empty_store.collection.name,
                astra_db_client=core_astra_db,
            )
        f_rec_warnings = [
            wrn for wrn in rec_warnings if issubclass(wrn.category, DeprecationWarning)
        ]
        assert len(f_rec_warnings) == 1
        assert store_init_core.mget(["key"]) == ["val123"]

    @pytest.mark.skipif(
        os.environ.get("ASTRA_DB_ENVIRONMENT", "prod").upper() != "PROD",
        reason="Can run on Astra DB production environment only",
    )
    async def test_store_coreclients_init_async(
        self,
        core_astra_db: AstraDB,
        astra_db_empty_store_async: AstraDBStore,
    ) -> None:
        """
        A deprecation warning from passing a (core) AstraDB, but it works.
        Async version.
        """
        await astra_db_empty_store_async.amset([("key", "val123")])
        # create an equivalent store with core AstraDB in init
        with pytest.warns(DeprecationWarning) as rec_warnings:
            store_init_core = AstraDBStore(
                collection_name=astra_db_empty_store_async.async_collection.name,
                astra_db_client=core_astra_db,
                setup_mode=SetupMode.ASYNC,
            )
        f_rec_warnings = [
            wrn for wrn in rec_warnings if issubclass(wrn.category, DeprecationWarning)
        ]
        assert len(f_rec_warnings) == 1
        assert await store_init_core.amget(["key"]) == ["val123"]

    @pytest.mark.usefixtures("ephemeral_indexing_collections_cleaner")
    def test_store_indexing_default_sync(
        self,
        astra_db_credentials: AstraDBCredentials,
        astra_db_empty_store: AstraDBStore,
    ) -> None:
        """Test of default-indexing re-instantiation."""
        AstraDBStore(
            collection_name=astra_db_empty_store.collection.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        )

    @pytest.mark.usefixtures("ephemeral_indexing_collections_cleaner")
    async def test_store_indexing_default_async(
        self,
        astra_db_credentials: AstraDBCredentials,
        astra_db_empty_store_async: AstraDBStore,
    ) -> None:
        """Test of default-indexing re-instantiation, async version"""
        await AstraDBStore(
            collection_name=astra_db_empty_store_async.async_collection.name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            setup_mode=SetupMode.ASYNC,
        ).amget(["some_key"])

    @pytest.mark.usefixtures("ephemeral_indexing_collections_cleaner")
    def test_store_indexing_on_legacy_sync(
        self,
        astra_db_credentials: AstraDBCredentials,
        database: Database,
    ) -> None:
        """Test of instantiation against a legacy collection."""
        database.create_collection(
            EPHEMERAL_LEGACY_IDX_NAME,
            indexing=None,
            check_exists=False,
        )
        with pytest.warns(UserWarning) as rec_warnings:
            AstraDBStore(
                collection_name=EPHEMERAL_LEGACY_IDX_NAME,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
            )
            f_rec_warnings = [
                wrn for wrn in rec_warnings if issubclass(wrn.category, UserWarning)
            ]
            assert len(f_rec_warnings) == 1

    @pytest.mark.usefixtures("ephemeral_indexing_collections_cleaner")
    async def test_store_indexing_on_legacy_async(
        self,
        astra_db_credentials: AstraDBCredentials,
        database: Database,
    ) -> None:
        """Test of instantiation against a legacy collection, async version."""
        await database.to_async().create_collection(
            EPHEMERAL_LEGACY_IDX_NAME,
            indexing=None,
            check_exists=False,
        )
        with pytest.warns(UserWarning) as rec_warnings:
            await AstraDBStore(
                collection_name=EPHEMERAL_LEGACY_IDX_NAME,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                setup_mode=SetupMode.ASYNC,
            ).amget(["some_key"])
            f_rec_warnings = [
                wrn for wrn in rec_warnings if issubclass(wrn.category, UserWarning)
            ]
            assert len(f_rec_warnings) == 1

    @pytest.mark.usefixtures("ephemeral_indexing_collections_cleaner")
    def test_store_indexing_on_custom_sync(
        self,
        astra_db_credentials: AstraDBCredentials,
        database: Database,
    ) -> None:
        """Test of instantiation against a legacy collection."""
        database.create_collection(
            EPHEMERAL_CUSTOM_IDX_NAME,
            indexing={"deny": ["useless", "forgettable"]},
            check_exists=False,
        )
        with pytest.raises(
            ValueError, match="is detected as having the following indexing policy"
        ):
            AstraDBStore(
                collection_name=EPHEMERAL_CUSTOM_IDX_NAME,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
            )

    @pytest.mark.usefixtures("ephemeral_indexing_collections_cleaner")
    async def test_store_indexing_on_custom_async(
        self,
        astra_db_credentials: AstraDBCredentials,
        database: Database,
    ) -> None:
        """Test of instantiation against a legacy collection, async version."""
        await database.to_async().create_collection(
            EPHEMERAL_CUSTOM_IDX_NAME,
            indexing={"deny": ["useless", "forgettable"]},
            check_exists=False,
        )
        with pytest.raises(
            ValueError, match="is detected as having the following indexing policy"
        ):
            await AstraDBStore(
                collection_name=EPHEMERAL_CUSTOM_IDX_NAME,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                setup_mode=SetupMode.ASYNC,
            ).amget(["some_key"])
