"""Implement integration tests for AstraDB storage."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from langchain_astradb.storage import AstraDBByteStore, AstraDBStore
from langchain_astradb.utils.astradb import SetupMode

from .conftest import _has_env_vars

if TYPE_CHECKING:
    from astrapy import Database
    from astrapy.db import AstraDB


def init_store(
    astra_db_credentials: dict[str, str | None],
    collection_name: str,
) -> AstraDBStore:
    store = AstraDBStore(
        collection_name=collection_name,
        token=astra_db_credentials["token"],
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
    )
    store.mset([("key1", [0.1, 0.2]), ("key2", "value2")])
    return store


def init_bytestore(
    astra_db_credentials: dict[str, str | None],
    collection_name: str,
) -> AstraDBByteStore:
    store = AstraDBByteStore(
        collection_name=collection_name,
        token=astra_db_credentials["token"],
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
    )
    store.mset([("key1", b"value1"), ("key2", b"value2")])
    return store


async def init_async_store(
    astra_db_credentials: dict[str, str | None], collection_name: str
) -> AstraDBStore:
    store = AstraDBStore(
        collection_name=collection_name,
        token=astra_db_credentials["token"],
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
        setup_mode=SetupMode.ASYNC,
    )
    await store.amset([("key1", [0.1, 0.2]), ("key2", "value2")])
    return store


@pytest.mark.skipif(not _has_env_vars(), reason="Missing Astra DB env. vars")
class TestAstraDBStore:
    def test_mget(
        self,
        astra_db_credentials: dict[str, str | None],
    ) -> None:
        """Test AstraDBStore mget method."""
        collection_name = "lc_test_store_mget"
        try:
            store = init_store(astra_db_credentials, collection_name)
            assert store.mget(["key1", "key2"]) == [[0.1, 0.2], "value2"]
        finally:
            store.astra_env.database.drop_collection(collection_name)

    async def test_amget(
        self,
        astra_db_credentials: dict[str, str | None],
    ) -> None:
        """Test AstraDBStore amget method."""
        collection_name = "lc_test_store_mget"
        try:
            store = await init_async_store(astra_db_credentials, collection_name)
            assert await store.amget(["key1", "key2"]) == [[0.1, 0.2], "value2"]
        finally:
            await store.astra_env.async_database.drop_collection(collection_name)

    def test_mset(
        self,
        astra_db_credentials: dict[str, str | None],
    ) -> None:
        """Test that multiple keys can be set with AstraDBStore."""
        collection_name = "lc_test_store_mset"
        try:
            store = init_store(astra_db_credentials, collection_name)
            result = store.collection.find_one({"_id": "key1"})
            assert (result or {})["value"] == [0.1, 0.2]
            result = store.collection.find_one({"_id": "key2"})
            assert (result or {})["value"] == "value2"
        finally:
            store.astra_env.database.drop_collection(collection_name)

    async def test_amset(
        self,
        astra_db_credentials: dict[str, str | None],
    ) -> None:
        """Test that multiple keys can be set with AstraDBStore."""
        collection_name = "lc_test_store_mset"
        try:
            store = await init_async_store(astra_db_credentials, collection_name)
            result = await store.async_collection.find_one({"_id": "key1"})
            assert (result or {})["value"] == [0.1, 0.2]
            result = await store.async_collection.find_one({"_id": "key2"})
            assert (result or {})["value"] == "value2"
        finally:
            await store.astra_env.async_database.drop_collection(collection_name)

    def test_store_massive_mset_with_replace(
        self,
        astra_db_credentials: dict[str, str | None],
    ) -> None:
        """Testing the insert-many-and-replace-some patterns thoroughly."""
        full_size = 300
        first_group_size = 150
        second_group_slicer = [30, 100, 2]
        max_values_in_in = 100
        collection_name = "lc_test_store_massive_mset"

        ids_and_texts = [
            (
                f"doc_{idx}",
                f"document number {idx}",
            )
            for idx in range(full_size)
        ]
        try:
            store = AstraDBStore(
                collection_name=collection_name,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
            )

            # massive insertion on empty (zip and rezip for uniformity with later)
            group0_ids, group0_texts = list(zip(*ids_and_texts[0:first_group_size]))
            store.mset(list(zip(group0_ids, group0_texts)))

            # massive insertion with many overwrites scattered through
            # (we change the text to later check on DB for successful update)
            _s, _e, _st = second_group_slicer
            group1_ids, group1_texts_pre = list(
                zip(
                    *(
                        ids_and_texts[_s:_e:_st]
                        + ids_and_texts[first_group_size:full_size]
                    )
                )
            )
            group1_texts = [txt.upper() for txt in group1_texts_pre]
            store.mset(list(zip(group1_ids, group1_texts)))

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
                for val in store.mget(
                    all_ids[chunk_start : chunk_start + max_values_in_in]
                )
            ]
            for val, doc_id in zip(all_vals, all_ids):
                assert val == expected_text_by_id[doc_id]
        finally:
            store.astra_env.database.drop_collection(collection_name)

    async def test_store_massive_amset_with_replace(
        self,
        astra_db_credentials: dict[str, str | None],
    ) -> None:
        """Testing the insert-many-and-replace-some patterns thoroughly."""
        full_size = 300
        first_group_size = 150
        second_group_slicer = [30, 100, 2]
        max_values_in_in = 100
        collection_name = "lc_test_store_massive_amset"

        ids_and_texts = [
            (
                f"doc_{idx}",
                f"document number {idx}",
            )
            for idx in range(full_size)
        ]

        try:
            store = AstraDBStore(
                collection_name=collection_name,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
            )

            # massive insertion on empty (zip and rezip for uniformity with later)
            group0_ids, group0_texts = list(zip(*ids_and_texts[0:first_group_size]))
            await store.amset(list(zip(group0_ids, group0_texts)))

            # massive insertion with many overwrites scattered through
            # (we change the text to later check on DB for successful update)
            _s, _e, _st = second_group_slicer
            group1_ids, group1_texts_pre = list(
                zip(
                    *(
                        ids_and_texts[_s:_e:_st]
                        + ids_and_texts[first_group_size:full_size]
                    )
                )
            )
            group1_texts = [txt.upper() for txt in group1_texts_pre]
            await store.amset(list(zip(group1_ids, group1_texts)))

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
                for val in await store.amget(
                    all_ids[chunk_start : chunk_start + max_values_in_in]
                )
            ]
            for val, doc_id in zip(all_vals, all_ids):
                assert val == expected_text_by_id[doc_id]
        finally:
            store.astra_env.database.drop_collection(collection_name)

    def test_mdelete(
        self,
        astra_db_credentials: dict[str, str | None],
    ) -> None:
        """Test that deletion works as expected."""
        collection_name = "lc_test_store_mdelete"
        try:
            store = init_store(astra_db_credentials, collection_name)
            store.mdelete(["key1", "key2"])
            result = store.mget(["key1", "key2"])
            assert result == [None, None]
        finally:
            store.astra_env.database.drop_collection(collection_name)

    async def test_amdelete(
        self,
        astra_db_credentials: dict[str, str | None],
    ) -> None:
        """Test that deletion works as expected."""
        collection_name = "lc_test_store_mdelete"
        try:
            store = await init_async_store(astra_db_credentials, collection_name)
            await store.amdelete(["key1", "key2"])
            result = await store.amget(["key1", "key2"])
            assert result == [None, None]
        finally:
            await store.astra_env.async_database.drop_collection(collection_name)

    def test_yield_keys(
        self,
        astra_db_credentials: dict[str, str | None],
    ) -> None:
        collection_name = "lc_test_store_yield_keys"
        try:
            store = init_store(astra_db_credentials, collection_name)
            assert set(store.yield_keys()) == {"key1", "key2"}
            assert set(store.yield_keys(prefix="key")) == {"key1", "key2"}
            assert set(store.yield_keys(prefix="lang")) == set()
        finally:
            store.astra_env.database.drop_collection(collection_name)

    async def test_ayield_keys(
        self,
        astra_db_credentials: dict[str, str | None],
    ) -> None:
        collection_name = "lc_test_store_yield_keys"
        try:
            store = await init_async_store(astra_db_credentials, collection_name)
            assert {key async for key in store.ayield_keys()} == {"key1", "key2"}
            assert {key async for key in store.ayield_keys(prefix="key")} == {
                "key1",
                "key2",
            }
            assert {key async for key in store.ayield_keys(prefix="lang")} == set()
        finally:
            await store.astra_env.async_database.drop_collection(collection_name)

    def test_bytestore_mget(
        self,
        astra_db_credentials: dict[str, str | None],
    ) -> None:
        """Test AstraDBByteStore mget method."""
        collection_name = "lc_test_bytestore_mget"
        try:
            store = init_bytestore(astra_db_credentials, collection_name)
            assert store.mget(["key1", "key2"]) == [b"value1", b"value2"]
        finally:
            store.astra_env.database.drop_collection(collection_name)

    def test_bytestore_mset(
        self,
        astra_db_credentials: dict[str, str | None],
    ) -> None:
        """Test that multiple keys can be set with AstraDBByteStore."""
        collection_name = "lc_test_bytestore_mset"
        try:
            store = init_bytestore(astra_db_credentials, collection_name)
            result = store.collection.find_one({"_id": "key1"})
            assert (result or {})["value"] == "dmFsdWUx"
            result = store.collection.find_one({"_id": "key2"})
            assert (result or {})["value"] == "dmFsdWUy"
        finally:
            store.astra_env.database.drop_collection(collection_name)

    def test_indexing_detection(
        self,
        astra_db_credentials: dict[str, str | None],
        database: Database,
    ) -> None:
        """Test the behaviour against preexisting legacy collections."""
        database.create_collection("lc_test_legacy_store")
        database.create_collection(
            "lc_test_custom_store", indexing={"allow": ["my_field"]}
        )
        AstraDBStore(
            collection_name="lc_test_regular_store",
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        )

        # repeated instantiation must work
        AstraDBStore(
            collection_name="lc_test_regular_store",
            token=astra_db_credentials["token"],
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        )
        # on a legacy collection must just give a warning
        with pytest.warns(UserWarning) as rec_warnings:
            AstraDBStore(
                collection_name="lc_test_legacy_store",
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
            )
            assert len(rec_warnings) == 1
        # on a custom collection must error
        with pytest.raises(
            ValueError, match="is detected as having the following indexing policy"
        ):
            AstraDBStore(
                collection_name="lc_test_custom_store",
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
            )

        database.drop_collection("lc_test_legacy_store")
        database.drop_collection("lc_test_custom_store")
        database.drop_collection("lc_test_regular_store")

    @pytest.mark.skipif(
        os.environ.get("ASTRA_DB_ENVIRONMENT", "prod").upper() != "PROD",
        reason="Can run on Astra DB prod only",
    )
    def test_store_coreclients_init_sync(
        self,
        astra_db_credentials: dict[str, str | None],
        core_astra_db: AstraDB,
    ) -> None:
        """A deprecation warning from passing a (core) AstraDB, but it works."""
        collection_name = "lc_test_bytestore_coreclsync"
        try:
            store_init_ok = AstraDBStore(
                collection_name=collection_name,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
            )
            store_init_ok.mset([("key", "val123")])
            # create an equivalent store with core AstraDB in init
            with pytest.warns(DeprecationWarning) as rec_warnings:
                store_init_core = AstraDBStore(
                    collection_name=collection_name,
                    astra_db_client=core_astra_db,
                )
            assert len(rec_warnings) == 1
            assert store_init_core.mget(["key"]) == ["val123"]
        finally:
            store_init_ok.astra_env.database.drop_collection(collection_name)

    @pytest.mark.skipif(
        os.environ.get("ASTRA_DB_ENVIRONMENT", "prod").upper() != "PROD",
        reason="Can run on Astra DB prod only",
    )
    async def test_store_coreclients_init_async(
        self,
        astra_db_credentials: dict[str, str | None],
        core_astra_db: AstraDB,
    ) -> None:
        """A deprecation warning from passing a (core) AstraDB, but it works."""
        collection_name = "lc_test_bytestore_coreclasync"
        try:
            store_init_ok = AstraDBStore(
                collection_name=collection_name,
                token=astra_db_credentials["token"],
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                setup_mode=SetupMode.ASYNC,
            )
            await store_init_ok.amset([("key", "val123")])
            # create an equivalent store with core AstraDB in init
            with pytest.warns(DeprecationWarning) as rec_warnings:
                store_init_core = AstraDBStore(
                    collection_name=collection_name,
                    astra_db_client=core_astra_db,
                    setup_mode=SetupMode.ASYNC,
                )
            assert len(rec_warnings) == 1
            assert await store_init_core.amget(["key"]) == ["val123"]
        finally:
            await store_init_ok.astra_env.async_database.drop_collection(
                collection_name
            )
