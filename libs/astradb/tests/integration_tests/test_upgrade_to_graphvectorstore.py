"""Test of Upgrading to Astra DB graph vector store class:
`AstraDBGraphVectorStore` from an existing collection used
by the Astra DB vector store class: `AstraDBVectorStore`

Refer to `test_vectorstores.py` for the requirements to run.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest
from astrapy.authentication import StaticTokenProvider
from langchain_core.documents import Document

from langchain_astradb.graph_vectorstores import AstraDBGraphVectorStore
from langchain_astradb.utils.astradb import SetupMode
from langchain_astradb.vectorstores import AstraDBVectorStore

from .conftest import (
    EPHEMERAL_ALLOW_IDX_NAME_D2,
    EPHEMERAL_DEFAULT_IDX_NAME_D2,
    EPHEMERAL_DENY_IDX_NAME_D2,
    astra_db_env_vars_available,
)

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

    from .conftest import AstraDBCredentials


def _vs_indexing_policy(collection_name: str) -> dict[str, Any] | None:
    if collection_name == EPHEMERAL_ALLOW_IDX_NAME_D2:
        return {"allow": ["test"]}
    if collection_name == EPHEMERAL_DEFAULT_IDX_NAME_D2:
        return None
    if collection_name == EPHEMERAL_DENY_IDX_NAME_D2:
        return {"deny": ["test"]}
    msg = f"Unknown collection_name: {collection_name} in _vs_indexing_policy()"
    raise ValueError(msg)


@pytest.mark.skipif(
    not astra_db_env_vars_available(), reason="Missing Astra DB env. vars"
)
@pytest.mark.skipif(
    "LANGCHAIN_TEST_ASTRADBGRAPHVECTORSTORE" not in os.environ,
    reason="AstraDBGraphVectorStore tests omitted by default",
)
class TestUpgradeToGraphVectorStore:
    @pytest.mark.usefixtures("ephemeral_indexing_collections_cleaner")
    @pytest.mark.parametrize(
        ("collection_name", "gvs_setup_mode", "gvs_indexing_policy"),
        [
            (EPHEMERAL_DEFAULT_IDX_NAME_D2, SetupMode.SYNC, None),
            (EPHEMERAL_DENY_IDX_NAME_D2, SetupMode.SYNC, {"deny": ["test"]}),
            (EPHEMERAL_DEFAULT_IDX_NAME_D2, SetupMode.OFF, None),
            (EPHEMERAL_DENY_IDX_NAME_D2, SetupMode.OFF, {"deny": ["test"]}),
            # for this one, even though the passed policy doesn't
            # match the policy used to create the collection,
            # there is no error since the SetupMode is OFF and
            # and no attempt is made to re-create the collection.
            (EPHEMERAL_DENY_IDX_NAME_D2, SetupMode.OFF, None),
        ],
        ids=[
            "default_upgrade_no_policy_sync",
            "deny_list_upgrade_same_policy_sync",
            "default_upgrade_no_policy_off",
            "deny_list_upgrade_same_policy_off",
            "deny_list_upgrade_change_policy_off",
        ],
    )
    def test_upgrade_to_gvs_success_sync(
        self,
        astra_db_credentials: AstraDBCredentials,
        embedding_d2: Embeddings,
        *,
        gvs_setup_mode: SetupMode,
        collection_name: str,
        gvs_indexing_policy: dict[str, Any] | None,
    ) -> None:
        # Create vector store using SetupMode.SYNC
        v_store = AstraDBVectorStore(
            embedding=embedding_d2,
            collection_name=collection_name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            collection_indexing_policy=_vs_indexing_policy(
                collection_name=collection_name
            ),
            setup_mode=SetupMode.SYNC,
        )

        # load a document to the vector store
        doc_id = "AL"
        doc_al = Document(id=doc_id, page_content="[-1, 9]", metadata={"label": "AL"})
        v_store.add_documents([doc_al])

        # get the document from the vector store
        v_doc = v_store.get_by_document_id(document_id=doc_id)
        assert v_doc is not None
        assert v_doc.page_content == doc_al.page_content

        # Create a GRAPH Vector Store using the existing collection from above
        # with setup_mode=gvs_setup_mode and indexing_policy=gvs_indexing_policy
        gv_store = AstraDBGraphVectorStore(
            embedding=embedding_d2,
            collection_name=collection_name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            collection_indexing_policy=gvs_indexing_policy,
            setup_mode=gvs_setup_mode,
        )

        # get the document from the GRAPH vector store
        gv_doc = gv_store.get_by_document_id(document_id=doc_id)
        assert gv_doc is not None
        assert gv_doc.page_content == doc_al.page_content

    @pytest.mark.usefixtures("ephemeral_indexing_collections_cleaner")
    @pytest.mark.parametrize(
        ("collection_name", "gvs_setup_mode", "gvs_indexing_policy"),
        [
            (EPHEMERAL_DEFAULT_IDX_NAME_D2, SetupMode.ASYNC, None),
            (EPHEMERAL_DENY_IDX_NAME_D2, SetupMode.ASYNC, {"deny": ["test"]}),
        ],
        ids=[
            "default_upgrade_no_policy_async",
            "deny_list_upgrade_same_policy_async",
        ],
    )
    async def test_upgrade_to_gvs_success_async(
        self,
        astra_db_credentials: AstraDBCredentials,
        embedding_d2: Embeddings,
        *,
        gvs_setup_mode: SetupMode,
        collection_name: str,
        gvs_indexing_policy: dict[str, Any] | None,
    ) -> None:
        # Create vector store using SetupMode.ASYNC
        v_store = AstraDBVectorStore(
            embedding=embedding_d2,
            collection_name=collection_name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            collection_indexing_policy=_vs_indexing_policy(
                collection_name=collection_name
            ),
            setup_mode=SetupMode.ASYNC,
        )

        # load a document to the vector store
        doc_id = "AL"
        doc_al = Document(id=doc_id, page_content="[-1, 9]", metadata={"label": "AL"})
        await v_store.aadd_documents([doc_al])

        # get the document from the vector store
        v_doc = await v_store.aget_by_document_id(document_id=doc_id)
        assert v_doc is not None
        assert v_doc.page_content == doc_al.page_content

        # Create a GRAPH Vector Store using the existing collection from above
        # with setup_mode=gvs_setup_mode and indexing_policy=gvs_indexing_policy
        gv_store = AstraDBGraphVectorStore(
            embedding=embedding_d2,
            collection_name=collection_name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            collection_indexing_policy=gvs_indexing_policy,
            setup_mode=gvs_setup_mode,
        )

        # get the document from the GRAPH vector store
        gv_doc = await gv_store.aget_by_document_id(document_id=doc_id)
        assert gv_doc is not None
        assert gv_doc.page_content == doc_al.page_content

    @pytest.mark.usefixtures("ephemeral_indexing_collections_cleaner")
    @pytest.mark.parametrize(
        ("collection_name", "gvs_setup_mode", "gvs_indexing_policy", "error_match"),
        [
            (
                EPHEMERAL_ALLOW_IDX_NAME_D2,
                SetupMode.SYNC,
                {"allow": ["test"]},
                "incompatible with vector graph",
            ),
            (
                EPHEMERAL_ALLOW_IDX_NAME_D2,
                SetupMode.SYNC,
                None,
                "incompatible with vector graph",
            ),
            (
                EPHEMERAL_DENY_IDX_NAME_D2,
                SetupMode.SYNC,
                None,
                "incompatible with vector graph",
            ),
            (
                EPHEMERAL_ALLOW_IDX_NAME_D2,
                SetupMode.OFF,
                {"allow": ["test"]},
                "not indexed",
            ),
            (EPHEMERAL_ALLOW_IDX_NAME_D2, SetupMode.OFF, None, "not indexed"),
        ],
        ids=[
            "allow_list_upgrade_same_policy_sync",
            "allow_list_upgrade_change_policy_sync",
            "deny_list_upgrade_change_policy_sync",
            "allow_list_upgrade_same_policy_off",
            "allow_list_upgrade_change_policy_off",
        ],
    )
    def test_upgrade_to_gvs_failure_sync(
        self,
        astra_db_credentials: AstraDBCredentials,
        embedding_d2: Embeddings,
        *,
        gvs_setup_mode: SetupMode,
        collection_name: str,
        gvs_indexing_policy: dict[str, Any] | None,
        error_match: str,
    ) -> None:
        # Create vector store using SetupMode.SYNC
        v_store = AstraDBVectorStore(
            embedding=embedding_d2,
            collection_name=collection_name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            collection_indexing_policy=_vs_indexing_policy(
                collection_name=collection_name
            ),
            setup_mode=SetupMode.SYNC,
        )

        # load a document to the vector store
        doc_id = "AL"
        doc_al = Document(id=doc_id, page_content="[-1, 9]", metadata={"label": "AL"})
        v_store.add_documents([doc_al])

        # get the document from the vector store
        v_doc = v_store.get_by_document_id(document_id=doc_id)
        assert v_doc is not None
        assert v_doc.page_content == doc_al.page_content
        with pytest.raises(Exception, match=error_match):
            # Create a GRAPH Vector Store using the existing collection from above
            # with setup_mode=gvs_setup_mode and indexing_policy=gvs_indexing_policy
            _ = AstraDBGraphVectorStore(
                embedding=embedding_d2,
                collection_name=collection_name,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                collection_indexing_policy=gvs_indexing_policy,
                setup_mode=gvs_setup_mode,
            )

    @pytest.mark.usefixtures("ephemeral_indexing_collections_cleaner")
    @pytest.mark.parametrize(
        ("collection_name", "gvs_setup_mode", "gvs_indexing_policy"),
        [
            (EPHEMERAL_ALLOW_IDX_NAME_D2, SetupMode.ASYNC, {"allow": ["test"]}),
            (EPHEMERAL_ALLOW_IDX_NAME_D2, SetupMode.ASYNC, None),
            (EPHEMERAL_DENY_IDX_NAME_D2, SetupMode.ASYNC, None),
        ],
        ids=[
            "allow_list_upgrade_same_policy_async",
            "allow_list_upgrade_change_policy_async",
            "deny_list_upgrade_change_policy_async",
        ],
    )
    async def test_upgrade_to_gvs_failure_async(
        self,
        astra_db_credentials: AstraDBCredentials,
        embedding_d2: Embeddings,
        *,
        gvs_setup_mode: SetupMode,
        collection_name: str,
        gvs_indexing_policy: dict[str, Any] | None,
    ) -> None:
        # Create vector store using SetupMode.ASYNC
        v_store = AstraDBVectorStore(
            embedding=embedding_d2,
            collection_name=collection_name,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            collection_indexing_policy=_vs_indexing_policy(
                collection_name=collection_name
            ),
            setup_mode=SetupMode.ASYNC,
        )

        # load a document to the vector store
        doc_id = "AL"
        doc_al = Document(id=doc_id, page_content="[-1, 9]", metadata={"label": "AL"})
        await v_store.aadd_documents([doc_al])

        # get the document from the vector store
        v_doc = await v_store.aget_by_document_id(document_id=doc_id)
        assert v_doc is not None
        assert v_doc.page_content == doc_al.page_content

        expected_msg = (
            "The collection configuration is incompatible with vector graph "
            "store. Please create a new collection and make sure the metadata "
            "path is not excluded by indexing."
        )
        with pytest.raises(ValueError, match=expected_msg):
            # Create a GRAPH Vector Store using the existing collection from above
            # with setup_mode=gvs_setup_mode and indexing_policy=gvs_indexing_policy
            _ = AstraDBGraphVectorStore(
                embedding=embedding_d2,
                collection_name=collection_name,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                collection_indexing_policy=gvs_indexing_policy,
                setup_mode=gvs_setup_mode,
            )
