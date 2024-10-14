"""Test of Upgrading to Astra DB graph vector store class:
`AstraDBGraphVectorStore` from an existing collection used
by the Astra DB vector store class: `AstraDBVectorStore`

Refer to `test_vectorstores.py` for the requirements to run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from astrapy.authentication import StaticTokenProvider
from langchain_core.documents import Document

from langchain_astradb.graph_vectorstores import AstraDBGraphVectorStore
from langchain_astradb.vectorstores import AstraDBVectorStore

from .conftest import (
    astra_db_env_vars_available,
)

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

    from .conftest import AstraDBCredentials


@pytest.fixture
def default_vector_store_d2(
    astra_db_credentials: AstraDBCredentials,
    embedding_d2: Embeddings,
    ephemeral_collection_cleaner_d2: str,
) -> AstraDBVectorStore:
    return AstraDBVectorStore(
        embedding=embedding_d2,
        collection_name=ephemeral_collection_cleaner_d2,
        token=StaticTokenProvider(astra_db_credentials["token"]),
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
    )


@pytest.fixture
def vector_store_d2_with_indexing_allow_list(
    astra_db_credentials: AstraDBCredentials,
    embedding_d2: Embeddings,
    ephemeral_collection_cleaner_d2: str,
) -> AstraDBVectorStore:
    return AstraDBVectorStore(
        embedding=embedding_d2,
        collection_name=ephemeral_collection_cleaner_d2,
        token=StaticTokenProvider(astra_db_credentials["token"]),
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
        # this is the only difference from the `default_vector_store_d2` fixture above
        collection_indexing_policy={"allow": ["test"]},
    )


@pytest.fixture
def vector_store_d2_with_indexing_deny_list(
    astra_db_credentials: AstraDBCredentials,
    embedding_d2: Embeddings,
    ephemeral_collection_cleaner_d2: str,
) -> AstraDBVectorStore:
    return AstraDBVectorStore(
        embedding=embedding_d2,
        collection_name=ephemeral_collection_cleaner_d2,
        token=StaticTokenProvider(astra_db_credentials["token"]),
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
        # this is the only difference from the `default_vector_store_d2` fixture above
        collection_indexing_policy={"deny": ["test"]},
    )


@pytest.mark.skipif(
    not astra_db_env_vars_available(), reason="Missing Astra DB env. vars"
)
class TestUpgradeToGraphVectorStore:
    @pytest.mark.parametrize(
        ("store_name", "indexing_policy", "expect_success"),
        [
            ("default_vector_store_d2", None, True),
            ("vector_store_d2_with_indexing_allow_list", {"allow": ["test"]}, False),
            ("vector_store_d2_with_indexing_allow_list", None, False),
            ("vector_store_d2_with_indexing_deny_list", {"deny": ["test"]}, True),
            ("vector_store_d2_with_indexing_deny_list", None, False),
        ],
        ids=[
            "default_store_upgrade_should_succeed",
            "allow_store_upgrade_with_allow_policy_should_fail",
            "allow_store_upgrade_with_no_policy_should_fail",
            "deny_store_upgrade_with_deny_policy_should_succeed",
            "deny_store_upgrade_with_no_policy_should_fail",
        ],
    )
    def test_upgrade_to_gvs(
        self,
        astra_db_credentials: AstraDBCredentials,
        embedding_d2: Embeddings,
        ephemeral_collection_cleaner_d2: str,
        *,
        store_name: str,
        indexing_policy: dict[str, Any] | None,
        expect_success: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        # Create Vector Store, load a document
        v_store: AstraDBVectorStore = request.getfixturevalue(store_name)
        doc_id = "AL"
        doc_al = Document(id=doc_id, page_content="[-1, 9]", metadata={"label": "AL"})
        v_store.add_documents([doc_al])

        # Try to create a GRAPH Vector Store using the existing collection from above
        try:
            gv_store = AstraDBGraphVectorStore(
                embedding=embedding_d2,
                collection_name=ephemeral_collection_cleaner_d2,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                collection_indexing_policy=indexing_policy,
            )

            if not expect_success:
                pytest.fail("Expected ValueError but none was raised")

        except ValueError as value_error:
            if expect_success:
                pytest.fail(f"Unexpected ValueError raised: {value_error}")
            else:
                assert (  # noqa: PT017
                    str(value_error)
                    == "The collection configuration is incompatible with vector graph store. Please create a new collection."  # noqa: E501
                )

        if expect_success:
            doc = gv_store.get_by_document_id(document_id=doc_id)
            assert doc is not None
            assert doc.page_content == doc_al.page_content
