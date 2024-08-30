import pytest
from astrapy.db import AstraDB
from astrapy.info import CollectionVectorServiceOptions

from langchain_astradb.utils.astradb import SetupMode
from langchain_astradb.vectorstores import (
    DEFAULT_INDEXING_OPTIONS,
    AstraDBVectorStore,
)
from tests.conftest import SomeEmbeddings

FAKE_TOKEN = "t"  # noqa: S105


class TestAstraDB:
    def test_initialization(self) -> None:
        """Unit test of vector store initialization modes."""
        # Using a 'core' AstraDB class (as opposed to secret-based)
        a_e_string = (
            "https://01234567-89ab-cdef-0123-456789abcdef-us-east1"
            ".apps.astra.datastax.com"
        )
        mock_astra_db = AstraDB(
            token=FAKE_TOKEN,
            api_endpoint=a_e_string,
            namespace="n",
        )
        embedding = SomeEmbeddings(dimension=2)
        with pytest.warns(DeprecationWarning):
            AstraDBVectorStore(
                embedding=embedding,
                collection_name="mock_coll_name",
                astra_db_client=mock_astra_db,
                setup_mode=SetupMode.OFF,
            )

        # With an embedding class
        AstraDBVectorStore(
            collection_name="mock_coll_name",
            token=FAKE_TOKEN,
            api_endpoint=a_e_string,
            namespace="n",
            embedding=embedding,
            setup_mode=SetupMode.OFF,
        )

        # With server-side embeddings ('vectorize')
        vector_options = CollectionVectorServiceOptions(
            provider="test", model_name="test"
        )
        AstraDBVectorStore(
            collection_name="mock_coll_name",
            token=FAKE_TOKEN,
            api_endpoint=a_e_string,
            namespace="n",
            collection_vector_service_options=vector_options,
            setup_mode=SetupMode.OFF,
        )

        # embedding and vectorize => error
        with pytest.raises(
            ValueError,
            match="Embedding cannot be provided for vectorize collections",
        ):
            AstraDBVectorStore(
                embedding=embedding,
                collection_name="mock_coll_name",
                token=FAKE_TOKEN,
                api_endpoint=a_e_string,
                namespace="n",
                collection_vector_service_options=vector_options,
                setup_mode=SetupMode.OFF,
            )

        # no embedding and no vectorize => error
        with pytest.raises(
            ValueError,
            match="Embedding is required for non-vectorize collections",
        ):
            AstraDBVectorStore(
                collection_name="mock_coll_name",
                token=FAKE_TOKEN,
                api_endpoint=a_e_string,
                namespace="n",
                setup_mode=SetupMode.OFF,
            )

    def test_astradb_vectorstore_unit_indexing_normalization(self) -> None:
        """Unit test of the indexing policy normalization"""
        n3_idx = AstraDBVectorStore._normalize_metadata_indexing_policy(
            metadata_indexing_include=None,
            metadata_indexing_exclude=None,
            collection_indexing_policy=None,
        )
        assert n3_idx == DEFAULT_INDEXING_OPTIONS

        al_idx = AstraDBVectorStore._normalize_metadata_indexing_policy(
            metadata_indexing_include=["a1", "a2"],
            metadata_indexing_exclude=None,
            collection_indexing_policy=None,
        )
        assert al_idx == {"allow": ["metadata.a1", "metadata.a2"]}

        dl_idx = AstraDBVectorStore._normalize_metadata_indexing_policy(
            metadata_indexing_include=None,
            metadata_indexing_exclude=["d1", "d2"],
            collection_indexing_policy=None,
        )
        assert dl_idx == {"deny": ["metadata.d1", "metadata.d2"]}

        custom_policy = {
            "deny": ["myfield", "other_field.subfield", "metadata.long_text"]
        }
        cip_idx = AstraDBVectorStore._normalize_metadata_indexing_policy(
            metadata_indexing_include=None,
            metadata_indexing_exclude=None,
            collection_indexing_policy=custom_policy,
        )
        assert cip_idx == custom_policy

        error_msg = (
            "At most one of the parameters `metadata_indexing_include`, "
            "`metadata_indexing_exclude` and `collection_indexing_policy` "
            "can be specified as non null."
        )

        with pytest.raises(ValueError, match=error_msg):
            AstraDBVectorStore._normalize_metadata_indexing_policy(
                metadata_indexing_include=["a"],
                metadata_indexing_exclude=["b"],
                collection_indexing_policy=None,
            )

        with pytest.raises(ValueError, match=error_msg):
            AstraDBVectorStore._normalize_metadata_indexing_policy(
                metadata_indexing_include=["a"],
                metadata_indexing_exclude=None,
                collection_indexing_policy={"a": "z"},
            )

        with pytest.raises(ValueError, match=error_msg):
            AstraDBVectorStore._normalize_metadata_indexing_policy(
                metadata_indexing_include=None,
                metadata_indexing_exclude=["b"],
                collection_indexing_policy={"a": "z"},
            )

        with pytest.raises(ValueError, match=error_msg):
            AstraDBVectorStore._normalize_metadata_indexing_policy(
                metadata_indexing_include=["a"],
                metadata_indexing_exclude=["b"],
                collection_indexing_policy={"a": "z"},
            )
