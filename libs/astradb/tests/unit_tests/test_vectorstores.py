import pytest
from astrapy.info import VectorServiceOptions

from langchain_astradb.utils.astradb import SetupMode
from langchain_astradb.utils.vector_store_codecs import (
    _DefaultVSDocumentCodec,
    _FlatVSDocumentCodec,
)
from langchain_astradb.vectorstores import AstraDBVectorStore
from tests.conftest import ParserEmbeddings

FAKE_TOKEN = "t"  # noqa: S105


class TestAstraDB:
    def test_initialization(self) -> None:
        """Unit test of vector store initialization modes."""
        a_e_string = (
            "https://01234567-89ab-cdef-0123-456789abcdef-us-east1"
            ".apps.astra.datastax.com"
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
        vector_options = VectorServiceOptions(provider="test", model_name="test")
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
        """Unit test of the indexing policy normalization.

        We use just a couple of codecs to check the idx policy fallbacks.
        """

        the_f_codec = _FlatVSDocumentCodec(
            content_field="content_x",
            ignore_invalid_documents=False,
        )
        the_f_default_policy = the_f_codec.default_collection_indexing_policy
        the_d_codec = _DefaultVSDocumentCodec(
            content_field="content_y",
            ignore_invalid_documents=False,
        )

        # default (non-flat): hardcoding expected indexing from including
        al_d_idx = AstraDBVectorStore._normalize_metadata_indexing_policy(
            metadata_indexing_include=["a1", "a2"],
            metadata_indexing_exclude=None,
            collection_indexing_policy=None,
            document_codec=the_d_codec,
        )
        assert al_d_idx == {"allow": ["metadata.a1", "metadata.a2"]}

        # default (non-flat): hardcoding expected indexing from excluding
        dl_d_idx = AstraDBVectorStore._normalize_metadata_indexing_policy(
            metadata_indexing_include=None,
            metadata_indexing_exclude=["d1", "d2"],
            collection_indexing_policy=None,
            document_codec=the_d_codec,
        )
        assert dl_d_idx == {"deny": ["metadata.d1", "metadata.d2"]}

        n3_idx = AstraDBVectorStore._normalize_metadata_indexing_policy(
            metadata_indexing_include=None,
            metadata_indexing_exclude=None,
            collection_indexing_policy=None,
            document_codec=the_f_codec,
        )
        assert n3_idx == the_f_default_policy

        al_idx = AstraDBVectorStore._normalize_metadata_indexing_policy(
            metadata_indexing_include=["a1", "a2"],
            metadata_indexing_exclude=None,
            collection_indexing_policy=None,
            document_codec=the_f_codec,
        )
        assert al_idx == {"allow": ["a1", "a2"]}

        dl_idx = AstraDBVectorStore._normalize_metadata_indexing_policy(
            metadata_indexing_include=None,
            metadata_indexing_exclude=["d1", "d2"],
            collection_indexing_policy=None,
            document_codec=the_f_codec,
        )
        assert dl_idx == {"deny": ["d1", "d2"]}

        custom_policy = {
            "deny": ["myfield", "other_field.subfield", "metadata.long_text"]
        }
        cip_idx = AstraDBVectorStore._normalize_metadata_indexing_policy(
            metadata_indexing_include=None,
            metadata_indexing_exclude=None,
            collection_indexing_policy=custom_policy,
            document_codec=the_f_codec,
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
                document_codec=the_f_codec,
            )

        with pytest.raises(ValueError, match=error_msg):
            AstraDBVectorStore._normalize_metadata_indexing_policy(
                metadata_indexing_include=["a"],
                metadata_indexing_exclude=None,
                collection_indexing_policy={"a": "z"},
                document_codec=the_f_codec,
            )

        with pytest.raises(ValueError, match=error_msg):
            AstraDBVectorStore._normalize_metadata_indexing_policy(
                metadata_indexing_include=None,
                metadata_indexing_exclude=["b"],
                collection_indexing_policy={"a": "z"},
                document_codec=the_f_codec,
            )

        with pytest.raises(ValueError, match=error_msg):
            AstraDBVectorStore._normalize_metadata_indexing_policy(
                metadata_indexing_include=["a"],
                metadata_indexing_exclude=["b"],
                collection_indexing_policy={"a": "z"},
                document_codec=the_f_codec,
            )
