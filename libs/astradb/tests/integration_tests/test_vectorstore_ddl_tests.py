"""DDL-heavy parts of the tests for the Astra DB vector store class `AstraDBVectorStore`

Refer to `test_vectorstores.py` for the requirements to run.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pytest
from astrapy.authentication import EmbeddingAPIKeyHeaderProvider, StaticTokenProvider
from astrapy.exceptions import InsertManyException

from langchain_astradb.utils.astradb import SetupMode
from langchain_astradb.vectorstores import AstraDBVectorStore

from .conftest import (
    EPHEMERAL_CUSTOM_IDX_NAME_D2,
    EPHEMERAL_DEFAULT_IDX_NAME_D2,
    EPHEMERAL_LEGACY_IDX_NAME_D2,
    INCOMPATIBLE_INDEXING_MSG,
    LEGACY_INDEXING_MSG,
    OPENAI_VECTORIZE_OPTIONS_HEADER,
    OPENAI_VECTORIZE_OPTIONS_KMS,
    astra_db_env_vars_available,
)

if TYPE_CHECKING:
    from astrapy import Database
    from langchain_core.embeddings import Embeddings

    from .conftest import AstraDBCredentials


@pytest.mark.skipif(
    not astra_db_env_vars_available(), reason="Missing Astra DB env. vars"
)
class TestAstraDBVectorStoreDDLs:
    def test_astradb_vectorstore_create_delete_sync(
        self,
        astra_db_credentials: AstraDBCredentials,
        database: Database,
        embedding_d2: Embeddings,
        ephemeral_collection_cleaner_d2: str,
    ) -> None:
        """Create and delete."""
        v_store = AstraDBVectorStore(
            embedding=embedding_d2,
            collection_name=ephemeral_collection_cleaner_d2,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            metric="cosine",
        )
        v_store.add_texts(["[1,2]"])
        v_store.delete_collection()
        assert ephemeral_collection_cleaner_d2 not in database.list_collection_names()

    def test_astradb_vectorstore_create_delete_vectorize_sync(
        self,
        astra_db_credentials: AstraDBCredentials,
        openai_api_key: str,
        database: Database,
        ephemeral_collection_cleaner_vz: str,
    ) -> None:
        """Create and delete with vectorize option."""
        v_store = AstraDBVectorStore(
            collection_name=ephemeral_collection_cleaner_vz,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            metric="cosine",
            collection_vector_service_options=OPENAI_VECTORIZE_OPTIONS_HEADER,
            collection_embedding_api_key=openai_api_key,
        )
        v_store.add_texts(["This is text"])
        v_store.delete_collection()
        assert ephemeral_collection_cleaner_vz not in database.list_collection_names()

    async def test_astradb_vectorstore_create_delete_async(
        self,
        astra_db_credentials: AstraDBCredentials,
        database: Database,
        embedding_d2: Embeddings,
        ephemeral_collection_cleaner_d2: str,
    ) -> None:
        """Create and delete, async."""
        v_store = AstraDBVectorStore(
            embedding=embedding_d2,
            collection_name=ephemeral_collection_cleaner_d2,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            metric="cosine",
        )
        await v_store.aadd_texts(["[1,2]"])
        await v_store.adelete_collection()
        assert ephemeral_collection_cleaner_d2 not in database.list_collection_names()

    async def test_astradb_vectorstore_create_delete_vectorize_async(
        self,
        astra_db_credentials: AstraDBCredentials,
        openai_api_key: str,
        database: Database,
        ephemeral_collection_cleaner_vz: str,
    ) -> None:
        """Create and delete with vectorize option, async."""
        v_store = AstraDBVectorStore(
            collection_name=ephemeral_collection_cleaner_vz,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            metric="cosine",
            collection_vector_service_options=OPENAI_VECTORIZE_OPTIONS_HEADER,
            collection_embedding_api_key=openai_api_key,
        )
        await v_store.aadd_texts(["[1,2]"])
        await v_store.adelete_collection()
        assert ephemeral_collection_cleaner_vz not in database.list_collection_names()

    def test_astradb_vectorstore_pre_delete_collection_sync(
        self,
        astra_db_credentials: AstraDBCredentials,
        embedding_d2: Embeddings,
        ephemeral_collection_cleaner_d2: str,
    ) -> None:
        """Use of the pre_delete_collection flag."""
        v_store = AstraDBVectorStore(
            embedding=embedding_d2,
            collection_name=ephemeral_collection_cleaner_d2,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            metric="cosine",
        )
        v_store.add_texts(texts=["[1,2]"])
        res1 = v_store.similarity_search("[-1,-1]", k=5)
        assert len(res1) == 1
        v_store = AstraDBVectorStore(
            embedding=embedding_d2,
            collection_name=ephemeral_collection_cleaner_d2,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            metric="cosine",
            pre_delete_collection=True,
        )
        res1 = v_store.similarity_search("[-1,-1]", k=5)
        assert len(res1) == 0

    async def test_astradb_vectorstore_pre_delete_collection_async(
        self,
        astra_db_credentials: AstraDBCredentials,
        embedding_d2: Embeddings,
        ephemeral_collection_cleaner_d2: str,
    ) -> None:
        v_store = AstraDBVectorStore(
            embedding=embedding_d2,
            collection_name=ephemeral_collection_cleaner_d2,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            setup_mode=SetupMode.ASYNC,
            metric="cosine",
        )
        await v_store.aadd_texts(texts=["[1,2]"])
        res1 = await v_store.asimilarity_search("[-1,-1]", k=5)
        assert len(res1) == 1
        v_store = AstraDBVectorStore(
            embedding=embedding_d2,
            collection_name=ephemeral_collection_cleaner_d2,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            setup_mode=SetupMode.ASYNC,
            metric="cosine",
            pre_delete_collection=True,
        )
        res1 = await v_store.asimilarity_search("[-1,-1]", k=5)
        assert len(res1) == 0

    @pytest.mark.usefixtures("ephemeral_indexing_collections_cleaner")
    def test_astradb_vectorstore_indexing_legacy_sync(
        self,
        astra_db_credentials: AstraDBCredentials,
        database: Database,
        embedding_d2: Embeddings,
    ) -> None:
        """
        Test of the vector store behaviour for various indexing settings,
        with an existing 'legacy' collection (i.e. unspecified indexing policy).
        """
        database.create_collection(
            EPHEMERAL_LEGACY_IDX_NAME_D2,
            dimension=2,
            check_exists=False,
        )

        with pytest.raises(
            ValueError,
            match=LEGACY_INDEXING_MSG,
        ):
            AstraDBVectorStore(
                collection_name=EPHEMERAL_LEGACY_IDX_NAME_D2,
                embedding=embedding_d2,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                metadata_indexing_exclude={"long_summary", "the_divine_comedy"},
            )

        # one case should result in just a warning:
        with pytest.warns(UserWarning) as rec_warnings:
            AstraDBVectorStore(
                collection_name=EPHEMERAL_LEGACY_IDX_NAME_D2,
                embedding=embedding_d2,
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
    def test_astradb_vectorstore_indexing_default_sync(
        self,
        astra_db_credentials: AstraDBCredentials,
        embedding_d2: Embeddings,
    ) -> None:
        """
        Test of the vector store behaviour for various indexing settings,
        with an existing 'default' collection.
        """
        AstraDBVectorStore(
            collection_name=EPHEMERAL_DEFAULT_IDX_NAME_D2,
            embedding=embedding_d2,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            AstraDBVectorStore(
                collection_name=EPHEMERAL_DEFAULT_IDX_NAME_D2,
                embedding=embedding_d2,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
            )

        # unacceptable for a pre-existing (default-indexing) collection:
        with pytest.raises(ValueError, match=INCOMPATIBLE_INDEXING_MSG):
            AstraDBVectorStore(
                collection_name=EPHEMERAL_DEFAULT_IDX_NAME_D2,
                embedding=embedding_d2,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                metadata_indexing_exclude={"long_summary", "the_divine_comedy"},
            )

    @pytest.mark.usefixtures("ephemeral_indexing_collections_cleaner")
    def test_astradb_vectorstore_indexing_custom_sync(
        self,
        astra_db_credentials: AstraDBCredentials,
        embedding_d2: Embeddings,
    ) -> None:
        """
        Test of the vector store behaviour for various indexing settings,
        with an existing custom-indexing collection.
        """
        AstraDBVectorStore(
            collection_name=EPHEMERAL_CUSTOM_IDX_NAME_D2,
            embedding=embedding_d2,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            metadata_indexing_exclude={"long_summary", "the_divine_comedy"},
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            AstraDBVectorStore(
                collection_name=EPHEMERAL_CUSTOM_IDX_NAME_D2,
                embedding=embedding_d2,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                metadata_indexing_exclude={"long_summary", "the_divine_comedy"},
            )

        with pytest.raises(ValueError, match=INCOMPATIBLE_INDEXING_MSG):
            AstraDBVectorStore(
                collection_name=EPHEMERAL_CUSTOM_IDX_NAME_D2,
                embedding=embedding_d2,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                metadata_indexing_exclude={"changed_fields"},
            )

        with pytest.raises(ValueError, match=INCOMPATIBLE_INDEXING_MSG):
            AstraDBVectorStore(
                collection_name=EPHEMERAL_CUSTOM_IDX_NAME_D2,
                embedding=embedding_d2,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
            )

    @pytest.mark.usefixtures("ephemeral_indexing_collections_cleaner")
    async def test_astradb_vectorstore_indexing_legacy_async(
        self,
        astra_db_credentials: AstraDBCredentials,
        database: Database,
        embedding_d2: Embeddings,
    ) -> None:
        """
        Test of the vector store behaviour for various indexing settings,
        with an existing 'legacy' collection (i.e. unspecified indexing policy).
        """
        database.create_collection(
            EPHEMERAL_LEGACY_IDX_NAME_D2,
            dimension=2,
            check_exists=False,
        )

        with pytest.raises(
            ValueError,
            match=LEGACY_INDEXING_MSG,
        ):
            await AstraDBVectorStore(
                collection_name=EPHEMERAL_LEGACY_IDX_NAME_D2,
                embedding=embedding_d2,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                setup_mode=SetupMode.ASYNC,
                metadata_indexing_exclude={"long_summary", "the_divine_comedy"},
            ).aadd_texts(["[4,13]"])

        # one case should result in just a warning:
        with pytest.warns(UserWarning) as rec_warnings:
            await AstraDBVectorStore(
                collection_name=EPHEMERAL_LEGACY_IDX_NAME_D2,
                embedding=embedding_d2,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                setup_mode=SetupMode.ASYNC,
            ).aadd_texts(["[4,13]"])
            f_rec_warnings = [
                wrn for wrn in rec_warnings if issubclass(wrn.category, UserWarning)
            ]
            assert len(f_rec_warnings) == 1

    @pytest.mark.usefixtures("ephemeral_indexing_collections_cleaner")
    async def test_astradb_vectorstore_indexing_default_async(
        self,
        astra_db_credentials: AstraDBCredentials,
        embedding_d2: Embeddings,
    ) -> None:
        """
        Test of the vector store behaviour for various indexing settings,
        with an existing 'default' collection.
        """
        await AstraDBVectorStore(
            collection_name=EPHEMERAL_DEFAULT_IDX_NAME_D2,
            embedding=embedding_d2,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            setup_mode=SetupMode.ASYNC,
        ).aadd_texts(["[4,13]"])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            await AstraDBVectorStore(
                collection_name=EPHEMERAL_DEFAULT_IDX_NAME_D2,
                embedding=embedding_d2,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                setup_mode=SetupMode.ASYNC,
            ).aadd_texts(["[4,13]"])

        # unacceptable for a pre-existing (default-indexing) collection:
        with pytest.raises(ValueError, match=INCOMPATIBLE_INDEXING_MSG):
            await AstraDBVectorStore(
                collection_name=EPHEMERAL_DEFAULT_IDX_NAME_D2,
                embedding=embedding_d2,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                setup_mode=SetupMode.ASYNC,
                metadata_indexing_exclude={"long_summary", "the_divine_comedy"},
            ).aadd_texts(["[4,13]"])

    @pytest.mark.usefixtures("ephemeral_indexing_collections_cleaner")
    async def test_astradb_vectorstore_indexing_custom_async(
        self,
        astra_db_credentials: AstraDBCredentials,
        embedding_d2: Embeddings,
    ) -> None:
        """
        Test of the vector store behaviour for various indexing settings,
        with an existing custom-indexing collection.
        """
        await AstraDBVectorStore(
            collection_name=EPHEMERAL_CUSTOM_IDX_NAME_D2,
            embedding=embedding_d2,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            setup_mode=SetupMode.ASYNC,
            metadata_indexing_exclude={"long_summary", "the_divine_comedy"},
        ).aadd_texts(["[4,13]"])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            await AstraDBVectorStore(
                collection_name=EPHEMERAL_CUSTOM_IDX_NAME_D2,
                embedding=embedding_d2,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                setup_mode=SetupMode.ASYNC,
                metadata_indexing_exclude={"long_summary", "the_divine_comedy"},
            ).aadd_texts(["[4,13]"])

        with pytest.raises(ValueError, match=INCOMPATIBLE_INDEXING_MSG):
            await AstraDBVectorStore(
                collection_name=EPHEMERAL_CUSTOM_IDX_NAME_D2,
                embedding=embedding_d2,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                setup_mode=SetupMode.ASYNC,
                metadata_indexing_exclude={"changed_fields"},
            ).aadd_texts(["[4,13]"])

        with pytest.raises(ValueError, match=INCOMPATIBLE_INDEXING_MSG):
            await AstraDBVectorStore(
                collection_name=EPHEMERAL_CUSTOM_IDX_NAME_D2,
                embedding=embedding_d2,
                token=StaticTokenProvider(astra_db_credentials["token"]),
                api_endpoint=astra_db_credentials["api_endpoint"],
                namespace=astra_db_credentials["namespace"],
                environment=astra_db_credentials["environment"],
                setup_mode=SetupMode.ASYNC,
            ).aadd_texts(["[4,13]"])

    @pytest.mark.skipif(
        OPENAI_VECTORIZE_OPTIONS_KMS is None,
        reason="A KMS ('shared secret') API Key name is required",
    )
    def test_astradb_vectorstore_vectorize_headers_precedence_stringheader(
        self,
        astra_db_credentials: AstraDBCredentials,
        ephemeral_collection_cleaner_vz_kms: str,
    ) -> None:
        """
        Test that header, if passed, takes precedence over vectorize setting.
        To do so, a faulty header is passed, expecting the call to fail.
        """
        v_store = AstraDBVectorStore(
            collection_name=ephemeral_collection_cleaner_vz_kms,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            collection_vector_service_options=OPENAI_VECTORIZE_OPTIONS_KMS,
            collection_embedding_api_key="verywrong",
        )
        # More specific messages are provider-specific, such as OpenAI returning:
        # "... Incorrect API key provided: verywrong ..."
        with pytest.raises(InsertManyException, match="Embedding Provider returned"):
            v_store.add_texts(["Failing"])

    @pytest.mark.skipif(
        OPENAI_VECTORIZE_OPTIONS_KMS is None,
        reason="A KMS ('shared secret') API Key name is required",
    )
    def test_astradb_vectorstore_vectorize_headers_precedence_headerprovider(
        self,
        astra_db_credentials: AstraDBCredentials,
        ephemeral_collection_cleaner_vz_kms: str,
    ) -> None:
        """
        Test that header, if passed, takes precedence over vectorize setting.
        To do so, a faulty header is passed, expecting the call to fail.
        This version passes the header through an EmbeddingHeaderProvider
        """
        v_store = AstraDBVectorStore(
            collection_name=ephemeral_collection_cleaner_vz_kms,
            token=StaticTokenProvider(astra_db_credentials["token"]),
            api_endpoint=astra_db_credentials["api_endpoint"],
            namespace=astra_db_credentials["namespace"],
            environment=astra_db_credentials["environment"],
            collection_vector_service_options=OPENAI_VECTORIZE_OPTIONS_KMS,
            collection_embedding_api_key=EmbeddingAPIKeyHeaderProvider("verywrong"),
        )
        # More specific messages are provider-specific, such as OpenAI returning:
        # "... Incorrect API key provided: verywrong ..."
        with pytest.raises(InsertManyException, match="Embedding Provider returned"):
            v_store.add_texts(["Failing"])
