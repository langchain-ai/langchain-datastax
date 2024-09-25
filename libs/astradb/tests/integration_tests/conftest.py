from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, TypedDict

import pytest
from astrapy import DataAPIClient
from astrapy.authentication import StaticTokenProvider
from astrapy.db import AstraDB
from astrapy.info import CollectionVectorServiceOptions

from langchain_astradb.utils.astradb import SetupMode
from langchain_astradb.vectorstores import DEFAULT_INDEXING_OPTIONS, AstraDBVectorStore
from tests.conftest import ParserEmbeddings

if TYPE_CHECKING:
    from astrapy import Collection, Database
    from langchain_core.embeddings import Embeddings


# Getting the absolute path of the current file's directory
ABS_PATH = (Path(__file__)).parent

# Getting the absolute path of the project's root directory
PROJECT_DIR = Path(ABS_PATH).parent.parent

# Long-lasting collection names
COLLECTION_NAME_D2 = "lc_test_d2_euclidean"
COLLECTION_NAME_VZ = "lc_test_vz_euclidean"
# Function-lived collection names:
# of generic use
EPHEMERAL_COLLECTION_NAME_D2 = "lc_test_d2_cosine_short"
EPHEMERAL_COLLECTION_NAME_VZ = "lc_test_vz_cosine_short"
# for KMS (aka shared_secret) vectorize setup
EPHEMERAL_COLLECTION_NAME_VZ_KMS = "lc_test_vz_kms_short"
# indexing-related collection names (function-lived)
EPHEMERAL_CUSTOM_IDX_NAME_D2 = "lc_test_custom_idx_d2_short"
EPHEMERAL_DEFAULT_IDX_NAME_D2 = "lc_test_default_idx_d2_short"
EPHEMERAL_LEGACY_IDX_NAME_D2 = "lc_test_legacy_idx_d2_short"
# autodetect assets
CUSTOM_CONTENT_KEY = "xcontent"
LONG_TEXT = "This is the textual content field in the doc."
# vectorstore-related utilities/constants
INCOMPATIBLE_INDEXING_MSG = "is detected as having the following indexing policy"
LEGACY_INDEXING_MSG = "is detected as having indexing turned on for all fields"
# similarity threshold definitions
EUCLIDEAN_MIN_SIM_UNIT_VECTORS = 0.2
MATCH_EPSILON = 0.0001


# Loading the .env file if it exists
def _load_env() -> None:
    dotenv_path = Path(PROJECT_DIR) / "tests" / "integration_tests" / ".env"
    if Path(dotenv_path).exists():
        from dotenv import load_dotenv

        load_dotenv(dotenv_path)


def _has_env_vars() -> bool:
    return all(
        [
            "ASTRA_DB_APPLICATION_TOKEN" in os.environ,
            "ASTRA_DB_API_ENDPOINT" in os.environ,
        ]
    )


_load_env()


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_VECTORIZE_OPTIONS_HEADER = CollectionVectorServiceOptions(
    provider="openai",
    model_name="text-embedding-3-small",
)

OPENAI_SHARED_SECRET_KEY_NAME = os.environ.get("SHARED_SECRET_NAME_OPENAI")
OPENAI_VECTORIZE_OPTIONS_KMS: CollectionVectorServiceOptions | None
if OPENAI_SHARED_SECRET_KEY_NAME:
    OPENAI_VECTORIZE_OPTIONS_KMS = CollectionVectorServiceOptions(
        provider="openai",
        model_name="text-embedding-3-small",
        authentication={
            "providerKey": OPENAI_SHARED_SECRET_KEY_NAME,
        },
    )
else:
    OPENAI_VECTORIZE_OPTIONS_KMS = None


class AstraDBCredentials(TypedDict):
    token: str
    api_endpoint: str
    namespace: str | None
    environment: str | None


@pytest.fixture(scope="session")
def embedding_d2() -> Embeddings:
    return ParserEmbeddings(dimension=2)


@pytest.fixture(scope="session")
def astra_db_credentials() -> AstraDBCredentials:
    return {
        "token": os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        "api_endpoint": os.environ["ASTRA_DB_API_ENDPOINT"],
        "namespace": os.environ.get("ASTRA_DB_KEYSPACE"),
        "environment": os.environ.get("ASTRA_DB_ENVIRONMENT"),
    }


@pytest.fixture(scope="session")
def is_astra_db(astra_db_credentials: AstraDBCredentials) -> bool:
    environment = astra_db_credentials["environment"]
    return environment is not None and environment.lower() not in {
        "prod",
        "test",
        "dev",
    }


@pytest.fixture(scope="session")
def database(
    is_astra_db: bool,
    astra_db_credentials: AstraDBCredentials,
) -> Database:
    client = DataAPIClient(environment=astra_db_credentials["environment"])
    db = client.get_database(
        astra_db_credentials["api_endpoint"],
        token=StaticTokenProvider(astra_db_credentials["token"]),
        namespace=astra_db_credentials["namespace"],
    )
    if not is_astra_db:
        if astra_db_credentials["namespace"] is None:
            msg = "Cannot test on non-Astra without a namespace set."
            raise ValueError(msg)
        db.get_database_admin().create_namespace(astra_db_credentials["namespace"])

    return db


@pytest.fixture(scope="session")
def core_astra_db(astra_db_credentials: AstraDBCredentials) -> AstraDB:
    """An instance of the 'core' (pre-1.0, legacy) astrapy database."""
    return AstraDB(
        token=astra_db_credentials["token"],
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
    )


@pytest.fixture(scope="session")
def collection_d2(
    database: Database,
) -> Iterable[Collection]:
    """A general-purpose D=2(Euclidean) collection for per-test reuse."""
    collection = database.create_collection(
        COLLECTION_NAME_D2,
        dimension=2,
        check_exists=False,
        indexing=DEFAULT_INDEXING_OPTIONS,
        metric="euclidean",
    )
    yield collection

    collection.drop()


@pytest.fixture
def empty_collection_d2(
    collection_d2: Collection,
) -> Collection:
    """A per-test-function empty d=2(Euclidean) collection."""
    collection_d2.delete_many({})
    return collection_d2


@pytest.fixture
def vector_store_d2(
    empty_collection_d2: Collection,  # noqa: ARG001
    astra_db_credentials: AstraDBCredentials,
    embedding_d2: Embeddings,
) -> AstraDBVectorStore:
    """A fresh vector store on a d=2(Euclidean) collection."""
    return AstraDBVectorStore(
        embedding=embedding_d2,
        collection_name=COLLECTION_NAME_D2,
        token=StaticTokenProvider(astra_db_credentials["token"]),
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
        setup_mode=SetupMode.OFF,
    )


@pytest.fixture
def vector_store_d2_stringtoken(
    empty_collection_d2: Collection,  # noqa: ARG001
    astra_db_credentials: AstraDBCredentials,
    embedding_d2: Embeddings,
) -> AstraDBVectorStore:
    """
    A fresh vector store on a d=2(Euclidean) collection,
    but initialized with a token string instead of a TokenProvider.
    """
    return AstraDBVectorStore(
        embedding=embedding_d2,
        collection_name=COLLECTION_NAME_D2,
        token=astra_db_credentials["token"],
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
        setup_mode=SetupMode.OFF,
    )


@pytest.fixture
def ephemeral_collection_cleaner_d2(
    database: Database,
) -> Iterable[str]:
    """
    A nominal fixture to ensure the ephemeral collection is deleted
    after the test function has finished.
    """

    yield EPHEMERAL_COLLECTION_NAME_D2

    if EPHEMERAL_COLLECTION_NAME_D2 in database.list_collection_names():
        database.drop_collection(EPHEMERAL_COLLECTION_NAME_D2)


@pytest.fixture(scope="session")
def collection_vz(
    database: Database,
) -> Iterable[Collection]:
    """A general-purpose $vectorize collection for per-test reuse."""
    collection = database.create_collection(
        COLLECTION_NAME_VZ,
        dimension=16,
        check_exists=False,
        indexing=DEFAULT_INDEXING_OPTIONS,
        metric="euclidean",
        service=OPENAI_VECTORIZE_OPTIONS_HEADER,
    )
    yield collection

    collection.drop()


@pytest.fixture
def empty_collection_vz(
    collection_vz: Collection,
) -> Collection:
    """A per-test-function empty $vecorize collection."""
    collection_vz.delete_many({})
    return collection_vz


@pytest.fixture
def vector_store_vz(
    empty_collection_vz: Collection,  # noqa: ARG001
    astra_db_credentials: AstraDBCredentials,
) -> AstraDBVectorStore:
    """A fresh vector store on a $vectorize collection."""
    return AstraDBVectorStore(
        collection_name=COLLECTION_NAME_VZ,
        token=StaticTokenProvider(astra_db_credentials["token"]),
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
        setup_mode=SetupMode.OFF,
        collection_vector_service_options=OPENAI_VECTORIZE_OPTIONS_HEADER,
        collection_embedding_api_key=OPENAI_API_KEY,
    )


@pytest.fixture
def ephemeral_collection_cleaner_vz(
    database: Database,
) -> Iterable[str]:
    """
    A nominal fixture to ensure the ephemeral vectorize collection is deleted
    after the test function has finished.
    """

    yield EPHEMERAL_COLLECTION_NAME_VZ

    if EPHEMERAL_COLLECTION_NAME_VZ in database.list_collection_names():
        database.drop_collection(EPHEMERAL_COLLECTION_NAME_VZ)


@pytest.fixture
def ephemeral_indexing_collections_cleaner(
    database: Database,
) -> Iterable[list[str]]:
    """
    A nominal fixture to ensure the ephemeral collections for indexing testing
    are deleted after the test function has finished.
    """

    collection_names = [
        EPHEMERAL_CUSTOM_IDX_NAME_D2,
        EPHEMERAL_DEFAULT_IDX_NAME_D2,
        EPHEMERAL_LEGACY_IDX_NAME_D2,
    ]
    yield collection_names

    for collection_name in collection_names:
        if collection_name in database.list_collection_names():
            database.drop_collection(collection_name)


@pytest.fixture
def ephemeral_collection_cleaner_vz_kms(
    database: Database,
) -> Iterable[str]:
    """
    A nominal fixture to ensure the ephemeral vectorize collection with KMS
    is deleted after the test function has finished.
    """

    yield EPHEMERAL_COLLECTION_NAME_VZ_KMS

    if EPHEMERAL_COLLECTION_NAME_VZ_KMS in database.list_collection_names():
        database.drop_collection(EPHEMERAL_COLLECTION_NAME_VZ_KMS)
