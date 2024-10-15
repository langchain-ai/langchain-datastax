"""
Integration tests on Astra DB.

Required to run this test:
    - a recent `astrapy` Python package available
    - an Astra DB instance;
    - the two environment variables set:
        export ASTRA_DB_API_ENDPOINT="https://<DB-ID>-us-east1.apps.astra.datastax.com"
        export ASTRA_DB_APPLICATION_TOKEN="AstraCS:........."
    - optionally this as well (otherwise defaults are used):
        export ASTRA_DB_KEYSPACE="my_keyspace"
    - optionally (if not on prod)
        export ASTRA_DB_ENVIRONMENT="dev"  # or similar
    - an openai key name on KMS for SHARED_SECRET vectorize mode, associated to the DB:
        export SHARED_SECRET_NAME_OPENAI="the_api_key_name_in_Astra_KMS"
    - an OpenAI key for the vectorize test (in HEADER mode):
        export OPENAI_API_KEY="..."

Please refer to testing.env.sample.
"""

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
from tests.conftest import IdentityLLM, ParserEmbeddings

if TYPE_CHECKING:
    from astrapy import Collection, Database
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import LLM


# Getting the absolute path of the current file's directory
ABS_PATH = (Path(__file__)).parent

# Getting the absolute path of the project's root directory
PROJECT_DIR = Path(ABS_PATH).parent.parent

# Long-lasting collection names (default-indexed for vectorstores)
COLLECTION_NAME_D2 = "lc_test_d2_euclidean"
COLLECTION_NAME_VZ = "lc_test_vz_euclidean"
# (All-indexed) for general-purpose and autodetect
COLLECTION_NAME_IDXALL_D2 = "lc_test_d2_idxall_euclidean"
COLLECTION_NAME_IDXALL_VZ = "lc_test_vz_idxall_euclidean"
# non-vector all-indexed collection
COLLECTION_NAME_IDXALL = "lc_test_idxall"
# non-vector store-like-indexed collection
COLLECTION_NAME_IDXID = "lc_test_idxid"
# Function-lived collection names:
# (all-indexed) for autodetect:
EPHEMERAL_COLLECTION_NAME_IDXALL_D2 = "lc_test_d2_idxall_euclidean"
# of generic use for vectorstores
EPHEMERAL_COLLECTION_NAME_D2 = "lc_test_d2_cosine_short"
EPHEMERAL_COLLECTION_NAME_VZ = "lc_test_vz_cosine_short"
# for KMS (aka shared_secret) vectorize setup (vectorstores)
EPHEMERAL_COLLECTION_NAME_VZ_KMS = "lc_test_vz_kms_short"
# indexing-related collection names (function-lived) (vectorstores)
EPHEMERAL_ALLOW_IDX_NAME_D2 = "lc_test_allow_idx_d2_short"
EPHEMERAL_CUSTOM_IDX_NAME_D2 = "lc_test_custom_idx_d2_short"
EPHEMERAL_DEFAULT_IDX_NAME_D2 = "lc_test_default_idx_d2_short"
EPHEMERAL_DENY_IDX_NAME_D2 = "lc_test_deny_idx_d2_short"
EPHEMERAL_LEGACY_IDX_NAME_D2 = "lc_test_legacy_idx_d2_short"
# indexing-related collection names (function-lived) (storage)
EPHEMERAL_CUSTOM_IDX_NAME = "lc_test_custom_idx_short"
EPHEMERAL_LEGACY_IDX_NAME = "lc_test_legacy_idx_short"

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


def astra_db_env_vars_available() -> bool:
    return all(
        [
            "ASTRA_DB_APPLICATION_TOKEN" in os.environ,
            "ASTRA_DB_API_ENDPOINT" in os.environ,
        ]
    )


_load_env()


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
    environment: str


@pytest.fixture(scope="session")
def openai_api_key() -> str:
    return os.environ["OPENAI_API_KEY"]


@pytest.fixture(scope="session")
def embedding_d2() -> Embeddings:
    return ParserEmbeddings(dimension=2)


@pytest.fixture
def test_llm() -> LLM:
    return IdentityLLM()


@pytest.fixture(scope="session")
def astra_db_credentials() -> AstraDBCredentials:
    return {
        "token": os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        "api_endpoint": os.environ["ASTRA_DB_API_ENDPOINT"],
        "namespace": os.environ.get("ASTRA_DB_KEYSPACE"),
        "environment": os.environ.get("ASTRA_DB_ENVIRONMENT", "prod"),
    }


@pytest.fixture(scope="session")
def is_astra_db(astra_db_credentials: AstraDBCredentials) -> bool:
    return astra_db_credentials["environment"].lower() in {
        "prod",
        "test",
        "dev",
    }


@pytest.fixture(scope="session")
def database(
    *,
    is_astra_db: bool,
    astra_db_credentials: AstraDBCredentials,
) -> Database:
    client = DataAPIClient(environment=astra_db_credentials["environment"])
    db = client.get_database(
        astra_db_credentials["api_endpoint"],
        token=StaticTokenProvider(astra_db_credentials["token"]),
        keyspace=astra_db_credentials["namespace"],
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


@pytest.fixture(scope="module")
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
    empty_collection_d2: Collection,
    astra_db_credentials: AstraDBCredentials,
    embedding_d2: Embeddings,
) -> AstraDBVectorStore:
    """A fresh vector store on a d=2(Euclidean) collection."""
    return AstraDBVectorStore(
        embedding=embedding_d2,
        collection_name=empty_collection_d2.name,
        token=StaticTokenProvider(astra_db_credentials["token"]),
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
        setup_mode=SetupMode.OFF,
    )


@pytest.fixture
def vector_store_d2_stringtoken(
    empty_collection_d2: Collection,
    astra_db_credentials: AstraDBCredentials,
    embedding_d2: Embeddings,
) -> AstraDBVectorStore:
    """
    A fresh vector store on a d=2(Euclidean) collection,
    but initialized with a token string instead of a TokenProvider.
    """
    return AstraDBVectorStore(
        embedding=embedding_d2,
        collection_name=empty_collection_d2.name,
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


@pytest.fixture(scope="module")
def collection_idxall(
    database: Database,
) -> Iterable[Collection]:
    """
    A general-purpose collection for per-test reuse.
    This one has default indexing (i.e. all fields are covered).
    """
    collection = database.create_collection(
        COLLECTION_NAME_IDXALL,
        check_exists=False,
    )
    yield collection

    collection.drop()


@pytest.fixture
def empty_collection_idxall(
    collection_idxall: Collection,
) -> Collection:
    """
    A per-test-function empty collection.
    This one has default indexing (i.e. all fields are covered).
    """
    collection_idxall.delete_many({})
    return collection_idxall


@pytest.fixture(scope="module")
def collection_idxid(
    database: Database,
) -> Iterable[Collection]:
    """
    A general-purpose collection for per-test reuse.
    This one has id-only indexing (i.e. for Storage classes).
    """
    collection = database.create_collection(
        COLLECTION_NAME_IDXID,
        indexing={"allow": ["_id"]},
        check_exists=False,
    )
    yield collection

    collection.drop()


@pytest.fixture(scope="module")
def collection_idxall_d2(
    database: Database,
) -> Iterable[Collection]:
    """
    A general-purpose D=2(Euclidean) collection for per-test reuse.
    This one has default indexing (i.e. all fields are covered).
    """
    collection = database.create_collection(
        COLLECTION_NAME_IDXALL_D2,
        dimension=2,
        check_exists=False,
        metric="euclidean",
    )
    yield collection

    collection.drop()


@pytest.fixture
def empty_collection_idxall_d2(
    collection_idxall_d2: Collection,
) -> Collection:
    """
    A per-test-function empty d=2(Euclidean) collection.
    This one has default indexing (i.e. all fields are covered).
    """
    collection_idxall_d2.delete_many({})
    return collection_idxall_d2


@pytest.fixture
def vector_store_idxall_d2(
    empty_collection_idxall_d2: Collection,
    astra_db_credentials: AstraDBCredentials,
    embedding_d2: Embeddings,
) -> AstraDBVectorStore:
    """A fresh vector store on a d=2(Euclidean) collection."""
    return AstraDBVectorStore(
        embedding=embedding_d2,
        collection_name=empty_collection_idxall_d2.name,
        token=StaticTokenProvider(astra_db_credentials["token"]),
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
        collection_indexing_policy={"allow": ["*"]},
        setup_mode=SetupMode.OFF,
    )


@pytest.fixture
def ephemeral_collection_cleaner_idxall_d2(
    database: Database,
) -> Iterable[str]:
    """
    A nominal fixture to ensure the ephemeral collection is deleted
    after the test function has finished.
    """

    yield EPHEMERAL_COLLECTION_NAME_IDXALL_D2

    if EPHEMERAL_COLLECTION_NAME_IDXALL_D2 in database.list_collection_names():
        database.drop_collection(EPHEMERAL_COLLECTION_NAME_IDXALL_D2)


@pytest.fixture(scope="module")
def collection_vz(
    openai_api_key: str,
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
        embedding_api_key=openai_api_key,
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
    astra_db_credentials: AstraDBCredentials,
    openai_api_key: str,
    empty_collection_vz: Collection,
) -> AstraDBVectorStore:
    """A fresh vector store on a $vectorize collection."""
    return AstraDBVectorStore(
        collection_name=empty_collection_vz.name,
        token=StaticTokenProvider(astra_db_credentials["token"]),
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
        setup_mode=SetupMode.OFF,
        collection_vector_service_options=OPENAI_VECTORIZE_OPTIONS_HEADER,
        collection_embedding_api_key=openai_api_key,
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


@pytest.fixture(scope="module")
def collection_idxall_vz(
    openai_api_key: str,
    database: Database,
) -> Iterable[Collection]:
    """
    A general-purpose $vectorize collection for per-test reuse.
    This one has default indexing (i.e. all fields are covered).
    """
    collection = database.create_collection(
        COLLECTION_NAME_IDXALL_VZ,
        dimension=16,
        check_exists=False,
        metric="euclidean",
        service=OPENAI_VECTORIZE_OPTIONS_HEADER,
        embedding_api_key=openai_api_key,
    )
    yield collection

    collection.drop()


@pytest.fixture
def empty_collection_idxall_vz(
    collection_idxall_vz: Collection,
) -> Collection:
    """
    A per-test-function empty $vecorize collection.
    This one has default indexing (i.e. all fields are covered).
    """
    collection_idxall_vz.delete_many({})
    return collection_idxall_vz


@pytest.fixture
def vector_store_idxall_vz(
    openai_api_key: str,
    empty_collection_idxall_vz: Collection,
    astra_db_credentials: AstraDBCredentials,
) -> AstraDBVectorStore:
    """A fresh vector store on a d=2(Euclidean) collection."""
    return AstraDBVectorStore(
        collection_name=empty_collection_idxall_vz.name,
        token=StaticTokenProvider(astra_db_credentials["token"]),
        api_endpoint=astra_db_credentials["api_endpoint"],
        namespace=astra_db_credentials["namespace"],
        environment=astra_db_credentials["environment"],
        collection_indexing_policy={"allow": ["*"]},
        setup_mode=SetupMode.OFF,
        collection_vector_service_options=OPENAI_VECTORIZE_OPTIONS_HEADER,
        collection_embedding_api_key=openai_api_key,
    )


@pytest.fixture
def ephemeral_indexing_collections_cleaner(
    database: Database,
) -> Iterable[list[str]]:
    """
    A nominal fixture to ensure the ephemeral collections for indexing testing
    are deleted after the test function has finished.
    """

    collection_names = [
        EPHEMERAL_ALLOW_IDX_NAME_D2,
        EPHEMERAL_CUSTOM_IDX_NAME_D2,
        EPHEMERAL_DEFAULT_IDX_NAME_D2,
        EPHEMERAL_DENY_IDX_NAME_D2,
        EPHEMERAL_LEGACY_IDX_NAME_D2,
        EPHEMERAL_CUSTOM_IDX_NAME,
        EPHEMERAL_LEGACY_IDX_NAME,
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
