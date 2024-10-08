from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from astrapy.constants import Environment

from langchain_astradb import (
    AstraDBByteStore,
    AstraDBCache,
    AstraDBChatMessageHistory,
    AstraDBGraphVectorStore,
    AstraDBLoader,
    AstraDBSemanticCache,
    AstraDBStore,
    AstraDBVectorStore,
)
from langchain_astradb.utils.astradb import (
    COMPONENT_NAME_BYTESTORE,
    COMPONENT_NAME_CACHE,
    COMPONENT_NAME_CHATMESSAGEHISTORY,
    COMPONENT_NAME_GRAPHVECTORSTORE,
    COMPONENT_NAME_LOADER,
    COMPONENT_NAME_SEMANTICCACHE,
    COMPONENT_NAME_STORE,
    COMPONENT_NAME_VECTORSTORE,
    _AstraDBCollectionEnvironment,
)
from tests.conftest import ParserEmbeddings

if TYPE_CHECKING:
    from pytest_httpserver import HTTPServer


def hv_prefix_matcher(hk: str, hv: str | None, ev: str) -> bool:
    """Custom header matcher function for httpserver.

    We require that the UA to start with the provided string,
    and also that "langchain/<v> is found later in the UA.
    """
    if hk.lower() == "user-agent":
        if hv is not None:
            return hv.startswith(ev) and " langchain/" in hv
        return False
    return True


class TestCallers:
    def test_callers_environment(self, httpserver: HTTPServer) -> None:
        """Mechanism for the 'ext_callers' and the 'component_name' params."""
        base_endpoint = httpserver.url_for("/")
        base_path = "/v1/ks"

        # prefix check, empty ext_callers
        httpserver.expect_oneshot_request(
            base_path,
            method="POST",
            headers={
                "User-Agent": "langchain_my_compo/",
            },
            header_value_matcher=hv_prefix_matcher,
        ).respond_with_json({})
        _AstraDBCollectionEnvironment(
            "my_coll",
            api_endpoint=base_endpoint,
            keyspace="ks",
            environment=Environment.OTHER,
            component_name="langchain_my_compo",
        )

        # prefix check, one ext_caller
        httpserver.expect_oneshot_request(
            base_path,
            method="POST",
            headers={
                "User-Agent": "ec0/ev0 langchain_my_compo/",
            },
            header_value_matcher=hv_prefix_matcher,
        ).respond_with_json({})
        _AstraDBCollectionEnvironment(
            "my_coll",
            api_endpoint=base_endpoint,
            keyspace="ks",
            environment=Environment.OTHER,
            ext_callers=[("ec0", "ev0")],
            component_name="langchain_my_compo",
        )

        # prefix check, two ext_callers
        httpserver.expect_oneshot_request(
            base_path,
            method="POST",
            headers={
                "User-Agent": "ec0/ev0 ec1/ev1 langchain_my_compo/",
            },
            header_value_matcher=hv_prefix_matcher,
        ).respond_with_json({})
        _AstraDBCollectionEnvironment(
            "my_coll",
            api_endpoint=base_endpoint,
            keyspace="ks",
            environment=Environment.OTHER,
            ext_callers=[("ec0", "ev0"), ("ec1", "ev1")],
            component_name="langchain_my_compo",
        )

    def test_callers_component_loader(self, httpserver: HTTPServer) -> None:
        """
        End-to-end testing of callers passed through the components.
        The loader, which does not create a collection, requires separate handling.
        """
        base_endpoint = httpserver.url_for("/")
        base_path = "/v1/ks/my_coll"

        # prefix check, empty ext_callers
        httpserver.expect_oneshot_request(
            base_path,
            method="POST",
            headers={
                "User-Agent": f"{COMPONENT_NAME_LOADER}/",
            },
            header_value_matcher=hv_prefix_matcher,
        ).respond_with_json({"data": {"nextPageState": None, "documents": [{}]}})

        loader = AstraDBLoader(
            collection_name="my_coll",
            api_endpoint=base_endpoint,
            environment=Environment.OTHER,
            namespace="ks",
        )
        next(loader.lazy_load())

    def test_callers_component_vectorstore(self, httpserver: HTTPServer) -> None:
        """
        End-to-end testing of callers passed through the components.
        The vectorstore, which can also do autodetect operations,
        requires separate handling.
        """
        base_endpoint = httpserver.url_for("/")
        base_path = "/v1/ks"

        # prefix check, empty ext_callers
        httpserver.expect_oneshot_request(
            base_path,
            method="POST",
            headers={
                "User-Agent": f"{COMPONENT_NAME_VECTORSTORE}/",
            },
            header_value_matcher=hv_prefix_matcher,
        ).respond_with_json({})

        AstraDBVectorStore(
            collection_name="my_coll",
            api_endpoint=base_endpoint,
            environment=Environment.OTHER,
            namespace="ks",
            embedding=ParserEmbeddings(2),
        )

        # autodetect mode independently surveys the collection:
        httpserver.expect_oneshot_request(
            base_path + "/my_coll",
            method="POST",
            headers={
                "User-Agent": f"{COMPONENT_NAME_VECTORSTORE}/",
            },
            header_value_matcher=hv_prefix_matcher,
        ).respond_with_json({"data": {"nextPageState": None, "documents": []}})
        httpserver.expect_request(
            base_path,
            method="POST",
            headers={
                "User-Agent": f"{COMPONENT_NAME_VECTORSTORE}/",
            },
            data='{"findCollections":{"options":{"explain":true}}}',
            header_value_matcher=hv_prefix_matcher,
        ).respond_with_json(
            {
                "status": {
                    "collections": [
                        {"name": "my_coll", "options": {"vector": {"dimension": 2}}}
                    ]
                }
            }
        )
        AstraDBVectorStore(
            collection_name="my_coll",
            api_endpoint=base_endpoint,
            environment=Environment.OTHER,
            namespace="ks",
            embedding=ParserEmbeddings(2),
            autodetect_collection=True,
            content_field="confie",
        )

    @pytest.mark.parametrize(
        ("component_class", "component_name", "kwargs"),
        [
            (AstraDBByteStore, COMPONENT_NAME_BYTESTORE, {}),
            (AstraDBCache, COMPONENT_NAME_CACHE, {}),
            (
                AstraDBChatMessageHistory,
                COMPONENT_NAME_CHATMESSAGEHISTORY,
                {"session_id": "x"},
            ),
            (
                AstraDBGraphVectorStore,
                COMPONENT_NAME_GRAPHVECTORSTORE,
                {"embedding": ParserEmbeddings(2)},
            ),
            (
                AstraDBSemanticCache,
                COMPONENT_NAME_SEMANTICCACHE,
                {"embedding": ParserEmbeddings(2)},
            ),
            (AstraDBStore, COMPONENT_NAME_STORE, {}),
        ],
        ids=[
            "Byte store",
            "Cache",
            "Chat message history",
            "Graph vector store",
            "Semantic cache",
            "Store",
        ],
    )
    def test_callers_component_generic(
        self,
        httpserver: HTTPServer,
        component_class: Any,  # noqa: ANN401
        component_name: str,
        kwargs: dict[str, Any],
    ) -> None:
        """End-to-end testing of callers passed through the components."""
        base_endpoint = httpserver.url_for("/")
        base_path = "/v1/ks"

        # prefix check, empty ext_callers
        httpserver.expect_oneshot_request(
            base_path,
            method="POST",
            headers={
                "User-Agent": f"{component_name}/",
            },
            header_value_matcher=hv_prefix_matcher,
        ).respond_with_json({})

        component_class(  # type: ignore[operator]
            collection_name="my_coll",
            api_endpoint=base_endpoint,
            environment=Environment.OTHER,
            namespace="ks",
            **kwargs,
        )
