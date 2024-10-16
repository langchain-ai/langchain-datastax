from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

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


def hv_prefix_matcher_factory(
    expected_name_after_lc: str | None = None,
) -> Callable[[str, str | None, str], bool]:
    """Prepare a customized header matcher function for httpserver.

    The matcher does exact string equality for all headers except User-Agent.
    For User-Agent:

    1. whatever is in the "header" part of `expect_request()` and similar must
    be at the beginning of the intercepted UA string
    2. "langchain/<something" must be the one block immediately after that
    3. if provided, there must be at least another block, whose first half
    (before the "/") is the `expected_name_after_lc` param, if such is provided to
    this factory.

    In other words, if the expect_request has something like
        headers={"User-Agent": ""}
    then this matcher verifies that the UA *starts with* "langchain"
    Otherwise, the string there will come before "langchain".
    Independently of the above, if a "expected_name_after_lc" is given, its
    presence right after the "langchain/<version>" part is validated.

    Note: This contrived implementation is to comply with the header matcher
    signature of pytest_httpsserver.
    """

    def _matcher(hk: str, hv: str | None, ev: str) -> bool:
        if hk.lower() == "user-agent":
            if hv is not None:
                if hv.startswith(ev):
                    # condition 1 is OK. Look at the rest
                    remaining = hv[len(ev) :]
                    blocks = [bl.strip() for bl in remaining.split(" ") if bl.strip()]
                    if any(bl.count("/") != 1 for bl in blocks):
                        return False
                    block_pairs = [bl.split("/") for bl in blocks]
                    if not block_pairs:
                        return False
                    # check 2:
                    if block_pairs[0][0] != "langchain":
                        return False
                    further_block_pairs = block_pairs[1:]
                    # check 3 if an `expected_name_after_lc` is given
                    if expected_name_after_lc:
                        return further_block_pairs[0][0] == expected_name_after_lc
                    # otherwise (if 3 is not required), 1 and 2 are satisfied by now:
                    return True
                return False
            return False
        return hv == ev

    return _matcher


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
                "User-Agent": "",
            },
            header_value_matcher=hv_prefix_matcher_factory("langchain_my_compo"),
        ).respond_with_json({})
        _AstraDBCollectionEnvironment(
            "my_coll",
            api_endpoint=base_endpoint,
            keyspace="ks",
            environment=Environment.OTHER,
            component_name="langchain_my_compo",
        )

        # prefix check, empty ext_callers and no component name
        httpserver.expect_oneshot_request(
            base_path,
            method="POST",
            headers={
                "User-Agent": "",
            },
            header_value_matcher=hv_prefix_matcher_factory(),
        ).respond_with_json({})
        _AstraDBCollectionEnvironment(
            "my_coll",
            api_endpoint=base_endpoint,
            keyspace="ks",
            environment=Environment.OTHER,
        )

        # prefix check, one ext_caller
        httpserver.expect_oneshot_request(
            base_path,
            method="POST",
            headers={
                "User-Agent": "ec0/ev0",
            },
            header_value_matcher=hv_prefix_matcher_factory("langchain_my_compo"),
        ).respond_with_json({})
        _AstraDBCollectionEnvironment(
            "my_coll",
            api_endpoint=base_endpoint,
            keyspace="ks",
            environment=Environment.OTHER,
            ext_callers=[("ec0", "ev0")],
            component_name="langchain_my_compo",
        )

        # prefix check, one ext_caller and no component name
        httpserver.expect_oneshot_request(
            base_path,
            method="POST",
            headers={
                "User-Agent": "ec0/ev0",
            },
            header_value_matcher=hv_prefix_matcher_factory(),
        ).respond_with_json({})
        _AstraDBCollectionEnvironment(
            "my_coll",
            api_endpoint=base_endpoint,
            keyspace="ks",
            environment=Environment.OTHER,
            ext_callers=[("ec0", "ev0")],
        )

        # prefix check, two ext_callers
        httpserver.expect_oneshot_request(
            base_path,
            method="POST",
            headers={
                "User-Agent": "ec0/ev0 ec1/ev1",
            },
            header_value_matcher=hv_prefix_matcher_factory("langchain_my_compo"),
        ).respond_with_json({})
        _AstraDBCollectionEnvironment(
            "my_coll",
            api_endpoint=base_endpoint,
            keyspace="ks",
            environment=Environment.OTHER,
            ext_callers=[("ec0", "ev0"), ("ec1", "ev1")],
            component_name="langchain_my_compo",
        )

        # prefix check, incomplete callers
        httpserver.expect_oneshot_request(
            base_path,
            method="POST",
            headers={
                "User-Agent": "ic0 ic1 ic2",
            },
            header_value_matcher=hv_prefix_matcher_factory("langchain_my_compo"),
        ).respond_with_json({})
        _AstraDBCollectionEnvironment(
            "my_coll",
            api_endpoint=base_endpoint,
            keyspace="ks",
            environment=Environment.OTHER,
            ext_callers=[
                None,
                (None, None),
                ("ic0", None),
                "ic1",
                (None, "zzz"),
                "ic2",
            ],
            component_name="langchain_my_compo",
        )

    def test_callers_component_loader(self, httpserver: HTTPServer) -> None:
        """
        End-to-end testing of callers passed through the components.
        The loader, which does not create a collection, requires separate handling.
        """
        base_endpoint = httpserver.url_for("/")
        base_path = "/v1/ks/my_coll"

        httpserver.expect_oneshot_request(
            base_path,
            method="POST",
            headers={
                "User-Agent": "ec0/ev0",
            },
            header_value_matcher=hv_prefix_matcher_factory(COMPONENT_NAME_LOADER),
        ).respond_with_json({"data": {"nextPageState": None, "documents": [{}]}})

        loader = AstraDBLoader(
            collection_name="my_coll",
            api_endpoint=base_endpoint,
            environment=Environment.OTHER,
            namespace="ks",
            ext_callers=[("ec0", "ev0")],
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

        # through the init flow
        httpserver.expect_oneshot_request(
            base_path,
            method="POST",
            headers={
                "User-Agent": "ec0/ev0",
            },
            header_value_matcher=hv_prefix_matcher_factory(COMPONENT_NAME_VECTORSTORE),
        ).respond_with_json({})

        AstraDBVectorStore(
            collection_name="my_coll",
            api_endpoint=base_endpoint,
            environment=Environment.OTHER,
            namespace="ks",
            embedding=ParserEmbeddings(2),
            ext_callers=[("ec0", "ev0")],
        )

        # autodetect mode independently surveys the collection:
        httpserver.expect_oneshot_request(
            base_path + "/my_coll",
            method="POST",
            headers={
                "User-Agent": "ec0/ev0",
            },
            header_value_matcher=hv_prefix_matcher_factory(COMPONENT_NAME_VECTORSTORE),
        ).respond_with_json({"data": {"nextPageState": None, "documents": []}})
        httpserver.expect_request(
            base_path,
            method="POST",
            headers={
                "User-Agent": "ec0/ev0",
            },
            data='{"findCollections":{"options":{"explain":true}}}',
            header_value_matcher=hv_prefix_matcher_factory(COMPONENT_NAME_VECTORSTORE),
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
            ext_callers=[("ec0", "ev0")],
        )

    def test_callers_component_graphvectorstore(self, httpserver: HTTPServer) -> None:
        """
        End-to-end testing of callers passed through the components.
        The graphvectorstore, which can also do autodetect operations,
        requires separate handling.
        """
        base_endpoint = httpserver.url_for("/")
        base_path = "/v1/ks"

        # through the init flow
        httpserver.expect_oneshot_request(
            base_path,
            method="POST",
            headers={
                "User-Agent": "ec0/ev0",
            },
            header_value_matcher=hv_prefix_matcher_factory(
                COMPONENT_NAME_GRAPHVECTORSTORE
            ),
        ).respond_with_json({})

        # the metadata_search test call
        httpserver.expect_oneshot_request(
            base_path + "/my_graph_coll",
            method="POST",
            headers={
                "User-Agent": "ec0/ev0",
            },
            header_value_matcher=hv_prefix_matcher_factory(
                COMPONENT_NAME_GRAPHVECTORSTORE
            ),
        ).respond_with_json(
            {
                "data": {
                    "nextPageState": None,
                    "documents": [],
                },
            }
        )

        AstraDBGraphVectorStore(
            collection_name="my_graph_coll",
            api_endpoint=base_endpoint,
            environment=Environment.OTHER,
            namespace="ks",
            embedding=ParserEmbeddings(2),
            ext_callers=[("ec0", "ev0")],
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

        httpserver.expect_oneshot_request(
            base_path,
            method="POST",
            headers={
                "User-Agent": "ec0/ev0",
            },
            header_value_matcher=hv_prefix_matcher_factory(component_name),
        ).respond_with_json({})

        component_class(  # type: ignore[operator]
            collection_name="my_coll",
            api_endpoint=base_endpoint,
            environment=Environment.OTHER,
            namespace="ks",
            ext_callers=[("ec0", "ev0")],
            **kwargs,
        )
