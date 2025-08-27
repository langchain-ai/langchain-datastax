from __future__ import annotations

import json
from typing import Any

import pytest

from langchain_astradb.utils.vector_store_autodetect import (
    _detect_document_content_field,
    _detect_document_flatness,
    _detect_documents_content_field,
    _detect_documents_flatness,
    _detect_documents_have_lexical,
)

FLAT_DOCUMENT = {"$vector": [0], "metadata": "m", "_id": "a", "x": "x"}
DEEP_DOCUMENT = {"$vector": [0], "metadata": {}, "_id": "a", "x": "x"}
UNKNOWN_FLATNESS_DOCUMENT = {"metadata": {}, "_id": "a", "$vectorize": "a", "x": "x"}
DOCUMENT_WITH_CF_X = {"x": "LL", "y": "s", "_id": "a"}
DOCUMENT_WITH_CF_Y = {"x": 1234, "y": "s", "_id": "a"}
DOCUMENT_WITH_UNKNOWN_CF = {"x": 1234, "y": 987, "_id": "a"}

# Test-case assets:
DOC_FLATNESS_PAIRS = [
    (UNKNOWN_FLATNESS_DOCUMENT, None),
    ({"$vector": [0], "metadata": {}, "_id": "a"}, None),
    (
        {
            "$vector": [0],
            "metadata": {},
            "_id": "a",
            "$vectorize": "a",
            "x": "x",
        },
        False,
    ),
    (DEEP_DOCUMENT, False),
    ({"$vector": [0], "metadata": "m", "_id": "a"}, True),
    (FLAT_DOCUMENT, True),
    ({"$vector": [0], "metadata": "m", "_id": "a", "x": 9}, True),
    (
        {
            "$vector": [0],
            "metadata": "m",
            "_id": "a",
            "$vectorize": "a",
            "x": "x",
        },
        True,
    ),
    (
        {
            "$vector": [0],
            "metadata": "m",
            "_id": "a",
            "$vectorize": "a",
            "x": 9,
        },
        True,
    ),
    ({"$vector": [0], "metadata": 9, "_id": "a"}, None),
    ({"$vector": [0], "metadata": 9, "_id": "a", "x": "x"}, True),
    ({"$vector": [0], "metadata": 9, "_id": "a", "x": 9}, None),
    (
        {
            "$vector": [0],
            "metadata": 9,
            "_id": "a",
            "$vectorize": "a",
            "x": "x",
        },
        True,
    ),
    ({"$vector": [0], "metadata": 9, "_id": "a", "$vectorize": "a", "x": 9}, True),
    ({"$vector": [0], "_id": "a"}, None),
    ({"$vector": [0], "_id": "a", "x": "x"}, True),
    ({"$vector": [0], "_id": "a", "x": 9}, None),
    ({"$vector": [0], "_id": "a", "$vectorize": "a", "x": "x"}, True),
    ({"$vector": [0], "_id": "a", "$vectorize": "a", "x": 9}, True),
]
DOC_FLATNESS_TEST_IDS = [f"DOC=<{json.dumps(doc)}>" for doc, _ in DOC_FLATNESS_PAIRS]

ff = FLAT_DOCUMENT
df = DEEP_DOCUMENT
uf = UNKNOWN_FLATNESS_DOCUMENT
DOCS_FLATNESS_TEST_PARAMS = [
    pytest.param([], False, id=" docs=[] "),
    pytest.param([uf], False, id=" docs=[u] "),
    pytest.param([uf, uf], False, id=" docs=[u, u] "),
    pytest.param([df], False, id=" docs=[d] "),
    pytest.param([df, df], False, id=" docs=[d, d] "),
    pytest.param([df, uf], False, id=" docs=[d, u] "),
    pytest.param([ff], True, id=" docs=[f] "),
    pytest.param([ff, ff], True, id=" docs=[f, f] "),
    pytest.param([ff, uf], True, id=" docs=[f, u] "),
    pytest.param([ff, df], ValueError(), id=" docs=[f, d] "),
]

DOC_CF_TEST_PARAMS = [
    pytest.param(DOCUMENT_WITH_CF_X, "x", id="cf=x"),
    pytest.param(DOCUMENT_WITH_CF_Y, "y", id="cf=y"),
    pytest.param(DOCUMENT_WITH_UNKNOWN_CF, None, id="unknown-cf"),
    pytest.param({"x": "LL", "_id": "a"}, "x", id="only-x"),
    pytest.param({"x": 1234, "_id": "a"}, None, id="x-is-number"),
    pytest.param(
        {"$lexical": "thelexical", "bla": "abc", "_id": "a"},
        "bla",
        id="prefer-nonlexical",
    ),
    pytest.param(
        {"$lexical": "lex", "size": 123, "_id": "a"},
        "$lexical",
        id="pick-lexical",
    ),
    pytest.param({"_id": "a"}, None, id="no-fields"),
]

xc = DOCUMENT_WITH_CF_X
yc = DOCUMENT_WITH_CF_Y
uc = DOCUMENT_WITH_UNKNOWN_CF
DOCS_CF_TEST_PARAMS = [
    pytest.param([], "q", "q", id=" [],req='q' "),
    pytest.param([xc], "q", "q", id=" [x],req='q' "),
    pytest.param([xc, xc, yc], "q", "q", id=" [x, x, y],req='q' "),
    pytest.param([uc, uc], "q", "q", id=" [u, u],req='q' "),
    pytest.param([xc, uc, uc], "q", "q", id=" [x, u, u],req='q' "),
    pytest.param([xc, xc, yc, uc, uc, uc], "q", "q", id=" [x, x, y, u, u, u],req='q' "),
    pytest.param([], "*", ValueError, id=" [],req='*' "),
    pytest.param([xc], "*", "x", id=" [x],req='*' "),
    pytest.param([xc, xc, yc], "*", "x", id=" [x, x, y],req='*' "),
    pytest.param([uc, uc], "*", ValueError, id=" [u, u],req='*' "),
    pytest.param([xc, uc, uc], "*", "x", id=" [x, u, u],req='*' "),
    pytest.param([xc, xc, yc, uc, uc, uc], "*", "x", id=" [x, x, y, u, u, u],req='*' "),
]

DOCS_LEXICAL_TEST_PARAMS = [
    pytest.param(
        [{}, {}, {}, {"$lexical": "bla"}, {}],
        True,
        id="one_has_it",
    ),
    pytest.param(
        [],
        None,
        id="empty_doc_list",
    ),
    pytest.param(
        [{"blo": 1}, {"bla": 2}, {"ble": 3}, {"lexical": "no!"}, {"_id": 5}],
        False,
        id="none_has_it",
    ),
]


class TestVSAutodetectInferences:
    @pytest.mark.parametrize(
        ("document", "expected_flatness"), DOC_FLATNESS_PAIRS, ids=DOC_FLATNESS_TEST_IDS
    )
    def test_detect_document_flatness(
        self,
        document: dict[str, Any],
        *,
        expected_flatness: bool | None,
    ) -> None:
        """Test expected results for flatness detection."""
        assert _detect_document_flatness(document) is expected_flatness

    @pytest.mark.parametrize(
        ("documents", "expected_flatness"),
        DOCS_FLATNESS_TEST_PARAMS,
    )
    def test_detect_documents_flatness(
        self,
        documents: list[dict[str, Any]],
        *,
        expected_flatness: bool | Exception,
    ) -> None:
        """Test flatness detection from a list of documents."""
        if isinstance(expected_flatness, bool):
            assert _detect_documents_flatness(documents) is expected_flatness
        else:
            with pytest.raises(ValueError, match="Mixed"):
                _detect_documents_flatness(documents)

    @pytest.mark.parametrize(
        ("document", "expected_content_field"),
        DOC_CF_TEST_PARAMS,
    )
    def test_detect_document_content_field(
        self,
        document: dict[str, Any],
        expected_content_field: str | None,
    ) -> None:
        """Test content-field detection on a document."""
        if isinstance(expected_content_field, str):
            assert _detect_document_content_field(document) == expected_content_field
        elif expected_content_field is None:
            assert _detect_document_content_field(document) is None
        else:
            raise NotImplementedError

    @pytest.mark.parametrize(
        ("documents", "requested_content_field", "expected_content_field"),
        DOCS_CF_TEST_PARAMS,
    )
    def test_detect_documents_content_field(
        self,
        documents: list[dict[str, Any]],
        requested_content_field: str,
        expected_content_field: str | Exception,
    ) -> None:
        """Test content-field detection on a list of document."""
        if isinstance(expected_content_field, str):
            detected_cf = _detect_documents_content_field(
                documents=documents,
                requested_content_field=requested_content_field,
            )
            assert detected_cf == expected_content_field
        else:
            with pytest.raises(ValueError, match="not infer"):
                _detect_documents_content_field(
                    documents=documents,
                    requested_content_field=requested_content_field,
                )

    @pytest.mark.parametrize(
        ("documents", "expected_has_lexical"),
        DOCS_LEXICAL_TEST_PARAMS,
    )
    def test_detect_documents_have_lexical(
        self,
        documents: list[dict[str, Any]],
        *,
        expected_has_lexical: bool | None,
    ) -> None:
        assert _detect_documents_have_lexical(documents) is expected_has_lexical
