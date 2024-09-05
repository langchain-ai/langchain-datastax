from __future__ import annotations

import json
from typing import Any

import pytest

from langchain_astradb.utils.vector_store_autodetect import (
    _detect_document_content_field,
    _detect_document_flatness,
    _detect_documents_content_field,
    _detect_documents_flatness,
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
df = DEEP_DOCUMENT  # noqa: PD901
uf = UNKNOWN_FLATNESS_DOCUMENT
DOCS_FLATNESS_PAIRS = [
    ([], False),
    ([uf], False),
    ([uf, uf], False),
    ([df], False),
    ([df, df], False),
    ([df, uf], False),
    ([ff], True),
    ([ff, ff], True),
    ([ff, uf], True),
    ([ff, df], ValueError()),
]
DOCS_FLATNESS_TEST_IDS = [
    " docs=[] ",
    " docs=[u] ",
    " docs=[u, u] ",
    " docs=[d] ",
    " docs=[d, d] ",
    " docs=[d, u] ",
    " docs=[f] ",
    " docs=[f, f] ",
    " docs=[f, u] ",
    " docs=[f, d] ",
]
DOC_CF_PAIRS = [
    (DOCUMENT_WITH_CF_X, "x"),
    (DOCUMENT_WITH_CF_Y, "y"),
    (DOCUMENT_WITH_UNKNOWN_CF, None),
    ({"x": "LL", "_id": "a"}, "x"),
    ({"x": 1234, "_id": "a"}, None),
    ({"_id": "a"}, None),
]
DOC_CF_TEST_IDS = [
    "cf=x",
    "cf=y",
    "unknown-cf",
    "only-x",
    "x-is-number",
    "no-fields",
]
xc = DOCUMENT_WITH_CF_X
yc = DOCUMENT_WITH_CF_Y
uc = DOCUMENT_WITH_UNKNOWN_CF
DOCS_CF_TRIPLES = [
    ([], "q", "q"),
    ([xc], "q", "q"),
    ([xc, xc, yc], "q", "q"),
    ([uc, uc], "q", "q"),
    ([xc, uc, uc], "q", "q"),
    ([xc, xc, yc, uc, uc, uc], "q", "q"),
    ([], "*", ValueError),
    ([xc], "*", "x"),
    ([xc, xc, yc], "*", "x"),
    ([uc, uc], "*", ValueError),
    ([xc, uc, uc], "*", "x"),
    ([xc, xc, yc, uc, uc, uc], "*", "x"),
]
DOCS_CF_TEST_IDS = [
    "[]",
    "[x]",
    "[x, x, y]",
    "[u, u]",
    "[x, u, u]",
    "[x, x, y, u, u, u]",
    "[]",
    "[x]",
    "[x, x, y]",
    "[u, u]",
    "[x, u, u]",
    "[x, x, y, u, u, u]",
]


class TestVSAutodetectInferences:
    @pytest.mark.parametrize(
        ("document", "expected_flatness"), DOC_FLATNESS_PAIRS, ids=DOC_FLATNESS_TEST_IDS
    )
    def test_detect_document_flatness(
        self,
        document: dict[str, Any],
        expected_flatness: bool | None,
    ) -> None:
        """Test expected results for flatness detection."""
        assert _detect_document_flatness(document) is expected_flatness

    @pytest.mark.parametrize(
        ("documents", "expected_flatness"),
        DOCS_FLATNESS_PAIRS,
        ids=DOCS_FLATNESS_TEST_IDS,
    )
    def test_detect_documents_flatness(
        self,
        documents: list[dict[str, Any]],
        expected_flatness: bool | Exception,
    ) -> None:
        """Test flatness detection from a list of documents."""
        if isinstance(expected_flatness, bool):
            assert _detect_documents_flatness(documents) is expected_flatness
        else:
            with pytest.raises(ValueError, match="Mixed"):
                _detect_documents_flatness(documents)

    @pytest.mark.parametrize(
        ("document", "expected_content_field"), DOC_CF_PAIRS, ids=DOC_CF_TEST_IDS
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
        DOCS_CF_TRIPLES,
        ids=DOCS_CF_TEST_IDS,
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
