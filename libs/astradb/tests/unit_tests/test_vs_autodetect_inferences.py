from __future__ import annotations

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


class TestVSAutodetectInferences:
    def test_detect_document_flatness(self) -> None:
        """Test expected results for flatness detection."""
        doc_00 = UNKNOWN_FLATNESS_DOCUMENT
        assert _detect_document_flatness(doc_00) is None
        doc_01 = {"$vector": [0], "metadata": {}, "_id": "a"}
        assert _detect_document_flatness(doc_01) is None
        doc_02 = {
            "$vector": [0],
            "metadata": {},
            "_id": "a",
            "$vectorize": "a",
            "x": "x",
        }
        assert _detect_document_flatness(doc_02) is False
        doc_03 = DEEP_DOCUMENT
        assert _detect_document_flatness(doc_03) is False
        # note: as soon as `metadata` is a string, it is just a field name like any:
        doc_04 = {"$vector": [0], "metadata": "m", "_id": "a"}
        assert _detect_document_flatness(doc_04) is True
        doc_05 = FLAT_DOCUMENT
        assert _detect_document_flatness(doc_05) is True
        doc_06 = {"$vector": [0], "metadata": "m", "_id": "a", "x": 9}
        assert _detect_document_flatness(doc_06) is True
        doc_07 = {
            "$vector": [0],
            "metadata": "m",
            "_id": "a",
            "$vectorize": "a",
            "x": "x",
        }
        assert _detect_document_flatness(doc_07) is True
        doc_08 = {
            "$vector": [0],
            "metadata": "m",
            "_id": "a",
            "$vectorize": "a",
            "x": 9,
        }
        assert _detect_document_flatness(doc_08) is True
        doc_09 = {"$vector": [0], "metadata": 9, "_id": "a"}
        assert _detect_document_flatness(doc_09) is None
        doc_10 = {"$vector": [0], "metadata": 9, "_id": "a", "x": "x"}
        assert _detect_document_flatness(doc_10) is True
        doc_11 = {"$vector": [0], "metadata": 9, "_id": "a", "x": 9}
        assert _detect_document_flatness(doc_11) is None
        doc_12 = {
            "$vector": [0],
            "metadata": 9,
            "_id": "a",
            "$vectorize": "a",
            "x": "x",
        }
        assert _detect_document_flatness(doc_12) is True
        doc_13 = {"$vector": [0], "metadata": 9, "_id": "a", "$vectorize": "a", "x": 9}
        assert _detect_document_flatness(doc_13) is True
        doc_14 = {"$vector": [0], "_id": "a"}
        assert _detect_document_flatness(doc_14) is None
        doc_15 = {"$vector": [0], "_id": "a", "x": "x"}
        assert _detect_document_flatness(doc_15) is True
        doc_16 = {"$vector": [0], "_id": "a", "x": 9}
        assert _detect_document_flatness(doc_16) is None
        doc_17 = {"$vector": [0], "_id": "a", "$vectorize": "a", "x": "x"}
        assert _detect_document_flatness(doc_17) is True
        doc_18 = {"$vector": [0], "_id": "a", "$vectorize": "a", "x": 9}
        assert _detect_document_flatness(doc_18) is True

    def test_detect_documents_flatness(self) -> None:
        """Test flatness detection from a list of documents."""
        f = FLAT_DOCUMENT
        d = DEEP_DOCUMENT
        u = UNKNOWN_FLATNESS_DOCUMENT
        assert _detect_documents_flatness([]) is False
        assert _detect_documents_flatness([u]) is False
        assert _detect_documents_flatness([u, u]) is False
        assert _detect_documents_flatness([d]) is False
        assert _detect_documents_flatness([d, d]) is False
        assert _detect_documents_flatness([d, u]) is False
        assert _detect_documents_flatness([f]) is True
        assert _detect_documents_flatness([f, f]) is True
        assert _detect_documents_flatness([f, u]) is True
        with pytest.raises(ValueError, match="Mixed"):
            assert _detect_documents_flatness([f, d]) is True

    def test_detect_document_content_field(self) -> None:
        """Test content-field detection on a document."""
        doc_2ss = DOCUMENT_WITH_CF_X
        assert _detect_document_content_field(doc_2ss) == "x"
        doc_2sn = DOCUMENT_WITH_CF_Y
        assert _detect_document_content_field(doc_2sn) == "y"
        doc_2nn = DOCUMENT_WITH_UNKNOWN_CF
        assert _detect_document_content_field(doc_2nn) is None
        doc_1s = {"x": "LL", "_id": "a"}
        assert _detect_document_content_field(doc_1s) == "x"
        doc_1n = {"x": 1234, "_id": "a"}
        assert _detect_document_content_field(doc_1n) is None
        doc_0 = {"_id": "a"}
        assert _detect_document_content_field(doc_0) is None

    def test_detect_documents_content_field(self) -> None:
        """Test content-field detection on a list of document."""
        x = DOCUMENT_WITH_CF_X
        y = DOCUMENT_WITH_CF_Y
        u = DOCUMENT_WITH_UNKNOWN_CF

        assert (
            _detect_documents_content_field(documents=[], requested_content_field="q")
            == "q"
        )
        assert (
            _detect_documents_content_field(documents=[x], requested_content_field="q")
            == "q"
        )
        assert (
            _detect_documents_content_field(
                documents=[x, x, y], requested_content_field="q"
            )
            == "q"
        )
        assert (
            _detect_documents_content_field(
                documents=[u, u], requested_content_field="q"
            )
            == "q"
        )
        assert (
            _detect_documents_content_field(
                documents=[x, u, u], requested_content_field="q"
            )
            == "q"
        )
        assert (
            _detect_documents_content_field(
                documents=[x, x, y, u, u, u], requested_content_field="q"
            )
            == "q"
        )

        with pytest.raises(ValueError, match="not infer"):
            _detect_documents_content_field(documents=[], requested_content_field="*")
        assert (
            _detect_documents_content_field(documents=[x], requested_content_field="*")
            == "x"
        )
        assert (
            _detect_documents_content_field(
                documents=[x, x, y], requested_content_field="*"
            )
            == "x"
        )
        with pytest.raises(ValueError, match="not infer"):
            _detect_documents_content_field(
                documents=[u, u], requested_content_field="*"
            )
        assert (
            _detect_documents_content_field(
                documents=[x, u, u], requested_content_field="*"
            )
            == "x"
        )
        assert (
            _detect_documents_content_field(
                documents=[x, x, y, u, u, u], requested_content_field="*"
            )
            == "x"
        )
