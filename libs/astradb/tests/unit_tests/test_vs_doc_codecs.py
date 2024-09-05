from __future__ import annotations

from typing import Any

import pytest
from langchain_core.documents import Document

from langchain_astradb.utils.vector_store_codecs import (
    NO_NULL_VECTOR_MSG,
    VECTOR_REQUIRED_PREAMBLE_MSG,
    _DefaultVectorizeVSDocumentCodec,
    _DefaultVSDocumentCodec,
    _FlatVectorizeVSDocumentCodec,
    _FlatVSDocumentCodec,
)

METADATA: dict[str, Any] = {"m1": 1, "m2": "two"}
CONTENT = "The content"
VECTOR: list[float] = [1, 2, 3]
DOCUMENT_ID = "the_id"
LC_DOCUMENT = Document(page_content=CONTENT, metadata=METADATA)
LC_FILTER = {"a0": 0, "$or": [{"b1": 1}, {"b2": 2}]}

ASTRA_DEFAULT_DOCUMENT_NOVECTORIZE = {
    "_id": DOCUMENT_ID,
    "content_x": CONTENT,
    "metadata": METADATA,
    "$vector": VECTOR,
}
ASTRA_DEFAULT_DOCUMENT_VECTORIZE = {
    "_id": DOCUMENT_ID,
    "$vectorize": CONTENT,
    "metadata": METADATA,
}
ASTRA_DEFAULT_FILTER = {
    "metadata.a0": 0,
    "$or": [{"metadata.b1": 1}, {"metadata.b2": 2}],
}

ASTRA_FLAT_DOCUMENT_NOVECTORIZE = {
    "_id": DOCUMENT_ID,
    "content_x": CONTENT,
    "$vector": VECTOR,
    **METADATA,
}
ASTRA_FLAT_DOCUMENT_VECTORIZE_WRITTEN: dict[str, Any] = {
    "_id": DOCUMENT_ID,
    "$vectorize": CONTENT,
    **METADATA,
}
ASTRA_FLAT_DOCUMENT_VECTORIZE_READ = {
    "$vector": VECTOR,
    **ASTRA_FLAT_DOCUMENT_VECTORIZE_WRITTEN,
}
ASTRA_FLAT_FILTER = LC_FILTER


class TestVSDocCodecs:
    def test_default_novectorize_encoding(self) -> None:
        """Test encoding for default, no-vectorize."""
        codec = _DefaultVSDocumentCodec(
            content_field="content_x", ignore_invalid_documents=False
        )
        encoded_doc = codec.encode(
            content=CONTENT,
            document_id=DOCUMENT_ID,
            vector=VECTOR,
            metadata=METADATA,
        )
        assert encoded_doc == ASTRA_DEFAULT_DOCUMENT_NOVECTORIZE

    def test_default_novectorize_vector_passed(self) -> None:
        """Test vector is required for default encoding, no-vectorize."""
        codec = _DefaultVSDocumentCodec(
            content_field="content_x", ignore_invalid_documents=False
        )
        with pytest.raises(
            ValueError,
            match=NO_NULL_VECTOR_MSG,
        ):
            codec.encode(
                content=CONTENT,
                document_id=DOCUMENT_ID,
                vector=None,
                metadata=METADATA,
            )

    def test_default_novectorize_decoding(self) -> None:
        """Test decoding for default, no-vectorize."""
        codec = _DefaultVSDocumentCodec(
            content_field="content_x", ignore_invalid_documents=False
        )
        decoded_doc = codec.decode(ASTRA_DEFAULT_DOCUMENT_NOVECTORIZE)
        assert decoded_doc == LC_DOCUMENT

    def test_default_novectorize_decoding_invalid(self) -> None:
        """Test decoding for invalid documents, no-vectorize."""
        codec_e = _DefaultVSDocumentCodec(
            content_field="content_x", ignore_invalid_documents=False
        )
        with pytest.raises(KeyError):
            codec_e.decode({"_id": 0})
        codec_w = _DefaultVSDocumentCodec(
            content_field="content_x", ignore_invalid_documents=True
        )
        with pytest.warns(UserWarning) as rec_warnings:
            decoded_doc = codec_w.decode({"_id": 0})
        codec_w.decode({"_id": 0, "content_x": "a", "metadata": {"k": "v"}})
        codec_e.decode({"_id": 0, "content_x": "a", "metadata": {"k": "v"}})
        assert len(rec_warnings) == 1
        assert decoded_doc is None

    def test_default_novectorize_filtering(self) -> None:
        """Test filter-encoding for default, no-vectorize."""
        codec = _DefaultVSDocumentCodec(
            content_field="content_x", ignore_invalid_documents=False
        )
        encoded_flt = codec.encode_filter(LC_FILTER)
        assert encoded_flt == ASTRA_DEFAULT_FILTER

    def test_default_vectorize_encoding(self) -> None:
        """Test encoding for default, vectorize."""
        codec = _DefaultVectorizeVSDocumentCodec(ignore_invalid_documents=False)
        encoded_doc = codec.encode(
            content=CONTENT,
            document_id=DOCUMENT_ID,
            vector=None,
            metadata=METADATA,
        )
        assert encoded_doc == ASTRA_DEFAULT_DOCUMENT_VECTORIZE

    def test_default_vectorize_vector_passed(self) -> None:
        """Test vector is not allowed for default encoding, vectorize."""
        codec = _DefaultVectorizeVSDocumentCodec(ignore_invalid_documents=False)
        with pytest.raises(
            ValueError,
            match=VECTOR_REQUIRED_PREAMBLE_MSG,
        ):
            codec.encode(
                content=CONTENT,
                document_id=DOCUMENT_ID,
                vector=VECTOR,
                metadata=METADATA,
            )

    def test_default_vectorize_decoding(self) -> None:
        """Test decoding for default, vectorize."""
        codec = _DefaultVectorizeVSDocumentCodec(ignore_invalid_documents=False)
        decoded_doc = codec.decode(ASTRA_DEFAULT_DOCUMENT_VECTORIZE)
        assert decoded_doc == LC_DOCUMENT

    def test_default_vectorize_decoding_invalid(self) -> None:
        """Test decoding for invalid documents, no-vectorize."""
        codec_e = _DefaultVectorizeVSDocumentCodec(ignore_invalid_documents=False)
        with pytest.raises(KeyError):
            codec_e.decode({"_id": 0})
        codec_w = _DefaultVectorizeVSDocumentCodec(ignore_invalid_documents=True)
        with pytest.warns(UserWarning) as rec_warnings:
            decoded_doc = codec_w.decode({"_id": 0})
        codec_w.decode({"_id": 0, "$vectorize": "a", "metadata": {"k": "v"}})
        codec_e.decode({"_id": 0, "$vectorize": "a", "metadata": {"k": "v"}})
        assert len(rec_warnings) == 1
        assert decoded_doc is None

    def test_default_vectorize_filtering(self) -> None:
        """Test filter-encoding for default, vectorize."""
        codec = _DefaultVectorizeVSDocumentCodec(ignore_invalid_documents=False)
        encoded_flt = codec.encode_filter(LC_FILTER)
        assert encoded_flt == ASTRA_DEFAULT_FILTER

    def test_flat_novectorize_encoding(self) -> None:
        """Test encoding for flat, no-vectorize."""
        codec = _FlatVSDocumentCodec(
            content_field="content_x", ignore_invalid_documents=False
        )
        encoded_doc = codec.encode(
            content=CONTENT,
            document_id=DOCUMENT_ID,
            vector=VECTOR,
            metadata=METADATA,
        )
        assert encoded_doc == ASTRA_FLAT_DOCUMENT_NOVECTORIZE

    def test_flat_novectorize_vector_passed(self) -> None:
        """Test vector is required for flat encoding, no-vectorize."""
        codec = _FlatVSDocumentCodec(
            content_field="content_x", ignore_invalid_documents=False
        )
        with pytest.raises(
            ValueError,
            match=NO_NULL_VECTOR_MSG,
        ):
            codec.encode(
                content=CONTENT,
                document_id=DOCUMENT_ID,
                vector=None,
                metadata=METADATA,
            )

    def test_flat_novectorize_decoding(self) -> None:
        """Test decoding for flat, no-vectorize."""
        codec = _FlatVSDocumentCodec(
            content_field="content_x", ignore_invalid_documents=False
        )
        decoded_doc = codec.decode(ASTRA_FLAT_DOCUMENT_NOVECTORIZE)
        assert decoded_doc == LC_DOCUMENT

    def test_flat_novectorize_decoding_invalid(self) -> None:
        """Test decoding for invalid documents, no-vectorize."""
        codec_e = _FlatVSDocumentCodec(
            content_field="content_x", ignore_invalid_documents=False
        )
        with pytest.raises(KeyError):
            codec_e.decode({"_id": 0})
        codec_w = _FlatVSDocumentCodec(
            content_field="content_x", ignore_invalid_documents=True
        )
        with pytest.warns(UserWarning) as rec_warnings:
            decoded_doc = codec_w.decode({"_id": 0})
        codec_w.decode({"_id": 0, "content_x": "a", "m": "v"})
        codec_e.decode({"_id": 0, "content_x": "a", "m": "v"})
        assert len(rec_warnings) == 1
        assert decoded_doc is None

    def test_flat_novectorize_filtering(self) -> None:
        """Test filter-encoding for flat, no-vectorize."""
        codec = _FlatVSDocumentCodec(
            content_field="content_x", ignore_invalid_documents=False
        )
        encoded_flt = codec.encode_filter(LC_FILTER)
        assert encoded_flt == ASTRA_FLAT_FILTER

    def test_flat_vectorize_encoding(self) -> None:
        """Test encoding for flat, vectorize."""
        codec = _FlatVectorizeVSDocumentCodec(ignore_invalid_documents=False)
        encoded_doc = codec.encode(
            content=CONTENT,
            document_id=DOCUMENT_ID,
            vector=None,
            metadata=METADATA,
        )
        assert encoded_doc == ASTRA_FLAT_DOCUMENT_VECTORIZE_WRITTEN

    def test_flat_vectorize_vector_passed(self) -> None:
        """Test vector is forbidden for flat encoding, vectorize."""
        codec = _FlatVectorizeVSDocumentCodec(ignore_invalid_documents=False)
        with pytest.raises(
            ValueError,
            match=VECTOR_REQUIRED_PREAMBLE_MSG,
        ):
            codec.encode(
                content=CONTENT,
                document_id=DOCUMENT_ID,
                vector=VECTOR,
                metadata=METADATA,
            )

    def test_flat_vectorize_decoding(self) -> None:
        """Test decoding for flat, vectorize."""
        codec = _FlatVectorizeVSDocumentCodec(ignore_invalid_documents=False)
        decoded_doc = codec.decode(ASTRA_FLAT_DOCUMENT_VECTORIZE_READ)
        assert decoded_doc == LC_DOCUMENT

    def test_flat_vectorize_decoding_invalid(self) -> None:
        """Test decoding for invalid documents, vectorize."""
        codec_e = _FlatVectorizeVSDocumentCodec(ignore_invalid_documents=False)
        with pytest.raises(KeyError):
            codec_e.decode({"_id": 0})
        codec_w = _FlatVectorizeVSDocumentCodec(ignore_invalid_documents=True)
        with pytest.warns(UserWarning) as rec_warnings:
            decoded_doc = codec_w.decode({"_id": 0})
        codec_w.decode({"_id": 0, "$vectorize": "a", "m": "v"})
        codec_e.decode({"_id": 0, "$vectorize": "a", "m": "v"})
        assert len(rec_warnings) == 1
        assert decoded_doc is None

    def test_flat_vectorize_filtering(self) -> None:
        """Test filter-encoding for flat, vectorize."""
        codec = _FlatVectorizeVSDocumentCodec(ignore_invalid_documents=False)
        encoded_flt = codec.encode_filter(LC_FILTER)
        assert encoded_flt == ASTRA_FLAT_FILTER
