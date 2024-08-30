from __future__ import annotations

import pytest
from langchain_core.documents import Document

from langchain_astradb.utils.encoders import (
    NO_NULL_VECTOR_MSG,
    VECTOR_REQUIRED_PREAMBLE_MSG,
    _DefaultVectorizeVSDocumentEncoder,
    _DefaultVSDocumentEncoder,
    _FlatVectorizeVSDocumentEncoder,
    _FlatVSDocumentEncoder,
)

METADATA = {"m1": 1, "m2": "two"}
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
} | METADATA
ASTRA_FLAT_DOCUMENT_VECTORIZE_WRITTEN = {
    "_id": DOCUMENT_ID,
    "$vectorize": CONTENT,
} | METADATA
ASTRA_FLAT_DOCUMENT_VECTORIZE_READ = ASTRA_FLAT_DOCUMENT_VECTORIZE_WRITTEN | {
    "$vector": VECTOR
}
ASTRA_FLAT_FILTER = LC_FILTER


class TestVSDocEncoders:
    def test_default_novectorize_encoding(self) -> None:
        """Test encoding for default, no-vectorize."""
        encoder = _DefaultVSDocumentEncoder(
            content_field="content_x", ignore_invalid_documents=False
        )
        encoded_doc = encoder.encode(
            content=CONTENT,
            document_id=DOCUMENT_ID,
            vector=VECTOR,
            metadata=METADATA,
        )
        assert encoded_doc == ASTRA_DEFAULT_DOCUMENT_NOVECTORIZE

    def test_default_novectorize_vector_passed(self) -> None:
        """Test vector is required for default encoding, no-vectorize."""
        encoder = _DefaultVSDocumentEncoder(
            content_field="content_x", ignore_invalid_documents=False
        )
        with pytest.raises(
            ValueError,
            match=NO_NULL_VECTOR_MSG,
        ):
            encoder.encode(
                content=CONTENT,
                document_id=DOCUMENT_ID,
                vector=None,
                metadata=METADATA,
            )

    def test_default_novectorize_decoding(self) -> None:
        """Test decoding for default, no-vectorize."""
        encoder = _DefaultVSDocumentEncoder(
            content_field="content_x", ignore_invalid_documents=False
        )
        decoded_doc = encoder.decode(ASTRA_DEFAULT_DOCUMENT_NOVECTORIZE)
        assert decoded_doc == LC_DOCUMENT

    def test_default_novectorize_decoding_invalid(self) -> None:
        """Test decoding for invalid documents, no-vectorize."""
        encoder_e = _DefaultVSDocumentEncoder(
            content_field="content_x", ignore_invalid_documents=False
        )
        with pytest.raises(KeyError):
            encoder_e.decode({"_id": 0})
        encoder_w = _DefaultVSDocumentEncoder(
            content_field="content_x", ignore_invalid_documents=True
        )
        with pytest.warns(UserWarning) as rec_warnings:
            decoded_doc = encoder_w.decode({"_id": 0})
        assert len(rec_warnings) == 1
        assert decoded_doc is None

    def test_default_novectorize_filtering(self) -> None:
        """Test filter-encoding for default, no-vectorize."""
        encoder = _DefaultVSDocumentEncoder(
            content_field="content_x", ignore_invalid_documents=False
        )
        encoded_flt = encoder.encode_filter(LC_FILTER)
        assert encoded_flt == ASTRA_DEFAULT_FILTER

    def test_default_vectorize_encoding(self) -> None:
        """Test encoding for default, vectorize."""
        encoder = _DefaultVectorizeVSDocumentEncoder(ignore_invalid_documents=False)
        encoded_doc = encoder.encode(
            content=CONTENT,
            document_id=DOCUMENT_ID,
            vector=None,
            metadata=METADATA,
        )
        assert encoded_doc == ASTRA_DEFAULT_DOCUMENT_VECTORIZE

    def test_default_vectorize_vector_passed(self) -> None:
        """Test vector is not allowed for default encoding, vectorize."""
        encoder = _DefaultVectorizeVSDocumentEncoder(ignore_invalid_documents=False)
        with pytest.raises(
            ValueError,
            match=VECTOR_REQUIRED_PREAMBLE_MSG,
        ):
            encoder.encode(
                content=CONTENT,
                document_id=DOCUMENT_ID,
                vector=VECTOR,
                metadata=METADATA,
            )

    def test_default_vectorize_decoding(self) -> None:
        """Test decoding for default, vectorize."""
        encoder = _DefaultVectorizeVSDocumentEncoder(ignore_invalid_documents=False)
        decoded_doc = encoder.decode(ASTRA_DEFAULT_DOCUMENT_VECTORIZE)
        assert decoded_doc == LC_DOCUMENT

    def test_default_vectorize_decoding_invalid(self) -> None:
        """Test decoding for invalid documents, no-vectorize."""
        encoder_e = _DefaultVectorizeVSDocumentEncoder(ignore_invalid_documents=False)
        with pytest.raises(KeyError):
            encoder_e.decode({"_id": 0})
        encoder_w = _DefaultVectorizeVSDocumentEncoder(ignore_invalid_documents=True)
        with pytest.warns(UserWarning) as rec_warnings:
            decoded_doc = encoder_w.decode({"_id": 0})
        assert len(rec_warnings) == 1
        assert decoded_doc is None

    def test_default_vectorize_filtering(self) -> None:
        """Test filter-encoding for default, vectorize."""
        encoder = _DefaultVectorizeVSDocumentEncoder(ignore_invalid_documents=False)
        encoded_flt = encoder.encode_filter(LC_FILTER)
        assert encoded_flt == ASTRA_DEFAULT_FILTER

    def test_flat_novectorize_encoding(self) -> None:
        """Test encoding for flat, no-vectorize."""
        encoder = _FlatVSDocumentEncoder(
            content_field="content_x", ignore_invalid_documents=False
        )
        encoded_doc = encoder.encode(
            content=CONTENT,
            document_id=DOCUMENT_ID,
            vector=VECTOR,
            metadata=METADATA,
        )
        assert encoded_doc == ASTRA_FLAT_DOCUMENT_NOVECTORIZE

    def test_flat_novectorize_vector_passed(self) -> None:
        """Test vector is required for flat encoding, no-vectorize."""
        encoder = _FlatVSDocumentEncoder(
            content_field="content_x", ignore_invalid_documents=False
        )
        with pytest.raises(
            ValueError,
            match=NO_NULL_VECTOR_MSG,
        ):
            encoder.encode(
                content=CONTENT,
                document_id=DOCUMENT_ID,
                vector=None,
                metadata=METADATA,
            )

    def test_flat_novectorize_decoding(self) -> None:
        """Test decoding for flat, no-vectorize."""
        encoder = _FlatVSDocumentEncoder(
            content_field="content_x", ignore_invalid_documents=False
        )
        decoded_doc = encoder.decode(ASTRA_FLAT_DOCUMENT_NOVECTORIZE)
        assert decoded_doc == LC_DOCUMENT

    def test_flat_novectorize_decoding_invalid(self) -> None:
        """Test decoding for invalid documents, no-vectorize."""
        encoder_e = _FlatVSDocumentEncoder(
            content_field="content_x", ignore_invalid_documents=False
        )
        with pytest.raises(KeyError):
            encoder_e.decode({"_id": 0})
        encoder_w = _FlatVSDocumentEncoder(
            content_field="content_x", ignore_invalid_documents=True
        )
        with pytest.warns(UserWarning) as rec_warnings:
            decoded_doc = encoder_w.decode({"_id": 0})
        assert len(rec_warnings) == 1
        assert decoded_doc is None

    def test_flat_novectorize_filtering(self) -> None:
        """Test filter-encoding for flat, no-vectorize."""
        encoder = _FlatVSDocumentEncoder(
            content_field="content_x", ignore_invalid_documents=False
        )
        encoded_flt = encoder.encode_filter(LC_FILTER)
        assert encoded_flt == ASTRA_FLAT_FILTER

    def test_flat_vectorize_encoding(self) -> None:
        """Test encoding for flat, vectorize."""
        encoder = _FlatVectorizeVSDocumentEncoder(ignore_invalid_documents=False)
        encoded_doc = encoder.encode(
            content=CONTENT,
            document_id=DOCUMENT_ID,
            vector=None,
            metadata=METADATA,
        )
        assert encoded_doc == ASTRA_FLAT_DOCUMENT_VECTORIZE_WRITTEN

    def test_flat_vectorize_vector_passed(self) -> None:
        """Test vector is forbidden for flat encoding, vectorize."""
        encoder = _FlatVectorizeVSDocumentEncoder(ignore_invalid_documents=False)
        with pytest.raises(
            ValueError,
            match=VECTOR_REQUIRED_PREAMBLE_MSG,
        ):
            encoder.encode(
                content=CONTENT,
                document_id=DOCUMENT_ID,
                vector=VECTOR,
                metadata=METADATA,
            )

    def test_flat_vectorize_decoding(self) -> None:
        """Test decoding for flat, vectorize."""
        encoder = _FlatVectorizeVSDocumentEncoder(ignore_invalid_documents=False)
        decoded_doc = encoder.decode(ASTRA_FLAT_DOCUMENT_VECTORIZE_READ)
        assert decoded_doc == LC_DOCUMENT

    def test_flat_vectorize_decoding_invalid(self) -> None:
        """Test decoding for invalid documents, vectorize."""
        encoder_e = _FlatVectorizeVSDocumentEncoder(ignore_invalid_documents=False)
        with pytest.raises(KeyError):
            encoder_e.decode({"_id": 0})
        encoder_w = _FlatVectorizeVSDocumentEncoder(ignore_invalid_documents=True)
        with pytest.warns(UserWarning) as rec_warnings:
            decoded_doc = encoder_w.decode({"_id": 0})
        assert len(rec_warnings) == 1
        assert decoded_doc is None

    def test_flat_vectorize_filtering(self) -> None:
        """Test filter-encoding for flat, vectorize."""
        encoder = _FlatVectorizeVSDocumentEncoder(ignore_invalid_documents=False)
        encoded_flt = encoder.encode_filter(LC_FILTER)
        assert encoded_flt == ASTRA_FLAT_FILTER
