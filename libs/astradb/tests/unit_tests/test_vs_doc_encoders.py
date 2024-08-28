from __future__ import annotations

import pytest
from langchain_core.documents import Document

from langchain_astradb.utils.encoders import (
    NO_NULL_VECTOR_MSG,
    VECTOR_REQUIRED_PREAMBLE_MSG,
    _DefaultVectorizeVSDocumentEncoder,
    _DefaultVSDocumentEncoder,
)

METADATA = {"m1": 1, "m2": "two"}
CONTENT = "The content"
VECTOR: list[float] = [1, 2, 3]
DOCUMENT_ID = "the_id"
LC_DOCUMENT = Document(page_content=CONTENT, metadata=METADATA)
ASTRA_DEFAULT_DOCUMENT_NOVECTORIZE = {
    "_id": DOCUMENT_ID,
    "content": CONTENT,
    "metadata": METADATA,
    "$vector": VECTOR,
}
ASTRA_DEFAULT_DOCUMENT_VECTORIZE = {
    "_id": DOCUMENT_ID,
    "$vectorize": CONTENT,
    "metadata": METADATA,
}
LC_FILTER = {"a0": 0, "$or": [{"b1": 1}, {"b2": 2}]}
ASTRA_DEFAULT_FILTER = {
    "metadata.a0": 0,
    "$or": [{"metadata.b1": 1}, {"metadata.b2": 2}],
}


class TestVSDocEncoders:
    def test_default_novectorize_encoding(self) -> None:
        """Test encoding for default, no-vectorize."""
        encoder = _DefaultVSDocumentEncoder()
        encoded_doc = encoder.encode(
            content=CONTENT,
            document_id=DOCUMENT_ID,
            vector=VECTOR,
            metadata=METADATA,
        )
        assert encoded_doc == ASTRA_DEFAULT_DOCUMENT_NOVECTORIZE

    def test_default_novectorize_vector_required(self) -> None:
        """Test vector is required for default encoding, no-vectorize."""
        encoder = _DefaultVSDocumentEncoder()
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
        encoder = _DefaultVSDocumentEncoder()
        decoded_doc = encoder.decode(ASTRA_DEFAULT_DOCUMENT_NOVECTORIZE)
        assert decoded_doc == LC_DOCUMENT

    def test_default_novectorize_filtering(self) -> None:
        """Test filter-encoding for default, no-vectorize."""
        encoder = _DefaultVSDocumentEncoder()
        encoded_flt = encoder.encode_filter(LC_FILTER)
        assert encoded_flt == ASTRA_DEFAULT_FILTER

    def test_default_vectorize_encoding(self) -> None:
        """Test encoding for default, vectorize."""
        encoder = _DefaultVectorizeVSDocumentEncoder()
        encoded_doc = encoder.encode(
            content=CONTENT,
            document_id=DOCUMENT_ID,
            vector=None,
            metadata=METADATA,
        )
        assert encoded_doc == ASTRA_DEFAULT_DOCUMENT_VECTORIZE

    def test_default_vectorize_vector_forbidden(self) -> None:
        """Test vector is not allowed for default encoding, vectorize."""
        encoder = _DefaultVectorizeVSDocumentEncoder()
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
        encoder = _DefaultVectorizeVSDocumentEncoder()
        decoded_doc = encoder.decode(ASTRA_DEFAULT_DOCUMENT_VECTORIZE)
        assert decoded_doc == LC_DOCUMENT

    def test_default_vectorize_filtering(self) -> None:
        """Test filter-encoding for default, vectorize."""
        encoder = _DefaultVectorizeVSDocumentEncoder()
        encoded_flt = encoder.encode_filter(LC_FILTER)
        assert encoded_flt == ASTRA_DEFAULT_FILTER
