from __future__ import annotations

from typing import Any

import pytest
from langchain_core.documents import Document

from langchain_astradb.utils.vector_store_codecs import (
    _AstraDBVectorStoreDocumentCodec,
    _DefaultVectorizeVSDocumentCodec,
    _DefaultVSDocumentCodec,
    _FlatVectorizeVSDocumentCodec,
    _FlatVSDocumentCodec,
)

THE_VECTOR: list[float] = [1, 2, 3]
THE_CONTENT = "d_pc"
THE_ID = "d_id"
THE_METADATA = {"d_mdk": "d_mdv"}
THE_DOCUMENT = Document(page_content=THE_CONTENT, id=THE_ID, metadata=THE_METADATA)
THE_LEXICAL = "d_lexical"
THE_QUERY = "d_search_query"

EXPECTED_DEFAULT_NOVECTORIZE_DOC = {
    "content_x": THE_CONTENT,
    "_id": THE_ID,
    "$vector": THE_VECTOR,
    "metadata": THE_METADATA,
    "$lexical": THE_CONTENT,
}
EXPECTED_FLAT_NOVECTORIZE_DOC = {
    "content_x": THE_CONTENT,
    "_id": THE_ID,
    "$vector": THE_VECTOR,
    "$lexical": THE_CONTENT,
    **THE_METADATA,
}
EXPECTED_NOVECTORIZE_H_SORT = {
    "$hybrid": {
        "$vector": THE_VECTOR,
        "$lexical": THE_LEXICAL,
    },
}
EXPECTED_DEFAULT_VECTORIZE_DOC = {
    "_id": THE_ID,
    "$vectorize": THE_CONTENT,
    "metadata": THE_METADATA,
    "$lexical": THE_CONTENT,
}
EXPECTED_FLAT_VECTORIZE_DOC = {
    "_id": THE_ID,
    "$vectorize": THE_CONTENT,
    "$lexical": THE_CONTENT,
    **THE_METADATA,
}
EXPECTED_VECTORIZE_H_SORT = {
    "$hybrid": {
        "$vectorize": THE_QUERY,
        "$lexical": THE_LEXICAL,
    },
}


class TestVSDocHybridCodecs:
    @pytest.mark.parametrize(
        ("has_vectorize", "codec_class", "expected_encoded"),
        [
            (False, _DefaultVSDocumentCodec, EXPECTED_DEFAULT_NOVECTORIZE_DOC),
            (False, _FlatVSDocumentCodec, EXPECTED_FLAT_NOVECTORIZE_DOC),
            (True, _DefaultVectorizeVSDocumentCodec, EXPECTED_DEFAULT_VECTORIZE_DOC),
            (True, _FlatVectorizeVSDocumentCodec, EXPECTED_FLAT_VECTORIZE_DOC),
        ],
        ids=[
            "default/novectorize",
            "flat/novectorize",
            "default/vectorize",
            "flat/vectorize",
        ],
    )
    def test_novectorize_hybrid_encoding(
        self,
        *,
        has_vectorize: bool,
        codec_class: type[_AstraDBVectorStoreDocumentCodec],
        expected_encoded: dict[str, Any],
    ) -> None:
        codec: _AstraDBVectorStoreDocumentCodec
        if has_vectorize:
            codec = codec_class(  # type: ignore[call-arg]
                ignore_invalid_documents=False,
                has_lexical=True,
            )
        else:
            codec = codec_class(  # type: ignore[call-arg]
                content_field="content_x",
                ignore_invalid_documents=False,
                has_lexical=True,
            )

        encoded = codec.encode(
            content=THE_CONTENT,
            document_id=THE_ID,
            vector=None if has_vectorize else THE_VECTOR,
            metadata=THE_METADATA,
        )
        assert encoded == expected_encoded

        assert codec.decode(encoded) == THE_DOCUMENT

        encoded_hs = codec.encode_hybrid_sort(
            vector=None if has_vectorize else THE_VECTOR,
            vectorize=THE_QUERY if has_vectorize else None,
            lexical=THE_LEXICAL,
        )
        if has_vectorize:
            assert encoded_hs == EXPECTED_VECTORIZE_H_SORT
        else:
            assert encoded_hs == EXPECTED_NOVECTORIZE_H_SORT

        with pytest.raises(ValueError, match="hybrid sort requires"):
            codec.encode_hybrid_sort(
                vector=THE_VECTOR,
                vectorize="bla",
                lexical=THE_LEXICAL,
            )

        with pytest.raises(ValueError, match="hybrid sort requires"):
            codec.encode_hybrid_sort(
                vector=THE_VECTOR if has_vectorize else None,
                vectorize=None if has_vectorize else "bla",
                lexical=THE_LEXICAL,
            )

        if has_vectorize:
            assert codec.rerank_on is None
        else:
            assert codec.rerank_on == "content_x"
