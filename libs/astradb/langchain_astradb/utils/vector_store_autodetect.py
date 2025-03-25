"""Utilities for AstraDB vector store autodetect mode."""

from __future__ import annotations

import logging
import warnings
from collections import Counter
from operator import itemgetter
from typing import (
    Any,
)

from langchain_astradb.utils.vector_store_codecs import (
    LEXICAL_FIELD_NAME,
    _AstraDBVectorStoreDocumentCodec,
    _DefaultVectorizeVSDocumentCodec,
    _DefaultVSDocumentCodec,
    _FlatVectorizeVSDocumentCodec,
    _FlatVSDocumentCodec,
)

logger = logging.getLogger(__name__)


def _detect_document_flatness(document: dict[str, Any]) -> bool | None:
    """Try to guess, when possible, if this document has metadata-as-a-dict or not."""
    _metadata = document.get("metadata")
    _vector = document.get("$vector")
    _regularfields = set(document.keys()) - {"_id", "$vector"}
    _regularfields_m = _regularfields - {"metadata"}
    # cannot determine if ...
    if _vector is None:
        return None
    # now a determination
    if isinstance(_metadata, dict) and _regularfields_m:
        return False
    if isinstance(_metadata, dict) and not _regularfields_m:
        # this document should not contribute to the survey
        return None
    str_regularfields = {
        k for k, v in document.items() if isinstance(v, str) if k in _regularfields
    }
    if str_regularfields:
        # Note: even if the only string field is "metadata"
        return True
    return None


def _detect_documents_flatness(documents: list[dict[str, Any]]) -> bool:
    flatness_survey = [_detect_document_flatness(document) for document in documents]
    n_flats = flatness_survey.count(True)
    n_deeps = flatness_survey.count(False)
    if n_flats > 0 and n_deeps > 0:
        msg = "Mixed document shapes detected on collection during autodetect."
        raise ValueError(msg)

    # in absence of clues, 0 < 0 is False and default is NON FLAT (i.e. native)
    return n_deeps < n_flats


def _detect_document_content_field(document: dict[str, Any]) -> str | None:
    """Try to guess the content field by inspecting the passed document.

    Prefer other fields than $lexical if available.
    """
    strlen_map = {
        k: len(v) for k, v in document.items() if k != "_id" if isinstance(v, str)
    }
    if not strlen_map:
        return None

    preferred_strlen_map = {k: v for k, v in strlen_map.items() if k != "$lexical"}
    if preferred_strlen_map:
        return max(preferred_strlen_map.items(), key=itemgetter(1))[0]
    return max(strlen_map.items(), key=itemgetter(1))[0]


def _detect_documents_content_field(
    documents: list[dict[str, Any]],
    requested_content_field: str,
) -> str:
    if requested_content_field == "*":
        # guess content_field by docs inspection
        content_fields = [
            _detect_document_content_field(document) for document in documents
        ]
        valid_content_fields = [cf for cf in content_fields if cf is not None]
        logger.info(
            "vector store autodetect: inferring content_field from %i documents",
            len(valid_content_fields),
        )
        cf_stats = Counter(valid_content_fields)
        if not cf_stats:
            msg = "Could not infer content_field name from sampled documents."
            raise ValueError(msg)
        return cf_stats.most_common(1)[0][0]

    return requested_content_field


def _detect_documents_have_lexical(documents: list[dict[str, Any]]) -> bool:
    return any(LEXICAL_FIELD_NAME in document for document in documents)


def _detect_document_codec(
    documents: list[dict[str, Any]],
    *,
    has_vectorize: bool,
    has_lexical: bool,
    ignore_invalid_documents: bool,
    norm_content_field: str,
) -> _AstraDBVectorStoreDocumentCodec:
    logger.info("vector store autodetect: inspecting %i documents", len(documents))
    # survey and determine flatness
    is_flat = _detect_documents_flatness(documents)
    logger.info("vector store autodetect: is_flat = %s", is_flat)

    final_content_field = _detect_documents_content_field(
        documents=documents,
        requested_content_field=norm_content_field,
    )
    logger.info(
        "vector store autodetect: final_content_field = %s", final_content_field
    )

    lexical_in_docs = _detect_documents_have_lexical(documents)
    logger.info(
        "vector store autodetect: lexical_in_docs = %s (has_lexical = %s)",
        lexical_in_docs,
        has_lexical,
    )
    if has_lexical and not lexical_in_docs:
        warnings.warn(
            "Collection supports lexical; however, autodetect encountered documents "
            "without a $lexical. The $lexical field will be set for new documents, "
            "but the pre-existing documents might not be reached by lexical search "
            "when running hybrid search mode.",
            stacklevel=2,
        )
    if not has_lexical and lexical_in_docs:
        warnings.warn(
            "Documents with $lexical encountered on a non-hybrid-capable "
            "collection. This inconsistency will be ignored moving forward "
            "(and new documents will not be written with any $lexical).",
            stacklevel=2,
        )

    if has_vectorize:
        if is_flat:
            return _FlatVectorizeVSDocumentCodec(
                ignore_invalid_documents=ignore_invalid_documents,
                has_lexical=has_lexical,
            )

        return _DefaultVectorizeVSDocumentCodec(
            ignore_invalid_documents=ignore_invalid_documents,
            has_lexical=has_lexical,
        )
    # no vectorize:
    if is_flat:
        return _FlatVSDocumentCodec(
            content_field=final_content_field,
            ignore_invalid_documents=ignore_invalid_documents,
            has_lexical=has_lexical,
        )
    return _DefaultVSDocumentCodec(
        content_field=final_content_field,
        ignore_invalid_documents=ignore_invalid_documents,
        has_lexical=has_lexical,
    )
