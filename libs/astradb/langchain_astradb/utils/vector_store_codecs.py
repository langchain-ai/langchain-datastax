"""Classes to handle encoding of documents on DB for the Vector Store.."""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents import Document
from typing_extensions import override

NO_NULL_VECTOR_MSG = "Default (non-vectorize) codec cannot encode null vector."
VECTOR_REQUIRED_PREAMBLE_MSG = (
    "Default vectorize codec got a non-null vector to encode."
)
FLATTEN_CONFLICT_MSG = "Cannot flatten metadata: field name overlap for '{field}'."

STANDARD_INDEXING_OPTIONS_DEFAULT = {"allow": ["metadata"]}

logger = logging.getLogger(__name__)


def _default_decode_vector(astra_doc: dict[str, Any]) -> list[float] | None:
    return astra_doc.get("$vector")


def _default_metadata_key_to_field_identifier(md_key: str) -> str:
    return f"metadata.{md_key}"


def _flat_metadata_key_to_field_identifier(md_key: str) -> str:
    return md_key


def _default_encode_filter(filter_dict: dict[str, Any]) -> dict[str, Any]:
    metadata_filter = {}
    for k, v in filter_dict.items():
        # Key in this dict starting with $ are supposedly operators and as such
        # should not be nested within the `metadata.` prefix. For instance,
        # >>> _default_encode_filter({'a':1, '$or': [{'b':2}, {'c': 3}]})
        #     {'metadata.a': 1, '$or': [{'metadata.b': 2}, {'metadata.c': 3}]}
        if k and k[0] == "$":
            if isinstance(v, list):
                metadata_filter[k] = [_default_encode_filter(f) for f in v]
            else:
                # assume each list item can be fed back to this function
                metadata_filter[k] = _default_encode_filter(v)  # type: ignore[assignment]
        else:
            metadata_filter[_default_metadata_key_to_field_identifier(k)] = v

    return metadata_filter


def _default_encode_id(filter_id: str) -> dict[str, Any]:
    return {"_id": filter_id}


def _default_encode_ids(filter_ids: list[str]) -> dict[str, Any]:
    if len(filter_ids) == 1:
        return _default_encode_id(filter_ids[0])
    return {"_id": {"$in": filter_ids}}


def _default_encode_vector_sort(vector: list[float]) -> dict[str, Any]:
    return {"$vector": vector}


class _AstraDBVectorStoreDocumentCodec(ABC):
    """A document codec for the Astra DB vector store.

    The document codec contains the information for consistent interaction
    with documents as stored on the Astra DB collection.

    Implementations of this class must:
    - define how to encode/decode documents consistently to and from
      Astra DB collections. The two operations must, so to speak, combine
      to the identity on both sides (except for the quirks of their signatures).
    - provide the adequate projection dictionaries for running find
      operations on Astra DB, with and without the field containing the vector.
    - encode IDs to the `_id` field on Astra DB.
    - define the name of the field storing the textual content of the Document.
    - define whether embeddings are computed server-side (with $vectorize) or not.
    """

    server_side_embeddings: bool
    content_field: str
    base_projection: dict[str, bool]
    full_projection: dict[str, bool]
    ignore_invalid_documents: bool

    @abstractmethod
    def encode(
        self,
        content: str,
        document_id: str,
        vector: list[float] | None,
        metadata: dict | None,
    ) -> dict[str, Any]:
        """Create a document for storage on Astra DB.

        Args:
            content: textual content for the (LangChain) `Document`.
            document_id: unique ID for the (LangChain) `Document`.
            vector: a vector associated to the (LangChain) `Document`. This
                parameter must be None for and only for server-side embeddings.
            metadata: a metadata dictionary for the (LangChain) `Document`.

        Returns:
            a dictionary ready for storage onto Astra DB.
        """

    @abstractmethod
    def decode(self, astra_document: dict[str, Any]) -> Document | None:
        """Create a LangChain Document instance from a document retrieved from Astra DB.

        Args:
            astra_document: a dictionary as retrieved from Astra DB.

        Returns:
            a (langchain) Document corresponding to the input.
        """

    @abstractmethod
    def decode_vector(self, astra_document: dict[str, Any]) -> list[float] | None:
        """Create a vector from a document retrieved from Astra DB.

        Args:
            astra_document: a dictionary as retrieved from Astra DB.

        Returns:
            a vector corresponding to the input.
        """

    @abstractmethod
    def encode_filter(self, filter_dict: dict[str, Any]) -> dict[str, Any]:
        """Encode a LangChain filter for use in Astra DB queries.

        Make a LangChain filter into a filter clause suitable for operations
        against the Astra DB collection, consistently with the encoding scheme.

        Args:
            filter_dict: a filter in the standardized metadata-filtering form
                used throughout LangChain.

        Returns:
            an equivalent filter clause for use in Astra DB's find queries.
        """

    @abstractmethod
    def metadata_key_to_field_identifier(self, md_key: str) -> str:
        """Express an 'abstract' metadata key as a full Data API field identifier."""

    @property
    @abstractmethod
    def default_collection_indexing_policy(self) -> dict[str, list[str]]:
        """Provide the default indexing policy if the collection must be created."""

    def get_id(self, astra_document: dict[str, Any]) -> str:
        """Return the ID of an encoded document (= a raw JSON read from DB)."""
        return astra_document["_id"]

    def get_similarity(self, astra_document: dict[str, Any]) -> float:
        """Return the similarity of an encoded document (= a raw JSON read from DB).

        This method assumes its argument comes from a suitable vector search.
        """
        return astra_document["$similarity"]

    def encode_id(self, filter_id: str) -> dict[str, Any]:
        """Encode an ID as a filter for use in Astra DB queries.

        Args:
            filter_id: the ID value to filter on.

        Returns:
            a filter clause for use in Astra DB's find queries.
        """
        return _default_encode_id(filter_id)

    def encode_ids(self, filter_ids: list[str]) -> dict[str, Any]:
        """Encode a list of IDs as an appropriate search filter.

        The resulting filter expresses condition: "document ID is among filter_ids".

        Args:
            filter_ids: the ID values to filter on.

        Returns:
            a filter clause for use in Astra DB's find queries.
        """
        return _default_encode_ids(filter_ids)

    def encode_vector_sort(self, vector: list[float]) -> dict[str, Any]:
        """Encode a vector as a sort to use for Astra DB queries.

        Args:
            vector: the search vector to order results by.

        Returns:
            an order clause for use in Astra DB's find queries.
        """
        return _default_encode_vector_sort(vector)


class _DefaultVSDocumentCodec(_AstraDBVectorStoreDocumentCodec):
    """Codec for the default vector store usage with client-side embeddings.

    This codec expresses how document are stored for collections created
    and entirely managed by the AstraDBVectorStore class.
    """

    server_side_embeddings = False

    def __init__(self, content_field: str, *, ignore_invalid_documents: bool) -> None:
        """Initialize a new DefaultVSDocumentCodec.

        Args:
            content_field: name of the (top-level) field for textual content.
            ignore_invalid_documents: if True, noncompliant inputs to `decode`
                are logged and a None is returned (instead of raising an exception).
        """
        self.content_field = content_field
        self.base_projection = {"_id": True, self.content_field: True, "metadata": True}
        self.full_projection = {
            "_id": True,
            self.content_field: True,
            "metadata": True,
            "$vector": True,
        }
        self.ignore_invalid_documents = ignore_invalid_documents

    @override
    def encode(
        self,
        content: str,
        document_id: str,
        vector: list[float] | None,
        metadata: dict | None,
    ) -> dict[str, Any]:
        if vector is None:
            raise ValueError(NO_NULL_VECTOR_MSG)
        return {
            self.content_field: content,
            "_id": document_id,
            "$vector": vector,
            "metadata": metadata or {},
        }

    @override
    def decode(self, astra_document: dict[str, Any]) -> Document | None:
        _invalid_doc = (
            "metadata" not in astra_document or self.content_field not in astra_document
        )
        if _invalid_doc and self.ignore_invalid_documents:
            invalid_doc_warning = (
                "Ignoring document with _id = "
                f"{astra_document.get('_id', '(no _id)')}. "
                "Reason: missing required fields."
            )
            logger.warning(invalid_doc_warning)
            return None
        return Document(
            page_content=astra_document[self.content_field],
            metadata=astra_document["metadata"],
            id=astra_document["_id"],
        )

    @override
    def decode_vector(self, astra_document: dict[str, Any]) -> list[float] | None:
        return _default_decode_vector(astra_document)

    @override
    def encode_filter(self, filter_dict: dict[str, Any]) -> dict[str, Any]:
        return _default_encode_filter(filter_dict)

    @override
    def metadata_key_to_field_identifier(self, md_key: str) -> str:
        return _default_metadata_key_to_field_identifier(md_key)

    @property
    def default_collection_indexing_policy(self) -> dict[str, list[str]]:
        return STANDARD_INDEXING_OPTIONS_DEFAULT


class _DefaultVectorizeVSDocumentCodec(_AstraDBVectorStoreDocumentCodec):
    """Codec for the default vector store usage with server-side embeddings.

    This codec expresses how document are stored for collections created
    and entirely managed by the AstraDBVectorStore class, for the case of
    server-side embeddings (aka $vectorize).
    """

    server_side_embeddings = True
    content_field = "$vectorize"

    def __init__(self, *, ignore_invalid_documents: bool) -> None:
        """Initialize a new DefaultVectorizeVSDocumentCodec.

        Args:
            ignore_invalid_documents: if True, noncompliant inputs to `decode`
                are logged and a None is returned (instead of raising an exception).
        """
        self.base_projection = {"_id": True, "$vectorize": True, "metadata": True}
        self.full_projection = {
            "_id": True,
            "$vectorize": True,
            "metadata": True,
            "$vector": True,
        }
        self.ignore_invalid_documents = ignore_invalid_documents

    @override
    def encode(
        self,
        content: str,
        document_id: str,
        vector: list[float] | None,
        metadata: dict | None,
    ) -> dict[str, Any]:
        if vector is not None:
            msg = f"{VECTOR_REQUIRED_PREAMBLE_MSG}: {vector}"
            raise ValueError(msg)
        return {
            "$vectorize": content,
            "_id": document_id,
            "metadata": metadata or {},
        }

    @override
    def decode(self, astra_document: dict[str, Any]) -> Document | None:
        _invalid_doc = (
            "metadata" not in astra_document or "$vectorize" not in astra_document
        )
        if _invalid_doc and self.ignore_invalid_documents:
            invalid_doc_warning = (
                "Ignoring document with _id = "
                f"{astra_document.get('_id', '(no _id)')}. "
                "Reason: missing required fields."
            )
            warnings.warn(
                invalid_doc_warning,
                stacklevel=2,
            )
            return None
        return Document(
            page_content=astra_document["$vectorize"],
            metadata=astra_document["metadata"],
            id=astra_document["_id"],
        )

    @override
    def decode_vector(self, astra_document: dict[str, Any]) -> list[float] | None:
        return _default_decode_vector(astra_document)

    @override
    def encode_filter(self, filter_dict: dict[str, Any]) -> dict[str, Any]:
        return _default_encode_filter(filter_dict)

    @property
    def default_collection_indexing_policy(self) -> dict[str, list[str]]:
        return STANDARD_INDEXING_OPTIONS_DEFAULT

    @override
    def metadata_key_to_field_identifier(self, md_key: str) -> str:
        return _default_metadata_key_to_field_identifier(md_key)


class _FlatVSDocumentCodec(_AstraDBVectorStoreDocumentCodec):
    """Codec for collections populated externally, with client-side embeddings.

    This codec manages document structured as a flat key-value map, with one
    field being the textual content and the other implicitly forming the "metadata".
    """

    server_side_embeddings = False

    def __init__(self, content_field: str, *, ignore_invalid_documents: bool) -> None:
        """Initialize a new DefaultVSDocumentCodec.

        Args:
            content_field: name of the (top-level) field for textual content.
            ignore_invalid_documents: if True, noncompliant inputs to `decode`
                are logged and a None is returned (instead of raising an exception).
        """
        self.content_field = content_field
        self.base_projection = {"_id": True, "$vector": False}
        self.full_projection = {"*": True}
        self.ignore_invalid_documents = ignore_invalid_documents
        self._non_md_fields = {
            "_id",
            "$vector",
            "$vectorize",
            self.content_field,
            "$similarity",
        }

    @override
    def encode(
        self,
        content: str,
        document_id: str,
        vector: list[float] | None,
        metadata: dict | None,
    ) -> dict[str, Any]:
        if vector is None:
            raise ValueError(NO_NULL_VECTOR_MSG)
        if self.content_field in (metadata or {}):
            msg = FLATTEN_CONFLICT_MSG.format(field=self.content_field)
            raise ValueError(msg)
        return {
            self.content_field: content,
            "_id": document_id,
            "$vector": vector,
            **(metadata or {}),
        }

    @override
    def decode(self, astra_document: dict[str, Any]) -> Document | None:
        if self.content_field not in astra_document and self.ignore_invalid_documents:
            invalid_doc_warning = (
                "Ignoring document with _id = "
                f"{astra_document.get('_id', '(no _id)')}. "
                "Reason: missing required fields."
            )
            warnings.warn(
                invalid_doc_warning,
                stacklevel=2,
            )
            return None
        _metadata = {
            k: v for k, v in astra_document.items() if k not in self._non_md_fields
        }
        return Document(
            page_content=astra_document[self.content_field],
            metadata=_metadata,
            id=astra_document["_id"],
        )

    @override
    def decode_vector(self, astra_document: dict[str, Any]) -> list[float] | None:
        return _default_decode_vector(astra_document)

    @override
    def encode_filter(self, filter_dict: dict[str, Any]) -> dict[str, Any]:
        return filter_dict

    @property
    def default_collection_indexing_policy(self) -> dict[str, list[str]]:
        return {"deny": [self.content_field]}

    @override
    def metadata_key_to_field_identifier(self, md_key: str) -> str:
        return _flat_metadata_key_to_field_identifier(md_key)


class _FlatVectorizeVSDocumentCodec(_AstraDBVectorStoreDocumentCodec):
    """Codec for collections populated externally, with server-side embeddings.

    This codec manages document structured as a flat key-value map, with one
    field being the textual content and the other implicitly forming the "metadata".
    """

    server_side_embeddings = True
    content_field = "$vectorize"

    def __init__(self, *, ignore_invalid_documents: bool) -> None:
        """Initialize a new DefaultVectorizeVSDocumentCodec.

        Args:
            ignore_invalid_documents: if True, noncompliant inputs to `decode`
                are logged and a None is returned (instead of raising an exception).
        """
        self.base_projection = {"_id": True, "$vector": False, "$vectorize": True}
        self.full_projection = {"*": True}
        self.ignore_invalid_documents = ignore_invalid_documents
        self._non_md_fields = {"_id", "$vector", "$vectorize", "$similarity"}

    @override
    def encode(
        self,
        content: str,
        document_id: str,
        vector: list[float] | None,
        metadata: dict | None,
    ) -> dict[str, Any]:
        if vector is not None:
            msg = f"{VECTOR_REQUIRED_PREAMBLE_MSG}: {vector}"
            raise ValueError(msg)
        if "$vectorize" in (metadata or {}):
            msg = FLATTEN_CONFLICT_MSG.format(field="$vectorize")
            raise ValueError(msg)
        return {
            "$vectorize": content,
            "_id": document_id,
            **(metadata or {}),
        }

    @override
    def decode(self, astra_document: dict[str, Any]) -> Document | None:
        if "$vectorize" not in astra_document and self.ignore_invalid_documents:
            invalid_doc_warning = (
                "Ignoring document with _id = "
                f"{astra_document.get('_id', '(no _id)')}. "
                "Reason: missing required fields."
            )
            warnings.warn(
                invalid_doc_warning,
                stacklevel=2,
            )
            return None
        _metadata = {
            k: v for k, v in astra_document.items() if k not in self._non_md_fields
        }
        return Document(
            page_content=astra_document["$vectorize"],
            metadata=_metadata,
            id=astra_document["_id"],
        )

    @override
    def decode_vector(self, astra_document: dict[str, Any]) -> list[float] | None:
        return _default_decode_vector(astra_document)

    @override
    def encode_filter(self, filter_dict: dict[str, Any]) -> dict[str, Any]:
        return filter_dict

    @property
    def default_collection_indexing_policy(self) -> dict[str, list[str]]:
        # $vectorize cannot be de-indexed explicitly (the API manages it entirely).
        return {}

    @override
    def metadata_key_to_field_identifier(self, md_key: str) -> str:
        return _flat_metadata_key_to_field_identifier(md_key)
