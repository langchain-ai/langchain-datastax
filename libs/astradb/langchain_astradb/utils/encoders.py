"""Classes to handle encoding of documents on DB for the Vector Store.."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents import Document
from typing_extensions import override

NO_NULL_VECTOR_MSG = "Default encoder cannot receive null vector"
VECTOR_REQUIRED_PREAMBLE_MSG = "DefaultVectorize encoder got a non-null vector"


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
            metadata_filter[f"metadata.{k}"] = v

    return metadata_filter


class _AstraDBVectorStoreDocumentEncoder(ABC):
    """A document encoder for the Astra DB vector store.

    The document encoder contains the information for consistent interaction
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
    def decode(self, astra_document: dict[str, Any]) -> Document:
        """Create a LangChain Document instance from a document retrieved from Astra DB.

        Args:
            astra_document: a dictionary as retrieved from Astra DB.

        Returns:
            a (langchain) Document corresponding to the input.
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


class _DefaultVSDocumentEncoder(_AstraDBVectorStoreDocumentEncoder):
    """Encoder for the default vector store usage with client-side embeddings.

    This encoder expresses how document are stored for collections created
    and entirely managed by the AstraDBVectorStore class.
    """

    server_side_embeddings = False
    content_field = "content"

    def __init__(self) -> None:
        """Initialize a new DefaultVSDocumentEncoder."""
        self.base_projection = {"_id": True, "content": True, "metadata": True}
        self.full_projection = {
            "_id": True,
            "content": True,
            "metadata": True,
            "$vector": True,
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
        return {
            "content": content,
            "_id": document_id,
            "$vector": vector,
            "metadata": metadata,
        }

    @override
    def decode(self, astra_document: dict[str, Any]) -> Document:
        return Document(
            page_content=astra_document["content"],
            metadata=astra_document["metadata"],
        )

    @override
    def encode_filter(self, filter_dict: dict[str, Any]) -> dict[str, Any]:
        return _default_encode_filter(filter_dict)


class _DefaultVectorizeVSDocumentEncoder(_AstraDBVectorStoreDocumentEncoder):
    """Encoder for the default vector store usage with server-side embeddings.

    This encoder expresses how document are stored for collections created
    and entirely managed by the AstraDBVectorStore class, for the case of
    server-side embeddings (aka $vectorize).
    """

    server_side_embeddings = True
    content_field = "$vectorize"

    def __init__(self) -> None:
        """Initialize a new DefaultVectorizeVSDocumentEncoder."""
        self.base_projection = {"_id": True, "$vectorize": True, "metadata": True}
        self.full_projection = {
            "_id": True,
            "$vectorize": True,
            "metadata": True,
            "$vector": True,
        }

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
            "metadata": metadata,
        }

    @override
    def decode(self, astra_document: dict[str, Any]) -> Document:
        return Document(
            page_content=astra_document["$vectorize"],
            metadata=astra_document["metadata"],
        )

    @override
    def encode_filter(self, filter_dict: dict[str, Any]) -> dict[str, Any]:
        return _default_encode_filter(filter_dict)
