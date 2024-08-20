from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document


def default_filter_encoder(filter_dict: Dict[str, Any]) -> Dict[str, Any]:
    metadata_filter = {}
    for k, v in filter_dict.items():
        if k and k[0] == "$":
            if isinstance(v, list):
                metadata_filter[k] = [default_filter_encoder(f) for f in v]
            else:
                # assume each list item can be fed back to this function
                metadata_filter[k] = default_filter_encoder(v)  # type: ignore[assignment]
        else:
            metadata_filter[f"metadata.{k}"] = v

    return metadata_filter


class VSDocumentEncoder(ABC):
    """
    TODO
    """

    server_side_embeddings: bool
    content_field: str
    base_projection: Dict[str, bool]
    full_projection: Dict[str, bool]

    @abstractmethod
    def encode(
        self,
        content: str,
        id: str,
        vector: Optional[List[float]],
        metadata: Optional[dict],
    ) -> Dict[str, Any]:
        """
        TODO
        """
        ...

    @abstractmethod
    def decode(self, astra_document: Dict[str, Any]) -> Document:
        """
        TODO
        """
        ...

    @abstractmethod
    def encode_filter(self, filter_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        TODO
        """
        ...


class DefaultVSDocumentEncoder(VSDocumentEncoder):
    server_side_embeddings = False
    content_field = "content"
    base_projection = {"_id": True, "content": True, "metadata": True}
    full_projection = {"_id": True, "content": True, "metadata": True, "$vector": True}

    def encode(
        self,
        content: str,
        id: str,
        vector: Optional[List[float]],
        metadata: Optional[dict],
    ) -> Dict[str, Any]:
        assert vector is not None  # TODO remove
        return {
            "content": content,
            "_id": id,
            "$vector": vector,
            "metadata": metadata,
        }

    def decode(self, astra_document: Dict[str, Any]) -> Document:
        return Document(
            page_content=astra_document["content"],
            metadata=astra_document["metadata"],
        )

    def encode_filter(self, filter_dict: Dict[str, Any]) -> Dict[str, Any]:
        return default_filter_encoder(filter_dict)


class DefaultVectorizeVSDocumentEncoder(VSDocumentEncoder):
    server_side_embeddings = True
    content_field = "$vectorize"
    base_projection = {"_id": True, "$vectorize": True, "metadata": True}
    full_projection = {
        "_id": True,
        "$vectorize": True,
        "metadata": True,
        "$vector": True,
    }

    def encode(
        self,
        content: str,
        id: str,
        vector: Optional[List[float]],
        metadata: Optional[dict],
    ) -> Dict[str, Any]:
        assert vector is None  # TODO remove
        return {
            "$vectorize": content,
            "_id": id,
            "metadata": metadata,
        }

    def decode(self, astra_document: Dict[str, Any]) -> Document:
        return Document(
            page_content=astra_document["$vectorize"],
            metadata=astra_document["metadata"],
        )

    def encode_filter(self, filter_dict: Dict[str, Any]) -> Dict[str, Any]:
        return default_filter_encoder(filter_dict)
