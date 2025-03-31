"""Classes to handle encoding of Documents on DB for the Vector Store.."""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, Iterable

from langchain_core.documents import Document
from typing_extensions import override

NO_NULL_VECTOR_MSG = "Default (non-vectorize) codec cannot encode null vector."
VECTOR_REQUIRED_PREAMBLE_MSG = (
    "Default vectorize codec got a non-null vector to encode."
)
FLATTEN_CONFLICT_MSG = "Cannot flatten metadata: field name overlap for '{field}'."
VECTORIZE_NOT_AVAILABLE_MSG = "Vectorize not available for this codec."

VECTOR_FIELD_NAME = "$vector"
VECTORIZE_FIELD_NAME = "$vectorize"
LEXICAL_FIELD_NAME = "$lexical"
SIMILARITY_FIELD_NAME = "$similarity"
RERANK_SORT_TOP_FIELD_NAME = "$hybrid"
DEFAULT_METADATA_FIELD_NAME = "metadata"

STANDARD_INDEXING_OPTIONS_DEFAULT = {"allow": [DEFAULT_METADATA_FIELD_NAME]}

logger = logging.getLogger(__name__)


def _default_decode_vector(astra_doc: dict[str, Any]) -> list[float] | None:
    """Extract the embedding vector from an Astra DB document."""
    return astra_doc.get(VECTOR_FIELD_NAME)


def _default_metadata_key_to_field_identifier(md_key: str) -> str:
    """Rewrite a metadata key name to its full path in the 'default' encoding.

    The input `md_key` is an "abstract" metadata key, while the return value
    identifies its actual full-path location on an Astra DB document encoded in the
    'default' way (i.e. with a nested `metadata` dictionary).
    """
    return f"{DEFAULT_METADATA_FIELD_NAME}.{md_key}"


def _flat_metadata_key_to_field_identifier(md_key: str) -> str:
    """Rewrite a metadata key name to its full path in the 'flat' encoding.

    The input `md_key` is an "abstract" metadata key, while the return value
    identifies its actual full-path location on an Astra DB document encoded in the
    'flat' way (i.e. metadata fields appearing at top-level in the Astra DB document).
    """
    return md_key


def _default_encode_filter(filter_dict: dict[str, Any]) -> dict[str, Any]:
    """Encode an "abstract" filter/sort condition for the 'default' encoding.

    The input can express a query clause, or sort criterion, on metadata and uses
    just the metadata field names, possibly connected/nested through AND and ORs.
    The output makes key names into their full path-identifiers (e.g. "metadata.xyz")
    according to the 'default' encoding scheme for Astra DB documents.
    """
    metadata_filter = {}
    for k, v in filter_dict.items():
        # Key in this dict starting with $ are supposedly operators and as such
        # should not be nested within the `metadata.` prefix. For instance,
        # >>> _default_encode_filter({'a':1, '$or': [{'b':2}, {'c': 3}]})
        #     {'metadata.a': 1, '$or': [{'metadata.b': 2}, {'metadata.c': 3}]}
        if k and k[0] == "$":
            if isinstance(v, list) and k != VECTOR_FIELD_NAME:
                metadata_filter[k] = [_default_encode_filter(f) for f in v]
            elif isinstance(v, dict):
                # assume each list item can be fed back to this function
                metadata_filter[k] = _default_encode_filter(v)  # type: ignore[assignment]
            else:
                # a scalar. As this is a 'value', never touch it
                metadata_filter[k] = v
        else:
            metadata_filter[_default_metadata_key_to_field_identifier(k)] = v

    return metadata_filter


def _astra_generic_encode_id(filter_id: str) -> dict[str, Any]:
    """Encoding of a single Document ID as a query clause for an Astra DB document."""
    return {"_id": filter_id}


def _astra_generic_encode_ids(filter_ids: list[str]) -> dict[str, Any]:
    """Encoding of Document IDs as a query clause for an Astra DB document.

    This function picks the right, and most concise, expression based on the
    multiplicity of the provided IDs.
    """
    if len(filter_ids) == 1:
        return _astra_generic_encode_id(filter_ids[0])
    return {"_id": {"$in": filter_ids}}


def _astra_generic_encode_vector_sort(vector: list[float]) -> dict[str, Any]:
    """Encoding of a vector-based sort as a query clause for an Astra DB document."""
    return {VECTOR_FIELD_NAME: vector}


def _astra_generic_encode_vectorize_sort(query_text: str) -> dict[str, Any]:
    """Encoding of a vectorize-based sort as a query clause for an Astra DB document."""
    return {VECTORIZE_FIELD_NAME: query_text}


def _astra_vector_encode_hybrid_sort(
    vector: list[float],
    lexical: str,
) -> dict[str, Any]:
    """Encoding of a sort clause for hybrid search in the non-vectorize case."""
    return {
        RERANK_SORT_TOP_FIELD_NAME: {
            VECTOR_FIELD_NAME: vector,
            LEXICAL_FIELD_NAME: lexical,
        },
    }


def _astra_vectorize_encode_hybrid_sort(
    vectorize: str,
    lexical: str,
) -> dict[str, Any]:
    """Encoding of a sort clause for hybrid search in the vectorize case."""
    if vectorize == lexical:
        return {RERANK_SORT_TOP_FIELD_NAME: vectorize}
    return {
        RERANK_SORT_TOP_FIELD_NAME: {
            VECTORIZE_FIELD_NAME: vectorize,
            LEXICAL_FIELD_NAME: lexical,
        },
    }


class _AstraDBVectorStoreDocumentCodec(ABC):
    """A Document codec for the Astra DB vector store.

    Document codecs hold the logic consistent interaction
    with documents as stored on the Astra DB collection.

    In this context, 'Document' (capital D) refers to the LangChain class,
    while 'Astra DB document' refers to the JSON-like object stored on DB.

    Implementations of this class must:
    - define how to encode/decode documents consistently to and from
      Astra DB collections. The two operations must, so to speak, combine
      to the identity on both sides (except for the quirks of their signatures).
    - provide the adequate projection dictionaries for running find
      operations on Astra DB, with and without the field containing the vector.
    - encode Document IDs to the right field on Astra DB ("_id" for Collections).
    - define the name of the field storing the textual content of the Document.
    - define whether embeddings are computed server-side (with $vectorize) or not.
    """

    server_side_embeddings: bool
    has_lexical: bool
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

    @abstractmethod
    def encode_vectorize_sort(self, query_text: str) -> dict[str, Any]:
        """Encode a query text as a sort to use for Astra DB 'vectorize' queries.

        Note that calling this method on codecs that don't support 'vectorize'
        will result in an exception being thrown.

        Args:
            query_text: the search query text to order results by.

        Returns:
            an order clause for use in Astra DB's find queries.
        """

    @abstractmethod
    def encode_hybrid_sort(
        self,
        *,
        vector: list[float] | None,
        vectorize: str | None,
        lexical: str,
    ) -> dict[str, Any]:
        """Encode a 'sort' parameter for an Astra DB 'hybrid' search.

        The input parameters must be appropriate for the particular codec:
        supplying the wrong inputs (such as a vector on a vectorize-codec)
        will result in an exception being thrown.

        Args:
            vector: a query vector (if applicable) or None.
            vectorize: a query text for vectorize search (if applicable) or None.
                Exactly one of ``vector`` and ``vectorize`` must not be None.
            lexical: the search query text for the lexical part of the hybrid search.

        Returns:
            a sort clause for use in Astra DB's findAndRerank queries.
        """

    @property
    @abstractmethod
    def default_collection_indexing_policy(self) -> dict[str, list[str]]:
        """Provide the default indexing policy if the collection must be created."""

    @property
    def rerank_on(self) -> str | None:
        """The value for 'rerank_on' in a find_and_rerank command, or None.

        This property is not None if and only if the codec is a non-vectorize one.
        """
        if self.server_side_embeddings:
            return None
        return self.content_field

    def decode_vector(self, astra_document: dict[str, Any]) -> list[float] | None:
        """Create a vector from a document retrieved from Astra DB.

        Args:
            astra_document: a dictionary as retrieved from Astra DB.

        Returns:
            a vector corresponding to the input.
        """
        return _default_decode_vector(astra_document)

    def get_id(self, astra_document: dict[str, Any]) -> str:
        """Return the ID of an encoded document (= a raw JSON read from DB)."""
        return astra_document["_id"]

    def get_similarity(self, astra_document: dict[str, Any]) -> float | None:
        """Return the similarity of an encoded document (= a raw JSON read from DB).

        This method gives no guarantees as to whether said score applies/is found.
        """
        return astra_document.get(SIMILARITY_FIELD_NAME)

    def encode_query(
        self,
        *,
        ids: Iterable[str] | None = None,
        filter_dict: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Prepare an encoded query according to the Astra DB document encoding.

        The method optionally accepts both IDs and metadata filters. The two,
        if passed together, are automatically combined with an AND operation.

        In other words, if passing both IDs and a metadata filtering clause,
        the resulting query would return Astra DB documents matching the metadata
        clause AND having an ID among those provided to this method. If, instead,
        an OR is required, one should run two separate queries and subsequently merge
        the result (taking care of avoiding duplcates).

        Args:
            ids: an iterable over Document IDs. If provided, the resulting Astra DB
                query dictionary expresses the requirement that returning documents
                have an ID among those provided here. Passing an empty iterable,
                or None, results in a query with no conditions on the IDs at all.
            filter_dict: a metadata filtering part. If provided, if must refer to
                metadata keys by their bare name (such as `{"key": 123}`).
                This filter can combine nested conditions with "$or"/"$and" connectors,
                for example:
                - `{"tag": "a"}`
                - `{"$or": [{"tag": "a"}, "label": "b"]}`
                - `{"$and": [{"tag": {"$in": ["a", "z"]}}, "label": "b"]}`

        Returns:
            a query dictionary ready to be used in an Astra DB find operation on
            a collection.
        """
        clauses: list[dict[str, Any]] = []
        _ids_list = list(ids or [])
        if _ids_list:
            clauses.append(_astra_generic_encode_ids(_ids_list))
        if filter_dict:
            clauses.append(self.encode_filter(filter_dict))

        if clauses:
            if len(clauses) > 1:
                return {"$and": clauses}
            return clauses[0]
        return {}

    def encode_vector_sort(self, vector: list[float]) -> dict[str, Any]:
        """Encode a vector as a sort to use for Astra DB queries.

        Args:
            vector: the search vector to order results by.

        Returns:
            an order clause for use in Astra DB's find queries.
        """
        return _astra_generic_encode_vector_sort(vector)


class _DefaultVSDocumentCodec(_AstraDBVectorStoreDocumentCodec):
    """Codec for the default vector store usage with client-side embeddings.

    This codec expresses how document are stored for collections created
    and entirely managed by the AstraDBVectorStore class.
    """

    server_side_embeddings = False

    def __init__(
        self, content_field: str, *, ignore_invalid_documents: bool, has_lexical: bool
    ) -> None:
        """Initialize a new DefaultVSDocumentCodec.

        Args:
            content_field: name of the (top-level) field for textual content.
            ignore_invalid_documents: if True, noncompliant inputs to `decode`
                are logged and a None is returned (instead of raising an exception).
            has_lexical: whether the codec should use the lexical field (hybrid search)
        """
        self.content_field = content_field
        self.has_lexical = has_lexical
        self.base_projection = {
            "_id": True,
            self.content_field: True,
            DEFAULT_METADATA_FIELD_NAME: True,
        }
        self.full_projection = {
            "_id": True,
            self.content_field: True,
            DEFAULT_METADATA_FIELD_NAME: True,
            VECTOR_FIELD_NAME: True,
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
            VECTOR_FIELD_NAME: vector,
            DEFAULT_METADATA_FIELD_NAME: metadata or {},
            **({LEXICAL_FIELD_NAME: content} if self.has_lexical else {}),
        }

    @override
    def decode(self, astra_document: dict[str, Any]) -> Document | None:
        _invalid_doc = (
            DEFAULT_METADATA_FIELD_NAME not in astra_document
            or self.content_field not in astra_document
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
            metadata=astra_document[DEFAULT_METADATA_FIELD_NAME],
            id=astra_document["_id"],
        )

    @override
    def encode_filter(self, filter_dict: dict[str, Any]) -> dict[str, Any]:
        return _default_encode_filter(filter_dict)

    @override
    def metadata_key_to_field_identifier(self, md_key: str) -> str:
        return _default_metadata_key_to_field_identifier(md_key)

    @override
    def encode_vectorize_sort(self, query_text: str) -> dict[str, Any]:
        raise ValueError(VECTORIZE_NOT_AVAILABLE_MSG)

    @override
    def encode_hybrid_sort(
        self,
        *,
        vector: list[float] | None,
        vectorize: str | None,
        lexical: str,
    ) -> dict[str, Any]:
        if vector is None or vectorize is not None:
            msg = "This codec's hybrid sort requires `vectorize=None` and a vector."
            raise ValueError(msg)
        return _astra_vector_encode_hybrid_sort(vector=vector, lexical=lexical)

    @property
    @override
    def default_collection_indexing_policy(self) -> dict[str, list[str]]:
        return STANDARD_INDEXING_OPTIONS_DEFAULT


class _DefaultVectorizeVSDocumentCodec(_AstraDBVectorStoreDocumentCodec):
    """Codec for the default vector store usage with server-side embeddings.

    This codec expresses how document are stored for collections created
    and entirely managed by the AstraDBVectorStore class, for the case of
    server-side embeddings (aka $vectorize).
    """

    server_side_embeddings = True
    content_field = VECTORIZE_FIELD_NAME

    def __init__(self, *, ignore_invalid_documents: bool, has_lexical: bool) -> None:
        """Initialize a new DefaultVectorizeVSDocumentCodec.

        Args:
            ignore_invalid_documents: if True, noncompliant inputs to `decode`
                are logged and a None is returned (instead of raising an exception).
            has_lexical: whether the codec should use the lexical field (hybrid search)
        """
        self.base_projection = {
            "_id": True,
            VECTORIZE_FIELD_NAME: True,
            DEFAULT_METADATA_FIELD_NAME: True,
        }
        self.full_projection = {
            "_id": True,
            VECTORIZE_FIELD_NAME: True,
            DEFAULT_METADATA_FIELD_NAME: True,
            VECTOR_FIELD_NAME: True,
        }
        self.ignore_invalid_documents = ignore_invalid_documents
        self.has_lexical = has_lexical

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
            VECTORIZE_FIELD_NAME: content,
            "_id": document_id,
            DEFAULT_METADATA_FIELD_NAME: metadata or {},
            **({LEXICAL_FIELD_NAME: content} if self.has_lexical else {}),
        }

    @override
    def decode(self, astra_document: dict[str, Any]) -> Document | None:
        _invalid_doc = (
            DEFAULT_METADATA_FIELD_NAME not in astra_document
            or VECTORIZE_FIELD_NAME not in astra_document
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
            page_content=astra_document[VECTORIZE_FIELD_NAME],
            metadata=astra_document[DEFAULT_METADATA_FIELD_NAME],
            id=astra_document["_id"],
        )

    @override
    def encode_filter(self, filter_dict: dict[str, Any]) -> dict[str, Any]:
        return _default_encode_filter(filter_dict)

    @override
    def metadata_key_to_field_identifier(self, md_key: str) -> str:
        return _default_metadata_key_to_field_identifier(md_key)

    @override
    def encode_vectorize_sort(self, query_text: str) -> dict[str, Any]:
        return _astra_generic_encode_vectorize_sort(query_text)

    @override
    def encode_hybrid_sort(
        self,
        *,
        vector: list[float] | None,
        vectorize: str | None,
        lexical: str,
    ) -> dict[str, Any]:
        if vectorize is None or vector is not None:
            msg = "This codec's hybrid sort requires `vector=None` and a vectorize."
            raise ValueError(msg)
        return _astra_vectorize_encode_hybrid_sort(vectorize=vectorize, lexical=lexical)

    @property
    @override
    def default_collection_indexing_policy(self) -> dict[str, list[str]]:
        return STANDARD_INDEXING_OPTIONS_DEFAULT


class _FlatVSDocumentCodec(_AstraDBVectorStoreDocumentCodec):
    """Codec for collections populated externally, with client-side embeddings.

    This codec manages document structured as a flat key-value map, with one
    field being the textual content and the other implicitly forming the "metadata".
    """

    server_side_embeddings = False

    def __init__(
        self, content_field: str, *, ignore_invalid_documents: bool, has_lexical: bool
    ) -> None:
        """Initialize a new DefaultVSDocumentCodec.

        Args:
            content_field: name of the (top-level) field for textual content.
            ignore_invalid_documents: if True, noncompliant inputs to `decode`
                are logged and a None is returned (instead of raising an exception).
            has_lexical: whether the codec should use the lexical field (hybrid search)
        """
        self.content_field = content_field
        self.base_projection = {"_id": True, VECTOR_FIELD_NAME: False}
        self.full_projection = {"*": True}
        self.ignore_invalid_documents = ignore_invalid_documents
        self.has_lexical = has_lexical
        self._non_md_fields = {
            "_id",
            self.content_field,
            LEXICAL_FIELD_NAME,
            VECTOR_FIELD_NAME,
            VECTORIZE_FIELD_NAME,
            SIMILARITY_FIELD_NAME,
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
            VECTOR_FIELD_NAME: vector,
            **(metadata or {}),
            **({LEXICAL_FIELD_NAME: content} if self.has_lexical else {}),
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
    def encode_filter(self, filter_dict: dict[str, Any]) -> dict[str, Any]:
        return filter_dict

    @override
    def metadata_key_to_field_identifier(self, md_key: str) -> str:
        return _flat_metadata_key_to_field_identifier(md_key)

    @override
    def encode_vectorize_sort(self, query_text: str) -> dict[str, Any]:
        raise ValueError(VECTORIZE_NOT_AVAILABLE_MSG)

    @override
    def encode_hybrid_sort(
        self,
        *,
        vector: list[float] | None,
        vectorize: str | None,
        lexical: str,
    ) -> dict[str, Any]:
        if vector is None or vectorize is not None:
            msg = "This codec's hybrid sort requires `vectorize=None` and a vector."
            raise ValueError(msg)
        return _astra_vector_encode_hybrid_sort(vector=vector, lexical=lexical)

    @property
    @override
    def default_collection_indexing_policy(self) -> dict[str, list[str]]:
        return {"deny": [self.content_field]}


class _FlatVectorizeVSDocumentCodec(_AstraDBVectorStoreDocumentCodec):
    """Codec for collections populated externally, with server-side embeddings.

    This codec manages document structured as a flat key-value map, with one
    field being the textual content and the other implicitly forming the "metadata".
    """

    server_side_embeddings = True
    content_field = VECTORIZE_FIELD_NAME

    def __init__(self, *, ignore_invalid_documents: bool, has_lexical: bool) -> None:
        """Initialize a new DefaultVectorizeVSDocumentCodec.

        Args:
            ignore_invalid_documents: if True, noncompliant inputs to `decode`
                are logged and a None is returned (instead of raising an exception).
            has_lexical: whether the codec should use the lexical field (hybrid search)
        """
        self.base_projection = {
            "_id": True,
            VECTOR_FIELD_NAME: False,
            VECTORIZE_FIELD_NAME: True,
        }
        self.full_projection = {"*": True}
        self.ignore_invalid_documents = ignore_invalid_documents
        self.has_lexical = has_lexical
        self._non_md_fields = {
            "_id",
            LEXICAL_FIELD_NAME,
            VECTOR_FIELD_NAME,
            VECTORIZE_FIELD_NAME,
            SIMILARITY_FIELD_NAME,
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
        if VECTORIZE_FIELD_NAME in (metadata or {}):
            msg = FLATTEN_CONFLICT_MSG.format(field=VECTORIZE_FIELD_NAME)
            raise ValueError(msg)
        return {
            VECTORIZE_FIELD_NAME: content,
            "_id": document_id,
            **(metadata or {}),
            **({LEXICAL_FIELD_NAME: content} if self.has_lexical else {}),
        }

    @override
    def decode(self, astra_document: dict[str, Any]) -> Document | None:
        if VECTORIZE_FIELD_NAME not in astra_document and self.ignore_invalid_documents:
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
            page_content=astra_document[VECTORIZE_FIELD_NAME],
            metadata=_metadata,
            id=astra_document["_id"],
        )

    @override
    def encode_filter(self, filter_dict: dict[str, Any]) -> dict[str, Any]:
        return filter_dict

    @override
    def metadata_key_to_field_identifier(self, md_key: str) -> str:
        return _flat_metadata_key_to_field_identifier(md_key)

    @override
    def encode_vectorize_sort(self, query_text: str) -> dict[str, Any]:
        return _astra_generic_encode_vectorize_sort(query_text)

    @override
    def encode_hybrid_sort(
        self,
        *,
        vector: list[float] | None,
        vectorize: str | None,
        lexical: str,
    ) -> dict[str, Any]:
        if vectorize is None or vector is not None:
            msg = "This codec's hybrid sort requires `vector=None` and a vectorize."
            raise ValueError(msg)
        return _astra_vectorize_encode_hybrid_sort(vectorize=vectorize, lexical=lexical)

    @property
    @override
    def default_collection_indexing_policy(self) -> dict[str, list[str]]:
        # $vectorize cannot be de-indexed explicitly (the API manages it entirely).
        return {}
