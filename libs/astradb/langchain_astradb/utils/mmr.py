"""Tools for the Maximal Marginal Relevance (MMR) reranking.

Duplicated from langchain_community to avoid cross-dependencies.

Functions "maximal_marginal_relevance" and "cosine_similarity"
are duplicated in this utility respectively from modules:
    - "libs/community/langchain_community/vectorstores/utils.py"
    - "libs/community/langchain_community/utils/math.py"
"""

from __future__ import annotations

import logging
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)

Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]


def cosine_similarity(x: Matrix, y: Matrix) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices.

    Args:
        x: First matrix.
        y: Second matrix.

    Returns:
        np.ndarray: Cosine similarity matrix.
    """
    if len(x) == 0 or len(y) == 0:
        return np.array([])

    x = np.array(x)
    y = np.array(y)
    if x.shape[1] != y.shape[1]:
        msg = (
            f"Number of columns in X and Y must be the same. X has shape {x.shape} "
            f"and Y has shape {y.shape}."
        )
        raise ValueError(msg)
    try:
        import simsimd as simd  # type: ignore[import]
    except ImportError:
        logger.info(
            "Unable to import simsimd, defaulting to NumPy implementation. If you want "
            "to use simsimd please install with `pip install simsimd`."
        )
        x_norm = np.linalg.norm(x, axis=1)
        y_norm = np.linalg.norm(y, axis=1)
        # Ignore divide by zero errors run time warnings as those are handled below.
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.dot(x, y.T) / np.outer(x_norm, y_norm)
        similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
        return similarity
    else:
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        z = 1 - np.array(simd.cdist(x, y, metric="cosine"))
        if isinstance(z, float):
            return np.array([z])
        return z


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: list,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> list[int]:
    """Calculate maximal marginal relevance.

    Args:
        query_embedding: Query embedding to compare.
        embedding_list: List of embeddings to select from.
        lambda_mult: Number between 0 and 1 that determines the degree
            of diversity among the results with 0 corresponding
            to maximum diversity and 1 to minimum diversity.
        k: Number of embeddings to select.

    Returns:
        List of indices of selected embeddings.
    """
    if min(k, len(embedding_list)) <= 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = (
                lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    return idxs
