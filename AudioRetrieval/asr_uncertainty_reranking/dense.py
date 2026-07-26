"""Dependency-light exact dense retrieval over cached embedding arrays."""

from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np


def l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("embedding array must be a non-empty two-dimensional matrix")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("embedding array must use a floating dtype")
    converted = np.asarray(array, dtype=np.float32)
    if not np.isfinite(converted).all():
        raise ValueError("embedding array must be finite")
    norms = np.linalg.norm(converted, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("zero-norm embeddings are not allowed")
    return converted / norms


def exact_chunked_topk(
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    *,
    query_ids: Sequence[str],
    document_ids: Sequence[str],
    k: int,
    query_batch_size: int = 32,
    document_chunk_size: int = 16384,
    require_normalized: bool = True,
) -> Dict[str, Dict[str, list]]:
    """Return exact inner-product Top-K with deterministic document-ID ties.

    The implementation never materializes the full query-by-corpus score
    matrix.  It is intended for CPU correctness/smoke runs and cached-array
    validation.  Large NQ execution may use a separately audited GPU/FAISS
    backend, but must match this exact reference on a shared smoke subset.
    """

    queries = np.asarray(query_embeddings, dtype=np.float32)
    documents = np.asarray(document_embeddings, dtype=np.float32)
    if queries.ndim != 2 or documents.ndim != 2:
        raise ValueError("query and document embeddings must be two-dimensional")
    if queries.shape[1] != documents.shape[1]:
        raise ValueError("query/document embedding dimensions differ")
    if queries.shape[0] != len(query_ids) or documents.shape[0] != len(document_ids):
        raise ValueError("embedding rows and ID counts differ")
    if len(set(query_ids)) != len(query_ids) or len(set(document_ids)) != len(document_ids):
        raise ValueError("query_ids and document_ids must each be unique")
    if any(not isinstance(value, str) or not value for value in query_ids):
        raise ValueError("query IDs must be non-empty strings")
    if any(not isinstance(value, str) or not value for value in document_ids):
        raise ValueError("document IDs must be non-empty strings")
    if not np.isfinite(queries).all() or not np.isfinite(documents).all():
        raise ValueError("embeddings must be finite")
    if (
        isinstance(k, bool)
        or not isinstance(k, int)
        or k <= 0
        or k > len(document_ids)
    ):
        raise ValueError("k must be a positive integer no larger than the corpus")
    if query_batch_size <= 0 or document_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if require_normalized:
        for label, values in (("query", queries), ("document", documents)):
            norms = np.linalg.norm(values, axis=1)
            if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
                raise ValueError(f"{label} embeddings are not L2 normalized")

    result: Dict[str, Dict[str, list]] = {}
    for query_start in range(0, len(query_ids), query_batch_size):
        query_stop = min(query_start + query_batch_size, len(query_ids))
        batch = queries[query_start:query_stop]
        running = [[] for _ in range(len(batch))]
        for document_start in range(0, len(document_ids), document_chunk_size):
            document_stop = min(
                document_start + document_chunk_size,
                len(document_ids),
            )
            chunk_scores = batch @ documents[document_start:document_stop].T
            local_k = min(k, document_stop - document_start)
            for row_index in range(len(batch)):
                scores = chunk_scores[row_index]
                if local_k == len(scores):
                    local_indices = range(len(scores))
                else:
                    local_indices = np.argpartition(scores, -local_k)[-local_k:]
                candidates = running[row_index] + [
                    (
                        float(scores[local_index]),
                        document_ids[document_start + int(local_index)],
                    )
                    for local_index in local_indices
                ]
                running[row_index] = sorted(
                    candidates,
                    key=lambda pair: (-pair[0], pair[1]),
                )[:k]
        for local_index, pairs in enumerate(running):
            query_id = query_ids[query_start + local_index]
            if len(pairs) != k or any(not math.isfinite(pair[0]) for pair in pairs):
                raise AssertionError("exact Top-K retrieval produced an invalid result")
            result[query_id] = {
                "candidate_ids": [pair[1] for pair in pairs],
                "scores": [pair[0] for pair in pairs],
            }
    return result
