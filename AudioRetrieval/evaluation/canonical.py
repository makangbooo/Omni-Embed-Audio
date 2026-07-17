"""Deterministic, embedding-only retrieval evaluation primitives.

The paper reports R@1/5/10 for text-to-audio (T2A), text-to-text (T2T),
and UIQ-to-audio retrieval. It does not specify caption query selection or
tie handling. Those choices therefore remain explicit inputs here rather
than being hidden behind random sampling.

The default optimistic tie policy matches the public repository's existing
``ranks_from_scores`` implementation: rank is one plus the number of scores
strictly greater than the best positive score. Every report produced from
this module records that policy so it cannot be mistaken for a paper-stated
choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from AudioRetrieval.evaluation.metrics import compute_all_metrics, l2norm


@dataclass(frozen=True)
class CanonicalRetrievalResult:
    """Auditable retrieval result for a fixed query and candidate set."""

    ranks: np.ndarray
    similarities: np.ndarray
    rankings: np.ndarray
    positive_indices: tuple[tuple[int, ...], ...]
    ignored_indices: tuple[tuple[int, ...], ...]
    evaluated_query_indices: np.ndarray
    metrics: dict[str, float]
    tie_policy: str = "optimistic_strict_greater"


def _as_embedding_matrix(name: str, value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix, got shape {matrix.shape}")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got shape {matrix.shape}")
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values, got {matrix.dtype}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return matrix.astype(np.float32, copy=False)


def _normalize_strict(name: str, matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1)
    zero_rows = np.flatnonzero(norms <= 1e-12)
    if zero_rows.size:
        preview = zero_rows[:10].tolist()
        raise ValueError(f"{name} contains zero-norm rows: {preview}")
    return l2norm(matrix)


def _validate_query_indices(
    query_indices: Sequence[int] | None,
    query_count: int,
) -> np.ndarray:
    if query_indices is None:
        return np.arange(query_count, dtype=np.int64)
    indices = np.asarray(query_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("query_indices must be a non-empty one-dimensional sequence")
    if np.any(indices < 0) or np.any(indices >= query_count):
        raise IndexError(
            f"query_indices must be within [0, {query_count}), got {indices.tolist()}"
        )
    if len(set(indices.tolist())) != indices.size:
        raise ValueError("query_indices must not contain duplicates")
    return indices


def _normalize_index_rows(
    name: str,
    rows: Sequence[Iterable[int]],
    *,
    row_count: int,
    candidate_count: int,
    allow_empty: bool,
) -> tuple[tuple[int, ...], ...]:
    if len(rows) != row_count:
        raise ValueError(f"{name} has {len(rows)} rows, expected {row_count}")
    normalized: list[tuple[int, ...]] = []
    for row_index, row in enumerate(rows):
        values = tuple(sorted(set(int(index) for index in row)))
        if not values and not allow_empty:
            raise ValueError(f"{name}[{row_index}] has no candidates")
        invalid = [index for index in values if index < 0 or index >= candidate_count]
        if invalid:
            raise IndexError(
                f"{name}[{row_index}] contains out-of-range candidates: {invalid}"
            )
        normalized.append(values)
    return tuple(normalized)


def evaluate_query_to_candidates(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    positive_indices: Sequence[Iterable[int]],
    *,
    ignored_indices: Sequence[Iterable[int]] | None = None,
    query_indices: Sequence[int] | None = None,
    normalize: bool = True,
) -> CanonicalRetrievalResult:
    """Evaluate a fixed query set against a fixed candidate set.

    Multiple positives are supported. The rank is the optimistic rank of the
    highest-scoring positive after ignored candidates are removed. Rankings
    use descending score and candidate index as a deterministic tie breaker;
    metric ranks retain the public code's strict-greater tie convention.
    """

    queries = _as_embedding_matrix("query_embeddings", query_embeddings)
    candidates = _as_embedding_matrix("candidate_embeddings", candidate_embeddings)
    if queries.shape[1] != candidates.shape[1]:
        raise ValueError(
            "query and candidate embedding dimensions differ: "
            f"{queries.shape[1]} != {candidates.shape[1]}"
        )
    if normalize:
        queries = _normalize_strict("query_embeddings", queries)
        candidates = _normalize_strict("candidate_embeddings", candidates)

    positives = _normalize_index_rows(
        "positive_indices",
        positive_indices,
        row_count=queries.shape[0],
        candidate_count=candidates.shape[0],
        allow_empty=False,
    )
    if ignored_indices is None:
        ignored = tuple(() for _ in range(queries.shape[0]))
    else:
        ignored = _normalize_index_rows(
            "ignored_indices",
            ignored_indices,
            row_count=queries.shape[0],
            candidate_count=candidates.shape[0],
            allow_empty=True,
        )

    selected_queries = _validate_query_indices(query_indices, queries.shape[0])
    similarities = queries[selected_queries] @ candidates.T
    ranks = np.empty(selected_queries.size, dtype=np.int64)
    rankings = np.empty(
        (selected_queries.size, candidates.shape[0]), dtype=np.int64
    )
    selected_positives: list[tuple[int, ...]] = []

    candidate_order = np.arange(candidates.shape[0], dtype=np.int64)
    for output_row, query_index in enumerate(selected_queries.tolist()):
        positive_row = positives[query_index]
        ignored_row = ignored[query_index]
        overlap = set(positive_row).intersection(ignored_row)
        if overlap:
            raise ValueError(
                f"query {query_index} marks positive candidates as ignored: "
                f"{sorted(overlap)}"
            )

        scores = similarities[output_row].copy()
        valid_mask = np.ones(candidates.shape[0], dtype=bool)
        if ignored_row:
            valid_mask[list(ignored_row)] = False
            scores[list(ignored_row)] = -np.inf

        best_positive_score = max(scores[index] for index in positive_row)
        ranks[output_row] = 1 + int(
            np.count_nonzero(scores[valid_mask] > best_positive_score)
        )
        rankings[output_row] = np.lexsort((candidate_order, -scores))
        selected_positives.append(positive_row)

    return CanonicalRetrievalResult(
        ranks=ranks,
        similarities=similarities,
        rankings=rankings,
        positive_indices=tuple(selected_positives),
        ignored_indices=tuple(ignored[index] for index in selected_queries),
        evaluated_query_indices=selected_queries,
        metrics=compute_all_metrics(ranks),
    )


def evaluate_id_retrieval(
    query_embeddings: np.ndarray,
    query_target_ids: Sequence[str],
    candidate_embeddings: np.ndarray,
    candidate_ids: Sequence[str],
    *,
    query_indices: Sequence[int] | None = None,
    normalize: bool = True,
) -> CanonicalRetrievalResult:
    """Evaluate T2A or UIQ retrieval using exact target/candidate IDs."""

    if len(query_target_ids) != np.asarray(query_embeddings).shape[0]:
        raise ValueError("query_target_ids length does not match query embeddings")
    if len(candidate_ids) != np.asarray(candidate_embeddings).shape[0]:
        raise ValueError("candidate_ids length does not match candidate embeddings")
    candidate_id_strings = [str(value) for value in candidate_ids]
    if len(set(candidate_id_strings)) != len(candidate_id_strings):
        raise ValueError("candidate_ids must be unique")
    candidate_lookup = {
        candidate_id: index for index, candidate_id in enumerate(candidate_id_strings)
    }
    missing = sorted(
        {
            str(target_id)
            for target_id in query_target_ids
            if str(target_id) not in candidate_lookup
        }
    )
    if missing:
        raise KeyError(f"query target IDs are absent from candidates: {missing[:10]}")
    positives = [
        (candidate_lookup[str(target_id)],) for target_id in query_target_ids
    ]
    return evaluate_query_to_candidates(
        query_embeddings,
        candidate_embeddings,
        positives,
        query_indices=query_indices,
        normalize=normalize,
    )


def evaluate_caption_to_caption(
    caption_embeddings: np.ndarray,
    caption_clip_ids: Sequence[str],
    *,
    query_indices: Sequence[int] | None = None,
    normalize: bool = True,
) -> CanonicalRetrievalResult:
    """Evaluate caption T2T retrieval with the query caption excluded.

    Every other caption belonging to the query's clip is a positive. Callers
    must explicitly choose ``query_indices`` when reproducing a one-caption-
    per-clip protocol; omitting it evaluates every caption deterministically.
    """

    embeddings = np.asarray(caption_embeddings)
    if len(caption_clip_ids) != embeddings.shape[0]:
        raise ValueError("caption_clip_ids length does not match caption embeddings")
    clip_to_indices: dict[str, list[int]] = {}
    for index, clip_id in enumerate(caption_clip_ids):
        clip_to_indices.setdefault(str(clip_id), []).append(index)

    positives: list[tuple[int, ...]] = []
    ignored: list[tuple[int, ...]] = []
    for index, clip_id in enumerate(caption_clip_ids):
        sister_indices = tuple(
            other
            for other in clip_to_indices[str(clip_id)]
            if other != index
        )
        if not sister_indices:
            raise ValueError(
                f"caption {index} for clip {clip_id!r} has no sister caption"
            )
        positives.append(sister_indices)
        ignored.append((index,))

    return evaluate_query_to_candidates(
        embeddings,
        embeddings,
        positives,
        ignored_indices=ignored,
        query_indices=query_indices,
        normalize=normalize,
    )
