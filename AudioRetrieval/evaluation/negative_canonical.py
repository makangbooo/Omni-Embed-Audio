"""Deterministic embedding evaluation for paired negative audio queries.

The paper's discrimination metrics require an explicit target audio and an
explicit hard-negative audio for every query.  This module deliberately takes
those IDs as inputs; it never mines, normalizes, or guesses a pairing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from AudioRetrieval.evaluation.canonical import evaluate_query_to_candidates
from AudioRetrieval.evaluation.negative_metrics import (
    compute_negative_query_metrics,
)


@dataclass(frozen=True)
class CanonicalNegativeRetrievalResult:
    """Complete evidence for a fixed target/hard-negative retrieval run."""

    target_ranks: np.ndarray
    hard_negative_ranks: np.ndarray
    similarities: np.ndarray
    rankings: np.ndarray
    target_indices: np.ndarray
    hard_negative_indices: np.ndarray
    evaluated_query_indices: np.ndarray
    metrics: dict[str, float | int]
    tie_policy: str = "optimistic_strict_greater"


def _validated_strings(
    name: str,
    values: Sequence[str],
    *,
    expected_count: int,
    unique: bool,
) -> list[str]:
    if len(values) != expected_count:
        raise ValueError(
            f"{name} length {len(values)} does not match expected count "
            f"{expected_count}"
        )
    normalized: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name}[{index}] must be a non-empty string")
        normalized.append(value.strip())
    if unique and len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must be unique")
    return normalized


def evaluate_negative_id_retrieval(
    query_embeddings: np.ndarray,
    query_ids: Sequence[str],
    candidate_embeddings: np.ndarray,
    candidate_ids: Sequence[str],
    target_ids: Sequence[str],
    hard_negative_ids: Sequence[str],
    *,
    query_indices: Sequence[int] | None = None,
    normalize: bool = True,
    ks: Sequence[int] = (1, 5, 10),
) -> CanonicalNegativeRetrievalResult:
    """Evaluate explicit query/target/hard-negative ID triples.

    Both target and hard-negative ranks use the public evaluator's optimistic
    tie convention: one plus the number of candidates with a strictly higher
    score.  ``rankings`` separately records the deterministic full ordering,
    where candidate index breaks score ties.
    """

    queries = np.asarray(query_embeddings)
    candidates = np.asarray(candidate_embeddings)
    if queries.ndim != 2:
        raise ValueError(
            f"query_embeddings must be a 2-D matrix, got shape {queries.shape}"
        )
    if candidates.ndim != 2:
        raise ValueError(
            "candidate_embeddings must be a 2-D matrix, "
            f"got shape {candidates.shape}"
        )

    _validated_strings(
        "query_ids", query_ids, expected_count=queries.shape[0], unique=True
    )
    candidate_id_values = _validated_strings(
        "candidate_ids",
        candidate_ids,
        expected_count=candidates.shape[0],
        unique=True,
    )
    target_id_values = _validated_strings(
        "target_ids", target_ids, expected_count=queries.shape[0], unique=False
    )
    hard_negative_id_values = _validated_strings(
        "hard_negative_ids",
        hard_negative_ids,
        expected_count=queries.shape[0],
        unique=False,
    )

    candidate_lookup = {
        candidate_id: index
        for index, candidate_id in enumerate(candidate_id_values)
    }
    missing_targets = sorted(
        {value for value in target_id_values if value not in candidate_lookup}
    )
    if missing_targets:
        raise KeyError(
            "target IDs are absent from candidates: "
            f"{missing_targets[:10]}"
        )
    missing_hard_negatives = sorted(
        {
            value
            for value in hard_negative_id_values
            if value not in candidate_lookup
        }
    )
    if missing_hard_negatives:
        raise KeyError(
            "hard-negative IDs are absent from candidates: "
            f"{missing_hard_negatives[:10]}"
        )

    identical_pairs = [
        index
        for index, (target_id, hard_negative_id) in enumerate(
            zip(target_id_values, hard_negative_id_values)
        )
        if target_id == hard_negative_id
    ]
    if identical_pairs:
        raise ValueError(
            "target_id and hard_negative_id must differ; query rows="
            f"{identical_pairs[:10]}"
        )

    all_target_indices = np.asarray(
        [candidate_lookup[value] for value in target_id_values], dtype=np.int64
    )
    all_hard_negative_indices = np.asarray(
        [candidate_lookup[value] for value in hard_negative_id_values],
        dtype=np.int64,
    )
    target_result = evaluate_query_to_candidates(
        queries,
        candidates,
        [(int(index),) for index in all_target_indices],
        query_indices=query_indices,
        normalize=normalize,
    )

    selected = target_result.evaluated_query_indices
    selected_targets = all_target_indices[selected]
    selected_hard_negatives = all_hard_negative_indices[selected]
    hard_negative_ranks = np.empty(selected.size, dtype=np.int64)
    for output_row, candidate_index in enumerate(
        selected_hard_negatives.tolist()
    ):
        scores = target_result.similarities[output_row]
        hard_negative_ranks[output_row] = 1 + int(
            np.count_nonzero(scores > scores[candidate_index])
        )

    return CanonicalNegativeRetrievalResult(
        target_ranks=target_result.ranks,
        hard_negative_ranks=hard_negative_ranks,
        similarities=target_result.similarities,
        rankings=target_result.rankings,
        target_indices=selected_targets,
        hard_negative_indices=selected_hard_negatives,
        evaluated_query_indices=selected,
        metrics=compute_negative_query_metrics(
            target_result.ranks, hard_negative_ranks, ks=ks
        ),
        tie_policy=target_result.tie_policy,
    )
