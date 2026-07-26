"""Deterministic frozen-candidate generation and score-fusion baselines."""

from __future__ import annotations

import math
from numbers import Real
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .normalization import sort_scored_items


def _score_map(
    candidate_ids: Sequence[str],
    scores: Sequence[Real],
    *,
    label: str,
) -> Dict[str, float]:
    if len(candidate_ids) != len(scores) or not candidate_ids:
        raise ValueError(f"{label}: candidate IDs and scores must have equal non-zero length")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError(f"{label}: candidate IDs must be unique")
    result: Dict[str, float] = {}
    for candidate_id, raw_score in zip(candidate_ids, scores):
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f"{label}: candidate IDs must be non-empty strings")
        if isinstance(raw_score, bool) or not isinstance(raw_score, Real):
            raise TypeError(f"{label}: scores must be real numbers")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError(f"{label}: scores must be finite")
        result[candidate_id] = score
    return result


def rank_scores(
    candidate_ids: Sequence[str],
    scores: Sequence[Real],
) -> List[str]:
    score_by_id = _score_map(candidate_ids, scores, label="rank_scores")
    return [
        candidate_id
        for candidate_id, _ in sort_scored_items(list(score_by_id.items()))
    ]


def build_top_k(
    scores_by_query: Mapping[str, Mapping[str, Real]],
    *,
    k: int,
) -> Dict[str, List[Tuple[str, float]]]:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    result: Dict[str, List[Tuple[str, float]]] = {}
    for query_id, scores in scores_by_query.items():
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("query IDs must be non-empty strings")
        ranking = sort_scored_items(list(scores.items()))
        result[query_id] = ranking[: min(k, len(ranking))]
    if not result:
        raise ValueError("scores_by_query must not be empty")
    return result


def validate_fixed_candidate_sets(
    reference: Mapping[str, Sequence[str]],
    comparison: Mapping[str, Sequence[str]],
    *,
    require_order: bool = True,
) -> None:
    if set(reference) != set(comparison):
        raise ValueError("candidate query sets differ")
    for query_id in reference:
        expected = list(reference[query_id])
        actual = list(comparison[query_id])
        if len(expected) != len(set(expected)) or len(actual) != len(set(actual)):
            raise ValueError(f"duplicate candidate for query {query_id!r}")
        if require_order:
            if actual != expected:
                raise ValueError(f"candidate order differs for query {query_id!r}")
        elif set(actual) != set(expected):
            raise ValueError(f"candidate membership differs for query {query_id!r}")


def fixed_weight_fusion(
    candidate_ids: Sequence[str],
    left_scores: Sequence[Real],
    right_scores: Sequence[Real],
    *,
    right_weight: Real,
) -> List[float]:
    left = _score_map(candidate_ids, left_scores, label="left_scores")
    right = _score_map(candidate_ids, right_scores, label="right_scores")
    if isinstance(right_weight, bool) or not isinstance(right_weight, Real):
        raise TypeError("right_weight must be a real number")
    weight = float(right_weight)
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("right_weight must be in [0, 1]")
    return [
        (1.0 - weight) * left[candidate_id] + weight * right[candidate_id]
        for candidate_id in candidate_ids
    ]


def reciprocal_rank_fusion(
    candidate_ids: Sequence[str],
    left_scores: Sequence[Real],
    right_scores: Sequence[Real],
    *,
    rank_constant: Real = 60.0,
) -> List[float]:
    _score_map(candidate_ids, left_scores, label="left_scores")
    _score_map(candidate_ids, right_scores, label="right_scores")
    if isinstance(rank_constant, bool) or not isinstance(rank_constant, Real):
        raise TypeError("rank_constant must be a real number")
    constant = float(rank_constant)
    if not math.isfinite(constant) or constant <= 0.0:
        raise ValueError("rank_constant must be positive")
    left_rank = {
        candidate_id: rank
        for rank, candidate_id in enumerate(
            rank_scores(candidate_ids, left_scores),
            start=1,
        )
    }
    right_rank = {
        candidate_id: rank
        for rank, candidate_id in enumerate(
            rank_scores(candidate_ids, right_scores),
            start=1,
        )
    }
    return [
        1.0 / (constant + left_rank[candidate_id])
        + 1.0 / (constant + right_rank[candidate_id])
        for candidate_id in candidate_ids
    ]


def top_k_overlap(
    left_ranking: Sequence[str],
    right_ranking: Sequence[str],
    *,
    k: int,
    mode: str,
) -> float:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if len(set(left_ranking)) != len(left_ranking):
        raise ValueError("left_ranking contains duplicates")
    if len(set(right_ranking)) != len(right_ranking):
        raise ValueError("right_ranking contains duplicates")
    left = set(left_ranking[:k])
    right = set(right_ranking[:k])
    intersection = len(left.intersection(right))
    if mode == "overlap_coefficient":
        denominator = min(len(left), len(right))
    elif mode == "jaccard":
        denominator = len(left.union(right))
    else:
        raise ValueError("mode must be overlap_coefficient or jaccard")
    return intersection / denominator if denominator else 0.0
