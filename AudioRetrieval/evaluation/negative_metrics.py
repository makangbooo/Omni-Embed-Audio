"""Paper-defined metrics for exclusionary audio-retrieval queries.

All ranks are one-based and all rate metrics are returned as percentages in
the range [0, 100], matching the presentation in paper Table 17.  The paper's
Appendix L and Figure 3 call the strict target-first metric ``TFR-HN@k``;
Table 17 shortens that same column to ``TFR@k``.  This module uses the explicit
``TFR-HN@k`` name so it cannot be confused with the unconditioned ``TFR``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Dict, Tuple

import numpy as np


def _validated_ranks(ranks: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(ranks)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional rank array")
    if values.size == 0:
        raise ValueError(f"{name} must not be empty")
    if np.issubdtype(values.dtype, np.bool_):
        raise ValueError(f"{name} must contain positive integer ranks, not booleans")
    try:
        numeric = values.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric ranks") from exc
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} contains a non-finite rank")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} must contain integer-valued ranks")
    if np.any(numeric < 1):
        raise ValueError(f"{name} must contain one-based positive ranks")
    return numeric.astype(np.int64)


def _validated_rank_pair(
    target_ranks: Sequence[int] | np.ndarray,
    hard_negative_ranks: Sequence[int] | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    target = _validated_ranks(target_ranks, "target_ranks")
    hard_negative = _validated_ranks(hard_negative_ranks, "hard_negative_ranks")
    if target.shape != hard_negative.shape:
        raise ValueError(
            "target_ranks and hard_negative_ranks must contain the same number "
            "of queries"
        )
    return target, hard_negative


def _validated_k(k: int) -> int:
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be a positive integer")
    if int(k) < 1:
        raise ValueError("k must be a positive integer")
    return int(k)


def _percentage(condition: np.ndarray) -> float:
    return float(np.mean(condition) * 100.0)


def compute_delta_rank(
    target_ranks: Sequence[int] | np.ndarray,
    hard_negative_ranks: Sequence[int] | np.ndarray,
) -> float:
    """Return mean Rank(HN) - Rank(T); higher is better. [PAPER, Appendix L]"""

    target, hard_negative = _validated_rank_pair(target_ranks, hard_negative_ranks)
    return float(np.mean(hard_negative.astype(np.float64) - target))


def compute_hnsr(
    target_ranks: Sequence[int] | np.ndarray,
    hard_negative_ranks: Sequence[int] | np.ndarray,
) -> float:
    """Return percentage with Rank(HN) > Rank(T). [PAPER, Appendix L]"""

    target, hard_negative = _validated_rank_pair(target_ranks, hard_negative_ranks)
    return _percentage(hard_negative > target)


def compute_hnsr_at_k(
    target_ranks: Sequence[int] | np.ndarray,
    hard_negative_ranks: Sequence[int] | np.ndarray,
    k: int,
) -> float:
    """Return percentage with target in top-k and HN outside top-k. [PAPER, §3.3]"""

    target, hard_negative = _validated_rank_pair(target_ranks, hard_negative_ranks)
    cutoff = _validated_k(k)
    return _percentage((target <= cutoff) & (hard_negative > cutoff))


def compute_tfr(target_ranks: Sequence[int] | np.ndarray) -> float:
    """Return percentage where the target is ranked first. [PAPER, Appendix L]"""

    target = _validated_ranks(target_ranks, "target_ranks")
    return _percentage(target == 1)


def compute_tfr_hn_at_k(
    target_ranks: Sequence[int] | np.ndarray,
    hard_negative_ranks: Sequence[int] | np.ndarray,
    k: int,
) -> float:
    """Return percentage with target first and HN outside top-k. [PAPER, Appendix L]"""

    target, hard_negative = _validated_rank_pair(target_ranks, hard_negative_ranks)
    cutoff = _validated_k(k)
    return _percentage((target == 1) & (hard_negative > cutoff))


def compute_negative_query_metrics(
    target_ranks: Sequence[int] | np.ndarray,
    hard_negative_ranks: Sequence[int] | np.ndarray,
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, float | int]:
    """Compute ordinary retrieval and every paper-defined discrimination metric."""

    target, hard_negative = _validated_rank_pair(target_ranks, hard_negative_ranks)
    cutoffs = tuple(_validated_k(k) for k in ks)
    if not cutoffs:
        raise ValueError("ks must not be empty")
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("ks must not contain duplicate cutoffs")

    metrics: Dict[str, float | int] = {"num_queries": int(target.size)}
    for cutoff in cutoffs:
        metrics[f"R@{cutoff}"] = _percentage(target <= cutoff)
    metrics["Delta-Rank"] = float(
        np.mean(hard_negative.astype(np.float64) - target)
    )
    metrics["HNSR"] = _percentage(hard_negative > target)
    for cutoff in cutoffs:
        metrics[f"HNSR@{cutoff}"] = _percentage(
            (target <= cutoff) & (hard_negative > cutoff)
        )
    metrics["TFR"] = _percentage(target == 1)
    for cutoff in cutoffs:
        metrics[f"TFR-HN@{cutoff}"] = _percentage(
            (target == 1) & (hard_negative > cutoff)
        )
    return metrics
