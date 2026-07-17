"""Evaluation Module for Audio Retrieval.

This module provides tools for evaluating audio-text retrieval models:

1. Core evaluation logic (metrics computation)
2. Baseline retrieval evaluation
3. UIQ (User Intent Query) evaluation
4. Negative query evaluation
5. Precomputed embedding evaluation

Submodules:
    - metrics: Recall, MRR, DCG computations
    - runners: Different evaluation runners

Example usage:
    from AudioRetrieval.evaluation import run_baseline_evaluation
    from AudioRetrieval.evaluation.metrics import compute_recall_at_k

    results = run_baseline_evaluation(
        model="laion_clap",
        dataset="clotho",
        split="eval",
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from AudioRetrieval.evaluation.metrics import (
    compute_recall_at_k,
    compute_mrr,
    compute_dcg,
    compute_rsum,
    l2norm,
    cosine_sim,
)
from AudioRetrieval.evaluation.negative_metrics import (
    compute_delta_rank,
    compute_hnsr,
    compute_hnsr_at_k,
    compute_negative_query_metrics,
    compute_tfr,
    compute_tfr_hn_at_k,
)
from AudioRetrieval.evaluation.canonical import (
    CanonicalRetrievalResult,
    evaluate_caption_to_caption,
    evaluate_id_retrieval,
    evaluate_query_to_candidates,
)
from AudioRetrieval.evaluation.negative_canonical import (
    CanonicalNegativeRetrievalResult,
    evaluate_negative_id_retrieval,
)
from AudioRetrieval.evaluation.uiq_schema import (
    ReleasedUIQQuery,
    load_released_uiq,
    released_uiq_summary,
)

if TYPE_CHECKING:
    from AudioRetrieval.evaluation.runners import (
        BaselineRunner,
        NegativeQueryRunner,
        PrecomputedEmbeddingRunner,
        UIQRunner,
    )


_RUNNER_EXPORTS = {
    "BaselineRunner",
    "UIQRunner",
    "NegativeQueryRunner",
    "PrecomputedEmbeddingRunner",
}


def __getattr__(name: str):
    """Load optional runner dependencies only when a runner is requested."""

    if name in _RUNNER_EXPORTS:
        from AudioRetrieval.evaluation import runners

        return getattr(runners, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Metrics
    "compute_recall_at_k",
    "compute_mrr",
    "compute_dcg",
    "compute_rsum",
    "l2norm",
    "cosine_sim",
    "compute_delta_rank",
    "compute_hnsr",
    "compute_hnsr_at_k",
    "compute_negative_query_metrics",
    "compute_tfr",
    "compute_tfr_hn_at_k",
    "CanonicalRetrievalResult",
    "evaluate_caption_to_caption",
    "evaluate_id_retrieval",
    "evaluate_query_to_candidates",
    "CanonicalNegativeRetrievalResult",
    "evaluate_negative_id_retrieval",
    "ReleasedUIQQuery",
    "load_released_uiq",
    "released_uiq_summary",
    # Runners
    "BaselineRunner",
    "UIQRunner",
    "NegativeQueryRunner",
    "PrecomputedEmbeddingRunner",
]
