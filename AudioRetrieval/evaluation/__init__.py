"""
Evaluation Module for Audio Retrieval.

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

from AudioRetrieval.evaluation.runners import (
    BaselineRunner,
    UIQRunner,
    NegativeQueryRunner,
    PrecomputedEmbeddingRunner,
)

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
    # Runners
    "BaselineRunner",
    "UIQRunner",
    "NegativeQueryRunner",
    "PrecomputedEmbeddingRunner",
]
