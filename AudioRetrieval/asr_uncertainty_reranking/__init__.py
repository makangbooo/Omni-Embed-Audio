"""Core utilities for ASR-uncertainty-guided reranking.

The package is deliberately split into dependency-light CPU components and
model adapters.  The CPU components can be audited and tested before any
model, dataset, GPU inference, or training is authorized.
"""

from .aggregation import (
    ASRUncertaintyFeatures,
    aggregate_nbest_scores,
    build_asr_uncertainty_features,
    normalized_entropy,
    softmax_proxy_posteriors,
)
from .cache_manifest import (
    CacheManifestMismatchError,
    assert_cache_compatible,
    build_cache_manifest,
    manifest_fingerprint,
)
from .normalization import (
    rank_normalize_scores,
    sort_scored_items,
    zscore_scores,
)

__all__ = [
    "ASRUncertaintyFeatures",
    "CacheManifestMismatchError",
    "aggregate_nbest_scores",
    "assert_cache_compatible",
    "build_asr_uncertainty_features",
    "build_cache_manifest",
    "manifest_fingerprint",
    "normalized_entropy",
    "rank_normalize_scores",
    "softmax_proxy_posteriors",
    "sort_scored_items",
    "zscore_scores",
]
