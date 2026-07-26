"""Deterministic paired bootstrap significance for per-query metrics."""

from __future__ import annotations

import math
import random
from numbers import Real
from typing import Dict, Mapping, Sequence


def _finite_metric_mapping(
    values: Mapping[str, Real],
    *,
    label: str,
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for query_id, raw_value in values.items():
        if not isinstance(query_id, str) or not query_id:
            raise ValueError(f"{label}: query IDs must be non-empty strings")
        if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
            raise TypeError(f"{label}[{query_id!r}] must be a real number")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"{label}[{query_id!r}] must be finite")
        result[query_id] = value
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("sorted_values must not be empty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def paired_bootstrap(
    method: Mapping[str, Real],
    baseline: Mapping[str, Real],
    *,
    iterations: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, object]:
    """Bootstrap the paired mean delta ``method - baseline`` over queries."""

    method_values = _finite_metric_mapping(method, label="method")
    baseline_values = _finite_metric_mapping(baseline, label="baseline")
    if set(method_values) != set(baseline_values):
        raise ValueError("method and baseline query sets must match exactly")
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations <= 0
    ):
        raise ValueError("iterations must be a positive integer")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    query_ids = sorted(method_values)
    deltas = [
        method_values[query_id] - baseline_values[query_id]
        for query_id in query_ids
    ]
    observed = math.fsum(deltas) / len(deltas)
    generator = random.Random(seed)
    sampled_means = []
    for _ in range(iterations):
        sampled_means.append(
            math.fsum(
                deltas[generator.randrange(len(deltas))]
                for _ in range(len(deltas))
            )
            / len(deltas)
        )
    sampled_means.sort()
    alpha = 1.0 - confidence
    lower = _quantile(sampled_means, alpha / 2.0)
    upper = _quantile(sampled_means, 1.0 - alpha / 2.0)
    non_positive = sum(value <= 0.0 for value in sampled_means) / iterations
    non_negative = sum(value >= 0.0 for value in sampled_means) / iterations
    p_value = min(1.0, 2.0 * min(non_positive, non_negative))
    return {
        "num_queries": len(query_ids),
        "iterations": iterations,
        "seed": seed,
        "confidence": confidence,
        "mean_delta": observed,
        "confidence_interval": [lower, upper],
        "two_sided_p_value": p_value,
        "direction": "method_minus_baseline",
    }
