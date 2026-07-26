"""Seed, condition, WER-bin, and gate-weight summaries."""

from __future__ import annotations

import math
from numbers import Real
from typing import Dict, Mapping, Sequence


def summarize_seeds(values: Sequence[Real]) -> Dict[str, float]:
    if not values:
        raise ValueError("values must not be empty")
    converted = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("seed metrics must be real numbers")
        converted_value = float(value)
        if not math.isfinite(converted_value):
            raise ValueError("seed metrics must be finite")
        converted.append(converted_value)
    mean = math.fsum(converted) / len(converted)
    if len(converted) == 1:
        standard_deviation = 0.0
        half_width = 0.0
    else:
        variance = math.fsum((value - mean) ** 2 for value in converted) / (
            len(converted) - 1
        )
        standard_deviation = math.sqrt(variance)
        # Exact two-sided 95% t critical values for the preregistered 1/2/3
        # seed cases; normal approximation beyond that is sufficient here.
        critical = {2: 12.706, 3: 4.303}.get(len(converted), 1.96)
        half_width = critical * standard_deviation / math.sqrt(len(converted))
    return {
        "count": float(len(converted)),
        "mean": mean,
        "sample_standard_deviation": standard_deviation,
        "confidence_95_lower": mean - half_width,
        "confidence_95_upper": mean + half_width,
    }


def summarize_gate_by_condition(
    gates: Mapping[str, Sequence[Real]],
) -> Dict[str, Dict[str, float]]:
    result = {}
    for condition, values in sorted(gates.items()):
        summary = summarize_seeds(values)
        result[condition] = {
            "count": summary["count"],
            "mean": summary["mean"],
            "sample_standard_deviation": summary["sample_standard_deviation"],
        }
    return result


def summarize_metric_by_wer_bins(
    wer_by_query: Mapping[str, Real],
    metric_by_query: Mapping[str, Real],
    *,
    boundaries: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 1.0, math.inf),
) -> list[Dict[str, object]]:
    if set(wer_by_query) != set(metric_by_query):
        raise ValueError("WER and metric query sets must match exactly")
    if len(boundaries) < 2 or any(
        boundaries[index] >= boundaries[index + 1]
        for index in range(len(boundaries) - 1)
    ):
        raise ValueError("boundaries must be strictly increasing")
    buckets = [[] for _ in range(len(boundaries) - 1)]
    for query_id in sorted(wer_by_query):
        wer = float(wer_by_query[query_id])
        metric = float(metric_by_query[query_id])
        if not math.isfinite(wer) or wer < 0.0 or not math.isfinite(metric):
            raise ValueError("WER must be finite/non-negative and metric finite")
        for index, (lower, upper) in enumerate(zip(boundaries, boundaries[1:])):
            if lower <= wer < upper or (
                index == len(buckets) - 1 and wer == upper
            ):
                buckets[index].append(metric)
                break
    return [
        {
            "lower_inclusive": lower,
            "upper_exclusive": upper,
            "count": len(values),
            "metric_mean": math.fsum(values) / len(values) if values else None,
        }
        for lower, upper, values in zip(boundaries, boundaries[1:], buckets)
    ]
