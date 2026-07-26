"""Query-local score normalization with explicit tie semantics."""

from __future__ import annotations

import math
from numbers import Real
from typing import Hashable, Iterable, List, Sequence, Tuple, TypeVar

ItemId = TypeVar("ItemId", bound=Hashable)


def _finite_scores(scores: Iterable[Real]) -> List[float]:
    values: List[float] = []
    for index, score in enumerate(scores):
        if isinstance(score, bool) or not isinstance(score, Real):
            raise TypeError(f"score[{index}] must be a real number, got {type(score).__name__}")
        value = float(score)
        if not math.isfinite(value):
            raise ValueError(f"score[{index}] must be finite, got {value!r}")
        values.append(value)
    if not values:
        raise ValueError("scores must not be empty")
    return values


def zscore_scores(scores: Iterable[Real], *, epsilon: float = 1e-12) -> List[float]:
    """Return population z-scores for one query's candidate scores.

    If the population standard deviation is at most ``epsilon``, every
    normalized score is defined as zero.  This makes a degenerate score route
    neutral instead of amplifying floating-point noise.
    """

    if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
        raise TypeError("epsilon must be a real number")
    epsilon_value = float(epsilon)
    if not math.isfinite(epsilon_value) or epsilon_value < 0.0:
        raise ValueError("epsilon must be finite and non-negative")

    values = _finite_scores(scores)
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(max(variance, 0.0))
    if std <= epsilon_value:
        return [0.0] * len(values)
    return [(value - mean) / std for value in values]


def rank_normalize_scores(scores: Iterable[Real]) -> List[float]:
    """Map descending average ranks to ``[0, 1]`` within one query.

    The highest score maps to 1 and the lowest to 0. Exact score ties receive
    the same average-rank value, so the result is invariant to input order.
    A singleton candidate set maps to 1.
    """

    values = _finite_scores(scores)
    if len(values) == 1:
        return [1.0]

    indexed = sorted(enumerate(values), key=lambda pair: (-pair[1], pair[0]))
    average_ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        # Ranks are one-indexed. The tied block occupies start+1 through end.
        average_rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            original_index = indexed[position][0]
            average_ranks[original_index] = average_rank
        start = end

    denominator = len(values) - 1
    return [(len(values) - rank) / denominator for rank in average_ranks]


def sort_scored_items(
    items: Sequence[Tuple[ItemId, Real]],
) -> List[Tuple[ItemId, float]]:
    """Sort by descending finite score and a deterministic item-ID tie break.

    IDs are compared through ``(type-name, repr)``.  This supports common
    string/integer IDs without relying on otherwise invalid mixed-type
    comparisons. Duplicate IDs are rejected because a retrieval ranking must
    contain each candidate at most once.
    """

    seen = set()
    validated: List[Tuple[ItemId, float]] = []
    for index, (item_id, score) in enumerate(items):
        if item_id in seen:
            raise ValueError(f"duplicate item ID at position {index}: {item_id!r}")
        seen.add(item_id)
        value = _finite_scores([score])[0]
        validated.append((item_id, value))
    if not validated:
        raise ValueError("items must not be empty")
    return sorted(
        validated,
        key=lambda pair: (-pair[1], type(pair[0]).__name__, repr(pair[0])),
    )
