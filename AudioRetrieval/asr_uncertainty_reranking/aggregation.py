"""ASR N-best proxy posteriors, uncertainty features, and score aggregation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import combinations
from numbers import Real
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _finite_float(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def _positive_temperature(value: Real, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def logsumexp(values: Iterable[Real]) -> float:
    """Compute log-sum-exp stably, allowing negative infinity terms."""

    converted: List[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"value[{index}] must be a real number")
        converted_value = float(value)
        if math.isnan(converted_value) or converted_value == math.inf:
            raise ValueError(f"value[{index}] must be finite or -inf")
        converted.append(converted_value)
    if not converted:
        raise ValueError("values must not be empty")
    maximum = max(converted)
    if maximum == -math.inf:
        return -math.inf
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in converted))


def softmax_proxy_posteriors(
    proxy_logits: Iterable[Real],
    *,
    temperature: Real = 1.0,
) -> List[float]:
    """Convert reproducible ASR proxy logits to normalized proxy posteriors.

    These values are *not* claimed to be calibrated acoustic-model posterior
    probabilities.  In the planned Whisper protocol, each proxy logit is the
    mean valid-token log probability for one returned hypothesis.
    """

    temperature_value = _positive_temperature(temperature, name="temperature")
    logits = [
        _finite_float(value, name=f"proxy_logits[{index}]")
        for index, value in enumerate(proxy_logits)
    ]
    if not logits:
        raise ValueError("proxy_logits must not be empty")
    scaled = [value / temperature_value for value in logits]
    normalizer = logsumexp(scaled)
    probabilities = [math.exp(value - normalizer) for value in scaled]
    # Divide by fsum once more to remove the last few ulps of accumulated drift.
    total = math.fsum(probabilities)
    return [probability / total for probability in probabilities]


def _validated_probabilities(probabilities: Iterable[Real]) -> List[float]:
    values: List[float] = []
    for index, probability in enumerate(probabilities):
        value = _finite_float(probability, name=f"probabilities[{index}]")
        if value < 0.0 or value > 1.0:
            raise ValueError(f"probabilities[{index}] must be in [0, 1]")
        values.append(value)
    if not values:
        raise ValueError("probabilities must not be empty")
    total = math.fsum(values)
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"probabilities must sum to 1, got {total!r}")
    if total <= 0.0:
        raise ValueError("at least one probability must be positive")
    return [value / total for value in values]


def normalized_entropy(probabilities: Iterable[Real]) -> float:
    """Return entropy divided by ``log(M)`` for ``M`` hypotheses."""

    values = _validated_probabilities(probabilities)
    if len(values) == 1:
        return 0.0
    entropy = -math.fsum(value * math.log(value) for value in values if value > 0.0)
    return entropy / math.log(len(values))


def word_levenshtein_distance(reference: str, hypothesis: str) -> int:
    """Return case-folded, whitespace-tokenized word edit distance."""

    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        raise TypeError("reference and hypothesis must be strings")
    source = reference.casefold().split()
    target = hypothesis.casefold().split()
    previous = list(range(len(target) + 1))
    for source_index, source_token in enumerate(source, start=1):
        current = [source_index]
        for target_index, target_token in enumerate(target, start=1):
            substitution_cost = 0 if source_token == target_token else 1
            current.append(
                min(
                    previous[target_index] + 1,
                    current[target_index - 1] + 1,
                    previous[target_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def normalized_word_edit_distance(left: str, right: str) -> float:
    """Return word edit distance divided by the longer token sequence."""

    denominator = max(len(left.casefold().split()), len(right.casefold().split()))
    if denominator == 0:
        return 0.0
    return word_levenshtein_distance(left, right) / denominator


def pairwise_edit_distance_stats(hypotheses: Sequence[str]) -> Tuple[float, float]:
    """Return mean and maximum normalized distance over all N-best pairs."""

    if not hypotheses:
        raise ValueError("hypotheses must not be empty")
    if any(not isinstance(hypothesis, str) for hypothesis in hypotheses):
        raise TypeError("every hypothesis must be a string")
    if len(hypotheses) == 1:
        return 0.0, 0.0
    distances = [
        normalized_word_edit_distance(left, right)
        for left, right in combinations(hypotheses, 2)
    ]
    return math.fsum(distances) / len(distances), max(distances)


@dataclass(frozen=True)
class ASRUncertaintyFeatures:
    """Query-level ASR uncertainty features used by later gate modules."""

    nbest_size: int
    top1_average_token_logprob: float
    normalized_nbest_entropy: float
    top1_top2_proxy_margin: float
    hypothesis_edit_distance_mean: float
    hypothesis_edit_distance_max: float
    no_speech_probability: Optional[float]
    no_speech_probability_missing: bool

    def as_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable feature mapping."""

        return asdict(self)


def build_asr_uncertainty_features(
    hypotheses: Sequence[str],
    proxy_logits: Sequence[Real],
    *,
    proxy_temperature: Real = 1.0,
    top1_average_token_logprob: Optional[Real] = None,
    no_speech_probability: Optional[Real] = None,
) -> ASRUncertaintyFeatures:
    """Build deterministic uncertainty features from one query's N-best list."""

    if not hypotheses:
        raise ValueError("hypotheses must not be empty")
    if len(hypotheses) != len(proxy_logits):
        raise ValueError("hypotheses and proxy_logits must have the same length")
    if any(not isinstance(hypothesis, str) for hypothesis in hypotheses):
        raise TypeError("every hypothesis must be a string")

    posteriors = softmax_proxy_posteriors(
        proxy_logits,
        temperature=proxy_temperature,
    )
    sorted_posteriors = sorted(posteriors, reverse=True)
    margin = (
        sorted_posteriors[0] - sorted_posteriors[1]
        if len(sorted_posteriors) > 1
        else 1.0
    )
    edit_mean, edit_max = pairwise_edit_distance_stats(hypotheses)
    top1_logprob = _finite_float(
        proxy_logits[0]
        if top1_average_token_logprob is None
        else top1_average_token_logprob,
        name="top1_average_token_logprob",
    )

    if no_speech_probability is None:
        no_speech = None
        missing = True
    else:
        no_speech = _finite_float(
            no_speech_probability,
            name="no_speech_probability",
        )
        if not 0.0 <= no_speech <= 1.0:
            raise ValueError("no_speech_probability must be in [0, 1]")
        missing = False

    return ASRUncertaintyFeatures(
        nbest_size=len(hypotheses),
        top1_average_token_logprob=top1_logprob,
        normalized_nbest_entropy=normalized_entropy(posteriors),
        top1_top2_proxy_margin=margin,
        hypothesis_edit_distance_mean=edit_mean,
        hypothesis_edit_distance_max=edit_max,
        no_speech_probability=no_speech,
        no_speech_probability_missing=missing,
    )


def _validated_score_matrix(scores: Sequence[Sequence[Real]]) -> List[List[float]]:
    if not scores:
        raise ValueError("scores must contain at least one hypothesis")
    candidate_count = len(scores[0])
    if candidate_count == 0:
        raise ValueError("scores must contain at least one candidate")
    matrix: List[List[float]] = []
    for row_index, row in enumerate(scores):
        if len(row) != candidate_count:
            raise ValueError("scores must be a rectangular hypothesis-by-candidate matrix")
        matrix.append(
            [
                _finite_float(value, name=f"scores[{row_index}][{column_index}]")
                for column_index, value in enumerate(row)
            ]
        )
    return matrix


def aggregate_nbest_scores(
    scores: Sequence[Sequence[Real]],
    *,
    mode: str,
    proxy_posteriors: Optional[Sequence[Real]] = None,
    cross_encoder_temperature: Real = 1.0,
) -> List[float]:
    """Aggregate a hypothesis-by-candidate Cross-Encoder score matrix.

    Supported modes:

    - ``one_best``: use the first hypothesis;
    - ``equal_mean``: arithmetic mean over hypotheses;
    - ``max``: maximum score over hypotheses;
    - ``proxy_posterior_logsumexp``:
      ``logsumexp(log(p_m) + score_mi / T_ce)``.
    """

    matrix = _validated_score_matrix(scores)
    if mode == "one_best":
        return list(matrix[0])
    if mode == "equal_mean":
        return [
            math.fsum(row[candidate] for row in matrix) / len(matrix)
            for candidate in range(len(matrix[0]))
        ]
    if mode == "max":
        return [
            max(row[candidate] for row in matrix)
            for candidate in range(len(matrix[0]))
        ]
    if mode != "proxy_posterior_logsumexp":
        raise ValueError(
            "mode must be one of: one_best, equal_mean, max, "
            "proxy_posterior_logsumexp"
        )

    if proxy_posteriors is None:
        raise ValueError("proxy_posteriors are required for proxy-posterior aggregation")
    probabilities = _validated_probabilities(proxy_posteriors)
    if len(probabilities) != len(matrix):
        raise ValueError("proxy_posteriors length must match the number of hypotheses")
    temperature = _positive_temperature(
        cross_encoder_temperature,
        name="cross_encoder_temperature",
    )
    log_probabilities = [
        math.log(probability) if probability > 0.0 else -math.inf
        for probability in probabilities
    ]
    return [
        logsumexp(
            log_probabilities[hypothesis] + matrix[hypothesis][candidate] / temperature
            for hypothesis in range(len(matrix))
        )
        for candidate in range(len(matrix[0]))
    ]
