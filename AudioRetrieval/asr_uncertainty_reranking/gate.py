"""Deterministic small NumPy gate with multi-positive listwise training."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .features import select_feature_columns
from .schema import CandidateFeatureRow


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.exp(values - maximum).sum()))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def multi_positive_listwise_loss(
    scores: Sequence[float],
    relevances: Sequence[float],
) -> float:
    score_array = np.asarray(scores, dtype=np.float64)
    relevance_array = np.asarray(relevances, dtype=np.float64)
    if score_array.ndim != 1 or relevance_array.shape != score_array.shape:
        raise ValueError("scores and relevances must be equal-length one-dimensional arrays")
    if score_array.size == 0 or not np.isfinite(score_array).all():
        raise ValueError("scores must be non-empty and finite")
    if not np.isfinite(relevance_array).all() or np.any(relevance_array < 0.0):
        raise ValueError("relevances must be finite and non-negative")
    positive = relevance_array > 0.0
    if not positive.any():
        raise ValueError("at least one positive candidate is required")
    return _logsumexp(score_array) - _logsumexp(score_array[positive])


def blend_scores(
    oea_scores: Sequence[float],
    asr_scores: Sequence[float],
    gates: Sequence[float],
) -> list[float]:
    oea = np.asarray(oea_scores, dtype=np.float64)
    asr = np.asarray(asr_scores, dtype=np.float64)
    gate = np.asarray(gates, dtype=np.float64)
    if oea.ndim != 1 or asr.shape != oea.shape or gate.shape != oea.shape:
        raise ValueError("oea_scores, asr_scores, and gates must have equal 1-D shapes")
    if not np.isfinite(oea).all() or not np.isfinite(asr).all():
        raise ValueError("route scores must be finite")
    if not np.isfinite(gate).all() or np.any(gate < 0.0) or np.any(gate > 1.0):
        raise ValueError("gates must be finite and in [0, 1]")
    return ((1.0 - gate) * oea + gate * asr).tolist()


@dataclass(frozen=True)
class GateTrainingGroup:
    query_id: str
    rows: Tuple[CandidateFeatureRow, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id:
            raise ValueError("query_id must be a non-empty string")
        if not self.rows:
            raise ValueError("rows must not be empty")
        if any(row.query_id != self.query_id for row in self.rows):
            raise ValueError("every row must belong to query_id")
        schema = self.rows[0].feature_names
        if any(row.feature_names != schema for row in self.rows):
            raise ValueError("all rows must share one feature schema")
        if not any(row.relevance > 0.0 for row in self.rows):
            raise ValueError("group must contain at least one positive candidate")


def sample_group_candidates(
    rows: Sequence[CandidateFeatureRow],
    *,
    group_size: int = 16,
) -> Tuple[CandidateFeatureRow, ...]:
    """Keep all positives and deterministic high-ranked OEA negatives."""

    if (
        isinstance(group_size, bool)
        or not isinstance(group_size, int)
        or group_size <= 0
    ):
        raise ValueError("group_size must be a positive integer")
    if not rows:
        raise ValueError("rows must not be empty")
    positives = [row for row in rows if row.relevance > 0.0]
    if not positives:
        raise ValueError("rows must contain at least one positive")
    negatives = sorted(
        (row for row in rows if row.relevance <= 0.0),
        key=lambda row: (-row.oea_score, row.document_id),
    )
    selected_negatives = negatives[: max(group_size - len(positives), 0)]
    selected_ids = {
        row.document_id for row in positives + selected_negatives
    }
    return tuple(row for row in rows if row.document_id in selected_ids)


def build_training_groups(
    rows: Iterable[CandidateFeatureRow],
    *,
    group_size: Optional[int] = 16,
) -> Tuple[GateTrainingGroup, ...]:
    by_query: Dict[str, list[CandidateFeatureRow]] = {}
    for row in rows:
        by_query.setdefault(row.query_id, []).append(row)
    if not by_query:
        raise ValueError("rows must not be empty")
    groups = []
    for query_id in sorted(by_query):
        query_rows = tuple(by_query[query_id])
        if group_size is not None:
            query_rows = sample_group_candidates(query_rows, group_size=group_size)
        groups.append(GateTrainingGroup(query_id=query_id, rows=query_rows))
    return tuple(groups)


def retain_positive_candidate_queries(
    rows: Iterable[CandidateFeatureRow],
) -> tuple[list[CandidateFeatureRow], Tuple[str, ...]]:
    """Retain query groups whose frozen candidate set contains a positive.

    Queries without a positive candidate remain valid for retrieval evaluation,
    but multi-positive listwise gate training and validation losses are
    undefined for them.  The caller must persist the returned excluded IDs.
    """

    materialized = list(rows)
    if not materialized:
        raise ValueError("rows must not be empty")
    by_query: Dict[str, list[CandidateFeatureRow]] = {}
    for row in materialized:
        by_query.setdefault(row.query_id, []).append(row)
    excluded = tuple(
        query_id
        for query_id in sorted(by_query)
        if not any(row.relevance > 0.0 for row in by_query[query_id])
    )
    excluded_set = set(excluded)
    retained = [
        row for row in materialized if row.query_id not in excluded_set
    ]
    if not retained:
        raise ValueError("no query has a positive document in the frozen candidates")
    return retained, excluded


@dataclass
class CandidateGateMLP:
    feature_names: Tuple[str, ...]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    weight_input: np.ndarray
    bias_hidden: np.ndarray
    weight_output: np.ndarray
    bias_output: float
    seed: int

    def __post_init__(self) -> None:
        input_dim = len(self.feature_names)
        if input_dim == 0 or len(set(self.feature_names)) != input_dim:
            raise ValueError("feature_names must be unique and non-empty")
        if self.feature_mean.shape != (input_dim,) or self.feature_std.shape != (
            input_dim,
        ):
            raise ValueError("feature standardizer shape mismatch")
        if np.any(self.feature_std <= 0.0):
            raise ValueError("feature_std must be positive")
        if self.weight_input.shape[0] != input_dim:
            raise ValueError("weight_input shape mismatch")
        hidden_dim = self.weight_input.shape[1]
        if self.bias_hidden.shape != (hidden_dim,):
            raise ValueError("bias_hidden shape mismatch")
        if self.weight_output.shape != (hidden_dim,):
            raise ValueError("weight_output shape mismatch")

    def _matrix(self, rows: Sequence[CandidateFeatureRow]) -> np.ndarray:
        matrix = np.asarray(
            select_feature_columns(rows, self.feature_names),
            dtype=np.float64,
        )
        return (matrix - self.feature_mean) / self.feature_std

    def predict_gates(self, rows: Sequence[CandidateFeatureRow]) -> list[float]:
        if not rows:
            raise ValueError("rows must not be empty")
        matrix = self._matrix(rows)
        hidden = np.tanh(matrix @ self.weight_input + self.bias_hidden)
        logits = hidden @ self.weight_output + self.bias_output
        return _sigmoid(logits).tolist()

    def predict_scores(self, rows: Sequence[CandidateFeatureRow]) -> list[float]:
        gates = self.predict_gates(rows)
        return blend_scores(
            [row.oea_score for row in rows],
            [row.asr_score for row in rows],
            gates,
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "schema_version": 1,
            "model_type": "candidate_gate_mlp_numpy",
            "feature_names": list(self.feature_names),
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "weight_input": self.weight_input.tolist(),
            "bias_hidden": self.bias_hidden.tolist(),
            "weight_output": self.weight_output.tolist(),
            "bias_output": float(self.bias_output),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CandidateGateMLP":
        if value.get("schema_version") != 1:
            raise ValueError("unsupported gate schema_version")
        if value.get("model_type") != "candidate_gate_mlp_numpy":
            raise ValueError("unsupported gate model_type")
        return cls(
            feature_names=tuple(value["feature_names"]),  # type: ignore[arg-type]
            feature_mean=np.asarray(value["feature_mean"], dtype=np.float64),
            feature_std=np.asarray(value["feature_std"], dtype=np.float64),
            weight_input=np.asarray(value["weight_input"], dtype=np.float64),
            bias_hidden=np.asarray(value["bias_hidden"], dtype=np.float64),
            weight_output=np.asarray(value["weight_output"], dtype=np.float64),
            bias_output=float(value["bias_output"]),  # type: ignore[arg-type]
            seed=int(value["seed"]),  # type: ignore[arg-type]
        )


def _standardizer(
    groups: Sequence[GateTrainingGroup],
    feature_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    rows = [row for group in groups for row in group.rows]
    matrix = np.asarray(
        select_feature_columns(rows, feature_names),
        dtype=np.float64,
    )
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std <= 1e-12] = 1.0
    return mean, std


def _group_loss(model: CandidateGateMLP, group: GateTrainingGroup) -> float:
    return multi_positive_listwise_loss(
        model.predict_scores(group.rows),
        [row.relevance for row in group.rows],
    )


def mean_group_loss(
    model: CandidateGateMLP,
    groups: Sequence[GateTrainingGroup],
) -> float:
    if not groups:
        raise ValueError("groups must not be empty")
    return math.fsum(_group_loss(model, group) for group in groups) / len(groups)


def train_candidate_gate(
    train_groups: Sequence[GateTrainingGroup],
    dev_groups: Sequence[GateTrainingGroup],
    *,
    feature_names: Sequence[str],
    hidden_dim: int = 16,
    learning_rate: float = 1e-2,
    weight_decay: float = 1e-4,
    max_epochs: int = 200,
    patience: int = 20,
    seed: int = 42,
) -> tuple[CandidateGateMLP, Dict[str, object]]:
    """Train only the small gate; all upstream scores remain frozen."""

    if not train_groups or not dev_groups:
        raise ValueError("train_groups and dev_groups must not be empty")
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("feature_names must be unique and non-empty")
    if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
        raise ValueError("hidden_dim must be a positive integer")
    if max_epochs <= 0 or patience <= 0:
        raise ValueError("max_epochs and patience must be positive")
    if learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("invalid learning_rate or weight_decay")

    mean, std = _standardizer(train_groups, feature_names)
    generator = np.random.default_rng(seed)
    input_dim = len(feature_names)
    scale = math.sqrt(2.0 / (input_dim + hidden_dim))
    model = CandidateGateMLP(
        feature_names=tuple(feature_names),
        feature_mean=mean,
        feature_std=std,
        weight_input=generator.normal(0.0, scale, size=(input_dim, hidden_dim)),
        bias_hidden=np.zeros(hidden_dim, dtype=np.float64),
        weight_output=generator.normal(0.0, 1.0 / math.sqrt(hidden_dim), size=hidden_dim),
        bias_output=0.0,
        seed=seed,
    )

    parameter_names = (
        "weight_input",
        "bias_hidden",
        "weight_output",
        "bias_output",
    )
    first_moment = {
        "weight_input": np.zeros_like(model.weight_input),
        "bias_hidden": np.zeros_like(model.bias_hidden),
        "weight_output": np.zeros_like(model.weight_output),
        "bias_output": 0.0,
    }
    second_moment = copy.deepcopy(first_moment)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    best_model = copy.deepcopy(model)
    best_dev = math.inf
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        gradients = {
            "weight_input": np.zeros_like(model.weight_input),
            "bias_hidden": np.zeros_like(model.bias_hidden),
            "weight_output": np.zeros_like(model.weight_output),
            "bias_output": 0.0,
        }
        train_loss = 0.0
        for group in train_groups:
            rows = group.rows
            matrix = model._matrix(rows)
            hidden = np.tanh(matrix @ model.weight_input + model.bias_hidden)
            logits = hidden @ model.weight_output + model.bias_output
            gates = _sigmoid(logits)
            oea = np.asarray([row.oea_score for row in rows], dtype=np.float64)
            asr = np.asarray([row.asr_score for row in rows], dtype=np.float64)
            relevance = np.asarray(
                [row.relevance for row in rows],
                dtype=np.float64,
            )
            scores = oea + gates * (asr - oea)
            positive = relevance > 0.0
            train_loss += _logsumexp(scores) - _logsumexp(scores[positive])

            probability_all = np.exp(scores - _logsumexp(scores))
            probability_positive = np.zeros_like(scores)
            probability_positive[positive] = np.exp(
                scores[positive] - _logsumexp(scores[positive])
            )
            grad_scores = probability_all - probability_positive
            grad_logits = grad_scores * (asr - oea) * gates * (1.0 - gates)
            gradients["weight_output"] += hidden.T @ grad_logits
            gradients["bias_output"] += float(grad_logits.sum())
            grad_hidden = np.outer(grad_logits, model.weight_output)
            grad_activation = grad_hidden * (1.0 - hidden**2)
            gradients["weight_input"] += matrix.T @ grad_activation
            gradients["bias_hidden"] += grad_activation.sum(axis=0)

        train_loss /= len(train_groups)
        for name in parameter_names:
            gradient = gradients[name] / len(train_groups)
            if name in {"weight_input", "weight_output"}:
                gradient = gradient + weight_decay * getattr(model, name)
            first_moment[name] = beta1 * first_moment[name] + (1.0 - beta1) * gradient
            second_moment[name] = beta2 * second_moment[name] + (1.0 - beta2) * (
                gradient * gradient
            )
            corrected_first = first_moment[name] / (1.0 - beta1**epoch)
            corrected_second = second_moment[name] / (1.0 - beta2**epoch)
            updated = getattr(model, name) - learning_rate * corrected_first / (
                np.sqrt(corrected_second) + epsilon
            )
            setattr(model, name, updated)

        dev_loss = mean_group_loss(model, dev_groups)
        history.append(
            {
                "epoch": epoch,
                "train_listwise_loss": train_loss,
                "dev_listwise_loss": dev_loss,
            }
        )
        if dev_loss < best_dev - 1e-10:
            best_dev = dev_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    return best_model, {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_dev_listwise_loss": best_dev,
        "epochs_ran": len(history),
        "stopped_early": len(history) < max_epochs,
        "history": history,
    }
