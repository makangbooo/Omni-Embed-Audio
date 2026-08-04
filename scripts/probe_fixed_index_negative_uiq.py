#!/usr/bin/env python3
"""Probe whether frozen OEA audio embeddings retain negative-UIQ signal.

The probe trains only a low-rank residual transform on query embeddings. Audio
embeddings are loaded from completed evaluations, hash-verified, and never
modified. Cross-validation groups the target/hard-negative graph by connected
component, so no audio ID used in a test fold occurs in that fold's training
pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-metrics",
        type=Path,
        action="append",
        required=True,
        help="Completed negative-UIQ metrics.json; repeat for each dataset.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--anchor-weight", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260806)
    parser.add_argument("--maximum-fold-imbalance", type=float, default=4.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--allow-dirty-git",
        action="store_true",
        help="Testing only; formal runs require a clean worktree.",
    )
    return parser.parse_args()


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no JSONL rows")
    return rows


def normalized(values: np.ndarray, label: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError(f"{label} must be a non-empty matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} contains non-finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{label} contains a zero row")
    return matrix / norms


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        self.parent[high] = low


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    label: str
    model: str
    metrics_path: Path
    query_embeddings: np.ndarray
    candidate_embeddings: np.ndarray
    candidate_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    hard_negative_ids: tuple[str, ...]
    target_indices: np.ndarray
    hard_negative_indices: np.ndarray
    component_keys: tuple[str, ...]
    inputs: Mapping[str, Mapping[str, Any]]


def _strings(values: np.ndarray, label: str) -> list[str]:
    if values.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    result = [str(value).strip() for value in values.tolist()]
    if any(not value for value in result) or len(result) != len(set(result)):
        raise ValueError(f"{label} must contain unique non-empty IDs")
    return result


def _verified_input(record: Mapping[str, Any], label: str) -> Path:
    path_value = record.get("path")
    expected_hash = record.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected_hash, str):
        raise ValueError(f"{label}: missing path or SHA256")
    path = Path(path_value).resolve()
    observed = file_record(path)
    if observed["sha256"] != expected_hash:
        raise ValueError(f"{label}: SHA256 mismatch")
    expected_size = record.get("size_bytes")
    if expected_size is not None and observed["size_bytes"] != expected_size:
        raise ValueError(f"{label}: size mismatch")
    return path


def load_bundle(metrics_path: Path) -> DatasetBundle:
    from scripts.evaluate_negative_uiq_npz import resolve_pairing_candidate_ids

    path = metrics_path.resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise ValueError(f"incomplete source evaluation: {path}")
    if report.get("normalization_applied") is not True:
        raise ValueError(f"source evaluation did not normalize embeddings: {path}")
    inputs = report.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"source evaluation lacks inputs: {path}")
    audio_path = _verified_input(inputs["audio_npz"], "audio_npz")
    query_path = _verified_input(inputs["query_npz"], "query_npz")
    pairing_path = _verified_input(inputs["pairing_jsonl"], "pairing_jsonl")

    with np.load(audio_path, allow_pickle=True) as archive:
        candidates = normalized(archive["embeddings"], "candidate embeddings")
        candidate_ids = _strings(np.asarray(archive["clip_ids"]), "candidate IDs")
        filenames = (
            _strings(np.asarray(archive["filenames"]), "candidate filenames")
            if "filenames" in archive
            else None
        )
    with np.load(query_path, allow_pickle=True) as archive:
        queries = normalized(archive["embeddings"], "query embeddings")
    pairings = read_jsonl(pairing_path)

    candidate_count = int(report["candidate_count"])
    query_count = int(report["evaluated_query_count"])
    if candidates.shape[0] != candidate_count or len(candidate_ids) != candidate_count:
        raise ValueError(f"candidate count mismatch: {path}")
    if queries.shape[0] != query_count or len(pairings) != query_count:
        raise ValueError(f"query count mismatch: {path}")
    if candidates.shape[1] != queries.shape[1]:
        raise ValueError(f"audio/query dimensions differ: {path}")

    query_ids = []
    requested_targets = []
    requested_hard_negatives = []
    for row_number, row in enumerate(pairings):
        values = []
        for field in ("query_id", "target_id", "hard_negative_id"):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"pairing row {row_number}: invalid {field}")
            values.append(value.strip())
        query_ids.append(values[0])
        requested_targets.append(values[1])
        requested_hard_negatives.append(values[2])
    if len(query_ids) != len(set(query_ids)):
        raise ValueError(f"duplicate query IDs: {path}")
    target_ids, _ = resolve_pairing_candidate_ids(
        requested_targets,
        candidate_ids,
        label="target_id",
        candidate_filenames=filenames,
    )
    hard_negative_ids, _ = resolve_pairing_candidate_ids(
        requested_hard_negatives,
        candidate_ids,
        label="hard_negative_id",
        candidate_filenames=filenames,
    )
    if any(left == right for left, right in zip(target_ids, hard_negative_ids)):
        raise ValueError(f"target equals hard negative: {path}")
    lookup = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}

    dataset_name = path.parent.name
    graph = UnionFind()
    for target_id, hard_negative_id in zip(target_ids, hard_negative_ids):
        graph.union(target_id, hard_negative_id)
    component_keys = tuple(
        f"{dataset_name}:{graph.find(target_id)}"
        for target_id in target_ids
    )
    return DatasetBundle(
        name=dataset_name,
        label=str(report["dataset"]),
        model=str(report["model"]),
        metrics_path=path,
        query_embeddings=queries,
        candidate_embeddings=candidates,
        candidate_ids=tuple(candidate_ids),
        query_ids=tuple(query_ids),
        target_ids=tuple(target_ids),
        hard_negative_ids=tuple(hard_negative_ids),
        target_indices=np.asarray([lookup[value] for value in target_ids], dtype=np.int64),
        hard_negative_indices=np.asarray(
            [lookup[value] for value in hard_negative_ids], dtype=np.int64
        ),
        component_keys=component_keys,
        inputs=inputs,
    )


def _seeded_digest(seed: int, *values: str) -> str:
    text = "\x1f".join((str(seed), *values))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assign_component_folds(
    bundles: Sequence[DatasetBundle],
    *,
    folds: int,
    seed: int,
    maximum_imbalance: float,
) -> tuple[dict[str, int], dict[str, Any]]:
    if folds < 2 or maximum_imbalance < 1.0:
        raise ValueError("fold count and maximum imbalance are invalid")
    assignments: dict[str, int] = {}
    audit: dict[str, Any] = {}
    for bundle in bundles:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, component in enumerate(bundle.component_keys):
            groups[component].append(index)
        if len(groups) < folds:
            raise ValueError(
                f"{bundle.name}: {len(groups)} components cannot fill {folds} folds"
            )
        ordered = sorted(
            groups,
            key=lambda component: (
                -len(groups[component]),
                _seeded_digest(seed, bundle.name, component),
            ),
        )
        row_loads = [0] * folds
        component_loads = [0] * folds
        for component in ordered:
            fold = min(range(folds), key=lambda value: (row_loads[value], value))
            assignments[component] = fold
            row_loads[fold] += len(groups[component])
            component_loads[fold] += 1
        imbalance = max(row_loads) / min(row_loads)
        if imbalance > maximum_imbalance:
            raise ValueError(
                f"{bundle.name}: fold row imbalance {imbalance:.3f} exceeds "
                f"{maximum_imbalance:.3f}"
            )
        audit[bundle.name] = {
            "component_count": len(groups),
            "query_rows_by_fold": row_loads,
            "components_by_fold": component_loads,
            "maximum_to_minimum_row_ratio": imbalance,
        }
    return assignments, audit


def train_low_rank_adapter(
    queries: np.ndarray,
    pair_differences: np.ndarray,
    *,
    rank: int,
    epochs: int,
    learning_rate: float,
    temperature: float,
    anchor_weight: float,
    weight_decay: float,
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    import torch

    if queries.shape != pair_differences.shape or queries.ndim != 2:
        raise ValueError("training query and pair-difference matrices must match")
    if rank < 1 or rank > queries.shape[1] or epochs < 1:
        raise ValueError("rank or epoch count is invalid")
    if learning_rate <= 0 or temperature <= 0 or anchor_weight < 0:
        raise ValueError("training hyperparameters are invalid")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    selected_device = torch.device(device)
    query_tensor = torch.as_tensor(queries, dtype=torch.float32, device=selected_device)
    difference_tensor = torch.as_tensor(
        pair_differences, dtype=torch.float32, device=selected_device
    )
    dimension = queries.shape[1]
    down = torch.nn.Parameter(
        torch.empty(dimension, rank, dtype=torch.float32, device=selected_device)
    )
    up = torch.nn.Parameter(
        torch.zeros(rank, dimension, dtype=torch.float32, device=selected_device)
    )
    torch.nn.init.normal_(down, mean=0.0, std=1.0 / math.sqrt(dimension))
    optimizer = torch.optim.AdamW(
        (down, up), lr=learning_rate, weight_decay=weight_decay
    )

    def objective() -> tuple[Any, Any, Any]:
        adapted = torch.nn.functional.normalize(
            query_tensor + (query_tensor @ down) @ up,
            p=2,
            dim=-1,
        )
        margins = torch.sum(adapted * difference_tensor, dim=-1)
        pair_loss = torch.nn.functional.softplus(-margins / temperature).mean()
        anchor_loss = (1.0 - torch.sum(adapted * query_tensor, dim=-1)).mean()
        return pair_loss + anchor_weight * anchor_loss, pair_loss, anchor_loss

    with torch.no_grad():
        initial_total, initial_pair, initial_anchor = objective()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = objective()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_total, final_pair, final_anchor = objective()
    return (
        down.detach().cpu().numpy(),
        up.detach().cpu().numpy(),
        {
            "training_rows": int(queries.shape[0]),
            "trainable_parameter_count": int(2 * dimension * rank),
            "initial_total_loss": float(initial_total.item()),
            "final_total_loss": float(final_total.item()),
            "initial_pair_loss": float(initial_pair.item()),
            "final_pair_loss": float(final_pair.item()),
            "initial_anchor_loss": float(initial_anchor.item()),
            "final_anchor_loss": float(final_anchor.item()),
        },
    )


def apply_adapter(
    queries: np.ndarray, down: np.ndarray, up: np.ndarray
) -> np.ndarray:
    return normalized(queries + (queries @ down) @ up, "adapted queries")


def evaluate_queries(
    queries: np.ndarray,
    candidates: np.ndarray,
    target_indices: np.ndarray,
    hard_negative_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    scores = np.asarray(queries, dtype=np.float32) @ np.asarray(
        candidates, dtype=np.float32
    ).T
    rows = np.arange(scores.shape[0])
    target_scores = scores[rows, target_indices]
    hard_negative_scores = scores[rows, hard_negative_indices]
    return {
        "target_ranks": 1 + np.count_nonzero(scores > target_scores[:, None], axis=1),
        "hard_negative_ranks": 1
        + np.count_nonzero(scores > hard_negative_scores[:, None], axis=1),
        "target_scores": target_scores,
        "hard_negative_scores": hard_negative_scores,
        "margins": target_scores - hard_negative_scores,
    }


def cluster_bootstrap_delta(
    method: Sequence[float],
    baseline: Sequence[float],
    clusters: Sequence[str],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    left = np.asarray(method, dtype=np.float64)
    right = np.asarray(baseline, dtype=np.float64)
    cluster_values = np.asarray(clusters, dtype=object)
    if (
        left.ndim != 1
        or left.shape != right.shape
        or left.shape != cluster_values.shape
        or left.size < 2
        or iterations < 100
    ):
        raise ValueError("invalid clustered bootstrap inputs")
    differences = left - right
    unique_clusters = sorted(set(str(value) for value in cluster_values.tolist()))
    if len(unique_clusters) < 2:
        raise ValueError("clustered bootstrap requires at least two components")
    sums = np.asarray(
        [differences[cluster_values == value].sum() for value in unique_clusters],
        dtype=np.float64,
    )
    counts = np.asarray(
        [np.count_nonzero(cluster_values == value) for value in unique_clusters],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 1000):
        stop = min(iterations, start + 1000)
        indices = generator.integers(
            0, len(unique_clusters), size=(stop - start, len(unique_clusters))
        )
        draws[start:stop] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "sample_count": int(left.size),
        "cluster_count": len(unique_clusters),
        "observed_mean_delta": float(differences.mean()),
        "confidence_interval_95": [float(lower), float(upper)],
        "probability_delta_positive": float(np.mean(draws > 0.0)),
        "iterations": iterations,
        "seed": seed,
    }


def _metric_summary(
    baseline: Mapping[str, np.ndarray],
    adapted: Mapping[str, np.ndarray],
    components: Sequence[str],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    from AudioRetrieval.evaluation.negative_metrics import compute_negative_query_metrics

    baseline_hnsr = baseline["hard_negative_ranks"] > baseline["target_ranks"]
    adapted_hnsr = adapted["hard_negative_ranks"] > adapted["target_ranks"]
    hnsr_effect = cluster_bootstrap_delta(
        adapted_hnsr.astype(np.float64) * 100.0,
        baseline_hnsr.astype(np.float64) * 100.0,
        components,
        iterations=iterations,
        seed=seed,
    )
    margin_effect = cluster_bootstrap_delta(
        adapted["margins"],
        baseline["margins"],
        components,
        iterations=iterations,
        seed=seed + 1,
    )
    return {
        "untouched_oea": {
            "negative_retrieval": compute_negative_query_metrics(
                baseline["target_ranks"], baseline["hard_negative_ranks"]
            ),
            "mean_target_minus_hard_negative_cosine": float(
                np.mean(baseline["margins"])
            ),
        },
        "out_of_fold_query_adapter": {
            "negative_retrieval": compute_negative_query_metrics(
                adapted["target_ranks"], adapted["hard_negative_ranks"]
            ),
            "mean_target_minus_hard_negative_cosine": float(
                np.mean(adapted["margins"])
            ),
        },
        "paired_component_bootstrap_effects": {
            "HNSR_percentage_points": hnsr_effect,
            "mean_target_minus_hard_negative_cosine": margin_effect,
        },
        "predeclared_pass_rule": {
            "HNSR_delta_ci_excludes_zero": hnsr_effect["confidence_interval_95"][0]
            > 0.0,
            "margin_delta_ci_excludes_zero": margin_effect["confidence_interval_95"][0]
            > 0.0,
            "fixed_index_signal_supported": (
                hnsr_effect["confidence_interval_95"][0] > 0.0
                and margin_effect["confidence_interval_95"][0] > 0.0
            ),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.bootstrap_iterations < 100:
        raise ValueError("bootstrap iterations must be at least 100")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output: {output_dir}")
    git_status = git_output("status", "--short", "--untracked-files=all")
    if git_status and not args.allow_dirty_git:
        raise RuntimeError(f"formal probe requires clean Git: {git_status!r}")
    bundles = [load_bundle(path) for path in args.dataset_metrics]
    if len(bundles) < 2 or len({bundle.name for bundle in bundles}) != len(bundles):
        raise ValueError("provide at least two uniquely named dataset evaluations")
    models = {bundle.model for bundle in bundles}
    dimensions = {bundle.query_embeddings.shape[1] for bundle in bundles}
    if len(models) != 1 or len(dimensions) != 1:
        raise ValueError("all datasets must use one model and embedding dimension")
    assignments, split_audit = assign_component_folds(
        bundles,
        folds=args.folds,
        seed=args.seed,
        maximum_imbalance=args.maximum_fold_imbalance,
    )
    output_dir.mkdir(parents=True)

    baseline_by_dataset = {
        bundle.name: evaluate_queries(
            bundle.query_embeddings,
            bundle.candidate_embeddings,
            bundle.target_indices,
            bundle.hard_negative_indices,
        )
        for bundle in bundles
    }
    adapted_by_dataset = {
        bundle.name: {
            key: np.full_like(value, -1 if "ranks" in key else np.nan)
            for key, value in baseline_by_dataset[bundle.name].items()
        }
        for bundle in bundles
    }
    fold_rows_by_dataset = {
        bundle.name: np.asarray(
            [assignments[component] for component in bundle.component_keys],
            dtype=np.int64,
        )
        for bundle in bundles
    }
    fold_reports = []
    adapter_artifacts = []
    for fold in range(args.folds):
        training_queries = []
        training_differences = []
        leakage_rows = {}
        for bundle in bundles:
            test_mask = fold_rows_by_dataset[bundle.name] == fold
            train_mask = ~test_mask
            training_queries.append(bundle.query_embeddings[train_mask])
            training_differences.append(
                bundle.candidate_embeddings[bundle.target_indices[train_mask]]
                - bundle.candidate_embeddings[bundle.hard_negative_indices[train_mask]]
            )
            train_audio_ids = set(
                np.asarray(bundle.target_ids, dtype=object)[train_mask].tolist()
            ) | set(
                np.asarray(bundle.hard_negative_ids, dtype=object)[train_mask].tolist()
            )
            test_audio_ids = set(
                np.asarray(bundle.target_ids, dtype=object)[test_mask].tolist()
            ) | set(
                np.asarray(bundle.hard_negative_ids, dtype=object)[test_mask].tolist()
            )
            overlap = train_audio_ids & test_audio_ids
            if overlap:
                raise RuntimeError(
                    f"audio leakage in {bundle.name} fold {fold}: {sorted(overlap)[:10]}"
                )
            leakage_rows[bundle.name] = {
                "training_query_count": int(np.count_nonzero(train_mask)),
                "test_query_count": int(np.count_nonzero(test_mask)),
                "training_pair_audio_id_count": len(train_audio_ids),
                "test_pair_audio_id_count": len(test_audio_ids),
                "overlapping_pair_audio_id_count": 0,
            }
        down, up, training_report = train_low_rank_adapter(
            np.concatenate(training_queries, axis=0),
            np.concatenate(training_differences, axis=0),
            rank=args.rank,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            anchor_weight=args.anchor_weight,
            weight_decay=args.weight_decay,
            seed=args.seed + fold,
            device=args.device,
        )
        adapter_path = output_dir / f"fold_{fold}_adapter.npz"
        np.savez_compressed(adapter_path, down=down, up=up)
        adapter_artifacts.append(file_record(adapter_path))
        for bundle in bundles:
            test_mask = fold_rows_by_dataset[bundle.name] == fold
            indices = np.flatnonzero(test_mask)
            evaluated = evaluate_queries(
                apply_adapter(bundle.query_embeddings[test_mask], down, up),
                bundle.candidate_embeddings,
                bundle.target_indices[test_mask],
                bundle.hard_negative_indices[test_mask],
            )
            for key, values in evaluated.items():
                adapted_by_dataset[bundle.name][key][indices] = values
        fold_reports.append(
            {
                "fold": fold,
                "training": training_report,
                "audio_id_leakage_audit": leakage_rows,
                "adapter_artifact": file_record(adapter_path),
            }
        )

    for bundle in bundles:
        for key, values in adapted_by_dataset[bundle.name].items():
            if "ranks" in key and np.any(values < 1):
                raise RuntimeError(f"missing OOF ranks for {bundle.name}/{key}")
            if not np.isfinite(values).all():
                raise RuntimeError(f"missing OOF values for {bundle.name}/{key}")

    per_query_path = output_dir / "oof_predictions.jsonl"
    with per_query_path.open("w", encoding="utf-8", newline="\n") as stream:
        for bundle in bundles:
            baseline = baseline_by_dataset[bundle.name]
            adapted = adapted_by_dataset[bundle.name]
            for index, query_id in enumerate(bundle.query_ids):
                row = {
                    "dataset": bundle.name,
                    "query_id": query_id,
                    "fold": int(fold_rows_by_dataset[bundle.name][index]),
                    "component": bundle.component_keys[index],
                    "target_id": bundle.target_ids[index],
                    "hard_negative_id": bundle.hard_negative_ids[index],
                    "untouched_oea": {
                        key: float(values[index]) if "ranks" not in key else int(values[index])
                        for key, values in baseline.items()
                    },
                    "out_of_fold_query_adapter": {
                        key: float(values[index]) if "ranks" not in key else int(values[index])
                        for key, values in adapted.items()
                    },
                }
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    by_dataset = {}
    for index, bundle in enumerate(bundles):
        by_dataset[bundle.name] = _metric_summary(
            baseline_by_dataset[bundle.name],
            adapted_by_dataset[bundle.name],
            bundle.component_keys,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed + 10 * index,
        )
    combined_baseline = {
        key: np.concatenate([baseline_by_dataset[bundle.name][key] for bundle in bundles])
        for key in next(iter(baseline_by_dataset.values()))
    }
    combined_adapted = {
        key: np.concatenate([adapted_by_dataset[bundle.name][key] for bundle in bundles])
        for key in next(iter(adapted_by_dataset.values()))
    }
    combined_components = [
        component for bundle in bundles for component in bundle.component_keys
    ]
    overall = _metric_summary(
        combined_baseline,
        combined_adapted,
        combined_components,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed + 100,
    )
    audio_immutability = {}
    for bundle in bundles:
        before = bundle.inputs["audio_npz"]
        after = file_record(Path(str(before["path"])))
        unchanged = (
            after["sha256"] == before["sha256"]
            and after["size_bytes"] == before["size_bytes"]
        )
        if not unchanged:
            raise RuntimeError(f"audio NPZ changed during probe: {bundle.name}")
        audio_immutability[bundle.name] = {
            "before": before,
            "after": after,
            "unchanged": True,
        }
    return {
        "schema_version": 1,
        "status": "complete",
        "stage": "fixed_index_negative_uiq_query_adapter_probe",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_before_output": git_status,
        "model": next(iter(models)),
        "dataset_names": [bundle.name for bundle in bundles],
        "query_count": sum(len(bundle.query_ids) for bundle in bundles),
        "embedding_dimension": next(iter(dimensions)),
        "protocol": {
            "audio_embeddings_frozen": True,
            "audio_index_rebuilt": False,
            "trainable_component": "query_side_low_rank_residual_adapter_only",
            "adapter_equation": "normalize(q + (q @ down) @ up)",
            "fold_grouping": "target_hard_negative_audio_graph_connected_components",
            "out_of_fold_predictions_only": True,
            "test_audio_ids_present_in_fold_training_pairs": False,
            "hyperparameter_selection": "predeclared_fixed_no_validation_or_test_tuning",
            "pairing_source": "INFERRED_DETERMINISTIC_RELEASED_CAPTION_IDENTITY",
        },
        "hyperparameters": {
            "folds": args.folds,
            "rank": args.rank,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "temperature": args.temperature,
            "anchor_weight": args.anchor_weight,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_seed": args.bootstrap_seed,
            "device": args.device,
        },
        "split_audit": split_audit,
        "audio_embedding_immutability_audit": audio_immutability,
        "folds": fold_reports,
        "overall": overall,
        "by_dataset": by_dataset,
        "inputs": {
            bundle.name: {
                "source_metrics": file_record(bundle.metrics_path),
                **bundle.inputs,
            }
            for bundle in bundles
        },
        "artifacts": {
            "oof_predictions": file_record(per_query_path),
            "fold_adapters": adapter_artifacts,
        },
        "claim_boundary": (
            "A passing result shows that one fixed, predeclared low-rank query "
            "adapter can exploit transferable target-vs-hard-negative signal in "
            "immutable OEA audio embeddings for held-out audio-disjoint pairs. "
            "It is diagnostic evidence, not a strict paper benchmark, and does "
            "not by itself establish unseen-intent or unseen-dataset generalization."
        ),
    }


def main() -> int:
    args = parse_args()
    result = run(args)
    output_path = args.output_dir.resolve() / "metrics.json"
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(
        "FIXED_INDEX_PROBE_RESULT="
        + json.dumps(
            {
                "status": result["status"],
                "model": result["model"],
                "query_count": result["query_count"],
                "fixed_index_signal_supported": result["overall"][
                    "predeclared_pass_rule"
                ]["fixed_index_signal_supported"],
                "metrics_path": str(output_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
