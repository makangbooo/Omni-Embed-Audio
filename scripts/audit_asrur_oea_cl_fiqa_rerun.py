#!/usr/bin/env python3
"""Audit an independent OEA-Nemo3B-Cl FiQA rerun against the first run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("clean", "snr_20", "snr_10", "snr_0")
METRIC_NAMES = (
    "nDCG@10",
    "MRR@10",
    "Recall@10",
    "Recall@20",
    "Recall@50",
    "Recall@100",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-cache-root", type=Path, required=True)
    parser.add_argument("--reference-cache-root", type=Path, required=True)
    parser.add_argument("--fresh-result-root", type=Path, required=True)
    parser.add_argument("--reference-result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def read_ids(path: Path) -> list[str]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = json.loads(line)
            identifier = value.get("id")
            if not isinstance(identifier, str) or not identifier:
                raise ValueError(f"invalid identifier at {path}:{line_number}")
            rows.append(identifier)
    if not rows or len(rows) != len(set(rows)):
        raise ValueError(f"identifiers must be non-empty and unique: {path}")
    return rows


def embedding_comparison(
    fresh_path: Path,
    reference_path: Path,
    *,
    fresh_ids_path: Path,
    reference_ids_path: Path,
    batch_size: int = 4096,
) -> dict[str, Any]:
    fresh_ids = read_ids(fresh_ids_path)
    reference_ids = read_ids(reference_ids_path)
    if fresh_ids != reference_ids:
        raise ValueError(
            f"embedding identifier order differs: {fresh_ids_path} vs "
            f"{reference_ids_path}"
        )
    fresh = np.load(fresh_path, mmap_mode="r", allow_pickle=False)
    reference = np.load(reference_path, mmap_mode="r", allow_pickle=False)
    if fresh.shape != reference.shape or fresh.shape[0] != len(fresh_ids):
        raise ValueError(
            f"embedding shapes/IDs differ: {fresh.shape}, {reference.shape}, "
            f"{len(fresh_ids)}"
        )
    cosines = np.empty(fresh.shape[0], dtype=np.float64)
    maximum_differences = np.empty(fresh.shape[0], dtype=np.float64)
    fresh_norms = np.empty(fresh.shape[0], dtype=np.float64)
    reference_norms = np.empty(fresh.shape[0], dtype=np.float64)
    for start in range(0, fresh.shape[0], batch_size):
        stop = min(start + batch_size, fresh.shape[0])
        left = np.asarray(fresh[start:stop], dtype=np.float64)
        right = np.asarray(reference[start:stop], dtype=np.float64)
        left_norm = np.linalg.norm(left, axis=1)
        right_norm = np.linalg.norm(right, axis=1)
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("embedding arrays must contain only finite values")
        if np.any(left_norm == 0) or np.any(right_norm == 0):
            raise ValueError("embedding arrays contain a zero-norm row")
        cosines[start:stop] = np.sum(left * right, axis=1) / (
            left_norm * right_norm
        )
        maximum_differences[start:stop] = np.max(np.abs(left - right), axis=1)
        fresh_norms[start:stop] = left_norm
        reference_norms[start:stop] = right_norm
    order = np.argsort(cosines)[:10]
    return {
        "row_count": int(fresh.shape[0]),
        "embedding_dimension": int(fresh.shape[1]),
        "fresh_sha256": sha256(fresh_path),
        "reference_sha256": sha256(reference_path),
        "byte_identical": sha256(fresh_path) == sha256(reference_path),
        "row_cosine": {
            "mean": float(np.mean(cosines)),
            "minimum": float(np.min(cosines)),
            "p01": float(np.quantile(cosines, 0.01)),
            "p50": float(np.quantile(cosines, 0.50)),
            "count_below_0_999": int(np.sum(cosines < 0.999)),
            "count_below_0_99": int(np.sum(cosines < 0.99)),
            "count_below_0_90": int(np.sum(cosines < 0.90)),
        },
        "maximum_absolute_difference": {
            "maximum": float(np.max(maximum_differences)),
            "mean": float(np.mean(maximum_differences)),
        },
        "fresh_norm": {
            "minimum": float(np.min(fresh_norms)),
            "maximum": float(np.max(fresh_norms)),
            "mean": float(np.mean(fresh_norms)),
        },
        "reference_norm": {
            "minimum": float(np.min(reference_norms)),
            "maximum": float(np.max(reference_norms)),
            "mean": float(np.mean(reference_norms)),
        },
        "lowest_cosine_rows": [
            {
                "index": int(index),
                "id": fresh_ids[int(index)],
                "row_cosine": float(cosines[int(index)]),
                "maximum_absolute_difference": float(
                    maximum_differences[int(index)]
                ),
            }
            for index in order
        ],
    }


def load_oea_metrics(root: Path, condition: str) -> dict[str, Any]:
    path = root / condition / "metrics.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    evaluation = value["evaluations"]["B3_oea"]
    metrics = {name: float(evaluation[name]) for name in METRIC_NAMES}
    oracle = value.get("oea_candidate_oracle")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "metrics": metrics,
        "candidate_oracle": oracle,
    }


def metric_comparison(
    fresh_result_root: Path,
    reference_result_root: Path,
) -> dict[str, Any]:
    result = {}
    for condition in CONDITIONS:
        fresh = load_oea_metrics(fresh_result_root, condition)
        reference = load_oea_metrics(reference_result_root, condition)
        result[condition] = {
            "fresh": fresh,
            "reference": reference,
            "delta_fresh_minus_reference": {
                name: fresh["metrics"][name] - reference["metrics"][name]
                for name in METRIC_NAMES
            },
        }
    return result


def classify(metrics: dict[str, Any]) -> dict[str, Any]:
    fresh_recall = {
        condition: row["fresh"]["metrics"]["Recall@100"]
        for condition, row in metrics.items()
    }
    recall_deltas = {
        condition: abs(row["delta_fresh_minus_reference"]["Recall@100"])
        for condition, row in metrics.items()
    }
    failure_threshold = 0.01
    reproduction_tolerance = 0.005
    if all(value < failure_threshold for value in fresh_recall.values()) and all(
        value <= reproduction_tolerance for value in recall_deltas.values()
    ):
        decision = "REPRODUCED_OEA_CL_FIQA_TRANSFER_FAILURE"
    elif any(value >= 0.80 for value in fresh_recall.values()):
        decision = "NOT_REPRODUCED_REQUIRES_IMPLEMENTATION_AUDIT"
    else:
        decision = "INCONCLUSIVE_REQUIRES_REVIEW"
    return {
        "decision": decision,
        "fixed_thresholds": {
            "domain_failure_recall_at_100_below": failure_threshold,
            "reference_reproduction_absolute_recall_at_100_delta_at_most": (
                reproduction_tolerance
            ),
            "preregistered_go_recall_at_100_at_least": 0.80,
        },
        "fresh_recall_at_100": fresh_recall,
        "absolute_recall_at_100_delta": recall_deltas,
        "checkpoint_selection_performed": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    roots = {
        "fresh_cache_root": args.fresh_cache_root.resolve(),
        "reference_cache_root": args.reference_cache_root.resolve(),
        "fresh_result_root": args.fresh_result_root.resolve(),
        "reference_result_root": args.reference_result_root.resolve(),
    }
    for label, path in roots.items():
        if not path.is_dir():
            raise FileNotFoundError(f"{label} is absent: {path}")
    if roots["fresh_cache_root"] == roots["reference_cache_root"]:
        raise ValueError("fresh and reference cache roots must differ")

    embeddings: dict[str, Any] = {
        "document_clean": embedding_comparison(
            roots["fresh_cache_root"] / "oea/clean/document_embeddings.npy",
            roots["reference_cache_root"] / "oea/clean/document_embeddings.npy",
            fresh_ids_path=(
                roots["fresh_cache_root"] / "oea/clean/document_ids.jsonl"
            ),
            reference_ids_path=(
                roots["reference_cache_root"] / "oea/clean/document_ids.jsonl"
            ),
        )
    }
    for condition in CONDITIONS:
        embeddings[f"audio_{condition}"] = embedding_comparison(
            roots["fresh_cache_root"]
            / f"oea/{condition}/audio_embeddings.npy",
            roots["reference_cache_root"]
            / f"oea/{condition}/audio_embeddings.npy",
            fresh_ids_path=(
                roots["fresh_cache_root"] / f"oea/{condition}/audio_ids.jsonl"
            ),
            reference_ids_path=(
                roots["reference_cache_root"] / f"oea/{condition}/audio_ids.jsonl"
            ),
        )
    metrics = metric_comparison(
        roots["fresh_result_root"],
        roots["reference_result_root"],
    )
    report = {
        "schema_version": 1,
        "status": "complete",
        "stage": "OEA-Nemo3B-Cl_independent_FiQA_rerun",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output(
            "status", "--short", "--untracked-files=all"
        ),
        "roots": {name: str(path) for name, path in roots.items()},
        "embeddings": embeddings,
        "metrics": metrics,
        "classification": classify(metrics),
        "scope": {
            "checkpoint": "JudeJiwoo/OEA-Nemo3B-Cl",
            "candidate_depth": 100,
            "conditions": list(CONDITIONS),
            "fresh_document_and_audio_encoding": True,
            "old_oea_embedding_reuse": False,
            "training_performed": False,
            "download_performed": False,
            "checkpoint_selection_performed": False,
        },
    }
    for value in _walk_floats(report):
        if not math.isfinite(value):
            raise ValueError("audit report contains a non-finite float")
    return report


def _walk_floats(value: Any):
    if isinstance(value, float):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _walk_floats(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_floats(child)


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.output}")
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": report["classification"]["decision"],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
