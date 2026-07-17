#!/usr/bin/env python3
"""Verify a completed five-sample vanilla run before allowing full evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np


EXPECTED_CANDIDATES = 5
EXPECTED_QUERIES = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def require_identity(
    identity: Any,
    expected_path: Path,
    label: str,
) -> dict[str, Any]:
    if not isinstance(identity, dict):
        raise ValueError(f"{label} identity must be an object")
    path_value = identity.get("path")
    size_value = identity.get("size_bytes")
    sha_value = identity.get("sha256")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{label} identity path is invalid")
    if not isinstance(size_value, int) or isinstance(size_value, bool) or size_value < 0:
        raise ValueError(f"{label} identity size_bytes is invalid")
    if (
        not isinstance(sha_value, str)
        or len(sha_value) != 64
        or any(character not in "0123456789abcdef" for character in sha_value)
    ):
        raise ValueError(f"{label} identity SHA256 is invalid")
    actual_path = Path(path_value).resolve()
    expected_path = expected_path.resolve()
    if actual_path != expected_path:
        raise ValueError(f"{label} identity points outside the smoke run")
    if not actual_path.is_file():
        raise FileNotFoundError(actual_path)
    if actual_path.stat().st_size != size_value:
        raise ValueError(f"{label} size differs from generation metrics")
    actual_sha = sha256_file(actual_path)
    if actual_sha != sha_value:
        raise ValueError(f"{label} SHA256 differs from generation metrics")
    return {
        "path": str(actual_path),
        "size_bytes": size_value,
        "sha256": actual_sha,
    }


def require_model_lock_identity(
    metrics: Mapping[str, Any], model_lock: Path
) -> dict[str, Any]:
    binding = metrics.get("vanilla_model_lock")
    if not isinstance(binding, dict):
        raise ValueError("generation metrics are missing vanilla_model_lock")
    identity = binding.get("model_lock")
    if not isinstance(identity, dict):
        raise ValueError("generation metrics are missing model_lock identity")
    size_value = identity.get("size_bytes")
    sha_value = identity.get("sha256")
    if not model_lock.is_file():
        raise FileNotFoundError(model_lock)
    if model_lock.stat().st_size != size_value:
        raise ValueError("model lock size differs from smoke metrics")
    actual_sha = sha256_file(model_lock)
    if actual_sha != sha_value:
        raise ValueError("model lock SHA256 differs from smoke metrics")
    return {
        "path": str(model_lock.resolve()),
        "size_bytes": model_lock.stat().st_size,
        "sha256": actual_sha,
    }


def require_embedding(
    path: Path, expected_shape: list[int], label: str
) -> None:
    value = np.load(path, allow_pickle=False)
    if list(value.shape) != expected_shape:
        raise ValueError(f"{label} array shape differs from smoke metrics")
    if not np.isfinite(value).all():
        raise ValueError(f"{label} contains non-finite values")
    norms = np.linalg.norm(value, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(f"{label} is not L2 normalized")


def require_jsonl_count(path: Path, expected: int, label: str) -> None:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{label} line {line_number} is blank")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{label} line {line_number} is not an object")
            rows.append(value)
    if len(rows) != expected:
        raise ValueError(f"{label} row count differs from smoke metrics")


def verify_smoke_gate(
    metrics: Mapping[str, Any],
    *,
    metrics_path: Path,
    backbone: str,
    model_lock: Path,
    current_git_commit: str,
) -> dict[str, Any]:
    if metrics.get("schema_version") != 1 or metrics.get("status") != "complete":
        raise ValueError("smoke generation metrics are not complete schema-v1 output")
    if metrics.get("git_commit") != current_git_commit:
        raise ValueError("smoke generation commit differs from the current Git commit")
    if metrics.get("backbone_id") != backbone:
        raise ValueError("smoke generation backbone differs from requested full run")
    if metrics.get("candidate_count") != EXPECTED_CANDIDATES:
        raise ValueError("smoke generation must contain exactly five candidates")
    if metrics.get("query_count") != EXPECTED_QUERIES:
        raise ValueError("smoke generation must contain exactly 25 queries")
    dimension = metrics.get("embedding_dimension")
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
        raise ValueError("smoke embedding_dimension must be a positive integer")
    candidate_shape = [EXPECTED_CANDIDATES, dimension]
    query_shape = [EXPECTED_QUERIES, dimension]
    if metrics.get("candidate_embedding_shape") != candidate_shape:
        raise ValueError("candidate embedding shape differs from the smoke contract")
    if metrics.get("query_embedding_shape") != query_shape:
        raise ValueError("query embedding shape differs from the smoke contract")
    for field in (
        "projection_head_loaded",
        "lora_loaded",
        "oea_checkpoint_loaded",
    ):
        if metrics.get(field) is not False:
            raise ValueError(f"smoke generation must record {field}=false")
        model_load = metrics.get("model_load")
        if isinstance(model_load, dict) and model_load.get(field) is not False:
            raise ValueError(f"smoke model_load must record {field}=false")

    run_dir = metrics_path.resolve().parent
    artifacts = metrics.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("smoke generation metrics are missing artifacts")
    expected_artifacts = {
        "candidate_embeddings": run_dir / "candidate_embeddings.npy",
        "query_embeddings": run_dir / "query_embeddings.npy",
        "candidate_metadata": run_dir / "candidate_metadata.jsonl",
        "query_metadata": run_dir / "query_metadata.jsonl",
    }
    if set(artifacts) != set(expected_artifacts):
        raise ValueError("smoke artifact inventory is not exact")
    verified_artifacts = {
        name: require_identity(artifacts[name], path, name)
        for name, path in expected_artifacts.items()
    }
    require_embedding(
        expected_artifacts["candidate_embeddings"],
        candidate_shape,
        "candidate embeddings",
    )
    require_embedding(
        expected_artifacts["query_embeddings"], query_shape, "query embeddings"
    )
    require_jsonl_count(
        expected_artifacts["candidate_metadata"],
        EXPECTED_CANDIDATES,
        "candidate metadata",
    )
    require_jsonl_count(
        expected_artifacts["query_metadata"], EXPECTED_QUERIES, "query metadata"
    )
    return {
        "schema_version": 1,
        "status": "passed",
        "gate": "vanilla_backbone_five_sample_smoke_before_full",
        "backbone_id": backbone,
        "git_commit": current_git_commit,
        "smoke_metrics": {
            "path": str(metrics_path.resolve()),
            "size_bytes": metrics_path.stat().st_size,
            "sha256": sha256_file(metrics_path),
        },
        "model_lock": require_model_lock_identity(metrics, model_lock.resolve()),
        "candidate_count": EXPECTED_CANDIDATES,
        "query_count": EXPECTED_QUERIES,
        "embedding_dimension": dimension,
        "verified_artifacts": verified_artifacts,
    }


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], text=True, encoding="utf-8"
    ).strip()


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite smoke gate output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if git_output("status", "--short"):
        raise RuntimeError("smoke gate verification requires a clean Git worktree")
    metrics_path = args.metrics.resolve()
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    result = verify_smoke_gate(
        read_object(metrics_path, "smoke metrics"),
        metrics_path=metrics_path,
        backbone=args.backbone,
        model_lock=args.model_lock.resolve(),
        current_git_commit=git_output("rev-parse", "HEAD"),
    )
    write_once(args.output.resolve(), result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
