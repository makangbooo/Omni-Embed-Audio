#!/usr/bin/env python3
"""Verify an immutable lock-bound OEA smoke before a full embedding run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from scripts.build_official_oea_eval_config import (
    compare_protocol_to_lock,
    validate_model_lock,
    verify_official_model_lock_binding,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CANDIDATES = 5
EXPECTED_QUERIES = 25
CRITICAL_INFERENCE_PATHS = (
    "scripts/generate_oea_embeddings.py",
    "scripts/build_official_oea_eval_config.py",
    "AudioRetrieval/models/omni_embed_adapter.py",
    "AudioRetrieval/training/oea/train_omniembed_lora.py",
)
CORE_EQUAL_FIELDS = (
    "official_variant_id",
    "model",
    "seed",
    "caption_count_per_audio",
    "audio_batch_size",
    "text_batch_size",
    "audio_prompt_protocol",
    "model_config",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--full-protocol-config", type=Path, required=True)
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
    identity: Any, expected_path: Path, label: str
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
    if not isinstance(sha_value, str) or len(sha_value) != 64:
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


def require_repository_identity(
    identity: Any,
    expected_path: Path,
    repository_root: Path,
    label: str,
) -> dict[str, Any]:
    if not isinstance(identity, dict):
        raise ValueError(f"{label} identity must be an object")
    expected_path = expected_path.resolve()
    expected_relative = expected_path.relative_to(repository_root.resolve()).as_posix()
    if identity.get("repository_path") != expected_relative:
        raise ValueError(f"{label} repository path differs")
    if identity.get("size_bytes") != expected_path.stat().st_size:
        raise ValueError(f"{label} size differs")
    actual_sha = sha256_file(expected_path)
    if identity.get("sha256") != actual_sha:
        raise ValueError(f"{label} SHA256 differs")
    return {
        "repository_path": expected_relative,
        "size_bytes": expected_path.stat().st_size,
        "sha256": actual_sha,
    }


def require_embedding(path: Path, expected_shape: list[int], label: str) -> None:
    value = np.load(path, allow_pickle=False)
    if list(value.shape) != expected_shape:
        raise ValueError(f"{label} array shape differs from smoke metrics")
    if not np.isfinite(value).all():
        raise ValueError(f"{label} contains non-finite values")
    if not np.allclose(np.linalg.norm(value, axis=1), 1.0, atol=1e-3):
        raise ValueError(f"{label} is not L2 normalized")


def require_jsonl_count(path: Path, expected: int, label: str) -> None:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{label} line {line_number} is blank")
            if not isinstance(json.loads(line), dict):
                raise ValueError(f"{label} line {line_number} is not an object")
            count += 1
    if count != expected:
        raise ValueError(f"{label} row count differs from smoke metrics")


def require_protocol_core(
    smoke_config: Mapping[str, Any],
    full_protocol: Mapping[str, Any],
    locked: Mapping[str, Any],
) -> None:
    for field in CORE_EQUAL_FIELDS:
        if smoke_config.get(field) != full_protocol.get(field):
            raise ValueError(f"smoke/full protocol core differs at {field}")
    for field in ("repo_id", "revision", "local_subdir"):
        if full_protocol["base_model"].get(field) != locked["base_model"][field]:
            raise ValueError(f"full protocol base_model.{field} differs from lock")
        if smoke_config["base_model"].get(field) != locked["base_model"][field]:
            raise ValueError(f"smoke config base_model.{field} differs from lock")
    if smoke_config.get("checkpoint") != locked["checkpoint"]:
        raise ValueError("smoke config checkpoint differs from lock")


def verify_smoke_gate(
    metrics: Mapping[str, Any],
    *,
    metrics_path: Path,
    model_lock_path: Path,
    full_protocol_path: Path,
    current_git_commit: str,
    smoke_commit_is_ancestor: bool,
    changed_critical_files: list[str],
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    metrics_path = metrics_path.resolve()
    model_lock_path = model_lock_path.resolve()
    full_protocol_path = full_protocol_path.resolve()
    smoke_config_path = metrics_path.parent / "config.yaml"
    model_lock = read_object(model_lock_path, "official model lock")
    locked = validate_model_lock(model_lock)
    full_protocol = read_object(full_protocol_path, "full protocol config")
    compare_protocol_to_lock(full_protocol, locked)
    smoke_config = read_object(smoke_config_path, "resolved smoke config")
    smoke_binding = verify_official_model_lock_binding(
        smoke_config, repository_root=repository_root
    )

    if metrics.get("schema_version") != 1 or metrics.get("status") != "complete":
        raise ValueError("smoke generation metrics are not complete schema-v1 output")
    if metrics.get("error") is not None:
        raise ValueError("smoke generation records a non-null error")
    smoke_commit = metrics.get("git_commit")
    if not isinstance(smoke_commit, str) or len(smoke_commit) != 40:
        raise ValueError("smoke generation git_commit is invalid")
    if smoke_config.get("resolution_git_commit") != smoke_commit:
        raise ValueError("resolved smoke config commit differs from smoke metrics")
    if not smoke_commit_is_ancestor:
        raise ValueError("smoke generation commit is not an ancestor of current HEAD")
    if changed_critical_files:
        raise ValueError(
            "critical inference files changed after smoke: "
            + ", ".join(changed_critical_files)
        )
    if metrics.get("model") != locked["model"]:
        raise ValueError("smoke model differs from model lock")
    require_protocol_core(smoke_config, full_protocol, locked)
    run_dir = metrics_path.parent
    if metrics.get("experiment_id") != run_dir.name:
        raise ValueError("smoke experiment_id differs from its run directory")
    if not run_dir.name.startswith(smoke_config["experiment_prefix"] + "_"):
        raise ValueError("smoke run directory differs from its experiment prefix")
    resolved_paths = smoke_config.get("resolved_paths")
    if not isinstance(resolved_paths, dict):
        raise ValueError("resolved smoke config is missing resolved_paths")
    output_path = resolved_paths.get("output_dir")
    if not isinstance(output_path, str) or Path(output_path).resolve() != run_dir:
        raise ValueError("resolved smoke output directory differs from metrics")
    for field in ("dataset", "seed", "audio_prompt_protocol"):
        if metrics.get(field) != smoke_config.get(field):
            raise ValueError(f"smoke metrics/config differ at {field}")

    binding = metrics.get("official_model_lock")
    if not isinstance(binding, dict):
        raise ValueError("smoke metrics are missing official_model_lock")
    if binding.get("variant_id") != locked["variant_id"]:
        raise ValueError("smoke variant differs from model lock")
    if binding.get("model") != locked["model"]:
        raise ValueError("smoke binding model differs from model lock")
    lock_identity = require_repository_identity(
        binding.get("model_lock"), model_lock_path, repository_root, "model lock"
    )
    if smoke_binding["model_lock"] != lock_identity:
        raise ValueError("resolved smoke config model-lock binding differs")
    protocol_binding = binding.get("protocol_config")
    if not isinstance(protocol_binding, dict):
        raise ValueError("smoke metrics are missing protocol_config identity")
    repository_path = protocol_binding.get("repository_path")
    if not isinstance(repository_path, str) or not repository_path:
        raise ValueError("smoke protocol repository path is invalid")
    smoke_protocol_path = (repository_root / repository_path).resolve()
    protocol_identity = require_repository_identity(
        binding.get("protocol_config"),
        smoke_protocol_path,
        repository_root,
        "smoke protocol",
    )
    if smoke_binding["protocol_config"] != protocol_identity:
        raise ValueError("resolved smoke config protocol binding differs")

    strict_offline = metrics.get("strict_offline")
    expected_offline = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    if strict_offline != expected_offline:
        raise ValueError("smoke generation was not strict-offline")
    expected_scalars = {
        "candidate_count": EXPECTED_CANDIDATES,
        "query_count": EXPECTED_QUERIES,
        "completed_audio_chunks": EXPECTED_CANDIDATES,
        "completed_text_chunks": EXPECTED_QUERIES,
        "pending_audio_chunks": 0,
        "pending_text_chunks": 0,
    }
    for field, expected in expected_scalars.items():
        if metrics.get(field) != expected:
            raise ValueError(f"smoke {field} differs from fixed contract")
    dimension = full_protocol["model_config"]["projection_dim"]["value"]
    candidate_shape = [EXPECTED_CANDIDATES, dimension]
    query_shape = [EXPECTED_QUERIES, dimension]
    if metrics.get("candidate_embedding_shape") != candidate_shape:
        raise ValueError("candidate embedding shape differs from fixed contract")
    if metrics.get("query_embedding_shape") != query_shape:
        raise ValueError("query embedding shape differs from fixed contract")
    model_load = metrics.get("model_load")
    measured = model_lock.get("measured_structure")
    if not isinstance(model_load, dict) or not isinstance(measured, dict):
        raise ValueError("smoke or model lock is missing measured model structure")
    if model_load.get("lora_tensor_count") != measured.get("lora_tensor_count"):
        raise ValueError("smoke LoRA tensor count differs from model lock")
    if model_load.get("projection_dim") != dimension:
        raise ValueError("smoke projection dimension differs from full protocol")

    artifacts = metrics.get("artifacts")
    expected_artifacts = {
        "candidate_embeddings": run_dir / "candidate_embeddings.npy",
        "query_embeddings": run_dir / "query_embeddings.npy",
        "candidate_metadata": run_dir / "candidate_metadata.jsonl",
        "query_metadata": run_dir / "query_metadata.jsonl",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != set(expected_artifacts):
        raise ValueError("smoke artifact inventory is not exact")
    verified_artifacts = {
        name: require_identity(artifacts[name], path, name)
        for name, path in expected_artifacts.items()
    }
    require_embedding(
        expected_artifacts["candidate_embeddings"], candidate_shape, "candidate embeddings"
    )
    require_embedding(
        expected_artifacts["query_embeddings"], query_shape, "query embeddings"
    )
    require_jsonl_count(
        expected_artifacts["candidate_metadata"], EXPECTED_CANDIDATES, "candidate metadata"
    )
    require_jsonl_count(
        expected_artifacts["query_metadata"], EXPECTED_QUERIES, "query metadata"
    )
    return {
        "schema_version": 1,
        "status": "passed",
        "gate": "official_oea_lock_bound_five_sample_smoke_before_full",
        "smoke_git_commit": smoke_commit,
        "current_git_commit": current_git_commit,
        "critical_inference_paths": list(CRITICAL_INFERENCE_PATHS),
        "changed_critical_files": changed_critical_files,
        "smoke_metrics": {
            "path": str(metrics_path),
            "size_bytes": metrics_path.stat().st_size,
            "sha256": sha256_file(metrics_path),
        },
        "smoke_config": {
            "path": str(smoke_config_path.resolve()),
            "size_bytes": smoke_config_path.stat().st_size,
            "sha256": sha256_file(smoke_config_path),
        },
        "model_lock": lock_identity,
        "smoke_protocol": protocol_identity,
        "full_protocol": require_repository_identity(
            {
                "repository_path": full_protocol_path.relative_to(repository_root.resolve()).as_posix(),
                "size_bytes": full_protocol_path.stat().st_size,
                "sha256": sha256_file(full_protocol_path),
            },
            full_protocol_path,
            repository_root,
            "full protocol",
        ),
        "candidate_count": EXPECTED_CANDIDATES,
        "query_count": EXPECTED_QUERIES,
        "embedding_dimension": dimension,
        "verified_artifacts": verified_artifacts,
    }


def git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def require_tracked(path: Path, label: str) -> None:
    relative = path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    try:
        git_output("ls-files", "--error-unmatch", "--", relative)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{label} is not tracked by Git: {relative}") from exc


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
    model_lock_path = args.model_lock.resolve()
    full_protocol_path = args.full_protocol_config.resolve()
    for path, label in (
        (metrics_path, "smoke metrics"),
        (metrics_path.parent / "config.yaml", "resolved smoke config"),
        (model_lock_path, "official model lock"),
        (full_protocol_path, "full protocol config"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")
    require_tracked(model_lock_path, "official model lock")
    require_tracked(full_protocol_path, "full protocol config")
    metrics = read_object(metrics_path, "smoke metrics")
    smoke_commit = metrics.get("git_commit")
    if not isinstance(smoke_commit, str):
        raise ValueError("smoke generation git_commit is invalid")
    current_commit = git_output("rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", smoke_commit, current_commit],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    if ancestor.returncode not in (0, 1):
        raise RuntimeError("git could not compare smoke and current commits")
    changed = git_output(
        "diff",
        "--name-only",
        f"{smoke_commit}..{current_commit}",
        "--",
        *CRITICAL_INFERENCE_PATHS,
    ).splitlines()
    result = verify_smoke_gate(
        metrics,
        metrics_path=metrics_path,
        model_lock_path=model_lock_path,
        full_protocol_path=full_protocol_path,
        current_git_commit=current_commit,
        smoke_commit_is_ancestor=ancestor.returncode == 0,
        changed_critical_files=changed,
    )
    write_once(args.output.resolve(), result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
