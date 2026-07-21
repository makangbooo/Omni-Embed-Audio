#!/usr/bin/env python3
"""Evaluate fixed retrieval embeddings and write a complete audit bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from AudioRetrieval.evaluation.canonical import (  # noqa: E402
    CanonicalRetrievalResult,
    evaluate_caption_to_caption,
    evaluate_grouped_id_retrieval,
    evaluate_id_retrieval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def resolve_path(config_path: Path, raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported evaluation config schema_version")
    required = {
        "checkpoint",
        "dataset",
        "experiment_id",
        "model",
        "task",
        "paper_table",
        "protocol_label",
        "protocol_source",
        "query_embeddings",
        "query_metadata",
        "seed",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"evaluation config missing fields: {missing}")
    if config["task"] not in {"a2t", "t2a", "t2t", "uiq"}:
        raise ValueError("task must be one of: a2t, t2a, t2t, uiq")
    if config["protocol_source"] not in {"PAPER", "CODE", "INFERRED"}:
        raise ValueError("protocol_source must be PAPER, CODE, or INFERRED")
    if not isinstance(config["experiment_id"], str) or not config["experiment_id"]:
        raise ValueError("experiment_id must be a non-empty string")
    for field in ("checkpoint", "dataset", "model"):
        if not isinstance(config[field], str) or not config[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool):
        raise ValueError("seed must be an integer")
    return config


def load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no metadata rows")
    return rows


def metadata_strings(
    rows: list[Mapping[str, Any]], field: str, label: str
) -> list[str]:
    values: list[str] = []
    for index, row in enumerate(rows):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} row {index}: {field} must be non-empty")
        values.append(value.strip())
    return values


def load_query_indices(config_path: Path, config: Mapping[str, Any]) -> list[int] | None:
    selection = config.get("query_selection")
    indices_path = resolve_path(config_path, config.get("query_indices"))
    if selection == "all":
        if indices_path is not None:
            raise ValueError("query_selection=all cannot include query_indices")
        return None
    if selection != "indices":
        raise ValueError("query_selection must be exactly 'all' or 'indices'")
    if indices_path is None:
        raise ValueError("query_selection=indices requires query_indices")
    raw = json.loads(indices_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("query_indices file must contain a non-empty JSON list")
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in raw):
        raise ValueError("query_indices must contain integers only")
    return raw


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def save_result_artifacts(
    output_dir: Path,
    result: CanonicalRetrievalResult,
) -> None:
    write_npy(output_dir / "ranks.npy", result.ranks)
    write_npy(output_dir / "rankings.npy", result.rankings)
    write_npy(output_dir / "similarities.npy", result.similarities)
    write_json(
        output_dir / "positive_indices.json",
        [list(row) for row in result.positive_indices],
    )
    write_json(
        output_dir / "ignored_indices.json",
        [list(row) for row in result.ignored_indices],
    )
    write_json(
        output_dir / "evaluated_query_indices.json",
        result.evaluated_query_indices.tolist(),
    )


def run_evaluation(
    *,
    config_path: Path,
    output_dir: Path,
    argv: list[str] | None = None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_preexisting = {
        "command.sh",
        "environment.txt",
        "git_commit.txt",
        "git_status.txt",
        "gpu_info.txt",
        "stdout.log",
        "stderr.log",
    }
    unexpected = sorted(
        path.name
        for path in output_dir.iterdir()
        if path.name not in allowed_preexisting
    )
    if unexpected:
        raise FileExistsError(
            f"output directory contains prior artifacts: {unexpected}"
        )
    started_at = utc_now()
    try:
        config = load_config(config_path)
        if config.get("enforce_experiment_id_directory", True):
            if output_dir.name != config["experiment_id"]:
                raise ValueError(
                    "output directory basename must equal experiment_id: "
                    f"{output_dir.name!r} != {config['experiment_id']!r}"
                )
        git_commit = git_output("rev-parse", "HEAD")
        git_status = git_output("status", "--short")
    except BaseException as exc:
        early_report = {
            "schema_version": 1,
            "status": "failed",
            "started_at": started_at,
            "finished_at": utc_now(),
            "config": str(config_path),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(output_dir / "metrics.json", early_report)
        raise
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": started_at,
        "finished_at": None,
        "experiment_id": config["experiment_id"],
        "model": config["model"],
        "dataset": config["dataset"],
        "checkpoint": config["checkpoint"],
        "seed": config["seed"],
        "randomness_used_by_evaluator": False,
        "task": config["task"],
        "paper_table": config["paper_table"],
        "protocol_label": config["protocol_label"],
        "protocol_source": config["protocol_source"],
        "git_commit": git_commit,
        "git_status_short": git_status,
        "error": None,
    }
    write_json(output_dir / "metrics.json", report)
    (output_dir / "git_commit.txt").write_text(git_commit + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(git_status + "\n", encoding="utf-8")
    command = argv if argv is not None else sys.argv
    command_path = output_dir / (
        "python_command.sh" if (output_dir / "command.sh").exists() else "command.sh"
    )
    command_path.write_text(
        " ".join(shlex.quote(item) for item in command) + "\n", encoding="utf-8"
    )
    try:
        if config.get("require_clean_git", True) and git_status:
            raise RuntimeError(
                "formal evaluation requires a clean Git worktree; "
                f"status={git_status!r}"
            )
        query_embeddings_path = resolve_path(config_path, config["query_embeddings"])
        query_metadata_path = resolve_path(config_path, config["query_metadata"])
        candidate_embeddings_path = resolve_path(
            config_path, config.get("candidate_embeddings")
        )
        candidate_metadata_path = resolve_path(
            config_path, config.get("candidate_metadata")
        )
        assert query_embeddings_path is not None
        assert query_metadata_path is not None
        if config["task"] == "t2t":
            candidate_embeddings_path = candidate_embeddings_path or query_embeddings_path
            candidate_metadata_path = candidate_metadata_path or query_metadata_path
        elif candidate_embeddings_path is None or candidate_metadata_path is None:
            raise ValueError("a2t/t2a/uiq require candidate embeddings and metadata")

        input_paths = {
            "config": config_path,
            "query_embeddings": query_embeddings_path,
            "candidate_embeddings": candidate_embeddings_path,
            "query_metadata": query_metadata_path,
            "candidate_metadata": candidate_metadata_path,
        }
        indices_path = resolve_path(config_path, config.get("query_indices"))
        if indices_path is not None:
            input_paths["query_indices"] = indices_path
        for path in input_paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        report["inputs"] = {
            name: file_identity(path) for name, path in input_paths.items()
        }

        query_embeddings = np.load(query_embeddings_path, allow_pickle=False)
        candidate_embeddings = np.load(candidate_embeddings_path, allow_pickle=False)
        query_metadata = load_jsonl_objects(query_metadata_path)
        candidate_metadata = load_jsonl_objects(candidate_metadata_path)
        if query_embeddings.shape[0] != len(query_metadata):
            raise ValueError("query embedding rows do not match query metadata rows")
        if candidate_embeddings.shape[0] != len(candidate_metadata):
            raise ValueError(
                "candidate embedding rows do not match candidate metadata rows"
            )
        query_indices = load_query_indices(config_path, config)

        if config["task"] == "t2t":
            if query_embeddings_path != candidate_embeddings_path:
                raise ValueError(
                    "canonical caption T2T currently requires one shared embedding bank"
                )
            clip_ids = metadata_strings(
                query_metadata,
                str(config.get("query_clip_id_field", "clip_id")),
                "query metadata",
            )
            result = evaluate_caption_to_caption(
                query_embeddings,
                clip_ids,
                query_indices=query_indices,
                normalize=bool(config.get("normalize_embeddings", True)),
            )
        elif config["task"] == "a2t":
            target_ids = metadata_strings(
                query_metadata,
                str(config.get("query_target_id_field", "target_id")),
                "query metadata",
            )
            candidate_group_ids = metadata_strings(
                candidate_metadata,
                str(config.get("candidate_group_id_field", "clip_id")),
                "candidate metadata",
            )
            result = evaluate_grouped_id_retrieval(
                query_embeddings,
                target_ids,
                candidate_embeddings,
                candidate_group_ids,
                query_indices=query_indices,
                normalize=bool(config.get("normalize_embeddings", True)),
            )
        else:
            target_ids = metadata_strings(
                query_metadata,
                str(config.get("query_target_id_field", "target_id")),
                "query metadata",
            )
            candidate_ids = metadata_strings(
                candidate_metadata,
                str(config.get("candidate_id_field", "candidate_id")),
                "candidate metadata",
            )
            result = evaluate_id_retrieval(
                query_embeddings,
                target_ids,
                candidate_embeddings,
                candidate_ids,
                query_indices=query_indices,
                normalize=bool(config.get("normalize_embeddings", True)),
            )

        copy_file(query_embeddings_path, output_dir / "query_embeddings.npy")
        copy_file(candidate_embeddings_path, output_dir / "candidate_embeddings.npy")
        write_jsonl(output_dir / "query_metadata.jsonl", query_metadata)
        write_jsonl(output_dir / "candidate_metadata.jsonl", candidate_metadata)
        resolved_config = dict(config)
        resolved_config["resolved_inputs"] = {
            name: str(path) for name, path in input_paths.items()
        }
        write_json(output_dir / "config.yaml", resolved_config)
        save_result_artifacts(output_dir, result)
        environment_lines = [
            f"python={platform.python_version()}",
            f"python_executable={sys.executable}",
            f"platform={platform.platform()}",
            f"numpy={np.__version__}",
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
        ]
        environment_path = output_dir / (
            "python_environment.txt"
            if (output_dir / "environment.txt").exists()
            else "environment.txt"
        )
        environment_path.write_text(
            "\n".join(environment_lines) + "\n", encoding="utf-8"
        )
        report.update(
            {
                "status": "complete",
                "finished_at": utc_now(),
                "query_embedding_shape": list(query_embeddings.shape),
                "candidate_embedding_shape": list(candidate_embeddings.shape),
                "evaluated_query_count": int(result.ranks.size),
                "candidate_count": int(candidate_embeddings.shape[0]),
                "query_selection": config["query_selection"],
                "normalization_applied": bool(
                    config.get("normalize_embeddings", True)
                ),
                "tie_policy": result.tie_policy,
                "metrics": result.metrics,
                "artifacts": {
                    name: file_identity(output_dir / name)
                    for name in (
                        "query_embeddings.npy",
                        "candidate_embeddings.npy",
                        "query_metadata.jsonl",
                        "candidate_metadata.jsonl",
                        "ranks.npy",
                        "rankings.npy",
                        "similarities.npy",
                        "positive_indices.json",
                        "ignored_indices.json",
                        "evaluated_query_indices.json",
                    )
                },
            }
        )
        write_json(output_dir / "metrics.json", report)
        return report
    except BaseException as exc:
        report.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(output_dir / "metrics.json", report)
        raise


def main() -> int:
    args = parse_args()
    report = run_evaluation(
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
