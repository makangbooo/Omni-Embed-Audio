#!/usr/bin/env python3
"""Evaluate explicit negative-query embedding triples and save audit artifacts."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from AudioRetrieval.evaluation.negative_canonical import (  # noqa: E402
    CanonicalNegativeRetrievalResult,
    evaluate_negative_id_retrieval,
)
from scripts.evaluate_embedding_artifacts import (  # noqa: E402
    copy_file,
    file_identity,
    git_output as _shared_git_output,
    load_jsonl_objects,
    load_query_indices,
    metadata_strings,
    resolve_path,
    write_json,
    write_jsonl,
    write_npy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_output(*arguments: str) -> str:
    """Indirection retained so tests can isolate Git provenance calls."""

    return _shared_git_output(*arguments)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported negative evaluation config schema_version")
    required = {
        "candidate_embeddings",
        "candidate_metadata",
        "checkpoint",
        "dataset",
        "experiment_id",
        "ks",
        "model",
        "pairing_metadata",
        "paper_table",
        "protocol_label",
        "protocol_source",
        "query_embeddings",
        "query_metadata",
        "query_selection",
        "seed",
        "task",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"negative evaluation config missing fields: {missing}")
    if config["task"] != "negative":
        raise ValueError("negative evaluator requires task='negative'")
    if config["protocol_source"] not in {"PAPER", "CODE", "INFERRED"}:
        raise ValueError("protocol_source must be PAPER, CODE, or INFERRED")
    for field in (
        "checkpoint",
        "dataset",
        "experiment_id",
        "model",
        "paper_table",
        "protocol_label",
    ):
        if not isinstance(config[field], str) or not config[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool):
        raise ValueError("seed must be an integer")
    validate_ks(config["ks"])
    return config


def validate_ks(raw: Any) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("ks must be a non-empty JSON list")
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in raw):
        raise ValueError("ks must contain integers only")
    values = tuple(raw)
    if any(value < 1 for value in values):
        raise ValueError("ks must contain positive integers")
    if len(set(values)) != len(values):
        raise ValueError("ks must not contain duplicate cutoffs")
    return values


def align_explicit_pairings(
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
    pairing_rows: list[Mapping[str, Any]],
    *,
    pairing_query_id_field: str,
    pairing_target_id_field: str,
    pairing_hard_negative_id_field: str,
) -> tuple[list[str], list[str]]:
    """Align a complete explicit pairing table to query metadata order."""

    if len(set(query_ids)) != len(query_ids):
        raise ValueError("query metadata query IDs must be unique")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate metadata candidate IDs must be unique")

    by_query: dict[str, tuple[str, str]] = {}
    for row_index, row in enumerate(pairing_rows):
        values: list[str] = []
        for field in (
            pairing_query_id_field,
            pairing_target_id_field,
            pairing_hard_negative_id_field,
        ):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"pairing row {row_index}: {field} must be a non-empty string"
                )
            values.append(value.strip())
        query_id, target_id, hard_negative_id = values
        if query_id in by_query:
            raise ValueError(f"duplicate pairing query ID: {query_id!r}")
        if target_id == hard_negative_id:
            raise ValueError(
                "target_id and hard_negative_id must differ for query "
                f"{query_id!r}"
            )
        by_query[query_id] = (target_id, hard_negative_id)

    expected = set(query_ids)
    actual = set(by_query)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            "pairing query IDs must exactly match query metadata; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    candidate_set = set(candidate_ids)
    missing_targets = sorted(
        {target for target, _ in by_query.values() if target not in candidate_set}
    )
    missing_hard_negatives = sorted(
        {
            hard_negative
            for _, hard_negative in by_query.values()
            if hard_negative not in candidate_set
        }
    )
    if missing_targets:
        raise KeyError(
            f"target IDs are absent from candidates: {missing_targets[:10]}"
        )
    if missing_hard_negatives:
        raise KeyError(
            "hard-negative IDs are absent from candidates: "
            f"{missing_hard_negatives[:10]}"
        )

    target_ids = [by_query[query_id][0] for query_id in query_ids]
    hard_negative_ids = [by_query[query_id][1] for query_id in query_ids]
    return target_ids, hard_negative_ids


def per_query_rows(
    result: CanonicalNegativeRetrievalResult,
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
    ks: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for output_row, query_index in enumerate(
        result.evaluated_query_indices.tolist()
    ):
        target_index = int(result.target_indices[output_row])
        hard_negative_index = int(result.hard_negative_indices[output_row])
        target_rank = int(result.target_ranks[output_row])
        hard_negative_rank = int(result.hard_negative_ranks[output_row])
        target_score = float(result.similarities[output_row, target_index])
        hard_negative_score = float(
            result.similarities[output_row, hard_negative_index]
        )
        row: dict[str, Any] = {
            "query_index": query_index,
            "query_id": query_ids[query_index],
            "target_id": candidate_ids[target_index],
            "target_index": target_index,
            "target_rank": target_rank,
            "target_score": target_score,
            "hard_negative_id": candidate_ids[hard_negative_index],
            "hard_negative_index": hard_negative_index,
            "hard_negative_rank": hard_negative_rank,
            "hard_negative_score": hard_negative_score,
            "delta_rank": hard_negative_rank - target_rank,
            "HNSR": hard_negative_rank > target_rank,
            "TFR": target_rank == 1,
        }
        for cutoff in ks:
            row[f"R@{cutoff}"] = target_rank <= cutoff
            row[f"HNSR@{cutoff}"] = (
                target_rank <= cutoff and hard_negative_rank > cutoff
            )
            row[f"TFR-HN@{cutoff}"] = (
                target_rank == 1 and hard_negative_rank > cutoff
            )
        rows.append(row)
    return rows


def save_result_artifacts(
    output_dir: Path,
    result: CanonicalNegativeRetrievalResult,
    rows: list[Mapping[str, Any]],
) -> None:
    write_npy(output_dir / "target_ranks.npy", result.target_ranks)
    write_npy(
        output_dir / "hard_negative_ranks.npy", result.hard_negative_ranks
    )
    write_npy(output_dir / "rankings.npy", result.rankings)
    write_npy(output_dir / "similarities.npy", result.similarities)
    write_npy(output_dir / "target_indices.npy", result.target_indices)
    write_npy(
        output_dir / "hard_negative_indices.npy", result.hard_negative_indices
    )
    write_npy(
        output_dir / "evaluated_query_indices.npy",
        result.evaluated_query_indices,
    )
    write_jsonl(output_dir / "per_query.jsonl", rows)


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
        "task": "negative",
        "paper_table": config["paper_table"],
        "protocol_label": config["protocol_label"],
        "protocol_source": config["protocol_source"],
        "pairing_contract": "explicit_query_target_hard_negative_ids",
        "pairing_inference_used": False,
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
        candidate_embeddings_path = resolve_path(
            config_path, config["candidate_embeddings"]
        )
        query_metadata_path = resolve_path(config_path, config["query_metadata"])
        candidate_metadata_path = resolve_path(
            config_path, config["candidate_metadata"]
        )
        pairing_metadata_path = resolve_path(config_path, config["pairing_metadata"])
        assert query_embeddings_path is not None
        assert candidate_embeddings_path is not None
        assert query_metadata_path is not None
        assert candidate_metadata_path is not None
        assert pairing_metadata_path is not None
        input_paths = {
            "config": config_path,
            "query_embeddings": query_embeddings_path,
            "candidate_embeddings": candidate_embeddings_path,
            "query_metadata": query_metadata_path,
            "candidate_metadata": candidate_metadata_path,
            "pairing_metadata": pairing_metadata_path,
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
        candidate_embeddings = np.load(
            candidate_embeddings_path, allow_pickle=False
        )
        query_metadata = load_jsonl_objects(query_metadata_path)
        candidate_metadata = load_jsonl_objects(candidate_metadata_path)
        pairing_metadata = load_jsonl_objects(pairing_metadata_path)
        if query_embeddings.ndim != 2:
            raise ValueError("query embeddings must be a 2-D matrix")
        if candidate_embeddings.ndim != 2:
            raise ValueError("candidate embeddings must be a 2-D matrix")
        if query_embeddings.shape[0] != len(query_metadata):
            raise ValueError("query embedding rows do not match query metadata rows")
        if candidate_embeddings.shape[0] != len(candidate_metadata):
            raise ValueError(
                "candidate embedding rows do not match candidate metadata rows"
            )

        query_id_field = str(config.get("query_id_field", "query_id"))
        candidate_id_field = str(
            config.get("candidate_id_field", "candidate_id")
        )
        pairing_query_id_field = str(
            config.get("pairing_query_id_field", "query_id")
        )
        pairing_target_id_field = str(
            config.get("pairing_target_id_field", "target_id")
        )
        pairing_hard_negative_id_field = str(
            config.get("pairing_hard_negative_id_field", "hard_negative_id")
        )
        query_ids = metadata_strings(
            query_metadata, query_id_field, "query metadata"
        )
        candidate_ids = metadata_strings(
            candidate_metadata, candidate_id_field, "candidate metadata"
        )
        target_ids, hard_negative_ids = align_explicit_pairings(
            query_ids,
            candidate_ids,
            pairing_metadata,
            pairing_query_id_field=pairing_query_id_field,
            pairing_target_id_field=pairing_target_id_field,
            pairing_hard_negative_id_field=pairing_hard_negative_id_field,
        )
        query_indices = load_query_indices(config_path, config)
        ks = validate_ks(config["ks"])
        result = evaluate_negative_id_retrieval(
            query_embeddings,
            query_ids,
            candidate_embeddings,
            candidate_ids,
            target_ids,
            hard_negative_ids,
            query_indices=query_indices,
            normalize=bool(config.get("normalize_embeddings", True)),
            ks=ks,
        )
        rows = per_query_rows(result, query_ids, candidate_ids, ks)

        copy_file(query_embeddings_path, output_dir / "query_embeddings.npy")
        copy_file(
            candidate_embeddings_path, output_dir / "candidate_embeddings.npy"
        )
        write_jsonl(output_dir / "query_metadata.jsonl", query_metadata)
        write_jsonl(output_dir / "candidate_metadata.jsonl", candidate_metadata)
        write_jsonl(output_dir / "pairing_metadata.jsonl", pairing_metadata)
        resolved_config = dict(config)
        resolved_config["resolved_inputs"] = {
            name: str(path) for name, path in input_paths.items()
        }
        write_json(output_dir / "config.yaml", resolved_config)
        save_result_artifacts(output_dir, result, rows)

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

        artifact_names = (
            "query_embeddings.npy",
            "candidate_embeddings.npy",
            "query_metadata.jsonl",
            "candidate_metadata.jsonl",
            "pairing_metadata.jsonl",
            "target_ranks.npy",
            "hard_negative_ranks.npy",
            "rankings.npy",
            "similarities.npy",
            "target_indices.npy",
            "hard_negative_indices.npy",
            "evaluated_query_indices.npy",
            "per_query.jsonl",
        )
        report.update(
            {
                "status": "complete",
                "finished_at": utc_now(),
                "query_embedding_shape": list(query_embeddings.shape),
                "candidate_embedding_shape": list(candidate_embeddings.shape),
                "evaluated_query_count": int(result.target_ranks.size),
                "candidate_count": int(candidate_embeddings.shape[0]),
                "query_selection": config["query_selection"],
                "ks": list(ks),
                "normalization_applied": bool(
                    config.get("normalize_embeddings", True)
                ),
                "tie_policy": result.tie_policy,
                "metrics": result.metrics,
                "artifacts": {
                    name: file_identity(output_dir / name)
                    for name in artifact_names
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
    report = run_evaluation(config_path=args.config, output_dir=args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
