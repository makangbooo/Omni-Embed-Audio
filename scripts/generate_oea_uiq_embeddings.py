#!/usr/bin/env python3
"""Generate resumable OEA embeddings for fixed positive UIQ releases."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from AudioRetrieval.evaluation.uiq_schema import load_released_uiq
from scripts.build_official_oea_eval_config import (
    verify_official_model_lock_binding,
)
from scripts.generate_oea_embeddings import (
    atomic_write_json,
    consolidate_chunks,
    ensure_run_identity,
    file_identity,
    immutable_json_text,
    immutable_jsonl_text,
    load_config as load_base_embedding_config,
    load_manifest,
    load_model_bundle,
    load_verified_chunk,
    project_batch,
    ranges,
    save_chunk,
    tagged_value,
    verify_fixed_file,
    write_text_once_or_verify,
)


RELEASED_QUERY_TYPES = ("question", "imperative", "paraphrase", "tagging")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-embedding-config", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attempt-dir", type=Path)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def print_progress(
    *,
    completed: int,
    total: int,
    completed_this_attempt: int,
    stage_started: float,
    overall_started_at: str,
) -> None:
    stage_elapsed = max(time.monotonic() - stage_started, 1e-9)
    throughput = completed_this_attempt / stage_elapsed
    remaining = max(total - completed, 0)
    remaining_seconds = remaining / throughput if throughput > 0 else float("inf")
    try:
        overall_started = datetime.fromisoformat(overall_started_at)
        overall_elapsed = (datetime.now(timezone.utc) - overall_started).total_seconds()
    except (TypeError, ValueError):
        overall_elapsed = stage_elapsed
    expected_completion = (
        datetime.now(timezone.utc) + timedelta(seconds=remaining_seconds)
        if remaining_seconds != float("inf")
        else None
    )
    eta = format_duration(remaining_seconds) if expected_completion else "unknown"
    completion = expected_completion.isoformat() if expected_completion else "unknown"
    print(
        "[PROGRESS] stage=uiq_text "
        f"current={completed} total={total} percent={completed / total * 100:.2f}% "
        f"stage_elapsed={format_duration(stage_elapsed)} "
        f"overall_elapsed={format_duration(overall_elapsed)} "
        f"throughput={throughput:.3f}_queries_per_second "
        f"stage_remaining={eta} overall_remaining={eta} "
        f"expected_completion={completion}",
        flush=True,
    )


def git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def repository_file(
    raw_path: str, *, resource_root: Path = REPOSITORY_ROOT
) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("repository resource path must be a non-empty string")
    path = Path(raw_path)
    if path.is_absolute():
        raise ValueError("repository resource path must be relative")
    root = resource_root.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("repository resource path escapes repository root") from exc
    return resolved


def load_uiq_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported UIQ embedding config schema_version")
    required = {
        "base_embedding_config",
        "dataset",
        "expected_embedding_dim",
        "expected_examples",
        "expected_total_queries",
        "experiment_prefix",
        "model",
        "query_sources",
        "seed",
        "text_batch_size",
        "text_protocol",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"UIQ embedding config missing fields: {missing}")
    for field in (
        "expected_embedding_dim",
        "expected_examples",
        "expected_total_queries",
        "text_batch_size",
    ):
        value = config[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool):
        raise ValueError("seed must be an integer")
    sources = config["query_sources"]
    if not isinstance(sources, list) or len(sources) != len(RELEASED_QUERY_TYPES):
        raise ValueError("query_sources must contain exactly four entries")
    released_types = []
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(f"query source {index} must be an object")
        source_required = {
            "expected_rows",
            "paper_query_type",
            "path",
            "released_query_type",
            "sha256",
            "size_bytes",
        }
        source_missing = sorted(source_required - set(source))
        if source_missing:
            raise ValueError(
                f"query source {index} missing fields: {source_missing}"
            )
        released_types.append(source["released_query_type"])
        if source["expected_rows"] != config["expected_examples"]:
            raise ValueError(f"query source {index}: expected_rows mismatch")
    if tuple(released_types) != RELEASED_QUERY_TYPES:
        raise ValueError(
            "query_sources must preserve question/imperative/paraphrase/tagging order"
        )
    expected_total = config["expected_examples"] * len(RELEASED_QUERY_TYPES)
    if config["expected_total_queries"] != expected_total:
        raise ValueError("expected_total_queries does not match source counts")
    return config


def verify_repository_resource(
    specification: Mapping[str, Any], *, resource_root: Path = REPOSITORY_ROOT
) -> dict[str, Any]:
    root = resource_root.resolve()
    path = repository_file(
        str(specification["path"]), resource_root=root
    )
    identity = verify_fixed_file(path, specification)
    identity["repository_relative_path"] = path.relative_to(root).as_posix()
    return identity


def build_uiq_query_metadata(
    manifest_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    resource_root: Path = REPOSITORY_ROOT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sample_ids = [str(row["sample_id"]) for row in manifest_rows]
    expected_id_set = set(sample_ids)
    output: list[dict[str, Any]] = []
    source_identities: list[dict[str, Any]] = []

    for source in config["query_sources"]:
        identity = verify_repository_resource(source, resource_root=resource_root)
        source_path = Path(identity["path"])
        query_type = str(source["released_query_type"])
        released = load_released_uiq(
            source_path,
            expected_dataset="clotho",
            expected_query_type=query_type,
            require_unique_audio_ids=True,
        )
        if len(released) != int(source["expected_rows"]):
            raise ValueError(f"{query_type}: released row count mismatch")
        released_by_id = {row.audio_id: row for row in released}
        released_ids = set(released_by_id)
        if released_ids != expected_id_set:
            raise ValueError(
                f"{query_type}: UIQ/canonical manifest ID mismatch: "
                f"uiq_only={sorted(released_ids - expected_id_set)[:10]}, "
                f"manifest_only={sorted(expected_id_set - released_ids)[:10]}"
            )
        for manifest_row in manifest_rows:
            sample_id = str(manifest_row["sample_id"])
            row = released_by_id[sample_id]
            canonical_captions = tuple(str(value) for value in manifest_row["captions"])
            if row.original_captions != canonical_captions:
                raise ValueError(
                    f"{query_type}/{sample_id}: original_captions differ from "
                    "the canonical Clotho manifest"
                )
            if row.dataset_slug != "clotho_evaluation":
                raise ValueError(f"{query_type}/{sample_id}: dataset_slug mismatch")
            if dict(row.metadata) != {"split": "evaluation", "num_captions": 5}:
                raise ValueError(f"{query_type}/{sample_id}: metadata mismatch")
            output.append(
                {
                    "query_index": len(output),
                    "query_id": f"{sample_id}#uiq_{query_type}",
                    "target_id": sample_id,
                    "clip_id": sample_id,
                    "released_query_type": query_type,
                    "paper_query_type": str(source["paper_query_type"]),
                    "text": row.query,
                    "release_row_index": row.row_index,
                    "source_model": row.source_model,
                    "regen_model": row.regen_model,
                }
            )
        source_identities.append(
            {
                **identity,
                "released_query_type": query_type,
                "paper_query_type": str(source["paper_query_type"]),
                "rows": len(released),
            }
        )

    if len(output) != int(config["expected_total_queries"]):
        raise ValueError("combined UIQ query count mismatch")
    query_ids = [row["query_id"] for row in output]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("combined UIQ query IDs are not unique")
    return output, source_identities


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    model_root = args.model_root.resolve()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    attempt_dir = args.attempt_dir.resolve() if args.attempt_dir else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "generation_metrics.json"
    started_at = utc_now()
    metrics_preexisting = metrics_path.exists()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": started_at,
        "finished_at": None,
        "metrics_path": str(metrics_path),
        "attempt_dir": str(attempt_dir),
        "error": None,
    }
    identity_validated = False

    try:
        if metrics_preexisting:
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict) or existing.get("schema_version") != 1:
                raise RuntimeError(f"invalid existing generation metrics: {metrics_path}")
            report = existing
            report.update(
                {
                    "status": "running",
                    "finished_at": None,
                    "last_attempt_started_at": started_at,
                    "metrics_path": str(metrics_path),
                    "attempt_dir": str(attempt_dir),
                    "error": None,
                }
            )
            report.pop("traceback", None)

        config = load_uiq_config(config_path)
        base_protocol_identity = verify_repository_resource(
            config["base_embedding_config"]
        )
        base_config_path = args.base_embedding_config.resolve()
        if not base_config_path.is_file():
            raise FileNotFoundError(base_config_path)
        base_identity = file_identity(base_config_path)
        base_config = load_base_embedding_config(base_config_path)
        model_lock_binding = verify_official_model_lock_binding(base_config)
        resolved_protocol = base_config.get("protocol_config")
        if not isinstance(resolved_protocol, dict) or any(
            resolved_protocol.get(resolved_field) != expected_value
            for resolved_field, expected_value in (
                ("repository_path", config["base_embedding_config"]["path"]),
                ("size_bytes", config["base_embedding_config"]["size_bytes"]),
                ("sha256", config["base_embedding_config"]["sha256"]),
            )
        ):
            raise ValueError(
                "resolved base config does not derive from the fixed UIQ protocol"
            )
        if base_config["model"] != config["model"]:
            raise ValueError("UIQ/base config model mismatch")
        if int(tagged_value(base_config, "model_config", "projection_dim")) != int(
            config["expected_embedding_dim"]
        ):
            raise ValueError("UIQ/base config embedding dimension mismatch")
        if tagged_value(base_config, "model_config", "query_prefix") != config[
            "text_protocol"
        ]["prefix"]:
            raise ValueError("UIQ/base config query prefix mismatch")

        git_commit = git_output("rev-parse", "HEAD")
        git_status = git_output("status", "--short")
        if git_status:
            raise RuntimeError(
                f"formal UIQ generation requires a clean worktree: {git_status!r}"
            )
        if base_config.get("resolution_git_commit") != git_commit:
            raise RuntimeError(
                "resolved base embedding config was not generated at the current Git commit"
            )
        if not output_dir.name.startswith(config["experiment_prefix"] + "_"):
            raise ValueError("output directory name must start with experiment_prefix")
        strict_offline = {
            name: os.environ.get(name)
            for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
        }
        if any(value != "1" for value in strict_offline.values()):
            raise RuntimeError("all strict-offline variables must equal 1")

        manifest_rows = load_manifest(
            manifest_path,
            int(config["expected_examples"]),
            int(base_config["caption_count_per_audio"]),
        )
        queries, source_identities = build_uiq_query_metadata(manifest_rows, config)
        identity = {
            "schema_version": 1,
            "experiment_id": output_dir.name,
            "git_commit": git_commit,
            "config": file_identity(config_path),
            "base_embedding_protocol": base_protocol_identity,
            "base_embedding_config": base_identity,
            "official_model_lock": model_lock_binding,
            "manifest": file_identity(manifest_path),
            "query_sources": source_identities,
            "model": config["model"],
            "checkpoint_revision": base_config["checkpoint"]["revision"],
            "seed": config["seed"],
            "text_batch_size": config["text_batch_size"],
            "query_count": len(queries),
        }
        ensure_run_identity(output_dir, identity)
        identity_validated = True
        resolved_config = json.loads(json.dumps(config))
        resolved_config["resolved_paths"] = {
            "source_config": str(config_path),
            "base_embedding_protocol": str(base_protocol_identity["path"]),
            "base_embedding_config": str(base_config_path),
            "model_root": str(model_root),
            "manifest": str(manifest_path),
            "output_dir": str(output_dir),
        }
        write_text_once_or_verify(
            output_dir / "config.yaml", immutable_json_text(resolved_config)
        )
        write_text_once_or_verify(
            output_dir / "query_metadata.jsonl", immutable_jsonl_text(queries)
        )
        report.update(
            {
                "experiment_id": output_dir.name,
                "git_commit": git_commit,
                "git_status_short": git_status,
                "model": config["model"],
                "official_model_lock": model_lock_binding,
                "base_embedding_config": base_identity,
                "dataset": config["dataset"],
                "seed": config["seed"],
                "strict_offline": strict_offline,
                "query_count": len(queries),
                "query_type_counts": {
                    query_type: sum(
                        row["released_query_type"] == query_type for row in queries
                    )
                    for query_type in RELEASED_QUERY_TYPES
                },
                "query_sources": source_identities,
                "completed_text_chunks": 0,
            }
        )
        atomic_write_json(metrics_path, report)

        random.seed(config["seed"])
        np.random.seed(config["seed"])
        embedding_dim = int(config["expected_embedding_dim"])
        text_batch_size = int(config["text_batch_size"])
        chunk_root = output_dir / "chunks"
        chunk_root.mkdir(exist_ok=True)
        all_ranges = list(ranges(len(queries), text_batch_size))
        pending = [
            (start, stop)
            for start, stop in all_ranges
            if load_verified_chunk(
                chunk_root, "uiq_text", start, stop, embedding_dim
            )
            is None
        ]
        report["completed_text_chunks"] = len(all_ranges) - len(pending)
        report["pending_text_chunks"] = len(pending)
        atomic_write_json(metrics_path, report)

        stage_started = time.monotonic()
        completed_this_attempt = 0
        print_progress(
            completed=report["completed_text_chunks"],
            total=len(all_ranges),
            completed_this_attempt=completed_this_attempt,
            stage_started=stage_started,
            overall_started_at=str(report["started_at"]),
        )

        if pending:
            torch, adapter, model, _audio_head, text_head, device = load_model_bundle(
                base_config, model_root, report
            )
            torch.manual_seed(config["seed"])
            torch.cuda.manual_seed_all(config["seed"])
            torch.cuda.reset_peak_memory_stats(device)
            for start, stop in pending:
                begun = time.monotonic()
                embeddings = project_batch(
                    torch,
                    adapter,
                    model,
                    text_head,
                    device,
                    texts=[queries[index]["text"] for index in range(start, stop)],
                )
                save_chunk(
                    chunk_root,
                    "uiq_text",
                    start,
                    stop,
                    embeddings,
                    time.monotonic() - begun,
                )
                report["completed_text_chunks"] += 1
                report["pending_text_chunks"] -= 1
                atomic_write_json(metrics_path, report)
                completed_this_attempt += 1
                if (
                    completed_this_attempt == 1
                    or completed_this_attempt % 25 == 0
                    or report["pending_text_chunks"] == 0
                ):
                    print_progress(
                        completed=report["completed_text_chunks"],
                        total=len(all_ranges),
                        completed_this_attempt=completed_this_attempt,
                        stage_started=stage_started,
                        overall_started_at=str(report["started_at"]),
                    )
            report["gpu_peak_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(device)
            )
            report["gpu_peak_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(device)
            )

        query_identity = consolidate_chunks(
            chunk_root,
            "uiq_text",
            len(queries),
            text_batch_size,
            embedding_dim,
            output_dir / "query_embeddings.npy",
        )
        report.update(
            {
                "status": "complete",
                "finished_at": utc_now(),
                "query_embedding_shape": [len(queries), embedding_dim],
                "artifacts": {
                    "query_embeddings": query_identity,
                    "query_metadata": file_identity(
                        output_dir / "query_metadata.jsonl"
                    ),
                },
                "error": None,
            }
        )
        atomic_write_json(metrics_path, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except BaseException as exc:
        report.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        if identity_validated or not metrics_preexisting:
            atomic_write_json(metrics_path, report)
        atomic_write_json(
            attempt_dir / "failure.json",
            {
                "status": "failed",
                "finished_at": report["finished_at"],
                "error": report["error"],
                "traceback": report["traceback"],
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
