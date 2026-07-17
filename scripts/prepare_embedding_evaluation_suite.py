#!/usr/bin/env python3
"""Prepare or finalize an auditable retrieval suite from embedding artifacts."""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.evaluate_embedding_artifacts import (  # noqa: E402
    file_identity,
    load_jsonl_objects,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "finalize"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", type=Path, required=True)
        subparser.add_argument("--embedding-dir", type=Path, required=True)
        subparser.add_argument("--suite-dir", type=Path, required=True)
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


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_text_once_or_verify(path: Path, text: str) -> None:
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"immutable suite artifact differs: {path}")
        return
    atomic_write_text(path, text)


def write_failure_report(
    metrics_path: Path, suite_dir: Path, report: Mapping[str, Any]
) -> None:
    if metrics_path.is_file():
        try:
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = None
        if isinstance(existing, dict) and existing.get("status") == "complete":
            failure_name = datetime.now(timezone.utc).strftime(
                "failure_%Y%m%d_%H%M%S_%f.json"
            )
            write_json(suite_dir / "failures" / failure_name, report)
            return
    write_json(metrics_path, report)


def immutable_json_text(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def load_suite_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported retrieval suite config schema_version")
    required = {
        "checkpoint",
        "dataset",
        "embedding_protocol",
        "embedding_seed",
        "expected_candidate_count",
        "expected_captions_per_clip",
        "expected_embedding_dim",
        "expected_generation_config_sha256",
        "expected_query_count",
        "minimum_generator_commit",
        "model",
        "protocols",
        "suite_prefix",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"retrieval suite config missing fields: {missing}")
    for field in (
        "expected_candidate_count",
        "expected_captions_per_clip",
        "expected_embedding_dim",
        "expected_query_count",
    ):
        value = config[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if not isinstance(config["embedding_seed"], int) or isinstance(
        config["embedding_seed"], bool
    ):
        raise ValueError("embedding_seed must be an integer")
    for field in ("dataset", "model", "suite_prefix"):
        if not isinstance(config[field], str) or not config[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    digest = config["expected_generation_config_sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("expected_generation_config_sha256 must be SHA256")
    protocols = config["protocols"]
    if not isinstance(protocols, list) or not protocols:
        raise ValueError("protocols must be a non-empty list")
    protocol_ids: list[str] = []
    for index, protocol in enumerate(protocols):
        if not isinstance(protocol, dict):
            raise ValueError(f"protocol {index} must be an object")
        protocol_required = {
            "expected_evaluated_queries",
            "note",
            "paper_table",
            "protocol_id",
            "protocol_label",
            "protocol_source",
            "query_selection",
            "task",
        }
        protocol_missing = sorted(protocol_required - set(protocol))
        if protocol_missing:
            raise ValueError(
                f"protocol {index} missing fields: {protocol_missing}"
            )
        if protocol["task"] not in {"t2a", "t2t"}:
            raise ValueError(f"protocol {index}: task must be t2a or t2t")
        if protocol["query_selection"] not in {
            "all",
            "public_code_seed0_one_per_clip",
        }:
            raise ValueError(f"protocol {index}: unsupported query_selection")
        if protocol["protocol_source"] not in {"CODE", "INFERRED"}:
            raise ValueError(f"protocol {index}: invalid protocol_source")
        protocol_id = protocol["protocol_id"]
        if not isinstance(protocol_id, str) or not protocol_id.strip():
            raise ValueError(f"protocol {index}: invalid protocol_id")
        protocol_ids.append(protocol_id)
    if len(set(protocol_ids)) != len(protocol_ids):
        raise ValueError("protocol_id values must be unique")
    return config


def verify_file_identity(
    path: Path, expected: Mapping[str, Any], label: str
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = file_identity(path)
    if actual["size_bytes"] != expected.get("size_bytes"):
        raise RuntimeError(f"{label} size mismatch")
    if actual["sha256"].lower() != str(expected.get("sha256", "")).lower():
        raise RuntimeError(f"{label} SHA256 mismatch")
    return actual


def metadata_string(row: Mapping[str, Any], field: str, row_index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"metadata row {row_index}: invalid {field}")
    return value.strip()


def validate_metadata_and_select_indices(
    candidate_rows: list[Mapping[str, Any]],
    query_rows: list[Mapping[str, Any]],
    *,
    expected_candidates: int,
    expected_queries: int,
    captions_per_clip: int,
    seed: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    if len(candidate_rows) != expected_candidates:
        raise ValueError(
            f"candidate metadata count mismatch: {len(candidate_rows)}"
        )
    if len(query_rows) != expected_queries:
        raise ValueError(f"query metadata count mismatch: {len(query_rows)}")

    candidate_ids: list[str] = []
    for index, row in enumerate(candidate_rows):
        if row.get("candidate_index") != index:
            raise ValueError(f"candidate row {index}: candidate_index mismatch")
        candidate_ids.append(metadata_string(row, "candidate_id", index))
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate IDs must be unique")

    positions_by_clip = {candidate_id: [] for candidate_id in candidate_ids}
    caption_indices_by_clip = {candidate_id: [] for candidate_id in candidate_ids}
    expected_query_order = [
        (candidate_id, caption_index)
        for candidate_id in candidate_ids
        for caption_index in range(1, captions_per_clip + 1)
    ]
    for index, row in enumerate(query_rows):
        if row.get("query_index") != index:
            raise ValueError(f"query row {index}: query_index mismatch")
        query_id = metadata_string(row, "query_id", index)
        clip_id = metadata_string(row, "clip_id", index)
        target_id = metadata_string(row, "target_id", index)
        caption_index = row.get("caption_index")
        if clip_id not in positions_by_clip:
            raise KeyError(f"query row {index}: clip_id absent from candidates")
        if target_id != clip_id:
            raise ValueError(f"query row {index}: target_id != clip_id")
        if not isinstance(caption_index, int) or isinstance(caption_index, bool):
            raise ValueError(f"query row {index}: invalid caption_index")
        expected_query_id = f"{clip_id}#caption_{caption_index}"
        if query_id != expected_query_id:
            raise ValueError(f"query row {index}: query_id mismatch")
        if (clip_id, caption_index) != expected_query_order[index]:
            raise ValueError(
                f"query row {index}: flattened candidate/caption order mismatch"
            )
        positions_by_clip[clip_id].append(index)
        caption_indices_by_clip[clip_id].append(caption_index)

    expected_caption_indices = list(range(1, captions_per_clip + 1))
    for candidate_id in candidate_ids:
        if len(positions_by_clip[candidate_id]) != captions_per_clip:
            raise ValueError(
                f"candidate {candidate_id!r}: caption count mismatch"
            )
        if caption_indices_by_clip[candidate_id] != expected_caption_indices:
            raise ValueError(
                f"candidate {candidate_id!r}: caption index order mismatch"
            )

    rng = random.Random(seed)
    selected = [
        rng.choice(positions_by_clip[candidate_id])
        for candidate_id in candidate_ids
    ]
    selected.sort()
    selection_rows = [
        {
            "clip_index": clip_index,
            "clip_id": candidate_id,
            "query_index": selected_index,
            "query_id": query_rows[selected_index]["query_id"],
            "caption_index": query_rows[selected_index]["caption_index"],
        }
        for clip_index, (candidate_id, selected_index) in enumerate(
            zip(candidate_ids, selected)
        )
    ]
    return selected, selection_rows


def validate_embeddings(
    path: Path,
    *,
    expected_rows: int,
    expected_dim: int,
    label: str,
) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if array.shape != (expected_rows, expected_dim):
        raise ValueError(
            f"{label} shape mismatch: {array.shape} != "
            f"{(expected_rows, expected_dim)}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{label} must be numeric")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    norms = np.linalg.norm(array, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(f"{label} is not L2 normalized")
    return array


def audit_embedding_directory(
    config: Mapping[str, Any], embedding_dir: Path
) -> dict[str, Any]:
    required_paths = {
        "generation_metrics": embedding_dir / "generation_metrics.json",
        "run_identity": embedding_dir / "run_identity.json",
        "resolved_generation_config": embedding_dir / "config.yaml",
        "candidate_embeddings": embedding_dir / "candidate_embeddings.npy",
        "query_embeddings": embedding_dir / "query_embeddings.npy",
        "candidate_metadata": embedding_dir / "candidate_metadata.jsonl",
        "query_metadata": embedding_dir / "query_metadata.jsonl",
    }
    for path in required_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    metrics = json.loads(
        required_paths["generation_metrics"].read_text(encoding="utf-8")
    )
    identity = json.loads(
        required_paths["run_identity"].read_text(encoding="utf-8")
    )
    resolved_generation_config = json.loads(
        required_paths["resolved_generation_config"].read_text(encoding="utf-8")
    )
    if metrics.get("status") != "complete":
        raise RuntimeError("embedding generation is not complete")
    if metrics.get("model") != config["model"]:
        raise RuntimeError("embedding model identity mismatch")
    if metrics.get("dataset") != config["dataset"]:
        raise RuntimeError("embedding dataset identity mismatch")
    if metrics.get("seed") != config["embedding_seed"]:
        raise RuntimeError("embedding seed mismatch")
    if identity.get("candidate_count") != config["expected_candidate_count"]:
        raise RuntimeError("run identity candidate_count mismatch")
    if identity.get("query_count") != config["expected_query_count"]:
        raise RuntimeError("run identity query_count mismatch")
    if identity.get("checkpoint_revision") != config["checkpoint"]["revision"]:
        raise RuntimeError("checkpoint revision mismatch")
    if str(identity.get("config", {}).get("sha256", "")).lower() != str(
        config["expected_generation_config_sha256"]
    ).lower():
        raise RuntimeError("generation config SHA256 mismatch")
    if resolved_generation_config.get("checkpoint") != config["checkpoint"]:
        raise RuntimeError("resolved generation checkpoint specification mismatch")
    if resolved_generation_config.get("audio_prompt_protocol") != {
        "runtime": {
            "value": "audio-only chat message; passage_prefix parameter is not inserted",
            "source": "CODE",
            "note": (
                "The paper states passage:, but public _build_audio_messages() "
                "explicitly ignores it. This run is a public-code protocol, not "
                "a strict paper-protocol claim."
            ),
        }
    }:
        raise RuntimeError("resolved audio prompt protocol mismatch")

    recorded_artifacts = metrics.get("artifacts")
    if not isinstance(recorded_artifacts, dict):
        raise RuntimeError("generation metrics lacks artifact identities")
    verified_artifacts: dict[str, Any] = {}
    for name in (
        "candidate_embeddings",
        "query_embeddings",
        "candidate_metadata",
        "query_metadata",
    ):
        specification = recorded_artifacts.get(name)
        if not isinstance(specification, dict):
            raise RuntimeError(f"generation metrics lacks {name} identity")
        verified_artifacts[name] = verify_file_identity(
            required_paths[name], specification, name
        )

    candidate_array = validate_embeddings(
        required_paths["candidate_embeddings"],
        expected_rows=int(config["expected_candidate_count"]),
        expected_dim=int(config["expected_embedding_dim"]),
        label="candidate_embeddings",
    )
    query_array = validate_embeddings(
        required_paths["query_embeddings"],
        expected_rows=int(config["expected_query_count"]),
        expected_dim=int(config["expected_embedding_dim"]),
        label="query_embeddings",
    )
    del candidate_array, query_array
    candidate_rows = load_jsonl_objects(required_paths["candidate_metadata"])
    query_rows = load_jsonl_objects(required_paths["query_metadata"])
    selected_indices, selection_rows = validate_metadata_and_select_indices(
        candidate_rows,
        query_rows,
        expected_candidates=int(config["expected_candidate_count"]),
        expected_queries=int(config["expected_query_count"]),
        captions_per_clip=int(config["expected_captions_per_clip"]),
        seed=0,
    )
    return {
        "paths": {name: str(path) for name, path in required_paths.items()},
        "source_files": {
            name: file_identity(path) for name, path in required_paths.items()
        },
        "verified_generation_artifacts": verified_artifacts,
        "generation_metrics": metrics,
        "run_identity": identity,
        "selected_indices": selected_indices,
        "selection_rows": selection_rows,
    }


def build_evaluation_config(
    suite_config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    suite_id: str,
    selection_indices_path: Path,
) -> dict[str, Any]:
    paths = audit["paths"]
    checkpoint = suite_config["checkpoint"]
    evaluation_config: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": f"{suite_id}__{protocol['protocol_id']}",
        "model": suite_config["model"],
        "checkpoint": (
            f"{checkpoint['repo_id']}@{checkpoint['revision']}:"
            f"{checkpoint['local_subpath']}#sha256={checkpoint['sha256']}"
        ),
        "dataset": suite_config["dataset"],
        "task": protocol["task"],
        "paper_table": protocol["paper_table"],
        "protocol_label": protocol["protocol_label"],
        "protocol_source": protocol["protocol_source"],
        "protocol_note": protocol["note"],
        "seed": suite_config["embedding_seed"],
        "embedding_seed": suite_config["embedding_seed"],
        "query_embeddings": paths["query_embeddings"],
        "candidate_embeddings": (
            paths["query_embeddings"]
            if protocol["task"] == "t2t"
            else paths["candidate_embeddings"]
        ),
        "query_metadata": paths["query_metadata"],
        "candidate_metadata": (
            paths["query_metadata"]
            if protocol["task"] == "t2t"
            else paths["candidate_metadata"]
        ),
        "normalize_embeddings": True,
        "require_clean_git": True,
    }
    if protocol["query_selection"] == "all":
        evaluation_config["query_selection"] = "all"
    else:
        evaluation_config.update(
            {
                "query_selection": "indices",
                "query_indices": str(selection_indices_path),
                "query_selection_seed": protocol["query_selection_seed"],
                "query_selection_algorithm": (
                    "CODE eval_core: random.Random(seed).choice per clip in "
                    "candidate order, then sorted"
                ),
            }
        )
    return evaluation_config


def prepare_suite(config_path: Path, embedding_dir: Path, suite_dir: Path) -> int:
    config_path = config_path.resolve()
    embedding_dir = embedding_dir.resolve()
    suite_dir = suite_dir.resolve()
    suite_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    metrics_path = suite_dir / "suite_metrics.json"
    try:
        config = load_suite_config(config_path)
        if not suite_dir.name.startswith(config["suite_prefix"] + "_"):
            raise ValueError("suite directory name must start with suite_prefix")
        git_commit = git_output("rev-parse", "HEAD")
        git_status = git_output("status", "--short")
        if git_status:
            raise RuntimeError(f"suite preparation requires clean Git: {git_status!r}")
        minimum_commit = config["minimum_generator_commit"]
        audit = audit_embedding_directory(config, embedding_dir)
        generation_commit = audit["run_identity"].get("git_commit")
        if not isinstance(generation_commit, str) or not generation_commit:
            raise RuntimeError("embedding run identity lacks git_commit")
        for descendant in (generation_commit, git_commit):
            subprocess.run(
                [
                    "git",
                    "merge-base",
                    "--is-ancestor",
                    minimum_commit,
                    descendant,
                ],
                cwd=REPOSITORY_ROOT,
                check=True,
            )
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", generation_commit, git_commit],
            cwd=REPOSITORY_ROOT,
            check=True,
        )

        selection_indices_path = suite_dir / "selections" / "seed0_indices.json"
        selection_rows_path = suite_dir / "selections" / "seed0_selection.jsonl"
        selection_document = immutable_json_text(audit["selected_indices"])
        selection_rows_text = "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in audit["selection_rows"]
        )
        write_text_once_or_verify(selection_indices_path, selection_document)
        write_text_once_or_verify(selection_rows_path, selection_rows_text)

        suite_id = suite_dir.name
        plan_protocols: list[dict[str, Any]] = []
        for protocol in config["protocols"]:
            evaluation_config = build_evaluation_config(
                config,
                protocol,
                audit,
                suite_id=suite_id,
                selection_indices_path=selection_indices_path.resolve(),
            )
            protocol_id = protocol["protocol_id"]
            protocol_config_path = suite_dir / "configs" / f"{protocol_id}.json"
            output_dir = suite_dir / "runs" / evaluation_config["experiment_id"]
            write_text_once_or_verify(
                protocol_config_path, immutable_json_text(evaluation_config)
            )
            plan_protocols.append(
                {
                    "protocol_id": protocol_id,
                    "task": protocol["task"],
                    "paper_table": protocol["paper_table"],
                    "protocol_label": protocol["protocol_label"],
                    "protocol_source": protocol["protocol_source"],
                    "query_selection": protocol["query_selection"],
                    "query_selection_seed": protocol.get(
                        "query_selection_seed"
                    ),
                    "expected_evaluated_queries": protocol[
                        "expected_evaluated_queries"
                    ],
                    "experiment_id": evaluation_config["experiment_id"],
                    "config": str(protocol_config_path.resolve()),
                    "output_dir": str(output_dir.resolve()),
                }
            )

        plan = {
            "schema_version": 1,
            "suite_id": suite_id,
            "protocols": plan_protocols,
        }
        plan_path = suite_dir / "suite_plan.json"
        write_text_once_or_verify(plan_path, immutable_json_text(plan))
        suite_identity = {
            "schema_version": 1,
            "suite_id": suite_id,
            "git_commit": git_commit,
            "suite_config": file_identity(config_path),
            "embedding_dir": str(embedding_dir),
            "embedding_generation_commit": audit["run_identity"]["git_commit"],
            "embedding_run_identity": file_identity(
                Path(audit["paths"]["run_identity"])
            ),
            "embedding_generation_metrics": file_identity(
                Path(audit["paths"]["generation_metrics"])
            ),
            "source_files": audit["source_files"],
            "selection_indices": file_identity(selection_indices_path),
            "selection_rows": file_identity(selection_rows_path),
            "suite_plan": file_identity(plan_path),
            "protocol_ids": [item["protocol_id"] for item in plan_protocols],
        }
        write_text_once_or_verify(
            suite_dir / "suite_identity.json", immutable_json_text(suite_identity)
        )
        report = {
            "schema_version": 1,
            "status": "prepared",
            "started_at": started_at,
            "finished_at": utc_now(),
            "suite_id": suite_id,
            "git_commit": git_commit,
            "git_status_short": git_status,
            "embedding_dir": str(embedding_dir),
            "embedding_generation_status": audit["generation_metrics"]["status"],
            "candidate_count": config["expected_candidate_count"],
            "query_count": config["expected_query_count"],
            "selection_count": len(audit["selected_indices"]),
            "protocol_count": len(plan_protocols),
            "plan": file_identity(suite_dir / "suite_plan.json"),
            "error": None,
        }
        if metrics_path.exists():
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
            if existing.get("status") == "complete":
                return 0
        write_json(metrics_path, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except BaseException as exc:
        write_failure_report(
            metrics_path,
            suite_dir,
            {
                "schema_version": 1,
                "status": "failed",
                "started_at": started_at,
                "finished_at": utc_now(),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def csv_summary(rows: Sequence[Mapping[str, Any]]) -> str:
    output = io.StringIO(newline="")
    fieldnames = (
        "paper_table",
        "protocol_id",
        "task",
        "protocol_source",
        "query_selection",
        "query_selection_seed",
        "evaluated_queries",
        "metric",
        "value",
    )
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def assert_identity_matches(
    actual: Mapping[str, Any], recorded: Mapping[str, Any], label: str
) -> None:
    if actual.get("size_bytes") != recorded.get("size_bytes"):
        raise RuntimeError(f"{label} recorded size mismatch")
    if str(actual.get("sha256", "")).lower() != str(
        recorded.get("sha256", "")
    ).lower():
        raise RuntimeError(f"{label} recorded SHA256 mismatch")


def validate_protocol_run(
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    git_commit: str,
    suite_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_dir = Path(protocol["output_dir"])
    if output_dir.parent.parent.resolve() != suite_dir.resolve():
        raise RuntimeError("protocol output directory escapes suite directory")
    run_metrics_path = output_dir / "metrics.json"
    if not run_metrics_path.is_file():
        raise FileNotFoundError(run_metrics_path)
    run_metrics = json.loads(run_metrics_path.read_text(encoding="utf-8"))
    if run_metrics.get("status") != "complete":
        raise RuntimeError(f"protocol {protocol['protocol_id']} is not complete")

    checkpoint = config["checkpoint"]
    expected_checkpoint = (
        f"{checkpoint['repo_id']}@{checkpoint['revision']}:"
        f"{checkpoint['local_subpath']}#sha256={checkpoint['sha256']}"
    )
    expected_values = {
        "experiment_id": protocol["experiment_id"],
        "model": config["model"],
        "dataset": config["dataset"],
        "checkpoint": expected_checkpoint,
        "seed": config["embedding_seed"],
        "randomness_used_by_evaluator": False,
        "task": protocol["task"],
        "paper_table": protocol["paper_table"],
        "protocol_label": protocol["protocol_label"],
        "protocol_source": protocol["protocol_source"],
        "git_commit": git_commit,
        "git_status_short": "",
        "evaluated_query_count": protocol["expected_evaluated_queries"],
        "query_embedding_shape": [
            config["expected_query_count"],
            config["expected_embedding_dim"],
        ],
        "candidate_embedding_shape": [
            (
                config["expected_query_count"]
                if protocol["task"] == "t2t"
                else config["expected_candidate_count"]
            ),
            config["expected_embedding_dim"],
        ],
        "candidate_count": (
            config["expected_query_count"]
            if protocol["task"] == "t2t"
            else config["expected_candidate_count"]
        ),
        "query_selection": (
            "all" if protocol["query_selection"] == "all" else "indices"
        ),
        "normalization_applied": True,
        "tie_policy": "optimistic_strict_greater",
    }
    for field, expected in expected_values.items():
        if run_metrics.get(field) != expected:
            raise RuntimeError(
                f"protocol {protocol['protocol_id']}: {field} mismatch"
            )

    metrics = run_metrics.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != {
        "R@1",
        "R@5",
        "R@10",
        "MRR",
        "DCG",
    }:
        raise RuntimeError("protocol metric set mismatch")
    for name, value in metrics.items():
        upper_bound = 100.0 if name.startswith("R@") else 1.0
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or not 0.0 <= float(value) <= upper_bound
        ):
            raise RuntimeError(f"protocol metric {name} is invalid")

    source_files = audit["source_files"]
    config_path = Path(protocol["config"])
    expected_inputs = {
        "config": file_identity(config_path),
        "query_embeddings": source_files["query_embeddings"],
        "candidate_embeddings": source_files[
            "query_embeddings"
            if protocol["task"] == "t2t"
            else "candidate_embeddings"
        ],
        "query_metadata": source_files["query_metadata"],
        "candidate_metadata": source_files[
            "query_metadata"
            if protocol["task"] == "t2t"
            else "candidate_metadata"
        ],
    }
    if protocol["query_selection"] != "all":
        expected_inputs["query_indices"] = file_identity(
            suite_dir / "selections" / "seed0_indices.json"
        )
    recorded_inputs = run_metrics.get("inputs")
    if not isinstance(recorded_inputs, dict) or set(recorded_inputs) != set(
        expected_inputs
    ):
        raise RuntimeError("protocol input identity set mismatch")
    for name, expected in expected_inputs.items():
        recorded = recorded_inputs.get(name)
        if not isinstance(recorded, dict):
            raise RuntimeError(f"protocol lacks input identity: {name}")
        assert_identity_matches(expected, recorded, f"protocol input {name}")

    expected_artifact_names = {
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
    }
    recorded_artifacts = run_metrics.get("artifacts")
    if not isinstance(recorded_artifacts, dict) or set(recorded_artifacts) != (
        expected_artifact_names
    ):
        raise RuntimeError("protocol artifact identity set mismatch")
    verified_artifacts: dict[str, Any] = {}
    for name in sorted(expected_artifact_names):
        actual = file_identity(output_dir / name)
        recorded = recorded_artifacts[name]
        if not isinstance(recorded, dict):
            raise RuntimeError(f"invalid protocol artifact identity: {name}")
        assert_identity_matches(actual, recorded, f"protocol artifact {name}")
        verified_artifacts[name] = actual

    required_audit_files = (
        "config.yaml",
        "command.sh",
        "python_command.sh",
        "environment.txt",
        "python_environment.txt",
        "gpu_info.txt",
        "git_commit.txt",
        "git_status.txt",
        "stdout.log",
        "stderr.log",
        "exit_code.txt",
    )
    audit_files: dict[str, Any] = {}
    for name in required_audit_files:
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        audit_files[name] = file_identity(path)
    if (output_dir / "git_commit.txt").read_text(encoding="utf-8").strip() != (
        git_commit
    ):
        raise RuntimeError("protocol git_commit.txt mismatch")
    if (output_dir / "git_status.txt").read_text(encoding="utf-8").strip():
        raise RuntimeError("protocol git_status.txt is not clean")
    if (output_dir / "exit_code.txt").read_text(encoding="utf-8").strip() != "0":
        raise RuntimeError("protocol exit_code.txt is not zero")

    return run_metrics, {
        "metrics": file_identity(run_metrics_path),
        "artifacts": verified_artifacts,
        "audit_files": audit_files,
    }


def finalize_suite(config_path: Path, embedding_dir: Path, suite_dir: Path) -> int:
    config_path = config_path.resolve()
    embedding_dir = embedding_dir.resolve()
    suite_dir = suite_dir.resolve()
    metrics_path = suite_dir / "suite_metrics.json"
    started_at = utc_now()
    try:
        config = load_suite_config(config_path)
        git_commit = git_output("rev-parse", "HEAD")
        git_status = git_output("status", "--short")
        if git_status:
            raise RuntimeError(f"suite finalization requires clean Git: {git_status!r}")
        audit = audit_embedding_directory(config, embedding_dir)
        plan_path = suite_dir / "suite_plan.json"
        identity_path = suite_dir / "suite_identity.json"
        if not plan_path.is_file() or not identity_path.is_file():
            raise FileNotFoundError("suite must be prepared before finalization")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if identity.get("git_commit") != git_commit:
            raise RuntimeError("suite preparation and finalization commits differ")
        identity_checks = {
            "suite_config": file_identity(config_path),
            "suite_plan": file_identity(plan_path),
            "selection_indices": file_identity(
                suite_dir / "selections" / "seed0_indices.json"
            ),
            "selection_rows": file_identity(
                suite_dir / "selections" / "seed0_selection.jsonl"
            ),
        }
        for name, actual in identity_checks.items():
            recorded = identity.get(name)
            if not isinstance(recorded, dict):
                raise RuntimeError(f"suite identity lacks {name}")
            assert_identity_matches(actual, recorded, f"suite {name}")
        recorded_source_files = identity.get("source_files")
        if not isinstance(recorded_source_files, dict) or set(
            recorded_source_files
        ) != set(audit["source_files"]):
            raise RuntimeError("suite source file identity set mismatch")
        for name, actual in audit["source_files"].items():
            recorded = recorded_source_files[name]
            if not isinstance(recorded, dict):
                raise RuntimeError(f"invalid suite source identity: {name}")
            assert_identity_matches(actual, recorded, f"suite source {name}")
        planned_protocol_ids = [
            protocol.get("protocol_id") for protocol in plan.get("protocols", [])
        ]
        configured_protocol_ids = [
            protocol["protocol_id"] for protocol in config["protocols"]
        ]
        if (
            identity.get("protocol_ids") != configured_protocol_ids
            or planned_protocol_ids != configured_protocol_ids
        ):
            raise RuntimeError("suite protocol list mismatch")
        if identity.get("embedding_run_identity", {}).get("sha256") != file_identity(
            Path(audit["paths"]["run_identity"])
        )["sha256"]:
            raise RuntimeError("embedding run identity drifted after suite preparation")
        results: list[dict[str, Any]] = []
        table_rows: list[dict[str, Any]] = []
        for protocol in plan.get("protocols", []):
            run_metrics, run_evidence = validate_protocol_run(
                config,
                protocol,
                audit,
                git_commit=git_commit,
                suite_dir=suite_dir,
            )
            result = {
                "protocol_id": protocol["protocol_id"],
                "task": protocol["task"],
                "paper_table": protocol["paper_table"],
                "protocol_label": protocol["protocol_label"],
                "protocol_source": protocol["protocol_source"],
                "query_selection": protocol["query_selection"],
                "query_selection_seed": protocol.get("query_selection_seed"),
                "evaluated_queries": run_metrics["evaluated_query_count"],
                "metrics": run_metrics["metrics"],
                "run_evidence": run_evidence,
            }
            results.append(result)
            for metric, value in run_metrics["metrics"].items():
                table_rows.append(
                    {
                        "paper_table": protocol["paper_table"],
                        "protocol_id": protocol["protocol_id"],
                        "task": protocol["task"],
                        "protocol_source": protocol["protocol_source"],
                        "query_selection": protocol["query_selection"],
                        "query_selection_seed": protocol.get(
                            "query_selection_seed"
                        ),
                        "evaluated_queries": run_metrics[
                            "evaluated_query_count"
                        ],
                        "metric": metric,
                        "value": value,
                    }
                )
        summary_path = suite_dir / "retrieval_summary.csv"
        summary_text = csv_summary(table_rows)
        if summary_path.exists():
            if summary_path.read_text(encoding="utf-8") != summary_text:
                raise RuntimeError("existing retrieval_summary.csv differs")
        else:
            atomic_write_text(summary_path, summary_text)
        report = {
            "schema_version": 1,
            "status": "complete",
            "started_at": started_at,
            "finished_at": utc_now(),
            "suite_id": suite_dir.name,
            "git_commit": git_commit,
            "git_status_short": git_status,
            "embedding_dir": str(embedding_dir),
            "protocol_count": len(results),
            "results": results,
            "summary": file_identity(summary_path),
            "error": None,
        }
        if metrics_path.is_file():
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
            if existing.get("status") == "complete":
                if (
                    existing.get("suite_id") != report["suite_id"]
                    or existing.get("git_commit") != report["git_commit"]
                    or existing.get("results") != report["results"]
                    or existing.get("summary") != report["summary"]
                ):
                    raise RuntimeError("existing complete suite metrics differ")
                return 0
        write_json(metrics_path, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except BaseException as exc:
        write_failure_report(
            metrics_path,
            suite_dir,
            {
                "schema_version": 1,
                "status": "failed",
                "started_at": started_at,
                "finished_at": utc_now(),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        return prepare_suite(args.config, args.embedding_dir, args.suite_dir)
    return finalize_suite(args.config, args.embedding_dir, args.suite_dir)


if __name__ == "__main__":
    raise SystemExit(main())
