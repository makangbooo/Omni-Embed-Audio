#!/usr/bin/env python3
"""Prepare or finalize a strict positive-UIQ retrieval evaluation suite."""

from __future__ import annotations

import argparse
import csv
import io
import json
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

from scripts.evaluate_embedding_artifacts import (
    file_identity,
    load_jsonl_objects,
    write_json,
)
from scripts.build_official_oea_eval_config import (
    verify_official_model_lock_binding,
)
from scripts.prepare_embedding_evaluation_suite import (
    assert_identity_matches,
    atomic_write_text,
    audit_embedding_directory,
    immutable_json_text,
    validate_embeddings,
    verify_file_identity,
    write_failure_report,
    write_text_once_or_verify,
)


RELEASED_QUERY_TYPES = ("question", "imperative", "paraphrase", "tagging")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "finalize"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", type=Path, required=True)
        subparser.add_argument(
            "--caption-embedding-dir", type=Path, required=True
        )
        subparser.add_argument("--uiq-embedding-dir", type=Path, required=True)
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


def load_suite_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported positive UIQ suite schema_version")
    required = {
        "caption_dataset",
        "checkpoint",
        "dataset",
        "embedding_protocol",
        "embedding_seed",
        "expected_candidate_count",
        "expected_caption_generation_protocol_sha256",
        "expected_caption_query_count",
        "expected_captions_per_clip",
        "expected_embedding_dim",
        "expected_queries_per_type",
        "expected_uiq_generation_config_sha256",
        "expected_uiq_query_count",
        "minimum_caption_generator_commit",
        "minimum_uiq_generator_commit",
        "model",
        "official_model_lock_path",
        "official_variant_id",
        "protocols",
        "suite_prefix",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"positive UIQ suite config missing fields: {missing}")
    for field in (
        "expected_candidate_count",
        "expected_caption_query_count",
        "expected_captions_per_clip",
        "expected_embedding_dim",
        "expected_queries_per_type",
        "expected_uiq_query_count",
    ):
        value = config[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if not isinstance(config["embedding_seed"], int) or isinstance(
        config["embedding_seed"], bool
    ):
        raise ValueError("embedding_seed must be an integer")
    for field in (
        "caption_dataset",
        "dataset",
        "model",
        "official_model_lock_path",
        "official_variant_id",
        "suite_prefix",
    ):
        if not isinstance(config[field], str) or not config[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    for field in (
        "expected_caption_generation_protocol_sha256",
        "expected_uiq_generation_config_sha256",
    ):
        value = config[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{field} must be SHA256")
    protocols = config["protocols"]
    if not isinstance(protocols, list) or len(protocols) != 4:
        raise ValueError("protocols must contain exactly four entries")
    observed_types: list[str] = []
    protocol_ids: list[str] = []
    for index, protocol in enumerate(protocols):
        if not isinstance(protocol, dict):
            raise ValueError(f"protocol {index} must be an object")
        protocol_required = {
            "expected_evaluated_queries",
            "note",
            "paper_query_type",
            "paper_table",
            "protocol_id",
            "protocol_label",
            "protocol_source",
            "released_query_type",
        }
        protocol_missing = sorted(protocol_required - set(protocol))
        if protocol_missing:
            raise ValueError(
                f"protocol {index} missing fields: {protocol_missing}"
            )
        if protocol["protocol_source"] != "CODE":
            raise ValueError(f"protocol {index}: protocol_source must be CODE")
        if protocol["expected_evaluated_queries"] != config[
            "expected_queries_per_type"
        ]:
            raise ValueError(f"protocol {index}: evaluated query count mismatch")
        observed_types.append(str(protocol["released_query_type"]))
        protocol_ids.append(str(protocol["protocol_id"]))
    if tuple(observed_types) != RELEASED_QUERY_TYPES:
        raise ValueError("protocols must preserve released positive UIQ type order")
    if len(set(protocol_ids)) != len(protocol_ids):
        raise ValueError("protocol_id values must be unique")
    expected_total = config["expected_queries_per_type"] * len(protocols)
    if config["expected_uiq_query_count"] != expected_total:
        raise ValueError("expected_uiq_query_count does not match protocols")
    return config


def caption_audit_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(config)
    result.update(
        {
            "dataset": config["caption_dataset"],
            "expected_query_count": config["expected_caption_query_count"],
            "expected_generation_protocol_sha256": config[
                "expected_caption_generation_protocol_sha256"
            ],
            "minimum_generator_commit": config[
                "minimum_caption_generator_commit"
            ],
        }
    )
    return result


def metadata_string(
    row: Mapping[str, Any], field: str, row_index: int
) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"UIQ metadata row {row_index}: invalid {field}")
    return value.strip()


def validate_uiq_metadata(
    candidate_rows: list[Mapping[str, Any]],
    query_rows: list[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[dict[str, list[int]], dict[str, list[dict[str, Any]]]]:
    candidate_ids = [
        metadata_string(row, "candidate_id", index)
        for index, row in enumerate(candidate_rows)
    ]
    if len(candidate_ids) != int(config["expected_candidate_count"]):
        raise ValueError("positive UIQ candidate count mismatch")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("positive UIQ candidate IDs are not unique")
    if len(query_rows) != int(config["expected_uiq_query_count"]):
        raise ValueError("positive UIQ query count mismatch")

    indices_by_type: dict[str, list[int]] = {}
    selection_rows_by_type: dict[str, list[dict[str, Any]]] = {}
    for protocol_index, protocol in enumerate(config["protocols"]):
        query_type = str(protocol["released_query_type"])
        start = protocol_index * len(candidate_ids)
        indices = list(range(start, start + len(candidate_ids)))
        selections: list[dict[str, Any]] = []
        for candidate_index, (query_index, candidate_id) in enumerate(
            zip(indices, candidate_ids)
        ):
            row = query_rows[query_index]
            expected_query_id = f"{candidate_id}#uiq_{query_type}"
            expected_values = {
                "query_index": query_index,
                "query_id": expected_query_id,
                "target_id": candidate_id,
                "clip_id": candidate_id,
                "released_query_type": query_type,
                "paper_query_type": protocol["paper_query_type"],
            }
            for field, expected in expected_values.items():
                if row.get(field) != expected:
                    raise ValueError(
                        f"UIQ metadata row {query_index}: {field} mismatch"
                    )
            for field in ("text", "source_model", "regen_model"):
                metadata_string(row, field, query_index)
            selections.append(
                {
                    "selection_index": candidate_index,
                    "query_index": query_index,
                    "query_id": expected_query_id,
                    "target_id": candidate_id,
                    "released_query_type": query_type,
                    "paper_query_type": protocol["paper_query_type"],
                }
            )
        indices_by_type[query_type] = indices
        selection_rows_by_type[query_type] = selections
    return indices_by_type, selection_rows_by_type


def audit_uiq_embedding_directory(
    config: Mapping[str, Any],
    uiq_embedding_dir: Path,
    candidate_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    required_paths = {
        "generation_metrics": uiq_embedding_dir / "generation_metrics.json",
        "run_identity": uiq_embedding_dir / "run_identity.json",
        "resolved_base_embedding_config": (
            uiq_embedding_dir / "resolved_base_embedding_config.json"
        ),
        "resolved_generation_config": uiq_embedding_dir / "config.yaml",
        "query_embeddings": uiq_embedding_dir / "query_embeddings.npy",
        "query_metadata": uiq_embedding_dir / "query_metadata.jsonl",
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
    resolved_base_config = json.loads(
        required_paths["resolved_base_embedding_config"].read_text(
            encoding="utf-8"
        )
    )
    resolved_config = json.loads(
        required_paths["resolved_generation_config"].read_text(encoding="utf-8")
    )
    expected_values = {
        "status": "complete",
        "model": config["model"],
        "dataset": config["dataset"],
        "seed": config["embedding_seed"],
        "query_count": config["expected_uiq_query_count"],
    }
    for field, expected in expected_values.items():
        if metrics.get(field) != expected:
            raise RuntimeError(f"UIQ generation metrics {field} mismatch")
    if identity.get("query_count") != config["expected_uiq_query_count"]:
        raise RuntimeError("UIQ run identity query_count mismatch")
    if identity.get("checkpoint_revision") != config["checkpoint"]["revision"]:
        raise RuntimeError("UIQ checkpoint revision mismatch")
    verify_file_identity(
        required_paths["resolved_base_embedding_config"],
        identity.get("base_embedding_config", {}),
        "resolved UIQ base embedding config",
    )
    model_lock_binding = verify_official_model_lock_binding(resolved_base_config)
    if model_lock_binding.get("variant_id") != config["official_variant_id"]:
        raise RuntimeError("UIQ official model-lock variant mismatch")
    if model_lock_binding.get("model_lock", {}).get(
        "repository_path"
    ) != config["official_model_lock_path"]:
        raise RuntimeError("UIQ official model-lock path mismatch")
    observed_protocol_sha256 = str(
        model_lock_binding.get("protocol_config", {}).get("sha256", "")
    ).lower()
    if observed_protocol_sha256 != str(
        config["expected_caption_generation_protocol_sha256"]
    ).lower():
        raise RuntimeError("UIQ base generation protocol SHA256 mismatch")
    if metrics.get("official_model_lock") != model_lock_binding:
        raise RuntimeError("UIQ metrics model-lock binding mismatch")
    if identity.get("official_model_lock") != model_lock_binding:
        raise RuntimeError("UIQ run identity model-lock binding mismatch")
    if metrics.get("base_embedding_config") != identity.get(
        "base_embedding_config"
    ):
        raise RuntimeError("UIQ resolved base-config identities differ")
    if resolved_base_config.get("resolution_git_commit") != identity.get(
        "git_commit"
    ):
        raise RuntimeError("UIQ resolved base-config/Git commit mismatch")
    if str(identity.get("config", {}).get("sha256", "")).lower() != str(
        config["expected_uiq_generation_config_sha256"]
    ).lower():
        raise RuntimeError("UIQ generation config SHA256 mismatch")
    resolved_checks = {
        "model": config["model"],
        "dataset": config["dataset"],
        "seed": config["embedding_seed"],
        "expected_total_queries": config["expected_uiq_query_count"],
        "expected_embedding_dim": config["expected_embedding_dim"],
    }
    for field, expected in resolved_checks.items():
        if resolved_config.get(field) != expected:
            raise RuntimeError(f"resolved UIQ generation {field} mismatch")
    if metrics.get("query_sources") != identity.get("query_sources"):
        raise RuntimeError("UIQ source identities differ between audit artifacts")
    expected_type_counts = {
        protocol["released_query_type"]: config["expected_queries_per_type"]
        for protocol in config["protocols"]
    }
    if metrics.get("query_type_counts") != expected_type_counts:
        raise RuntimeError("UIQ query type counts mismatch")

    recorded_artifacts = metrics.get("artifacts")
    if not isinstance(recorded_artifacts, dict):
        raise RuntimeError("UIQ generation metrics lacks artifact identities")
    verified_artifacts = {}
    for name in ("query_embeddings", "query_metadata"):
        specification = recorded_artifacts.get(name)
        if not isinstance(specification, dict):
            raise RuntimeError(f"UIQ generation metrics lacks {name} identity")
        verified_artifacts[name] = verify_file_identity(
            required_paths[name], specification, f"UIQ {name}"
        )
    validate_embeddings(
        required_paths["query_embeddings"],
        expected_rows=int(config["expected_uiq_query_count"]),
        expected_dim=int(config["expected_embedding_dim"]),
        label="UIQ query_embeddings",
    )
    query_rows = load_jsonl_objects(required_paths["query_metadata"])
    indices, selection_rows = validate_uiq_metadata(
        candidate_rows, query_rows, config
    )
    return {
        "paths": {name: str(path) for name, path in required_paths.items()},
        "source_files": {
            name: file_identity(path) for name, path in required_paths.items()
        },
        "verified_generation_artifacts": verified_artifacts,
        "official_model_lock": model_lock_binding,
        "generation_metrics": metrics,
        "run_identity": identity,
        "indices_by_type": indices,
        "selection_rows_by_type": selection_rows,
    }


def combined_source_files(
    caption_audit: Mapping[str, Any], uiq_audit: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        **{
            f"caption_{name}": identity
            for name, identity in caption_audit["source_files"].items()
        },
        **{
            f"uiq_{name}": identity
            for name, identity in uiq_audit["source_files"].items()
        },
    }


def build_evaluation_config(
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    caption_audit: Mapping[str, Any],
    uiq_audit: Mapping[str, Any],
    *,
    suite_id: str,
    indices_path: Path,
) -> dict[str, Any]:
    checkpoint = config["checkpoint"]
    return {
        "schema_version": 1,
        "experiment_id": f"{suite_id}__{protocol['protocol_id']}",
        "model": config["model"],
        "checkpoint": (
            f"{checkpoint['repo_id']}@{checkpoint['revision']}:"
            f"{checkpoint['local_subpath']}#sha256={checkpoint['sha256']}"
        ),
        "dataset": config["dataset"],
        "task": "uiq",
        "paper_table": protocol["paper_table"],
        "protocol_label": protocol["protocol_label"],
        "protocol_source": protocol["protocol_source"],
        "protocol_note": protocol["note"],
        "released_query_type": protocol["released_query_type"],
        "paper_query_type": protocol["paper_query_type"],
        "seed": config["embedding_seed"],
        "embedding_seed": config["embedding_seed"],
        "query_embeddings": uiq_audit["paths"]["query_embeddings"],
        "candidate_embeddings": caption_audit["paths"]["candidate_embeddings"],
        "query_metadata": uiq_audit["paths"]["query_metadata"],
        "candidate_metadata": caption_audit["paths"]["candidate_metadata"],
        "query_selection": "indices",
        "query_indices": str(indices_path.resolve()),
        "query_selection_algorithm": (
            "fixed contiguous type-major rows after exact target/candidate ID "
            "and ordering validation"
        ),
        "normalize_embeddings": True,
        "require_clean_git": True,
    }


def assert_ancestry(ancestor: str, descendant: str) -> None:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=REPOSITORY_ROOT,
        check=True,
    )


def prepare_suite(
    config_path: Path,
    caption_embedding_dir: Path,
    uiq_embedding_dir: Path,
    suite_dir: Path,
) -> int:
    config_path = config_path.resolve()
    caption_embedding_dir = caption_embedding_dir.resolve()
    uiq_embedding_dir = uiq_embedding_dir.resolve()
    suite_dir = suite_dir.resolve()
    suite_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = suite_dir / "suite_metrics.json"
    started_at = utc_now()
    try:
        config = load_suite_config(config_path)
        if not suite_dir.name.startswith(config["suite_prefix"] + "_"):
            raise ValueError("suite directory name must start with suite_prefix")
        git_commit = git_output("rev-parse", "HEAD")
        git_status = git_output("status", "--short")
        if git_status:
            raise RuntimeError(f"suite preparation requires clean Git: {git_status!r}")
        caption_audit = audit_embedding_directory(
            caption_audit_config(config), caption_embedding_dir
        )
        candidate_rows = load_jsonl_objects(
            Path(caption_audit["paths"]["candidate_metadata"])
        )
        uiq_audit = audit_uiq_embedding_directory(
            config, uiq_embedding_dir, candidate_rows
        )
        caption_commit = caption_audit["run_identity"].get("git_commit")
        uiq_commit = uiq_audit["run_identity"].get("git_commit")
        if not isinstance(caption_commit, str) or not caption_commit:
            raise RuntimeError("caption embedding identity lacks git_commit")
        if not isinstance(uiq_commit, str) or not uiq_commit:
            raise RuntimeError("UIQ embedding identity lacks git_commit")
        for ancestor, generation_commit in (
            (config["minimum_caption_generator_commit"], caption_commit),
            (config["minimum_uiq_generator_commit"], uiq_commit),
        ):
            assert_ancestry(ancestor, generation_commit)
            assert_ancestry(ancestor, git_commit)
            assert_ancestry(generation_commit, git_commit)

        suite_id = suite_dir.name
        plan_protocols: list[dict[str, Any]] = []
        selection_identities: dict[str, Any] = {}
        for protocol in config["protocols"]:
            query_type = protocol["released_query_type"]
            indices_path = suite_dir / "selections" / f"{query_type}_indices.json"
            rows_path = suite_dir / "selections" / f"{query_type}_selection.jsonl"
            write_text_once_or_verify(
                indices_path,
                immutable_json_text(uiq_audit["indices_by_type"][query_type]),
            )
            write_text_once_or_verify(
                rows_path,
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in uiq_audit["selection_rows_by_type"][query_type]
                ),
            )
            selection_identities[query_type] = {
                "indices": file_identity(indices_path),
                "rows": file_identity(rows_path),
            }
            evaluation_config = build_evaluation_config(
                config,
                protocol,
                caption_audit,
                uiq_audit,
                suite_id=suite_id,
                indices_path=indices_path,
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
                    "released_query_type": query_type,
                    "paper_query_type": protocol["paper_query_type"],
                    "paper_table": protocol["paper_table"],
                    "protocol_label": protocol["protocol_label"],
                    "protocol_source": protocol["protocol_source"],
                    "expected_evaluated_queries": protocol[
                        "expected_evaluated_queries"
                    ],
                    "experiment_id": evaluation_config["experiment_id"],
                    "query_indices": str(indices_path.resolve()),
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
            "caption_embedding_dir": str(caption_embedding_dir),
            "uiq_embedding_dir": str(uiq_embedding_dir),
            "caption_generation_commit": caption_commit,
            "uiq_generation_commit": uiq_commit,
            "source_files": combined_source_files(caption_audit, uiq_audit),
            "selections": selection_identities,
            "suite_plan": file_identity(plan_path),
            "protocol_ids": [row["protocol_id"] for row in plan_protocols],
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
            "caption_embedding_dir": str(caption_embedding_dir),
            "uiq_embedding_dir": str(uiq_embedding_dir),
            "candidate_count": config["expected_candidate_count"],
            "uiq_query_count": config["expected_uiq_query_count"],
            "protocol_count": len(plan_protocols),
            "plan": file_identity(plan_path),
            "error": None,
        }
        if metrics_path.is_file():
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


def validate_protocol_run(
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    caption_audit: Mapping[str, Any],
    uiq_audit: Mapping[str, Any],
    *,
    git_commit: str,
    suite_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_dir = Path(protocol["output_dir"])
    if output_dir.parent.parent.resolve() != suite_dir.resolve():
        raise RuntimeError("protocol output directory escapes suite directory")
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    checkpoint = config["checkpoint"]
    expected_checkpoint = (
        f"{checkpoint['repo_id']}@{checkpoint['revision']}:"
        f"{checkpoint['local_subpath']}#sha256={checkpoint['sha256']}"
    )
    expected_values = {
        "status": "complete",
        "experiment_id": protocol["experiment_id"],
        "model": config["model"],
        "dataset": config["dataset"],
        "checkpoint": expected_checkpoint,
        "seed": config["embedding_seed"],
        "randomness_used_by_evaluator": False,
        "task": "uiq",
        "paper_table": protocol["paper_table"],
        "protocol_label": protocol["protocol_label"],
        "protocol_source": protocol["protocol_source"],
        "git_commit": git_commit,
        "git_status_short": "",
        "evaluated_query_count": protocol["expected_evaluated_queries"],
        "query_embedding_shape": [
            config["expected_uiq_query_count"],
            config["expected_embedding_dim"],
        ],
        "candidate_embedding_shape": [
            config["expected_candidate_count"],
            config["expected_embedding_dim"],
        ],
        "candidate_count": config["expected_candidate_count"],
        "query_selection": "indices",
        "normalization_applied": True,
        "tie_policy": "optimistic_strict_greater",
    }
    for field, expected in expected_values.items():
        if metrics.get(field) != expected:
            raise RuntimeError(
                f"protocol {protocol['protocol_id']}: {field} mismatch"
            )
    metric_values = metrics.get("metrics")
    if not isinstance(metric_values, dict) or set(metric_values) != {
        "R@1",
        "R@5",
        "R@10",
        "MRR",
        "DCG",
    }:
        raise RuntimeError("positive UIQ protocol metric set mismatch")
    for name, value in metric_values.items():
        upper = 100.0 if name.startswith("R@") else 1.0
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or not 0.0 <= float(value) <= upper
        ):
            raise RuntimeError(f"positive UIQ metric {name} is invalid")

    expected_inputs = {
        "config": file_identity(Path(protocol["config"])),
        "query_embeddings": uiq_audit["source_files"]["query_embeddings"],
        "candidate_embeddings": caption_audit["source_files"][
            "candidate_embeddings"
        ],
        "query_metadata": uiq_audit["source_files"]["query_metadata"],
        "candidate_metadata": caption_audit["source_files"][
            "candidate_metadata"
        ],
        "query_indices": file_identity(Path(protocol["query_indices"])),
    }
    recorded_inputs = metrics.get("inputs")
    if not isinstance(recorded_inputs, dict) or set(recorded_inputs) != set(
        expected_inputs
    ):
        raise RuntimeError("positive UIQ protocol input identity set mismatch")
    for name, expected in expected_inputs.items():
        recorded = recorded_inputs.get(name)
        if not isinstance(recorded, dict):
            raise RuntimeError(f"positive UIQ protocol lacks input {name}")
        assert_identity_matches(expected, recorded, f"protocol input {name}")

    artifact_names = {
        "candidate_embeddings.npy",
        "candidate_metadata.jsonl",
        "evaluated_query_indices.json",
        "ignored_indices.json",
        "positive_indices.json",
        "query_embeddings.npy",
        "query_metadata.jsonl",
        "rankings.npy",
        "ranks.npy",
        "similarities.npy",
    }
    recorded_artifacts = metrics.get("artifacts")
    if not isinstance(recorded_artifacts, dict) or set(recorded_artifacts) != (
        artifact_names
    ):
        raise RuntimeError("positive UIQ protocol artifact set mismatch")
    verified_artifacts = {}
    for name in sorted(artifact_names):
        actual = file_identity(output_dir / name)
        recorded = recorded_artifacts[name]
        if not isinstance(recorded, dict):
            raise RuntimeError(f"invalid positive UIQ artifact identity: {name}")
        assert_identity_matches(actual, recorded, f"protocol artifact {name}")
        verified_artifacts[name] = actual

    audit_names = (
        "command.sh",
        "config.yaml",
        "environment.txt",
        "exit_code.txt",
        "git_commit.txt",
        "git_status.txt",
        "gpu_info.txt",
        "python_command.sh",
        "python_environment.txt",
        "stderr.log",
        "stdout.log",
    )
    audit_files = {}
    for name in audit_names:
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        audit_files[name] = file_identity(path)
    if (output_dir / "git_commit.txt").read_text(encoding="utf-8").strip() != (
        git_commit
    ):
        raise RuntimeError("positive UIQ protocol git commit mismatch")
    if (output_dir / "git_status.txt").read_text(encoding="utf-8").strip():
        raise RuntimeError("positive UIQ protocol Git status is not clean")
    if (output_dir / "exit_code.txt").read_text(encoding="utf-8").strip() != "0":
        raise RuntimeError("positive UIQ protocol exit code is not zero")
    return metrics, {
        "metrics": file_identity(metrics_path),
        "artifacts": verified_artifacts,
        "audit_files": audit_files,
    }


def csv_summary(rows: Sequence[Mapping[str, Any]]) -> str:
    output = io.StringIO(newline="")
    fieldnames = (
        "paper_table",
        "protocol_id",
        "released_query_type",
        "paper_query_type",
        "protocol_source",
        "evaluated_queries",
        "metric",
        "value",
    )
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def finalize_suite(
    config_path: Path,
    caption_embedding_dir: Path,
    uiq_embedding_dir: Path,
    suite_dir: Path,
) -> int:
    config_path = config_path.resolve()
    caption_embedding_dir = caption_embedding_dir.resolve()
    uiq_embedding_dir = uiq_embedding_dir.resolve()
    suite_dir = suite_dir.resolve()
    metrics_path = suite_dir / "suite_metrics.json"
    started_at = utc_now()
    try:
        config = load_suite_config(config_path)
        git_commit = git_output("rev-parse", "HEAD")
        git_status = git_output("status", "--short")
        if git_status:
            raise RuntimeError(f"suite finalization requires clean Git: {git_status!r}")
        caption_audit = audit_embedding_directory(
            caption_audit_config(config), caption_embedding_dir
        )
        candidate_rows = load_jsonl_objects(
            Path(caption_audit["paths"]["candidate_metadata"])
        )
        uiq_audit = audit_uiq_embedding_directory(
            config, uiq_embedding_dir, candidate_rows
        )
        plan_path = suite_dir / "suite_plan.json"
        identity_path = suite_dir / "suite_identity.json"
        if not plan_path.is_file() or not identity_path.is_file():
            raise FileNotFoundError("positive UIQ suite must be prepared first")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if identity.get("git_commit") != git_commit:
            raise RuntimeError("suite preparation and finalization commits differ")
        for name, actual in {
            "suite_config": file_identity(config_path),
            "suite_plan": file_identity(plan_path),
        }.items():
            recorded = identity.get(name)
            if not isinstance(recorded, dict):
                raise RuntimeError(f"positive UIQ suite identity lacks {name}")
            assert_identity_matches(actual, recorded, f"suite {name}")
        actual_sources = combined_source_files(caption_audit, uiq_audit)
        recorded_sources = identity.get("source_files")
        if not isinstance(recorded_sources, dict) or set(recorded_sources) != set(
            actual_sources
        ):
            raise RuntimeError("positive UIQ suite source set mismatch")
        for name, actual in actual_sources.items():
            recorded = recorded_sources[name]
            if not isinstance(recorded, dict):
                raise RuntimeError(f"invalid positive UIQ source identity: {name}")
            assert_identity_matches(actual, recorded, f"suite source {name}")
        configured_ids = [row["protocol_id"] for row in config["protocols"]]
        planned_ids = [row.get("protocol_id") for row in plan.get("protocols", [])]
        if identity.get("protocol_ids") != configured_ids or planned_ids != configured_ids:
            raise RuntimeError("positive UIQ suite protocol list mismatch")
        for protocol in plan["protocols"]:
            query_type = protocol["released_query_type"]
            selection_record = identity.get("selections", {}).get(query_type)
            if not isinstance(selection_record, dict):
                raise RuntimeError(f"suite selection identity missing: {query_type}")
            actual_selection = {
                "indices": file_identity(Path(protocol["query_indices"])),
                "rows": file_identity(
                    suite_dir / "selections" / f"{query_type}_selection.jsonl"
                ),
            }
            for name, actual in actual_selection.items():
                recorded = selection_record.get(name)
                if not isinstance(recorded, dict):
                    raise RuntimeError(f"suite selection {query_type}/{name} missing")
                assert_identity_matches(
                    actual, recorded, f"suite selection {query_type}/{name}"
                )

        results = []
        table_rows = []
        for protocol in plan["protocols"]:
            run_metrics, evidence = validate_protocol_run(
                config,
                protocol,
                caption_audit,
                uiq_audit,
                git_commit=git_commit,
                suite_dir=suite_dir,
            )
            result = {
                "protocol_id": protocol["protocol_id"],
                "released_query_type": protocol["released_query_type"],
                "paper_query_type": protocol["paper_query_type"],
                "paper_table": protocol["paper_table"],
                "protocol_label": protocol["protocol_label"],
                "protocol_source": protocol["protocol_source"],
                "evaluated_queries": run_metrics["evaluated_query_count"],
                "metrics": run_metrics["metrics"],
                "run_evidence": evidence,
            }
            results.append(result)
            for metric, value in run_metrics["metrics"].items():
                table_rows.append(
                    {
                        "paper_table": protocol["paper_table"],
                        "protocol_id": protocol["protocol_id"],
                        "released_query_type": protocol["released_query_type"],
                        "paper_query_type": protocol["paper_query_type"],
                        "protocol_source": protocol["protocol_source"],
                        "evaluated_queries": run_metrics[
                            "evaluated_query_count"
                        ],
                        "metric": metric,
                        "value": value,
                    }
                )
        summary_path = suite_dir / "positive_uiq_summary.csv"
        summary_text = csv_summary(table_rows)
        if summary_path.exists():
            if summary_path.read_text(encoding="utf-8") != summary_text:
                raise RuntimeError("existing positive_uiq_summary.csv differs")
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
            "caption_embedding_dir": str(caption_embedding_dir),
            "uiq_embedding_dir": str(uiq_embedding_dir),
            "protocol_count": len(results),
            "results": results,
            "summary": file_identity(summary_path),
            "error": None,
        }
        if metrics_path.is_file():
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
            if existing.get("status") == "complete":
                comparison_fields = (
                    "suite_id",
                    "git_commit",
                    "results",
                    "summary",
                )
                if any(
                    existing.get(field) != report.get(field)
                    for field in comparison_fields
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
        return prepare_suite(
            args.config,
            args.caption_embedding_dir,
            args.uiq_embedding_dir,
            args.suite_dir,
        )
    return finalize_suite(
        args.config,
        args.caption_embedding_dir,
        args.uiq_embedding_dir,
        args.suite_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
