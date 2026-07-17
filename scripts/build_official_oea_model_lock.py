#!/usr/bin/env python3
"""Build a portable official-OEA evaluation lock from two complete audit reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.prepare_official_oea_checkpoint import (
    DEFAULT_REGISTRY,
    SHA256_PATTERN,
    atomic_write_json,
    file_identity,
    load_checkpoint_registry,
    positive_integer,
    safe_asset_path,
    sha256_file,
    variant_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--model-resource-audit", type=Path, required=True)
    parser.add_argument("--checkpoint-preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def portable_file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display = resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        display = resolved.name
    return file_identity(resolved, display)


def nonnegative_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def validate_evidence_header(
    report: Mapping[str, Any], label: str, allowed_statuses: set[str]
) -> None:
    if report.get("schema_version") != 1:
        raise ValueError(f"{label} has unsupported schema_version")
    if report.get("status") not in allowed_statuses:
        raise ValueError(f"{label} has unacceptable status: {report.get('status')}")
    commit = report.get("git_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError(f"{label} has invalid git_commit")
    if report.get("git_status_short") != "":
        raise ValueError(f"{label} was produced from a dirty Git worktree")


def validate_optional_official_scope(
    scope: Any, registry_path: Path, variant: Mapping[str, Any]
) -> None:
    if scope is None:
        return
    if not isinstance(scope, dict):
        raise ValueError("model resource audit scope is not an object")
    if scope.get("schema_version") != 1:
        raise ValueError("model resource audit scope has unsupported schema_version")
    if scope.get("type") != "official_oea_variant":
        raise ValueError("model resource audit has an unexpected scope type")
    if scope.get("variant_id") != variant["variant_id"]:
        raise ValueError("model resource audit scope variant_id differs from request")
    if scope.get("variant") != variant:
        raise ValueError("model resource audit scope variant differs from registry")
    expected_manifests: list[str] = []
    for field in ("base_manifest", "checkpoint_manifest"):
        if variant[field] not in expected_manifests:
            expected_manifests.append(variant[field])
    if scope.get("manifests") != expected_manifests:
        raise ValueError("model resource audit scope manifests differ from registry")
    expected_assets = [
        variant["base_asset"]["name"],
        variant["checkpoint_asset"]["name"],
    ]
    if scope.get("asset_names") != expected_assets:
        raise ValueError("model resource audit scope assets differ from registry")
    registry_identity = scope.get("checkpoint_registry")
    actual_registry_identity = portable_file_identity(registry_path)
    if not isinstance(registry_identity, dict) or any(
        registry_identity.get(field) != actual_registry_identity[field]
        for field in ("size_bytes", "sha256")
    ):
        raise ValueError("model resource audit scope used a different registry")


def matching_asset_report(
    resource_audit: Mapping[str, Any],
    expected: Mapping[str, Any],
    model_root: Path,
    label: str,
) -> Mapping[str, Any]:
    reports = resource_audit.get("assets")
    if not isinstance(reports, list):
        raise ValueError("model resource audit has no asset reports")
    matches = [
        report
        for report in reports
        if isinstance(report, dict) and report.get("name") == expected["name"]
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{label}: expected one asset report named {expected['name']}, "
            f"found {len(matches)}"
        )
    report = matches[0]
    for field in ("repo_id", "revision"):
        if report.get(field) != expected[field]:
            raise ValueError(f"{label}: asset {field} differs from registry")
    if report.get("status") != "complete":
        raise ValueError(f"{label}: selected asset is not complete")
    local_subdir = safe_asset_path(
        expected["local_subdir"], f"{label}.local_subdir"
    )
    expected_destination = (
        model_root / Path(*local_subdir.parts)
    ).resolve()
    expected_marker = (
        model_root / ".oea_asset_markers" / f"{expected['name']}.json"
    ).resolve()
    for field, expected_path in (
        ("destination", expected_destination),
        ("marker", expected_marker),
    ):
        observed_path = report.get(field)
        if (
            not isinstance(observed_path, str)
            or Path(observed_path).resolve() != expected_path
        ):
            raise ValueError(f"{label}: asset {field} differs from model root")
    if report.get("marker_identity") != {
        "repo_id": expected["repo_id"],
        "revision": expected["revision"],
    }:
        raise ValueError(f"{label}: revision marker identity differs from registry")
    for field in (
        "errors",
        "extra_files",
        "incomplete_files",
        "missing_files",
        "unsafe_symlinks",
    ):
        if report.get(field) not in (None, []):
            raise ValueError(f"{label}: selected asset has non-empty {field}")
    return report


def locked_file_inventory(
    asset_report: Mapping[str, Any], label: str
) -> dict[str, dict[str, Any]]:
    rows = asset_report.get("local_files")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label}: local file inventory is empty")
    expected_count = asset_report.get("expected_file_count")
    verified_count = asset_report.get("verified_local_file_count")
    if expected_count != len(rows) or verified_count != len(rows):
        raise ValueError(f"{label}: verified file counts are inconsistent")
    output: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        row_label = f"{label}.local_files[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{row_label} is not an object")
        relative = safe_asset_path(row.get("path"), f"{row_label}.path").as_posix()
        if relative in output:
            raise ValueError(f"{label}: duplicate local file path: {relative}")
        size = nonnegative_integer(
            row.get("size_bytes"), f"{row_label}.size_bytes"
        )
        if row.get("remote_size_bytes") != size:
            raise ValueError(f"{row_label} local and remote byte sizes differ")
        digest = row.get("sha256")
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"{row_label}.sha256 is invalid")
        if row.get("size_matches_remote") is not True:
            raise ValueError(f"{row_label} did not match remote size")
        remote_lfs = row.get("remote_lfs_sha256")
        if remote_lfs is not None:
            if (
                not isinstance(remote_lfs, str)
                or SHA256_PATTERN.fullmatch(remote_lfs) is None
            ):
                raise ValueError(f"{row_label}.remote_lfs_sha256 is invalid")
            if row.get("matches_remote_lfs_sha256") is not True:
                raise ValueError(f"{row_label} did not match remote LFS SHA256")
            if digest != remote_lfs:
                raise ValueError(f"{row_label} local and remote LFS SHA256 differ")
        else:
            remote_blob = row.get("remote_git_blob_id")
            if (
                not isinstance(remote_blob, str)
                or len(remote_blob) != 40
                or any(character not in "0123456789abcdef" for character in remote_blob)
            ):
                raise ValueError(f"{row_label}.remote_git_blob_id is invalid")
            if row.get("matches_remote_git_blob_id") is not True:
                raise ValueError(f"{row_label} did not match remote Git blob ID")
        output[relative] = {"size_bytes": size, "sha256": digest}
    total_bytes = sum(row["size_bytes"] for row in output.values())
    if asset_report.get("expected_bytes") != total_bytes:
        raise ValueError(f"{label}: expected byte count is inconsistent")
    if asset_report.get("verified_local_bytes") != total_bytes:
        raise ValueError(f"{label}: verified byte count is inconsistent")
    return dict(sorted(output.items()))


def validate_primary_source(
    checkpoint_files: Mapping[str, Mapping[str, Any]], variant: Mapping[str, Any]
) -> None:
    asset = variant["checkpoint_asset"]
    source_file = asset["source_file"]
    if source_file not in checkpoint_files:
        raise ValueError("fixed source checkpoint is absent from resource audit")
    observed = checkpoint_files[source_file]
    if observed["size_bytes"] != asset["source_size_bytes"]:
        raise ValueError("source checkpoint size differs from fixed registry")
    if observed["sha256"] != asset["source_sha256"]:
        raise ValueError("source checkpoint SHA256 differs from fixed registry")


def validate_preparation(
    report: Mapping[str, Any],
    registry_path: Path,
    variant: Mapping[str, Any],
    model_root: Path,
) -> tuple[str, dict[str, Any], dict[str, int]]:
    validate_evidence_header(report, "checkpoint preparation", {"complete"})
    if report.get("operation") != "inspect_and_extract":
        raise ValueError("checkpoint preparation did not extract a derived artifact")
    if report.get("variant_id") != variant["variant_id"]:
        raise ValueError("checkpoint preparation variant_id differs from request")
    if report.get("variant") != variant:
        raise ValueError(
            "checkpoint preparation resolved variant differs from registry"
        )

    registry_identity = report.get("registry")
    actual_registry_identity = portable_file_identity(registry_path)
    if not isinstance(registry_identity, dict) or any(
        registry_identity.get(field) != actual_registry_identity[field]
        for field in ("size_bytes", "sha256")
    ):
        raise ValueError("checkpoint preparation used a different registry")

    expected_source, expected_destination = variant_paths(variant, model_root)
    source_text = report.get("source_checkpoint")
    destination_text = report.get("derived_checkpoint")
    if (
        not isinstance(source_text, str)
        or Path(source_text).resolve() != expected_source
    ):
        raise ValueError("checkpoint preparation source path differs from registry")
    if (
        not isinstance(destination_text, str)
        or Path(destination_text).resolve() != expected_destination
    ):
        raise ValueError(
            "checkpoint preparation destination path differs from registry"
        )

    identity = report.get("derived_checkpoint_identity")
    if not isinstance(identity, dict):
        raise ValueError("checkpoint preparation has no derived identity")
    derived_size = positive_integer(
        identity.get("size_bytes"), "derived checkpoint size"
    )
    derived_sha256 = identity.get("sha256")
    if (
        not isinstance(derived_sha256, str)
        or SHA256_PATTERN.fullmatch(derived_sha256) is None
    ):
        raise ValueError("derived checkpoint SHA256 is invalid")
    if not expected_destination.is_file():
        raise FileNotFoundError(expected_destination)
    if expected_destination.stat().st_size != derived_size:
        raise ValueError("derived checkpoint file size drifted after preparation")
    if sha256_file(expected_destination) != derived_sha256:
        raise ValueError("derived checkpoint file SHA256 drifted after preparation")

    expectations = report.get("measured_structure_expectations")
    if not isinstance(expectations, dict):
        raise ValueError("checkpoint preparation has no structure expectations")
    structure = {
        "lora_tensor_count": positive_integer(
            expectations.get("expected_lora_tensors"), "LoRA tensor count"
        ),
        "lora_tensor_bytes": positive_integer(
            expectations.get("expected_lora_bytes"), "LoRA tensor bytes"
        ),
    }
    local_subpath = (
        PurePosixPath(variant["checkpoint_asset"]["local_subdir"])
        / variant["checkpoint_asset"]["derived_file"]
    ).as_posix()
    identity = {"size_bytes": derived_size, "sha256": derived_sha256}
    return local_subpath, identity, structure


def build_model_lock(
    *,
    variant_id: str,
    registry_path: Path,
    resource_audit_path: Path,
    preparation_path: Path,
) -> dict[str, Any]:
    registry_path = registry_path.resolve()
    resource_audit_path = resource_audit_path.resolve()
    preparation_path = preparation_path.resolve()
    registry = load_checkpoint_registry(registry_path)
    if variant_id not in registry:
        raise ValueError(f"unknown variant: {variant_id}")
    variant = registry[variant_id]

    resource_audit = read_json_object(resource_audit_path, "model resource audit")
    validate_evidence_header(
        resource_audit, "model resource audit", {"complete", "incomplete"}
    )
    if resource_audit.get("fatal_error") is not None:
        raise ValueError("model resource audit has a fatal_error")
    audit_scope = resource_audit.get("scope")
    validate_optional_official_scope(audit_scope, registry_path, variant)
    if audit_scope is not None and resource_audit.get("requested_assets") != [
        variant["base_asset"]["name"],
        variant["checkpoint_asset"]["name"],
    ]:
        raise ValueError(
            "model resource audit requested assets differ from official scope"
        )
    model_root_text = resource_audit.get("model_root")
    if not isinstance(model_root_text, str) or not model_root_text:
        raise ValueError("model resource audit has no model_root")
    model_root = Path(model_root_text).resolve()

    base_report = matching_asset_report(
        resource_audit, variant["base_asset"], model_root, "base model"
    )
    checkpoint_report = matching_asset_report(
        resource_audit,
        variant["checkpoint_asset"],
        model_root,
        "source checkpoint",
    )
    base_files = locked_file_inventory(base_report, "base model")
    checkpoint_files = locked_file_inventory(
        checkpoint_report, "source checkpoint"
    )
    validate_primary_source(checkpoint_files, variant)

    preparation = read_json_object(preparation_path, "checkpoint preparation")
    local_subpath, derived_identity, structure = validate_preparation(
        preparation,
        registry_path,
        variant,
        model_root,
    )
    checkpoint_asset = variant["checkpoint_asset"]
    return {
        "schema_version": 1,
        "status": "locked",
        "variant_id": variant_id,
        "model": variant["paper_model"],
        "training_variant": variant["training_variant"],
        "source_tags": {
            "base_model": (
                "[CODE] immutable official revision; every selected file "
                "verified by resource audit"
            ),
            "source_checkpoint": (
                "[CODE] immutable official LFS identity verified by resource audit"
            ),
            "derived_checkpoint": (
                "[DERIVED] weights-only LoRA/projection artifact verified "
                "against the fixed source checkpoint"
            ),
            "structure": (
                "[CODE] measured from the fixed source checkpoint; not "
                "architecture-inferred"
            ),
        },
        "base_model": {
            "repo_id": variant["base_asset"]["repo_id"],
            "revision": variant["base_asset"]["revision"],
            "local_subdir": variant["base_asset"]["local_subdir"],
            "source": "CODE+AUDIT",
            "files": base_files,
        },
        "checkpoint": {
            "repo_id": checkpoint_asset["repo_id"],
            "revision": checkpoint_asset["revision"],
            "local_subpath": local_subpath,
            "size_bytes": derived_identity["size_bytes"],
            "sha256": derived_identity["sha256"],
            "source": "DERIVED_FROM_CODE_CHECKPOINT",
            "source_checkpoint": {
                "local_subpath": (
                    PurePosixPath(checkpoint_asset["local_subdir"])
                    / checkpoint_asset["source_file"]
                ).as_posix(),
                "size_bytes": checkpoint_asset["source_size_bytes"],
                "sha256": checkpoint_asset["source_sha256"],
            },
        },
        "measured_structure": structure,
        "evidence": {
            "checkpoint_registry": portable_file_identity(registry_path),
            "model_resource_audit": portable_file_identity(resource_audit_path),
            "checkpoint_preparation": portable_file_identity(preparation_path),
            "resource_audit_git_commit": resource_audit["git_commit"],
            "checkpoint_preparation_git_commit": preparation["git_commit"],
        },
    }


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing model lock: {output}")
    if git_output("status", "--short"):
        raise RuntimeError("model lock generation requires a clean Git worktree")
    lock = build_model_lock(
        variant_id=args.variant,
        registry_path=args.registry,
        resource_audit_path=args.model_resource_audit,
        preparation_path=args.checkpoint_preparation,
    )
    lock["builder_git_commit"] = git_output("rev-parse", "HEAD")
    atomic_write_json(output, lock)
    print(f"[INFO] Wrote locked official model resources: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
