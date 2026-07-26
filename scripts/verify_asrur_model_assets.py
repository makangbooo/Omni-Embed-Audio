#!/usr/bin/env python3
"""Offline verification for pinned ASR-uncertainty model assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe relative path: {value!r}")
    return path


def git_head(directory: Path) -> tuple[str | None, str | None]:
    completed = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        return None, message or f"git exited {completed.returncode}"
    return completed.stdout.strip(), None


def verify_file(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": expected["path"],
        "expected_size_bytes": int(expected["size_bytes"]),
        "status": "failed",
        "errors": [],
    }
    if not path.is_file():
        record["errors"].append("missing_regular_file")
        return record

    actual_size = path.stat().st_size
    record["actual_size_bytes"] = actual_size
    if actual_size != record["expected_size_bytes"]:
        record["errors"].append("size_mismatch")

    with path.open("rb") as handle:
        prefix = handle.read(len(LFS_POINTER_PREFIX))
    record["git_lfs_pointer"] = prefix == LFS_POINTER_PREFIX
    if record["git_lfs_pointer"]:
        record["errors"].append("unresolved_git_lfs_pointer")

    expected_digest = expected.get("sha256") or expected.get("lfs_sha256")
    record["expected_sha256"] = expected_digest
    actual_digest = sha256_file(path)
    record["actual_sha256"] = actual_digest
    if actual_digest != expected_digest:
        record["errors"].append("sha256_mismatch")

    if not record["errors"]:
        record["status"] = "complete"
    return record


def audit_assets(
    manifest: dict[str, Any],
    model_root: Path,
    *,
    require_git_revision: bool,
) -> dict[str, Any]:
    root = model_root.expanduser().resolve()
    assets: list[dict[str, Any]] = []
    total_expected = 0
    total_actual = 0

    for specification in manifest["assets"]:
        relative_directory = safe_relative_path(specification["local_subdir"])
        destination = root.joinpath(*relative_directory.parts)
        record: dict[str, Any] = {
            "name": specification["name"],
            "repo_id": specification["repo_id"],
            "expected_revision": specification["revision"],
            "destination": str(destination),
            "status": "failed",
            "errors": [],
            "files": [],
        }

        if not destination.is_dir():
            record["errors"].append("missing_model_directory")
            assets.append(record)
            continue

        head, git_error = git_head(destination)
        record["actual_revision"] = head
        if require_git_revision:
            if git_error is not None:
                record["errors"].append(f"git_revision_unavailable: {git_error}")
            elif head != specification["revision"]:
                record["errors"].append("git_revision_mismatch")
        elif git_error is not None:
            record["git_revision_note"] = git_error

        expected_files = specification["expected_files"]
        required_files = specification["required_files"]
        expected_paths = [entry["path"] for entry in expected_files]
        if expected_paths != required_files:
            record["errors"].append("manifest_inventory_mismatch")

        for expected in expected_files:
            relative_file = safe_relative_path(expected["path"])
            candidate = destination.joinpath(*relative_file.parts)
            resolved_parent = candidate.parent.resolve()
            if not resolved_parent.is_relative_to(destination.resolve()):
                raise ValueError(
                    f"resolved file escapes model directory: {candidate}"
                )
            file_record = verify_file(candidate, expected)
            record["files"].append(file_record)
            total_expected += int(expected["size_bytes"])
            total_actual += int(file_record.get("actual_size_bytes", 0))
            if file_record["status"] != "complete":
                record["errors"].append(
                    f"file_failed: {expected['path']}"
                )

        expected_asset_bytes = int(specification["expected_selected_bytes"])
        computed_asset_bytes = sum(
            int(entry["size_bytes"]) for entry in expected_files
        )
        if computed_asset_bytes != expected_asset_bytes:
            record["errors"].append("manifest_asset_byte_total_mismatch")

        if not record["errors"]:
            record["status"] = "complete"
        assets.append(record)

    declared_total = int(manifest["expected_selected_bytes"])
    report_errors: list[str] = []
    if total_expected != declared_total:
        report_errors.append("manifest_total_byte_mismatch")
    if any(asset["status"] != "complete" for asset in assets):
        report_errors.append("one_or_more_assets_failed")

    return {
        "schema_version": 1,
        "resource_id": manifest.get("resource_id"),
        "status": "complete" if not report_errors else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "network_access_performed": False,
        "model_root": str(root),
        "require_git_revision": require_git_revision,
        "expected_selected_bytes": declared_total,
        "verified_expected_bytes": total_expected,
        "observed_selected_bytes": total_actual,
        "errors": report_errors,
        "assets": assets,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-missing-git-revision",
        action="store_true",
        help=(
            "Do not fail when a snapshot directory lacks .git metadata. "
            "Manual git clones must not use this option."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = audit_assets(
        manifest,
        args.model_root,
        require_git_revision=not args.allow_missing_git_revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
