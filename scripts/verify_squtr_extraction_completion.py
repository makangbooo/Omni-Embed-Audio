#!/usr/bin/env python3
"""Verify a DATA-13B completion marker without rereading all extracted bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-root", type=Path, required=True)
    parser.add_argument("--resource-manifest", type=Path, required=True)
    parser.add_argument("--structure-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def validate_completion(
    extract_root: Path,
    resource: dict[str, Any],
    structure: dict[str, Any],
) -> dict[str, Any]:
    if extract_root.is_symlink() or not extract_root.is_dir():
        raise ValueError(f"missing/unsafe extract root: {extract_root}")
    files = resource.get("files")
    if not isinstance(files, list) or len(files) != 1:
        raise ValueError("resource manifest must contain exactly one archive")
    asset = files[0]
    marker = extract_root / ".data13b_extraction_complete.json"
    if marker.is_symlink() or not marker.is_file():
        raise ValueError(f"missing/unsafe completion marker: {marker}")
    observed = read_json(marker)
    expected = {
        "archive_sha256": asset["sha256"],
        "archive_root": structure["observed_archive_root"],
        "expected_file_members": structure["observed_file_members"],
        "expected_uncompressed_member_bytes": structure[
            "observed_uncompressed_member_bytes"
        ],
        "file_members": structure["observed_file_members"],
        "uncompressed_member_bytes": structure[
            "observed_uncompressed_member_bytes"
        ],
        "full_member_crc_verified": True,
    }
    if observed != expected:
        raise ValueError(f"completion marker mismatch: {observed!r} != {expected!r}")

    dataset_root = extract_root / str(structure["observed_archive_root"])
    checked_paths: list[str] = []
    for subset in structure["subsets"]:
        subset_root = dataset_root.joinpath(
            *PurePosixPath(str(subset["relative_path"])).parts
        )
        for relative in structure["core_relative_paths"]:
            path = subset_root.joinpath(*PurePosixPath(str(relative)).parts)
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"missing/unsafe extracted core file: {path}")
            checked_paths.append(path.relative_to(extract_root).as_posix())
        for condition in structure["conditions"]:
            metadata = subset_root / str(condition["documented_query_file"])
            audio_dir = subset_root / str(condition["audio_directory"])
            if metadata.is_symlink() or not metadata.is_file():
                raise ValueError(f"missing/unsafe query metadata: {metadata}")
            if audio_dir.is_symlink() or not audio_dir.is_dir():
                raise ValueError(f"missing/unsafe audio directory: {audio_dir}")
            checked_paths.extend(
                [
                    metadata.relative_to(extract_root).as_posix(),
                    audio_dir.relative_to(extract_root).as_posix(),
                ]
            )
    return {
        "marker_path": str(marker),
        "marker_sha256": sha256_file(marker),
        "marker_identity_exact": True,
        "checked_core_paths": len(checked_paths),
        "checked_core_path_examples": checked_paths[:20],
        "reuse_policy": (
            "reuse prior full-CRC extraction only when the immutable completion "
            "marker exactly matches pinned resource and structure manifests"
        ),
        "full_tree_reread_performed": False,
        "extraction_performed": False,
    }


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "extract_root": str(args.extract_root.expanduser().absolute()),
    }
    write_json(args.output, report)
    try:
        completion = validate_completion(
            args.extract_root.expanduser().absolute(),
            read_json(args.resource_manifest),
            read_json(args.structure_manifest),
        )
        report.update(
            {
                "status": "complete",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                **completion,
            }
        )
        write_json(args.output, report)
        return 0
    except Exception as exc:  # noqa: BLE001 - preserve failure evidence
        report.update(
            {
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": repr(exc),
            }
        )
        write_json(args.output, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
