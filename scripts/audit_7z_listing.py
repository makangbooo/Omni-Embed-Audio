#!/usr/bin/env python3
"""Reject unsafe or unexpected members before extracting a pinned 7z archive."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listing", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-wav-files", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_slt_members(text: str) -> list[dict[str, str]]:
    """Parse member records following the 7-Zip ``----------`` separator."""

    lines = text.splitlines()
    try:
        member_start = next(
            index + 1 for index, line in enumerate(lines) if line.strip() == "----------"
        )
    except StopIteration as exc:
        raise ValueError("7-Zip technical listing has no member separator") from exc

    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for raw_line in [*lines[member_start:], ""]:
        line = raw_line.rstrip("\r\n")
        if not line.strip():
            if current:
                records.append(current)
                current = {}
            continue
        if " = " not in line:
            raise ValueError(f"unexpected 7-Zip technical-listing line: {line!r}")
        key, value = line.split(" = ", 1)
        if key in current:
            raise ValueError(f"duplicate field {key!r} in 7-Zip member record")
        current[key] = value
    if not records:
        raise ValueError("7-Zip technical listing contains no members")
    return records


def normalized_member_path(raw_path: str) -> str:
    candidate = raw_path.replace("\\", "/")
    if not candidate or candidate.startswith(("/", "//")):
        raise ValueError(f"empty or absolute archive member path: {raw_path!r}")
    if WINDOWS_DRIVE.match(candidate):
        raise ValueError(f"Windows-absolute archive member path: {raw_path!r}")
    path = PurePosixPath(candidate)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"non-canonical archive member path: {raw_path!r}")
    normalized = path.as_posix()
    if normalized != candidate.rstrip("/"):
        raise ValueError(f"non-canonical archive member path: {raw_path!r}")
    return normalized


def audit_members(
    records: list[dict[str, str]], expected_wav_files: int
) -> dict[str, Any]:
    if expected_wav_files <= 0:
        raise ValueError("expected_wav_files must be positive")

    paths: list[str] = []
    file_paths: list[str] = []
    directory_paths: list[str] = []
    for record in records:
        if "Path" not in record:
            raise ValueError(f"7-Zip member record has no Path: {record!r}")
        path = normalized_member_path(record["Path"])
        lowered_keys = {key.casefold() for key in record}
        if lowered_keys & {"symbolic link", "hard link"}:
            raise ValueError(f"link member is not allowed: {path!r}")
        attributes = record.get("Attributes", "").lstrip()
        is_directory = attributes.startswith("D") or record["Path"].endswith(("/", "\\"))
        paths.append(path)
        if is_directory:
            directory_paths.append(path)
        else:
            file_paths.append(path)

    duplicates = sorted(path for path, count in Counter(paths).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate archive member paths: {duplicates[:20]}")
    casefolded: dict[str, list[str]] = {}
    for path in paths:
        casefolded.setdefault(path.casefold(), []).append(path)
    case_collisions = sorted(
        values for values in casefolded.values() if len(values) > 1
    )
    if case_collisions:
        raise ValueError(f"case-insensitive archive path collisions: {case_collisions[:20]}")

    unexpected_files = sorted(
        path for path in file_paths if not path.casefold().endswith(".wav")
    )
    if unexpected_files:
        raise ValueError(f"unexpected non-WAV archive files: {unexpected_files[:20]}")
    if len(file_paths) != expected_wav_files:
        raise ValueError(
            f"archive WAV count mismatch: {len(file_paths)} != {expected_wav_files}"
        )

    return {
        "status": "complete",
        "member_records": len(records),
        "file_members": len(file_paths),
        "directory_members": len(directory_paths),
        "wav_members": len(file_paths),
        "expected_wav_files": expected_wav_files,
        "path_traversal_members": 0,
        "absolute_path_members": 0,
        "link_members": 0,
        "unexpected_file_members": 0,
        "case_insensitive_collisions": 0,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "archive": str(args.archive.resolve()),
        "listing": str(args.listing.resolve()),
        "expected_wav_files": args.expected_wav_files,
    }
    write_json(args.output, report)
    try:
        text = args.listing.read_text(encoding="utf-8")
        records = parse_slt_members(text)
        report.update(audit_members(records, args.expected_wav_files))
        write_json(args.output, report)
        return 0
    except Exception as exc:  # noqa: BLE001 - preserve the audit failure
        report["status"] = "failed"
        report["error"] = repr(exc)
        write_json(args.output, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
