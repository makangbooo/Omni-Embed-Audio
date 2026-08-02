#!/usr/bin/env python3
"""Safely extract the author-provided AudioCaps raw-audio ZIP once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--progress-files", type=int, default=500)
    parser.add_argument("--minimum-free-margin-bytes", type=int, default=1 << 30)
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def safe_member_parts(name: str) -> tuple[str, ...]:
    if not name or "\\" in name or name.startswith("/"):
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    path = PurePosixPath(name)
    parts = tuple(part for part in path.parts if part not in ("", "."))
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    if ":" in parts[0]:
        raise ValueError(f"unsafe ZIP member drive prefix: {name!r}")
    return parts


def is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = info.external_attr >> 16
    return stat.S_ISLNK(mode)


def plan_members(
    infos: list[zipfile.ZipInfo],
    wrapper_name: str,
) -> tuple[list[tuple[zipfile.ZipInfo, tuple[str, ...]]], int]:
    files: list[tuple[zipfile.ZipInfo, tuple[str, ...]]] = []
    for info in infos:
        parts = safe_member_parts(info.filename)
        if is_symlink(info):
            raise ValueError(f"ZIP symlink is not allowed: {info.filename!r}")
        if info.is_dir():
            continue
        files.append((info, parts))
    if not files:
        raise ValueError("ZIP contains no regular files")

    strip_levels = 0
    remaining = [parts for _, parts in files]
    while all(len(parts) > 1 and parts[0] == wrapper_name for parts in remaining):
        strip_levels += 1
        remaining = [parts[1:] for parts in remaining]

    planned: list[tuple[zipfile.ZipInfo, tuple[str, ...]]] = []
    seen: set[tuple[str, ...]] = set()
    for (info, _), destination_parts in zip(files, remaining, strict=True):
        if not destination_parts or destination_parts in seen:
            raise ValueError(
                f"duplicate or empty destination after flattening: {info.filename!r}"
            )
        seen.add(destination_parts)
        planned.append((info, destination_parts))
    return planned, strip_levels


def extract_archive(args: argparse.Namespace) -> dict[str, Any]:
    if args.progress_files < 1:
        raise ValueError("progress-files must be positive")
    if args.minimum_free_margin_bytes < 0:
        raise ValueError("minimum-free-margin-bytes must be non-negative")
    archive = args.archive.resolve()
    output_dir = args.output_dir.resolve()
    report_path = (
        args.report.resolve()
        if args.report
        else output_dir.with_name(output_dir.name + ".extraction_report.json")
    )
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if output_dir.exists():
        raise FileExistsError(f"output path already exists; refusing overwrite: {output_dir}")
    if report_path.exists():
        raise FileExistsError(f"report already exists; refusing overwrite: {report_path}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    print("STAGE_START=archive_sha256", flush=True)
    archive_sha256 = sha256_file(archive)
    print(f"STAGE_END=archive_sha256 SHA256={archive_sha256}", flush=True)
    with zipfile.ZipFile(archive) as handle:
        planned, stripped_levels = plan_members(handle.infolist(), output_dir.name)
        expected_bytes = sum(info.file_size for info, _ in planned)
        free_bytes = shutil.disk_usage(output_dir.parent).free
        required_bytes = expected_bytes + args.minimum_free_margin_bytes
        if free_bytes < required_bytes:
            raise OSError(
                f"insufficient free space: free={free_bytes}, required={required_bytes}"
            )

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        staging = output_dir.with_name(
            f".{output_dir.name}.extracting.{stamp}.{os.getpid()}"
        )
        staging.mkdir(parents=False, exist_ok=False)
        extracted_bytes = 0
        try:
            for index, (info, parts) in enumerate(planned, start=1):
                destination = staging.joinpath(*parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(info, "r") as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                actual_size = destination.stat().st_size
                if actual_size != info.file_size:
                    raise ValueError(
                        f"extracted size mismatch for {info.filename!r}: "
                        f"{actual_size} != {info.file_size}"
                    )
                extracted_bytes += actual_size
                if index % args.progress_files == 0 or index == len(planned):
                    print(
                        f"PROGRESS files={index}/{len(planned)} "
                        f"bytes={extracted_bytes}/{expected_bytes}",
                        flush=True,
                    )
            staging.replace(output_dir)
        except BaseException:
            print(f"FAILED_STAGING_DIRECTORY={staging}", file=sys.stderr)
            raise

    report = {
        "schema_version": 1,
        "status": "complete",
        "started_at": args.started_at,
        "finished_at": now(),
        "archive": {
            "path": str(archive),
            "size_bytes": archive.stat().st_size,
            "sha256": archive_sha256,
        },
        "output_directory": str(output_dir),
        "stripped_wrapper_name": output_dir.name if stripped_levels else None,
        "stripped_wrapper_levels": stripped_levels,
        "file_count": len(planned),
        "extracted_bytes": expected_bytes,
        "free_bytes_before_extraction": free_bytes,
        "overwrite_or_delete_performed": False,
    }
    write_json(report_path, report)
    print("AUDIOCAPS_EXTRACTION_STATUS=complete")
    print(f"OUTPUT_DIRECTORY={output_dir}")
    print(f"STRIPPED_WRAPPER_NAME={output_dir.name if stripped_levels else 'none'}")
    print(f"STRIPPED_WRAPPER_LEVELS={stripped_levels}")
    print(f"FILE_COUNT={len(planned)}")
    print(f"EXTRACTED_BYTES={expected_bytes}")
    print(f"ARCHIVE_SHA256={report['archive']['sha256']}")
    print(f"REPORT_PATH={report_path}")
    return report


def main() -> int:
    args = parse_args()
    args.started_at = now()
    try:
        extract_archive(args)
        return 0
    except BaseException as exc:
        print("AUDIOCAPS_EXTRACTION_STATUS=failed", file=sys.stderr)
        print(f"ERROR_TYPE={type(exc).__name__}", file=sys.stderr)
        print(f"ERROR={exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
