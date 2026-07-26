#!/usr/bin/env python3
"""Safely and resumably extract the pinned SQuTR ZIP without overwriting files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zlib
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from scripts.audit_squtr_archive import (
        member_kind,
        normalized_member_path,
        read_json,
    )
except ModuleNotFoundError:
    from audit_squtr_archive import member_kind, normalized_member_path, read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--extract-root", type=Path, required=True)
    parser.add_argument("--resource-manifest", type=Path, required=True)
    parser.add_argument("--structure-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crc32_file(path: Path) -> int:
    checksum = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum = zlib.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def verify_file_against_member(path: Path, info: zipfile.ZipInfo) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"existing extraction target is not a regular file: {path}")
    actual_size = path.stat().st_size
    if actual_size != info.file_size:
        raise ValueError(
            f"existing extraction target size mismatch; refusing to overwrite: "
            f"{path} ({actual_size} != {info.file_size})"
        )
    actual_crc = crc32_file(path)
    if actual_crc != info.CRC:
        raise ValueError(
            f"existing extraction target CRC mismatch; refusing to overwrite: "
            f"{path} ({actual_crc:08x} != {info.CRC:08x})"
        )


def ensure_safe_directory(root: Path, parts: tuple[str, ...]) -> Path:
    current = root
    for part in parts:
        current = current / part
        if current.exists() or current.is_symlink():
            if current.is_symlink() or not current.is_dir():
                raise ValueError(f"unsafe extraction directory component: {current}")
        else:
            current.mkdir()
    return current


def validate_manifests(
    archive: Path,
    resource_manifest: dict[str, Any],
    structure_manifest: dict[str, Any],
) -> tuple[str, int, str]:
    assets = resource_manifest.get("files")
    if not isinstance(assets, list) or len(assets) != 1:
        raise ValueError("SQuTR resource manifest must contain exactly one asset")
    asset = assets[0]
    expected_size = int(asset["size_bytes"])
    expected_sha256 = str(asset["sha256"])
    actual_size = archive.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"archive size mismatch: {actual_size} != {expected_size}")
    actual_sha256 = sha256_file(archive)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"archive SHA256 mismatch: {actual_sha256} != {expected_sha256}"
        )
    archive_root = structure_manifest.get("observed_archive_root")
    if not isinstance(archive_root, str) or not archive_root:
        raise ValueError("structure manifest has no observed_archive_root")
    return archive_root, actual_size, actual_sha256


def safe_extract(
    archive: Path,
    extract_root: Path,
    structure_manifest: dict[str, Any],
    progress_every: int,
) -> dict[str, Any]:
    expected_records = int(structure_manifest["observed_zip_member_records"])
    expected_files = int(structure_manifest["observed_file_members"])
    expected_directories = int(structure_manifest["observed_directory_members"])
    expected_uncompressed = int(
        structure_manifest["observed_uncompressed_member_bytes"]
    )
    extract_root.mkdir(parents=True, exist_ok=True)
    if extract_root.is_symlink() or not extract_root.is_dir():
        raise ValueError(f"extract root is not a regular directory: {extract_root}")

    extracted_files = 0
    verified_existing_files = 0
    restarted_partial_files = 0
    file_count = 0
    directory_count = 0
    uncompressed_bytes = 0
    expected_paths: set[str] = set()

    with zipfile.ZipFile(archive) as handle:
        infos = handle.infolist()
        if len(infos) != expected_records:
            raise ValueError(
                f"ZIP member count drift: {len(infos)} != {expected_records}"
            )
        for index, info in enumerate(infos, start=1):
            normalized = normalized_member_path(info.filename)
            kind = member_kind(info)
            target = extract_root.joinpath(*PurePosixPath(normalized).parts)
            expected_paths.add(normalized)
            if kind == "directory":
                directory_count += 1
                ensure_safe_directory(
                    extract_root,
                    tuple(PurePosixPath(normalized).parts),
                )
                continue

            file_count += 1
            uncompressed_bytes += info.file_size
            ensure_safe_directory(
                extract_root,
                tuple(PurePosixPath(normalized).parts[:-1]),
            )
            if target.exists() or target.is_symlink():
                verify_file_against_member(target, info)
                verified_existing_files += 1
            else:
                partial = target.with_name(target.name + ".data13b.part")
                if partial.exists() or partial.is_symlink():
                    if partial.is_symlink() or not partial.is_file():
                        raise ValueError(f"unsafe managed partial path: {partial}")
                    restarted_partial_files += 1
                with handle.open(info, "r") as source, partial.open("wb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                verify_file_against_member(partial, info)
                if target.exists() or target.is_symlink():
                    raise ValueError(
                        f"target appeared during extraction; refusing to overwrite: {target}"
                    )
                partial.replace(target)
                extracted_files += 1

            if file_count % progress_every == 0:
                print(
                    "[PROGRESS] files="
                    f"{file_count}/{expected_files} extracted={extracted_files} "
                    f"verified_existing={verified_existing_files}",
                    flush=True,
                )

    if file_count != expected_files:
        raise ValueError(f"extracted file count drift: {file_count} != {expected_files}")
    if directory_count != expected_directories:
        raise ValueError(
            f"directory member count drift: {directory_count} != {expected_directories}"
        )
    if uncompressed_bytes != expected_uncompressed:
        raise ValueError(
            f"uncompressed byte count drift: {uncompressed_bytes} != "
            f"{expected_uncompressed}"
        )

    marker_names = {
        ".data13b_extraction_in_progress.json",
        ".data13b_extraction_complete.json",
    }
    unexpected: list[str] = []
    managed_partials: list[str] = []
    for path in extract_root.rglob("*"):
        if path.is_dir():
            continue
        relative = path.relative_to(extract_root).as_posix()
        if relative in expected_paths or relative in marker_names:
            continue
        if relative.endswith(".data13b.part"):
            managed_partials.append(relative)
        else:
            unexpected.append(relative)
        if len(unexpected) >= 20:
            break
    if unexpected:
        raise ValueError(f"unexpected extracted files: {unexpected}")
    if managed_partials:
        raise ValueError(f"incomplete managed partial files remain: {managed_partials[:20]}")

    return {
        "zip_member_records": expected_records,
        "file_members": file_count,
        "directory_members": directory_count,
        "uncompressed_member_bytes": uncompressed_bytes,
        "extracted_files": extracted_files,
        "verified_existing_files": verified_existing_files,
        "restarted_partial_files": restarted_partial_files,
        "full_member_crc_verified": True,
        "unexpected_files": 0,
        "remaining_partial_files": 0,
    }


def main() -> int:
    args = parse_args()
    archive = args.archive.resolve()
    extract_root = args.extract_root.expanduser().absolute()
    if extract_root.is_symlink():
        raise ValueError(f"extract root must not be a symbolic link: {extract_root}")
    extract_root.mkdir(parents=True, exist_ok=True)
    if not extract_root.is_dir():
        raise ValueError(f"extract root is not a directory: {extract_root}")
    resource_manifest = read_json(args.resource_manifest)
    structure_manifest = read_json(args.structure_manifest)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "archive": str(archive),
        "extract_root": str(extract_root),
        "extraction_policy": (
            "verify-and-reuse exact existing files; refuse mismatched final files; "
            "restart only tool-owned .data13b.part files"
        ),
    }
    write_json(args.output, report)
    try:
        archive_root, archive_size, archive_sha256 = validate_manifests(
            archive,
            resource_manifest,
            structure_manifest,
        )
        in_progress = {
            "archive_sha256": archive_sha256,
            "archive_root": archive_root,
            "expected_file_members": structure_manifest["observed_file_members"],
            "expected_uncompressed_member_bytes": structure_manifest[
                "observed_uncompressed_member_bytes"
            ],
        }
        progress_marker = extract_root / ".data13b_extraction_in_progress.json"
        if progress_marker.exists():
            existing = read_json(progress_marker)
            if existing != in_progress:
                raise ValueError(
                    f"in-progress marker mismatch: {existing} != {in_progress}"
                )
        else:
            write_json(progress_marker, in_progress)

        extraction = safe_extract(
            archive,
            extract_root,
            structure_manifest,
            args.progress_every,
        )
        complete_marker = extract_root / ".data13b_extraction_complete.json"
        completed = {
            **in_progress,
            "file_members": extraction["file_members"],
            "uncompressed_member_bytes": extraction["uncompressed_member_bytes"],
            "full_member_crc_verified": True,
        }
        if complete_marker.exists():
            existing = read_json(complete_marker)
            if existing != completed:
                raise ValueError(
                    f"completion marker mismatch: {existing} != {completed}"
                )
        else:
            write_json(complete_marker, completed)

        report.update(
            {
                "status": "complete",
                "finished_at": utc_now(),
                "archive_size_bytes": archive_size,
                "archive_sha256": archive_sha256,
                "archive_root": archive_root,
                "extraction": extraction,
                "completion_marker": str(complete_marker),
            }
        )
        write_json(args.output, report)
        return 0
    except Exception as exc:  # noqa: BLE001 - preserve failure evidence
        report["status"] = "failed"
        report["finished_at"] = utc_now()
        report["error"] = repr(exc)
        write_json(args.output, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
