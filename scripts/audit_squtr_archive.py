#!/usr/bin/env python3
"""Read-only safety and structure audit for the pinned SQuTR ZIP archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--resource-manifest", type=Path, required=True)
    parser.add_argument("--structure-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_member_path(raw_path: str) -> str:
    if "\x00" in raw_path:
        raise ValueError("NUL byte in ZIP member path")
    if "\\" in raw_path:
        raise ValueError(f"backslash in ZIP member path: {raw_path!r}")
    if not raw_path or raw_path.startswith(("/", "//")):
        raise ValueError(f"empty or absolute ZIP member path: {raw_path!r}")
    if WINDOWS_DRIVE.match(raw_path):
        raise ValueError(f"Windows-absolute ZIP member path: {raw_path!r}")
    path = PurePosixPath(raw_path)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"non-canonical ZIP member path: {raw_path!r}")
    normalized = path.as_posix()
    if normalized != raw_path.rstrip("/"):
        raise ValueError(f"non-canonical ZIP member path: {raw_path!r}")
    return normalized


def member_kind(info: zipfile.ZipInfo) -> str:
    if info.flag_bits & 0x1:
        raise ValueError(f"encrypted ZIP member is not allowed: {info.filename!r}")
    if info.is_dir():
        return "directory"
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    if info.create_system == 3 and unix_mode:
        file_type = stat.S_IFMT(unix_mode)
        if file_type == stat.S_IFLNK:
            raise ValueError(f"symbolic-link ZIP member is not allowed: {info.filename!r}")
        if file_type not in {0, stat.S_IFREG}:
            raise ValueError(f"special ZIP member is not allowed: {info.filename!r}")
    return "file"


def find_archive_root(
    file_paths: set[str], subset_relative_paths: list[str]
) -> tuple[str, dict[str, str]]:
    roots: dict[str, str] = {}
    for subset_path in subset_relative_paths:
        suffix = f"{subset_path}/corpus.jsonl"
        matches = sorted(
            path for path in file_paths if path == suffix or path.endswith(f"/{suffix}")
        )
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one corpus anchor for {subset_path}, found {matches}"
            )
        match = matches[0]
        roots[subset_path] = match[: -len(suffix)].rstrip("/")
    unique_roots = sorted(set(roots.values()))
    if len(unique_roots) != 1 or not unique_roots[0]:
        raise ValueError(f"subsets do not share one non-empty archive root: {roots}")
    return unique_roots[0], roots


def audit_zip_structure(
    archive: Path,
    resource_manifest: dict[str, Any],
    structure_manifest: dict[str, Any],
) -> dict[str, Any]:
    assets = resource_manifest.get("files")
    if not isinstance(assets, list) or len(assets) != 1:
        raise ValueError("SQuTR resource manifest must contain exactly one asset")
    asset = assets[0]
    expected_size = asset.get("size_bytes")
    expected_sha256 = asset.get("sha256")
    if not isinstance(expected_size, int) or expected_size <= 0:
        raise ValueError("resource manifest has no valid archive size")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("resource manifest has no valid archive SHA256")
    actual_size = archive.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"archive size mismatch: {actual_size} != {expected_size}")
    actual_sha256 = hash_file(archive)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"archive SHA256 mismatch: {actual_sha256} != {expected_sha256}"
        )

    subsets = structure_manifest.get("subsets")
    conditions = structure_manifest.get("conditions")
    core_paths = structure_manifest.get("core_relative_paths")
    if not isinstance(subsets, list) or not subsets:
        raise ValueError("structure manifest has no subsets")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("structure manifest has no conditions")
    if not isinstance(core_paths, list) or not core_paths:
        raise ValueError("structure manifest has no core paths")

    with zipfile.ZipFile(archive) as handle:
        infos = handle.infolist()
    if not infos:
        raise ValueError("ZIP archive contains no members")

    normalized_paths: list[str] = []
    file_infos: list[tuple[str, zipfile.ZipInfo]] = []
    directory_count = 0
    for info in infos:
        path = normalized_member_path(info.filename)
        kind = member_kind(info)
        normalized_paths.append(path)
        if kind == "directory":
            directory_count += 1
        else:
            file_infos.append((path, info))

    duplicate_paths = sorted(
        path for path, count in Counter(normalized_paths).items() if count > 1
    )
    if duplicate_paths:
        raise ValueError(f"duplicate ZIP member paths: {duplicate_paths[:20]}")
    casefold_groups: dict[str, list[str]] = {}
    for path in normalized_paths:
        casefold_groups.setdefault(path.casefold(), []).append(path)
    case_collisions = sorted(
        paths for paths in casefold_groups.values() if len(paths) > 1
    )
    if case_collisions:
        raise ValueError(
            f"case-insensitive ZIP path collisions: {case_collisions[:20]}"
        )

    file_paths = {path for path, _ in file_infos}
    subset_paths = [str(item["relative_path"]) for item in subsets]
    archive_root, roots = find_archive_root(file_paths, subset_paths)
    violations: list[str] = []
    subset_reports: list[dict[str, Any]] = []
    total_condition_wavs = 0

    for subset in subsets:
        relative_path = str(subset["relative_path"])
        subset_prefix = f"{archive_root}/{relative_path}"
        expected_queries = int(subset["expected_unique_queries"])
        documented_paths = [
            *[f"{subset_prefix}/{path}" for path in core_paths],
            *[
                f"{subset_prefix}/{condition['documented_query_file']}"
                for condition in conditions
            ],
        ]
        presence = {path: path in file_paths for path in documented_paths}
        missing = sorted(path for path, present in presence.items() if not present)
        if missing:
            violations.append(f"{relative_path}: missing documented paths {missing}")

        condition_counts: dict[str, int] = {}
        for condition in conditions:
            condition_id = str(condition["id"])
            audio_prefix = f"{subset_prefix}/{condition['audio_directory']}/"
            count = sum(
                path.startswith(audio_prefix) and path.casefold().endswith(".wav")
                for path in file_paths
            )
            condition_counts[condition_id] = count
            total_condition_wavs += count
            if count != expected_queries:
                violations.append(
                    f"{relative_path}/{condition_id}: WAV count "
                    f"{count} != {expected_queries}"
                )

        query_candidates = sorted(
            path
            for path in file_paths
            if path.startswith(f"{subset_prefix}/")
            and PurePosixPath(path).name.casefold().startswith("quer")
            and path.casefold().endswith(".jsonl")
        )
        qrels_candidates = sorted(
            path
            for path in file_paths
            if path.startswith(f"{subset_prefix}/")
            and "qrel" in path.casefold()
            and path.casefold().endswith(".jsonl")
        )
        subset_reports.append(
            {
                "language": subset["language"],
                "name": subset["name"],
                "relative_path": relative_path,
                "archive_prefix": subset_prefix,
                "expected_unique_queries": expected_queries,
                "condition_wav_counts": condition_counts,
                "documented_path_presence": presence,
                "query_jsonl_candidates": query_candidates,
                "qrels_jsonl_candidates": qrels_candidates,
            }
        )

    expected_total = int(structure_manifest["expected_audio_instances"])
    if total_condition_wavs != expected_total:
        violations.append(
            f"total condition WAV count {total_condition_wavs} != {expected_total}"
        )

    extension_counts = Counter(
        PurePosixPath(path).suffix.casefold() or "[no_extension]"
        for path, _ in file_infos
    )
    compressed_bytes = sum(info.compress_size for _, info in file_infos)
    uncompressed_bytes = sum(info.file_size for _, info in file_infos)
    return {
        "status": "complete" if not violations else "failed",
        "dataset": structure_manifest.get("dataset"),
        "official_code_repository": structure_manifest.get(
            "official_code_repository"
        ),
        "official_code_revision": structure_manifest.get("official_code_revision"),
        "source_tags": structure_manifest.get("source_tags"),
        "archive_size_bytes": actual_size,
        "archive_sha256": actual_sha256,
        "zip_member_records": len(infos),
        "file_members": len(file_infos),
        "directory_members": directory_count,
        "compressed_member_bytes": compressed_bytes,
        "uncompressed_member_bytes": uncompressed_bytes,
        "compression_ratio": (
            uncompressed_bytes / compressed_bytes if compressed_bytes else None
        ),
        "archive_root": archive_root,
        "subset_anchor_roots": roots,
        "extension_counts": dict(sorted(extension_counts.items())),
        "condition_wav_files": total_condition_wavs,
        "expected_audio_instances": expected_total,
        "subsets": subset_reports,
        "violations": violations,
        "safety": {
            "path_traversal_members": 0,
            "absolute_path_members": 0,
            "encrypted_members": 0,
            "link_members": 0,
            "special_members": 0,
            "duplicate_paths": 0,
            "case_insensitive_collisions": 0,
        },
        "content_crc_test_performed": False,
        "extraction_performed": False,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "archive": str(args.archive.resolve()),
        "resource_manifest": str(args.resource_manifest.resolve()),
        "structure_manifest": str(args.structure_manifest.resolve()),
    }
    write_json(args.output, report)
    try:
        report.update(
            audit_zip_structure(
                args.archive,
                read_json(args.resource_manifest),
                read_json(args.structure_manifest),
            )
        )
        write_json(args.output, report)
        return 0 if report["status"] == "complete" else 1
    except Exception as exc:  # noqa: BLE001 - preserve the failed audit
        report["status"] = "failed"
        report["error"] = repr(exc)
        report["extraction_performed"] = False
        write_json(args.output, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
