#!/usr/bin/env python3
"""Safely extract and validate the pinned MECAT-Caption 00A test shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA256 = "644cf75e2509c633452a18e36c41b285a317c6cbc06198d7dfe406c5aa5122c4"
DATASET_REVISION = "be4a24c3f7309d74208e08a7cce49e72cb7a5834"
CAPTION_FIELDS = ("long", "short", "speech", "music", "sound", "environment")
POSITIVE_UIQ_TYPES = ("question", "imperative", "paraphrase", "tagging")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--extract-root", type=Path, required=True)
    parser.add_argument("--uiq-root", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--statistics-output", type=Path, required=True)
    parser.add_argument("--expected-examples", type=int, default=848)
    parser.add_argument("--expected-negative-rows", type=int, default=409)
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


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_member(name: str) -> PurePosixPath:
    if "\\" in name:
        raise ValueError(f"archive member uses a backslash: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe archive member path: {name!r}")
    return path


def archive_sample_index(archive: Path, expected_examples: int) -> dict[str, dict[str, str]]:
    samples: dict[str, dict[str, str]] = {}
    seen_members: set[str] = set()
    with tarfile.open(archive, mode="r:gz") as handle:
        for member in handle:
            relative = safe_relative_member(member.name)
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(
                    f"archive contains a non-regular member: {member.name!r}"
                )
            if member.name in seen_members:
                raise ValueError(f"duplicate archive member: {member.name!r}")
            seen_members.add(member.name)
            suffix = relative.suffix.lower()
            if suffix not in {".flac", ".json"}:
                raise ValueError(f"unexpected archive payload: {member.name!r}")
            sample_id = relative.stem
            entry = samples.setdefault(sample_id, {})
            label = suffix.removeprefix(".")
            if label in entry:
                raise ValueError(f"duplicate {label} for sample {sample_id!r}")
            entry[label] = relative.as_posix()

    incomplete = sorted(
        sample_id
        for sample_id, files in samples.items()
        if set(files) != {"flac", "json"}
    )
    if incomplete:
        raise ValueError(f"samples without exact FLAC/JSON pairs: {incomplete[:20]}")
    if len(samples) != expected_examples:
        raise ValueError(
            f"archive sample count mismatch: {len(samples)} != {expected_examples}"
        )
    return samples


def extract_regular_members(
    archive: Path,
    destination: Path,
    sample_index: dict[str, dict[str, str]],
) -> None:
    expected_members = {
        member_name
        for files in sample_index.values()
        for member_name in files.values()
    }
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, mode="r:gz") as handle:
        for member_name in sorted(expected_members):
            member = handle.getmember(member_name)
            if not member.isfile():
                raise ValueError(f"member changed type during extraction: {member_name}")
            source = handle.extractfile(member)
            if source is None:
                raise ValueError(f"unable to read archive member: {member_name}")
            target = destination.joinpath(*PurePosixPath(member_name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            with temporary.open("xb") as output:
                while chunk := source.read(8 * 1024 * 1024):
                    output.write(chunk)
            temporary.replace(target)


def load_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"MECAT metadata is not an object: {path}")
    missing = sorted(set(CAPTION_FIELDS) - set(value))
    if missing:
        raise ValueError(f"MECAT metadata fields missing in {path}: {missing}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL line in {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row in {path}:{line_number}")
            rows.append(value)
    return rows


def validate_uiq(
    uiq_root: Path,
    sample_ids: set[str],
    expected_negative_rows: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for query_type in POSITIVE_UIQ_TYPES:
        path = uiq_root / f"mecat_{query_type}_queries.jsonl"
        rows = read_jsonl(path)
        ids = [str(row.get("audio_id", "")) for row in rows]
        if len(rows) != len(sample_ids) or len(set(ids)) != len(sample_ids):
            raise ValueError(
                f"positive UIQ count/uniqueness mismatch for {query_type}: "
                f"rows={len(rows)}, unique={len(set(ids))}"
            )
        if set(ids) != sample_ids:
            raise ValueError(
                f"positive UIQ ID mismatch for {query_type}: "
                f"uiq_only={sorted(set(ids) - sample_ids)[:20]}, "
                f"archive_only={sorted(sample_ids - set(ids))[:20]}"
            )
        invalid = [
            index
            for index, row in enumerate(rows, start=1)
            if row.get("dataset") != "mecat"
            or row.get("dataset_slug") != "mecat"
            or row.get("query_type") != query_type
            or not isinstance(row.get("generated_query"), str)
            or not row["generated_query"].strip()
        ]
        if invalid:
            raise ValueError(f"invalid positive UIQ schema for {query_type}: {invalid[:20]}")
        report[query_type] = {
            "rows": len(rows),
            "unique_audio_ids": len(set(ids)),
            "exact_archive_id_set_match": True,
        }

    negative_path = uiq_root / "mecat_negative_queries.jsonl"
    negative_rows = read_jsonl(negative_path)
    if len(negative_rows) != expected_negative_rows:
        raise ValueError(
            f"negative UIQ row count mismatch: {len(negative_rows)} != "
            f"{expected_negative_rows}"
        )
    negative_ids = [str(row.get("audio_id", "")) for row in negative_rows]
    unknown = sorted(set(negative_ids) - sample_ids)
    if unknown:
        raise ValueError(f"negative UIQ IDs absent from archive: {unknown[:20]}")
    invalid = [
        index
        for index, row in enumerate(negative_rows, start=1)
        if row.get("dataset") != "mecat"
        or row.get("dataset_slug") != "mecat"
        or row.get("query_type") != "negative"
        or not isinstance(row.get("negative_query"), str)
        or not row["negative_query"].strip()
    ]
    if invalid:
        raise ValueError(f"invalid negative UIQ schema rows: {invalid[:20]}")
    report["negative"] = {
        "rows": len(negative_rows),
        "unique_audio_ids": len(set(negative_ids)),
        "all_audio_ids_in_archive": True,
        "hard_negative_audio_id_source": "[MISSING] not present in released JSONL",
    }
    return report


def decode_audio(path: Path) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    info = sf.info(str(path))
    frames = 0
    all_finite = True
    for block in sf.blocks(str(path), blocksize=65536, dtype="float32", always_2d=True):
        frames += int(block.shape[0])
        all_finite = all_finite and bool(np.isfinite(block).all())
    if frames != info.frames:
        raise ValueError(f"decoded frame mismatch for {path}: {frames} != {info.frames}")
    if not all_finite:
        raise ValueError(f"non-finite decoded samples in {path}")
    return {
        "duration_seconds": float(info.duration),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "format": info.format,
        "subtype": info.subtype,
        "decode_ok": True,
        "all_samples_finite": True,
    }


def main() -> int:
    args = parse_args()
    archive = args.archive.resolve()
    extract_root = args.extract_root.resolve()
    uiq_root = args.uiq_root.resolve()
    manifest_output = args.manifest_output.resolve()
    statistics_output = args.statistics_output.resolve()
    archive_sha256 = sha256_file(archive)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output("status", "--short"),
        "dataset": "MECAT-Caption",
        "dataset_revision": DATASET_REVISION,
        "configuration": "00A",
        "split": "test",
        "expected_examples": args.expected_examples,
        "paper_reported_examples": 847,
        "archive": str(archive),
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": archive_sha256,
        "extract_root": str(extract_root),
        "uiq_root": str(uiq_root),
        "manifest_output": str(manifest_output),
    }
    write_json(statistics_output, report)

    try:
        if archive_sha256 != ARCHIVE_SHA256:
            raise ValueError(
                f"archive SHA256 mismatch: {archive_sha256} != {ARCHIVE_SHA256}"
            )
        archive_index = archive_sample_index(archive, args.expected_examples)
        marker = extract_root / ".data05_mecat_extraction_complete.json"
        if extract_root.exists():
            if not marker.is_file():
                raise ValueError(
                    f"existing extract root has no completion marker: {extract_root}"
                )
            marker_value = json.loads(marker.read_text(encoding="utf-8"))
            expected_marker = {
                "archive_sha256": ARCHIVE_SHA256,
                "dataset_revision": DATASET_REVISION,
                "examples": args.expected_examples,
            }
            if marker_value != expected_marker:
                raise ValueError(
                    f"extraction marker mismatch: {marker_value} != {expected_marker}"
                )
        else:
            staging = extract_root.parent / f".{extract_root.name}.extracting-{os.getpid()}"
            if staging.exists():
                raise ValueError(f"staging path already exists: {staging}")
            report["staging_root"] = str(staging)
            write_json(statistics_output, report)
            extract_regular_members(archive, staging, archive_index)
            write_json(
                staging / ".data05_mecat_extraction_complete.json",
                {
                    "archive_sha256": ARCHIVE_SHA256,
                    "dataset_revision": DATASET_REVISION,
                    "examples": args.expected_examples,
                },
            )
            extract_root.parent.mkdir(parents=True, exist_ok=True)
            staging.replace(extract_root)

        sample_ids = set(archive_index)
        uiq_report = validate_uiq(uiq_root, sample_ids, args.expected_negative_rows)
        manifest_rows: list[dict[str, Any]] = []
        sample_rates: Counter[int] = Counter()
        channel_counts: Counter[int] = Counter()
        caption_field_types: dict[str, Counter[str]] = {
            field: Counter() for field in CAPTION_FIELDS
        }
        durations: list[float] = []
        total_audio_bytes = 0
        for sample_id in sorted(sample_ids):
            files = archive_index[sample_id]
            audio_path = extract_root.joinpath(*PurePosixPath(files["flac"]).parts)
            json_path = extract_root.joinpath(*PurePosixPath(files["json"]).parts)
            metadata = load_json_object(json_path)
            decoded = decode_audio(audio_path)
            for field in CAPTION_FIELDS:
                caption_field_types[field][type(metadata[field]).__name__] += 1
            durations.append(decoded["duration_seconds"])
            sample_rates[decoded["sample_rate"]] += 1
            channel_counts[decoded["channels"]] += 1
            total_audio_bytes += audio_path.stat().st_size
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "audio_path": str(audio_path),
                    "audio_relpath": files["flac"],
                    "metadata_path": str(json_path),
                    "metadata_relpath": files["json"],
                    "caption_fields": {field: metadata[field] for field in CAPTION_FIELDS},
                    "retrieval_caption_field": None,
                    "retrieval_caption_source": (
                        "[MISSING] paper/code do not identify the MECAT field used "
                        "for retrieval"
                    ),
                    "split": "test",
                    "duration_seconds": decoded["duration_seconds"],
                    "sample_rate": decoded["sample_rate"],
                    "channels": decoded["channels"],
                    "frames": decoded["frames"],
                    "format": decoded["format"],
                    "subtype": decoded["subtype"],
                    "audio_size_bytes": audio_path.stat().st_size,
                    "audio_sha256": sha256_file(audio_path),
                    "metadata_sha256": sha256_file(json_path),
                    "file_exists": True,
                    "decode_ok": True,
                    "all_samples_finite": True,
                    "include_in_training": False,
                    "filter_reason": "evaluation_only",
                    "leakage_blocklist_status": "NOT_APPLICABLE_EVALUATION_SPLIT",
                }
            )

        write_jsonl(manifest_output, manifest_rows)
        report.update(
            {
                "status": "complete",
                "finished_at": utc_now(),
                "archive_examples": len(archive_index),
                "decoded_files": len(manifest_rows),
                "total_audio_bytes": total_audio_bytes,
                "total_duration_seconds": math.fsum(durations),
                "duration_min_seconds": min(durations),
                "duration_max_seconds": max(durations),
                "duration_mean_seconds": math.fsum(durations) / len(durations),
                "sample_rate_counts": dict(sorted(sample_rates.items())),
                "channel_counts": dict(sorted(channel_counts.items())),
                "caption_field_type_counts": {
                    field: dict(sorted(counts.items()))
                    for field, counts in caption_field_types.items()
                },
                "uiq": uiq_report,
                "manifest_size_bytes": manifest_output.stat().st_size,
                "manifest_sha256": sha256_file(manifest_output),
                "source_tags": {
                    "dataset_revision": "[CODE] pinned official Hugging Face revision",
                    "archive_count": "[CODE] official 00A test contains 848 pairs",
                    "paper_count_difference": (
                        "[MISSING] paper reports 847; no excluded sample or filter is published"
                    ),
                    "retrieval_caption_field": (
                        "[MISSING] all six official fields are preserved without choosing one"
                    ),
                },
            }
        )
        write_json(statistics_output, report)
        return 0
    except Exception as exc:  # noqa: BLE001 - retain complete failure evidence
        report["status"] = "failed"
        report["finished_at"] = utc_now()
        report["error"] = repr(exc)
        write_json(statistics_output, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
