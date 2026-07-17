#!/usr/bin/env python3
"""Validate Clotho v2.1 development/validation audio and build manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.validate_clotho_evaluation import decode_audio, discover_audio
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from validate_clotho_evaluation import decode_audio, discover_audio


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CAPTION_COLUMNS = ["file_name", *(f"caption_{index}" for index in range(1, 6))]
METADATA_COLUMNS = [
    "file_name",
    "keywords",
    "sound_id",
    "sound_link",
    "start_end_samples",
    "manufacturer",
    "license",
]
INVALID_SOUND_IDS = {"", "na", "n/a", "none", "not found", "null"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--development-captions", type=Path, required=True)
    parser.add_argument("--validation-captions", type=Path, required=True)
    parser.add_argument("--development-metadata", type=Path, required=True)
    parser.add_argument("--validation-metadata", type=Path, required=True)
    parser.add_argument("--development-manifest-output", type=Path, required=True)
    parser.add_argument("--validation-manifest-output", type=Path, required=True)
    parser.add_argument("--statistics-output", type=Path, required=True)
    parser.add_argument("--expected-development", type=int, default=3839)
    parser.add_argument("--expected-validation", type=int, default=1045)
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


def hash_file(path: Path, algorithm: str) -> str:
    if algorithm == "md5":
        digest = hashlib.md5(usedforsecurity=False)
    else:
        digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def serialize_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def write_reproducible(path: Path, content: bytes) -> str:
    """Write a canonical artifact once; reuse equal content and reject drift."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file():
            raise ValueError(f"artifact path is not a regular file: {path}")
        if path.read_bytes() != content:
            raise ValueError(f"refusing to overwrite different canonical artifact: {path}")
        return "verified_existing"
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise ValueError(f"stale temporary artifact exists: {temporary}")
    temporary.write_bytes(content)
    temporary.replace(path)
    return "created"


def read_csv_exact(
    path: Path, expected_columns: list[str], encoding: str
) -> list[dict[str, str]]:
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            raise ValueError(
                f"unexpected CSV columns for {path}: {reader.fieldnames} != "
                f"{expected_columns}"
            )
        rows = [dict(row) for row in reader]
    if any(any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"CSV contains missing parsed fields: {path}")
    return rows


def metadata_encoding_report(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    try:
        content.decode("utf-8-sig")
        utf8_decodable = True
        utf8_error = None
    except UnicodeDecodeError as exc:
        utf8_decodable = False
        utf8_error = {
            "start": exc.start,
            "end": exc.end,
            "bytes_hex": content[exc.start : exc.end].hex(),
        }
    latin1 = content.decode("iso-8859-1")
    codepoints = Counter(f"U+{ord(character):04X}" for character in latin1 if ord(character) > 127)
    return {
        "encoding_used": "iso-8859-1",
        "utf8_decodable": utf8_decodable,
        "utf8_error": utf8_error,
        "non_ascii_codepoint_counts": dict(sorted(codepoints.items())),
        "decode_error_policy": "strict; errors are never ignored or replaced",
    }


def validate_unique_names(rows: list[dict[str, str]], label: str) -> list[str]:
    names = [row["file_name"] for row in rows]
    if any(not name for name in names):
        raise ValueError(f"empty file_name in {label}")
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate file_name values in {label}: {duplicates[:20]}")
    casefolded: dict[str, list[str]] = defaultdict(list)
    for name in names:
        casefolded[name.casefold()].append(name)
    collisions = sorted(values for values in casefolded.values() if len(values) > 1)
    if collisions:
        raise ValueError(f"case-insensitive file_name collisions in {label}: {collisions[:20]}")
    return names


def validate_split_text(
    split: str,
    captions_csv: Path,
    metadata_csv: Path,
    expected_examples: int,
) -> tuple[dict[str, list[str]], dict[str, dict[str, str]], dict[str, Any]]:
    caption_rows = read_csv_exact(captions_csv, CAPTION_COLUMNS, "utf-8-sig")
    # The fixed v2.1 development/validation metadata contains raw 0xC1 bytes.
    metadata_rows = read_csv_exact(metadata_csv, METADATA_COLUMNS, "iso-8859-1")
    if len(caption_rows) != expected_examples:
        raise ValueError(
            f"{split} caption row count mismatch: {len(caption_rows)} != {expected_examples}"
        )
    if len(metadata_rows) != expected_examples:
        raise ValueError(
            f"{split} metadata row count mismatch: {len(metadata_rows)} != {expected_examples}"
        )

    caption_names = validate_unique_names(caption_rows, f"{split} captions")
    metadata_names = validate_unique_names(metadata_rows, f"{split} metadata")
    caption_set = set(caption_names)
    metadata_set = set(metadata_names)
    if caption_set != metadata_set:
        raise ValueError(
            f"{split} caption/metadata filename mismatch: "
            f"caption_only={sorted(caption_set - metadata_set)[:20]}, "
            f"metadata_only={sorted(metadata_set - caption_set)[:20]}"
        )

    captions_by_name: dict[str, list[str]] = {}
    duplicate_caption_audio = 0
    for row in caption_rows:
        captions = [row[f"caption_{index}"].strip() for index in range(1, 6)]
        if any(not caption for caption in captions):
            raise ValueError(f"empty {split} caption for {row['file_name']}")
        duplicate_caption_audio += len(set(captions)) < len(captions)
        captions_by_name[row["file_name"]] = captions

    metadata_by_name = {row["file_name"]: row for row in metadata_rows}
    whitespace_names = sorted(name for name in caption_names if name != name.strip())
    valid_sound_ids = {
        row["sound_id"].strip()
        for row in metadata_rows
        if row["sound_id"].strip().casefold() not in INVALID_SOUND_IDS
    }
    summary = {
        "caption_rows": len(caption_rows),
        "metadata_rows": len(metadata_rows),
        "unique_file_names": len(caption_set),
        "caption_metadata_exact_file_set_match": True,
        "captions_per_audio": 5,
        "caption_audio_pairs_if_public_loader_is_used": len(caption_rows) * 5,
        "audio_with_duplicate_caption_text": duplicate_caption_audio,
        "valid_sound_ids": len(valid_sound_ids),
        "file_names_with_leading_or_trailing_whitespace": whitespace_names,
        "file_name_policy": "preserve exact official CSV value; do not strip or rename",
        "metadata_encoding": metadata_encoding_report(metadata_csv),
    }
    return captions_by_name, metadata_by_name, summary


def validate_audio_split(
    split: str,
    audio_root: Path,
    captions_by_name: dict[str, list[str]],
    metadata_by_name: dict[str, dict[str, str]],
    include_in_training: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audio_by_name = discover_audio(audio_root)
    expected_names = set(captions_by_name)
    audio_names = set(audio_by_name)
    if audio_names != expected_names:
        raise ValueError(
            f"{split} audio/caption mismatch: audio_count={len(audio_names)}, "
            f"audio_only={sorted(audio_names - expected_names)[:20]}, "
            f"caption_only={sorted(expected_names - audio_names)[:20]}"
        )

    rows: list[dict[str, Any]] = []
    durations: list[float] = []
    sample_rates: Counter[int] = Counter()
    channel_counts: Counter[int] = Counter()
    format_counts: Counter[str] = Counter()
    subtype_counts: Counter[str] = Counter()
    total_audio_bytes = 0
    for file_name in sorted(expected_names):
        path = audio_by_name[file_name]
        decoded = decode_audio(path)
        duration = decoded["duration_seconds"]
        durations.append(duration)
        sample_rates[decoded["sample_rate"]] += 1
        channel_counts[decoded["channels"]] += 1
        format_counts[decoded["format"]] += 1
        subtype_counts[decoded["subtype"]] += 1
        total_audio_bytes += path.stat().st_size
        metadata = metadata_by_name[file_name]
        rows.append(
            {
                "sample_id": file_name,
                "audio_path": str(path),
                "audio_relpath": path.relative_to(audio_root).as_posix(),
                "captions": captions_by_name[file_name],
                "split": split,
                "source_metadata": {
                    key: metadata[key] for key in METADATA_COLUMNS if key != "file_name"
                },
                "duration_seconds": duration,
                "sample_rate": decoded["sample_rate"],
                "channels": decoded["channels"],
                "frames": decoded["frames"],
                "format": decoded["format"],
                "subtype": decoded["subtype"],
                "file_exists": True,
                "decode_ok": decoded["decode_ok"],
                "all_samples_finite": decoded["all_samples_finite"],
                "include_in_training": include_in_training,
                "filter_reason": (
                    "included_official_development_split"
                    if include_in_training
                    else "official_validation_split_not_selected_by_public_launcher"
                ),
                "leakage_blocklist_status": "NOT_APPLIED_CLOTHO_STAGE_AUDIT",
            }
        )

    summary = {
        "audio_files": len(rows),
        "decoded_files": len(rows),
        "total_audio_bytes": total_audio_bytes,
        "total_duration_seconds": math.fsum(durations),
        "duration_min_seconds": min(durations),
        "duration_max_seconds": max(durations),
        "duration_mean_seconds": math.fsum(durations) / len(durations),
        "sample_rate_counts": dict(sorted(sample_rates.items())),
        "channel_counts": dict(sorted(channel_counts.items())),
        "format_counts": dict(sorted(format_counts.items())),
        "subtype_counts": dict(sorted(subtype_counts.items())),
    }
    return rows, summary


def valid_sound_id(value: str) -> bool:
    return value.strip().casefold() not in INVALID_SOUND_IDS


def cross_split_audit(
    development_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    development_by_name = {row["sample_id"]: row for row in development_rows}
    validation_by_name = {row["sample_id"]: row for row in validation_rows}
    filename_overlap = sorted(set(development_by_name) & set(validation_by_name))

    development_by_sound: dict[str, list[dict[str, Any]]] = defaultdict(list)
    validation_by_sound: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in development_rows:
        sound_id = row["source_metadata"]["sound_id"].strip()
        if valid_sound_id(sound_id):
            development_by_sound[sound_id].append(row)
    for row in validation_rows:
        sound_id = row["source_metadata"]["sound_id"].strip()
        if valid_sound_id(sound_id):
            validation_by_sound[sound_id].append(row)
    sound_id_overlap = sorted(set(development_by_sound) & set(validation_by_sound))

    hash_cache: dict[str, str] = {}

    def audio_sha(row: dict[str, Any]) -> str:
        path = row["audio_path"]
        if path not in hash_cache:
            hash_cache[path] = hash_file(Path(path), "sha256")
        return hash_cache[path]

    filename_details = []
    for name in filename_overlap:
        development = development_by_name[name]
        validation = validation_by_name[name]
        development_sha = audio_sha(development)
        validation_sha = audio_sha(validation)
        filename_details.append(
            {
                "file_name": name,
                "development_sound_id": development["source_metadata"]["sound_id"],
                "validation_sound_id": validation["source_metadata"]["sound_id"],
                "development_audio_sha256": development_sha,
                "validation_audio_sha256": validation_sha,
                "exact_audio_bytes_match": development_sha == validation_sha,
                "caption_lists_match": development["captions"] == validation["captions"],
            }
        )

    sound_id_details = []
    for sound_id in sound_id_overlap:
        development = development_by_sound[sound_id]
        validation = validation_by_sound[sound_id]
        development_hashes = [audio_sha(row) for row in development]
        validation_hashes = [audio_sha(row) for row in validation]
        sound_id_details.append(
            {
                "sound_id": sound_id,
                "development_file_names": [row["sample_id"] for row in development],
                "validation_file_names": [row["sample_id"] for row in validation],
                "development_audio_sha256": development_hashes,
                "validation_audio_sha256": validation_hashes,
                "any_exact_audio_bytes_match": bool(
                    set(development_hashes) & set(validation_hashes)
                ),
            }
        )

    return {
        "file_name_overlap_count": len(filename_overlap),
        "file_name_overlaps": filename_details,
        "valid_sound_id_overlap_count": len(sound_id_overlap),
        "valid_sound_id_overlaps": sound_id_details,
        "invalid_sound_id_values_excluded": sorted(INVALID_SOUND_IDS),
        "policy": (
            "[CODE] report official-release cross-split overlaps without silently "
            "deleting samples; the paper does not publish a Clotho split blocklist"
        ),
    }


def main() -> int:
    args = parse_args()
    development_root = args.development_root.resolve()
    validation_root = args.validation_root.resolve()
    development_captions = args.development_captions.resolve()
    validation_captions = args.validation_captions.resolve()
    development_metadata = args.development_metadata.resolve()
    validation_metadata = args.validation_metadata.resolve()
    development_manifest = args.development_manifest_output.resolve()
    validation_manifest = args.validation_manifest_output.resolve()
    statistics_output = args.statistics_output.resolve()

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output("status", "--short"),
        "dataset": "Clotho",
        "version": "2.1",
        "splits": ["development", "validation"],
        "expected_development": args.expected_development,
        "expected_validation": args.expected_validation,
        "paths": {
            "development_root": str(development_root),
            "validation_root": str(validation_root),
            "development_captions": str(development_captions),
            "validation_captions": str(validation_captions),
            "development_metadata": str(development_metadata),
            "validation_metadata": str(validation_metadata),
            "development_manifest": str(development_manifest),
            "validation_manifest": str(validation_manifest),
        },
        "source_checksums": {
            "development_captions_md5": hash_file(development_captions, "md5"),
            "validation_captions_md5": hash_file(validation_captions, "md5"),
            "development_metadata_md5": hash_file(development_metadata, "md5"),
            "validation_metadata_md5": hash_file(validation_metadata, "md5"),
        },
    }
    write_json(statistics_output, report)

    try:
        development_captions_by_name, development_metadata_by_name, development_text = (
            validate_split_text(
                "development",
                development_captions,
                development_metadata,
                args.expected_development,
            )
        )
        validation_captions_by_name, validation_metadata_by_name, validation_text = (
            validate_split_text(
                "validation",
                validation_captions,
                validation_metadata,
                args.expected_validation,
            )
        )
        development_rows, development_audio = validate_audio_split(
            "development",
            development_root,
            development_captions_by_name,
            development_metadata_by_name,
            include_in_training=True,
        )
        validation_rows, validation_audio = validate_audio_split(
            "validation",
            validation_root,
            validation_captions_by_name,
            validation_metadata_by_name,
            include_in_training=False,
        )

        development_content = serialize_jsonl(development_rows)
        validation_content = serialize_jsonl(validation_rows)
        development_write_status = write_reproducible(
            development_manifest, development_content
        )
        validation_write_status = write_reproducible(validation_manifest, validation_content)
        cross_split = cross_split_audit(development_rows, validation_rows)
        warnings = []
        if cross_split["file_name_overlap_count"]:
            warnings.append("official development/validation file_name overlap detected")
        if cross_split["valid_sound_id_overlap_count"]:
            warnings.append("official development/validation FreeSound sound_id overlap detected")

        report.update(
            {
                "status": "complete",
                "finished_at": utc_now(),
                "development": {**development_text, **development_audio},
                "validation": {**validation_text, **validation_audio},
                "cross_split_audit": cross_split,
                "warnings": warnings,
                "manifests": {
                    "development": {
                        "write_status": development_write_status,
                        "rows": len(development_rows),
                        "size_bytes": len(development_content),
                        "sha256": hashlib.sha256(development_content).hexdigest(),
                    },
                    "validation": {
                        "write_status": validation_write_status,
                        "rows": len(validation_rows),
                        "size_bytes": len(validation_content),
                        "sha256": hashlib.sha256(validation_content).hexdigest(),
                    },
                },
                "split_role_sources": {
                    "development": (
                        "[PAPER][CODE] paper reports 3,839 additional Clotho clips; "
                        "the public launcher trains on development"
                    ),
                    "paper_early_stopping_split": (
                        "[MISSING] paper says validation R@10 but does not identify "
                        "the Clotho split"
                    ),
                    "public_launcher_early_stopping_split": (
                        "[CODE] public launcher passes Clotho evaluation, not the "
                        "official validation split, as --val-csv/--val-audio-dir"
                    ),
                    "official_validation": (
                        "[CODE] validated and preserved; not silently substituted for "
                        "the public launcher's evaluation split"
                    ),
                },
            }
        )
        write_json(statistics_output, report)
        return 0
    except Exception as exc:  # noqa: BLE001 - preserve complete audit evidence
        report["status"] = "failed"
        report["finished_at"] = utc_now()
        report["error"] = repr(exc)
        write_json(statistics_output, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
