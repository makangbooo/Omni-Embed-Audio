#!/usr/bin/env python3
"""Build and validate the canonical Clotho evaluation audio manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
POSITIVE_UIQ_TYPES = ("question", "imperative", "paraphrase", "tagging")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-root", type=Path, required=True)
    parser.add_argument("--captions-csv", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--uiq-root", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--statistics-output", type=Path, required=True)
    parser.add_argument("--expected-examples", type=int, default=1045)
    parser.add_argument("--expected-negative-rows", type=int, default=542)
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


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_exact(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            raise ValueError(
                f"unexpected CSV columns for {path}: {reader.fieldnames} != "
                f"{expected_columns}"
            )
        return [dict(row) for row in reader]


def unique_names(rows: list[dict[str, str]], label: str) -> set[str]:
    names = [row["file_name"] for row in rows]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {label} file_name values: {duplicates[:20]}")
    if any(not name for name in names):
        raise ValueError(f"empty {label} file_name")
    return set(names)


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


def validate_text_and_uiq(
    captions_csv: Path,
    metadata_csv: Path,
    uiq_root: Path,
    expected_examples: int,
    expected_negative_rows: int,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    caption_rows = read_csv_exact(captions_csv, CAPTION_COLUMNS)
    metadata_rows = read_csv_exact(metadata_csv, METADATA_COLUMNS)
    if len(caption_rows) != expected_examples:
        raise ValueError(
            f"caption row count mismatch: {len(caption_rows)} != {expected_examples}"
        )
    if len(metadata_rows) != expected_examples:
        raise ValueError(
            f"metadata row count mismatch: {len(metadata_rows)} != {expected_examples}"
        )

    caption_names = unique_names(caption_rows, "caption")
    metadata_names = unique_names(metadata_rows, "metadata")
    if caption_names != metadata_names:
        raise ValueError(
            "caption/metadata filename mismatch: "
            f"caption_only={sorted(caption_names - metadata_names)[:20]}, "
            f"metadata_only={sorted(metadata_names - caption_names)[:20]}"
        )

    captions_by_name: dict[str, list[str]] = {}
    for row in caption_rows:
        captions = [row[f"caption_{index}"].strip() for index in range(1, 6)]
        if any(not caption for caption in captions):
            raise ValueError(f"empty caption for {row['file_name']}")
        captions_by_name[row["file_name"]] = captions

    uiq_report: dict[str, Any] = {}
    for query_type in POSITIVE_UIQ_TYPES:
        path = uiq_root / f"clotho_evaluation_{query_type}_queries.jsonl"
        rows = read_jsonl(path)
        ids = [str(row.get("audio_id", "")) for row in rows]
        if len(rows) != expected_examples or len(set(ids)) != expected_examples:
            raise ValueError(
                f"positive UIQ count/uniqueness mismatch for {query_type}: "
                f"rows={len(rows)}, unique={len(set(ids))}"
            )
        if set(ids) != caption_names:
            raise ValueError(
                f"positive UIQ audio_id mismatch for {query_type}: "
                f"uiq_only={sorted(set(ids) - caption_names)[:20]}, "
                f"caption_only={sorted(caption_names - set(ids))[:20]}"
            )
        invalid_schema = [
            index
            for index, row in enumerate(rows, start=1)
            if row.get("dataset") != "clotho"
            or row.get("dataset_slug") != "clotho_evaluation"
            or row.get("query_type") != query_type
            or not isinstance(row.get("generated_query"), str)
            or not row["generated_query"].strip()
        ]
        if invalid_schema:
            raise ValueError(
                f"invalid positive UIQ schema for {query_type}: {invalid_schema[:20]}"
            )
        uiq_report[query_type] = {
            "rows": len(rows),
            "unique_audio_ids": len(set(ids)),
            "exact_audio_id_set_match": True,
        }

    negative_path = uiq_root / "clotho_evaluation_negative_queries.jsonl"
    negative_rows = read_jsonl(negative_path)
    if len(negative_rows) != expected_negative_rows:
        raise ValueError(
            f"negative UIQ row count mismatch: {len(negative_rows)} != "
            f"{expected_negative_rows}"
        )
    negative_ids = [str(row.get("audio_id", "")) for row in negative_rows]
    inferred_ids = [
        value if value.lower().endswith(".wav") else f"{value}.wav"
        for value in negative_ids
    ]
    inferred_unmatched = sorted(set(inferred_ids) - caption_names)
    invalid_negative_schema = [
        index
        for index, row in enumerate(negative_rows, start=1)
        if row.get("dataset") != "clotho"
        or row.get("dataset_slug") != "clotho_evaluation"
        or row.get("query_type") != "negative"
        or not isinstance(row.get("negative_query"), str)
        or not row["negative_query"].strip()
    ]
    if invalid_negative_schema:
        raise ValueError(
            f"invalid negative UIQ schema rows: {invalid_negative_schema[:20]}"
        )
    uiq_report["negative"] = {
        "rows": len(negative_rows),
        "unique_raw_audio_ids": len(set(negative_ids)),
        "raw_exact_matches": sum(value in caption_names for value in negative_ids),
        "inferred_append_wav_matches": sum(
            value in caption_names for value in inferred_ids
        ),
        "inferred_unmatched_ids": inferred_unmatched,
        "mapping_source": (
            "[INFERRED] released negative audio_id values omit the .wav suffix; "
            "the suffix-appended mapping is reported but is not treated as an "
            "officially published target/HN pairing"
        ),
    }
    return captions_by_name, uiq_report


def discover_audio(extract_root: Path) -> dict[str, Path]:
    audio_paths = sorted(
        path for path in extract_root.rglob("*") if path.is_file() and path.suffix.lower() == ".wav"
    )
    by_name: dict[str, Path] = {}
    duplicates: dict[str, list[str]] = {}
    for path in audio_paths:
        if path.name in by_name:
            duplicates.setdefault(path.name, [str(by_name[path.name])]).append(str(path))
        else:
            by_name[path.name] = path
    if duplicates:
        raise ValueError(f"duplicate WAV basenames: {dict(list(duplicates.items())[:20])}")
    return by_name


def decode_audio(path: Path) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    info = sf.info(str(path))
    decoded_frames = 0
    all_finite = True
    for block in sf.blocks(str(path), blocksize=65536, dtype="float32", always_2d=True):
        decoded_frames += int(block.shape[0])
        all_finite = all_finite and bool(np.isfinite(block).all())
    if decoded_frames != info.frames:
        raise ValueError(
            f"decoded frame mismatch for {path}: {decoded_frames} != {info.frames}"
        )
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
    extract_root = args.extract_root.resolve()
    captions_csv = args.captions_csv.resolve()
    metadata_csv = args.metadata_csv.resolve()
    uiq_root = args.uiq_root.resolve()
    manifest_output = args.manifest_output.resolve()
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
        "split": "evaluation",
        "expected_examples": args.expected_examples,
        "extract_root": str(extract_root),
        "captions_csv": str(captions_csv),
        "metadata_csv": str(metadata_csv),
        "uiq_root": str(uiq_root),
        "manifest_output": str(manifest_output),
        "source_checksums": {
            "captions_md5": md5_file(captions_csv),
            "metadata_md5": md5_file(metadata_csv),
        },
    }
    write_json(statistics_output, report)

    try:
        captions_by_name, uiq_report = validate_text_and_uiq(
            captions_csv,
            metadata_csv,
            uiq_root,
            args.expected_examples,
            args.expected_negative_rows,
        )
        audio_by_name = discover_audio(extract_root)
        caption_names = set(captions_by_name)
        audio_names = set(audio_by_name)
        if len(audio_by_name) != args.expected_examples or audio_names != caption_names:
            raise ValueError(
                "audio/caption mismatch: "
                f"audio_count={len(audio_by_name)}, "
                f"audio_only={sorted(audio_names - caption_names)[:20]}, "
                f"caption_only={sorted(caption_names - audio_names)[:20]}"
            )

        manifest_rows: list[dict[str, Any]] = []
        sample_rates: Counter[int] = Counter()
        channel_counts: Counter[int] = Counter()
        durations: list[float] = []
        total_audio_bytes = 0
        for file_name in sorted(caption_names):
            path = audio_by_name[file_name]
            decoded = decode_audio(path)
            durations.append(decoded["duration_seconds"])
            sample_rates[decoded["sample_rate"]] += 1
            channel_counts[decoded["channels"]] += 1
            total_audio_bytes += path.stat().st_size
            manifest_rows.append(
                {
                    "sample_id": file_name,
                    "audio_path": str(path),
                    "audio_relpath": path.relative_to(extract_root).as_posix(),
                    "captions": captions_by_name[file_name],
                    "split": "evaluation",
                    "duration_seconds": decoded["duration_seconds"],
                    "sample_rate": decoded["sample_rate"],
                    "channels": decoded["channels"],
                    "frames": decoded["frames"],
                    "format": decoded["format"],
                    "subtype": decoded["subtype"],
                    "file_exists": True,
                    "decode_ok": decoded["decode_ok"],
                    "all_samples_finite": decoded["all_samples_finite"],
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
                "caption_rows": len(captions_by_name),
                "metadata_rows": args.expected_examples,
                "audio_files": len(audio_by_name),
                "decoded_files": len(manifest_rows),
                "caption_count_per_audio": 5,
                "total_audio_bytes": total_audio_bytes,
                "total_duration_seconds": math.fsum(durations),
                "duration_min_seconds": min(durations),
                "duration_max_seconds": max(durations),
                "duration_mean_seconds": math.fsum(durations) / len(durations),
                "sample_rate_counts": dict(sorted(sample_rates.items())),
                "channel_counts": dict(sorted(channel_counts.items())),
                "uiq": uiq_report,
                "manifest_size_bytes": manifest_output.stat().st_size,
                "manifest_md5": md5_file(manifest_output),
                "source_tags": {
                    "dataset_version": (
                        "[INFERRED] paper says Clotho v2; use repaired official v2.1 release"
                    ),
                    "evaluation_count": "[PAPER][CODE] 1045 clips",
                    "positive_uiq_counts": "[CODE] released UIQ JSONL files",
                    "negative_id_suffix": (
                        "[INFERRED] append .wav only for alignment audit; not an "
                        "official target/HN pairing"
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
