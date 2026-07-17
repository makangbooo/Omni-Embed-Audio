#!/usr/bin/env python3
"""Audit source-video provenance overlap between MECAT and WavCaps AudioSet_SL."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.validate_wavcaps_metadata import (
        REPOSITORY_ROOT,
        SOURCE_SPECS,
        WAVCAPS_REVISION,
        normalize_wavcaps_audioset_id,
        sha256_file,
        validate_pinned_file,
        write_json,
        write_jsonl_without_overwriting_mismatch,
    )
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from validate_wavcaps_metadata import (
        REPOSITORY_ROOT,
        SOURCE_SPECS,
        WAVCAPS_REVISION,
        normalize_wavcaps_audioset_id,
        sha256_file,
        validate_pinned_file,
        write_json,
        write_jsonl_without_overwriting_mismatch,
    )


MECAT_REVISION = "be4a24c3f7309d74208e08a7cce49e72cb7a5834"
YOUTUBE_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mecat-manifest", type=Path, required=True)
    parser.add_argument("--wavcaps-root", type=Path, required=True)
    parser.add_argument("--overlap-output", type=Path, required=True)
    parser.add_argument("--statistics-output", type=Path, required=True)
    parser.add_argument("--expected-mecat-examples", type=int, default=848)
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL line: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row: {path}:{line_number}")
            rows.append(value)
    return rows


def mecat_youtube_id(sample_id: str) -> str:
    """Extract only the published 11-character source ID, not segment times."""

    if len(sample_id) < 13 or sample_id[11] != "_":
        raise ValueError(f"unexpected MECAT sample_id structure: {sample_id!r}")
    youtube_id = sample_id[:11]
    segment_suffix = sample_id[12:]
    if not YOUTUBE_ID.fullmatch(youtube_id) or not segment_suffix:
        raise ValueError(f"unexpected MECAT sample_id structure: {sample_id!r}")
    return youtube_id


def validate_mecat_manifest(
    path: Path, expected_examples: int
) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != expected_examples:
        raise ValueError(
            f"MECAT manifest count mismatch: {len(rows)} != {expected_examples}"
        )
    sample_ids: list[str] = []
    for index, row in enumerate(rows, start=1):
        sample_id = row.get("sample_id")
        audio_relpath = row.get("audio_relpath")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"invalid MECAT sample_id at row {index}")
        if not isinstance(audio_relpath, str) or Path(audio_relpath).stem != sample_id:
            raise ValueError(f"MECAT audio_relpath/sample_id mismatch at row {index}")
        if row.get("split") != "test" or row.get("include_in_training") is not False:
            raise ValueError(f"unexpected MECAT split/training status at row {index}")
        if row.get("file_exists") is not True or row.get("decode_ok") is not True:
            raise ValueError(f"unvalidated MECAT audio row at index {index}")
        mecat_youtube_id(sample_id)
        sample_ids.append(sample_id)
    duplicates = sorted(
        sample_id for sample_id, count in Counter(sample_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate MECAT sample IDs: {duplicates[:20]}")
    return rows


def load_wavcaps_audioset(root: Path) -> list[dict[str, Any]]:
    spec = SOURCE_SPECS["AudioSet_SL"]
    path = root / spec["path"]
    validate_pinned_file(path, spec)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError(f"unexpected WavCaps AudioSet_SL schema: {path}")
    rows = payload["data"]
    if len(rows) != spec["rows"]:
        raise ValueError(
            f"WavCaps AudioSet_SL count mismatch: {len(rows)} != {spec['rows']}"
        )
    source_ids: list[str] = []
    youtube_ids: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not all(
            key in row for key in ("id", "caption", "duration")
        ):
            raise ValueError(f"invalid WavCaps AudioSet_SL row: {index}")
        source_id = str(row["id"])
        caption = str(row["caption"])
        duration = float(row["duration"])
        if not source_id or not caption.strip() or not math.isfinite(duration) or duration < 0:
            raise ValueError(f"invalid WavCaps AudioSet_SL row: {index}")
        source_ids.append(source_id)
        youtube_ids.append(normalize_wavcaps_audioset_id(source_id))
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("duplicate WavCaps AudioSet_SL source IDs")
    if len(youtube_ids) != len(set(youtube_ids)):
        raise ValueError("duplicate normalized WavCaps AudioSet_SL YouTube IDs")
    return rows


def derive_source_video_overlaps(
    mecat_rows: list[dict[str, Any]], wavcaps_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    mecat_by_video: dict[str, list[str]] = defaultdict(list)
    for row in mecat_rows:
        sample_id = row["sample_id"]
        mecat_by_video[mecat_youtube_id(sample_id)].append(sample_id)

    wavcaps_by_video: dict[str, dict[str, Any]] = {}
    for row in wavcaps_rows:
        video_id = normalize_wavcaps_audioset_id(str(row["id"]))
        if video_id in wavcaps_by_video:
            raise ValueError(f"duplicate normalized WavCaps video ID: {video_id}")
        wavcaps_by_video[video_id] = row

    overlap_ids = sorted(set(mecat_by_video) & set(wavcaps_by_video))
    overlap_rows = []
    for video_id in overlap_ids:
        wavcaps = wavcaps_by_video[video_id]
        overlap_rows.append(
            {
                "youtube_id": video_id,
                "mecat_sample_ids": sorted(mecat_by_video[video_id]),
                "wavcaps_source": "AudioSet_SL",
                "wavcaps_id": str(wavcaps["id"]),
                "wavcaps_duration_seconds": float(wavcaps["duration"]),
                "wavcaps_caption": str(wavcaps["caption"]),
                "match_protocol": (
                    "[CODE][INFERRED] exact 11-character YouTube source-video ID; "
                    "this does not prove temporal or audio-content overlap"
                ),
                "blocklist_status": "NOT_APPLIED_SOURCE_VIDEO_CANDIDATE_ONLY",
            }
        )
    return {
        "mecat_examples": len(mecat_rows),
        "mecat_unique_source_videos": len(mecat_by_video),
        "mecat_videos_with_multiple_segments": sum(
            len(sample_ids) > 1 for sample_ids in mecat_by_video.values()
        ),
        "wavcaps_audioset_rows": len(wavcaps_rows),
        "source_video_overlap_ids": overlap_ids,
        "source_video_overlap_count": len(overlap_ids),
        "mecat_overlap_sample_count": sum(
            len(mecat_by_video[video_id]) for video_id in overlap_ids
        ),
        "overlap_rows": overlap_rows,
    }


def main() -> int:
    args = parse_args()
    mecat_manifest = args.mecat_manifest.resolve()
    wavcaps_root = args.wavcaps_root.resolve()
    overlap_output = args.overlap_output.resolve()
    statistics_output = args.statistics_output.resolve()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output("status", "--short"),
        "mecat_revision": MECAT_REVISION,
        "wavcaps_revision": WAVCAPS_REVISION,
        "mecat_manifest": str(mecat_manifest),
        "mecat_manifest_sha256": sha256_file(mecat_manifest),
        "wavcaps_root": str(wavcaps_root),
        "overlap_output": str(overlap_output),
    }
    write_json(statistics_output, report)
    try:
        mecat_rows = validate_mecat_manifest(
            mecat_manifest, args.expected_mecat_examples
        )
        wavcaps_rows = load_wavcaps_audioset(wavcaps_root)
        overlap = derive_source_video_overlaps(mecat_rows, wavcaps_rows)
        artifact = write_jsonl_without_overwriting_mismatch(
            overlap_output, overlap["overlap_rows"]
        )
        report.update(
            {
                "status": "complete",
                "finished_at": utc_now(),
                "mecat_examples": overlap["mecat_examples"],
                "mecat_unique_source_videos": overlap[
                    "mecat_unique_source_videos"
                ],
                "mecat_videos_with_multiple_segments": overlap[
                    "mecat_videos_with_multiple_segments"
                ],
                "wavcaps_audioset_rows": overlap["wavcaps_audioset_rows"],
                "source_video_overlap_count": overlap[
                    "source_video_overlap_count"
                ],
                "mecat_overlap_sample_count": overlap[
                    "mecat_overlap_sample_count"
                ],
                "source_video_overlap_ids": overlap["source_video_overlap_ids"],
                "overlap_artifact": artifact,
                "interpretation": {
                    "proven": (
                        "[CODE] exact source-video ID candidates between released "
                        "MECAT 00A/test and pinned WavCaps AudioSet_SL metadata"
                    ),
                    "not_proven": (
                        "[INFERRED] a shared video ID does not prove that time "
                        "segments or audio content overlap"
                    ),
                    "paper_protocol": (
                        "[MISSING] paper audio/embedding comparison implementation, "
                        "model, similarity threshold, and reviewed candidate list"
                    ),
                    "policy": (
                        "report candidates without deleting samples or changing "
                        "evaluation/training sets"
                    ),
                },
            }
        )
        write_json(statistics_output, report)
        return 0
    except Exception as exc:  # noqa: BLE001 - preserve full audit failure evidence
        report["status"] = "failed"
        report["finished_at"] = utc_now()
        report["error"] = repr(exc)
        write_json(statistics_output, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
