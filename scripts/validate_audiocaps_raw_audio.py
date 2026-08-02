#!/usr/bin/env python3
"""Bind the author-provided AudioCaps audio archive to the pinned test manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "d87bc8f58c1abe625b4914614461bccb1987e5267b9da211e63925d62695c90a"
)
EXPECTED_ARCHIVE_SHA256 = (
    "27d6edbd623bff6e00bdda53520dd0b77c3b58e39224bc4d75e58a6f8321dcc1"
)
EXPECTED_ARCHIVE_FILES = 101715
EXPECTED_ARCHIVE_BYTES = 74648530543
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--extraction-report", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--expected-test-items", type=int, default=975)
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl_idempotent(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"existing output manifest differs: {path}")
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return digest


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("source manifest contains a non-object row")
    return rows


def validate_extraction_report(path: Path, audio_root: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "complete",
        "output_directory": str(audio_root),
        "stripped_wrapper_name": "audiocaps_raw_audio",
        "stripped_wrapper_levels": 1,
        "file_count": EXPECTED_ARCHIVE_FILES,
        "extracted_bytes": EXPECTED_ARCHIVE_BYTES,
        "overwrite_or_delete_performed": False,
    }
    for field, value in expected.items():
        observed = report.get(field)
        if field == "output_directory":
            observed = str(Path(str(observed)).resolve())
            value = str(audio_root.resolve())
        if observed != value:
            raise ValueError(
                f"AudioCaps extraction report {field} mismatch: {observed!r} != {value!r}"
            )
    if report.get("archive", {}).get("sha256") != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("AudioCaps archive SHA256 mismatch in extraction report")
    return report


def index_audio_files(root: Path) -> tuple[dict[str, list[Path]], dict[str, Any]]:
    by_stem: dict[str, list[Path]] = defaultdict(list)
    suffix_counts: Counter[str] = Counter()
    total_files = 0
    total_bytes = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        total_files += 1
        total_bytes += path.stat().st_size
        suffix = path.suffix.lower()
        suffix_counts[suffix or "<none>"] += 1
        if suffix in AUDIO_SUFFIXES:
            by_stem[path.stem].append(path.resolve())
    return dict(by_stem), {
        "total_files": total_files,
        "total_bytes": total_bytes,
        "audio_files": sum(len(paths) for paths in by_stem.values()),
        "unique_audio_stems": len(by_stem),
        "suffix_counts": dict(sorted(suffix_counts.items())),
    }


def resolve_audio_path(
    row: dict[str, Any],
    by_stem: dict[str, list[Path]],
) -> tuple[Path, list[str]]:
    expected_basename = str(row.get("expected_audio_basename", ""))
    expected_stem = Path(expected_basename).stem
    if not expected_stem:
        raise ValueError(f"empty expected audio basename for {row.get('sample_id')}")
    candidates = sorted(by_stem.get(expected_stem, []), key=lambda value: str(value))
    if not candidates:
        raise FileNotFoundError(expected_basename)
    if len(candidates) == 1:
        return candidates[0], []
    identities = [(path, path.stat().st_size, sha256_file(path)) for path in candidates]
    if len({(size, digest) for _, size, digest in identities}) != 1:
        raise ValueError(
            f"ambiguous non-identical AudioCaps files for {expected_basename}: "
            f"{[str(path) for path in candidates]}"
        )
    preferred = next(
        (path for path in candidates if path.name == expected_basename),
        candidates[0],
    )
    return preferred, [str(path) for path in candidates if path != preferred]


def decode_audio(path: Path) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    frames_read = 0
    finite = True
    with sf.SoundFile(path) as handle:
        sample_rate = int(handle.samplerate)
        channels = int(handle.channels)
        format_name = str(handle.format)
        subtype = str(handle.subtype)
        expected_frames = int(handle.frames)
        for block in handle.blocks(blocksize=262144, dtype="float32", always_2d=True):
            frames_read += int(block.shape[0])
            finite = finite and bool(np.isfinite(block).all())
    if sample_rate <= 0 or channels <= 0 or frames_read <= 0:
        raise ValueError(f"invalid decoded audio properties: {path}")
    if frames_read != expected_frames or not finite:
        raise ValueError(f"AudioCaps full-decode gate failed: {path}")
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": frames_read,
        "duration_seconds": frames_read / sample_rate,
        "format": format_name,
        "subtype": subtype,
        "finite_samples": finite,
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    audio_root = args.audio_root.resolve()
    source_manifest = args.source_manifest.resolve()
    extraction_report = args.extraction_report.resolve()
    output_manifest = args.output_manifest.resolve()
    output_report = args.output_report.resolve()
    if not audio_root.is_dir():
        raise FileNotFoundError(audio_root)
    if (audio_root / audio_root.name).exists():
        raise ValueError("nested audiocaps_raw_audio directory is not allowed")
    if sha256_file(source_manifest) != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise ValueError("AudioCaps source test manifest SHA256 mismatch")
    extraction = validate_extraction_report(extraction_report, audio_root)
    rows = read_jsonl(source_manifest)
    if len(rows) != args.expected_test_items:
        raise ValueError(f"AudioCaps test row mismatch: {len(rows)}")
    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    if sample_ids != sorted(sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("AudioCaps test sample IDs must be sorted and unique")
    if any(len(row.get("captions", [])) != 5 for row in rows):
        raise ValueError("each AudioCaps test item must contain five captions")

    print("STAGE_START=recursive_audio_index", flush=True)
    by_stem, inventory = index_audio_files(audio_root)
    print(
        f"STAGE_END=recursive_audio_index files={inventory['total_files']} "
        f"audio_files={inventory['audio_files']}",
        flush=True,
    )
    if inventory["total_files"] != EXPECTED_ARCHIVE_FILES:
        raise ValueError("extracted AudioCaps file count differs from extraction report")
    if inventory["total_bytes"] != EXPECTED_ARCHIVE_BYTES:
        raise ValueError("extracted AudioCaps byte count differs from extraction report")

    output_rows: list[dict[str, Any]] = []
    selected_paths: set[Path] = set()
    duplicate_aliases = 0
    sample_rates: Counter[int] = Counter()
    channels: Counter[int] = Counter()
    durations: list[float] = []
    for index, row in enumerate(rows, start=1):
        audio_path, aliases = resolve_audio_path(row, by_stem)
        if audio_path in selected_paths:
            raise ValueError(f"AudioCaps test rows resolve to the same file: {audio_path}")
        selected_paths.add(audio_path)
        decoded = decode_audio(audio_path)
        sample_rates[decoded["sample_rate"]] += 1
        channels[decoded["channels"]] += 1
        durations.append(decoded["duration_seconds"])
        duplicate_aliases += len(aliases)
        output_rows.append(
            {
                **row,
                "audio_path": str(audio_path),
                "audio_size_bytes": audio_path.stat().st_size,
                "audio_sha256": sha256_file(audio_path),
                "file_exists": True,
                "decode_ok": True,
                "decode": decoded,
                "identical_duplicate_aliases": aliases,
                "filter_reason": None,
            }
        )
        if index % 50 == 0 or index == len(rows):
            print(f"PROGRESS decoded={index}/{len(rows)}", flush=True)

    manifest_sha256 = write_jsonl_idempotent(output_manifest, output_rows)
    report = {
        "schema_version": 1,
        "status": "complete",
        "started_at": args.started_at,
        "finished_at": now(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
        ).strip(),
        "git_status_short": subprocess.check_output(
            ["git", "status", "--short"], cwd=REPOSITORY_ROOT, text=True
        ).strip(),
        "dataset": "AudioCaps v2 test",
        "candidate_count": len(output_rows),
        "caption_count": sum(len(row["captions"]) for row in output_rows),
        "source_manifest": {
            "path": str(source_manifest),
            "sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        },
        "extraction_report": extraction,
        "inventory": inventory,
        "selected_audio": {
            "count": len(selected_paths),
            "identical_duplicate_alias_count": duplicate_aliases,
            "sample_rate_counts": {str(k): v for k, v in sorted(sample_rates.items())},
            "channel_counts": {str(k): v for k, v in sorted(channels.items())},
            "minimum_duration_seconds": min(durations),
            "maximum_duration_seconds": max(durations),
            "all_finite_and_fully_decoded": True,
        },
        "output_manifest": {
            "path": str(output_manifest),
            "rows": len(output_rows),
            "size_bytes": output_manifest.stat().st_size,
            "sha256": manifest_sha256,
        },
    }
    if not all(math.isfinite(value) and value > 0 for value in durations):
        raise ValueError("AudioCaps duration gate failed")
    write_json(output_report, report)
    return report


def main() -> int:
    args = parse_args()
    args.started_at = now()
    try:
        report = validate(args)
    except BaseException as exc:
        write_json(
            args.output_report,
            {
                "schema_version": 1,
                "status": "failed",
                "started_at": args.started_at,
                "finished_at": now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print("AUDIOCAPS_AUDIO_VALIDATION_STATUS=complete")
    print(f"CANDIDATE_COUNT={report['candidate_count']}")
    print(f"CAPTION_COUNT={report['caption_count']}")
    print(f"OUTPUT_MANIFEST={report['output_manifest']['path']}")
    print(f"OUTPUT_MANIFEST_SHA256={report['output_manifest']['sha256']}")
    print(f"OUTPUT_REPORT={args.output_report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
