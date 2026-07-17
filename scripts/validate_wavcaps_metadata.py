#!/usr/bin/env python3
"""Audit pinned WavCaps metadata, paper count reconstruction, and leakage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WAVCAPS_REVISION = "0930ec11ded28fa0eaa910fde2f6fc3538acbeac"
AUDIOCAPS_COMMIT = "d004db3ea1b01cf4fd0347dd8d27db90cadc8809"
CLOTHO_METADATA_MD5 = "13946f054d4e1bf48079813aac61bf77"
AUDIOCAPS_TEST_SHA256 = (
    "365c8a8a71c9070a8d5dba5dd44f3a781f906632a49cb82b38bebaaa8f3c5ccb"
)

SOURCE_SPECS = {
    "AudioSet_SL": {
        "path": "json_files/AudioSet_SL/as_final.json",
        "rows": 108317,
        "size_bytes": 15341826,
        "sha256": "c26e4c7f1fd00f346b54b7808c0d90c3465a2d98469023a8ad2735b9c9d8d4f3",
    },
    "BBC_Sound_Effects": {
        "path": "json_files/BBC_Sound_Effects/bbc_final.json",
        "rows": 31201,
        "size_bytes": 11670806,
        "sha256": "151c37b00aa0823a697fbc8b24b124bef2c2654a5165ed4245addf7a0388f657",
    },
    "FreeSound": {
        "path": "json_files/FreeSound/fsd_final.json",
        "rows": 262300,
        "size_bytes": 147300685,
        "sha256": "4499ada1a7a76666695d880feb298f557efb3aa8af3b57fd9b713c75319ff53f",
    },
    "SoundBible": {
        "path": "json_files/SoundBible/sb_final.json",
        "rows": 1232,
        "size_bytes": 469675,
        "sha256": "c64329d13c58322d932a7e802c3f4b95d1fcd2ba0e52083cecb35a41b00f6a1c",
    },
}

OFFICIAL_BLACKLIST_SPECS = {
    "blacklist_exclude_all_ac.json": {
        "path": "json_files/blacklist/blacklist_exclude_all_ac.json",
        "size_bytes": 1525390,
        "sha256": "ab53786c5d0a0fe174013395cae46dfcec8676ba71c8539605691290b63b4803",
        "counts": {"AudioSet": 50725, "FreeSound": 5929},
    },
    "blacklist_exclude_test_ac.json": {
        "path": "json_files/blacklist/blacklist_exclude_test_ac.json",
        "size_bytes": 77724,
        "sha256": "54a84d4e24f5c7eda582ce75d0e34eae51a22df9f1a946db9e43501a6fb386a8",
        "counts": {"AudioSet": 1451, "FreeSound": 2090},
    },
    "blacklist_exclude_ub8k_esc50_vggsound.json": {
        "path": "json_files/blacklist/blacklist_exclude_ub8k_esc50_vggsound.json",
        "size_bytes": 473374,
        "sha256": "64e278412489767ce8578e651985c310d8409ed78855bfcc78f9e398ca838c50",
        "counts": {"AudioSet": 15510, "FreeSound": 2219},
    },
}

EXPECTED_FULL_AUDIT = {
    "all_rows": 403050,
    "duration_lt_31": 275624,
    "duration_le_31": 275691,
    "positive_duration_lt_31": 275618,
    "positive_duration_le_31": 275685,
    "non_positive_duration": 6,
    "duration_equal_31": 67,
    "audiocaps_unique_test_ids": 975,
    "audiocaps_overlap": 173,
    "clotho_evaluation_rows": 1045,
    "clotho_filename_overlap": 638,
    "clotho_wavcaps_candidate_ids": 1017,
    "clotho_sound_id_filename_confirmed": 611,
    "clotho_ambiguous_filenames": 64,
    "eligible_audiocaps_blocked": 173,
    "eligible_clotho_candidate_blocked": 383,
    "conservative_leakage_filtered": 275062,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavcaps-root", type=Path, required=True)
    parser.add_argument("--audiocaps-test-csv", type=Path, required=True)
    parser.add_argument("--clotho-metadata-csv", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--blocklist-root", type=Path, required=True)
    parser.add_argument("--statistics-output", type=Path, required=True)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - source release checksum, not security
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def jsonl_payload(row: dict[str, Any]) -> bytes:
    return (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def write_jsonl_without_overwriting_mismatch(
    path: Path, rows: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise RuntimeError(f"stale manifest staging file exists: {temporary}")
    digest = hashlib.sha256()
    count = 0
    size_bytes = 0
    try:
        with temporary.open("xb") as handle:
            for row in rows:
                payload = jsonl_payload(row)
                handle.write(payload)
                digest.update(payload)
                size_bytes += len(payload)
                count += 1
        new_sha256 = digest.hexdigest()
        if path.exists():
            existing_sha256 = sha256_file(path)
            if existing_sha256 != new_sha256:
                raise RuntimeError(
                    f"refusing to overwrite different generated artifact: {path}; "
                    f"existing={existing_sha256}, generated={new_sha256}"
                )
            temporary.unlink()
            status = "verified_existing"
        else:
            temporary.replace(path)
            status = "created"
        return {
            "path": str(path),
            "rows": count,
            "size_bytes": size_bytes,
            "sha256": new_sha256,
            "status": status,
        }
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def validate_pinned_file(path: Path, spec: dict[str, Any]) -> None:
    if path.stat().st_size != spec["size_bytes"]:
        raise ValueError(
            f"pinned size mismatch for {path}: {path.stat().st_size} != "
            f"{spec['size_bytes']}"
        )
    digest = sha256_file(path)
    if digest != spec["sha256"]:
        raise ValueError(
            f"pinned SHA256 mismatch for {path}: {digest} != {spec['sha256']}"
        )


def load_wavcaps_sources(root: Path) -> dict[str, list[dict[str, Any]]]:
    sources: dict[str, list[dict[str, Any]]] = {}
    for source, spec in SOURCE_SPECS.items():
        path = root / spec["path"]
        validate_pinned_file(path, spec)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError(f"unexpected WavCaps JSON schema: {path}")
        rows = payload["data"]
        if len(rows) != spec["rows"]:
            raise ValueError(
                f"WavCaps {source} count mismatch: {len(rows)} != {spec['rows']}"
            )
        ids: list[str] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"non-object WavCaps row: {source}:{index}")
            if not all(key in row for key in ("id", "caption", "duration")):
                raise ValueError(f"missing required WavCaps field: {source}:{index}")
            source_id = str(row["id"])
            caption = str(row["caption"])
            duration = float(row["duration"])
            if not source_id or not caption.strip() or duration < 0:
                raise ValueError(f"invalid WavCaps row: {source}:{index}")
            ids.append(source_id)
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate WavCaps IDs within {source}")
        sources[source] = rows
    return sources


def load_official_blacklists(root: Path) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    for name, spec in OFFICIAL_BLACKLIST_SPECS.items():
        path = root / spec["path"]
        validate_pinned_file(path, spec)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if set(payload) != {"AudioSet", "FreeSound"}:
            raise ValueError(f"unexpected official WavCaps blacklist schema: {path}")
        counts = {key: len(payload[key]) for key in sorted(payload)}
        if counts != spec["counts"]:
            raise ValueError(
                f"official blacklist count mismatch for {name}: "
                f"{counts} != {spec['counts']}"
            )
        result[name] = payload
    return result


def duration_policy_counts(
    sources: dict[str, list[dict[str, Any]]]
) -> dict[str, int]:
    durations = [
        float(row["duration"])
        for rows in sources.values()
        for row in rows
    ]
    return {
        "all_rows": len(durations),
        "duration_lt_31": sum(value < 31 for value in durations),
        "duration_le_31": sum(value <= 31 for value in durations),
        "positive_duration_lt_31": sum(0 < value < 31 for value in durations),
        "positive_duration_le_31": sum(0 < value <= 31 for value in durations),
        "non_positive_duration": sum(value <= 0 for value in durations),
        "duration_equal_31": sum(value == 31 for value in durations),
    }


def normalize_wavcaps_audioset_id(value: str) -> str:
    stem = Path(value).stem
    if len(stem) != 12 or not stem.startswith("Y"):
        raise ValueError(f"unexpected WavCaps AudioSet_SL ID: {value!r}")
    return stem[1:]


def load_audiocaps_test(path: Path) -> list[dict[str, str]]:
    if sha256_file(path) != AUDIOCAPS_TEST_SHA256:
        raise ValueError(f"AudioCaps test SHA256 mismatch: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != [
            "audiocap_id",
            "youtube_id",
            "start_time",
            "caption",
        ]:
            raise ValueError(f"unexpected AudioCaps test header: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != 4875 or any(not row["youtube_id"] for row in rows):
        raise ValueError(f"unexpected AudioCaps test rows: {len(rows)}")
    counts = Counter(row["youtube_id"] for row in rows)
    if len(counts) != 975 or set(counts.values()) != {5}:
        raise ValueError(f"unexpected AudioCaps test grouping: {counts}")
    return rows


def load_clotho_metadata(path: Path) -> list[dict[str, str]]:
    if md5_file(path) != CLOTHO_METADATA_MD5:
        raise ValueError(f"Clotho metadata MD5 mismatch: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    required = {"file_name", "sound_id"}
    if not required <= set(reader.fieldnames or []):
        raise ValueError(f"unexpected Clotho metadata header: {reader.fieldnames}")
    if len(rows) != 1045:
        raise ValueError(f"unexpected Clotho evaluation rows: {len(rows)}")
    normalized = [row["file_name"].casefold() for row in rows]
    if any(not value for value in normalized) or len(normalized) != len(set(normalized)):
        raise ValueError("Clotho evaluation filenames are empty or non-unique")
    return rows


def derive_overlaps(
    sources: dict[str, list[dict[str, Any]]],
    audiocaps_rows: list[dict[str, str]],
    clotho_rows: list[dict[str, str]],
) -> dict[str, Any]:
    audioset_by_youtube: dict[str, dict[str, Any]] = {}
    for row in sources["AudioSet_SL"]:
        youtube_id = normalize_wavcaps_audioset_id(str(row["id"]))
        if youtube_id in audioset_by_youtube:
            raise ValueError(f"duplicate normalized AudioSet ID: {youtube_id}")
        audioset_by_youtube[youtube_id] = row
    audiocaps_ids = sorted({row["youtube_id"] for row in audiocaps_rows})
    audiocaps_overlap_ids = sorted(set(audiocaps_ids) & set(audioset_by_youtube))
    audiocaps_blocked_ids = {
        str(audioset_by_youtube[youtube_id]["id"])
        for youtube_id in audiocaps_overlap_ids
    }

    freesound_by_filename: dict[str, list[dict[str, Any]]] = defaultdict(list)
    freesound_by_id: dict[str, dict[str, Any]] = {}
    for row in sources["FreeSound"]:
        key = str(row["file_name"]).casefold()
        freesound_by_filename[key].append(row)
        freesound_by_id[str(row["id"])] = row

    clotho_matches: list[dict[str, Any]] = []
    conservative_candidate_ids: set[str] = set()
    confirmed_ids: set[str] = set()
    for row in clotho_rows:
        key = row["file_name"].casefold()
        candidates = sorted(
            freesound_by_filename.get(key, []), key=lambda item: str(item["id"])
        )
        if not candidates:
            continue
        candidate_ids = [str(item["id"]) for item in candidates]
        conservative_candidate_ids.update(candidate_ids)
        sound_id = str(row["sound_id"])
        confirmed = sound_id in set(candidate_ids)
        if confirmed:
            confirmed_ids.add(sound_id)
        clotho_matches.append(
            {
                "clotho_file_name": row["file_name"],
                "normalized_filename": key,
                "clotho_sound_id": sound_id,
                "wavcaps_candidate_count": len(candidates),
                "wavcaps_candidate_ids": candidate_ids,
                "sound_id_and_filename_confirmed": confirmed,
            }
        )

    candidate_rows = [
        freesound_by_id[source_id]
        for source_id in sorted(conservative_candidate_ids)
    ]
    confirmed_rows = [
        freesound_by_id[source_id] for source_id in sorted(confirmed_ids)
    ]
    return {
        "audiocaps_unique_test_ids": len(audiocaps_ids),
        "audiocaps_overlap_ids": audiocaps_overlap_ids,
        "audiocaps_blocked_ids": audiocaps_blocked_ids,
        "clotho_evaluation_rows": len(clotho_rows),
        "clotho_matches": sorted(
            clotho_matches, key=lambda item: item["normalized_filename"]
        ),
        "clotho_candidate_rows": candidate_rows,
        "clotho_candidate_ids": conservative_candidate_ids,
        "clotho_confirmed_rows": confirmed_rows,
        "clotho_confirmed_ids": confirmed_ids,
        "clotho_ambiguous_filenames": sum(
            item["wavcaps_candidate_count"] > 1 for item in clotho_matches
        ),
    }


def manifest_rows(
    sources: dict[str, list[dict[str, Any]]],
    *,
    audiocaps_blocked_ids: set[str],
    clotho_candidate_ids: set[str],
) -> Iterable[dict[str, Any]]:
    for source in SOURCE_SPECS:
        for row in sorted(sources[source], key=lambda item: str(item["id"])):
            source_id = str(row["id"])
            duration = float(row["duration"])
            duration_eligible = 0 < duration < 31
            blocked_audiocaps = (
                source == "AudioSet_SL" and source_id in audiocaps_blocked_ids
            )
            blocked_clotho = (
                source == "FreeSound" and source_id in clotho_candidate_ids
            )
            reasons: list[str] = []
            if duration <= 0:
                reasons.append("NON_POSITIVE_DURATION")
            elif duration >= 31:
                reasons.append("DURATION_NOT_STRICTLY_BELOW_31_SECONDS")
            if blocked_audiocaps:
                reasons.append("AUDIOCAPS_TEST_OVERLAP_EXACT_YOUTUBE_ID")
            if blocked_clotho:
                reasons.append("CLOTHO_EVALUATION_FILENAME_MATCH_CONSERVATIVE")
            leakage_status: list[str] = []
            if blocked_audiocaps:
                leakage_status.append("BLOCKED_AUDIOCAPS_TEST")
            if blocked_clotho:
                leakage_status.append("BLOCKED_CLOTHO_EVALUATION_CONSERVATIVE")
            if not leakage_status:
                leakage_status.append("CLEAR_UNDER_IMPLEMENTED_EXACT_MATCHES")
            yield {
                "sample_id": f"{source}:{Path(source_id).stem}",
                "source": source,
                "source_id": source_id,
                "expected_audio_stem": Path(source_id).stem,
                "audio_path": None,
                "caption": str(row["caption"]),
                "split": "training",
                "duration_seconds": duration,
                "sample_rate": None,
                "file_exists": None,
                "decode_ok": None,
                "duration_eligible": duration_eligible,
                "include_in_training": (
                    duration_eligible and not blocked_audiocaps and not blocked_clotho
                ),
                "filter_reasons": reasons,
                "leakage_blocklist_status": leakage_status,
                "duration_policy": (
                    "[INFERRED] 0 < duration < 31 exactly reconstructs the "
                    "paper's 275,618 count; paper text says <=31"
                ),
                "audio_validation_status": "METADATA_ONLY_NOT_CHECKED",
            }


def summarize_manifest_counts(
    sources: dict[str, list[dict[str, Any]]], overlaps: dict[str, Any]
) -> dict[str, int]:
    eligible_audiocaps = sum(
        0 < float(row["duration"]) < 31
        for row in sources["AudioSet_SL"]
        if str(row["id"]) in overlaps["audiocaps_blocked_ids"]
    )
    eligible_clotho = sum(
        0 < float(row["duration"]) < 31
        for row in sources["FreeSound"]
        if str(row["id"]) in overlaps["clotho_candidate_ids"]
    )
    eligible_total = duration_policy_counts(sources)["positive_duration_lt_31"]
    return {
        "eligible_audiocaps_blocked": eligible_audiocaps,
        "eligible_clotho_candidate_blocked": eligible_clotho,
        "conservative_leakage_filtered": (
            eligible_total - eligible_audiocaps - eligible_clotho
        ),
    }


def validate_expected_full_audit(actual: dict[str, int]) -> None:
    for key, expected in EXPECTED_FULL_AUDIT.items():
        if actual[key] != expected:
            raise ValueError(f"full WavCaps audit mismatch for {key}: {actual[key]} != {expected}")


def main() -> int:
    args = parse_args()
    wavcaps_root = args.wavcaps_root.resolve()
    audiocaps_test_csv = args.audiocaps_test_csv.resolve()
    clotho_metadata_csv = args.clotho_metadata_csv.resolve()
    manifest_root = args.manifest_root.resolve()
    blocklist_root = args.blocklist_root.resolve()
    statistics_output = args.statistics_output.resolve()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output("status", "--short"),
        "dataset": "WavCaps",
        "wavcaps_revision": WAVCAPS_REVISION,
        "audiocaps_commit": AUDIOCAPS_COMMIT,
        "wavcaps_root": str(wavcaps_root),
        "audiocaps_test_csv": str(audiocaps_test_csv),
        "clotho_metadata_csv": str(clotho_metadata_csv),
        "manifest_root": str(manifest_root),
        "blocklist_root": str(blocklist_root),
    }
    write_json(statistics_output, report)

    try:
        sources = load_wavcaps_sources(wavcaps_root)
        official_blacklists = load_official_blacklists(wavcaps_root)
        audiocaps_rows = load_audiocaps_test(audiocaps_test_csv)
        clotho_rows = load_clotho_metadata(clotho_metadata_csv)
        durations = duration_policy_counts(sources)
        overlaps = derive_overlaps(sources, audiocaps_rows, clotho_rows)
        manifest_counts = summarize_manifest_counts(sources, overlaps)
        full_actual = {
            **durations,
            "audiocaps_unique_test_ids": overlaps["audiocaps_unique_test_ids"],
            "audiocaps_overlap": len(overlaps["audiocaps_overlap_ids"]),
            "clotho_evaluation_rows": overlaps["clotho_evaluation_rows"],
            "clotho_filename_overlap": len(overlaps["clotho_matches"]),
            "clotho_wavcaps_candidate_ids": len(overlaps["clotho_candidate_ids"]),
            "clotho_sound_id_filename_confirmed": len(overlaps["clotho_confirmed_ids"]),
            "clotho_ambiguous_filenames": overlaps["clotho_ambiguous_filenames"],
            **manifest_counts,
        }
        validate_expected_full_audit(full_actual)

        audiocaps_blocklist_rows = [
            {
                "source": "AudioSet_SL",
                "wavcaps_id": source_id,
                "youtube_id": normalize_wavcaps_audioset_id(source_id),
                "match_protocol": (
                    "[PAPER] strip WavCaps leading Y and trailing .wav, then "
                    "exact-match the AudioCaps test youtube_id"
                ),
            }
            for source_id in sorted(overlaps["audiocaps_blocked_ids"])
        ]
        clotho_filename_rows = [
            {
                **item,
                "match_protocol": (
                    "[PAPER] case-insensitive exact file_name match; candidate "
                    "multiplicity is retained instead of silently choosing an ID"
                ),
            }
            for item in overlaps["clotho_matches"]
        ]
        clotho_candidate_rows = [
            {
                "source": "FreeSound",
                "wavcaps_id": str(row["id"]),
                "file_name": str(row["file_name"]),
                "duration_seconds": float(row["duration"]),
                "blocklist_protocol": (
                    "[INFERRED] conservative removal of every WavCaps row whose "
                    "case-folded filename matches a Clotho evaluation filename"
                ),
            }
            for row in overlaps["clotho_candidate_rows"]
        ]
        clotho_confirmed_rows = [
            {
                "source": "FreeSound",
                "wavcaps_id": str(row["id"]),
                "file_name": str(row["file_name"]),
                "duration_seconds": float(row["duration"]),
                "confirmation_protocol": (
                    "[CODE] Clotho sound_id and case-folded file_name both match "
                    "the same WavCaps FreeSound record"
                ),
            }
            for row in overlaps["clotho_confirmed_rows"]
        ]

        artifacts = {
            "all_metadata_manifest": write_jsonl_without_overwriting_mismatch(
                manifest_root / "wavcaps_all_metadata_audit_manifest.jsonl",
                manifest_rows(
                    sources,
                    audiocaps_blocked_ids=overlaps["audiocaps_blocked_ids"],
                    clotho_candidate_ids=overlaps["clotho_candidate_ids"],
                ),
            ),
            "audiocaps_test_blocklist": write_jsonl_without_overwriting_mismatch(
                blocklist_root / "audiocaps_test_wavcaps_audioset_sl_ids.jsonl",
                audiocaps_blocklist_rows,
            ),
            "clotho_filename_overlap": write_jsonl_without_overwriting_mismatch(
                blocklist_root / "clotho_evaluation_filename_matches.jsonl",
                clotho_filename_rows,
            ),
            "clotho_conservative_candidate_blocklist": (
                write_jsonl_without_overwriting_mismatch(
                    blocklist_root
                    / "clotho_evaluation_wavcaps_freesound_candidate_ids.jsonl",
                    clotho_candidate_rows,
                )
            ),
            "clotho_sound_id_filename_confirmed": (
                write_jsonl_without_overwriting_mismatch(
                    blocklist_root
                    / "clotho_evaluation_sound_id_filename_confirmed.jsonl",
                    clotho_confirmed_rows,
                )
            ),
        }

        official_summary: dict[str, Any] = {}
        for name, payload in official_blacklists.items():
            official_summary[name] = {
                "counts": {key: len(payload[key]) for key in sorted(payload)},
                "derived_audiocaps_ids_present": len(
                    overlaps["audiocaps_blocked_ids"] & set(payload["AudioSet"])
                ),
                "confirmed_clotho_ids_present": len(
                    overlaps["clotho_confirmed_ids"]
                    & {str(value) for value in payload["FreeSound"]}
                ),
                "interpretation": (
                    "[CODE] released WavCaps artifact; its filename does not prove "
                    "that it is the unpublished OEA training blocklist"
                ),
            }

        report.update(
            {
                "status": "complete",
                "finished_at": utc_now(),
                "source_counts": {
                    source: len(rows) for source, rows in sources.items()
                },
                "duration_reconciliation": {
                    **durations,
                    "paper_reported_samples": 275618,
                    "paper_written_predicate": "duration <= 31 seconds",
                    "exact_public_metadata_reconstruction": "0 < duration < 31 seconds",
                    "source_tag": (
                        "[INFERRED] count-equivalent predicate; the paper does not "
                        "publish its exact filtered manifest"
                    ),
                },
                "audiocaps_overlap": {
                    "audiocaps_unique_test_ids": overlaps[
                        "audiocaps_unique_test_ids"
                    ],
                    "wavcaps_audioset_sl_overlaps": len(
                        overlaps["audiocaps_overlap_ids"]
                    ),
                    "eligible_blocked_rows": manifest_counts[
                        "eligible_audiocaps_blocked"
                    ],
                    "paper_reported_overlap": 173,
                    "match_protocol": "[PAPER] normalized exact YouTube ID",
                },
                "clotho_overlap": {
                    "clotho_evaluation_rows": overlaps["clotho_evaluation_rows"],
                    "filename_matched_clotho_rows": len(overlaps["clotho_matches"]),
                    "paper_reported_overlap": 638,
                    "wavcaps_candidate_rows": len(overlaps["clotho_candidate_ids"]),
                    "ambiguous_clotho_filenames": overlaps[
                        "clotho_ambiguous_filenames"
                    ],
                    "sound_id_and_filename_confirmed_rows": len(
                        overlaps["clotho_confirmed_ids"]
                    ),
                    "eligible_conservative_candidate_rows": manifest_counts[
                        "eligible_clotho_candidate_blocked"
                    ],
                    "match_protocol": (
                        "[PAPER] case-insensitive exact filename for the 638 count; "
                        "[INFERRED] all 1,017 candidate WavCaps IDs are blocked in "
                        "the conservative metadata manifest"
                    ),
                },
                "training_count_reconciliation": {
                    "duration_eligible_before_leakage": durations[
                        "positive_duration_lt_31"
                    ],
                    "audiocaps_rows_removed": manifest_counts[
                        "eligible_audiocaps_blocked"
                    ],
                    "clotho_candidate_rows_removed": manifest_counts[
                        "eligible_clotho_candidate_blocked"
                    ],
                    "conservative_metadata_only_training_rows": manifest_counts[
                        "conservative_leakage_filtered"
                    ],
                    "exact_paper_post_blocklist_count": (
                        "[MISSING] not reported and exact OEA blocklist not released"
                    ),
                },
                "official_wavcaps_blacklists": official_summary,
                "artifacts": artifacts,
                "source_tags": {
                    "wavcaps_metadata": "[CODE] pinned official Hugging Face revision",
                    "audiocaps_test": "[CODE] pinned official AudioCaps 2.0 commit",
                    "clotho_metadata": "[INFERRED] repaired official Clotho v2.1 evaluation metadata",
                    "paper_duration_count": "[PAPER] 275,618 and written <=31 seconds",
                    "duration_reconstruction": "[INFERRED] 0 < duration < 31",
                    "exact_oea_training_blocklist": "[MISSING]",
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
