#!/usr/bin/env python3
"""Audit pinned AudioCaps 2.0 CSVs and build metadata-only manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "d004db3ea1b01cf4fd0347dd8d27db90cadc8809"
CSV_COLUMNS = ["audiocap_id", "youtube_id", "start_time", "caption"]
POSITIVE_UIQ_TYPES = ("question", "imperative", "paraphrase", "tagging")
EXPECTED_FILES = {
    "README.md": {
        "size_bytes": 753,
        "sha256": "ae2fcbcf7e4f93964dfaf02c094998b2cfc199a16f4eb0350f65f170ad01a906",
    },
    "train.csv": {
        "size_bytes": 6311901,
        "sha256": "25659eee0ff887972b6a8a74008dfffe582343930e89db71519d66e9dd014f6e",
        "lf_count": 91257,
        "bare_cr_count": 3,
        "parsed_rows": 91257,
        "valid_rows": 91254,
        "audio_groups": 91254,
    },
    "val.csv": {
        "size_bytes": 169508,
        "sha256": "dfdd0f83c70fb818fbb86dcba42754abfb97f7774d811fff91a1c89d84d868c5",
        "lf_count": 2476,
        "bare_cr_count": 0,
        "parsed_rows": 2475,
        "valid_rows": 2475,
        "audio_groups": 495,
    },
    "test.csv": {
        "size_bytes": 397498,
        "sha256": "365c8a8a71c9070a8d5dba5dd44f3a781f906632a49cb82b38bebaaa8f3c5ccb",
        "lf_count": 4876,
        "bare_cr_count": 0,
        "parsed_rows": 4875,
        "valid_rows": 4875,
        "audio_groups": 975,
    },
}
EXPECTED_TRAIN_ORPHANS = [
    "a dog is whimpering",
    "bang",
    "speech in the background",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--uiq-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def normalize_start_time(value: str) -> str:
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"invalid AudioCaps start_time: {value!r}") from exc
    if not number.is_finite() or number < 0:
        raise ValueError(f"invalid AudioCaps start_time: {value!r}")
    result = format(number, "f")
    if "." in result:
        result = result.rstrip("0").rstrip(".")
    return result


def order_preserving_casefold_unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def parse_source_csv(path: Path, split: str) -> dict[str, Any]:
    payload = path.read_bytes()
    text = payload.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != CSV_COLUMNS:
        raise ValueError(f"unexpected AudioCaps header in {path}: {reader.fieldnames}")

    valid: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    previous_valid: dict[str, Any] | None = None
    previous_line_number = 1
    multiline_rows: list[dict[str, Any]] = []
    parsed_rows = 0
    for ordinal, raw in enumerate(reader, start=1):
        parsed_rows += 1
        line_advance = reader.line_num - previous_line_number
        previous_line_number = reader.line_num
        is_valid = (
            set(raw) == set(CSV_COLUMNS)
            and all(isinstance(raw.get(column), str) for column in CSV_COLUMNS)
            and all(raw[column].strip() for column in CSV_COLUMNS)
        )
        if not is_valid:
            populated = [value.strip() for value in raw.values() if isinstance(value, str) and value.strip()]
            fragment = populated[0] if len(populated) == 1 else None
            anomaly = {
                "parsed_ordinal": ordinal,
                "source_line_number": reader.line_num,
                "raw": raw,
                "orphan_fragment": fragment,
                "attached_to_previous_audiocap_id": (
                    previous_valid["audiocap_id"] if previous_valid is not None else None
                ),
            }
            malformed.append(anomaly)
            if fragment is not None and previous_valid is not None:
                previous_valid["orphan_fragments"].append(fragment)
                previous_valid["caption_integrity_status"] = (
                    "TRUNCATED_BY_UNQUOTED_BARE_CR_IN_SOURCE"
                )
            continue

        record: dict[str, Any] = {
            column: raw[column].strip() for column in CSV_COLUMNS
        }
        record.update(
            {
                "split": split,
                "parsed_ordinal": ordinal,
                "source_line_number": reader.line_num,
                "orphan_fragments": [],
                "caption_integrity_status": "SOURCE_RECORD_COMPLETE",
            }
        )
        record["normalized_audio_id"] = (
            f"{record['youtube_id']}_{normalize_start_time(record['start_time'])}"
        )
        valid.append(record)
        previous_valid = record
        if line_advance > 1:
            multiline_rows.append(
                {
                    "audiocap_id": record["audiocap_id"],
                    "source_line_number": reader.line_num,
                    "physical_line_span": line_advance,
                }
            )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in valid:
        groups[record["normalized_audio_id"]].append(record)

    duplicate_caption_ids = [
        audio_id
        for audio_id, records in groups.items()
        if len(
            order_preserving_casefold_unique(
                [record["caption"] for record in records]
            )
        )
        != len(records)
    ]
    return {
        "split": split,
        "path": str(path.resolve()),
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "lf_count": payload.count(b"\n"),
        "cr_count": payload.count(b"\r"),
        "crlf_count": payload.count(b"\r\n"),
        "bare_cr_count": payload.count(b"\r") - payload.count(b"\r\n"),
        "parsed_rows": parsed_rows,
        "valid_rows": len(valid),
        "malformed_rows": malformed,
        "multiline_rows": multiline_rows,
        "records": valid,
        "groups": dict(groups),
        "audio_groups": len(groups),
        "duplicate_caption_audio_ids": sorted(duplicate_caption_ids),
    }


def validate_expected_source(source: dict[str, Any], expected: dict[str, Any]) -> None:
    for field in (
        "size_bytes",
        "sha256",
        "lf_count",
        "bare_cr_count",
        "parsed_rows",
        "valid_rows",
        "audio_groups",
    ):
        if source[field] != expected[field]:
            raise ValueError(
                f"AudioCaps {source['split']} {field} mismatch: "
                f"{source[field]} != {expected[field]}"
            )
    audiocap_ids = [record["audiocap_id"] for record in source["records"]]
    if len(set(audiocap_ids)) != len(audiocap_ids):
        raise ValueError(f"duplicate audiocap_id in {source['split']}")
    expected_captions_per_audio = 1 if source["split"] == "train" else 5
    distribution = Counter(len(records) for records in source["groups"].values())
    if distribution != Counter({expected_captions_per_audio: source["audio_groups"]}):
        raise ValueError(
            f"unexpected captions-per-audio distribution in {source['split']}: "
            f"{dict(distribution)}"
        )


def build_manifest_rows(
    source: dict[str, Any],
    *,
    repair_bare_cr: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for audio_id in sorted(source["groups"]):
        records = source["groups"][audio_id]
        captions: list[str] = []
        caption_status: list[str] = []
        orphan_fragments: list[str] = []
        for record in records:
            caption = record["caption"]
            fragments = record["orphan_fragments"]
            if repair_bare_cr and fragments:
                caption = " ".join([caption, *fragments])
            captions.append(caption)
            caption_status.append(record["caption_integrity_status"])
            orphan_fragments.extend(fragments)
        first = records[0]
        rows.append(
            {
                "sample_id": audio_id,
                "youtube_id": first["youtube_id"],
                "start_time": first["start_time"],
                "audiocap_ids": [record["audiocap_id"] for record in records],
                "captions": captions,
                "split": source["split"],
                "expected_audio_basename": (
                    f"{first['youtube_id']}_{first['start_time']}.wav"
                ),
                "audio_path": None,
                "duration_seconds": 10.0,
                "sample_rate": None,
                "file_exists": None,
                "decode_ok": None,
                "include_in_training": source["split"] == "train",
                "filter_reason": "metadata_only_audio_not_checked",
                "leakage_blocklist_status": "NOT_CHECKED",
                "caption_integrity_status": caption_status,
                "orphan_fragments": orphan_fragments,
                "bare_cr_policy": (
                    "[INFERRED] append orphan fragment with one space"
                    if repair_bare_cr
                    else "[CODE] preserve public OEA csv.DictReader result"
                ),
            }
        )
    return rows


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
    test_source: dict[str, Any],
    *,
    expected_positive_rows: int = 975,
    expected_negative_rows: int = 630,
    expected_negative_unique_ids: int = 255,
) -> dict[str, Any]:
    test_ids = set(test_source["groups"])
    captions = {
        audio_id: [record["caption"] for record in records]
        for audio_id, records in test_source["groups"].items()
    }
    report: dict[str, Any] = {}
    for query_type in POSITIVE_UIQ_TYPES:
        path = uiq_root / f"audiocaps_test_{query_type}_queries.jsonl"
        rows = read_jsonl(path)
        ids = [str(row.get("audio_id", "")) for row in rows]
        if (
            len(rows) != expected_positive_rows
            or len(set(ids)) != expected_positive_rows
            or set(ids) != test_ids
        ):
            raise ValueError(
                f"AudioCaps positive UIQ ID mismatch for {query_type}: "
                f"rows={len(rows)}, unique={len(set(ids))}, "
                f"uiq_only={sorted(set(ids) - test_ids)[:20]}, "
                f"csv_only={sorted(test_ids - set(ids))[:20]}"
            )
        invalid = [
            index
            for index, row in enumerate(rows, start=1)
            if row.get("dataset") != "audiocaps"
            or row.get("dataset_slug") != "audiocaps_test"
            or row.get("query_type") != query_type
            or not isinstance(row.get("generated_query"), str)
            or not row["generated_query"].strip()
            or row.get("original_captions")
            != order_preserving_casefold_unique(captions[row["audio_id"]])
        ]
        if invalid:
            raise ValueError(
                f"AudioCaps positive UIQ schema/caption mismatch for {query_type}: "
                f"{invalid[:20]}"
            )
        report[query_type] = {
            "rows": len(rows),
            "unique_audio_ids": len(set(ids)),
            "exact_csv_id_set_match": True,
            "original_captions_match_order_preserving_casefold_deduplication": True,
        }

    negative_path = uiq_root / "audiocaps_test_negative_queries.jsonl"
    negative_rows = read_jsonl(negative_path)
    ids = [str(row.get("audio_id", "")) for row in negative_rows]
    if (
        len(negative_rows) != expected_negative_rows
        or len(set(ids)) != expected_negative_unique_ids
        or not set(ids) <= test_ids
    ):
        raise ValueError(
            f"AudioCaps negative UIQ count/ID mismatch: rows={len(negative_rows)}, "
            f"unique={len(set(ids))}, unknown={sorted(set(ids) - test_ids)[:20]}"
        )
    invalid = [
        index
        for index, row in enumerate(negative_rows, start=1)
        if row.get("dataset") != "audiocaps"
        or row.get("dataset_slug") != "audiocaps_test"
        or row.get("query_type") != "negative"
        or not isinstance(row.get("negative_query"), str)
        or not row["negative_query"].strip()
        or row.get("original_captions") != captions[row["audio_id"]]
    ]
    if invalid:
        raise ValueError(f"AudioCaps negative UIQ schema/caption mismatch: {invalid[:20]}")
    report["negative"] = {
        "rows": len(negative_rows),
        "unique_audio_ids": len(set(ids)),
        "all_audio_ids_in_test_csv": True,
        "original_captions_exact_match": True,
        "hard_negative_audio_id_source": "[MISSING] not present in released JSONL",
    }
    return report


def source_summary(source: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in source.items()
        if key not in {"records", "groups"}
    }


def main() -> int:
    args = parse_args()
    metadata_root = args.metadata_root.resolve()
    uiq_root = args.uiq_root.resolve()
    manifest_root = args.manifest_root.resolve()
    statistics_output = args.statistics_output.resolve()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output("status", "--short"),
        "dataset": "AudioCaps",
        "version": "2.0",
        "source_commit": SOURCE_COMMIT,
        "metadata_root": str(metadata_root),
        "uiq_root": str(uiq_root),
        "manifest_root": str(manifest_root),
    }
    write_json(statistics_output, report)

    try:
        for filename, expected in EXPECTED_FILES.items():
            path = metadata_root / filename
            if path.stat().st_size != expected["size_bytes"]:
                raise ValueError(f"AudioCaps source size mismatch: {path}")
            if sha256_file(path) != expected["sha256"]:
                raise ValueError(f"AudioCaps source SHA256 mismatch: {path}")

        sources = {
            split: parse_source_csv(metadata_root / filename, split)
            for split, filename in (
                ("train", "train.csv"),
                ("validation", "val.csv"),
                ("test", "test.csv"),
            )
        }
        for split, filename in (
            ("train", "train.csv"),
            ("validation", "val.csv"),
            ("test", "test.csv"),
        ):
            validate_expected_source(sources[split], EXPECTED_FILES[filename])

        orphan_fragments = [
            row["orphan_fragment"] for row in sources["train"]["malformed_rows"]
        ]
        if orphan_fragments != EXPECTED_TRAIN_ORPHANS:
            raise ValueError(
                f"unexpected AudioCaps train orphan fragments: {orphan_fragments}"
            )
        if len(sources["train"]["multiline_rows"]) != 2:
            raise ValueError("expected exactly two quoted multiline train captions")
        if len(sources["test"]["duplicate_caption_audio_ids"]) != 97:
            raise ValueError(
                "expected exactly 97 test clips with duplicate caption strings"
            )

        manifests = {
            "train_public_loader": build_manifest_rows(
                sources["train"], repair_bare_cr=False
            ),
            "train_repaired_bare_cr": build_manifest_rows(
                sources["train"], repair_bare_cr=True
            ),
            "validation": build_manifest_rows(
                sources["validation"], repair_bare_cr=False
            ),
            "test": build_manifest_rows(sources["test"], repair_bare_cr=False),
        }
        manifest_paths: dict[str, Path] = {}
        for name, rows in manifests.items():
            path = manifest_root / f"audiocaps_v2_{name}_manifest.jsonl"
            write_jsonl(path, rows)
            manifest_paths[name] = path

        uiq_report = validate_uiq(uiq_root, sources["test"])
        report.update(
            {
                "status": "complete",
                "finished_at": utc_now(),
                "sources": {
                    split: source_summary(source) for split, source in sources.items()
                },
                "uiq": uiq_report,
                "manifests": {
                    name: {
                        "path": str(path),
                        "rows": len(manifests[name]),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                    for name, path in manifest_paths.items()
                },
                "count_reconciliation": {
                    "paper_train_samples": 91256,
                    "official_readme_train_count": 91256,
                    "source_lf_count_minus_header": 91256,
                    "public_oea_loader_valid_train_records": 91254,
                    "bare_cr_repaired_train_records": 91254,
                    "unresolved_difference_from_paper": 2,
                    "status": (
                        "[MISSING] paper does not publish a manifest or loader that "
                        "produces 91,256 valid training records"
                    ),
                },
                "source_tags": {
                    "metadata": "[CODE] official AudioCaps 2.0 fixed commit",
                    "public_loader_manifest": (
                        "[CODE] exact csv.DictReader behavior used by public OEA trainer"
                    ),
                    "repaired_manifest": (
                        "[INFERRED] append three orphan bare-CR fragments; record count "
                        "remains 91,254"
                    ),
                    "paper_count": (
                        "[PAPER][MISSING] 91,256 is source LF count minus header, but "
                        "not the number of valid CSV records"
                    ),
                },
            }
        )
        write_json(statistics_output, report)
        return 0
    except Exception as exc:  # noqa: BLE001 - retain complete audit evidence
        report["status"] = "failed"
        report["finished_at"] = utc_now()
        report["error"] = repr(exc)
        write_json(statistics_output, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
