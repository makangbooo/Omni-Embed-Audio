#!/usr/bin/env python3
"""Audit public MECAT caption fields before fixing a Table 2/3 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CAPTION_FIELDS = ("long", "short", "speech", "music", "sound", "environment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-examples", type=int, default=848)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_captions(value: Any, *, field: str, sample_id: str) -> list[str]:
    if value is None:
        raw_values: list[Any] = []
    elif isinstance(value, str):
        raw_values = [value]
    elif isinstance(value, list):
        raw_values = value
    else:
        raise TypeError(
            f"unsupported {field} value for {sample_id}: {type(value).__name__}"
        )

    captions: list[str] = []
    for index, item in enumerate(raw_values):
        if not isinstance(item, str):
            raise TypeError(
                f"non-string {field}[{index}] value for {sample_id}: "
                f"{type(item).__name__}"
            )
        caption = item.strip()
        if caption and caption.casefold() != "none":
            captions.append(caption)
    return captions


def protocol_summary(counts: list[int], expected_examples: int) -> dict[str, Any]:
    histogram = Counter(counts)
    return {
        "sample_count": len(counts),
        "samples_with_at_least_one_caption": sum(value >= 1 for value in counts),
        "samples_with_at_least_two_captions": sum(value >= 2 for value in counts),
        "minimum_captions_per_sample": min(counts),
        "maximum_captions_per_sample": max(counts),
        "total_captions": sum(counts),
        "multiplicity_histogram": {
            str(key): histogram[key] for key in sorted(histogram)
        },
        "complete_t2a_candidate": all(value >= 1 for value in counts),
        "complete_t2t_self_exclusion_candidate": all(value >= 2 for value in counts),
        "expected_examples": expected_examples,
    }


def audit_manifest(
    manifest: Path,
    expected_sha256: str,
    expected_examples: int,
) -> dict[str, Any]:
    manifest = manifest.resolve()
    observed_sha256 = sha256_file(manifest)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"MECAT manifest SHA256 mismatch: {observed_sha256} != {expected_sha256}"
        )
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != expected_examples:
        raise ValueError(
            f"MECAT manifest row mismatch: {len(rows)} != {expected_examples}"
        )

    sample_ids = [str(row.get("sample_id", "")).strip() for row in rows]
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError("MECAT manifest contains an empty sample_id")
    if len(set(sample_ids)) != expected_examples:
        raise ValueError("MECAT manifest sample IDs are not unique")
    if sample_ids != sorted(sample_ids):
        raise ValueError("MECAT manifest is not in canonical sample_id order")

    per_field: dict[str, list[list[str]]] = {field: [] for field in CAPTION_FIELDS}
    for sample_id, row in zip(sample_ids, rows, strict=True):
        caption_fields = row.get("caption_fields")
        if not isinstance(caption_fields, dict):
            raise TypeError(f"caption_fields is not an object for {sample_id}")
        missing = sorted(set(CAPTION_FIELDS) - set(caption_fields))
        if missing:
            raise ValueError(f"caption fields missing for {sample_id}: {missing}")
        for field in CAPTION_FIELDS:
            per_field[field].append(
                normalize_captions(
                    caption_fields[field], field=field, sample_id=sample_id
                )
            )

    field_statistics: dict[str, Any] = {}
    for field in CAPTION_FIELDS:
        groups = per_field[field]
        flattened = [caption for group in groups for caption in group]
        field_statistics[field] = {
            **protocol_summary([len(group) for group in groups], expected_examples),
            "unique_caption_strings": len(set(flattened)),
            "duplicate_caption_occurrences": len(flattened) - len(set(flattened)),
        }

    short_first_counts = [1 if group else 0 for group in per_field["short"]]
    short_all_counts = [len(group) for group in per_field["short"]]
    all_fields_counts = [
        sum(len(per_field[field][index]) for field in CAPTION_FIELDS)
        for index in range(expected_examples)
    ]
    return {
        "schema_version": 1,
        "status": "complete",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "MECAT-Caption 00A/test public release",
        "public_candidate_count": expected_examples,
        "paper_candidate_count": 847,
        "manifest": {
            "path": str(manifest),
            "size_bytes": manifest.stat().st_size,
            "sha256": observed_sha256,
        },
        "field_statistics": field_statistics,
        "candidate_protocols": {
            "short_first": {
                "source": "CODE: uiq_toolkit.load_examples(dataset='mecat')",
                **protocol_summary(short_first_counts, expected_examples),
            },
            "short_all": {
                "source": "CODE: mine_hard_negatives_laion.load_mecat_captions",
                **protocol_summary(short_all_counts, expected_examples),
            },
            "all_six_fields_flat": {
                "source": "INFERRED sensitivity only",
                **protocol_summary(all_fields_counts, expected_examples),
            },
        },
        "claim_boundary": {
            "strict_paper_reproduction": False,
            "paper_caption_protocol": "[MISSING]",
            "paper_excluded_candidate_id": "[MISSING]",
            "next_decision": (
                "Fix a controlled public-848 T2A/T2T protocol only after reviewing "
                "the observed caption multiplicities."
            ),
        },
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    report = audit_manifest(
        args.manifest,
        args.manifest_sha256,
        args.expected_examples,
    )
    write_json(args.output, report)
    print("MECAT_RETRIEVAL_CAPTION_AUDIT_STATUS=complete")
    print(f"PUBLIC_CANDIDATES={report['public_candidate_count']}")
    for protocol_id, protocol in report["candidate_protocols"].items():
        print(
            f"PROTOCOL={protocol_id} "
            f"TOTAL_CAPTIONS={protocol['total_captions']} "
            f"MIN_PER_SAMPLE={protocol['minimum_captions_per_sample']} "
            f"MAX_PER_SAMPLE={protocol['maximum_captions_per_sample']} "
            f"T2A_COMPLETE={str(protocol['complete_t2a_candidate']).lower()} "
            f"T2T_SELF_EXCLUSION_COMPLETE="
            f"{str(protocol['complete_t2t_self_exclusion_candidate']).lower()}"
        )
    print(f"OUTPUT={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
