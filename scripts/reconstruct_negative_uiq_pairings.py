#!/usr/bin/env python3
"""Reconstruct negative-UIQ audio pairings from exact released captions."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-jsonl", type=Path, required=True)
    parser.add_argument("--reference-positive-jsonl", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            value = dict(value)
            value["_release_line_number"] = line_number
            rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def required_string(row: Mapping[str, Any], field: str, context: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: {field} must be a non-empty string")
    return value.strip()


def caption_tuple(row: Mapping[str, Any], field: str, context: str) -> tuple[str, ...]:
    value = row.get(field)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context}: {field} must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{context}: {field} contains an empty/non-string caption")
    return tuple(item.strip() for item in value)


def _id_aliases(candidate_id: str) -> tuple[str, ...]:
    value = candidate_id.strip().casefold()
    stem = Path(value).stem
    return tuple(dict.fromkeys((value, stem)))


def reconstruct_pairings(
    negative_rows: Sequence[Mapping[str, Any]],
    positive_rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
) -> dict[str, Any]:
    positive_ids: list[str] = []
    captions_to_ids: dict[tuple[str, ...], list[str]] = defaultdict(list)
    casefold_captions_to_ids: dict[tuple[str, ...], list[str]] = defaultdict(list)
    alias_to_ids: dict[str, list[str]] = defaultdict(list)

    for index, row in enumerate(positive_rows, start=1):
        context = f"positive row {index}"
        candidate_id = required_string(row, "audio_id", context)
        captions = caption_tuple(row, "original_captions", context)
        positive_ids.append(candidate_id)
        captions_to_ids[captions].append(candidate_id)
        casefold_captions_to_ids[
            tuple(caption.casefold() for caption in captions)
        ].append(candidate_id)
        for alias in _id_aliases(candidate_id):
            alias_to_ids[alias].append(candidate_id)

    if len(set(positive_ids)) != len(positive_ids):
        raise ValueError("reference positive UIQ audio IDs are not unique")

    pairings: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    target_methods: Counter[str] = Counter()
    hard_negative_methods: Counter[str] = Counter()

    for logical_index, row in enumerate(negative_rows):
        line_number = int(row.get("_release_line_number", logical_index + 1))
        context = f"negative row {line_number}"
        raw_target_id = required_string(row, "audio_id", context)
        query = required_string(row, "negative_query", context)
        negative_captions = caption_tuple(row, "negative_captions", context)
        query_id = f"{dataset}:negative:{logical_index:06d}"

        if raw_target_id in set(positive_ids):
            target_candidates = [raw_target_id]
            target_method = "exact_audio_id"
        else:
            target_candidates = alias_to_ids.get(raw_target_id.casefold(), [])
            if not target_candidates:
                target_candidates = alias_to_ids.get(Path(raw_target_id).stem.casefold(), [])
            target_method = "unique_casefold_or_stem_audio_id"

        hard_negative_candidates = captions_to_ids.get(negative_captions, [])
        hard_negative_method = "exact_ordered_caption_list"
        casefold_candidates = casefold_captions_to_ids.get(
            tuple(caption.casefold() for caption in negative_captions), []
        )

        reasons: list[str] = []
        if len(target_candidates) != 1:
            reasons.append(f"target_candidate_count={len(target_candidates)}")
        if len(hard_negative_candidates) != 1:
            reasons.append(
                f"exact_hard_negative_candidate_count={len(hard_negative_candidates)}"
            )
        if (
            len(hard_negative_candidates) != 1
            and len(casefold_candidates) == 1
        ):
            reasons.append("casefold_unique_but_exact_ordered_missing")

        if not reasons:
            target_id = target_candidates[0]
            hard_negative_id = hard_negative_candidates[0]
            if target_id == hard_negative_id:
                reasons.append("target_equals_hard_negative")

        if reasons:
            failures.append(
                {
                    "query_id": query_id,
                    "release_line_number": line_number,
                    "raw_target_id": raw_target_id,
                    "reasons": reasons,
                    "target_candidates": sorted(set(target_candidates)),
                    "exact_hard_negative_candidates": sorted(
                        set(hard_negative_candidates)
                    ),
                    "casefold_hard_negative_candidates": sorted(
                        set(casefold_candidates)
                    ),
                    "negative_captions": list(negative_captions),
                }
            )
            continue

        target_methods[target_method] += 1
        hard_negative_methods[hard_negative_method] += 1
        pairings.append(
            {
                "query_id": query_id,
                "target_id": target_id,
                "hard_negative_id": hard_negative_id,
                "release_line_number": line_number,
                "pairing_source": "INFERRED_EXACT_RELEASED_NEGATIVE_CAPTIONS",
                "target_match_method": target_method,
                "hard_negative_match_method": hard_negative_method,
            }
        )
        queries.append(
            {
                "query_id": query_id,
                "text": query,
                "target_id": target_id,
                "release_line_number": line_number,
            }
        )

    return {
        "status": "complete" if not failures else "incomplete",
        "dataset": dataset,
        "negative_row_count": len(negative_rows),
        "candidate_count": len(positive_rows),
        "matched_pairing_count": len(pairings),
        "failed_pairing_count": len(failures),
        "full_coverage": len(pairings) == len(negative_rows),
        "pairing_source": "INFERRED_EXACT_RELEASED_NEGATIVE_CAPTIONS",
        "target_match_methods": dict(sorted(target_methods.items())),
        "hard_negative_match_methods": dict(sorted(hard_negative_methods.items())),
        "pairings": pairings,
        "queries": queries,
        "failures": failures,
    }


def run(
    negative_jsonl: Path,
    reference_positive_jsonl: Path,
    dataset: str,
    output_dir: Path,
) -> dict[str, Any]:
    negative_jsonl = negative_jsonl.resolve()
    reference_positive_jsonl = reference_positive_jsonl.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    negative_rows = read_jsonl(negative_jsonl)
    positive_rows = read_jsonl(reference_positive_jsonl)
    result = reconstruct_pairings(negative_rows, positive_rows, dataset=dataset)

    pairing_path = output_dir / "pairing_metadata.jsonl"
    query_path = output_dir / "query_metadata.jsonl"
    failure_path = output_dir / "pairing_failures.jsonl"
    write_jsonl(pairing_path, result.pop("pairings"))
    write_jsonl(query_path, result.pop("queries"))
    write_jsonl(failure_path, result.pop("failures"))

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    report = {
        "schema_version": 1,
        **result,
        "generated_at": utc_now(),
        "git_commit": git_commit,
        "strict_paper_pairing_reproduction": False,
        "claim_boundary": (
            "The release omits hard-negative audio IDs. This reconstruction "
            "uses only unique exact ordered matches between negative_captions "
            "and the released positive UIQ original_captions."
        ),
        "inputs": {
            "negative_jsonl": file_identity(negative_jsonl),
            "reference_positive_jsonl": file_identity(reference_positive_jsonl),
        },
        "outputs": {
            "pairing_metadata": file_identity(pairing_path),
            "query_metadata": file_identity(query_path),
            "pairing_failures": file_identity(failure_path),
        },
    }
    write_json(output_dir / "reconstruction_report.json", report)
    return report


def main() -> int:
    args = parse_args()
    report = run(
        args.negative_jsonl,
        args.reference_positive_jsonl,
        args.dataset,
        args.output_dir,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
