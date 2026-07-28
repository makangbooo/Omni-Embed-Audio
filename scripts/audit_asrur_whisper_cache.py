#!/usr/bin/env python3
"""Fail-closed content audit for one completed formal Whisper N-best cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from AudioRetrieval.asr_uncertainty_reranking.artifacts import (  # noqa: E402
    load_nbest,
    load_unbounded_qrels,
)
from AudioRetrieval.asr_uncertainty_reranking.data import (  # noqa: E402
    load_text_queries,
    normalized_query_text,
)
from AudioRetrieval.asr_uncertainty_reranking.metrics import (  # noqa: E402
    corpus_wer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nbest", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-hypotheses", type=int, default=4)
    parser.add_argument("--min-unique-top1", type=int, default=8)
    parser.add_argument("--max-mode-fraction", type=float, default=0.95)
    parser.add_argument("--max-corpus-wer", type=float, default=0.95)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def audit_cache(
    *,
    nbest_path: Path,
    queries_path: Path,
    qrels_path: Path,
    condition: str,
    expected_hypotheses: int,
    min_unique_top1: int,
    max_mode_fraction: float,
    max_corpus_wer: float,
) -> dict[str, Any]:
    if expected_hypotheses <= 0:
        raise ValueError("expected_hypotheses must be positive")
    if min_unique_top1 < 2:
        raise ValueError("min_unique_top1 must be at least 2")
    if not 0.0 < max_mode_fraction < 1.0:
        raise ValueError("max_mode_fraction must be between zero and one")
    if not 0.0 < max_corpus_wer <= 1.0:
        raise ValueError("max_corpus_wer must be in (0, 1]")
    if not condition:
        raise ValueError("condition must be non-empty")

    nbest = load_nbest(nbest_path, expected_size=expected_hypotheses)
    queries = load_text_queries(queries_path)
    qrels = load_unbounded_qrels(qrels_path)
    expected_query_ids = set(qrels)
    if set(nbest) != expected_query_ids:
        missing = sorted(expected_query_ids - set(nbest))
        unexpected = sorted(set(nbest) - expected_query_ids)
        raise ValueError(
            "Whisper N-best/query contract differs: "
            f"missing={missing[:20]} unexpected={unexpected[:20]}"
        )
    absent_queries = sorted(expected_query_ids - set(queries))
    if absent_queries:
        raise ValueError(
            f"qrels reference absent query text: {absent_queries[:20]}"
        )

    query_ids = sorted(expected_query_ids)
    references = {query_id: queries[query_id].text for query_id in query_ids}
    top1 = {
        query_id: nbest[query_id].hypotheses[0].text
        for query_id in query_ids
    }
    normalized = {
        query_id: normalized_query_text(top1[query_id])
        for query_id in query_ids
    }
    frequency = Counter(normalized.values())
    mode_text, mode_count = frequency.most_common(1)[0]
    wer = corpus_wer(
        (references[query_id], top1[query_id])
        for query_id in query_ids
    )
    unique_count = len(frequency)
    mode_fraction = mode_count / len(query_ids)
    violations = []
    if unique_count < min_unique_top1:
        violations.append(
            {
                "code": "top1_diversity_below_integrity_floor",
                "observed": unique_count,
                "required_minimum": min_unique_top1,
            }
        )
    if mode_fraction > max_mode_fraction:
        violations.append(
            {
                "code": "single_transcript_mode_exceeds_integrity_ceiling",
                "observed": mode_fraction,
                "required_maximum": max_mode_fraction,
            }
        )
    if float(wer["WER"]) > max_corpus_wer:
        violations.append(
            {
                "code": "corpus_wer_exceeds_integrity_ceiling",
                "observed": float(wer["WER"]),
                "required_maximum": max_corpus_wer,
            }
        )

    return {
        "schema_version": 1,
        "status": "complete" if not violations else "failed",
        "stage": "asrur_whisper_cache_content_integrity_gate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output(
            "status",
            "--short",
            "--untracked-files=all",
        ),
        "condition": condition,
        "query_count": len(query_ids),
        "expected_hypotheses": expected_hypotheses,
        "top1": {
            "unique_normalized_count": unique_count,
            "empty_normalized_count": sum(not value for value in normalized.values()),
            "mode": {"text": mode_text, "count": mode_count},
            "mode_fraction": mode_fraction,
            "most_common": [
                {"text": text, "count": count}
                for text, count in frequency.most_common(10)
            ],
        },
        "whitespace_casefold_corpus_wer": wer,
        "integrity_thresholds": {
            "min_unique_top1": min_unique_top1,
            "max_mode_fraction": max_mode_fraction,
            "max_corpus_wer": max_corpus_wer,
        },
        "violations": violations,
        "provenance": {
            "nbest": file_record(nbest_path),
            "queries": file_record(queries_path),
            "qrels": file_record(qrels_path),
        },
        "claim_boundary": (
            "This is a catastrophic-content integrity gate, not model "
            "selection or test-set hyperparameter tuning. It accepts or "
            "rejects the complete frozen cache and never edits hypotheses."
        ),
    }


def write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    report = audit_cache(
        nbest_path=args.nbest.resolve(),
        queries_path=args.queries.resolve(),
        qrels_path=args.qrels.resolve(),
        condition=args.condition,
        expected_hypotheses=args.expected_hypotheses,
        min_unique_top1=args.min_unique_top1,
        max_mode_fraction=args.max_mode_fraction,
        max_corpus_wer=args.max_corpus_wer,
    )
    write_json_once(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "condition": report["condition"],
                "query_count": report["query_count"],
                "unique_top1": report["top1"]["unique_normalized_count"],
                "mode_fraction": report["top1"]["mode_fraction"],
                "WER": report["whitespace_casefold_corpus_wer"]["WER"],
                "violations": report["violations"],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
    )
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
