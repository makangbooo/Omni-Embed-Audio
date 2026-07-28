#!/usr/bin/env python3
"""Stratify a valid Phase-2 retrieval baseline by per-query Whisper WER."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from AudioRetrieval.asr_uncertainty_reranking.artifacts import load_nbest  # noqa: E402
from AudioRetrieval.asr_uncertainty_reranking.data import load_text_queries  # noqa: E402
from AudioRetrieval.asr_uncertainty_reranking.metrics import (  # noqa: E402
    corpus_wer,
    word_error_counts,
)
from AudioRetrieval.asr_uncertainty_reranking.reporting import (  # noqa: E402
    summarize_metric_by_wer_bins,
)

CONDITIONS = ("clean", "snr_20", "snr_10", "snr_0")
DEFAULT_METRICS = (
    "nDCG@10",
    "MRR@10",
    "Recall@10",
    "Recall@20",
    "Recall@50",
    "Recall@100",
)
DEFAULT_BOUNDARIES = (0.0, 0.1, 0.25, 0.5, 1.0, math.inf)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--method",
        default="B1_whisper_1best_bge_dense",
    )
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        choices=DEFAULT_METRICS,
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _finite_metric(
    per_query: Mapping[str, Any],
    query_id: str,
    metric_name: str,
) -> float:
    row = per_query.get(query_id)
    if not isinstance(row, Mapping):
        raise KeyError(f"missing per-query metrics for {query_id!r}")
    raw = row.get(metric_name)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TypeError(f"{query_id}:{metric_name} must be numeric")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{query_id}:{metric_name} must be finite")
    return value


def _json_safe_bins(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        converted = dict(row)
        upper = float(converted["upper_exclusive"])
        converted["upper_exclusive"] = None if math.isinf(upper) else upper
        result.append(converted)
    return result


def analyze(
    *,
    cache_root: Path,
    result_root: Path,
    queries_path: Path,
    method: str,
    metric_names: Sequence[str] = DEFAULT_METRICS,
    boundaries: Sequence[float] = DEFAULT_BOUNDARIES,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if not method:
        raise ValueError("method must be non-empty")
    if not metric_names or len(metric_names) != len(set(metric_names)):
        raise ValueError("metric_names must be non-empty and unique")

    queries = load_text_queries(queries_path)
    condition_reports: dict[str, Any] = {}
    bucket_rows: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []
    provenance_inputs = [queries_path.resolve()]

    for condition in CONDITIONS:
        nbest_path = (cache_root / "whisper" / condition / "nbest.jsonl").resolve()
        metrics_path = (result_root / condition / "metrics.json").resolve()
        if not nbest_path.is_file():
            raise FileNotFoundError(nbest_path)
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        provenance_inputs.extend([nbest_path, metrics_path])

        nbest = load_nbest(nbest_path, expected_size=4)
        metrics_document = _load_json_object(metrics_path)
        evaluations = metrics_document.get("evaluations")
        if not isinstance(evaluations, Mapping):
            raise TypeError(f"evaluations must be an object: {metrics_path}")
        method_evaluation = evaluations.get(method)
        if not isinstance(method_evaluation, Mapping):
            raise KeyError(f"missing evaluation method {method!r}: {metrics_path}")
        per_query = method_evaluation.get("per_query")
        if not isinstance(per_query, Mapping):
            raise TypeError(f"{method}.per_query must be an object: {metrics_path}")

        query_ids = set(nbest)
        if query_ids != set(per_query):
            raise ValueError(f"{condition}: N-best/evaluation query sets differ")
        absent_queries = sorted(query_ids - set(queries))
        if absent_queries:
            raise ValueError(
                f"{condition}: query text missing for IDs {absent_queries[:20]}"
            )

        wer_by_query: dict[str, float] = {}
        counts_by_query = {}
        for query_id in sorted(query_ids):
            counts = word_error_counts(
                queries[query_id].text,
                nbest[query_id].hypotheses[0].text,
            )
            if counts.wer is None:
                raise ValueError(f"{condition}:{query_id} has an empty reference")
            wer_by_query[query_id] = float(counts.wer)
            counts_by_query[query_id] = counts

        metric_summaries = {}
        metric_values = {}
        for metric_name in metric_names:
            by_query = {
                query_id: _finite_metric(per_query, query_id, metric_name)
                for query_id in sorted(query_ids)
            }
            metric_values[metric_name] = by_query
            bins = _json_safe_bins(
                summarize_metric_by_wer_bins(
                    wer_by_query,
                    by_query,
                    boundaries=boundaries,
                )
            )
            metric_summaries[metric_name] = bins
            for row in bins:
                bucket_rows.append(
                    {
                        "condition": condition,
                        "method": method,
                        "metric": metric_name,
                        **row,
                    }
                )

        corpus_summary = corpus_wer(
            (
                queries[query_id].text,
                nbest[query_id].hypotheses[0].text,
            )
            for query_id in sorted(query_ids)
        )
        condition_reports[condition] = {
            "query_count": len(query_ids),
            "corpus_wer": corpus_summary,
            "per_query_wer": {
                "mean": math.fsum(wer_by_query.values()) / len(wer_by_query),
                "minimum": min(wer_by_query.values()),
                "maximum": max(wer_by_query.values()),
            },
            "metric_bins": metric_summaries,
        }

        for query_id in sorted(query_ids):
            counts = counts_by_query[query_id]
            per_query_rows.append(
                {
                    "condition": condition,
                    "query_id": query_id,
                    "wer": wer_by_query[query_id],
                    "substitutions": counts.substitutions,
                    "deletions": counts.deletions,
                    "insertions": counts.insertions,
                    "reference_words": counts.reference_words,
                    **{
                        metric_name: metric_values[metric_name][query_id]
                        for metric_name in metric_names
                    },
                }
            )

    report = {
        "schema_version": 1,
        "status": "complete",
        "stage": "asrur_phase2_wer_stratification",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "conditions": condition_reports,
        "metric_names": list(metric_names),
        "wer_bin_boundaries": [
            None if math.isinf(float(value)) else float(value)
            for value in boundaries
        ],
        "provenance": {
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_status_short": git_output(
                "status",
                "--short",
                "--untracked-files=all",
            ),
            "input_sha256": {
                str(path): sha256(path)
                for path in provenance_inputs
            },
        },
        "claim_boundary": (
            "This is a post-hoc diagnostic over fixed formal outputs. It does "
            "not select a model, threshold, candidate set, or test parameter."
        ),
    }
    return report, bucket_rows, per_query_rows


def write_outputs(
    output_dir: Path,
    report: Mapping[str, Any],
    bucket_rows: Sequence[Mapping[str, Any]],
    per_query_rows: Sequence[Mapping[str, Any]],
) -> None:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "wer_analysis.json").write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with (output_dir / "wer_bins.csv").open(
        "x",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "condition",
                "method",
                "metric",
                "lower_inclusive",
                "upper_exclusive",
                "count",
                "metric_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(bucket_rows)
    with (output_dir / "per_query.jsonl").open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as stream:
        for row in per_query_rows:
            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                )
                + "\n"
            )


def main() -> int:
    args = parse_args()
    report, bucket_rows, per_query_rows = analyze(
        cache_root=args.cache_root.resolve(),
        result_root=args.result_root.resolve(),
        queries_path=args.queries.resolve(),
        method=args.method,
        metric_names=tuple(args.metrics or DEFAULT_METRICS),
    )
    write_outputs(args.output_dir, report, bucket_rows, per_query_rows)
    print(
        json.dumps(
            {
                "status": "complete",
                "method": report["method"],
                "condition_count": len(report["conditions"]),
                "per_query_row_count": len(per_query_rows),
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
