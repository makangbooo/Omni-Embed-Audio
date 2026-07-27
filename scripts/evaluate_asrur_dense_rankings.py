#!/usr/bin/env python3
"""Evaluate cached full-corpus B1/B2/B3/U1 dense rankings on global qrels."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from AudioRetrieval.asr_uncertainty_reranking.artifacts import (  # noqa: E402
    load_frozen_candidates,
    load_unbounded_qrels,
)
from AudioRetrieval.asr_uncertainty_reranking.candidates import (  # noqa: E402
    rank_scores,
)
from AudioRetrieval.asr_uncertainty_reranking.metrics import (  # noqa: E402
    evaluate_candidate_oracle,
    evaluate_rankings,
)

ALLOWED_METHODS = {
    "B1_whisper_1best_bge_dense",
    "B2_original_omni",
    "B3_oea",
    "U1_gold_bge_dense",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranking",
        action="append",
        required=True,
        metavar="METHOD=PATH",
    )
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_rankings(values: list[str]) -> dict[str, Path]:
    result = {}
    for raw in values:
        method, separator, path = raw.partition("=")
        if not separator or method not in ALLOWED_METHODS or not path:
            raise ValueError(
                "--ranking must use B1/B2/B3/U1 canonical METHOD=PATH names"
            )
        if method in result:
            raise ValueError(f"duplicate ranking method: {method}")
        result[method] = Path(path)
    return result


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


if __name__ == "__main__":
    args = parse_args()
    ranking_paths = parse_rankings(args.ranking)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    qrels = load_unbounded_qrels(args.qrels)
    evaluations = {}
    rankings = {}
    candidates = {}
    for method, path in sorted(ranking_paths.items()):
        records = load_frozen_candidates(path)
        if set(records) != set(qrels):
            raise ValueError(f"{method} ranking/qrels query sets differ")
        method_rankings = {
            query_id: rank_scores(record.candidate_ids, record.scores)
            for query_id, record in records.items()
        }
        rankings[method] = method_rankings
        candidates[method] = {
            query_id: record.candidate_ids
            for query_id, record in records.items()
        }
        evaluations[method] = evaluate_rankings(method_rankings, qrels)
    oea_candidate_oracle = (
        evaluate_candidate_oracle(candidates["B3_oea"], qrels)
        if "B3_oea" in candidates
        else None
    )
    result = {
        "schema_version": 1,
        "scale": "fraction",
        "evaluations": evaluations,
        "oea_candidate_oracle": oea_candidate_oracle,
        "rankings": rankings,
        "provenance": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_status_short": git_output(
                "status",
                "--short",
                "--untracked-files=all",
            ),
            "command": sys.argv,
            "input_sha256": {
                str(path.resolve()): sha256(path)
                for path in [args.qrels, *ranking_paths.values()]
            },
        },
    }
    with (args.output_dir / "metrics.json").open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as stream:
        json.dump(
            result,
            stream,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "methods": sorted(evaluations),
                "query_count": len(qrels),
                "output_dir": str(args.output_dir.resolve()),
            },
            indent=2,
        )
    )
