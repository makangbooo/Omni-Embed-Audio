#!/usr/bin/env python3
"""Evaluate CE baselines that require no FiQA-dev hyperparameter selection."""

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
    assemble_rerank_inputs,
    load_cross_encoder_scores,
    load_frozen_candidates,
    load_nbest,
    load_unbounded_qrels,
)
from AudioRetrieval.asr_uncertainty_reranking.configuration import (  # noqa: E402
    load_main_experiment_config,
)
from AudioRetrieval.asr_uncertainty_reranking.candidates import (  # noqa: E402
    rank_scores,
)
from AudioRetrieval.asr_uncertainty_reranking.experiment import (  # noqa: E402
    evaluate_cached_reranking,
)
from AudioRetrieval.asr_uncertainty_reranking.metrics import (  # noqa: E402
    evaluate_rankings,
)

UNSELECTED_METHODS = (
    "B4_1best_ce",
    "B7a_4best_equal",
    "B7b_4best_max",
)
GOLD_METHOD = "U2_gold_ce"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--top100", type=Path, required=True)
    parser.add_argument("--nbest", type=Path, required=True)
    parser.add_argument("--cross-encoder", type=Path, required=True)
    parser.add_argument("--gold-cross-encoder", type=Path)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def evaluate_unselected(
    *,
    config_path: Path,
    top100_path: Path,
    nbest_path: Path,
    cross_encoder_path: Path,
    gold_cross_encoder_path: Path | None,
    qrels_path: Path,
) -> dict:
    config = load_main_experiment_config(config_path)
    inputs = assemble_rerank_inputs(
        load_frozen_candidates(top100_path),
        load_nbest(nbest_path, expected_size=4),
        load_cross_encoder_scores(cross_encoder_path),
    )
    qrels = load_unbounded_qrels(qrels_path)
    if set(inputs) != set(qrels):
        raise ValueError("formal cached inputs and qrels query sets must match")
    expected_queries = int(config["datasets"]["fiqa"]["test_queries_expected"])
    if len(inputs) != expected_queries:
        raise ValueError("query count does not match locked FiQA test split")

    protocol = config["candidate_protocol"]
    complete = evaluate_cached_reranking(
        inputs,
        qrels,
        normalization=protocol["normalization"],
        overlap_mode=protocol["overlap_mode"],
        # These defaults are deliberately irrelevant to the three retained
        # methods. Selected fusion/RRF/proxy methods are not emitted here.
        fixed_asr_weight=0.5,
        rrf_rank_constant=60.0,
        proxy_temperature=1.0,
        cross_encoder_temperature=1.0,
        fusion_asr_source=protocol["fusion_asr_source"],
    )
    evaluations = {
        method: complete["evaluations"][method] for method in UNSELECTED_METHODS
    }
    rankings = {
        method: complete["rankings"][method] for method in UNSELECTED_METHODS
    }
    reported_methods = list(UNSELECTED_METHODS)
    if gold_cross_encoder_path is not None:
        gold_records = load_cross_encoder_scores(gold_cross_encoder_path)
        if set(gold_records) != set(inputs):
            raise ValueError("gold CE and formal input query sets must match")
        gold_rankings = {}
        for query_id, record in sorted(gold_records.items()):
            if record.candidate_ids != inputs[query_id].candidate_ids:
                raise ValueError(
                    f"gold CE candidate order differs for {query_id!r}"
                )
            if len(record.scores) != 1:
                raise ValueError(
                    "gold CE artifact must contain exactly one score row"
                )
            gold_rankings[query_id] = rank_scores(
                record.candidate_ids,
                record.scores[0],
            )
        evaluations[GOLD_METHOD] = evaluate_rankings(gold_rankings, qrels)
        rankings[GOLD_METHOD] = gold_rankings
        reported_methods.append(GOLD_METHOD)

    input_paths = [
        config_path,
        top100_path,
        nbest_path,
        cross_encoder_path,
        qrels_path,
    ]
    if gold_cross_encoder_path is not None:
        input_paths.append(gold_cross_encoder_path)
    return {
        "schema_version": 1,
        "status": "complete",
        "scale": "fraction",
        "selection_status": "no_dev_selection_required",
        "reported_methods": reported_methods,
        "excluded_until_fiqa_dev_selection": [
            "B5_fixed_fusion",
            "B6_rrf",
            "B7c_4best_proxy",
        ],
        "evaluations": evaluations,
        "rankings": rankings,
        "candidate_membership_reference": {
            query_id: list(query.candidate_ids)
            for query_id, query in sorted(inputs.items())
        },
        "provenance": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_status_short": git_output(
                "status",
                "--short",
                "--untracked-files=all",
            ),
            "input_sha256": {
                str(path.resolve()): sha256(path)
                for path in input_paths
            },
            "test_qrels_used_for_parameter_selection": False,
            "command": sys.argv,
        },
    }


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    result = evaluate_unselected(
        config_path=args.config,
        top100_path=args.top100,
        nbest_path=args.nbest,
        cross_encoder_path=args.cross_encoder,
        gold_cross_encoder_path=args.gold_cross_encoder,
        qrels_path=args.qrels,
    )
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "metrics.json"
    output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "methods": result["reported_methods"],
                "output": str(output.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
