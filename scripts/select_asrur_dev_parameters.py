#!/usr/bin/env python3
"""Select only preregistered reranking hyperparameters on FiQA dev."""

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
from AudioRetrieval.asr_uncertainty_reranking.experiment import (  # noqa: E402
    evaluate_cached_reranking,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--top100", type=Path, required=True)
    parser.add_argument("--nbest", type=Path, required=True)
    parser.add_argument("--cross-encoder", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_once(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def select_best(rows, *, parameter_key):
    # Highest dev nDCG wins; lexicographically smallest parameter tuple breaks ties.
    ordered = sorted(rows, key=lambda row: (-row["nDCG@10"], parameter_key(row)))
    return ordered[0]


if __name__ == "__main__":
    args = parse_args()
    config = load_main_experiment_config(args.config)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    inputs = assemble_rerank_inputs(
        load_frozen_candidates(args.top100),
        load_nbest(args.nbest),
        load_cross_encoder_scores(args.cross_encoder),
    )
    qrels = load_unbounded_qrels(args.qrels)
    if set(inputs) != set(qrels):
        raise ValueError("FiQA dev cached inputs and qrels query sets must match")

    protocol = config["candidate_protocol"]
    grids = config["selection_grids"]
    fixed_rows = []
    for weight in grids["fixed_asr_weight"]:
        result = evaluate_cached_reranking(
            inputs,
            qrels,
            normalization=protocol["normalization"],
            overlap_mode=protocol["overlap_mode"],
            fusion_asr_source=protocol["fusion_asr_source"],
            fixed_asr_weight=weight,
        )
        fixed_rows.append(
            {
                "fixed_asr_weight": weight,
                "nDCG@10": result["evaluations"]["B5_fixed_fusion"]["mean"]["nDCG@10"],
            }
        )
    rrf_rows = []
    for constant in grids["rrf_rank_constant"]:
        result = evaluate_cached_reranking(
            inputs,
            qrels,
            normalization=protocol["normalization"],
            overlap_mode=protocol["overlap_mode"],
            fusion_asr_source=protocol["fusion_asr_source"],
            rrf_rank_constant=constant,
        )
        rrf_rows.append(
            {
                "rrf_rank_constant": constant,
                "nDCG@10": result["evaluations"]["B6_rrf"]["mean"]["nDCG@10"],
            }
        )
    temperature_rows = []
    for proxy_temperature in grids["proxy_temperature"]:
        for ce_temperature in grids["cross_encoder_temperature"]:
            result = evaluate_cached_reranking(
                inputs,
                qrels,
                normalization=protocol["normalization"],
                overlap_mode=protocol["overlap_mode"],
                fusion_asr_source=protocol["fusion_asr_source"],
                proxy_temperature=proxy_temperature,
                cross_encoder_temperature=ce_temperature,
            )
            temperature_rows.append(
                {
                    "proxy_temperature": proxy_temperature,
                    "cross_encoder_temperature": ce_temperature,
                    "nDCG@10": result["evaluations"]["B7c_4best_proxy"]["mean"]["nDCG@10"],
                }
            )

    fixed_best = select_best(
        fixed_rows,
        parameter_key=lambda row: row["fixed_asr_weight"],
    )
    rrf_best = select_best(
        rrf_rows,
        parameter_key=lambda row: row["rrf_rank_constant"],
    )
    temperature_best = select_best(
        temperature_rows,
        parameter_key=lambda row: (
            row["proxy_temperature"],
            row["cross_encoder_temperature"],
        ),
    )
    selection = {
        "schema_version": 1,
        "selected_on_split": "fiqa_dev",
        "selection_metric": "nDCG@10",
        "tie_break": "smallest_parameter_tuple",
        "fixed_asr_weight": fixed_best["fixed_asr_weight"],
        "rrf_rank_constant": rrf_best["rrf_rank_constant"],
        "proxy_temperature": temperature_best["proxy_temperature"],
        "cross_encoder_temperature": temperature_best["cross_encoder_temperature"],
        "normalization": protocol["normalization"],
        "overlap_mode": protocol["overlap_mode"],
        "fusion_asr_source": protocol["fusion_asr_source"],
        "grid_results": {
            "fixed_fusion": fixed_rows,
            "rrf": rrf_rows,
            "temperatures": temperature_rows,
        },
        "input_sha256": {
            str(path.resolve()): sha256(path)
            for path in (args.config, args.top100, args.nbest, args.cross_encoder, args.qrels)
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output("status", "--short", "--untracked-files=all"),
        "command": sys.argv,
        "test_qrels_used": False,
    }
    write_json_once(args.output_dir / "frozen_selection.json", selection)
    print(
        json.dumps(
            {
                key: selection[key]
                for key in (
                    "fixed_asr_weight",
                    "rrf_rank_constant",
                    "proxy_temperature",
                    "cross_encoder_temperature",
                )
            },
            indent=2,
        )
    )
