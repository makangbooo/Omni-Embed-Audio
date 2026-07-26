#!/usr/bin/env python3
"""Evaluate cached B3-B7/QG/Ours/U3-U4 with frozen FiQA-dev selections."""

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
from AudioRetrieval.asr_uncertainty_reranking.candidates import (  # noqa: E402
    rank_scores,
)
from AudioRetrieval.asr_uncertainty_reranking.bootstrap import (  # noqa: E402
    paired_bootstrap,
)
from AudioRetrieval.asr_uncertainty_reranking.configuration import (  # noqa: E402
    load_main_experiment_config,
)
from AudioRetrieval.asr_uncertainty_reranking.experiment import (  # noqa: E402
    evaluate_cached_reranking,
    truncate_query_input,
)
from AudioRetrieval.asr_uncertainty_reranking.gate import (  # noqa: E402
    CandidateGateMLP,
)
from AudioRetrieval.asr_uncertainty_reranking.metrics import (  # noqa: E402
    evaluate_rankings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--top100", type=Path, required=True)
    parser.add_argument("--nbest", type=Path, required=True)
    parser.add_argument("--cross-encoder", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--query-gate", type=Path, required=True)
    parser.add_argument("--candidate-gate", type=Path, required=True)
    parser.add_argument(
        "--ablation-gate",
        action="append",
        default=[],
        metavar="A4|A5|A6=PATH",
    )
    parser.add_argument("--gold-cross-encoder", type=Path)
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


def parse_named_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    allowed = {
        "A4_no_asr_confidence",
        "A5_no_nbest_entropy",
        "A6_no_rank_disagreement",
    }
    for raw in values:
        name, separator, path = raw.partition("=")
        if not separator or name not in allowed or not path:
            raise ValueError(
                "--ablation-gate must be NAME=PATH for A4/A5/A6 locked names"
            )
        if name in result:
            raise ValueError(f"duplicate ablation gate: {name}")
        result[name] = Path(path)
    return result


if __name__ == "__main__":
    args = parse_args()
    config = load_main_experiment_config(args.config)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("selected_on_split") != "fiqa_dev":
        raise ValueError("formal evaluation requires a FiQA-dev frozen selection")
    if selection.get("test_qrels_used") is not False:
        raise ValueError("selection provenance must state test_qrels_used=false")
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
        raise ValueError("formal cached inputs and qrels query sets must match")
    query_gate = CandidateGateMLP.from_dict(
        json.loads(args.query_gate.read_text(encoding="utf-8"))
    )
    candidate_gate = CandidateGateMLP.from_dict(
        json.loads(args.candidate_gate.read_text(encoding="utf-8"))
    )
    ablation_paths = parse_named_paths(args.ablation_gate)
    result = evaluate_cached_reranking(
        inputs,
        qrels,
        normalization=selection["normalization"],
        overlap_mode=selection["overlap_mode"],
        fixed_asr_weight=selection["fixed_asr_weight"],
        rrf_rank_constant=selection["rrf_rank_constant"],
        proxy_temperature=selection["proxy_temperature"],
        cross_encoder_temperature=selection["cross_encoder_temperature"],
        fusion_asr_source=selection["fusion_asr_source"],
        query_gate=query_gate,
        candidate_gate=candidate_gate,
    )
    for name, path in sorted(ablation_paths.items()):
        model = CandidateGateMLP.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
        ablation_result = evaluate_cached_reranking(
            inputs,
            qrels,
            normalization=selection["normalization"],
            overlap_mode=selection["overlap_mode"],
            fixed_asr_weight=selection["fixed_asr_weight"],
            rrf_rank_constant=selection["rrf_rank_constant"],
            proxy_temperature=selection["proxy_temperature"],
            cross_encoder_temperature=selection["cross_encoder_temperature"],
            fusion_asr_source=selection["fusion_asr_source"],
            candidate_gate=model,
        )
        result["evaluations"][name] = ablation_result["evaluations"][
            "Ours_candidate_gate"
        ]
        result["rankings"][name] = ablation_result["rankings"][
            "Ours_candidate_gate"
        ]
        result["diagnostics"][f"{name}_gate"] = {
            query_id: values["gate_outputs"]["Ours_candidate_gate"]
            for query_id, values in ablation_result["diagnostics"].items()
        }

    top50_inputs = {
        query_id: truncate_query_input(query, k=min(50, len(query.candidate_ids)))
        for query_id, query in inputs.items()
    }
    top50_result = evaluate_cached_reranking(
        top50_inputs,
        qrels,
        normalization=selection["normalization"],
        overlap_mode=selection["overlap_mode"],
        fixed_asr_weight=selection["fixed_asr_weight"],
        rrf_rank_constant=selection["rrf_rank_constant"],
        proxy_temperature=selection["proxy_temperature"],
        cross_encoder_temperature=selection["cross_encoder_temperature"],
        fusion_asr_source=selection["fusion_asr_source"],
        candidate_gate=candidate_gate,
    )
    result["evaluations"]["A9_candidate_gate_top50"] = top50_result[
        "evaluations"
    ]["Ours_candidate_gate"]
    result["rankings"]["A9_candidate_gate_top50"] = top50_result["rankings"][
        "Ours_candidate_gate"
    ]
    result["evaluations"]["A9_candidate_recall_top50"] = top50_result[
        "evaluations"
    ]["U4_candidate_recall"]

    if args.gold_cross_encoder is not None:
        gold_records = load_cross_encoder_scores(args.gold_cross_encoder)
        if set(gold_records) != set(inputs):
            raise ValueError("gold CE and formal input query sets must match")
        gold_rankings = {}
        for query_id, record in gold_records.items():
            if record.candidate_ids != inputs[query_id].candidate_ids:
                raise ValueError(f"gold CE candidate order differs for {query_id!r}")
            if len(record.scores) != 1:
                raise ValueError("gold CE artifact must contain exactly one score row")
            gold_rankings[query_id] = rank_scores(
                record.candidate_ids,
                record.scores[0],
            )
        result["rankings"]["U2_gold_ce"] = gold_rankings
        result["evaluations"]["U2_gold_ce"] = evaluate_rankings(
            gold_rankings,
            qrels,
        )

    primary = config["metrics"]["primary"]
    baseline_methods = [
        method
        for method in (
            "B3_oea",
            "B4_1best_ce",
            "B5_fixed_fusion",
            "B6_rrf",
            "B7a_4best_equal",
            "B7b_4best_max",
            "B7c_4best_proxy",
            "QG_query_gate",
        )
        if method in result["evaluations"]
    ]
    significance = {}
    ours_per_query = {
        query_id: values[primary]
        for query_id, values in result["evaluations"]["Ours_candidate_gate"][
            "per_query"
        ].items()
    }
    for baseline in baseline_methods:
        baseline_per_query = {
            query_id: values[primary]
            for query_id, values in result["evaluations"][baseline]["per_query"].items()
        }
        significance[baseline] = paired_bootstrap(
            ours_per_query,
            baseline_per_query,
            iterations=config["metrics"]["paired_bootstrap_iterations"],
            seed=config["metrics"]["paired_bootstrap_seed"],
        )
    strongest_baseline = max(
        baseline_methods,
        key=lambda method: result["evaluations"][method]["mean"][primary],
    )
    result["significance"] = significance
    result["strongest_baseline_on_this_evaluation"] = strongest_baseline
    result["strongest_baseline_selection_note"] = (
        "Reported descriptively; pairwise tests are retained for every preregistered baseline."
    )
    input_paths = [
        args.config,
        args.selection,
        args.top100,
        args.nbest,
        args.cross_encoder,
        args.qrels,
        args.query_gate,
        args.candidate_gate,
        *ablation_paths.values(),
    ]
    if args.gold_cross_encoder is not None:
        input_paths.append(args.gold_cross_encoder)
    result["provenance"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output("status", "--short", "--untracked-files=all"),
        "command": sys.argv,
        "input_sha256": {
            str(path.resolve()): sha256(path) for path in input_paths
        },
        "formal_components_present": {
            "A4_A6_ablation_gates": sorted(ablation_paths),
            "U2_gold_cross_encoder": args.gold_cross_encoder is not None,
        },
    }
    write_json_once(args.output_dir / "metrics.json", result)
    print(
        json.dumps(
            {
                "status": "complete",
                "query_count": len(inputs),
                "strongest_baseline": strongest_baseline,
                "output_dir": str(args.output_dir.resolve()),
            },
            indent=2,
        )
    )
