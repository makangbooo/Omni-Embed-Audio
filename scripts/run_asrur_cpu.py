#!/usr/bin/env python3
"""Validate or smoke-test the CPU half of the ASR uncertainty experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from AudioRetrieval.asr_uncertainty_reranking.aggregation import (  # noqa: E402
    aggregate_nbest_scores,
    build_asr_uncertainty_features,
    softmax_proxy_posteriors,
)
from AudioRetrieval.asr_uncertainty_reranking.bootstrap import (  # noqa: E402
    paired_bootstrap,
)
from AudioRetrieval.asr_uncertainty_reranking.configuration import (  # noqa: E402
    experiment_inventory,
    load_main_experiment_config,
)
from AudioRetrieval.asr_uncertainty_reranking.experiment import (  # noqa: E402
    RerankQueryInput,
    evaluate_cached_reranking,
)
from AudioRetrieval.asr_uncertainty_reranking.features import (  # noqa: E402
    FEATURE_NAMES,
    QUERY_ONLY_FEATURE_NAMES,
    build_candidate_feature_rows,
)
from AudioRetrieval.asr_uncertainty_reranking.gate import (  # noqa: E402
    build_training_groups,
    train_candidate_gate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT
        / "configs"
        / "asr_uncertainty_reranking"
        / "main_experiment.json",
    )
    parser.add_argument(
        "--stage",
        choices=("validate_config", "plan", "synthetic_smoke"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def write_json_once(path: Path, value: object) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def synthetic_query(index: int, *, noisy: bool) -> tuple[RerankQueryInput, dict[str, float]]:
    query_id = f"q{index:02d}"
    candidate_ids = tuple(f"{query_id}_d{candidate}" for candidate in range(6))
    positive_index = index % len(candidate_ids)
    qrels = {candidate_ids[positive_index]: 1.0}

    # OEA and ASR deliberately contain complementary errors so the gate and
    # fusion implementations are exercised without implying a real result.
    oea_scores = [0.85 - 0.08 * candidate for candidate in range(6)]
    oea_scores[positive_index] += 0.12 if noisy else -0.04
    proxy_logits = (-0.20, -0.28, -0.70 if noisy else -0.38, -0.95)
    hypotheses = (
        f"financial query {index}",
        f"finance query {index}",
        f"noisy fiscal request {index}",
        f"financial question {index}",
    )
    cross_encoder_scores = []
    for hypothesis_index in range(4):
        scores = [-0.5 - 0.07 * candidate for candidate in range(6)]
        scores[positive_index] = (
            0.25
            if noisy and hypothesis_index == 0
            else 1.4 - 0.15 * hypothesis_index
        )
        if noisy and hypothesis_index > 0:
            scores[positive_index] += 0.35
        cross_encoder_scores.append(tuple(scores))
    return (
        RerankQueryInput(
            query_id=query_id,
            candidate_ids=candidate_ids,
            oea_scores=tuple(oea_scores),
            hypotheses=hypotheses,
            proxy_logits=proxy_logits,
            cross_encoder_scores=tuple(cross_encoder_scores),
            top1_average_token_logprob=proxy_logits[0],
        ),
        qrels,
    )


def feature_rows_for_query(
    query: RerankQueryInput,
    qrels: dict[str, float],
    *,
    proxy_temperature: float,
    cross_encoder_temperature: float,
    normalization: str,
    overlap_mode: str,
):
    posteriors = softmax_proxy_posteriors(
        query.proxy_logits,
        temperature=proxy_temperature,
    )
    asr_scores = aggregate_nbest_scores(
        query.cross_encoder_scores,
        mode="proxy_posterior_logsumexp",
        proxy_posteriors=posteriors,
        cross_encoder_temperature=cross_encoder_temperature,
    )
    uncertainty = build_asr_uncertainty_features(
        query.hypotheses,
        query.proxy_logits,
        proxy_temperature=proxy_temperature,
        top1_average_token_logprob=query.top1_average_token_logprob,
        no_speech_probability=query.no_speech_probability,
    )
    return build_candidate_feature_rows(
        query_id=query.query_id,
        candidate_ids=query.candidate_ids,
        oea_scores=query.oea_scores,
        asr_scores=asr_scores,
        qrels=qrels,
        uncertainty=uncertainty,
        normalization=normalization,
        overlap_mode=overlap_mode,
    )


def run_synthetic_smoke(
    config: dict[str, Any],
    output_dir: Path,
    *,
    config_path: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    protocol = config["candidate_protocol"]
    gate_config = config["gate"]
    normalization = protocol["normalization"]
    overlap_mode = protocol["overlap_mode"]
    proxy_temperature = 1.0
    cross_encoder_temperature = 1.0

    synthetic_pairs = [
        synthetic_query(index, noisy=index % 2 == 1) for index in range(10)
    ]
    queries = {query.query_id: query for query, _ in synthetic_pairs}
    qrels = {
        query.query_id: query_qrels
        for query, query_qrels in synthetic_pairs
    }
    rows_by_query = {
        query_id: feature_rows_for_query(
            query,
            qrels[query_id],
            proxy_temperature=proxy_temperature,
            cross_encoder_temperature=cross_encoder_temperature,
            normalization=normalization,
            overlap_mode=overlap_mode,
        )
        for query_id, query in queries.items()
    }
    train_ids = [f"q{index:02d}" for index in range(6)]
    dev_ids = ["q06", "q07"]
    test_ids = ["q08", "q09"]
    train_groups = build_training_groups(
        [row for query_id in train_ids for row in rows_by_query[query_id]],
        group_size=gate_config["group_size"],
    )
    dev_groups = build_training_groups(
        [row for query_id in dev_ids for row in rows_by_query[query_id]],
        group_size=None,
    )
    candidate_gate, candidate_history = train_candidate_gate(
        train_groups,
        dev_groups,
        feature_names=FEATURE_NAMES,
        hidden_dim=gate_config["hidden_dim"],
        learning_rate=gate_config["learning_rate"],
        weight_decay=gate_config["weight_decay"],
        max_epochs=50,
        patience=10,
        seed=42,
    )
    query_gate, query_history = train_candidate_gate(
        train_groups,
        dev_groups,
        feature_names=QUERY_ONLY_FEATURE_NAMES,
        hidden_dim=gate_config["hidden_dim"],
        learning_rate=gate_config["learning_rate"],
        weight_decay=gate_config["weight_decay"],
        max_epochs=50,
        patience=10,
        seed=42,
    )
    evaluation = evaluate_cached_reranking(
        {query_id: queries[query_id] for query_id in test_ids},
        {query_id: qrels[query_id] for query_id in test_ids},
        normalization=normalization,
        overlap_mode=overlap_mode,
        fixed_asr_weight=0.5,
        rrf_rank_constant=60.0,
        proxy_temperature=proxy_temperature,
        cross_encoder_temperature=cross_encoder_temperature,
        fusion_asr_source=protocol["fusion_asr_source"],
        query_gate=query_gate,
        candidate_gate=candidate_gate,
    )
    per_query_ours = {
        query_id: metrics["nDCG@10"]
        for query_id, metrics in evaluation["evaluations"]["Ours_candidate_gate"][
            "per_query"
        ].items()
    }
    per_query_fixed = {
        query_id: metrics["nDCG@10"]
        for query_id, metrics in evaluation["evaluations"]["B5_fixed_fusion"][
            "per_query"
        ].items()
    }
    bootstrap = paired_bootstrap(
        per_query_ours,
        per_query_fixed,
        iterations=200,
        seed=42,
    )
    result = {
        "schema_version": 1,
        "status": "synthetic_smoke_only_not_a_research_result",
        "split_query_counts": {
            "train": len(train_ids),
            "dev": len(dev_ids),
            "test": len(test_ids),
        },
        "evaluation": evaluation,
        "paired_bootstrap_smoke": bootstrap,
        "candidate_gate_training": candidate_history,
        "query_gate_training": query_history,
    }
    write_json_once(output_dir / "metrics.json", result)
    write_json_once(
        output_dir / "gate_models.json",
        {
            "candidate_gate": candidate_gate.as_dict(),
            "query_gate": query_gate.as_dict(),
        },
    )
    write_json_once(
        output_dir / "run_manifest.json",
        {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_status_short": git_output("status", "--short", "--untracked-files=all"),
            "command": sys.argv,
            "config": str(config_path.resolve()),
            "gpu_disabled": os.environ.get("CUDA_VISIBLE_DEVICES", "") == "",
            "warning": "Synthetic data and metrics must never enter the research result table.",
        },
    )
    return result


if __name__ == "__main__":
    args = parse_args()
    config = load_main_experiment_config(args.config)
    inventory = experiment_inventory(config)
    if args.stage == "validate_config":
        print(json.dumps({"status": "valid", "inventory": inventory}, indent=2))
    elif args.stage == "plan":
        print(json.dumps(inventory, ensure_ascii=False, indent=2))
    else:
        if args.output_dir is None:
            raise SystemExit("--output-dir is required for synthetic_smoke")
        if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
            raise SystemExit("synthetic_smoke requires CUDA_VISIBLE_DEVICES to be empty")
        result = run_synthetic_smoke(
            config,
            args.output_dir,
            config_path=args.config,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "output_dir": str(args.output_dir.resolve()),
                    "methods": sorted(result["evaluation"]["evaluations"]),
                },
                indent=2,
            )
        )
