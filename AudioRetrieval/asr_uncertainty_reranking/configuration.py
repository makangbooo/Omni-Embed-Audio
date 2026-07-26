"""Validation for the preregistered main-experiment JSON configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

REQUIRED_METHODS = {
    "B1_whisper_1best_bge_dense",
    "B2_original_omni",
    "B3_oea",
    "B4_oea_top100_1best_ce",
    "B5_fixed_fusion",
    "B6_rrf",
    "B7a_4best_equal",
    "B7b_4best_max",
    "B7c_4best_proxy_posterior",
    "QG_query_gate",
    "Ours_candidate_gate",
    "U1_gold_bge_dense",
    "U2_gold_ce",
    "U3_candidate_oracle",
    "U4_candidate_recall",
}
REQUIRED_ABLATIONS = {f"A{index}" for index in range(1, 10)}
REQUIRED_CONDITIONS = ["clean", "snr_20", "snr_10", "snr_0"]


def validate_main_experiment_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must equal 1")
    required_top_level = {
        "project",
        "status",
        "datasets",
        "models",
        "candidate_protocol",
        "selection_grids",
        "gate",
        "methods",
        "ablations",
        "metrics",
        "go_no_go",
    }
    missing = sorted(required_top_level - set(config))
    if missing:
        raise ValueError(f"missing main config fields: {missing}")
    if set(config["methods"]) != REQUIRED_METHODS:
        raise ValueError("method registry does not match B1-B7/QG/Ours/U1-U4")
    if set(config["ablations"]) != REQUIRED_ABLATIONS:
        raise ValueError("ablation registry must contain A1-A9")
    datasets = config["datasets"]
    for dataset in ("fiqa", "nq"):
        if datasets[dataset]["conditions"] != REQUIRED_CONDITIONS:
            raise ValueError(f"{dataset} acoustic conditions are not protocol-locked")
    protocol = config["candidate_protocol"]
    if protocol["depth"] != 100 or not protocol["immutable_across_rerankers"]:
        raise ValueError("the main protocol requires one immutable OEA Top-100")
    if protocol["normalization"] not in {"zscore", "rank"}:
        raise ValueError("unsupported score normalization")
    if protocol["overlap_mode"] not in {"overlap_coefficient", "jaccard"}:
        raise ValueError("unsupported overlap mode")
    if protocol["fusion_asr_source"] not in {"one_best", "proxy_posterior"}:
        raise ValueError("unsupported B5/B6 fusion ASR source")
    if config["models"]["oea"]["embedding_dim"] != 512:
        raise ValueError("OEA output must be 512-dimensional")
    if config["models"]["asr"]["num_return_sequences"] != 4:
        raise ValueError("the complete method requires Whisper 4-best")
    seeds = config["gate"]["seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) != 3
        or len(set(seeds)) != 3
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
    ):
        raise ValueError("gate.seeds must contain three distinct integers")
    if config["metrics"]["primary"] != "nDCG@10":
        raise ValueError("the primary metric must be nDCG@10")
    if config["go_no_go"]["candidate_generation_change_authorized"]:
        raise ValueError("dual-route candidate generation is not authorized")
    json.dumps(config, allow_nan=False)


def load_main_experiment_config(path: Path | str) -> Dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("main experiment config must be a JSON object")
    validate_main_experiment_config(config)
    return config


def experiment_inventory(config: Mapping[str, Any]) -> Dict[str, object]:
    validate_main_experiment_config(config)
    conditions = config["datasets"]["fiqa"]["conditions"]
    seeds = config["gate"]["seeds"]
    return {
        "baseline_and_upper_bound_methods": list(config["methods"]),
        "baseline_and_upper_bound_method_count": len(config["methods"]),
        "fiqa_conditions": list(conditions),
        "gate_seeds": list(seeds),
        "ablations": dict(config["ablations"]),
        "ablation_count": len(config["ablations"]),
        "fiqa_formal_gate_cells": len(conditions) * len(seeds),
        "nq_execution_status": "blocked_until_fiqa_go_and_new_gpu_approval",
    }
