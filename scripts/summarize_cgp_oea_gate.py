#!/usr/bin/env python3
"""Apply the predeclared CGP-OEA recovery gate to paired JSON diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--cgp-dir", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def metric(report: dict[str, Any], space: str, key: str) -> float:
    values = report["spaces"][space]["query_recall"]
    if key == "MRR":
        return float(report["spaces"][space]["mean_reciprocal_rank"])
    return float(values[key])


def paired_bootstrap(
    method: list[float], baseline: list[float], *, seed: int, iterations: int = 10000
) -> dict[str, Any]:
    import numpy as np

    left = np.asarray(method, dtype=np.float64)
    right = np.asarray(baseline, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.size < 2:
        raise ValueError("paired bootstrap inputs must be matching vectors")
    differences = left - right
    generator = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 1000):
        stop = min(start + 1000, iterations)
        indices = generator.integers(0, left.size, size=(stop - start, left.size))
        draws[start:stop] = differences[indices].mean(axis=1)
    return {
        "observed_delta": float(differences.mean()),
        "confidence_interval_95": [
            float(value) for value in np.quantile(draws, [0.025, 0.975])
        ],
        "iterations": iterations,
        "seed": seed,
    }


def per_query_values(report: dict[str, Any], space: str, key: str) -> list[float]:
    field = "reciprocal_rank" if key == "MRR" else "hit_at_10"
    return [float(row[field]) for row in report["spaces"][space]["per_query"]]


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for variant in ("oea_nemo3b_cl", "oea_qwen3b_cl"):
        for subset in ("fiqa", "nq"):
            official_path = args.official_dir / f"{variant}_{subset}.json"
            cgp_path = args.cgp_dir / f"{variant}_{subset}.json"
            official = json.loads(official_path.read_text(encoding="utf-8"))
            cgp = json.loads(cgp_path.read_text(encoding="utf-8"))
            if official["query_ids"] != cgp["query_ids"]:
                raise ValueError(f"official/CGP query order differs for {variant}/{subset}")
            if official["document_ids"] != cgp["document_ids"]:
                raise ValueError(f"official/CGP candidate order differs for {variant}/{subset}")
            recovery = {}
            for metric_index, key in enumerate(("MRR", "Recall@10")):
                base = metric(cgp, "base_hidden", key)
                lora = metric(cgp, "lora_hidden", key)
                full = metric(official, "lora_plus_oea_heads", key)
                candidate = metric(cgp, "lora_plus_oea_heads", key)
                projection_gap = lora - full
                base_gap = base - full
                improvement = paired_bootstrap(
                    per_query_values(cgp, "lora_plus_oea_heads", key),
                    per_query_values(official, "lora_plus_oea_heads", key),
                    seed=20260805 + len(rows) * 10 + metric_index,
                )
                recovery[key] = {
                    "base": base,
                    "lora_hidden": lora,
                    "official_full": full,
                    "cgp": candidate,
                    "projection_gap_recovered_fraction": (
                        None
                        if projection_gap <= 0
                        else (candidate - full) / projection_gap
                    ),
                    "base_gap_recovered_fraction_secondary": (
                        None if base_gap <= 0 else (candidate - full) / base_gap
                    ),
                    "absolute_delta_vs_official_full": candidate - full,
                    "paired_improvement": improvement,
                }
            rows.append({"variant": variant, "subset": subset, "recovery": recovery})
    summaries = []
    for path in args.training_summary:
        value = json.loads(path.read_text(encoding="utf-8"))
        baseline = value["baseline_metrics"]
        best = value["best_metrics"]
        deltas = {
            key: float(best[key]) - float(baseline[key])
            for key in ("R@1", "R@5", "R@10")
        }
        baseline_ranks = [int(rank) for rank in baseline["ranks"]]
        best_ranks = [int(rank) for rank in best["ranks"]]
        paired = {
            key: paired_bootstrap(
                [float(rank <= cutoff) for rank in best_ranks],
                [float(rank <= cutoff) for rank in baseline_ranks],
                seed=20260900 + cutoff,
            )
            for key, cutoff in (("R@1", 1), ("R@5", 5), ("R@10", 10))
        }
        summaries.append(
            {
                "path": str(path.resolve()),
                "dataset": value.get("dataset"),
                "deltas": deltas,
                "paired_deltas": paired,
                "in_domain_noninferior_at_2pp": all(
                    result["confidence_interval_95"][0] >= -0.02
                    for result in paired.values()
                ),
            }
        )
    recovery_ok = all(
        item["recovery"][key]["projection_gap_recovered_fraction"] is not None
        and item["recovery"][key]["projection_gap_recovered_fraction"] >= 0.5
        and item["recovery"][key]["paired_improvement"]["confidence_interval_95"][0] > 0.0
        for item in rows for key in ("MRR", "Recall@10")
    )
    in_domain_ok = all(item["in_domain_noninferior_at_2pp"] for item in summaries) if summaries else False
    result = {
        "schema_version": 1,
        "status": "pass" if recovery_ok and in_domain_ok else "hold_negative_uiq",
        "recovery_gate": {
            "reference": "lora_hidden_to_official_full_projection_gap",
            "required_fraction": 0.5,
            "requires_positive_paired_95ci": True,
            "passed": recovery_ok,
        },
        "in_domain_gate": {
            "noninferiority_margin": 0.02,
            "requires_paired_95ci_lower_bound": True,
            "passed": in_domain_ok,
        },
        "rows": rows,
        "training_summaries": summaries,
        "decision": "Only a pass permits adding negative UIQ loss; intent gate remains deferred until after that experiment.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
