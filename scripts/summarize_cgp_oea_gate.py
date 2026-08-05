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


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for variant in ("oea_nemo3b_cl", "oea_qwen3b_cl"):
        for subset in ("fiqa", "nq"):
            official_path = args.official_dir / f"{variant}_{subset}.json"
            cgp_path = args.cgp_dir / f"{variant}_{subset}.json"
            official = json.loads(official_path.read_text(encoding="utf-8"))
            cgp = json.loads(cgp_path.read_text(encoding="utf-8"))
            recovery = {}
            for key in ("MRR", "Recall@10"):
                base = metric(cgp, "base_hidden", key)
                full = metric(official, "lora_plus_oea_heads", key)
                candidate = metric(cgp, "lora_plus_oea_heads", key)
                gap = base - full
                recovery[key] = {
                    "base": base,
                    "official_full": full,
                    "cgp": candidate,
                    "gap_recovered_fraction": None if gap <= 0 else (candidate - full) / gap,
                    "absolute_delta_vs_official_full": candidate - full,
                }
            rows.append({"variant": variant, "subset": subset, "recovery": recovery})
    summaries = []
    for path in args.training_summary:
        value = json.loads(path.read_text(encoding="utf-8"))
        baseline = value["baseline_metrics"]
        best = value["best_metrics"]
        deltas = {key: float(best[key]) - float(baseline[key]) for key in ("R@1", "R@5", "R@10")}
        summaries.append({"path": str(path.resolve()), "dataset": value.get("dataset"), "deltas": deltas, "in_domain_drop_within_2pp": all(delta >= -0.02 for delta in deltas.values())})
    recovery_ok = all(
        item["recovery"][key]["gap_recovered_fraction"] is not None
        and item["recovery"][key]["gap_recovered_fraction"] >= 0.5
        for item in rows for key in ("MRR", "Recall@10")
    )
    in_domain_ok = all(item["in_domain_drop_within_2pp"] for item in summaries) if summaries else False
    result = {
        "schema_version": 1,
        "status": "pass" if recovery_ok and in_domain_ok else "hold_negative_uiq",
        "recovery_gate": {"required_fraction": 0.5, "passed": recovery_ok},
        "in_domain_gate": {"required_drop_at_most": 0.02, "passed": in_domain_ok},
        "rows": rows,
        "training_summaries": summaries,
        "decision": "Only a pass permits adding negative UIQ loss; intent gate remains deferred until after that experiment.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
