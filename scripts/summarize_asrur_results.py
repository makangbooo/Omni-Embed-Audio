#!/usr/bin/env python3
"""Build FiQA/NQ main, seed, noise, gate, and significance result tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from AudioRetrieval.asr_uncertainty_reranking.configuration import (  # noqa: E402
    load_main_experiment_config,
)
from AudioRetrieval.asr_uncertainty_reranking.metrics import (  # noqa: E402
    noise_degradation,
)
from AudioRetrieval.asr_uncertainty_reranking.reporting import (  # noqa: E402
    summarize_seeds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cell-index", type=Path, required=True)
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


def load_metrics(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("scale") != "fraction" or not isinstance(
        value.get("evaluations"),
        dict,
    ):
        raise ValueError(f"invalid metrics artifact: {path}")
    return value


def write_csv_once(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    args = parse_args()
    config = load_main_experiment_config(args.config)
    index = json.loads(args.cell_index.read_text(encoding="utf-8"))
    dataset = index.get("dataset")
    if dataset not in {"fiqa", "nq"}:
        raise ValueError("cell index dataset must be fiqa or nq")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    conditions = config["datasets"][dataset]["conditions"]
    seeds = config["gate"]["seeds"]
    cells = index.get("cells")
    if not isinstance(cells, list):
        raise TypeError("cell index cells must be a list")
    expected = {(condition, seed) for condition in conditions for seed in seeds}
    actual = set()
    input_paths = [args.config, args.cell_index]
    metric_values = defaultdict(list)
    gate_values = defaultdict(list)
    significance_rows = []
    for raw_cell in cells:
        if not isinstance(raw_cell, dict):
            raise TypeError("every cell must be an object")
        condition = raw_cell.get("condition")
        seed = raw_cell.get("seed")
        cell_key = (condition, seed)
        if cell_key not in expected or cell_key in actual:
            raise ValueError(f"invalid or duplicate cell: {cell_key}")
        actual.add(cell_key)
        paths = []
        for key in ("dense_metrics", "rerank_metrics"):
            path = Path(raw_cell[key])
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(path)
            input_paths.append(path)
        merged_evaluations = {}
        rerank_payload = None
        for path in paths:
            payload = load_metrics(path)
            overlap = set(merged_evaluations).intersection(payload["evaluations"])
            for method in overlap:
                if merged_evaluations[method] != payload["evaluations"][method]:
                    raise ValueError(
                        f"conflicting duplicate method across cell artifacts: {method}"
                    )
            merged_evaluations.update(
                {
                    method: evaluation
                    for method, evaluation in payload["evaluations"].items()
                    if method not in overlap
                }
            )
            if "Ours_candidate_gate" in payload["evaluations"]:
                rerank_payload = payload
        for method, evaluation in sorted(merged_evaluations.items()):
            mean = evaluation.get("mean")
            if not isinstance(mean, dict):
                continue
            for metric, value in sorted(mean.items()):
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError("result metrics must be finite")
                metric_values[(condition, method, metric)].append((seed, numeric))
        if rerank_payload is not None:
            for query_id, diagnostics in rerank_payload.get(
                "diagnostics",
                {},
            ).items():
                gate_output = diagnostics.get("gate_outputs", {}).get(
                    "Ours_candidate_gate"
                )
                if gate_output is not None:
                    for gate in gate_output["gates"]:
                        gate_values[(condition, seed)].append(float(gate))
            for baseline, value in rerank_payload.get("significance", {}).items():
                significance_rows.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "seed": seed,
                        "ours": "Ours_candidate_gate",
                        "baseline": baseline,
                        "mean_difference": value["mean_delta"],
                        "confidence_95_lower": value["confidence_interval"][0],
                        "confidence_95_upper": value["confidence_interval"][1],
                        "two_sided_p_value": value["two_sided_p_value"],
                    }
                )
    if actual != expected:
        raise ValueError(f"missing formal cells: {sorted(expected - actual)}")

    result_rows = []
    aggregate_lookup = {}
    for (condition, method, metric), values in sorted(metric_values.items()):
        if {seed for seed, _ in values} != set(seeds):
            raise ValueError(f"incomplete seed set for {(condition, method, metric)}")
        summary = summarize_seeds([value for _, value in sorted(values)])
        aggregate_lookup[(condition, method, metric)] = summary["mean"]
        for seed, value in sorted(values):
            result_rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "method": method,
                    "metric": metric,
                    "seed": seed,
                    "value": value,
                    "seed_mean": "",
                    "seed_sample_std": "",
                    "confidence_95_lower": "",
                    "confidence_95_upper": "",
                }
            )
        result_rows.append(
            {
                "dataset": dataset,
                "condition": condition,
                "method": method,
                "metric": metric,
                "seed": "ALL",
                "value": "",
                "seed_mean": summary["mean"],
                "seed_sample_std": summary["sample_standard_deviation"],
                "confidence_95_lower": summary["confidence_95_lower"],
                "confidence_95_upper": summary["confidence_95_upper"],
            }
        )

    primary = config["metrics"]["primary"]
    methods = sorted(
        {
            method
            for condition, method, metric in aggregate_lookup
            if metric == primary
        }
    )
    noise_rows = []
    for method in methods:
        clean_key = ("clean", method, primary)
        noisy_key = ("snr_0", method, primary)
        if clean_key not in aggregate_lookup or noisy_key not in aggregate_lookup:
            raise ValueError(f"missing clean/snr_0 primary metric for {method}")
        degradation = noise_degradation(
            {primary: aggregate_lookup[clean_key]},
            {primary: aggregate_lookup[noisy_key]},
        )[primary]
        noise_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "metric": primary,
                **degradation,
            }
        )

    gate_rows = []
    for (condition, seed), values in sorted(gate_values.items()):
        if not values:
            continue
        gate_rows.append(
            {
                "dataset": dataset,
                "condition": condition,
                "seed": seed,
                "candidate_count": len(values),
                "mean_gate": math.fsum(values) / len(values),
                "min_gate": min(values),
                "max_gate": max(values),
            }
        )

    write_csv_once(
        args.output_dir / "main_results.csv",
        [
            "dataset",
            "condition",
            "method",
            "metric",
            "seed",
            "value",
            "seed_mean",
            "seed_sample_std",
            "confidence_95_lower",
            "confidence_95_upper",
        ],
        result_rows,
    )
    write_csv_once(
        args.output_dir / "noise_degradation.csv",
        [
            "dataset",
            "method",
            "metric",
            "clean",
            "noisy",
            "absolute_drop",
            "relative_drop",
        ],
        noise_rows,
    )
    write_csv_once(
        args.output_dir / "gate_by_condition.csv",
        [
            "dataset",
            "condition",
            "seed",
            "candidate_count",
            "mean_gate",
            "min_gate",
            "max_gate",
        ],
        gate_rows,
    )
    write_csv_once(
        args.output_dir / "paired_bootstrap.csv",
        [
            "dataset",
            "condition",
            "seed",
            "ours",
            "baseline",
            "mean_difference",
            "confidence_95_lower",
            "confidence_95_upper",
            "two_sided_p_value",
        ],
        significance_rows,
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "dataset": dataset,
        "conditions": conditions,
        "seeds": seeds,
        "cell_count": len(cells),
        "result_row_count": len(result_rows),
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
            for path in dict.fromkeys(input_paths)
        },
    }
    with (args.output_dir / "summary_manifest.json").open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as stream:
        json.dump(
            manifest,
            stream,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
