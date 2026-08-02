#!/usr/bin/env python3
"""Build controlled Table 11 means from the fixed Table 2/3 audit artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = REPOSITORY_ROOT / "results/audits"
PAPER_REGISTRY = REPOSITORY_ROOT / "configs/results/paper_reported_metrics.jsonl"
DEFAULT_JSON_OUTPUT = (
    AUDIT_DIR / "table11_public_controlled_means_eval_20260803.json"
)
DEFAULT_CSV_OUTPUT = REPOSITORY_ROOT / "results/tables/table11_public_controlled_means.csv"
METRICS = ("R@1", "R@5", "R@10")
DATASETS = ("AudioCaps", "Clotho", "MECAT")
TASK_PROTOCOLS = {
    "T2A": "t2a_public_code_default_joint_all_captions",
    "T2T": "t2t_public_code_default_seed0",
}
MECAT_TASK_PROTOCOLS = {
    "T2A": "t2a_public848_short_all",
    "T2T": "t2t_public848_short_seed0",
}
MODELS = (
    "LAION-CLAP",
    "Robust-CLAP",
    "MGA-CLAP",
    "M2D-CLAP",
    "Nemotron-3B",
    "Qwen2.5-Omni-3B",
    "Qwen2.5-Omni-7B",
    "OEA-Nemo3B",
    "OEA-Nemo3B (+Cl)",
    "OEA-Qwen3B",
    "OEA-Qwen3B (+Cl)",
    "OEA-Qwen7B",
    "OEA-Qwen7B (+Cl)",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def protocol_metrics(protocols: dict[str, Any], task: str) -> dict[str, float]:
    protocol = protocols[TASK_PROTOCOLS[task]]
    if isinstance(protocol, list):
        return dict(zip(METRICS, map(float, protocol), strict=True))
    metrics = protocol["metrics"]
    return {metric: float(metrics[metric]) for metric in METRICS}


def comparison_metrics(comparison: dict[str, Any], task: str) -> dict[str, float]:
    values = comparison[MECAT_TASK_PROTOCOLS[task]]["reproduced"]
    return dict(zip(METRICS, map(float, values), strict=True))


def collect_inputs() -> tuple[dict[str, dict[str, dict[str, dict[str, float]]]], list[Path]]:
    values: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        model: {} for model in MODELS
    }
    sources: list[Path] = []

    def add(model: str, dataset: str, protocols: dict[str, Any], tasks=("T2A", "T2T")) -> None:
        values[model].setdefault(dataset, {})
        for task in tasks:
            values[model][dataset][task] = protocol_metrics(protocols, task)

    audio_batch_path = AUDIT_DIR / "audiocaps_remaining_seven_main_eval_20260802.json"
    audio_batch = read_json(audio_batch_path)
    sources.append(audio_batch_path)
    for run in audio_batch["runs"]:
        add(run["paper_model"], "AudioCaps", run["protocols"])

    audio_files = (
        "laion_clap_audiocaps_main_eval_20260802.json",
        "mga_clap_audiocaps_main_eval_20260802.json",
        "m2d_clap_audiocaps_main_eval_20260802.json",
        "oea_qwen7b_audiocaps_main_eval_20260802.json",
        "oea_qwen7b_cl_audiocaps_main_eval_20260802.json",
    )
    for filename in audio_files:
        path = AUDIT_DIR / filename
        audit = read_json(path)
        sources.append(path)
        add(audit["model"]["paper_name"], "AudioCaps", audit["protocols"])

    robust_path = AUDIT_DIR / "robust_clap_audiocaps_mecat_eval_20260802.json"
    robust = read_json(robust_path)
    sources.append(robust_path)
    add("Robust-CLAP", "AudioCaps", robust["audiocaps"]["protocols"])

    clotho_files = (
        "laion_clap_clotho_main_eval_20260730.json",
        "mga_clap_clotho_main_eval_20260731.json",
        "m2d_clap_clotho_main_eval_20260730.json",
        "robust_clap_clotho_main_eval_20260802.json",
        "vanilla_nemotron_3b_clotho_main_eval_20260730.json",
        "vanilla_qwen2_5_omni_3b_clotho_main_eval_20260730.json",
        "vanilla_qwen2_5_omni_7b_clotho_main_eval_20260730.json",
        "oea_qwen7b_ac_clotho_official_source_eval_20260730.json",
        "oea_qwen7b_cl_clotho_official_source_eval_20260730.json",
    )
    for filename in clotho_files:
        path = AUDIT_DIR / filename
        audit = read_json(path)
        sources.append(path)
        add(audit["model"]["paper_name"], "Clotho", audit["protocols"])

    clotho_nested = (
        (
            "oea_nemo3b_ac_clotho_official_source_eval_20260729.json",
            "OEA-Nemo3B",
            ("final_suite", "protocols"),
        ),
        (
            "qwen3b_clotho_retrieval_suite_20260720.json",
            "OEA-Qwen3B",
            ("retrieval_suite", "protocols"),
        ),
        (
            "qwen3b_cl_clotho_official_eval_20260719.json",
            "OEA-Qwen3B (+Cl)",
            ("retrieval_suite", "protocols"),
        ),
    )
    for filename, model, keys in clotho_nested:
        path = AUDIT_DIR / filename
        audit = read_json(path)
        sources.append(path)
        add(model, "Clotho", audit[keys[0]][keys[1]])

    nemo_cl_paths = {
        "T2A": AUDIT_DIR / "asrur_nemo_clotho_t2a_20260727.json",
        "T2T": AUDIT_DIR / "oea_nemo3b_clotho_t2t_20260729.json",
    }
    for task, path in nemo_cl_paths.items():
        audit = read_json(path)
        sources.append(path)
        add(
            "OEA-Nemo3B (+Cl)",
            "Clotho",
            audit["retrieval_suite"]["protocols"],
            tasks=(task,),
        )

    for filename in (
        "mecat_weighted_table2_table3_public848_short_eval_20260802.json",
        "mecat_vanilla_table2_table3_public848_short_eval_20260803.json",
    ):
        path = AUDIT_DIR / filename
        audit = read_json(path)
        sources.append(path)
        for run in audit["runs"]:
            model = run["paper_model"]
            values[model]["MECAT"] = {
                task: comparison_metrics(run["comparison"], task)
                for task in ("T2A", "T2T")
            }

    return values, sorted(set(sources))


def paper_table11() -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    for line in PAPER_REGISTRY.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["paper_table"] != "Table 11":
            continue
        result.setdefault(row["model"], {}).setdefault(row["task"], {})[
            row["metric"]
        ] = float(row["paper_value"])
    return result


def build_audit() -> dict[str, Any]:
    inputs, sources = collect_inputs()
    paper = paper_table11()
    results: list[dict[str, Any]] = []
    for model in MODELS:
        if set(inputs[model]) != set(DATASETS):
            raise ValueError(f"incomplete datasets for {model}: {sorted(inputs[model])}")
        for task in ("T2A", "T2T"):
            reproduced = {
                metric: round(
                    sum(inputs[model][dataset][task][metric] for dataset in DATASETS)
                    / len(DATASETS),
                    6,
                )
                for metric in METRICS
            }
            paper_values = {metric: paper[model][task][metric] for metric in METRICS}
            deltas = {
                metric: round(reproduced[metric] - paper_values[metric], 6)
                for metric in METRICS
            }
            results.append(
                {
                    "model": model,
                    "task": task,
                    "input_values": {
                        dataset: inputs[model][dataset][task] for dataset in DATASETS
                    },
                    "reproduced_mean": reproduced,
                    "paper_mean": paper_values,
                    "signed_delta_percentage_points": deltas,
                    "maximum_absolute_delta_percentage_points": max(
                        abs(value) for value in deltas.values()
                    ),
                }
            )
    return {
        "schema_version": 1,
        "status": "complete",
        "evidence_type": "derived_public_controlled_table11",
        "paper_scope": "DER-01 / Table 11",
        "model_count": len(MODELS),
        "task_count": 2,
        "result_count": len(results),
        "metric_count": len(results) * len(METRICS),
        "source_artifacts": [source_identity(path) for path in sources]
        + [source_identity(PAPER_REGISTRY)],
        "protocol": {
            "datasets": list(DATASETS),
            "t2a": "predeclared default all-caption protocol",
            "t2t": "predeclared default seed0 protocol",
            "mecat": "controlled public 848-row short-caption protocol",
            "aggregation": "unweighted arithmetic mean across AudioCaps, Clotho, and MECAT",
            "no_post_hoc_protocol_selection": True,
        },
        "results": results,
        "claim_boundary": {
            "status": "CONTROLLED_PUBLIC_PROTOCOL_DERIVATION",
            "strict_paper_reproduction": False,
            "reason": "MECAT PAPER uses an unpublished 847-row subset, and PAPER does not publish caption selection or T2T self/tie handling. The controlled means therefore cannot be promoted to strict Table 11 reproduction.",
        },
    }


def csv_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in audit["results"]:
        row: dict[str, Any] = {"model": result["model"], "task": result["task"]}
        for metric in METRICS:
            key = metric.lower().replace("@", "_at_")
            row[f"reproduced_{key}"] = result["reproduced_mean"][metric]
            row[f"paper_{key}"] = result["paper_mean"][metric]
            row[f"delta_pp_{key}"] = result["signed_delta_percentage_points"][metric]
        row["maximum_absolute_delta_pp"] = result[
            "maximum_absolute_delta_percentage_points"
        ]
        row["strict_paper_reproduction"] = False
        rows.append(row)
    return rows


def render_json(audit: dict[str, Any]) -> str:
    return json.dumps(audit, indent=2, ensure_ascii=True) + "\n"


def render_csv(rows: list[dict[str, Any]]) -> str:
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def main() -> int:
    args = parse_args()
    audit = build_audit()
    expected_json = render_json(audit)
    expected_csv = render_csv(csv_rows(audit))
    if args.check:
        if not args.json_output.exists() or not args.csv_output.exists():
            raise SystemExit("Table 11 outputs are missing")
        if args.json_output.read_text(encoding="utf-8") != expected_json:
            raise SystemExit(f"stale output: {args.json_output}")
        if args.csv_output.read_text(encoding="utf-8") != expected_csv:
            raise SystemExit(f"stale output: {args.csv_output}")
        return 0
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(expected_json, encoding="utf-8", newline="\n")
    args.csv_output.write_text(expected_csv, encoding="utf-8", newline="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
