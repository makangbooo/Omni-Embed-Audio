#!/usr/bin/env python3
"""Build the committed OEA Table 4/17 audit from compact remote output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER_REGISTRY = REPOSITORY_ROOT / "configs/results/paper_reported_metrics.jsonl"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "results/audits/oea_negative_uiq_eval_20260803.json"
VARIANTS = (
    "oea_nemo3b",
    "oea_nemo3b_cl",
    "oea_qwen3b",
    "oea_qwen3b_cl",
    "oea_qwen7b",
    "oea_qwen7b_cl",
)
METRICS = ("R@5", "R@10", "Delta-Rank", "HNSR", "HNSR@10", "TFR", "TFR-HN@10")
EXPECTED_COUNTS = {
    "audiocaps": (975, 630),
    "clotho": (1045, 542),
    "mecat": (848, 409),
}
CHECKPOINTS = {
    "oea_nemo3b": (
        "step_400_best.pt",
        "55579dfbd4f6621b5d842c5e731d6a1d37dbfd04b26b1c55bf8cdea980e67d25",
    ),
    "oea_nemo3b_cl": (
        "step_450_best_inference_only.pt",
        "2a5bee9039a28c0028cf205d1e2f4302fda540dd913f4cc1b08301edfc6680c4",
    ),
    "oea_qwen3b": (
        "step_350_inference_only.pt",
        "b1d0f559711b70f5a80dbdeb8cd46d80ed7b38b8e5524b9a871878d8bcd5f101",
    ),
    "oea_qwen3b_cl": (
        "step_40_inference_only.pt",
        "f084bf3c3ad645809e4c7e22cf148caef788cb96c019729576b881291268432a",
    ),
    "oea_qwen7b": (
        "step_300.pt",
        "cd751e3a71f0b47b9ecbbc0f5a11e4673097f0ebdbd80f610387f5778dd70a46",
    ),
    "oea_qwen7b_cl": (
        "step_330.pt",
        "09ce41af7e7ac23106fa74d25e45b7229a046d687774cbd4ab2c00c05097d92e",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-log", type=Path, required=True)
    parser.add_argument("--paper-registry", type=Path, default=DEFAULT_PAPER_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_results(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    records: dict[str, dict[str, Any]] = {}
    metadata: dict[str, str] = {}
    in_compact_output = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "COMPACT_OUTPUT_START":
            in_compact_output = True
            continue
        if line == "COMPACT_OUTPUT_END":
            in_compact_output = False
            continue
        if in_compact_output and line.startswith("OEA_NEGATIVE_RESULT="):
            record = json.loads(line.split("=", 1)[1])
            records[record["variant"]] = record
        elif in_compact_output and "=" in line:
            key, value = line.split("=", 1)
            metadata[key] = value
    if tuple(records) != VARIANTS:
        raise ValueError(f"expected variants {VARIANTS}, observed {tuple(records)}")
    if metadata.get("RUN_RC") != "0" or metadata.get("MATRIX_STATUS") != "complete":
        raise ValueError(f"batch did not complete: {metadata}")
    if metadata.get("COMPLETED_VARIANTS", "").split() != list(VARIANTS):
        raise ValueError(f"completed variant mismatch: {metadata}")
    return records, metadata


def read_paper_values(path: Path) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["paper_table"] != "Table 17" or not row["model"].startswith("OEA-"):
            continue
        metric = "TFR-HN@10" if row["metric"] == "TFR@10" else row["metric"]
        values.setdefault(row["model"], {})[metric] = float(row["paper_value"])
    return values


def validate_record(record: dict[str, Any]) -> None:
    if record["status"] != "complete":
        raise ValueError(f"incomplete record: {record['variant']}")
    if set(record["datasets"]) != set(EXPECTED_COUNTS):
        raise ValueError(f"dataset mismatch: {record['variant']}")
    for dataset, row in record["datasets"].items():
        if (row["candidate_count"], row["query_count"]) != EXPECTED_COUNTS[dataset]:
            raise ValueError(f"count mismatch: {record['variant']} / {dataset}")
        if set(row["metrics"]) != set(METRICS):
            raise ValueError(f"metric mismatch: {record['variant']} / {dataset}")
        for field in (
            "metrics_sha256",
            "audio_npz_sha256",
            "query_npz_sha256",
            "pairing_sha256",
        ):
            if len(row[field]) != 64:
                raise ValueError(f"invalid {field}: {record['variant']} / {dataset}")


def build_audit(compact_log: Path, registry: Path) -> dict[str, Any]:
    records, metadata = read_results(compact_log)
    paper_values = read_paper_values(registry)
    models: dict[str, dict[str, Any]] = {}
    commits = set()
    for variant in VARIANTS:
        record = records[variant]
        validate_record(record)
        commits.add(record["git_commit"])
        paper = paper_values[record["model"]]
        means = record["three_dataset_mean"]
        deltas = {metric: means[metric] - paper[metric] for metric in METRICS}
        maximum_metric = max(METRICS, key=lambda metric: abs(deltas[metric]))
        checkpoint, checkpoint_sha256 = CHECKPOINTS[variant]
        models[record["model"]] = {
            "variant": variant,
            "resource_binding": {
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_sha256,
                "strict_paper_checkpoint_identity": True,
            },
            "run_id": Path(record["run"]).name,
            "artifact_manifest_sha256": record["artifact_manifest_sha256"],
            "datasets": record["datasets"],
            "three_dataset_mean": means,
            "paper_values": paper,
            "comparison": {
                "signed_delta": deltas,
                "maximum_absolute_delta": abs(deltas[maximum_metric]),
                "maximum_absolute_delta_metric": maximum_metric,
                "assessment": "CONTROLLED_COMPLETE_STRICT_PAPER_PROTOCOL_UNAVAILABLE",
            },
        }
    if len(commits) != 1:
        raise ValueError(f"expected one git commit, observed {sorted(commits)}")
    return {
        "schema_version": 1,
        "status": "complete",
        "evidence_type": (
            "controlled_remote_oea_negative_uiq_six_model_three_dataset_audit"
        ),
        "paper_scope": [
            "EXP-16 / Table 17 / six OEA models / Negative UIQ standard retrieval",
            "EXP-17 / Tables 4 and 17 / six OEA models / hard-negative discrimination",
        ],
        "execution": {
            "git_commit": commits.pop(),
            "git_status_short": "",
            "gpu": "NVIDIA GeForce RTX 4090",
            "model_run_count": 6,
            "dataset_run_count": 18,
            "complete_model_run_count": 6,
            "failed_run_count": 0,
            "run_rc": int(metadata["RUN_RC"]),
            "matrix_status": metadata["MATRIX_STATUS"],
            "batch_log": metadata["BATCH_LOG"],
            "compact_log_sha256": sha256(compact_log),
        },
        "official_source": {
            "oea_official_source_used": True,
            "source_files": [
                "AudioRetrieval/preprocessing/embeddings/oea.py",
                "AudioRetrieval/preprocessing/embeddings/uiq_text.py",
                "AudioRetrieval/evaluation/negative_canonical.py",
                "AudioRetrieval/evaluation/negative_metrics.py",
                "scripts/evaluate_negative_uiq_npz.py",
                "scripts/run_oea_negative_uiq.sh",
            ],
        },
        "protocol": {
            "source": "INFERRED_DETERMINISTIC_RELEASED_CAPTION_IDENTITY",
            "strict_paper_reproduction": False,
            "pairing_audit_run": "negative_uiq_exact_pairing_audit_20260803_125256",
            "tie_policy": "optimistic_strict_greater",
            "normalization_applied": True,
            "aggregation": "unweighted arithmetic mean of the three dataset percentages",
            "paper_metric_mapping": {"TFR@10": "implementation TFR-HN@10"},
            "claim_boundary": (
                "The released negative JSONL files omit explicit hard-negative audio IDs. "
                "The controlled deterministic caption-identity pairing is complete and uses "
                "no embedding or nearest-neighbor inference, but it is not claimed as the "
                "unpublished strict PAPER pairing. MECAT uses all 848 public candidates "
                "while PAPER reports 847."
            ),
        },
        "models": models,
    }


def main() -> int:
    args = parse_args()
    audit = build_audit(args.compact_log.resolve(), args.paper_registry.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
