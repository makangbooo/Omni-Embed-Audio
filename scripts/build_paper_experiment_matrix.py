#!/usr/bin/env python3
"""Build the paper experiment x dataset x model completion matrix."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER_REGISTRY = REPOSITORY_ROOT / "configs/results/paper_reported_metrics.jsonl"
DEFAULT_OBSERVATIONS = REPOSITORY_ROOT / "results/observations/reproduction_observations.jsonl"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "results/tables/paper_experiment_matrix.csv"
ALLOWED_STATUSES = {
    "REPRODUCED_CLOSE",
    "CONTROLLED_ONLY",
    "PARTIAL",
    "TODO",
    "BLOCKED",
}
VALUE_STATUSES = {"exact", "close", "trend_reproduced"}
MATRIX_COLUMNS = (
    "inventory_id",
    "paper_table",
    "model",
    "dataset",
    "task",
    "paper_status",
    "reproduction_status",
    "paper_metric_count",
    "observed_metric_count",
    "evidence",
    "assessment",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-registry", type=Path, default=DEFAULT_PAPER_REGISTRY)
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank JSONL line at {path}:{number}")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row must be an object at {path}:{number}")
        rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def inventory_id(row: dict[str, Any]) -> str:
    table = row["paper_table"]
    task = row["task"]
    metric = row["metric"]
    if table == "Table 1":
        return "EXP-02" if row["model"].startswith("Human") else "EXP-03"
    if table == "Table 2":
        return "EXP-10"
    if table == "Table 3":
        return "EXP-11"
    if table == "Table 4":
        return "DER-02"
    if table in {"Table 5", "Table 16"}:
        return "EXP-19" if "parameter" in metric.lower() else "EXP-18"
    if table == "Table 7":
        return "EXP-02"
    if table == "Table 11":
        return "DER-01"
    if table == "Table 12":
        return "EXP-12"
    if table == "Table 13":
        return "EXP-13"
    if table == "Table 14":
        return "EXP-14"
    if table == "Table 15":
        return "EXP-15"
    if table == "Table 17":
        return "EXP-16" if task == "Negative UIQ standard retrieval" else "EXP-17"
    raise ValueError(f"unmapped paper row: {table} / {task} / {metric}")


def evidence_paths(observations: Iterable[dict[str, Any]]) -> str:
    paths = sorted(
        {
            observation["evidence"]["path"]
            for observation in observations
            if isinstance(observation.get("evidence"), dict)
            and isinstance(observation["evidence"].get("path"), str)
        }
    )
    return ";".join(paths)


def derived_input_is_partial(model: str, task: str) -> bool:
    if model in {"OEA-Qwen3B", "OEA-Qwen3B (+Cl)"}:
        return True
    return model == "OEA-Nemo3B (+Cl)" and task == "T2A"


def assess_cell(
    key: tuple[str, str, str, str, str],
    metric_ids: set[str],
    observations_by_metric: dict[str, list[dict[str, Any]]],
) -> tuple[str, int, str, str]:
    inventory, table, model, dataset, task = key
    observations = [
        observation
        for metric_id in metric_ids
        for observation in observations_by_metric.get(metric_id, [])
    ]
    observed_value_metrics = {
        observation["paper_metric_id"]
        for observation in observations
        if observation.get("status") in VALUE_STATUSES
        and observation.get("reproduced_value") is not None
    }
    if observed_value_metrics:
        status = (
            "REPRODUCED_CLOSE"
            if observed_value_metrics == metric_ids
            else "PARTIAL"
        )
        return (
            status,
            len(observed_value_metrics),
            evidence_paths(observations),
            "Predeclared public-code or sensitivity protocol has value evidence; missing paper protocol remains separately blocked.",
        )

    if table in {"Table 5", "Table 16"} and model == "OEA-Qwen3B":
        return (
            "CONTROLLED_ONLY",
            0,
            "results/audits/qwen3b_clotho_a100_efficiency_20260721.json",
            "A100 controlled measurement exists for the released +Cl checkpoint, but PAPER omits timing, memory, parameter-count, and checkpoint-variant definitions.",
        )

    if table in {"Table 1", "Table 7"}:
        return (
            "BLOCKED",
            0,
            "docs/paper_experiment_inventory.md",
            "Original sample IDs, ratings or closed-model responses are not published.",
        )

    if table == "Table 17" or (
        table == "Table 4" and "Negative UIQ" in task
    ):
        if model == "M2D-CLAP":
            return (
                "CONTROLLED_ONLY",
                0,
                "results/audits/m2d_clap_negative_uiq_eval_20260803.json",
                "All three public dataset runs are complete under deterministic inferred caption-identity pairing; PAPER does not publish strict target-HN audio IDs.",
            )
        if model in {"LAION-CLAP", "MGA-CLAP", "Robust-CLAP"}:
            checkpoint_boundary = (
                " Robust-CLAP also lacks a verifiable PAPER checkpoint identity."
                if model == "Robust-CLAP"
                else ""
            )
            return (
                "CONTROLLED_ONLY",
                0,
                "results/audits/clap_negative_uiq_eval_20260803.json",
                "All three public dataset runs are complete under deterministic inferred caption-identity pairing; PAPER does not publish strict target-HN audio IDs."
                + checkpoint_boundary,
            )
        if model in {
            "OEA-Nemo3B",
            "OEA-Nemo3B (+Cl)",
            "OEA-Qwen3B",
            "OEA-Qwen3B (+Cl)",
            "OEA-Qwen7B",
            "OEA-Qwen7B (+Cl)",
        }:
            return (
                "CONTROLLED_ONLY",
                0,
                "results/audits/oea_negative_uiq_eval_20260803.json",
                "All three public dataset runs are complete under deterministic inferred "
                "caption-identity pairing; PAPER does not publish strict target-HN audio "
                "IDs and MECAT uses public 848 rather than PAPER 847 candidates.",
            )
        return (
            "BLOCKED",
            0,
            "docs/paper_experiment_inventory.md",
            "Released negative JSONL omits hard-negative audio IDs; strict target-HN pairing cannot be reconstructed without author evidence.",
        )

    if table in {"Table 5", "Table 16"}:
        return (
            "BLOCKED",
            0,
            "docs/evaluation_protocol.md",
            "Strict efficiency/counting protocol is unpublished and no controlled run exists for this model.",
        )

    if table == "Table 11":
        status = "PARTIAL" if derived_input_is_partial(model, task) else "TODO"
        return (
            status,
            0,
            "results/tables/reproduction_summary.csv" if status == "PARTIAL" else "",
            "Only Clotho inputs exist; all three datasets are required for the paper mean."
            if status == "PARTIAL"
            else "No reproduced input cell is available for this three-dataset mean.",
        )

    if table == "Table 4":
        if model in {"OEA-Qwen3B", "OEA-Qwen3B (+Cl)"}:
            return (
                "PARTIAL",
                0,
                "results/tables/reproduction_summary.csv",
                "Clotho positive-UIQ inputs exist, but AudioCaps, MECAT and/or negative inputs required by the aggregate are missing.",
            )
        return (
            "TODO",
            0,
            "",
            "No reproduced three-dataset UIQ inputs are available for this model.",
        )

    if (
        table == "Table 3"
        and model == "OEA-Nemo3B (+Cl)"
        and dataset == "Clotho"
    ):
        return (
            "PARTIAL",
            0,
            "results/audits/asrur_nemo_g1_full_embeddings_20260727.json",
            "Complete 1,045 audio and 5,225 caption embeddings exist; CPU T2T finalization is pending remote artifact access.",
        )

    if (
        table in {"Table 12", "Table 13", "Table 14", "Table 15"}
        and model == "OEA-Nemo3B (+Cl)"
        and dataset == "Clotho"
    ):
        return (
            "PARTIAL",
            0,
            "results/audits/asrur_nemo_g1_full_embeddings_20260727.json",
            "Candidate audio bank and checkpoint are complete; the corresponding UIQ query embeddings/evaluation are pending.",
        )

    if dataset == "MECAT":
        return (
            "BLOCKED",
            0,
            "docs/mecat_data_audit.md" if (REPOSITORY_ROOT / "docs/mecat_data_audit.md").exists() else "docs/paper_experiment_inventory.md",
            "PAPER uses 847 items while the public release has 848; excluded ID and caption construction are missing.",
        )

    return (
        "TODO",
        0,
        "",
        "Paper value is transcribed, but no qualifying reproduction observation is committed for this cell.",
    )


def build_matrix(
    paper_rows: list[dict[str, Any]], observations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    observations_by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        observations_by_metric[observation["paper_metric_id"]].append(observation)

    grouped: dict[tuple[str, str, str, str, str], set[str]] = defaultdict(set)
    for row in paper_rows:
        key = (
            inventory_id(row),
            row["paper_table"],
            row["model"],
            row["dataset"],
            row["task"],
        )
        grouped[key].add(row["paper_metric_id"])

    matrix: list[dict[str, Any]] = []
    for key, metric_ids in grouped.items():
        status, observed_count, evidence, assessment = assess_cell(
            key, metric_ids, observations_by_metric
        )
        if status not in ALLOWED_STATUSES:
            raise AssertionError(f"unsupported matrix status: {status}")
        inventory, table, model, dataset, task = key
        matrix.append(
            {
                "inventory_id": inventory,
                "paper_table": table,
                "model": model,
                "dataset": dataset,
                "task": task,
                "paper_status": "PAPER",
                "reproduction_status": status,
                "paper_metric_count": len(metric_ids),
                "observed_metric_count": observed_count,
                "evidence": evidence,
                "assessment": assessment,
            }
        )
    matrix.sort(
        key=lambda row: (
            int(row["paper_table"].split()[1]),
            row["inventory_id"],
            row["model"],
            row["dataset"],
            row["task"],
        )
    )
    return matrix


def csv_text(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=MATRIX_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    args = parse_args()
    paper_rows = read_jsonl(args.paper_registry)
    observations = read_jsonl(args.observations)
    rows = build_matrix(paper_rows, observations)
    rendered = csv_text(rows)
    audit_path = args.audit_output or args.output.with_suffix(".audit.json")
    if args.check:
        if args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"matrix output is stale: {args.output}")
        return 0
    atomic_write(args.output, rendered)
    audit = {
        "schema_version": 1,
        "status": "complete",
        "scope": "committed evidence snapshot; remote shared artifacts not revalidated",
        "paper_cell_count": len(rows),
        "paper_metric_count": sum(int(row["paper_metric_count"]) for row in rows),
        "status_counts": dict(sorted(Counter(row["reproduction_status"] for row in rows).items())),
        "inputs": {
            "paper_registry": file_identity(args.paper_registry),
            "observations": file_identity(args.observations),
        },
        "output": file_identity(args.output),
    }
    atomic_write(audit_path, json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
