#!/usr/bin/env python3
"""Render currently reproduced cells for paper Tables 2/3/12-15."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPOSITORY_ROOT / "results/tables/reproduction_summary.csv"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "results/tables/oea_partial_tables.md"
TARGET_TABLES = ("Table 2", "Table 3", "Table 12", "Table 13", "Table 14", "Table 15")
METRIC_ORDER = ("R@1", "R@5", "R@10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


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


def protocol_label(experiment: str, table: str) -> str:
    if "default_joint_all_captions" in experiment:
        return "[CODE] all-caption"
    if "t2a_only_seed0" in experiment:
        return "[CODE] seed-0 one-caption"
    if "default_seed0" in experiment:
        return "[CODE] seed-0 one-caption/self-exclusion"
    if "all_captions_sensitivity" in experiment:
        return "[INFERRED] all-caption sensitivity"
    if table in {"Table 12", "Table 13", "Table 14", "Table 15"}:
        return "[CODE] released positive UIQ"
    return experiment


def selected_groups(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[tuple[str, str, str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["paper_table"] not in TARGET_TABLES or row["status"] not in {
            "exact",
            "close",
            "trend_reproduced",
        }:
            continue
        key = (
            row["paper_table"],
            row["model"],
            row["dataset"],
            row["task"],
            row["experiment"],
        )
        if row["metric"] in grouped[key]:
            raise ValueError(f"duplicate metric within reproduction protocol: {key} / {row['metric']}")
        grouped[key][row["metric"]] = row

    result: list[dict[str, Any]] = []
    for key, metrics in grouped.items():
        if set(metrics) != set(METRIC_ORDER):
            raise ValueError(f"incomplete R@1/5/10 protocol group: {key}")
        table, model, dataset, task, experiment = key
        result.append(
            {
                "paper_table": table,
                "model": model,
                "dataset": dataset,
                "task": task,
                "experiment": experiment,
                "protocol": protocol_label(experiment, table),
                "paper": [metrics[name]["paper_value"] for name in METRIC_ORDER],
                "reproduced": [metrics[name]["reproduced_value"] for name in METRIC_ORDER],
                "max_absolute_delta": max(
                    float(metrics[name]["absolute_delta"]) for name in METRIC_ORDER
                ),
                "checkpoint": metrics["R@1"]["checkpoint"],
            }
        )
    result.sort(
        key=lambda row: (
            int(row["paper_table"].split()[1]),
            row["model"],
            row["dataset"],
            row["experiment"],
        )
    )
    return result


def triple(values: list[str]) -> str:
    return " / ".join(values)


def markdown_text(groups: list[dict[str, Any]]) -> str:
    lines = [
        "# Partial Paper Tables",
        "",
        "This view includes only committed value evidence for paper Tables 2, 3 and 12-15. "
        "It excludes all extension experiments. Multiple predeclared protocols remain separate; "
        "no protocol is selected after observing proximity to the paper value.",
        "",
    ]
    by_table: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_table[group["paper_table"]].append(group)
    for table in TARGET_TABLES:
        lines.extend(
            [
                f"## {table}",
                "",
                "| Model | Dataset | Protocol | Paper R@1 / R@5 / R@10 | Reproduced R@1 / R@5 / R@10 | Max abs delta (pp) |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for group in by_table.get(table, []):
            lines.append(
                "| {model} | {dataset} | {protocol} | {paper} | {reproduced} | {delta:.6f} |".format(
                    model=group["model"],
                    dataset=group["dataset"],
                    protocol=group["protocol"],
                    paper=triple(group["paper"]),
                    reproduced=triple(
                        [f"{float(value):.4f}" for value in group["reproduced"]]
                    ),
                    delta=group["max_absolute_delta"],
                )
            )
        if not by_table.get(table):
            lines.append("| [MISSING] | [MISSING] | [MISSING] | - | - | - |")
        lines.append("")
    lines.extend(
        [
            "## Boundaries",
            "",
            "- [PAPER] Clotho caption selection is not specified.",
            "- [PAPER] T2T self-exclusion and tie handling are not fully specified.",
            "- [CODE] The public audio path omits the `passage:` prefix stated by the paper.",
            "- [OBSERVED] Nemo3B (+Cl) T2T and all four released positive-UIQ protocols are listed from completed suites.",
            "",
        ]
    )
    return "\n".join(lines)


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
    groups = selected_groups(args.input)
    rendered = markdown_text(groups)
    audit_path = args.audit_output or args.output.with_suffix(".audit.json")
    if args.check:
        if args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"partial tables are stale: {args.output}")
        return 0
    atomic_write(args.output, rendered)
    audit = {
        "schema_version": 1,
        "status": "complete",
        "scope": "committed paper Tables 2, 3 and 12-15 only; extension experiments excluded",
        "protocol_group_count": len(groups),
        "table_protocol_counts": {
            table: sum(group["paper_table"] == table for group in groups)
            for table in TARGET_TABLES
        },
        "input": file_identity(args.input),
        "output": file_identity(args.output),
    }
    atomic_write(audit_path, json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
