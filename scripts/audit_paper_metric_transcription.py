#!/usr/bin/env python3
"""Cross-check every configured paper table cell against PDF-extracted text."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.build_paper_metric_registry import (
    DEFAULT_OUTPUT,
    DEFAULT_SOURCE,
    atomic_write_text,
    expand_registry,
    registry_text,
)


DEFAULT_AUDIT_OUTPUT = (
    REPOSITORY_ROOT / "results/paper_audits/paper_metric_transcription.json"
)
NUMBER_PATTERN = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--registry", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_AUDIT_OUTPUT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        display_path = resolved.name
    return {
        "path": display_path,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def normalized_text(value: str) -> str:
    return "".join(value.split())


def table_row_label(row: Mapping[str, Any]) -> str:
    displayed_label = row.get("displayed_label")
    if isinstance(displayed_label, str):
        return displayed_label
    model = row.get("model")
    if isinstance(model, str):
        return model
    task = row.get("task")
    if not isinstance(task, str):
        raise ValueError("table row has neither model nor task label")
    if not task.startswith("UIQ ") or not task.endswith(" validity"):
        raise ValueError(f"cannot derive displayed task label: {task}")
    return task[len("UIQ ") : -len(" validity")]


def displayed_values(row: Mapping[str, Any]) -> list[str]:
    values = row.get("values")
    if not isinstance(values, list):
        raise ValueError("table row values must be a list")
    result: list[str] = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("value")
        if not isinstance(value, str):
            raise ValueError("table row value is not a displayed string")
        result.append(value)
    return result


def line_matches_row(line: str, label: str, expected_values: list[str]) -> bool:
    normalized_line = normalized_text(line)
    normalized_label = normalized_text(label)
    if normalized_label not in normalized_line:
        return False
    observed_values = NUMBER_PATTERN.findall(line)
    width = len(expected_values)
    return any(
        observed_values[start : start + width] == expected_values
        for start in range(len(observed_values) - width + 1)
    )


def audit_transcription(
    pdf_path: Path,
    source_path: Path,
    registry_path: Path,
) -> dict[str, Any]:
    try:
        import pdfplumber
    except ImportError as error:
        raise RuntimeError(
            "pdfplumber is required only for the local PDF transcription audit"
        ) from error

    config = json.loads(source_path.read_text(encoding="utf-8"))
    expanded = expand_registry(config)
    if registry_path.read_text(encoding="utf-8") != registry_text(expanded):
        raise ValueError("generated registry is stale before PDF audit")
    expected_pdf_hash = config["paper_pdf_sha256"]
    actual_pdf_hash = sha256_file(pdf_path)
    if actual_pdf_hash != expected_pdf_hash:
        raise ValueError(
            f"paper PDF SHA256 mismatch: {actual_pdf_hash} != {expected_pdf_hash}"
        )

    with pdfplumber.open(pdf_path) as pdf:
        page_lines = {
            page_number: (pdf.pages[page_number - 1].extract_text(layout=True) or "").splitlines()
            for page_number in {
                int(table["pdf_page"])
                for table in config["numeric_result_tables"]
            }
        }
        page_count = len(pdf.pages)

    table_reports: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    matched_cells = 0
    for table in config["numeric_result_tables"]:
        table_name = table["paper_table"]
        page_number = int(table["pdf_page"])
        lines = page_lines[page_number]
        row_reports: list[dict[str, Any]] = []
        for row_index, row in enumerate(table["rows"]):
            label = table_row_label(row)
            values = displayed_values(row)
            matching_line_numbers = [
                line_number
                for line_number, line in enumerate(lines, start=1)
                if line_matches_row(line, label, values)
            ]
            row_report = {
                "row_index": row_index,
                "displayed_label": label,
                "value_count": len(values),
                "matching_line_numbers": matching_line_numbers,
                "status": "matched" if matching_line_numbers else "unmatched",
            }
            row_reports.append(row_report)
            if matching_line_numbers:
                matched_cells += len(values)
            else:
                unmatched.append(
                    {
                        "paper_table": table_name,
                        "pdf_page": page_number,
                        "row_index": row_index,
                        "displayed_label": label,
                        "expected_values": values,
                    }
                )
        table_reports.append(
            {
                "paper_table": table_name,
                "pdf_page": page_number,
                "row_count": len(row_reports),
                "metric_cell_count": sum(
                    row["value_count"] for row in row_reports
                ),
                "matched_row_count": sum(
                    row["status"] == "matched" for row in row_reports
                ),
                "rows": row_reports,
            }
        )

    report = {
        "schema_version": 1,
        "status": "complete" if not unmatched else "failed",
        "method": (
            "Each configured numeric row label and its complete displayed value "
            "sequence must occur on one pdfplumber layout-extracted line from the "
            "pinned PDF page; extracted copyrighted table text is not retained."
        ),
        "paper_pdf": file_identity(pdf_path),
        "paper_page_count": page_count,
        "source": file_identity(source_path),
        "registry": file_identity(registry_path),
        "numeric_table_count": len(table_reports),
        "metric_count": len(expanded),
        "matched_metric_cell_count": matched_cells,
        "unmatched_metric_cell_count": len(expanded) - matched_cells,
        "unit_counts": dict(
            sorted(Counter(row["unit"] for row in expanded).items())
        ),
        "tables": sorted(
            table_reports,
            key=lambda item: int(item["paper_table"].split()[1]),
        ),
        "unmatched_rows": unmatched,
    }
    return report


def main() -> int:
    args = parse_args()
    report = audit_transcription(
        args.pdf.resolve(), args.source.resolve(), args.registry.resolve()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        args.output,
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    print(
        f"[INFO] Matched {report['matched_metric_cell_count']} / "
        f"{report['metric_count']} metric cells: {args.output}"
    )
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
