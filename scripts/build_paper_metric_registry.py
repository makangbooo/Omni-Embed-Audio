#!/usr/bin/env python3
"""Expand the visually audited paper table matrices into metric JSONL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPOSITORY_ROOT / "configs/results/paper_reported_tables.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "configs/results/paper_reported_metrics.jsonl"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ID_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
TABLE_PATTERN = re.compile(r"^Table ([1-9][0-9]*)$")
NUMERIC_PATTERN = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def identifier(value: Any, label: str) -> str:
    result = nonempty_string(value, label)
    if ID_PATTERN.fullmatch(result) is None:
        raise ValueError(f"{label} is not a canonical identifier: {result}")
    return result


def table_number(value: Any, label: str) -> int:
    table = nonempty_string(value, label)
    match = TABLE_PATTERN.fullmatch(table)
    if match is None:
        raise ValueError(f"{label} is invalid: {table}")
    return int(match.group(1))


def positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def resolve_axis_field(
    row: Mapping[str, Any],
    dimension: Mapping[str, Any],
    table: Mapping[str, Any],
    field: str,
    label: str,
) -> str:
    for source in (row, dimension, table):
        if field in source:
            return nonempty_string(source[field], f"{label}.{field}")
    raise ValueError(f"{label}: unresolved field: {field}")


def resolve_axis_id(
    row: Mapping[str, Any],
    dimension: Mapping[str, Any],
    table: Mapping[str, Any],
    field: str,
    label: str,
) -> str:
    for source in (row, dimension, table):
        if field in source:
            return identifier(source[field], f"{label}.{field}")
    raise ValueError(f"{label}: unresolved field: {field}")


def parse_displayed_value(value: Any, label: str) -> tuple[str, str | None]:
    qualifier: str | None = None
    if isinstance(value, dict):
        unexpected = sorted(set(value) - {"value", "qualifier"})
        if unexpected:
            raise ValueError(f"{label}: unsupported value fields: {unexpected}")
        qualifier = nonempty_string(value.get("qualifier"), f"{label}.qualifier")
        value = value.get("value")
    if not isinstance(value, str) or NUMERIC_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must preserve a displayed numeric string")
    return value, qualifier


def expand_registry(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    if config.get("schema_version") != 1:
        raise ValueError("unsupported paper table source schema_version")
    pdf_hash = config.get("paper_pdf_sha256")
    if not isinstance(pdf_hash, str) or SHA256_PATTERN.fullmatch(pdf_hash) is None:
        raise ValueError("paper_pdf_sha256 must be lowercase SHA256")
    expected_tables = config.get("expected_paper_tables")
    if expected_tables != list(range(1, 18)):
        raise ValueError("expected_paper_tables must be exactly 1 through 17")
    tables = config.get("numeric_result_tables")
    excluded = config.get("non_metric_tables")
    if not isinstance(tables, list) or not isinstance(excluded, list):
        raise ValueError("numeric_result_tables and non_metric_tables must be lists")

    numeric_numbers: list[int] = []
    rows_out: list[dict[str, Any]] = []
    metric_ids: set[str] = set()
    for table_index, table in enumerate(tables):
        label = f"numeric_result_tables[{table_index}]"
        if not isinstance(table, dict):
            raise ValueError(f"{label} must be an object")
        number = table_number(table.get("paper_table"), f"{label}.paper_table")
        numeric_numbers.append(number)
        pdf_page = positive_integer(table.get("pdf_page"), f"{label}.pdf_page")
        note = nonempty_string(table.get("note"), f"{label}.note")
        dimensions = table.get("dimensions")
        rows = table.get("rows")
        if not isinstance(dimensions, list) or not dimensions:
            raise ValueError(f"{label}.dimensions must be a non-empty list")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{label}.rows must be a non-empty list")
        for dimension_index, dimension in enumerate(dimensions):
            dimension_label = f"{label}.dimensions[{dimension_index}]"
            if not isinstance(dimension, dict):
                raise ValueError(f"{dimension_label} must be an object")
            nonempty_string(dimension.get("metric"), f"{dimension_label}.metric")
            identifier(dimension.get("metric_id"), f"{dimension_label}.metric_id")
            nonempty_string(dimension.get("unit"), f"{dimension_label}.unit")
            sort_order = dimension.get("sort_order")
            if sort_order is not None:
                positive_integer(sort_order, f"{dimension_label}.sort_order")
        for row_index, row in enumerate(rows):
            row_label = f"{label}.rows[{row_index}]"
            if not isinstance(row, dict):
                raise ValueError(f"{row_label} must be an object")
            values = row.get("values")
            if not isinstance(values, list) or len(values) != len(dimensions):
                raise ValueError(
                    f"{row_label}.values must match dimension count {len(dimensions)}"
                )
            for dimension_index, (dimension, raw_value) in enumerate(
                zip(dimensions, values)
            ):
                cell_label = f"{row_label}.values[{dimension_index}]"
                model = resolve_axis_field(row, dimension, table, "model", cell_label)
                model_id = resolve_axis_id(
                    row, dimension, table, "model_id", cell_label
                )
                dataset = resolve_axis_field(
                    row, dimension, table, "dataset", cell_label
                )
                dataset_id = resolve_axis_id(
                    row, dimension, table, "dataset_id", cell_label
                )
                task = resolve_axis_field(row, dimension, table, "task", cell_label)
                task_id = resolve_axis_id(
                    row, dimension, table, "task_id", cell_label
                )
                metric = nonempty_string(
                    dimension.get("metric"), f"{cell_label}.metric"
                )
                metric_id = identifier(
                    dimension.get("metric_id"), f"{cell_label}.metric_id"
                )
                unit = nonempty_string(
                    dimension.get("unit"), f"{cell_label}.unit"
                )
                paper_value, qualifier = parse_displayed_value(raw_value, cell_label)
                paper_metric_id = (
                    f"table{number}_{model_id}_{dataset_id}_{task_id}_{metric_id}"
                )
                if paper_metric_id in metric_ids:
                    raise ValueError(f"duplicate paper_metric_id: {paper_metric_id}")
                metric_ids.add(paper_metric_id)
                output_row: dict[str, Any] = {
                    "paper_metric_id": paper_metric_id,
                    "paper_table": f"Table {number}",
                    "model": model,
                    "dataset": dataset,
                    "task": task,
                    "metric": metric,
                    "paper_value": paper_value,
                    "unit": unit,
                    "sort_order": dimension.get("sort_order", dimension_index + 1),
                    "source": "[PAPER]",
                    "paper_pdf_sha256": pdf_hash,
                    "pdf_page": pdf_page,
                    "note": note,
                }
                if qualifier is not None:
                    output_row["paper_value_qualifier"] = qualifier
                for optional_field in ("released_query_type",):
                    if optional_field in dimension:
                        output_row[optional_field] = nonempty_string(
                            dimension[optional_field],
                            f"{cell_label}.{optional_field}",
                        )
                    elif optional_field in table:
                        output_row[optional_field] = nonempty_string(
                            table[optional_field], f"{label}.{optional_field}"
                        )
                rows_out.append(output_row)

    excluded_numbers: list[int] = []
    for index, item in enumerate(excluded):
        label = f"non_metric_tables[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{label} must be an object")
        excluded_numbers.append(
            table_number(item.get("paper_table"), f"{label}.paper_table")
        )
        nonempty_string(item.get("reason"), f"{label}.reason")
        positive_integer(item.get("pdf_page"), f"{label}.pdf_page")
    classified = numeric_numbers + excluded_numbers
    if len(set(classified)) != len(classified):
        raise ValueError("paper tables are classified more than once")
    if sorted(classified) != expected_tables:
        raise ValueError("paper table classification does not cover Tables 1-17")
    if config.get("expected_metric_count") != len(rows_out):
        raise ValueError(
            "expected_metric_count mismatch: "
            f"{config.get('expected_metric_count')} != {len(rows_out)}"
        )
    rows_out.sort(
        key=lambda row: (
            table_number(row["paper_table"], "generated paper_table"),
            row["paper_metric_id"],
        )
    )
    return rows_out


def registry_text(rows: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    args = parse_args()
    config = json.loads(args.source.read_text(encoding="utf-8"))
    rows = expand_registry(config)
    expected_text = registry_text(rows)
    if args.check:
        if not args.output.is_file():
            raise ValueError(f"generated registry is missing: {args.output}")
        if args.output.read_text(encoding="utf-8") != expected_text:
            raise ValueError("generated paper metric registry is stale")
        print(f"[INFO] Registry is current: {args.output} ({len(rows)} metrics)")
        return 0
    atomic_write_text(args.output, expected_text)
    print(f"[INFO] Wrote {len(rows)} paper metrics: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
