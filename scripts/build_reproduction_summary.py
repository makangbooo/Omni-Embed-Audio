#!/usr/bin/env python3
"""Build the auditable paper-versus-reproduction summary table."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER_REGISTRY = (
    REPOSITORY_ROOT / "configs/results/paper_reported_metrics.jsonl"
)
DEFAULT_OBSERVATIONS = (
    REPOSITORY_ROOT / "results/observations/reproduction_observations.jsonl"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "results/tables/reproduction_summary.csv"
SUMMARY_COLUMNS = (
    "paper_table",
    "experiment",
    "model",
    "dataset",
    "task",
    "metric",
    "paper_value",
    "reproduced_value",
    "absolute_delta",
    "relative_delta",
    "seed",
    "checkpoint",
    "status",
    "notes",
)
ALLOWED_STATUSES = {
    "exact",
    "close",
    "trend_reproduced",
    "failed",
    "blocked",
    "not_reproducible",
}
VALUE_STATUSES = {"exact", "close", "trend_reproduced"}
MISSING_VALUE_STATUSES = {"blocked", "not_reproducible"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
PAPER_VALUE_PATTERN = re.compile(r"^(?:0|[1-9][0-9]*)\.[0-9]{2}$")
TABLE_PATTERN = re.compile(r"^Table ([1-9][0-9]*)$")
METRIC_ORDER = {"R@1": 1, "R@5": 5, "R@10": 10}
SIX_PLACES = Decimal("0.000001")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-registry", type=Path, default=DEFAULT_PAPER_REGISTRY
    )
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--audit-output",
        type=Path,
        help="Defaults to <output stem>.audit.json.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_repository_path(path: Path) -> str:
    resolved = path.resolve(strict=True)
    try:
        return resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"path is outside repository: {path}") from error


def file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required file is missing: {path}")
    return {
        "path": relative_repository_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"{label} file is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            raise ValueError(f"{label} line {line_number} is blank")
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{label} line {line_number} is invalid JSON: {error}"
            ) from error
        if not isinstance(row, dict):
            raise ValueError(f"{label} line {line_number} must be an object")
        rows.append(row)
    return rows


def required_nonempty_string(
    row: Mapping[str, Any], field: str, label: str
) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: {field} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{label}: {field} must not have surrounding whitespace")
    return value


def decimal_value(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{label} must be a finite number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not result.is_finite():
        raise ValueError(f"{label} must be a finite number")
    return result


def decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def rounded_delta_text(value: Decimal) -> str:
    return decimal_text(value.quantize(SIX_PLACES, rounding=ROUND_HALF_UP))


def validate_paper_registry(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    required_fields = {
        "dataset",
        "metric",
        "model",
        "note",
        "paper_metric_id",
        "paper_pdf_sha256",
        "paper_table",
        "paper_value",
        "pdf_page",
        "source",
        "task",
        "unit",
    }
    registry: dict[str, dict[str, Any]] = {}
    for index, raw_row in enumerate(rows, start=1):
        label = f"paper registry row {index}"
        missing = sorted(required_fields - set(raw_row))
        if missing:
            raise ValueError(f"{label}: missing fields: {missing}")
        row = dict(raw_row)
        metric_id = required_nonempty_string(row, "paper_metric_id", label)
        if metric_id in registry:
            raise ValueError(f"{label}: duplicate paper_metric_id: {metric_id}")
        table = required_nonempty_string(row, "paper_table", label)
        if TABLE_PATTERN.fullmatch(table) is None:
            raise ValueError(f"{label}: invalid paper_table: {table}")
        for field in ("model", "dataset", "task", "note"):
            required_nonempty_string(row, field, label)
        metric = required_nonempty_string(row, "metric", label)
        if metric not in METRIC_ORDER:
            raise ValueError(f"{label}: unsupported metric: {metric}")
        paper_value = row.get("paper_value")
        if not isinstance(paper_value, str) or PAPER_VALUE_PATTERN.fullmatch(
            paper_value
        ) is None:
            raise ValueError(
                f"{label}: paper_value must be a string with two decimals"
            )
        numeric_paper_value = decimal_value(paper_value, f"{label}: paper_value")
        if not Decimal("0") <= numeric_paper_value <= Decimal("100"):
            raise ValueError(f"{label}: paper_value is outside [0, 100]")
        if row.get("unit") != "percent":
            raise ValueError(f"{label}: unit must be percent")
        if row.get("source") != "[PAPER]":
            raise ValueError(f"{label}: source must be [PAPER]")
        pdf_hash = row.get("paper_pdf_sha256")
        if not isinstance(pdf_hash, str) or SHA256_PATTERN.fullmatch(pdf_hash) is None:
            raise ValueError(f"{label}: invalid paper_pdf_sha256")
        pdf_page = row.get("pdf_page")
        if not isinstance(pdf_page, int) or isinstance(pdf_page, bool) or pdf_page <= 0:
            raise ValueError(f"{label}: pdf_page must be a positive integer")
        registry[metric_id] = row
    if not registry:
        raise ValueError("paper registry is empty")
    return registry


def resolve_evidence_path(relative_path: str) -> Path:
    candidate = (REPOSITORY_ROOT / relative_path).resolve()
    try:
        candidate.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"evidence path escapes repository: {relative_path}") from error
    if candidate == REPOSITORY_ROOT.resolve():
        raise ValueError("evidence path must identify a file")
    return candidate


def verify_evidence(
    evidence: Any, label: str, require_json_value: bool
) -> Decimal | None:
    if not isinstance(evidence, dict):
        raise ValueError(f"{label}: evidence must be an object")
    expected_fields = {"path", "sha256", "size_bytes"}
    if require_json_value:
        expected_fields.add("json_path")
    missing = sorted(expected_fields - set(evidence))
    if missing:
        raise ValueError(f"{label}: evidence missing fields: {missing}")
    path_text = evidence.get("path")
    if not isinstance(path_text, str) or not path_text.strip():
        raise ValueError(f"{label}: evidence.path must be a non-empty string")
    path = resolve_evidence_path(path_text)
    if not path.is_file():
        raise ValueError(f"{label}: evidence file is missing: {path_text}")
    expected_size = evidence.get("size_bytes")
    if (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size < 0
    ):
        raise ValueError(f"{label}: evidence.size_bytes must be non-negative")
    if path.stat().st_size != expected_size:
        raise ValueError(f"{label}: evidence size mismatch: {path_text}")
    expected_hash = evidence.get("sha256")
    if not isinstance(expected_hash, str) or SHA256_PATTERN.fullmatch(
        expected_hash
    ) is None:
        raise ValueError(f"{label}: invalid evidence.sha256")
    if sha256_file(path) != expected_hash:
        raise ValueError(f"{label}: evidence SHA256 mismatch: {path_text}")
    json_path = evidence.get("json_path")
    if json_path is None:
        if require_json_value:
            raise ValueError(f"{label}: evidence.json_path is required")
        return None
    if not isinstance(json_path, list) or not json_path:
        raise ValueError(f"{label}: evidence.json_path must be a non-empty list")
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    for component in json_path:
        if isinstance(component, bool) or not isinstance(component, (str, int)):
            raise ValueError(
                f"{label}: evidence.json_path components must be strings or integers"
            )
        try:
            if isinstance(component, int):
                if not isinstance(value, list):
                    raise TypeError
                value = value[component]
            else:
                if not isinstance(value, dict):
                    raise TypeError
                value = value[component]
        except (IndexError, KeyError, TypeError) as error:
            raise ValueError(
                f"{label}: evidence.json_path does not resolve: {json_path}"
            ) from error
    return decimal_value(value, f"{label}: resolved evidence value")


def validate_observations(
    rows: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    required_fields = {
        "checkpoint",
        "experiment",
        "notes",
        "observation_id",
        "paper_metric_id",
        "reproduced_value",
        "seed",
        "status",
    }
    observations: list[dict[str, Any]] = []
    observation_ids: set[str] = set()
    experiment_keys: set[tuple[str, str, int | None]] = set()
    for index, raw_row in enumerate(rows, start=1):
        label = f"observation row {index}"
        missing = sorted(required_fields - set(raw_row))
        if missing:
            raise ValueError(f"{label}: missing fields: {missing}")
        row = dict(raw_row)
        observation_id = required_nonempty_string(row, "observation_id", label)
        if observation_id in observation_ids:
            raise ValueError(f"{label}: duplicate observation_id: {observation_id}")
        observation_ids.add(observation_id)
        metric_id = required_nonempty_string(row, "paper_metric_id", label)
        if metric_id not in registry:
            raise ValueError(f"{label}: unknown paper_metric_id: {metric_id}")
        experiment = required_nonempty_string(row, "experiment", label)
        required_nonempty_string(row, "notes", label)
        status = row.get("status")
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"{label}: unsupported status: {status}")
        seed = row.get("seed")
        if seed is not None and (
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
        ):
            raise ValueError(f"{label}: seed must be null or a non-negative integer")
        checkpoint = row.get("checkpoint")
        if checkpoint is not None and (
            not isinstance(checkpoint, str) or not checkpoint.strip()
        ):
            raise ValueError(f"{label}: checkpoint must be null or non-empty")
        key = (experiment, metric_id, seed)
        if key in experiment_keys:
            raise ValueError(
                f"{label}: duplicate experiment/paper_metric_id/seed observation"
            )
        experiment_keys.add(key)

        reproduced_raw = row.get("reproduced_value")
        reproduced: Decimal | None = None
        if status in VALUE_STATUSES:
            reproduced = decimal_value(
                reproduced_raw, f"{label}: reproduced_value"
            )
            if not Decimal("0") <= reproduced <= Decimal("100"):
                raise ValueError(
                    f"{label}: reproduced_value is outside [0, 100]"
                )
            evidence_value = verify_evidence(
                row.get("evidence"), label, require_json_value=True
            )
            if evidence_value != reproduced:
                raise ValueError(
                    f"{label}: reproduced_value does not match evidence JSON value"
                )
        elif status == "failed":
            if reproduced_raw is not None:
                reproduced = decimal_value(
                    reproduced_raw, f"{label}: reproduced_value"
                )
                evidence_value = verify_evidence(
                    row.get("evidence"), label, require_json_value=True
                )
                if evidence_value != reproduced:
                    raise ValueError(
                        f"{label}: reproduced_value does not match evidence JSON value"
                    )
            else:
                verify_evidence(
                    row.get("evidence"), label, require_json_value=False
                )
        elif status in MISSING_VALUE_STATUSES:
            if reproduced_raw is not None:
                raise ValueError(
                    f"{label}: {status} observation must have null reproduced_value"
                )
            if "evidence" in row:
                verify_evidence(row["evidence"], label, require_json_value=False)

        paper_value = decimal_value(
            registry[metric_id]["paper_value"], f"{label}: paper_value"
        )
        if status == "exact" and reproduced != paper_value:
            raise ValueError(
                f"{label}: exact status requires reproduced_value == paper_value"
            )
        row["_reproduced_decimal"] = reproduced
        observations.append(row)
    return observations


def table_number(value: str) -> int:
    match = TABLE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid paper table: {value}")
    return int(match.group(1))


def summary_rows(
    observations: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    output_rows: list[dict[str, str]] = []
    for observation in observations:
        paper = registry[str(observation["paper_metric_id"])]
        paper_value = decimal_value(paper["paper_value"], "paper_value")
        reproduced = observation["_reproduced_decimal"]
        if reproduced is None:
            reproduced_text = ""
            absolute_delta = ""
            relative_delta = ""
        else:
            reproduced_text = decimal_text(reproduced)
            absolute_delta = rounded_delta_text(abs(reproduced - paper_value))
            relative_delta = (
                ""
                if paper_value == 0
                else rounded_delta_text(
                    (reproduced - paper_value) / paper_value * Decimal("100")
                )
            )
        seed = observation.get("seed")
        output_rows.append(
            {
                "paper_table": str(paper["paper_table"]),
                "experiment": str(observation["experiment"]),
                "model": str(paper["model"]),
                "dataset": str(paper["dataset"]),
                "task": str(paper["task"]),
                "metric": str(paper["metric"]),
                "paper_value": str(paper["paper_value"]),
                "reproduced_value": reproduced_text,
                "absolute_delta": absolute_delta,
                "relative_delta": relative_delta,
                "seed": "" if seed is None else str(seed),
                "checkpoint": ""
                if observation.get("checkpoint") is None
                else str(observation["checkpoint"]),
                "status": str(observation["status"]),
                "notes": str(observation["notes"]),
            }
        )
    output_rows.sort(
        key=lambda row: (
            table_number(row["paper_table"]),
            row["experiment"],
            row["task"],
            METRIC_ORDER[row["metric"]],
        )
    )
    return output_rows


def csv_text(rows: Sequence[Mapping[str, str]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


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


def build_summary(
    paper_registry_path: Path,
    observations_path: Path,
    output_path: Path,
    audit_output_path: Path,
) -> dict[str, Any]:
    registry = validate_paper_registry(
        read_jsonl(paper_registry_path, "paper registry")
    )
    observations = validate_observations(
        read_jsonl(observations_path, "observations"), registry
    )
    rows = summary_rows(observations, registry)
    atomic_write_text(output_path, csv_text(rows))
    observed_metric_ids = {str(row["paper_metric_id"]) for row in observations}
    audit = {
        "schema_version": 1,
        "status": "complete",
        "inputs": {
            "paper_registry": file_identity(paper_registry_path),
            "observations": file_identity(observations_path),
        },
        "paper_metric_count": len(registry),
        "observation_count": len(observations),
        "summary_row_count": len(rows),
        "unobserved_paper_metric_count": len(registry) - len(observed_metric_ids),
        "unobserved_paper_metric_ids": sorted(set(registry) - observed_metric_ids),
        "status_counts": dict(
            sorted(Counter(str(row["status"]) for row in observations).items())
        ),
        "delta_contract": {
            "absolute_delta": "abs(reproduced_value - paper_value), percentage points",
            "relative_delta": "(reproduced_value - paper_value) / paper_value * 100, signed percent",
            "rounding": "ROUND_HALF_UP to 6 decimal places",
        },
        "output": file_identity(output_path),
    }
    atomic_write_text(
        audit_output_path,
        json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    return audit


def main() -> int:
    args = parse_args()
    output_path = args.output.resolve()
    audit_output_path = (
        args.audit_output.resolve()
        if args.audit_output is not None
        else output_path.with_suffix(".audit.json")
    )
    build_summary(
        args.paper_registry.resolve(),
        args.observations.resolve(),
        output_path,
        audit_output_path,
    )
    print(f"[INFO] Summary: {output_path}")
    print(f"[INFO] Audit: {audit_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
