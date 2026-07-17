"""Strict adapter for the UIQ JSONL schema released with OEA.

The released files do not use the legacy ``clip_id/uiq[].bucket/query``
schema consumed by :mod:`AudioRetrieval.evaluation.runners.uiq`. Positive
rows store text in ``generated_query``; negative rows store it in
``negative_query`` and do not include a hard-negative audio ID.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


POSITIVE_QUERY_TYPES = frozenset(
    {"question", "imperative", "paraphrase", "tagging"}
)
QUERY_TYPES = POSITIVE_QUERY_TYPES | {"negative"}


@dataclass(frozen=True)
class ReleasedUIQQuery:
    """One validated row from the released UIQ benchmark."""

    row_index: int
    audio_id: str
    dataset: str
    dataset_slug: str
    query_type: str
    query: str
    original_captions: tuple[str, ...]
    negative_captions: tuple[str, ...]
    metadata: Mapping[str, Any]
    source_model: str
    regen_model: str


def _required_string(row: Mapping[str, Any], field: str, row_index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"row {row_index}: {field} must be a non-empty string")
    return value.strip()


def _string_list(
    row: Mapping[str, Any],
    field: str,
    row_index: int,
    *,
    required: bool,
) -> tuple[str, ...]:
    value = row.get(field)
    if value is None and not required:
        return ()
    if not isinstance(value, list) or not value:
        raise ValueError(f"row {row_index}: {field} must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(
            f"row {row_index}: {field} must contain only non-empty strings"
        )
    return tuple(item.strip() for item in value)


def parse_released_uiq_row(
    row: Mapping[str, Any],
    *,
    row_index: int,
) -> ReleasedUIQQuery:
    """Validate and normalize one released JSON object without changing IDs."""

    if not isinstance(row, Mapping):
        raise TypeError(f"row {row_index}: JSON value must be an object")
    query_type = _required_string(row, "query_type", row_index).lower()
    if query_type not in QUERY_TYPES:
        raise ValueError(
            f"row {row_index}: unsupported query_type {query_type!r}"
        )
    query_field = "negative_query" if query_type == "negative" else "generated_query"
    forbidden_field = (
        "generated_query" if query_type == "negative" else "negative_query"
    )
    if forbidden_field in row:
        raise ValueError(
            f"row {row_index}: {query_type} row unexpectedly contains "
            f"{forbidden_field}"
        )
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"row {row_index}: metadata must be an object")

    return ReleasedUIQQuery(
        row_index=row_index,
        audio_id=_required_string(row, "audio_id", row_index),
        dataset=_required_string(row, "dataset", row_index).lower(),
        dataset_slug=_required_string(row, "dataset_slug", row_index),
        query_type=query_type,
        query=_required_string(row, query_field, row_index),
        original_captions=_string_list(
            row, "original_captions", row_index, required=True
        ),
        negative_captions=_string_list(
            row,
            "negative_captions",
            row_index,
            required=query_type == "negative",
        ),
        metadata=dict(metadata),
        source_model=_required_string(row, "source_model", row_index),
        regen_model=_required_string(row, "regen_model", row_index),
    )


def load_released_uiq(
    path: Path,
    *,
    expected_dataset: str | None = None,
    expected_query_type: str | None = None,
    require_unique_audio_ids: bool | None = None,
) -> tuple[ReleasedUIQQuery, ...]:
    """Load one released JSONL file and enforce file-level invariants.

    Positive releases are expected to contain one query per audio ID and are
    unique by default. Negative releases intentionally contain multiple rows
    per target audio, so duplicates are retained by default.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[ReleasedUIQQuery] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            rows.append(parse_released_uiq_row(raw, row_index=line_number))
    if not rows:
        raise ValueError(f"{path}: no UIQ rows")

    datasets = {row.dataset for row in rows}
    query_types = {row.query_type for row in rows}
    if len(datasets) != 1:
        raise ValueError(f"{path}: mixed datasets: {sorted(datasets)}")
    if len(query_types) != 1:
        raise ValueError(f"{path}: mixed query types: {sorted(query_types)}")
    if expected_dataset is not None and datasets != {expected_dataset.lower()}:
        raise ValueError(
            f"{path}: dataset {sorted(datasets)} != expected {expected_dataset!r}"
        )
    if expected_query_type is not None and query_types != {
        expected_query_type.lower()
    }:
        raise ValueError(
            f"{path}: query type {sorted(query_types)} != expected "
            f"{expected_query_type!r}"
        )

    query_type = next(iter(query_types))
    if require_unique_audio_ids is None:
        require_unique_audio_ids = query_type in POSITIVE_QUERY_TYPES
    if require_unique_audio_ids:
        seen: set[str] = set()
        duplicates: list[str] = []
        for row in rows:
            if row.audio_id in seen:
                duplicates.append(row.audio_id)
            seen.add(row.audio_id)
        if duplicates:
            raise ValueError(
                f"{path}: duplicate audio IDs: {sorted(set(duplicates))[:10]}"
            )
    return tuple(rows)


def released_uiq_summary(rows: Iterable[ReleasedUIQQuery]) -> dict[str, Any]:
    """Return small auditable counts without discarding duplicate rows."""

    materialized = tuple(rows)
    if not materialized:
        raise ValueError("cannot summarize an empty UIQ collection")
    return {
        "rows": len(materialized),
        "unique_audio_ids": len({row.audio_id for row in materialized}),
        "dataset": sorted({row.dataset for row in materialized}),
        "dataset_slug": sorted({row.dataset_slug for row in materialized}),
        "query_type": sorted({row.query_type for row in materialized}),
        "source_model": sorted({row.source_model for row in materialized}),
        "regen_model": sorted({row.regen_model for row in materialized}),
        "hard_negative_audio_id_available": False,
    }
