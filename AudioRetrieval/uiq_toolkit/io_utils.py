"""
Dataset loading and JSONL helpers for standalone UIQ generation.
"""

from __future__ import annotations

import csv
import json
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from uiq_toolkit.query_types import QueryRecord, QueryType, legacy_bucket_for_query_type


@dataclass(frozen=True)
class SourceExample:
    """Minimal input example required for UIQ generation."""

    clip_id: str
    caption: str


def load_clotho_examples(csv_path: Path, caption_index: int = 1) -> List[SourceExample]:
    """Load one caption per clip from a Clotho CSV file."""
    if caption_index < 1 or caption_index > 5:
        raise ValueError("caption_index must be between 1 and 5 for Clotho.")

    examples: List[SourceExample] = []
    with Path(csv_path).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            file_name = (row.get("file_name") or "").strip()
            caption = (row.get(f"caption_{caption_index}") or "").strip()
            if not file_name or not caption:
                continue
            examples.append(SourceExample(clip_id=Path(file_name).stem, caption=caption))
    return examples


def load_audiocaps_examples(csv_path: Path) -> List[SourceExample]:
    """Load one caption per clip from an AudioCaps CSV file."""
    examples: List[SourceExample] = []
    with Path(csv_path).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            youtube_id = (row.get("youtube_id") or "").strip()
            start_time = (row.get("start_time") or "").strip()
            caption = (row.get("caption") or "").strip()
            if not youtube_id or not start_time or not caption:
                continue
            examples.append(SourceExample(clip_id=f"{youtube_id}_{start_time}", caption=caption))
    return examples


def load_mecat_examples(
    meta_dir: Path,
    caption_field: str = "short",
    caption_index: int = 0,
) -> List[SourceExample]:
    """Load examples from a directory of MeCAT JSON metadata files."""
    meta_dir = Path(meta_dir)
    examples: List[SourceExample] = []
    for json_path in sorted(meta_dir.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        captions = data.get(caption_field, [])
        if not captions or caption_index >= len(captions):
            continue
        caption = captions[caption_index].strip()
        if not caption or caption.lower() == "none":
            continue
        examples.append(SourceExample(clip_id=json_path.stem, caption=caption))
    return examples


def load_examples(
    dataset: str,
    csv_path: Path,
    caption_index: int = 1,
) -> List[SourceExample]:
    """Load source examples for a supported dataset."""
    dataset = dataset.lower().strip()
    if dataset == "clotho":
        return load_clotho_examples(csv_path, caption_index=caption_index)
    if dataset == "audiocaps":
        return load_audiocaps_examples(csv_path)
    if dataset == "mecat":
        return load_mecat_examples(csv_path, caption_field="short", caption_index=0)
    raise ValueError(f"Unsupported dataset: {dataset}")


def _first_non_empty(value: Any) -> Optional[str]:
    """Return the first non-empty string from a scalar or sequence."""
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    if isinstance(value, Sequence):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return None


def _extract_hard_negative_caption(item: Dict[str, Any]) -> Optional[str]:
    """Extract a usable hard-negative caption from a JSONL record."""
    direct_caption = _first_non_empty(
        item.get("hard_negative_caption")
        or item.get("negative_caption")
        or item.get("hard_negative_captions")
    )
    if direct_caption:
        return direct_caption

    negatives = item.get("hard_negatives") or item.get("neighbors") or item.get("negatives") or []
    if not isinstance(negatives, Sequence):
        return None

    for negative in negatives:
        if not isinstance(negative, dict):
            continue
        caption = _first_non_empty(
            negative.get("captions")
            or negative.get("caption")
            or negative.get("hard_negative_caption")
        )
        if caption:
            return caption

    return None


def load_hard_negative_caption_map(jsonl_path: Path) -> Dict[str, str]:
    """
    Load a mapping from clip ID to one hard-negative caption.

    This supports the Stage-2 filtered hard-negative JSONL already produced in
    this repository, plus a few simpler hand-written mapping formats.
    """

    mapping: Dict[str, str] = {}
    with Path(jsonl_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            clip_id = (
                item.get("audio_id")
                or item.get("clip_id")
                or item.get("target_audio")
                or item.get("source_audio")
            )
            if not clip_id:
                continue

            caption = _extract_hard_negative_caption(item)
            if caption:
                mapping[str(clip_id).strip()] = caption

    return mapping


def save_flat_results(records: Iterable[QueryRecord], output_path: Path) -> None:
    """Write flat JSONL records."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_flat_dict(), ensure_ascii=False) + "\n")


def group_records_by_clip(records: Iterable[QueryRecord]) -> List[Dict[str, Any]]:
    """
    Convert flat per-query records into the grouped UIQ JSONL schema.

    The grouped schema is compatible with the legacy evaluator in this repo:
    each line contains one clip and a `uiq` list with legacy bucket names.
    """

    grouped: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    for record in records:
        query = record.query.strip()
        if not query:
            continue

        entry = grouped.setdefault(
            record.clip_id,
            {
                "clip_id": record.clip_id,
                "source_captions": [],
                "uiq": [],
            },
        )

        if record.original_caption and record.original_caption not in entry["source_captions"]:
            entry["source_captions"].append(record.original_caption)

        uiq_item: Dict[str, Any] = {
            "bucket": legacy_bucket_for_query_type(record.query_type),
            "query_type": record.query_type.value,
            "query": query,
        }
        if record.hard_negative_caption:
            uiq_item["hard_negative_caption"] = record.hard_negative_caption
        if record.metadata:
            uiq_item["metadata"] = record.metadata

        entry["uiq"].append(uiq_item)

    return list(grouped.values())


def save_grouped_results(records: Iterable[QueryRecord], output_path: Path) -> None:
    """Write grouped evaluator-compatible UIQ JSONL records."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grouped = group_records_by_clip(records)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in grouped:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_metadata(output_path: Path, payload: Dict[str, Any]) -> None:
    """Write run metadata as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
