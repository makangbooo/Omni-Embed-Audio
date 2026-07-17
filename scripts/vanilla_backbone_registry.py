#!/usr/bin/env python3
"""Load the fixed three-backbone vanilla evaluation registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPOSITORY_ROOT / "configs/checkpoints/vanilla_backbones.json"
EXPECTED_BACKBONE_IDS = (
    "vanilla_nemotron_3b",
    "vanilla_qwen2_5_omni_3b",
    "vanilla_qwen2_5_omni_7b",
)


def _tagged_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be a non-empty object")
    return value


def load_vanilla_backbone_registry(
    path: Path = DEFAULT_REGISTRY,
) -> dict[str, dict[str, Any]]:
    from scripts.prepare_official_oea_checkpoint import (
        find_asset,
        identifier,
        nonempty_string,
        validate_asset,
    )

    path = path.resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported vanilla backbone registry schema_version")
    if tuple(config.get("expected_backbone_ids", [])) != EXPECTED_BACKBONE_IDS:
        raise ValueError("expected_backbone_ids must use the fixed three-model order")
    source_tags = _tagged_mapping(config.get("source_tags"), "source_tags")
    if any(
        not isinstance(value, str) or not value.startswith("[")
        for value in source_tags.values()
    ):
        raise ValueError("vanilla backbone source_tags must be explicit")
    protocol = _tagged_mapping(config.get("protocol"), "protocol")
    expected_protocol_fields = {
        "text_prompt",
        "audio_prompt",
        "pooling",
        "normalization",
        "projection",
    }
    if set(protocol) != expected_protocol_fields:
        raise ValueError("vanilla backbone protocol fields are not exact")
    if protocol["audio_prompt"].get("status") != "CONFLICT":
        raise ValueError("audio prompt paper/code conflict must remain explicit")

    rows = config.get("backbones")
    if not isinstance(rows, list):
        raise ValueError("vanilla backbone registry backbones must be a list")
    output: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        label = f"backbones[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{label} must be an object")
        backbone_id = identifier(row.get("backbone_id"), f"{label}.backbone_id")
        if backbone_id in output:
            raise ValueError(f"duplicate backbone_id: {backbone_id}")
        reference = row.get("base_asset")
        if not isinstance(reference, dict):
            raise ValueError(f"{label}.base_asset must be an object")
        _, raw_asset, manifest_path = find_asset(
            reference, REPOSITORY_ROOT, f"{label}.base_asset"
        )
        output[backbone_id] = {
            "backbone_id": backbone_id,
            "paper_model": nonempty_string(
                row.get("paper_model"), f"{label}.paper_model"
            ),
            "family": identifier(row.get("family"), f"{label}.family"),
            "base_manifest": manifest_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "base_asset": validate_asset(raw_asset, f"{label}.base_asset"),
            "protocol": protocol,
            "source_tags": source_tags,
        }
    if tuple(output) != EXPECTED_BACKBONE_IDS:
        raise ValueError("resolved vanilla backbones differ from fixed order")
    return output
