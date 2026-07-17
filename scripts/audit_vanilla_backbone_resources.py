#!/usr/bin/env python3
"""Audit exactly one immutable vanilla multimodal backbone snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.prepare_official_oea_checkpoint import file_identity
from scripts.vanilla_backbone_registry import (
    DEFAULT_REGISTRY,
    load_vanilla_backbone_registry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def portable_registry_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display = resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        display = resolved.name
    return file_identity(resolved, display)


def resolve_backbone_audit_plan(
    backbone_id: str, registry_path: Path = DEFAULT_REGISTRY
) -> dict[str, Any]:
    registry_path = registry_path.resolve()
    backbones = load_vanilla_backbone_registry(registry_path)
    if backbone_id not in backbones:
        raise ValueError(
            f"unknown backbone {backbone_id}; expected one of {list(backbones)}"
        )
    backbone = backbones[backbone_id]
    return {
        "schema_version": 1,
        "type": "vanilla_backbone",
        "backbone_id": backbone_id,
        "model": backbone["paper_model"],
        "vanilla_registry": portable_registry_identity(registry_path),
        "backbone": backbone,
        "manifests": [backbone["base_manifest"]],
        "asset_names": [backbone["base_asset"]["name"]],
    }


def main() -> int:
    args = parse_args()
    plan = resolve_backbone_audit_plan(args.backbone, args.registry)
    from scripts.audit_model_resources import audit_resources

    return audit_resources(
        manifest_paths=[REPOSITORY_ROOT / plan["manifests"][0]],
        model_root=args.model_root,
        output=args.output,
        asset_names=tuple(plan["asset_names"]),
        scope=plan,
    )


if __name__ == "__main__":
    raise SystemExit(main())
