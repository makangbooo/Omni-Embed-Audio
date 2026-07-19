#!/usr/bin/env python3
"""Audit exactly the base model and checkpoint for one official OEA variant."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.prepare_official_oea_checkpoint import (
    DEFAULT_REGISTRY,
    file_identity,
    load_checkpoint_registry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-registered-derived",
        action="store_true",
        help=(
            "Allow only the selected variant's registry-declared derived checkpoint "
            "beside the immutable source snapshot. The checkpoint preparation stage "
            "must independently verify its tensors."
        ),
    )
    return parser.parse_args()


def portable_registry_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display = resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        display = resolved.name
    return file_identity(resolved, display)


def resolve_variant_audit_plan(
    variant_id: str, registry_path: Path = DEFAULT_REGISTRY
) -> dict[str, Any]:
    registry_path = registry_path.resolve()
    variants = load_checkpoint_registry(registry_path)
    if variant_id not in variants:
        raise ValueError(
            f"unknown variant {variant_id}; expected one of {list(variants)}"
        )
    variant = variants[variant_id]
    manifests: list[str] = []
    for field in ("base_manifest", "checkpoint_manifest"):
        manifest = variant[field]
        if manifest not in manifests:
            manifests.append(manifest)
    asset_names: list[str] = []
    for field in ("base_asset", "checkpoint_asset"):
        name = variant[field]["name"]
        if name in asset_names:
            raise ValueError(f"variant resolves the same asset twice: {name}")
        asset_names.append(name)
    return {
        "schema_version": 1,
        "type": "official_oea_variant",
        "variant_id": variant_id,
        "model": variant["paper_model"],
        "training_variant": variant["training_variant"],
        "checkpoint_registry": portable_registry_identity(registry_path),
        "variant": variant,
        "manifests": manifests,
        "asset_names": asset_names,
    }


def registered_derived_allowance(plan: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    checkpoint = plan["variant"]["checkpoint_asset"]
    return {checkpoint["name"]: (checkpoint["derived_file"],)}


def main() -> int:
    args = parse_args()
    plan = resolve_variant_audit_plan(args.variant, args.registry)

    # Imported lazily so registry planning and its unit tests do not require the
    # Hugging Face client. The formal CPU environment supplies that dependency.
    from scripts.audit_model_resources import audit_resources

    return audit_resources(
        manifest_paths=[REPOSITORY_ROOT / path for path in plan["manifests"]],
        model_root=args.model_root,
        output=args.output,
        asset_names=tuple(plan["asset_names"]),
        scope=plan,
        allowed_extra_files_by_asset=(
            registered_derived_allowance(plan)
            if args.allow_registered_derived
            else None
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
