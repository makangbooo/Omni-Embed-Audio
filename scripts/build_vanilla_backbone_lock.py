#!/usr/bin/env python3
"""Build a portable base-only vanilla-backbone lock from a complete audit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.build_official_oea_model_lock import (
    git_output,
    locked_file_inventory,
    matching_asset_report,
    portable_file_identity,
    read_json_object,
    validate_evidence_header,
)
from scripts.prepare_official_oea_checkpoint import atomic_write_json
from scripts.vanilla_backbone_registry import (
    DEFAULT_REGISTRY,
    load_vanilla_backbone_registry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--model-resource-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_scope(
    scope: Any, registry_path: Path, backbone: Mapping[str, Any]
) -> None:
    if not isinstance(scope, dict):
        raise ValueError("model resource audit must have a vanilla backbone scope")
    if scope.get("schema_version") != 1:
        raise ValueError("model resource audit scope has unsupported schema_version")
    if scope.get("type") != "vanilla_backbone":
        raise ValueError("model resource audit has an unexpected scope type")
    if scope.get("backbone_id") != backbone["backbone_id"]:
        raise ValueError("model resource audit scope backbone_id differs from request")
    if scope.get("backbone") != backbone:
        raise ValueError("model resource audit scope backbone differs from registry")
    if scope.get("manifests") != [backbone["base_manifest"]]:
        raise ValueError("model resource audit scope manifest differs from registry")
    if scope.get("asset_names") != [backbone["base_asset"]["name"]]:
        raise ValueError("model resource audit scope asset differs from registry")
    identity = scope.get("vanilla_registry")
    actual = portable_file_identity(registry_path)
    if not isinstance(identity, dict) or any(
        identity.get(field) != actual[field] for field in ("size_bytes", "sha256")
    ):
        raise ValueError("model resource audit scope used a different registry")


def build_vanilla_lock(
    *,
    backbone_id: str,
    registry_path: Path,
    resource_audit_path: Path,
) -> dict[str, Any]:
    registry_path = registry_path.resolve()
    resource_audit_path = resource_audit_path.resolve()
    backbones = load_vanilla_backbone_registry(registry_path)
    if backbone_id not in backbones:
        raise ValueError(f"unknown backbone: {backbone_id}")
    backbone = backbones[backbone_id]

    audit = read_json_object(resource_audit_path, "model resource audit")
    validate_evidence_header(audit, "model resource audit", {"complete"})
    if audit.get("fatal_error") is not None:
        raise ValueError("model resource audit has a fatal_error")
    validate_scope(audit.get("scope"), registry_path, backbone)
    expected_assets = [backbone["base_asset"]["name"]]
    if audit.get("requested_assets") != expected_assets:
        raise ValueError("model resource audit requested asset differs from scope")
    model_root_text = audit.get("model_root")
    if not isinstance(model_root_text, str) or not model_root_text:
        raise ValueError("model resource audit has no model_root")
    model_root = Path(model_root_text).resolve()
    base_report = matching_asset_report(
        audit, backbone["base_asset"], model_root, "base model"
    )
    base_files = locked_file_inventory(base_report, "base model")

    return {
        "schema_version": 1,
        "status": "locked",
        "lock_type": "vanilla_backbone_base_only",
        "backbone_id": backbone_id,
        "model": backbone["paper_model"],
        "family": backbone["family"],
        "claim_boundary": (
            "This lock proves the immutable base snapshot and public-code "
            "pooling/prompt protocol only; it is not a reproduced metric."
        ),
        "protocol": backbone["protocol"],
        "base_model": {
            "repo_id": backbone["base_asset"]["repo_id"],
            "revision": backbone["base_asset"]["revision"],
            "local_subdir": backbone["base_asset"]["local_subdir"],
            "source": "CODE+AUDIT",
            "files": base_files,
        },
        "embedding_output": {
            "projection_head": "none",
            "dimension": "backbone text hidden size resolved at runtime",
            "source": "[CODE] public base adapter returns normalized pooled hidden states",
        },
        "source_tags": backbone["source_tags"],
        "evidence": {
            "vanilla_registry": portable_file_identity(registry_path),
            "model_resource_audit": portable_file_identity(resource_audit_path),
            "resource_audit_git_commit": audit["git_commit"],
        },
    }


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    expected = (
        REPOSITORY_ROOT / "results/model_locks" / f"{args.backbone}.json"
    ).resolve()
    if output != expected:
        raise ValueError(f"lock output must use the canonical backbone path: {expected}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing model lock: {output}")
    if git_output("status", "--short"):
        raise RuntimeError("vanilla model lock generation requires a clean Git worktree")
    lock = build_vanilla_lock(
        backbone_id=args.backbone,
        registry_path=args.registry,
        resource_audit_path=args.model_resource_audit,
    )
    lock["builder_git_commit"] = git_output("rev-parse", "HEAD")
    atomic_write_json(output, lock)
    print(f"[INFO] Wrote locked vanilla base resources: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
