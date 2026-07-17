#!/usr/bin/env python3
"""Read-only audit of baseline code coverage and immutable-resource readiness."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = (
    REPOSITORY_ROOT / "configs/baselines/baseline_readiness.json"
)
EXPECTED_MODEL_IDS = (
    "laion_clap",
    "robust_clap",
    "mga_clap",
    "m2d_clap",
    "vanilla_nemotron_3b",
    "vanilla_qwen2_5_omni_3b",
    "vanilla_qwen2_5_omni_7b",
)
ALLOWED_STATUSES = {
    "TODO",
    "IN_PROGRESS",
    "WAITING_USER",
    "RUNNING_REMOTE",
    "COMPLETED",
    "FAILED",
    "BLOCKED",
}
ALLOWED_CATEGORIES = {"clap_baseline", "vanilla_backbone"}
ALLOWED_IDENTITY_STATUSES = {"PINNED", "MISSING"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def safe_relative_path(value: Any, label: str) -> str:
    text = nonempty_string(value, label)
    if "\\" in text:
        raise ValueError(f"{label} must use POSIX separators")
    path = PurePosixPath(text)
    if path.is_absolute() or any(
        part in {"", ".", ".."} or ":" in part for part in path.parts
    ):
        raise ValueError(f"{label} must be a portable repository-relative path")
    return path.as_posix()


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("baseline registry has unsupported schema_version")
    if document.get("expected_model_ids") != list(EXPECTED_MODEL_IDS):
        raise ValueError("expected_model_ids must exactly match the audited scope")
    source_tags = document.get("source_tags")
    if not isinstance(source_tags, dict) or not source_tags:
        raise ValueError("source_tags must be a non-empty object")
    for key, value in source_tags.items():
        nonempty_string(key, "source_tags key")
        nonempty_string(value, f"source_tags.{key}")

    data_gate = document.get("common_data_gate")
    if not isinstance(data_gate, dict) or data_gate.get("status") != "BLOCKED":
        raise ValueError("common_data_gate must explicitly remain BLOCKED")
    nonempty_string(data_gate.get("reason"), "common_data_gate.reason")

    models = document.get("models")
    if not isinstance(models, list) or len(models) != len(EXPECTED_MODEL_IDS):
        raise ValueError("models must cover the exact audited scope")
    seen: list[str] = []
    for index, model in enumerate(models):
        label = f"models[{index}]"
        if not isinstance(model, dict):
            raise ValueError(f"{label} must be an object")
        model_id = nonempty_string(model.get("model_id"), f"{label}.model_id")
        seen.append(model_id)
        nonempty_string(model.get("paper_model"), f"{model_id}.paper_model")
        if model.get("category") not in ALLOWED_CATEGORIES:
            raise ValueError(f"{model_id}: invalid category")
        if model.get("status") not in ALLOWED_STATUSES:
            raise ValueError(f"{model_id}: invalid status")
        if model.get("status") != "BLOCKED":
            raise ValueError(f"{model_id}: readiness audit must not pre-unblock runs")

        surfaces = model.get("surfaces")
        if not isinstance(surfaces, list) or not surfaces:
            raise ValueError(f"{model_id}: surfaces must be non-empty")
        surface_ids: set[str] = set()
        for surface in surfaces:
            if not isinstance(surface, dict):
                raise ValueError(f"{model_id}: surface must be an object")
            surface_id = nonempty_string(
                surface.get("surface_id"), f"{model_id}.surface_id"
            )
            if surface_id in surface_ids:
                raise ValueError(f"{model_id}: duplicate surface_id {surface_id}")
            surface_ids.add(surface_id)
            surface["path"] = safe_relative_path(
                surface.get("path"), f"{model_id}.{surface_id}.path"
            )
            if "needle" in surface:
                nonempty_string(
                    surface.get("needle"), f"{model_id}.{surface_id}.needle"
                )
            if not isinstance(surface.get("expected_present"), bool):
                raise ValueError(f"{model_id}.{surface_id}: expected_present must be bool")
            if not isinstance(surface.get("required_for_formal"), bool):
                raise ValueError(f"{model_id}.{surface_id}: required_for_formal must be bool")

        resources = model.get("required_local_resources")
        if not isinstance(resources, list):
            raise ValueError(f"{model_id}: required_local_resources must be a list")
        resource_ids: set[str] = set()
        for resource in resources:
            if not isinstance(resource, dict):
                raise ValueError(f"{model_id}: resource must be an object")
            resource_id = nonempty_string(
                resource.get("resource_id"), f"{model_id}.resource_id"
            )
            if resource_id in resource_ids:
                raise ValueError(f"{model_id}: duplicate resource_id {resource_id}")
            resource_ids.add(resource_id)
            if resource.get("root") != "repository":
                raise ValueError(f"{model_id}.{resource_id}: unsupported root")
            resource["path"] = safe_relative_path(
                resource.get("path"), f"{model_id}.{resource_id}.path"
            )
            if resource.get("kind") not in {"file", "directory"}:
                raise ValueError(f"{model_id}.{resource_id}: invalid kind")

        identity = model.get("resource_identity")
        if not isinstance(identity, dict):
            raise ValueError(f"{model_id}: resource_identity must be an object")
        if identity.get("status") not in ALLOWED_IDENTITY_STATUSES:
            raise ValueError(f"{model_id}: invalid resource identity status")
        if identity["status"] == "PINNED":
            registry_reference = nonempty_string(
                identity.get("registry_reference"),
                f"{model_id}.registry_reference",
            )
            parts = registry_reference.split("#", 1)
            if len(parts) != 2 or not parts[1]:
                raise ValueError(
                    f"{model_id}: registry_reference must include an asset fragment"
                )
            safe_relative_path(
                parts[0],
                f"{model_id}.registry_reference",
            )
        else:
            nonempty_string(
                identity.get("code_reference"), f"{model_id}.code_reference"
            )
            if identity.get("immutable_revision") is not None:
                raise ValueError(f"{model_id}: MISSING identity has a revision")
            if identity.get("checkpoint_sha256") is not None:
                raise ValueError(f"{model_id}: MISSING identity has a SHA256")

        blockers = model.get("blockers")
        if not isinstance(blockers, list) or not blockers:
            raise ValueError(f"{model_id}: blockers must be non-empty")
        for blocker in blockers:
            nonempty_string(blocker, f"{model_id}.blocker")
        nonempty_string(model.get("next_action"), f"{model_id}.next_action")

    if seen != list(EXPECTED_MODEL_IDS):
        raise ValueError("models must appear once in expected_model_ids order")
    return document


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repository_root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def audit_surface(repository_root: Path, surface: dict[str, Any]) -> dict[str, Any]:
    path = repository_root / Path(*PurePosixPath(surface["path"]).parts)
    path_exists = path.exists()
    needle = surface.get("needle")
    if needle is None:
        actual_present = path_exists
    elif path.is_file():
        actual_present = needle in path.read_text(encoding="utf-8")
    else:
        actual_present = False
    return {
        "surface_id": surface["surface_id"],
        "path": surface["path"],
        "needle": needle,
        "expected_present": surface["expected_present"],
        "actual_present": actual_present,
        "matches_expected": actual_present == surface["expected_present"],
        "required_for_formal": surface["required_for_formal"],
        "file_sha256": sha256_file(path) if path.is_file() else None,
    }


def audit_resource(repository_root: Path, resource: dict[str, Any]) -> dict[str, Any]:
    path = repository_root / Path(*PurePosixPath(resource["path"]).parts)
    exists = path.is_file() if resource["kind"] == "file" else path.is_dir()
    return {
        "resource_id": resource["resource_id"],
        "root": resource["root"],
        "path": resource["path"],
        "kind": resource["kind"],
        "exists": exists,
        "size_bytes": path.stat().st_size if exists and path.is_file() else None,
        "sha256": sha256_file(path) if exists and path.is_file() else None,
    }


def audit_resource_identity(
    repository_root: Path, identity: dict[str, Any]
) -> dict[str, Any]:
    if identity["status"] == "MISSING":
        return {**identity, "complete": False}
    registry_name, asset_name = identity["registry_reference"].split("#", 1)
    registry_path = repository_root / Path(
        *PurePosixPath(registry_name).parts
    )
    result: dict[str, Any] = {
        **identity,
        "registry_exists": registry_path.is_file(),
        "registry_sha256": (
            sha256_file(registry_path) if registry_path.is_file() else None
        ),
        "asset_name": asset_name,
        "asset_found": False,
        "complete": False,
    }
    if not registry_path.is_file():
        return result
    document = json.loads(registry_path.read_text(encoding="utf-8"))
    assets = document.get("assets", []) if isinstance(document, dict) else []
    matches = [
        asset
        for asset in assets
        if isinstance(asset, dict) and asset.get("name") == asset_name
    ]
    if len(matches) != 1:
        return result
    asset = matches[0]
    revision = asset.get("revision")
    required_files = asset.get("required_files")
    complete = (
        isinstance(asset.get("repo_id"), str)
        and bool(asset["repo_id"])
        and isinstance(revision, str)
        and len(revision) == 40
        and isinstance(asset.get("local_subdir"), str)
        and bool(asset["local_subdir"])
        and isinstance(required_files, list)
        and bool(required_files)
        and all(isinstance(path, str) and path for path in required_files)
    )
    result.update(
        {
            "asset_found": True,
            "repo_id": asset.get("repo_id"),
            "revision": revision,
            "local_subdir": asset.get("local_subdir"),
            "required_files": required_files,
            "complete": complete,
        }
    )
    return result


def portable_path(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def audit_registry(
    registry: dict[str, Any],
    repository_root: Path,
    registry_path: Path,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    model_reports: list[dict[str, Any]] = []
    drift_count = 0
    for model in registry["models"]:
        surfaces = [
            audit_surface(repository_root, surface) for surface in model["surfaces"]
        ]
        resources = [
            audit_resource(repository_root, resource)
            for resource in model["required_local_resources"]
        ]
        drift_count += sum(not surface["matches_expected"] for surface in surfaces)
        required_surfaces = [
            surface for surface in surfaces if surface["required_for_formal"]
        ]
        code_surface_complete = all(
            surface["actual_present"] for surface in required_surfaces
        )
        local_resources_present: bool | None = (
            all(resource["exists"] for resource in resources) if resources else None
        )
        resource_identity = audit_resource_identity(
            repository_root, model["resource_identity"]
        )
        identity_complete = resource_identity["complete"]
        formal_ready = (
            code_surface_complete
            and identity_complete
            and local_resources_present is not False
            and registry["common_data_gate"]["status"] != "BLOCKED"
            and model["status"] != "BLOCKED"
        )
        model_reports.append(
            {
                "model_id": model["model_id"],
                "paper_model": model["paper_model"],
                "category": model["category"],
                "declared_status": model["status"],
                "surfaces": surfaces,
                "code_surface_complete": code_surface_complete,
                "required_local_resources": resources,
                "local_resources_present": local_resources_present,
                "resource_identity": resource_identity,
                "resource_identity_complete": identity_complete,
                "common_data_gate_status": registry["common_data_gate"]["status"],
                "formal_ready": formal_ready,
                "blockers": model["blockers"],
                "next_action": model["next_action"],
            }
        )

    return {
        "schema_version": 1,
        "report_status": "complete" if drift_count == 0 else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value(repository_root, "rev-parse", "HEAD"),
        "git_status_short": git_value(repository_root, "status", "--short"),
        "repository_root": ".",
        "registry": portable_path(registry_path, repository_root),
        "registry_sha256": sha256_file(registry_path),
        "source_tags": registry["source_tags"],
        "common_data_gate": registry["common_data_gate"],
        "summary": {
            "models": len(model_reports),
            "clap_baselines": sum(
                model["category"] == "clap_baseline" for model in model_reports
            ),
            "vanilla_backbones": sum(
                model["category"] == "vanilla_backbone"
                for model in model_reports
            ),
            "code_surface_complete": sum(
                model["code_surface_complete"] for model in model_reports
            ),
            "resource_identity_complete": sum(
                model["resource_identity_complete"] for model in model_reports
            ),
            "formal_ready": sum(model["formal_ready"] for model in model_reports),
            "evidence_drift": drift_count,
        },
        "models": model_reports,
    }


def main() -> int:
    args = parse_args()
    registry_path = args.registry.resolve()
    registry = load_registry(registry_path)
    report = audit_registry(registry, args.repository_root, registry_path)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing report: {output}")
        with output.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
    else:
        print(rendered, end="")
    return 0 if report["report_status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
