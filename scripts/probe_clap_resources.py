#!/usr/bin/env python3
"""Probe local CLAP source, package, and checkpoint resources without loading models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.audit_baseline_readiness import DEFAULT_REGISTRY, load_registry


CLAP_MODEL_IDS = ("laion_clap", "robust_clap", "mga_clap", "m2d_clap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command_output(cwd: Path, *arguments: str) -> tuple[str | None, str | None]:
    try:
        completed = subprocess.run(
            list(arguments),
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        stderr = getattr(error, "stderr", None)
        return None, (stderr or repr(error)).strip()
    return completed.stdout.strip(), None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested_git_snapshot(path: Path) -> dict[str, Any] | None:
    if not (path / ".git").exists():
        return None
    head, head_error = command_output(path, "git", "rev-parse", "HEAD")
    status, status_error = command_output(path, "git", "status", "--short")
    return {
        "head": head,
        "status_short": status,
        "dirty": None if status is None else bool(status),
        "error": head_error or status_error,
    }


def inspect_resource(repository_root: Path, resource: dict[str, Any]) -> dict[str, Any]:
    relative = PurePosixPath(resource["path"])
    path = repository_root / Path(*relative.parts)
    expected_kind = resource["kind"]
    exists = path.is_file() if expected_kind == "file" else path.is_dir()
    result: dict[str, Any] = {
        "resource_id": resource["resource_id"],
        "path": resource["path"],
        "kind": expected_kind,
        "exists": exists,
        "size_bytes": None,
        "sha256": None,
        "nested_git": None,
    }
    if exists and expected_kind == "file":
        result["size_bytes"] = path.stat().st_size
        result["sha256"] = sha256_file(path)
    elif exists:
        result["nested_git"] = nested_git_snapshot(path)
    return result


def inspect_laion_package() -> dict[str, Any]:
    distribution = None
    distribution_name = None
    errors: list[str] = []
    for candidate in ("laion-clap", "laion_clap"):
        try:
            distribution = importlib.metadata.distribution(candidate)
            distribution_name = candidate
            break
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception as error:  # noqa: BLE001 - preserve local probe evidence
            errors.append(f"{candidate}: {error!r}")

    try:
        spec = importlib.util.find_spec("laion_clap")
    except Exception as error:  # noqa: BLE001 - preserve local probe evidence
        spec = None
        errors.append(f"find_spec: {error!r}")

    locations = [] if spec is None or spec.submodule_search_locations is None else [
        str(Path(location).resolve()) for location in spec.submodule_search_locations
    ]
    return {
        "installed": distribution is not None and spec is not None,
        "distribution_name": distribution_name,
        "distribution_version": (
            None if distribution is None else distribution.version
        ),
        "module_origin": None if spec is None or spec.origin is None else str(Path(spec.origin).resolve()),
        "module_search_locations": locations,
        "errors": errors,
    }


def surface_present(repository_root: Path, surface: dict[str, Any]) -> bool:
    relative = PurePosixPath(surface["path"])
    path = repository_root / Path(*relative.parts)
    if not path.is_file():
        return False
    needle = surface.get("needle")
    return needle is None or needle in path.read_text(encoding="utf-8")


def probe_registry(
    registry: dict[str, Any], repository_root: Path
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    by_id = {model["model_id"]: model for model in registry["models"]}
    if any(model_id not in by_id for model_id in CLAP_MODEL_IDS):
        raise ValueError("baseline registry does not cover all four CLAP models")

    laion_package = inspect_laion_package()
    models: list[dict[str, Any]] = []
    for model_id in CLAP_MODEL_IDS:
        model = by_id[model_id]
        resources = [
            inspect_resource(repository_root, resource)
            for resource in model["required_local_resources"]
        ]
        required_surfaces = [
            surface for surface in model["surfaces"] if surface["required_for_formal"]
        ]
        code_surface_complete = all(
            surface_present(repository_root, surface) for surface in required_surfaces
        )
        local_resources_complete = all(resource["exists"] for resource in resources)
        if model_id == "laion_clap":
            local_resources_complete = laion_package["installed"]
        execution_candidate = code_surface_complete and local_resources_complete
        models.append(
            {
                "model_id": model_id,
                "paper_model": model["paper_model"],
                "code_surface_complete": code_surface_complete,
                "local_resources_complete": local_resources_complete,
                "execution_candidate": execution_candidate,
                "resource_identity_status": model["resource_identity"]["status"],
                "formal_reproduction_authorized": False,
                "laion_package": laion_package if model_id == "laion_clap" else None,
                "resources": resources,
                "claim_boundary": (
                    "Local execution candidate only; immutable upstream source and "
                    "checkpoint provenance remain unverified."
                    if execution_candidate
                    else "Local resources and/or the committed evaluation entrypoint are incomplete."
                ),
            }
        )

    candidates = [model["model_id"] for model in models if model["execution_candidate"]]
    return {
        "models": models,
        "summary": {
            "models": len(models),
            "local_resources_complete": sum(
                model["local_resources_complete"] for model in models
            ),
            "code_surface_complete": sum(
                model["code_surface_complete"] for model in models
            ),
            "execution_candidates": candidates,
            "formal_reproduction_authorized": 0,
        },
    }


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    logs_root = (REPOSITORY_ROOT / "logs").resolve()
    if output == logs_root or not output.is_relative_to(logs_root):
        raise ValueError("output must be a unique file below repository logs/")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing probe: {output}")

    registry_path = args.registry.resolve()
    started_at = utc_now()
    registry = load_registry(registry_path)
    probe = probe_registry(registry, REPOSITORY_ROOT)
    git_commit, git_commit_error = command_output(
        REPOSITORY_ROOT, "git", "rev-parse", "HEAD"
    )
    git_status, git_status_error = command_output(
        REPOSITORY_ROOT, "git", "status", "--short"
    )
    report = {
        "schema_version": 1,
        "status": "complete",
        "started_at": started_at,
        "finished_at": utc_now(),
        "operation": (
            "read-only local CLAP resource probe; checkpoint hashing only; "
            "no model loading, network, downloads, or GPU"
        ),
        "git_commit": git_commit,
        "git_status_short": git_status,
        "git_error": git_commit_error or git_status_error,
        "registry": registry_path.relative_to(REPOSITORY_ROOT).as_posix(),
        **probe,
    }
    write_new_json(output, report)

    print("CLAP_RESOURCE_PROBE_STATUS=complete")
    print("GPU_USED=no")
    print("OEA_OFFICIAL_SOURCE_USED=no")
    package = report["models"][0]["laion_package"]
    print(
        "LAION_PACKAGE "
        f"installed={str(package['installed']).lower()} "
        f"version={package['distribution_version']} "
        f"origin={package['module_origin']}"
    )
    for model in report["models"]:
        print(
            "MODEL "
            f"id={model['model_id']} "
            f"code={str(model['code_surface_complete']).lower()} "
            f"resources={str(model['local_resources_complete']).lower()} "
            f"candidate={str(model['execution_candidate']).lower()}"
        )
        for resource in model["resources"]:
            print(
                "RESOURCE "
                f"model={model['model_id']} "
                f"id={resource['resource_id']} "
                f"exists={str(resource['exists']).lower()} "
                f"bytes={resource['size_bytes']} "
                f"sha256={resource['sha256']} "
                f"git_head={None if resource['nested_git'] is None else resource['nested_git']['head']} "
                f"git_dirty={None if resource['nested_git'] is None else resource['nested_git']['dirty']}"
            )
    candidates = report["summary"]["execution_candidates"]
    print("EXECUTION_CANDIDATES=" + (",".join(candidates) if candidates else "none"))
    print(f"OUTPUT={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
