#!/usr/bin/env python3
"""Audit immutable Hugging Face resources without downloading or modifying them."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import HfApi, __version__ as huggingface_hub_version

try:
    from scripts.download_model_assets import (
        REPOSITORY_ROOT,
        ensure_external_model_root,
        pinned_file_specs,
        repository_info,
        selected,
        sha256_file,
        validate_remote_file_pins,
    )
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from download_model_assets import (
        REPOSITORY_ROOT,
        ensure_external_model_root,
        pinned_file_specs,
        repository_info,
        selected,
        sha256_file,
        validate_remote_file_pins,
    )


SAFE_ASSET_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
        help="Resource manifest; repeat for multiple manifests.",
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_output(*arguments: str) -> str:
    import subprocess

    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def git_blob_sha1(path: Path) -> str:
    """Compute the Git object ID for a regular non-LFS file."""

    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(value: str, label: str) -> PurePosixPath:
    candidate = value.replace("\\", "/")
    path = PurePosixPath(candidate)
    if (
        not candidate
        or candidate.startswith("/")
        or WINDOWS_DRIVE.match(candidate)
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != candidate
    ):
        raise ValueError(f"unsafe {label}: {value!r}")
    return path


def destination_for(model_root: Path, asset: dict[str, Any]) -> Path:
    relative = safe_relative_path(asset["local_subdir"], "local_subdir")
    destination = (model_root / Path(*relative.parts)).resolve()
    if model_root != destination and model_root not in destination.parents:
        raise ValueError(f"asset destination escapes model root: {destination}")
    return destination


def marker_for(model_root: Path, asset: dict[str, Any]) -> Path:
    name = asset["name"]
    if not SAFE_ASSET_NAME.fullmatch(name):
        raise ValueError(f"unsafe asset name: {name!r}")
    return model_root / ".oea_asset_markers" / f"{name}.json"


def validate_manifest(specification: dict[str, Any], path: Path) -> None:
    if specification.get("schema_version") != 1:
        raise ValueError(f"unsupported schema_version in {path}")
    assets = specification.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError(f"manifest has no assets: {path}")
    required = {"name", "repo_id", "revision", "local_subdir", "required_files"}
    for asset in assets:
        if not isinstance(asset, dict) or not required <= set(asset):
            raise ValueError(f"malformed asset in {path}: {asset!r}")
        if not isinstance(asset["name"], str) or not SAFE_ASSET_NAME.fullmatch(
            asset["name"]
        ):
            raise ValueError(f"unsafe asset name in {path}: {asset['name']!r}")
        if (
            not isinstance(asset["revision"], str)
            or len(asset["revision"]) != 40
            or not all(
                character in "0123456789abcdef" for character in asset["revision"]
            )
        ):
            raise ValueError(f"asset revision is not an immutable SHA: {asset['name']}")
        if not isinstance(asset["required_files"], list) or not asset["required_files"]:
            raise ValueError(f"asset has no required_files: {asset['name']}")
        safe_relative_path(asset["local_subdir"], "local_subdir")
        for filename in asset["required_files"]:
            safe_relative_path(filename, "required file")
        if len(set(asset["required_files"])) != len(asset["required_files"]):
            raise ValueError(f"duplicate required_files for asset: {asset['name']}")
        for item in pinned_file_specs(asset):
            safe_relative_path(item["path"], "pinned file")


def remote_inventory(api: HfApi, asset: dict[str, Any]) -> list[dict[str, Any]]:
    info = repository_info(
        api,
        repo_id=asset["repo_id"],
        repo_type=asset.get("repo_type", "model"),
        revision=asset["revision"],
    )
    if info.sha != asset["revision"]:
        raise RuntimeError(
            f"resolved revision mismatch for {asset['repo_id']}: "
            f"{info.sha} != {asset['revision']}"
        )

    inventory: list[dict[str, Any]] = []
    for sibling in info.siblings or []:
        if not selected(sibling.rfilename, asset.get("allow_patterns")):
            continue
        safe_relative_path(sibling.rfilename, "remote file")
        lfs = getattr(sibling, "lfs", None)
        if isinstance(lfs, dict):
            lfs_sha256 = lfs.get("sha256")
        else:
            lfs_sha256 = getattr(lfs, "sha256", None) if lfs else None
        inventory.append(
            {
                "path": sibling.rfilename,
                "size_bytes": getattr(sibling, "size", None),
                "lfs_sha256": lfs_sha256,
                "blob_id": getattr(sibling, "blob_id", None),
            }
        )
    inventory.sort(key=lambda item: item["path"])
    if not inventory:
        raise RuntimeError(f"remote selected inventory is empty: {asset['name']}")
    if len({item["path"] for item in inventory}) != len(inventory):
        raise RuntimeError(f"remote inventory has duplicate paths: {asset['name']}")
    missing_size = [item["path"] for item in inventory if item["size_bytes"] is None]
    if missing_size:
        raise RuntimeError(
            f"remote inventory is missing file sizes for {asset['name']}: {missing_size}"
        )
    missing_identity = [
        item["path"]
        for item in inventory
        if item["lfs_sha256"] is None and item["blob_id"] is None
    ]
    if missing_identity:
        raise RuntimeError(
            f"remote inventory has no content identity for {asset['name']}: "
            f"{missing_identity}"
        )
    required_missing = sorted(
        set(asset["required_files"]) - {item["path"] for item in inventory}
    )
    if required_missing:
        raise RuntimeError(
            f"required files absent remotely for {asset['name']}: {required_missing}"
        )
    validate_remote_file_pins(
        asset_name=asset["name"],
        remote_files=inventory,
        specifications=pinned_file_specs(asset),
    )
    return inventory


def local_inventory(destination: Path) -> tuple[dict[str, Path], list[str]]:
    files: dict[str, Path] = {}
    unsafe_links: list[str] = []
    for path in sorted(destination.rglob("*")):
        relative = path.relative_to(destination).as_posix()
        if ".cache" in path.relative_to(destination).parts:
            continue
        if path.is_symlink():
            unsafe_links.append(relative)
            continue
        if path.is_file():
            files[relative] = path
    return files, unsafe_links


def audit_local_asset(
    *,
    model_root: Path,
    asset: dict[str, Any],
    remote_files: list[dict[str, Any]],
) -> dict[str, Any]:
    model_root = model_root.resolve()
    destination = destination_for(model_root, asset)
    marker = marker_for(model_root, asset)
    report: dict[str, Any] = {
        "name": asset["name"],
        "repo_id": asset["repo_id"],
        "repo_type": asset.get("repo_type", "model"),
        "revision": asset["revision"],
        "destination": str(destination),
        "marker": str(marker),
        "status": "running",
        "remote_files": remote_files,
        "local_files": [],
        "missing_files": [],
        "extra_files": [],
        "incomplete_files": [],
        "unsafe_symlinks": [],
        "errors": [],
    }

    if not destination.is_dir():
        report.update(
            {
                "status": "incomplete",
                "missing_files": [item["path"] for item in remote_files],
                "errors": ["destination directory does not exist"],
            }
        )
        return report

    identity = {"repo_id": asset["repo_id"], "revision": asset["revision"]}
    if not marker.is_file():
        report["errors"].append("non-empty or partial destination has no revision marker")
    else:
        try:
            marker_identity = json.loads(marker.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - preserve malformed marker evidence
            report["errors"].append(f"unable to parse revision marker: {exc!r}")
        else:
            report["marker_identity"] = marker_identity
            if marker_identity != identity:
                report["errors"].append(
                    f"revision marker mismatch: {marker_identity!r} != {identity!r}"
                )

    report["incomplete_files"] = sorted(
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*.incomplete")
        if path.is_file()
    )
    local_by_path, unsafe_links = local_inventory(destination)
    report["unsafe_symlinks"] = unsafe_links
    if unsafe_links:
        report["errors"].append("symlinks are not allowed in local_dir resources")

    remote_by_path = {item["path"]: item for item in remote_files}
    pinned_by_path = {item["path"]: item for item in pinned_file_specs(asset)}
    report["missing_files"] = sorted(set(remote_by_path) - set(local_by_path))
    report["extra_files"] = sorted(set(local_by_path) - set(remote_by_path))
    if report["extra_files"]:
        report["errors"].append("local resource contains files absent from pinned snapshot")

    for relative in sorted(set(local_by_path) & set(remote_by_path)):
        path = local_by_path[relative]
        remote = remote_by_path[relative]
        size = path.stat().st_size
        digest = sha256_file(path)
        lfs_digest = remote["lfs_sha256"]
        blob_id = remote.get("blob_id")
        size_matches = size == remote["size_bytes"]
        lfs_matches = None if lfs_digest is None else digest == lfs_digest
        blob_matches = None
        if lfs_digest is None and blob_id is not None:
            blob_matches = git_blob_sha1(path) == blob_id
        report["local_files"].append(
            {
                "path": relative,
                "size_bytes": size,
                "sha256": digest,
                "remote_size_bytes": remote["size_bytes"],
                "size_matches_remote": size_matches,
                "remote_lfs_sha256": lfs_digest,
                "matches_remote_lfs_sha256": lfs_matches,
                "remote_git_blob_id": blob_id,
                "matches_remote_git_blob_id": blob_matches,
            }
        )
        if not size_matches:
            report["errors"].append(f"size mismatch: {relative}")
        if lfs_matches is False:
            report["errors"].append(f"LFS SHA256 mismatch: {relative}")
        if blob_matches is False:
            report["errors"].append(f"Git blob mismatch: {relative}")
        pinned_sha256 = pinned_by_path.get(relative, {}).get("sha256")
        if pinned_sha256 is not None and digest != pinned_sha256:
            report["errors"].append(f"pinned SHA256 mismatch: {relative}")

    report["expected_file_count"] = len(remote_files)
    report["verified_local_file_count"] = len(report["local_files"])
    report["expected_bytes"] = sum(item["size_bytes"] for item in remote_files)
    report["verified_local_bytes"] = sum(
        item["size_bytes"] for item in report["local_files"]
    )

    if report["errors"]:
        report["status"] = "failed"
    elif report["incomplete_files"] or report["missing_files"]:
        report["status"] = "incomplete"
    else:
        report["status"] = "complete"
    return report


def combined_status(assets: list[dict[str, Any]]) -> tuple[str, int]:
    statuses = {asset["status"] for asset in assets}
    if statuses <= {"complete"}:
        return "complete", 0
    if "failed" in statuses:
        return "failed", 1
    return "incomplete", 2


def main() -> int:
    args = parse_args()
    model_root = args.model_root.resolve()
    output = args.output.resolve()
    ensure_external_model_root(model_root)

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output("status", "--short"),
        "model_root": str(model_root),
        "manifests": [],
        "hf_endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        "huggingface_hub_version": huggingface_hub_version,
        "operation": "read-only metadata and local-content audit; no downloads",
        "assets": [],
    }
    write_json(output, report)

    try:
        api = HfApi(endpoint=report["hf_endpoint"])
        seen_names: set[str] = set()
        seen_destinations: set[str] = set()
        for manifest_argument in args.manifest:
            manifest = manifest_argument.resolve()
            specification = json.loads(manifest.read_text(encoding="utf-8"))
            validate_manifest(specification, manifest)
            report["manifests"].append(str(manifest))
            for asset in specification["assets"]:
                name = asset["name"]
                destination = str(destination_for(model_root, asset))
                if name in seen_names:
                    raise ValueError(f"duplicate asset name across manifests: {name}")
                if destination in seen_destinations:
                    raise ValueError(
                        f"duplicate local_subdir across manifests: {destination}"
                    )
                seen_names.add(name)
                seen_destinations.add(destination)
                try:
                    remote_files = remote_inventory(api, asset)
                    asset_report = audit_local_asset(
                        model_root=model_root,
                        asset=asset,
                        remote_files=remote_files,
                    )
                except Exception as exc:  # noqa: BLE001 - audit every remaining asset
                    asset_report = {
                        "name": name,
                        "repo_id": asset["repo_id"],
                        "revision": asset["revision"],
                        "destination": destination,
                        "status": "failed",
                        "errors": [repr(exc)],
                    }
                report["assets"].append(asset_report)
                write_json(output, report)

        report["status"], exit_code = combined_status(report["assets"])
        report["summary"] = {
            status: sum(asset["status"] == status for asset in report["assets"])
            for status in ("complete", "incomplete", "failed")
        }
    except Exception as exc:  # noqa: BLE001 - preserve fatal audit evidence
        report["status"] = "failed"
        report["fatal_error"] = repr(exc)
        exit_code = 1
    report["finished_at"] = utc_now()
    write_json(output, report)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
