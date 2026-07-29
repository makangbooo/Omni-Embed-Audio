#!/usr/bin/env python3
"""Inspect official OEA model presence without hashing files or using network."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
DEFAULT_REGISTRY = (
    REPOSITORY_ROOT / "configs/checkpoints/official_oea_checkpoints.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def safe_destination(model_root: Path, local_subdir: str) -> Path:
    relative = PurePosixPath(local_subdir)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"unsafe local_subdir: {local_subdir!r}")
    destination = (model_root / Path(*relative.parts)).resolve()
    if destination != model_root and model_root not in destination.parents:
        raise ValueError(f"asset destination escapes model root: {destination}")
    return destination


def load_assets(registry_path: Path) -> list[dict[str, Any]]:
    from scripts.prepare_official_oea_checkpoint import load_checkpoint_registry

    variants = load_checkpoint_registry(registry_path)
    assets: dict[str, dict[str, Any]] = {}
    consumers: dict[str, list[str]] = {}
    for variant_id, variant in variants.items():
        for kind in ("base", "checkpoint"):
            asset = variant[f"{kind}_asset"]
            manifest_path = REPOSITORY_ROOT / variant[f"{kind}_manifest"]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            matches = [row for row in manifest["assets"] if row.get("name") == asset["name"]]
            if len(matches) != 1:
                raise ValueError(f"expected one manifest asset named {asset['name']}")
            resolved = dict(matches[0])
            existing = assets.setdefault(asset["name"], resolved)
            if existing != resolved:
                raise ValueError(f"asset definition drift: {asset['name']}")
            consumers.setdefault(asset["name"], []).append(variant_id)
    return [
        {**asset, "variant_consumers": consumers[name]}
        for name, asset in assets.items()
    ]


def revision_marker(model_root: Path, asset: dict[str, Any]) -> dict[str, Any]:
    marker_path = model_root / ".oea_asset_markers" / f"{asset['name']}.json"
    result: dict[str, Any] = {
        "path": str(marker_path),
        "exists": marker_path.is_file(),
        "matches_expected": False,
    }
    if not marker_path.is_file():
        return result
    try:
        value = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001 - preserve malformed marker evidence
        result["error"] = repr(error)
        return result
    expected = {"repo_id": asset["repo_id"], "revision": asset["revision"]}
    result["value"] = value
    result["matches_expected"] = value == expected
    return result


def inspect_tree(destination: Path) -> dict[str, Any]:
    selected_files: list[dict[str, Any]] = []
    incomplete_files: list[str] = []
    symlinks: list[str] = []
    if not destination.is_dir():
        return {
            "destination_exists": False,
            "selected_file_count": 0,
            "selected_bytes": 0,
            "incomplete_files": [],
            "symlinks": [],
            "selected_files": [],
        }

    for root, directories, filenames in os.walk(destination, followlinks=False):
        root_path = Path(root)
        relative_root = root_path.relative_to(destination)
        for name in list(directories):
            path = root_path / name
            if path.is_symlink():
                symlinks.append(path.relative_to(destination).as_posix())
                directories.remove(name)
        for name in filenames:
            path = root_path / name
            relative = path.relative_to(destination).as_posix()
            if path.is_symlink():
                symlinks.append(relative)
                continue
            if name.endswith(".incomplete"):
                incomplete_files.append(relative)
            if ".cache" in relative_root.parts:
                continue
            stat = path.stat()
            selected_files.append({"path": relative, "size_bytes": stat.st_size})

    selected_files.sort(key=lambda row: row["path"])
    return {
        "destination_exists": True,
        "selected_file_count": len(selected_files),
        "selected_bytes": sum(row["size_bytes"] for row in selected_files),
        "incomplete_files": sorted(incomplete_files),
        "symlinks": sorted(symlinks),
        "selected_files": selected_files,
    }


def inspect_asset(model_root: Path, asset: dict[str, Any]) -> dict[str, Any]:
    destination = safe_destination(model_root, asset["local_subdir"])
    tree = inspect_tree(destination)
    local_by_path = {row["path"]: row for row in tree["selected_files"]}
    required = [
        {
            "path": path,
            "exists": path in local_by_path,
            "size_bytes": local_by_path.get(path, {}).get("size_bytes"),
        }
        for path in asset["required_files"]
    ]
    primary = asset.get("expected_primary_file")
    primary_presence = None
    if isinstance(primary, dict):
        local = local_by_path.get(primary["path"])
        primary_presence = {
            "path": primary["path"],
            "expected_size_bytes": primary["size_bytes"],
            "exists": local is not None,
            "actual_size_bytes": None if local is None else local["size_bytes"],
            "size_matches": (
                local is not None and local["size_bytes"] == primary["size_bytes"]
            ),
            "content_hash_checked": False,
        }
    marker = revision_marker(model_root, asset)
    required_present = all(row["exists"] for row in required)
    primary_size_ok = primary_presence is None or primary_presence["size_matches"]
    presence_candidate = (
        tree["destination_exists"]
        and marker["matches_expected"]
        and required_present
        and primary_size_ok
        and not tree["incomplete_files"]
        and not tree["symlinks"]
    )
    return {
        "name": asset["name"],
        "repo_id": asset["repo_id"],
        "revision": asset["revision"],
        "variant_consumers": asset["variant_consumers"],
        "destination": str(destination),
        **tree,
        "revision_marker": marker,
        "required_files": required,
        "expected_primary_file": primary_presence,
        "presence_candidate": presence_candidate,
        "claim_boundary": (
            "OBSERVED metadata-only presence candidate; full file inventory, "
            "content hashes, and remote provenance were not checked"
            if presence_candidate
            else "OBSERVED local presence is absent or visibly incomplete"
        ),
    }


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    model_root = args.model_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    if model_root == REPOSITORY_ROOT or REPOSITORY_ROOT in model_root.parents:
        raise ValueError("model root must remain outside the Git repository")

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "started_at": utc_now(),
        "finished_at": None,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output("status", "--short"),
        "model_root": str(model_root),
        "operation": "local metadata-only presence scan; no hashing, network, or downloads",
        "claim_boundary": (
            "Presence candidates are not complete resource audits and do not authorize "
            "evaluation until the existing hash/provenance audit passes."
        ),
        "assets": [],
    }
    for asset in load_assets(args.registry.resolve()):
        report["assets"].append(inspect_asset(model_root, asset))
    report["summary"] = {
        "asset_count": len(report["assets"]),
        "presence_candidate_count": sum(
            asset["presence_candidate"] for asset in report["assets"]
        ),
        "absent_or_visibly_incomplete_count": sum(
            not asset["presence_candidate"] for asset in report["assets"]
        ),
    }
    report["finished_at"] = utc_now()
    write_new_json(output, report)

    print("MODEL_PRESENCE_SCAN_STATUS=complete")
    print(f"ASSET_COUNT={report['summary']['asset_count']}")
    for asset in report["assets"]:
        print(
            "ASSET "
            f"name={asset['name']} "
            f"candidate={str(asset['presence_candidate']).lower()} "
            f"files={asset['selected_file_count']} "
            f"bytes={asset['selected_bytes']} "
            f"incomplete={len(asset['incomplete_files'])} "
            f"marker_match={str(asset['revision_marker']['matches_expected']).lower()}"
        )
    print(f"OUTPUT={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
