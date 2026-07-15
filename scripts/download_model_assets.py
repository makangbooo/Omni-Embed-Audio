#!/usr/bin/env python3
"""Download immutable Hugging Face assets and produce an auditable manifest."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, __version__ as huggingface_hub_version
from huggingface_hub import snapshot_download


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def selected(filename: str, patterns: list[str] | None) -> bool:
    if patterns is None:
        return True
    return any(fnmatch.fnmatch(filename, pattern) for pattern in patterns)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def local_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(directory).parts
    )


def ensure_external_model_root(model_root: Path) -> None:
    resolved = model_root.resolve()
    if resolved == REPOSITORY_ROOT or REPOSITORY_ROOT in resolved.parents:
        raise ValueError(
            f"model root must be outside the Git repository: {resolved}"
        )


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    model_root = args.model_root.resolve()
    output = args.output.resolve()
    ensure_external_model_root(model_root)
    model_root.mkdir(parents=True, exist_ok=True)

    specification = json.loads(manifest_path.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output("status", "--short"),
        "manifest": str(manifest_path),
        "model_root": str(model_root),
        "hf_endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        "huggingface_hub_version": huggingface_hub_version,
        "assets": [],
    }
    write_json(output, report)

    api = HfApi(
        endpoint=os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    )
    marker_root = model_root / ".oea_asset_markers"
    marker_root.mkdir(parents=True, exist_ok=True)

    try:
        for asset in specification["assets"]:
            name = asset["name"]
            repo_id = asset["repo_id"]
            revision = asset["revision"]
            patterns = asset.get("allow_patterns")
            destination = model_root / asset["local_subdir"]
            marker = marker_root / f"{name}.json"
            identity = {"repo_id": repo_id, "revision": revision}

            if marker.exists():
                existing_identity = json.loads(marker.read_text(encoding="utf-8"))
                if existing_identity != identity:
                    raise RuntimeError(
                        f"asset marker mismatch for {name}: {existing_identity} != {identity}"
                    )
            elif destination.exists() and any(destination.iterdir()):
                raise RuntimeError(
                    f"non-empty destination has no revision marker: {destination}"
                )
            else:
                write_json(marker, identity)

            info = api.model_info(
                repo_id=repo_id,
                revision=revision,
                files_metadata=True,
            )
            if info.sha != revision:
                raise RuntimeError(
                    f"resolved revision mismatch for {repo_id}: {info.sha} != {revision}"
                )

            remote_files: list[dict[str, Any]] = []
            expected_bytes = 0
            remote_sha256: dict[str, str] = {}
            for sibling in info.siblings or []:
                if not selected(sibling.rfilename, patterns):
                    continue
                size = getattr(sibling, "size", None)
                lfs = getattr(sibling, "lfs", None)
                if isinstance(lfs, dict):
                    lfs_sha256 = lfs.get("sha256")
                else:
                    lfs_sha256 = getattr(lfs, "sha256", None) if lfs else None
                remote_files.append(
                    {
                        "path": sibling.rfilename,
                        "size_bytes": size,
                        "lfs_sha256": lfs_sha256,
                    }
                )
                if size is not None:
                    expected_bytes += int(size)
                if lfs_sha256:
                    remote_sha256[sibling.rfilename] = lfs_sha256

            free_bytes = shutil.disk_usage(model_root).free
            required_free = int(expected_bytes * 1.2) + 5 * 1024**3
            if expected_bytes and free_bytes < required_free:
                raise RuntimeError(
                    f"insufficient free space for {name}: free={free_bytes}, "
                    f"required={required_free}"
                )

            asset_report: dict[str, Any] = {
                "name": name,
                "repo_id": repo_id,
                "requested_revision": revision,
                "resolved_revision": info.sha,
                "destination": str(destination),
                "allow_patterns": patterns,
                "expected_download_bytes": expected_bytes,
                "remote_files": remote_files,
                "status": "downloading",
                "local_files": [],
            }
            report["assets"].append(asset_report)
            write_json(output, report)

            snapshot_download(
                repo_id=repo_id,
                repo_type=asset.get("repo_type", "model"),
                revision=revision,
                local_dir=destination,
                allow_patterns=patterns,
            )

            missing = [
                filename
                for filename in asset["required_files"]
                if not (destination / filename).is_file()
            ]
            if missing:
                raise RuntimeError(f"required files missing for {name}: {missing}")

            for path in local_files(destination):
                relative = path.relative_to(destination).as_posix()
                digest = sha256_file(path)
                remote_digest = remote_sha256.get(relative)
                matches_remote = None if remote_digest is None else digest == remote_digest
                if matches_remote is False:
                    raise RuntimeError(f"SHA256 mismatch: {repo_id}/{relative}")
                asset_report["local_files"].append(
                    {
                        "path": relative,
                        "size_bytes": path.stat().st_size,
                        "sha256": digest,
                        "matches_remote_lfs_sha256": matches_remote,
                    }
                )

            asset_report["status"] = "complete"
            write_json(output, report)

        report["status"] = "complete"
        report["finished_at"] = utc_now()
        write_json(output, report)
        return 0
    except Exception as exc:  # noqa: BLE001 - preserve download failures in audit output
        report["status"] = "failed"
        report["finished_at"] = utc_now()
        report["error"] = repr(exc)
        write_json(output, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
