#!/usr/bin/env python3
"""Verify downloaded LAION-CLAP resources and write a portable local model lock."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from AudioRetrieval.models.laion_clap_tokenizers import local_tokenizer_redirect


EXPECTED_DISTRIBUTIONS = {
    "braceexpand": "0.1.7",
    "ftfy": "6.1.1",
    "laion_clap": "1.1.6",
    "progressbar": "2.5",
    "torchlibrosa": "0.1.0",
    "webdataset": "0.2.48",
    "wget": "3.2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--overlay-root", type=Path, required=True)
    parser.add_argument("--download-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def distribution_versions(overlay_root: Path) -> dict[str, str]:
    found = {
        distribution.metadata["Name"].lower().replace("-", "_"): distribution.version
        for distribution in importlib.metadata.distributions(path=[str(overlay_root)])
    }
    missing = sorted(set(EXPECTED_DISTRIBUTIONS) - set(found))
    if missing:
        raise RuntimeError(f"LAION overlay distributions are missing: {missing}")
    actual = {name: found[name] for name in EXPECTED_DISTRIBUTIONS}
    if actual != EXPECTED_DISTRIBUTIONS:
        raise RuntimeError(
            f"LAION overlay distribution versions differ: {actual!r}"
        )
    return actual


def verify_downloads(
    model_root: Path, manifest_path: Path, report_path: Path
) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise RuntimeError("LAION resource download report is not complete")
    report_assets = {asset["name"]: asset for asset in report.get("assets", [])}
    verified: list[dict[str, Any]] = []
    for asset in manifest["assets"]:
        recorded = report_assets.get(asset["name"])
        if not isinstance(recorded, dict) or recorded.get("status") != "complete":
            raise RuntimeError(f"download report lacks asset {asset['name']}")
        destination = (model_root / asset["local_subdir"]).resolve()
        files = []
        for item in recorded.get("local_files", []):
            path = destination / item["path"]
            actual = identity(path)
            if (
                actual["size_bytes"] != item["size_bytes"]
                or actual["sha256"] != item["sha256"]
            ):
                raise RuntimeError(
                    f"downloaded resource changed after verification: {path}"
                )
            files.append(actual)
        if not files:
            raise RuntimeError(f"download report has no files for {asset['name']}")
        verified.append(
            {
                "name": asset["name"],
                "repo_id": asset["repo_id"],
                "revision": asset["revision"],
                "local_subdir": asset["local_subdir"],
                "files": files,
            }
        )
    return verified


def tokenizer_paths(model_root: Path) -> dict[str, str]:
    return {
        "bert-base-uncased": str(
            (model_root / "laion-clap-tokenizers/bert-base-uncased").resolve()
        ),
        "roberta-base": str(
            (model_root / "laion-clap-tokenizers/roberta-base").resolve()
        ),
        "facebook/bart-base": str(
            (model_root / "laion-clap-tokenizers/bart-base").resolve()
        ),
    }


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> int:
    args = parse_args()
    model_root = args.model_root.expanduser().resolve()
    overlay_root = args.overlay_root.expanduser().resolve()
    manifest = args.manifest.resolve()
    requirements = args.requirements.resolve()
    download_report = args.download_report.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite LAION lock: {output}")
    if not overlay_root.is_dir():
        raise FileNotFoundError(overlay_root)
    git_commit = git_output("rev-parse", "HEAD")
    git_status = git_output("status", "--short")
    if git_status:
        raise RuntimeError("LAION lock build requires a clean Git worktree")

    versions = distribution_versions(overlay_root)
    resources = verify_downloads(model_root, manifest, download_report)
    sys.path.insert(0, str(overlay_root))
    paths = tokenizer_paths(model_root)
    with local_tokenizer_redirect(paths):
        module = importlib.import_module("laion_clap")

    runtime_versions = {}
    for name in ("numpy", "torch", "torchaudio", "torchvision", "transformers"):
        runtime_versions[name] = importlib.metadata.version(name)
    report = {
        "schema_version": 1,
        "status": "complete",
        "started_at": utc_now(),
        "finished_at": utc_now(),
        "git_commit": git_commit,
        "git_status_short": git_status,
        "model_id": "laion_clap",
        "paper_model": "LAION-CLAP",
        "protocol_status": "controlled_public_code_reproduction",
        "claim_boundary": (
            "The paper/public repository do not publish exact dependency and resource "
            "revisions; this lock fixes the public-code defaults without claiming the "
            "authors' hidden environment."
        ),
        "package_overlay": {
            "root": str(overlay_root),
            "requirements": identity(requirements),
            "distributions": versions,
            "laion_clap_module": str(Path(module.__file__).resolve()),
        },
        "runtime_distributions": runtime_versions,
        "resource_manifest": identity(manifest),
        "download_report": identity(download_report),
        "resources": resources,
        "checkpoint": identity(model_root / "laion-clap/630k-audioset-best.pt"),
        "tokenizer_paths": paths,
        "gpu_used": False,
        "model_loaded": False,
        "network_used_during_import": False,
    }
    write_new_json(output, report)
    print("LAION_LOCK_STATUS=complete")
    print(f"GIT_COMMIT={git_commit}")
    print("GPU_USED=no")
    print("OEA_OFFICIAL_SOURCE_USED=yes")
    print("LAION_CLAP_VERSION=1.1.6")
    print(f"CHECKPOINT_SHA256={report['checkpoint']['sha256']}")
    print(f"CHECKPOINT_BYTES={report['checkpoint']['size_bytes']}")
    print(f"OUTPUT={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
