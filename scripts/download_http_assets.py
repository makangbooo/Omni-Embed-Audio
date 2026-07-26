#!/usr/bin/env python3
"""Download checksum-pinned HTTP assets without placing data in Git."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--curl-retries", type=int, default=8)
    args = parser.parse_args()
    if args.curl_retries < 0:
        parser.error("--curl-retries must be non-negative")
    return args


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


def hash_file(path: Path, algorithm: str = "md5") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_external_data_root(data_root: Path) -> None:
    resolved = data_root.resolve()
    if resolved == REPOSITORY_ROOT or REPOSITORY_ROOT in resolved.parents:
        raise ValueError(f"data root must be outside the Git repository: {resolved}")


def validate_specification(specification: dict[str, Any]) -> None:
    if specification.get("schema_version") != 1:
        raise ValueError("unsupported resource manifest schema")
    files = specification.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("resource manifest must contain a non-empty files list")
    local_subdir = specification.get("local_subdir")
    if (
        not isinstance(local_subdir, str)
        or Path(local_subdir).is_absolute()
        or ".." in Path(local_subdir).parts
    ):
        raise ValueError(f"unsafe or invalid local_subdir: {local_subdir!r}")
    names: set[str] = set()
    for asset in files:
        name = asset.get("name")
        checksum = asset.get("md5")
        sha256 = asset.get("sha256")
        size_bytes = asset.get("size_bytes")
        url = asset.get("url")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"unsafe or invalid file name: {name!r}")
        if name in names:
            raise ValueError(f"duplicate file name: {name}")
        names.add(name)
        if not isinstance(url, str) or not url.startswith("https://"):
            raise ValueError(f"asset URL must use HTTPS: {name}")
        if checksum is None and sha256 is None:
            raise ValueError(f"asset must pin at least one checksum: {name}")
        if checksum is not None:
            if not isinstance(checksum, str) or len(checksum) != 32:
                raise ValueError(f"invalid MD5 checksum: {name}")
            int(checksum, 16)
        if sha256 is not None:
            if not isinstance(sha256, str) or len(sha256) != 64:
                raise ValueError(f"invalid SHA256 checksum: {name}")
            int(sha256, 16)
        if size_bytes is not None and (
            not isinstance(size_bytes, int) or size_bytes < 0
        ):
            raise ValueError(f"invalid size_bytes: {name}")


def curl_download(url: str, partial: Path, retries: int) -> None:
    if shutil.which("curl") is None:
        raise RuntimeError("curl is required for resumable HTTP downloads")
    command = [
        "curl",
        "--location",
        "--fail",
        "--show-error",
        "--http1.1",
        "--continue-at",
        "-",
        "--connect-timeout",
        "60",
        "--speed-time",
        "120",
        "--speed-limit",
        "1024",
        "--output",
        str(partial),
        url,
    ]
    for attempt in range(1, retries + 2):
        size_before = partial.stat().st_size if partial.exists() else 0
        completed = subprocess.run(command, check=False)
        if completed.returncode == 0:
            return
        size_after = partial.stat().st_size if partial.exists() else 0
        if attempt > retries:
            raise subprocess.CalledProcessError(completed.returncode, command)
        print(
            "[WARN] curl attempt "
            f"{attempt}/{retries + 1} failed with exit {completed.returncode}; "
            f"preserving partial bytes {size_before} -> {size_after} and "
            "resuming in 15 seconds",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(15)


def verify_asset_file(path: Path, asset: dict[str, Any]) -> dict[str, Any]:
    size_bytes = path.stat().st_size
    expected_size = asset.get("size_bytes")
    if expected_size is not None and size_bytes != expected_size:
        raise RuntimeError(
            f"file size mismatch for {path}: {size_bytes} != {expected_size}"
        )

    checksums: dict[str, str] = {}
    expected_md5 = asset.get("md5")
    if expected_md5 is not None:
        checksums["md5"] = hash_file(path)
        if checksums["md5"] != expected_md5:
            raise RuntimeError(
                f"MD5 mismatch for {path}: {checksums['md5']} != {expected_md5}"
            )

    expected_sha256 = asset.get("sha256")
    if expected_sha256 is not None:
        checksums["sha256"] = hash_file(path, "sha256")
        if checksums["sha256"] != expected_sha256:
            raise RuntimeError(
                f"SHA256 mismatch for {path}: "
                f"{checksums['sha256']} != {expected_sha256}"
            )

    return {"size_bytes": size_bytes, **checksums}


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    ensure_external_data_root(data_root)

    specification = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_specification(specification)
    destination = data_root / specification["local_subdir"]
    destination.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output("status", "--short"),
        "manifest": str(manifest_path),
        "data_root": str(data_root),
        "destination": str(destination),
        "source_record": specification["source_record"],
        "dataset": specification["dataset"],
        "version": specification["version"],
        "split": specification["split"],
        "expected_examples": specification["expected_examples"],
        "assets": [],
    }
    write_json(output, report)

    try:
        for asset in specification["files"]:
            final_path = destination / asset["name"]
            partial_path = final_path.with_name(final_path.name + ".part")
            asset_report: dict[str, Any] = {
                "name": asset["name"],
                "kind": asset["kind"],
                "url": asset["url"],
                "expected_md5": asset.get("md5"),
                "expected_sha256": asset.get("sha256"),
                "expected_size_bytes": asset.get("size_bytes"),
                "path": str(final_path),
                "partial_path": str(partial_path),
                "status": "pending",
            }
            report["assets"].append(asset_report)
            write_json(output, report)

            if final_path.exists():
                try:
                    verified = verify_asset_file(final_path, asset)
                except RuntimeError as exc:
                    raise RuntimeError(
                        "existing final file failed pinned verification; "
                        f"refusing to overwrite: {final_path} ({exc})"
                    ) from exc
                final_status = "verified_existing"
            else:
                asset_report["status"] = "downloading"
                asset_report["partial_size_before_bytes"] = (
                    partial_path.stat().st_size if partial_path.exists() else 0
                )
                write_json(output, report)
                curl_download(asset["url"], partial_path, args.curl_retries)
                try:
                    verified = verify_asset_file(partial_path, asset)
                except RuntimeError as exc:
                    raise RuntimeError(
                        "downloaded file failed pinned verification; partial file "
                        f"preserved: {partial_path} ({exc})"
                    ) from exc
                partial_path.replace(final_path)
                final_status = "downloaded_and_verified"

            asset_report.update(verified)
            asset_report["status"] = final_status
            asset_report["finished_at"] = utc_now()
            write_json(output, report)

        report["status"] = "complete"
        report["finished_at"] = utc_now()
        write_json(output, report)
        return 0
    except Exception as exc:  # noqa: BLE001 - preserve complete failure evidence
        report["status"] = "failed"
        report["finished_at"] = utc_now()
        report["error"] = repr(exc)
        write_json(output, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
