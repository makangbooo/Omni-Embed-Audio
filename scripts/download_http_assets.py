#!/usr/bin/env python3
"""Download checksum-pinned HTTP assets without placing data in Git."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
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
    names: set[str] = set()
    for asset in files:
        name = asset.get("name")
        checksum = asset.get("md5")
        url = asset.get("url")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"unsafe or invalid file name: {name!r}")
        if name in names:
            raise ValueError(f"duplicate file name: {name}")
        names.add(name)
        if not isinstance(url, str) or not url.startswith("https://"):
            raise ValueError(f"asset URL must use HTTPS: {name}")
        if not isinstance(checksum, str) or len(checksum) != 32:
            raise ValueError(f"invalid MD5 checksum: {name}")
        int(checksum, 16)


def curl_download(url: str, partial: Path, retries: int) -> None:
    if shutil.which("curl") is None:
        raise RuntimeError("curl is required for resumable HTTP downloads")
    subprocess.run(
        [
            "curl",
            "--location",
            "--fail",
            "--show-error",
            "--continue-at",
            "-",
            "--retry",
            str(retries),
            "--retry-all-errors",
            "--retry-delay",
            "15",
            "--connect-timeout",
            "60",
            "--speed-time",
            "120",
            "--speed-limit",
            "1024",
            "--output",
            str(partial),
            url,
        ],
        check=True,
    )


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    ensure_external_data_root(data_root)

    specification = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_specification(specification)
    destination = data_root / "clotho_v2.1" / "source"
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
                "expected_md5": asset["md5"],
                "path": str(final_path),
                "partial_path": str(partial_path),
                "status": "pending",
            }
            report["assets"].append(asset_report)
            write_json(output, report)

            if final_path.exists():
                existing_md5 = hash_file(final_path)
                if existing_md5 != asset["md5"]:
                    raise RuntimeError(
                        f"existing final file has wrong MD5; refusing to overwrite: "
                        f"{final_path} ({existing_md5})"
                    )
                asset_report["status"] = "verified_existing"
            else:
                asset_report["status"] = "downloading"
                asset_report["partial_size_before_bytes"] = (
                    partial_path.stat().st_size if partial_path.exists() else 0
                )
                write_json(output, report)
                curl_download(asset["url"], partial_path, args.curl_retries)
                downloaded_md5 = hash_file(partial_path)
                asset_report["downloaded_md5"] = downloaded_md5
                if downloaded_md5 != asset["md5"]:
                    raise RuntimeError(
                        f"downloaded file MD5 mismatch; partial file preserved: "
                        f"{partial_path} ({downloaded_md5})"
                    )
                partial_path.replace(final_path)
                asset_report["status"] = "downloaded_and_verified"

            asset_report["size_bytes"] = final_path.stat().st_size
            asset_report["md5"] = hash_file(final_path)
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
