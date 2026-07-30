#!/usr/bin/env python3
"""Create a portable identity lock for the official MGA-CLAP resources."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE_REVISION = "48ca5a5cd22cd34427e118bd8cf332090ec54770"
CHECKPOINT_SOURCE = (
    "https://drive.google.com/file/d/1RWTuVMEPy-L0uK6WYIX2wwxHjD1YSQFz/view"
)
TOKENIZER_FILES = {
    "tokenizer.json": 466062,
    "tokenizer_config.json": 48,
    "vocab.txt": 231508,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def tree_identity(root: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    total_bytes = 0
    file_count = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        item = identity(path)
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(str(item["size_bytes"]).encode("ascii") + b"\0")
        digest.update(str(item["sha256"]).encode("ascii") + b"\n")
        total_bytes += int(item["size_bytes"])
        file_count += 1
    if file_count == 0:
        raise RuntimeError(f"MGA source tree is empty: {root}")
    return {
        "path": str(root.resolve()),
        "file_count": file_count,
        "size_bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def partition_git_status(status: str) -> tuple[str, str]:
    ignored_pattern = re.compile(r"\?\? scripts/\.__dpc[0-9a-f]+")
    meaningful = []
    ignored = []
    for line in status.splitlines():
        (ignored if ignored_pattern.fullmatch(line) else meaningful).append(line)
    return "\n".join(meaningful), "\n".join(ignored)


def main() -> int:
    args = parse_args()
    model_root = args.model_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite MGA model lock: {output}")

    source = model_root / "mga-clap/source"
    revision = (source / ".source_revision").read_text(encoding="utf-8").strip()
    if revision != EXPECTED_SOURCE_REVISION:
        raise RuntimeError("MGA source revision marker mismatch")
    for relative in ("models/ase_model.py", "settings/inference_example.yaml"):
        if not (source / relative).is_file():
            raise RuntimeError(f"required MGA source file is missing: {relative}")

    checkpoint = model_root / "mga-clap/pretrained_models/models/model.pt"
    checkpoint_identity = identity(checkpoint)
    if int(checkpoint_identity["size_bytes"]) < 1024 * 1024:
        raise RuntimeError("MGA checkpoint is unexpectedly smaller than 1 MiB")
    with checkpoint.open("rb") as stream:
        prefix = stream.read(256).lower()
    if b"<html" in prefix or b"<!doctype" in prefix:
        raise RuntimeError("MGA checkpoint is an HTML response")

    tokenizer_root = model_root / "laion-clap-tokenizers/bert-base-uncased"
    tokenizer = {}
    for name, expected_size in TOKENIZER_FILES.items():
        item = identity(tokenizer_root / name)
        if item["size_bytes"] != expected_size:
            raise RuntimeError(f"BERT tokenizer size mismatch: {name}")
        tokenizer[name] = item

    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    raw_status = subprocess.check_output(
        ["git", "status", "--short"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    git_status, git_status_ignored = partition_git_status(raw_status)
    if git_status:
        raise RuntimeError("MGA resource lock requires a clean Git worktree")

    report = {
        "schema_version": 1,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_status_short": git_status,
        "git_status_ignored_ephemeral": git_status_ignored,
        "model": "MGA-CLAP",
        "source_revision": revision,
        "source_tree": tree_identity(source),
        "checkpoint": checkpoint_identity,
        "checkpoint_source": CHECKPOINT_SOURCE,
        "checkpoint_identity_status": "observed_official_download_no_published_hash",
        "bert_tokenizer_path": str(tokenizer_root.resolve()),
        "bert_tokenizer": tokenizer,
        "protocol_status": "controlled_public_code_reproduction",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("MGA_LOCK_STATUS=complete")
    print(f"CHECKPOINT_SHA256={checkpoint_identity['sha256']}")
    print(f"CHECKPOINT_BYTES={checkpoint_identity['size_bytes']}")
    print(f"OUTPUT={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
