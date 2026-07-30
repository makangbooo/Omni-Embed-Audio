#!/usr/bin/env python3
"""Verify the pinned M2D release archive and its extracted runtime resources."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE_REVISION = "3d0c4de9447c404a8d3f9f37e04f53bc902e09b3"
EXPECTED_ARCHIVE_SIZE = 1469703420
EXPECTED_ARCHIVE_SHA256 = (
    "fd193ae591720df7f1e27ed728ce127e0309b8bd427f0f4b3e5cd17d7ee5e1e1"
)
CHECKPOINT_NAME = "checkpoint-30.pth"
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


def hash_stream(stream) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(8 * 1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        digest = hash_stream(stream)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": digest,
    }


def main() -> int:
    args = parse_args()
    root = args.model_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite model lock: {output}")

    source = root / "m2d-clap/source"
    marker = source / ".source_revision"
    if marker.read_text(encoding="utf-8").strip() != EXPECTED_SOURCE_REVISION:
        raise RuntimeError("M2D source revision marker mismatch")

    archive = root / "downloads/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025.zip"
    archive_identity = identity(archive)
    if archive_identity["size_bytes"] != EXPECTED_ARCHIVE_SIZE:
        raise RuntimeError("M2D release archive size mismatch")
    if archive_identity["sha256"] != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("M2D release archive SHA256 mismatch")

    checkpoint = (
        root
        / "m2d-clap/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025"
        / CHECKPOINT_NAME
    )
    checkpoint_identity = identity(checkpoint)
    with zipfile.ZipFile(archive) as bundle:
        members = [name for name in bundle.namelist() if name.endswith(f"/{CHECKPOINT_NAME}")]
        if len(members) != 1:
            raise RuntimeError(f"expected one checkpoint archive member, found {members}")
        info = bundle.getinfo(members[0])
        with bundle.open(info) as stream:
            archived_sha256 = hash_stream(stream)
    if info.file_size != checkpoint_identity["size_bytes"]:
        raise RuntimeError("extracted M2D checkpoint size differs from pinned archive")
    if archived_sha256 != checkpoint_identity["sha256"]:
        raise RuntimeError("extracted M2D checkpoint differs from pinned archive")

    tokenizer_root = root / "laion-clap-tokenizers/bert-base-uncased"
    tokenizer = {}
    for name, size in TOKENIZER_FILES.items():
        file_identity = identity(tokenizer_root / name)
        if file_identity["size_bytes"] != size:
            raise RuntimeError(f"BERT tokenizer size mismatch: {name}")
        tokenizer[name] = file_identity

    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    git_status = subprocess.check_output(
        ["git", "status", "--short"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    if git_status:
        raise RuntimeError("M2D model lock requires a clean Git worktree")

    report = {
        "schema_version": 1,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_status_short": git_status,
        "model": "M2D-CLAP",
        "source_revision": EXPECTED_SOURCE_REVISION,
        "source_path": str(source.resolve()),
        "release_archive": archive_identity,
        "checkpoint": checkpoint_identity,
        "checkpoint_archive_member": members[0],
        "bert_tokenizer_path": str(tokenizer_root.resolve()),
        "bert_tokenizer": tokenizer,
        "protocol_status": "controlled_public_code_reproduction",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("M2D_LOCK_STATUS=complete")
    print(f"CHECKPOINT_SHA256={checkpoint_identity['sha256']}")
    print(f"CHECKPOINT_BYTES={checkpoint_identity['size_bytes']}")
    print(f"OUTPUT={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
