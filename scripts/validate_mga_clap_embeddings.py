#!/usr/bin/env python3
"""Validate MGA-CLAP embeddings and preserve generation evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--expected-audio", type=int, required=True)
    parser.add_argument("--expected-captions", type=int, required=True)
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


def load_embeddings(path: Path, rows: int) -> tuple[dict[str, object], int]:
    with np.load(path, allow_pickle=False) as archive:
        embeddings = np.asarray(archive["embeddings"])
        clip_ids = np.asarray(archive["clip_ids"])
        keys = sorted(archive.files)
    if embeddings.ndim != 2 or embeddings.shape[0] != rows:
        raise RuntimeError(f"embedding shape mismatch: {path} {embeddings.shape}")
    if clip_ids.shape != (rows,):
        raise RuntimeError(f"clip ID shape mismatch: {path} {clip_ids.shape}")
    if not np.isfinite(embeddings).all():
        raise RuntimeError(f"non-finite embeddings: {path}")
    norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise RuntimeError(f"embeddings are not L2 normalized: {path}")
    return ({
        "identity": identity(path),
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "archive_keys": keys,
        "all_finite": True,
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
    }, int(embeddings.shape[1]))


def partition_git_status(status: str) -> tuple[str, str]:
    ignored_pattern = re.compile(r"\?\? scripts/\.__dpc[0-9a-f]+")
    meaningful = []
    ignored = []
    for line in status.splitlines():
        (ignored if ignored_pattern.fullmatch(line) else meaningful).append(line)
    return "\n".join(meaningful), "\n".join(ignored)


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    lock = json.loads(args.model_lock.read_text(encoding="utf-8"))
    if lock.get("status") != "complete" or lock.get("model") != "MGA-CLAP":
        raise RuntimeError("MGA model lock is incomplete or has the wrong model")
    audio, audio_dim = load_embeddings(
        args.embedding_dir / "audio_embeddings.npz", args.expected_audio
    )
    captions, caption_dim = load_embeddings(
        args.embedding_dir / "caption_embeddings.npz", args.expected_captions
    )
    if audio_dim != caption_dim:
        raise RuntimeError(f"audio/text dimension mismatch: {audio_dim}/{caption_dim}")

    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    raw_status = subprocess.check_output(
        ["git", "status", "--short"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    git_status, git_status_ignored = partition_git_status(raw_status)
    if git_status:
        raise RuntimeError("MGA embedding validation requires a clean Git worktree")

    report = {
        "schema_version": 1,
        "status": "complete",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_status_short": git_status,
        "git_status_ignored_ephemeral": git_status_ignored,
        "model": "MGA-CLAP",
        "protocol_status": "controlled_public_code_reproduction",
        "embedding_dimension": audio_dim,
        "model_lock": identity(args.model_lock),
        "checkpoint": lock["checkpoint"],
        "audio": audio,
        "captions": captions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("MGA_EMBEDDING_VALIDATION_STATUS=complete")
    print(f"AUDIO_SHAPE={audio['shape']}")
    print(f"CAPTION_SHAPE={captions['shape']}")
    print(f"OUTPUT={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
