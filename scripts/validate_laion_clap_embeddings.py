#!/usr/bin/env python3
"""Validate LAION-CLAP NPZ embeddings and preserve generation evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

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


def load_array(path: Path, expected_rows: int) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        embeddings = np.asarray(archive["embeddings"])
        clip_ids = np.asarray(archive["clip_ids"])
        keys = sorted(archive.files)
    expected_shape = (expected_rows, 512)
    if embeddings.shape != expected_shape:
        raise RuntimeError(
            f"embedding shape mismatch for {path.name}: {embeddings.shape}"
        )
    if clip_ids.shape != (expected_rows,):
        raise RuntimeError(f"clip-id shape mismatch for {path.name}: {clip_ids.shape}")
    if not np.isfinite(embeddings).all():
        raise RuntimeError(f"non-finite embeddings in {path.name}")
    norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise RuntimeError(
            f"embeddings are not L2 normalized in {path.name}: "
            f"min={norms.min()} max={norms.max()}"
        )
    return {
        "identity": identity(path),
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "archive_keys": keys,
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "all_finite": True,
    }


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> int:
    args = parse_args()
    embedding_dir = args.embedding_dir.resolve()
    model_lock_path = args.model_lock.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite validation report: {output}")
    git_commit = git_output("rev-parse", "HEAD")
    git_status = git_output("status", "--short")
    if git_status:
        raise RuntimeError("embedding validation requires a clean Git worktree")
    model_lock = json.loads(model_lock_path.read_text(encoding="utf-8"))
    if model_lock.get("status") != "complete":
        raise RuntimeError("LAION model lock is not complete")
    audio = load_array(
        embedding_dir / "audio_embeddings.npz", args.expected_audio
    )
    captions = load_array(
        embedding_dir / "caption_embeddings.npz", args.expected_captions
    )
    report = {
        "schema_version": 1,
        "status": "complete",
        "started_at": utc_now(),
        "finished_at": utc_now(),
        "git_commit": git_commit,
        "git_status_short": git_status,
        "model": "LAION-CLAP",
        "protocol_status": "controlled_public_code_reproduction",
        "model_lock": identity(model_lock_path),
        "checkpoint": model_lock["checkpoint"],
        "package_overlay": model_lock["package_overlay"],
        "protocol": {
            "architecture": "HTSAT-tiny + RoBERTa",
            "checkpoint": "630k-audioset-best.pt",
            "fusion": False,
            "sample_rate": 48000,
            "audio_length_seconds": 10.0,
            "audio_crop": "center crop or zero pad",
            "normalization": "adapter L2",
            "source": "CODE",
        },
        "audio": audio,
        "captions": captions,
        "source_files": {
            relative: identity(REPOSITORY_ROOT / relative)
            for relative in (
                "AudioRetrieval/models/laion_clap_adapter.py",
                "AudioRetrieval/models/laion_clap_tokenizers.py",
                "AudioRetrieval/preprocessing/embeddings/laion_clap.py",
                "AudioRetrieval/evaluation/metrics.py",
            )
        },
    }
    write_new_json(output, report)
    print("LAION_EMBEDDING_VALIDATION_STATUS=complete")
    print(f"AUDIO_SHAPE={audio['shape']}")
    print(f"CAPTION_SHAPE={captions['shape']}")
    print(f"OUTPUT={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
