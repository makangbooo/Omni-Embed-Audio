#!/usr/bin/env python3
"""Validate Robust-CLAP NPZ embeddings and their controlled resource binding."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.precompute_mecat_audio_embeddings import identity, write_json


EXPECTED_SOURCE_REVISION = "d08d0e3c545fa22df0930fc0d090741aaa9e2cc1"
EXPECTED_CHECKPOINT_SHA256 = (
    "8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-audio", type=int, required=True)
    parser.add_argument("--expected-captions", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate(args: argparse.Namespace) -> dict:
    source = args.source_dir.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != EXPECTED_SOURCE_REVISION:
        raise ValueError(f"Robust-CLAP source revision mismatch: {revision}")
    checkpoint = identity(args.checkpoint.resolve())
    if checkpoint["sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Robust-CLAP checkpoint SHA256 mismatch")

    audio_path = args.embedding_dir / "audio_embeddings.npz"
    caption_path = args.embedding_dir / "caption_embeddings.npz"
    with np.load(audio_path, allow_pickle=True) as archive:
        audio = np.asarray(archive["embeddings"])
        audio_ids = [str(value) for value in archive["clip_ids"].tolist()]
    with np.load(caption_path, allow_pickle=True) as archive:
        captions = np.asarray(archive["embeddings"])
        caption_ids = [str(value) for value in archive["clip_ids"].tolist()]
    if audio.shape != (args.expected_audio, 512):
        raise ValueError(f"unexpected Robust-CLAP audio shape: {audio.shape}")
    if captions.shape != (args.expected_captions, 512):
        raise ValueError(f"unexpected Robust-CLAP caption shape: {captions.shape}")
    if len(set(audio_ids)) != args.expected_audio:
        raise ValueError("Robust-CLAP audio IDs are not unique")
    if len(caption_ids) != args.expected_captions:
        raise ValueError("Robust-CLAP caption IDs are incomplete")
    if not np.isfinite(audio).all() or not np.isfinite(captions).all():
        raise ValueError("Robust-CLAP embeddings contain non-finite values")
    report = {
        "schema_version": 1,
        "status": "complete",
        "model": "Robust-CLAP controlled standard-checkpoint binding",
        "strict_paper_checkpoint_reproduction": False,
        "source_revision": revision,
        "checkpoint": checkpoint,
        "enable_fusion": False,
        "audio_shape": list(audio.shape),
        "caption_shape": list(captions.shape),
        "audio_embeddings": identity(audio_path),
        "caption_embeddings": identity(caption_path),
    }
    write_json(args.output, report)
    return report


def main() -> int:
    report = validate(parse_args())
    print("ROBUST_CLAP_EMBEDDING_VALIDATION_STATUS=complete")
    print(f"AUDIO_SHAPE={report['audio_shape']}")
    print(f"CAPTION_SHAPE={report['caption_shape']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
