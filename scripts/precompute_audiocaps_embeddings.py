#!/usr/bin/env python3
"""Precompute AudioCaps test audio and caption embeddings from the audited manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.precompute_mecat_audio_embeddings import (
    build_precomputer,
    identity,
    sha256_file,
    write_json,
)


@dataclass(frozen=True)
class AudioCapsEntry:
    clip_id: str
    audio_path: Path
    captions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        choices=("laion_clap", "mga_clap", "m2d_clap", "oea"),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-examples", type=int, default=975)
    parser.add_argument("--expected-captions", type=int, default=4875)
    parser.add_argument("--expected-dim", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size-audio", type=int, default=16)
    parser.add_argument("--batch-size-text", type=int, default=128)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--repo-id")
    parser.add_argument("--local-path", type=Path)
    parser.add_argument("--laion-ckpt", type=Path)
    parser.add_argument("--laion-bert-tokenizer", type=Path)
    parser.add_argument("--laion-roberta-tokenizer", type=Path)
    parser.add_argument("--laion-bart-tokenizer", type=Path)
    parser.add_argument("--mga-repo", type=Path)
    parser.add_argument("--mga-ckpt", type=Path)
    parser.add_argument("--mga-bert-tokenizer", type=Path)
    parser.add_argument("--mga-checkpoint-sha256")
    parser.add_argument("--m2d-ckpt", type=Path)
    parser.add_argument("--m2d-bert-tokenizer", type=Path)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifest(
    path: Path,
    expected_sha256: str,
    expected_examples: int,
    expected_captions: int,
) -> tuple[list[AudioCapsEntry], list[dict[str, Any]]]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("AudioCaps audio manifest SHA256 mismatch")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != expected_examples:
        raise ValueError(
            f"AudioCaps manifest row mismatch: {len(rows)} != {expected_examples}"
        )
    sample_ids = [str(row.get("sample_id", "")).strip() for row in rows]
    if any(not value for value in sample_ids):
        raise ValueError("AudioCaps manifest contains an empty sample_id")
    if sample_ids != sorted(sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("AudioCaps sample IDs must be sorted and unique")

    entries: list[AudioCapsEntry] = []
    caption_count = 0
    for index, (sample_id, row) in enumerate(zip(sample_ids, rows, strict=True)):
        captions = row.get("captions")
        if not isinstance(captions, list) or len(captions) != 5:
            raise ValueError(f"AudioCaps {sample_id} must have five captions")
        if any(not isinstance(value, str) or not value.strip() for value in captions):
            raise ValueError(f"AudioCaps {sample_id} contains an empty caption")
        audio_path = Path(str(row.get("audio_path", ""))).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        if audio_path.stat().st_size != row.get("audio_size_bytes"):
            raise ValueError(f"AudioCaps audio size mismatch for {sample_id}")
        if sha256_file(audio_path) != row.get("audio_sha256"):
            raise ValueError(f"AudioCaps audio SHA256 mismatch for {sample_id}")
        if row.get("decode_ok") is not True or row.get("file_exists") is not True:
            raise ValueError(f"AudioCaps decode gate is incomplete for {sample_id}")
        entries.append(
            AudioCapsEntry(
                clip_id=sample_id,
                audio_path=audio_path,
                captions=[value.strip() for value in captions],
                metadata={"manifest_index": index},
            )
        )
        caption_count += len(captions)
    if caption_count != expected_captions:
        raise ValueError(
            f"AudioCaps caption count mismatch: {caption_count} != {expected_captions}"
        )
    return entries, rows


def strings(values: np.ndarray) -> list[str]:
    return [str(value) for value in values.tolist()]


def validate_outputs(
    output_dir: Path,
    entries: list[AudioCapsEntry],
    expected_dim: int,
) -> tuple[Path, Path]:
    audio_path = output_dir / "audio_embeddings.npz"
    caption_path = output_dir / "caption_embeddings.npz"
    with np.load(audio_path, allow_pickle=True) as archive:
        audio_embeddings = np.asarray(archive["embeddings"])
        audio_ids = strings(archive["clip_ids"])
    with np.load(caption_path, allow_pickle=True) as archive:
        caption_embeddings = np.asarray(archive["embeddings"])
        caption_ids = strings(archive["clip_ids"])
        caption_texts = strings(archive["texts"])

    expected_ids = [entry.clip_id for entry in entries]
    expected_caption_ids = [
        entry.clip_id for entry in entries for _ in entry.captions
    ]
    expected_texts = [caption for entry in entries for caption in entry.captions]
    if audio_embeddings.shape != (len(entries), expected_dim):
        raise ValueError(f"unexpected AudioCaps audio shape: {audio_embeddings.shape}")
    if caption_embeddings.shape != (len(expected_texts), expected_dim):
        raise ValueError(
            f"unexpected AudioCaps caption shape: {caption_embeddings.shape}"
        )
    if audio_ids != expected_ids:
        raise ValueError("AudioCaps audio IDs differ from manifest order")
    if caption_ids != expected_caption_ids or caption_texts != expected_texts:
        raise ValueError("AudioCaps caption artifacts differ from manifest order")
    if not np.isfinite(audio_embeddings).all() or not np.isfinite(caption_embeddings).all():
        raise ValueError("AudioCaps embeddings contain non-finite values")
    if Counter(caption_ids) != Counter({value: 5 for value in expected_ids}):
        raise ValueError("AudioCaps caption ownership mismatch")
    return audio_path, caption_path


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    report_path = output_dir / "generation_metrics.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "model": args.model,
        "dataset": "AudioCaps v2 test",
    }
    try:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        entries, _ = load_manifest(
            args.manifest.resolve(),
            args.manifest_sha256,
            args.expected_examples,
            args.expected_captions,
        )
        print(
            f"AUDIOCAPS_MANIFEST_STATUS=complete rows={len(entries)} "
            f"captions={sum(len(entry.captions) for entry in entries)}"
        )
        precomputer = build_precomputer(args)
        precomputer.batch_size_text = args.batch_size_text
        precomputer.precompute_dataset(
            entries,
            output_dir,
            compute_audio=True,
            compute_captions=True,
        )
        audio_path, caption_path = validate_outputs(
            output_dir, entries, args.expected_dim
        )
        report.update(
            {
                "status": "complete",
                "finished_at": utc_now(),
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
                ).strip(),
                "git_status_short": subprocess.check_output(
                    ["git", "status", "--short"], cwd=REPOSITORY_ROOT, text=True
                ).strip(),
                "candidate_count": len(entries),
                "caption_count": sum(len(entry.captions) for entry in entries),
                "embedding_dimension": args.expected_dim,
                "manifest": identity(args.manifest.resolve()),
                "audio_embeddings": identity(audio_path),
                "caption_embeddings": identity(caption_path),
            }
        )
        write_json(report_path, report)
        print("AUDIOCAPS_EMBEDDING_STATUS=complete")
        print(f"AUDIO_SHAPE={[len(entries), args.expected_dim]}")
        print(f"CAPTION_SHAPE={[args.expected_captions, args.expected_dim]}")
        print(f"OUTPUT={output_dir}")
        return 0
    except BaseException as exc:
        report.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(report_path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
