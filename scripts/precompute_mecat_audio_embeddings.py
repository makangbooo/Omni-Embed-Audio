#!/usr/bin/env python3
"""Precompute MECAT public-848 audio embeddings from the audited manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


@dataclass(frozen=True)
class MecatAudioEntry:
    """Minimal entry contract consumed by the official precomputers."""

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
    parser.add_argument("--expected-examples", type=int, default=848)
    parser.add_argument("--expected-dim", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size-audio", type=int, default=16)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_manifest(
    path: Path,
    expected_sha256: str,
    expected_examples: int,
) -> tuple[list[MecatAudioEntry], list[dict[str, Any]]]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("MECAT manifest SHA256 mismatch")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != expected_examples:
        raise ValueError(
            f"MECAT manifest row mismatch: {len(rows)} != {expected_examples}"
        )
    sample_ids = [str(row.get("sample_id", "")).strip() for row in rows]
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError("MECAT manifest contains an empty sample_id")
    if len(set(sample_ids)) != expected_examples:
        raise ValueError("MECAT manifest sample_id values are not unique")
    if sample_ids != sorted(sample_ids):
        raise ValueError("MECAT manifest must preserve canonical sample_id order")

    entries: list[MecatAudioEntry] = []
    for index, (sample_id, row) in enumerate(zip(sample_ids, rows, strict=True)):
        audio_path = Path(str(row.get("audio_path", ""))).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        expected_size = row.get("audio_size_bytes")
        expected_audio_sha256 = row.get("audio_sha256")
        if audio_path.stat().st_size != expected_size:
            raise ValueError(f"MECAT audio size mismatch for {sample_id}")
        if sha256_file(audio_path) != expected_audio_sha256:
            raise ValueError(f"MECAT audio SHA256 mismatch for {sample_id}")
        if row.get("decode_ok") is not True or row.get("file_exists") is not True:
            raise ValueError(f"MECAT manifest gate is incomplete for {sample_id}")
        entries.append(
            MecatAudioEntry(
                clip_id=sample_id,
                audio_path=audio_path,
                metadata={"manifest_index": index},
            )
        )
    return entries, rows


def build_precomputer(args: argparse.Namespace):
    if args.model == "laion_clap":
        from AudioRetrieval.preprocessing.embeddings import (
            LaionClapEmbeddingPrecomputer,
        )

        required = (
            args.laion_ckpt,
            args.laion_bert_tokenizer,
            args.laion_roberta_tokenizer,
            args.laion_bart_tokenizer,
        )
        if not all(required):
            raise ValueError("LAION-CLAP requires checkpoint and three tokenizers")
        return LaionClapEmbeddingPrecomputer(
            ckpt_path=str(args.laion_ckpt),
            device=args.device,
            batch_size_audio=args.batch_size_audio,
            bert_tokenizer_path=str(args.laion_bert_tokenizer),
            roberta_tokenizer_path=str(args.laion_roberta_tokenizer),
            bart_tokenizer_path=str(args.laion_bart_tokenizer),
        )
    if args.model == "mga_clap":
        from AudioRetrieval.preprocessing.embeddings import MGAClapEmbeddingPrecomputer

        required = (
            args.mga_repo,
            args.mga_ckpt,
            args.mga_bert_tokenizer,
            args.mga_checkpoint_sha256,
        )
        if not all(required):
            raise ValueError("MGA-CLAP requires source, checkpoint, tokenizer, SHA256")
        return MGAClapEmbeddingPrecomputer(
            repo_path=str(args.mga_repo),
            ckpt_path=str(args.mga_ckpt),
            bert_tokenizer_path=str(args.mga_bert_tokenizer),
            checkpoint_sha256=args.mga_checkpoint_sha256,
            device=args.device,
            batch_size_audio=args.batch_size_audio,
        )
    if args.model == "m2d_clap":
        from AudioRetrieval.preprocessing.embeddings import M2DClapEmbeddingPrecomputer

        if not args.m2d_ckpt or not args.m2d_bert_tokenizer:
            raise ValueError("M2D-CLAP requires checkpoint and tokenizer")
        return M2DClapEmbeddingPrecomputer(
            checkpoint=str(args.m2d_ckpt),
            bert_tokenizer_path=str(args.m2d_bert_tokenizer),
            device=args.device,
            batch_size_audio=args.batch_size_audio,
        )
    from AudioRetrieval.preprocessing.embeddings import OEAEmbeddingPrecomputer

    if not args.checkpoint or not args.repo_id or not args.local_path:
        raise ValueError("OEA requires checkpoint, repo ID, and local base model")
    return OEAEmbeddingPrecomputer(
        checkpoint=str(args.checkpoint),
        repo_id=args.repo_id,
        local_path=str(args.local_path),
        device=args.device,
        batch_size_audio=args.batch_size_audio,
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    metrics_path = output_dir / "generation_metrics.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "model": args.model,
        "dataset": "MECAT-Caption 00A/test public 848-row release",
        "error": None,
    }
    try:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        entries, manifest_rows = load_manifest(
            args.manifest.resolve(),
            args.manifest_sha256,
            args.expected_examples,
        )
        print(f"MECAT_MANIFEST_STATUS=complete rows={len(entries)}")
        precomputer = build_precomputer(args)
        precomputer.precompute_dataset(
            entries,
            output_dir,
            compute_audio=True,
            compute_captions=False,
        )
        output_path = output_dir / "audio_embeddings.npz"
        with np.load(output_path, allow_pickle=True) as archive:
            embeddings = np.asarray(archive["embeddings"])
            clip_ids = [str(value) for value in archive["clip_ids"].tolist()]
        expected_ids = [str(row["sample_id"]) for row in manifest_rows]
        expected_shape = (args.expected_examples, args.expected_dim)
        if embeddings.shape != expected_shape:
            raise ValueError(
                f"MECAT audio embedding shape mismatch: "
                f"{embeddings.shape} != {expected_shape}"
            )
        if clip_ids != expected_ids:
            raise ValueError("MECAT audio embedding IDs differ from manifest order")
        if not np.isfinite(embeddings).all():
            raise ValueError("MECAT audio embeddings contain non-finite values")
        report.update(
            {
                "status": "complete",
                "finished_at": utc_now(),
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=REPOSITORY_ROOT,
                    text=True,
                ).strip(),
                "git_status_short": subprocess.check_output(
                    ["git", "status", "--short"],
                    cwd=REPOSITORY_ROOT,
                    text=True,
                ).strip(),
                "candidate_count": args.expected_examples,
                "paper_reported_candidate_count": 847,
                "embedding_shape": list(embeddings.shape),
                "manifest": identity(args.manifest.resolve()),
                "audio_embeddings": identity(output_path),
                "error": None,
            }
        )
        write_json(metrics_path, report)
        print("MECAT_AUDIO_EMBEDDING_STATUS=complete")
        print(f"AUDIO_SHAPE={list(embeddings.shape)}")
        print(f"OUTPUT={output_path}")
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
        write_json(metrics_path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
