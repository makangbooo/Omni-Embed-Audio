#!/usr/bin/env python3
"""Precompute MECAT public-848 audio and/or caption embeddings."""

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
        choices=("laion_clap", "robust_clap", "mga_clap", "m2d_clap", "oea"),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-examples", type=int, default=848)
    parser.add_argument("--expected-dim", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size-audio", type=int, default=16)
    parser.add_argument("--batch-size-text", type=int, default=128)
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--compute-captions", action="store_true")
    parser.add_argument(
        "--caption-field",
        choices=("long", "short", "speech", "music", "sound", "environment"),
    )
    parser.add_argument("--captions-per-example", type=int)
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
    parser.add_argument("--robust-ckpt", type=Path)
    parser.add_argument("--robust-repo", type=Path)
    parser.add_argument("--robust-bert-tokenizer", type=Path)
    parser.add_argument("--robust-roberta-tokenizer", type=Path)
    parser.add_argument("--robust-bart-tokenizer", type=Path)
    parser.add_argument("--robust-bpe-vocab", type=Path)
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
    *,
    caption_field: str | None = None,
    captions_per_example: int | None = None,
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
        captions: list[str] = []
        if caption_field is not None:
            caption_fields = row.get("caption_fields")
            if not isinstance(caption_fields, dict):
                raise TypeError(f"caption_fields is not an object for {sample_id}")
            raw_captions = caption_fields.get(caption_field)
            if not isinstance(raw_captions, list):
                raise TypeError(
                    f"{caption_field} captions are not a list for {sample_id}"
                )
            captions = [
                value.strip()
                for value in raw_captions
                if isinstance(value, str)
                and value.strip()
                and value.strip().casefold() != "none"
            ]
            if len(captions) != captions_per_example:
                raise ValueError(
                    f"{caption_field} caption count mismatch for {sample_id}: "
                    f"{len(captions)} != {captions_per_example}"
                )
        entries.append(
            MecatAudioEntry(
                clip_id=sample_id,
                audio_path=audio_path,
                captions=captions,
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
    if args.model == "robust_clap":
        from AudioRetrieval.preprocessing.embeddings import (
            RobustClapEmbeddingPrecomputer,
        )

        if not all(
            (
                args.robust_ckpt,
                args.robust_repo,
                args.robust_bert_tokenizer,
                args.robust_roberta_tokenizer,
                args.robust_bart_tokenizer,
                args.robust_bpe_vocab,
            )
        ):
            raise ValueError(
                "Robust-CLAP requires source, checkpoint, and local BERT, "
                "RoBERTa, and BART tokenizers plus BPE vocabulary"
            )
        return RobustClapEmbeddingPrecomputer(
            ckpt_path=str(args.robust_ckpt),
            repo_root=str(args.robust_repo),
            bert_tokenizer_path=str(args.robust_bert_tokenizer),
            roberta_tokenizer_path=str(args.robust_roberta_tokenizer),
            bart_tokenizer_path=str(args.robust_bart_tokenizer),
            bpe_vocab_path=str(args.robust_bpe_vocab),
            enable_fusion=False,
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
    if args.skip_audio and not args.compute_captions:
        raise ValueError("at least one of audio or caption computation is required")
    if args.compute_captions and (
        args.caption_field is None or args.captions_per_example is None
    ):
        raise ValueError(
            "caption computation requires --caption-field and --captions-per-example"
        )
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
            caption_field=args.caption_field,
            captions_per_example=args.captions_per_example,
        )
        print(f"MECAT_MANIFEST_STATUS=complete rows={len(entries)}")
        precomputer = build_precomputer(args)
        precomputer.batch_size_text = args.batch_size_text
        precomputer.precompute_dataset(
            entries,
            output_dir,
            compute_audio=not args.skip_audio,
            compute_captions=args.compute_captions,
        )
        expected_ids = [str(row["sample_id"]) for row in manifest_rows]
        generated: dict[str, Any] = {}
        if not args.skip_audio:
            audio_path = output_dir / "audio_embeddings.npz"
            with np.load(audio_path, allow_pickle=True) as archive:
                audio_embeddings = np.asarray(archive["embeddings"])
                audio_ids = [str(value) for value in archive["clip_ids"].tolist()]
            expected_audio_shape = (args.expected_examples, args.expected_dim)
            if audio_embeddings.shape != expected_audio_shape:
                raise ValueError(
                    f"MECAT audio embedding shape mismatch: "
                    f"{audio_embeddings.shape} != {expected_audio_shape}"
                )
            if audio_ids != expected_ids:
                raise ValueError("MECAT audio embedding IDs differ from manifest order")
            if not np.isfinite(audio_embeddings).all():
                raise ValueError("MECAT audio embeddings contain non-finite values")
            generated["audio_embedding_shape"] = list(audio_embeddings.shape)
            generated["audio_embeddings"] = identity(audio_path)
        if args.compute_captions:
            caption_path = output_dir / "caption_embeddings.npz"
            with np.load(caption_path, allow_pickle=True) as archive:
                caption_embeddings = np.asarray(archive["embeddings"])
                caption_ids = [str(value) for value in archive["clip_ids"].tolist()]
            expected_caption_ids = [
                sample_id
                for sample_id in expected_ids
                for _ in range(args.captions_per_example)
            ]
            expected_caption_shape = (
                args.expected_examples * args.captions_per_example,
                args.expected_dim,
            )
            if caption_embeddings.shape != expected_caption_shape:
                raise ValueError(
                    f"MECAT caption embedding shape mismatch: "
                    f"{caption_embeddings.shape} != {expected_caption_shape}"
                )
            if caption_ids != expected_caption_ids:
                raise ValueError("MECAT caption ownership/order differs from manifest")
            if not np.isfinite(caption_embeddings).all():
                raise ValueError("MECAT caption embeddings contain non-finite values")
            generated["caption_embedding_shape"] = list(caption_embeddings.shape)
            generated["caption_embeddings"] = identity(caption_path)
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
                "caption_protocol": (
                    None
                    if not args.compute_captions
                    else {
                        "field": args.caption_field,
                        "captions_per_example": args.captions_per_example,
                        "source": (
                            "CODE: AudioRetrieval/scripts/"
                            "mine_hard_negatives_laion.py:load_mecat_captions"
                        ),
                    }
                ),
                **generated,
                "manifest": identity(args.manifest.resolve()),
                "error": None,
            }
        )
        write_json(metrics_path, report)
        if not args.skip_audio:
            print("MECAT_AUDIO_EMBEDDING_STATUS=complete")
            print(f"AUDIO_SHAPE={generated['audio_embedding_shape']}")
        if args.compute_captions:
            print("MECAT_CAPTION_EMBEDDING_STATUS=complete")
            print(f"CAPTION_SHAPE={generated['caption_embedding_shape']}")
            print(f"CAPTION_FIELD={args.caption_field}")
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
        write_json(metrics_path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
