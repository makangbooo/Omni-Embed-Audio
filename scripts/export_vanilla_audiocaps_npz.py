#!/usr/bin/env python3
"""Export locked vanilla backbone artifacts to the canonical AudioCaps NPZ format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.precompute_mecat_audio_embeddings import identity, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-candidates", type=int, default=975)
    parser.add_argument("--captions-per-audio", type=int, default=5)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def export(args: argparse.Namespace) -> dict:
    source = args.generation_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    metrics_path = source / "generation_metrics.json"
    generation = json.loads(metrics_path.read_text(encoding="utf-8"))
    if generation.get("status") != "complete":
        raise ValueError("vanilla generation is not complete")
    candidates = read_jsonl(source / "candidate_metadata.jsonl")
    queries = read_jsonl(source / "query_metadata.jsonl")
    audio = np.load(source / "candidate_embeddings.npy", allow_pickle=False)
    captions = np.load(source / "query_embeddings.npy", allow_pickle=False)
    expected_captions = args.expected_candidates * args.captions_per_audio
    if len(candidates) != args.expected_candidates or len(queries) != expected_captions:
        raise ValueError("vanilla AudioCaps metadata count mismatch")
    if audio.ndim != 2 or captions.shape != (expected_captions, audio.shape[1]):
        raise ValueError("vanilla AudioCaps embedding shape mismatch")
    if not np.isfinite(audio).all() or not np.isfinite(captions).all():
        raise ValueError("vanilla AudioCaps embeddings contain non-finite values")

    audio_ids = [str(row["candidate_id"]) for row in candidates]
    caption_ids = [str(row["target_id"]) for row in queries]
    texts = [str(row["text"]) for row in queries]
    if audio_ids != sorted(audio_ids) or len(set(audio_ids)) != len(audio_ids):
        raise ValueError("vanilla AudioCaps candidate IDs are not canonical")
    expected_caption_ids = [value for value in audio_ids for _ in range(args.captions_per_audio)]
    if caption_ids != expected_caption_ids:
        raise ValueError("vanilla AudioCaps caption ownership/order mismatch")

    audio_path = output / "audio_embeddings.npz"
    caption_path = output / "caption_embeddings.npz"
    np.savez_compressed(
        audio_path,
        embeddings=np.asarray(audio, dtype=np.float32),
        clip_ids=np.asarray(audio_ids, dtype=object),
    )
    np.savez_compressed(
        caption_path,
        embeddings=np.asarray(captions, dtype=np.float32),
        clip_ids=np.asarray(caption_ids, dtype=object),
        texts=np.asarray(texts, dtype=object),
    )
    report = {
        "schema_version": 1,
        "status": "complete",
        "conversion": "locked vanilla NPY/JSONL to canonical AudioCaps NPZ",
        "candidate_count": len(audio_ids),
        "caption_count": len(caption_ids),
        "embedding_dimension": int(audio.shape[1]),
        "source_generation_metrics": identity(metrics_path),
        "audio_embeddings": identity(audio_path),
        "caption_embeddings": identity(caption_path),
    }
    write_json(output / "conversion_metrics.json", report)
    return report


def main() -> int:
    report = export(parse_args())
    print("VANILLA_AUDIOCAPS_EXPORT_STATUS=complete")
    print(f"AUDIO_SHAPE={[report['candidate_count'], report['embedding_dimension']]}")
    print(f"CAPTION_SHAPE={[report['caption_count'], report['embedding_dimension']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
