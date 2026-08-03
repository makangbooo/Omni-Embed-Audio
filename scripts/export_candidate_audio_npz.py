#!/usr/bin/env python3
"""Export locked candidate NPY/JSONL artifacts to canonical audio NPZ."""

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
    parser.add_argument("--expected-candidates", type=int, required=True)
    parser.add_argument("--dataset-name", required=True)
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
        raise ValueError("candidate generation is not complete")

    metadata_path = source / "candidate_metadata.jsonl"
    embeddings_path = source / "candidate_embeddings.npy"
    candidates = read_jsonl(metadata_path)
    embeddings = np.load(embeddings_path, allow_pickle=False)
    expected = args.expected_candidates
    if len(candidates) != expected or embeddings.ndim != 2:
        raise ValueError("candidate metadata or embedding count mismatch")
    if embeddings.shape[0] != expected or not np.isfinite(embeddings).all():
        raise ValueError("candidate embeddings have invalid shape or values")

    indices = [int(row["candidate_index"]) for row in candidates]
    candidate_ids = [str(row["candidate_id"]).strip() for row in candidates]
    filenames = [Path(str(row.get("audio_path", value))).name for row, value in zip(candidates, candidate_ids)]
    if indices != list(range(expected)):
        raise ValueError("candidate indices are not contiguous")
    if any(not value for value in candidate_ids) or len(set(candidate_ids)) != expected:
        raise ValueError("candidate IDs are empty or duplicated")

    audio_path = output / "audio_embeddings.npz"
    np.savez_compressed(
        audio_path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        clip_ids=np.asarray(candidate_ids, dtype=object),
        filenames=np.asarray(filenames, dtype=object),
    )
    report = {
        "schema_version": 1,
        "status": "complete",
        "conversion": "locked candidate NPY/JSONL to canonical audio NPZ",
        "dataset": args.dataset_name,
        "candidate_count": expected,
        "embedding_dimension": int(embeddings.shape[1]),
        "source_generation_metrics": identity(metrics_path),
        "source_candidate_embeddings": identity(embeddings_path),
        "source_candidate_metadata": identity(metadata_path),
        "audio_embeddings": identity(audio_path),
    }
    write_json(output / "conversion_metrics.json", report)
    return report


def main() -> int:
    report = export(parse_args())
    print("CANDIDATE_AUDIO_EXPORT_STATUS=complete")
    print(f"AUDIO_SHAPE={[report['candidate_count'], report['embedding_dimension']]}")
    print(f"AUDIO_NPZ={report['audio_embeddings']['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
