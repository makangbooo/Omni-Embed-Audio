#!/usr/bin/env python3
"""Encode three released negative UIQ sets with one OEA model load."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.precompute_mecat_audio_embeddings import identity, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--local-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size-text", type=int, default=16)
    parser.add_argument(
        "--dataset-spec",
        action="append",
        nargs=4,
        metavar=("NAME", "JSONL", "OUTPUT_DIR", "EXPECTED_QUERIES"),
        required=True,
    )
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def generate(
    args: argparse.Namespace,
    encoder_factory: Callable[..., Any] | None = None,
) -> dict:
    if encoder_factory is None:
        from AudioRetrieval.preprocessing.embeddings import (
            UIQTextEmbeddingPrecomputer,
        )

        encoder_factory = UIQTextEmbeddingPrecomputer
    names = [spec[0] for spec in args.dataset_spec]
    if len(names) != len(set(names)):
        raise ValueError("dataset names must be unique")
    if args.batch_size_text < 1:
        raise ValueError("batch size must be positive")

    resolved_specs = []
    for name, jsonl_value, output_value, expected_value in args.dataset_spec:
        jsonl_path = Path(jsonl_value).resolve()
        output_dir = Path(output_value).resolve()
        expected = int(expected_value)
        if not jsonl_path.is_file():
            raise FileNotFoundError(jsonl_path)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        resolved_specs.append((name, jsonl_path, output_dir, expected))

    encoder = encoder_factory(
        model="oea",
        device=args.device,
        batch_size_text=args.batch_size_text,
        oea_checkpoint=str(args.checkpoint.resolve()),
        oea_repo_id=args.repo_id,
        oea_local_path=str(args.local_path.resolve()),
    )

    datasets = {}
    for name, jsonl_path, output_dir, expected in resolved_specs:
        result = encoder.precompute(
            uiq_jsonl=jsonl_path,
            output_dir=output_dir,
            dataset=name,
        )
        if set(result) != {"negative"}:
            raise ValueError(f"{name} did not produce exactly one negative bucket")
        npz_path = output_dir / "uiq_negative_embeddings.npz"
        with np.load(npz_path, allow_pickle=True) as payload:
            shape = list(np.asarray(payload["embeddings"]).shape)
            clip_count = len(payload["clip_ids"])
        if shape != [expected, 512] or clip_count != expected:
            raise ValueError(f"{name} query embedding shape/count mismatch")
        datasets[name] = {
            "query_count": expected,
            "embedding_shape": shape,
            "source_jsonl": identity(jsonl_path),
            "query_npz": identity(npz_path),
        }

    report = {
        "schema_version": 1,
        "status": "complete",
        "model": "OEA",
        "model_load_count": 1,
        "checkpoint": identity(args.checkpoint.resolve()),
        "repo_id": args.repo_id,
        "local_path": str(args.local_path.resolve()),
        "device": args.device,
        "batch_size_text": args.batch_size_text,
        "total_query_count": sum(row["query_count"] for row in datasets.values()),
        "datasets": datasets,
    }
    write_json(args.summary.resolve(), report)
    print("OEA_NEGATIVE_EMBEDDING_STATUS=complete")
    print(f"MODEL_LOAD_COUNT={report['model_load_count']}")
    print(f"TOTAL_QUERY_COUNT={report['total_query_count']}")
    return report


def main() -> int:
    generate(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
