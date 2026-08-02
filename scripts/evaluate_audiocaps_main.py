#!/usr/bin/env python3
"""Evaluate AudioCaps Tables 2/3 and positive UIQ Tables 12-15."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from AudioRetrieval.evaluation.canonical import (  # noqa: E402
    evaluate_caption_to_caption,
    evaluate_id_retrieval,
)
from scripts.evaluate_official_source_oea_clotho import (  # noqa: E402
    QUERY_TYPES,
    identity,
    load_npz,
    now,
    save_result,
    strings,
    write_json,
)
from scripts.precompute_mecat_audio_embeddings import sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--uiq-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-candidates", type=int, default=975)
    parser.add_argument("--captions-per-audio", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-uiq",
        action="store_true",
        help="Evaluate Tables 2/3 only (used by vanilla backbones).",
    )
    return parser.parse_args()


def select_one_per_clip(
    clip_ids: list[str], ordered_ids: list[str], seed: int
) -> list[int]:
    positions: dict[str, list[int]] = {clip_id: [] for clip_id in ordered_ids}
    for index, clip_id in enumerate(clip_ids):
        if clip_id not in positions:
            raise ValueError(f"caption ID absent from audio candidates: {clip_id}")
        positions[clip_id].append(index)
    rng = random.Random(seed)
    selected = [rng.choice(positions[clip_id]) for clip_id in ordered_ids]
    selected.sort()
    return selected


def read_manifest(
    path: Path,
    expected_sha256: str,
    expected_candidates: int,
    captions_per_audio: int,
) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("AudioCaps audio manifest SHA256 mismatch")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != expected_candidates:
        raise ValueError("AudioCaps manifest candidate count mismatch")
    ids = [str(row.get("sample_id", "")) for row in rows]
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        raise ValueError("AudioCaps manifest IDs must be sorted and unique")
    if any(len(row.get("captions", [])) != captions_per_audio for row in rows):
        raise ValueError("AudioCaps manifest caption count mismatch")
    return rows


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = now()

    rows = read_manifest(
        args.manifest.resolve(),
        args.manifest_sha256,
        args.expected_candidates,
        args.captions_per_audio,
    )
    expected_ids = [str(row["sample_id"]) for row in rows]
    expected_caption_count = args.expected_candidates * args.captions_per_audio
    audio_path = args.embedding_dir / "audio_embeddings.npz"
    caption_path = args.embedding_dir / "caption_embeddings.npz"
    audio = load_npz(audio_path)
    captions = load_npz(caption_path)
    audio_embeddings = np.asarray(audio["embeddings"])
    caption_embeddings = np.asarray(captions["embeddings"])
    audio_ids = strings(audio["clip_ids"])
    caption_ids = strings(captions["clip_ids"])
    expected_dim = audio_embeddings.shape[1] if audio_embeddings.ndim == 2 else -1

    if audio_embeddings.shape != (args.expected_candidates, expected_dim):
        raise ValueError(f"unexpected AudioCaps audio shape: {audio_embeddings.shape}")
    if caption_embeddings.shape != (expected_caption_count, expected_dim):
        raise ValueError(
            f"unexpected AudioCaps caption shape: {caption_embeddings.shape}"
        )
    if audio_ids != expected_ids:
        raise ValueError("AudioCaps audio IDs differ from manifest order")
    if Counter(caption_ids) != Counter(
        {clip_id: args.captions_per_audio for clip_id in expected_ids}
    ):
        raise ValueError("AudioCaps caption ownership mismatch")
    if not np.isfinite(audio_embeddings).all() or not np.isfinite(caption_embeddings).all():
        raise ValueError("AudioCaps main embeddings contain non-finite values")

    seed0_indices = select_one_per_clip(caption_ids, audio_ids, args.seed)
    protocols: list[dict[str, Any]] = []
    protocols.append(
        save_result(
            output_dir,
            {
                "protocol_id": "t2a_public_code_default_joint_all_captions",
                "task": "t2a",
                "paper_table": "Table 2",
                "protocol_source": "CODE",
                "query_selection": "all",
            },
            evaluate_id_retrieval(
                caption_embeddings, caption_ids, audio_embeddings, audio_ids
            ),
        )
    )
    protocols.append(
        save_result(
            output_dir,
            {
                "protocol_id": "t2a_public_code_t2a_only_seed0",
                "task": "t2a",
                "paper_table": "Table 2 sensitivity",
                "protocol_source": "CODE",
                "query_selection": "public_code_seed0_one_per_clip",
                "query_selection_seed": args.seed,
            },
            evaluate_id_retrieval(
                caption_embeddings,
                caption_ids,
                audio_embeddings,
                audio_ids,
                query_indices=seed0_indices,
            ),
        )
    )
    protocols.append(
        save_result(
            output_dir,
            {
                "protocol_id": "t2t_public_code_default_seed0",
                "task": "t2t",
                "paper_table": "Table 3",
                "protocol_source": "CODE",
                "query_selection": "public_code_seed0_one_per_clip",
                "query_selection_seed": args.seed,
            },
            evaluate_caption_to_caption(
                caption_embeddings, caption_ids, query_indices=seed0_indices
            ),
        )
    )
    protocols.append(
        save_result(
            output_dir,
            {
                "protocol_id": "t2t_all_captions_sensitivity",
                "task": "t2t",
                "paper_table": "Table 3 sensitivity",
                "protocol_source": "INFERRED",
                "query_selection": "all",
            },
            evaluate_caption_to_caption(caption_embeddings, caption_ids),
        )
    )

    uiq_inputs: dict[str, dict[str, Any]] = {}
    query_types = () if getattr(args, "skip_uiq", False) else QUERY_TYPES
    for query_type in query_types:
        path = args.uiq_dir / f"uiq_{query_type}_embeddings.npz"
        data = load_npz(path)
        embeddings = np.asarray(data["embeddings"])
        target_ids = strings(data["clip_ids"])
        if embeddings.shape != (args.expected_candidates, expected_dim):
            raise ValueError(f"unexpected AudioCaps {query_type} shape: {embeddings.shape}")
        if len(set(target_ids)) != args.expected_candidates:
            raise ValueError(f"AudioCaps {query_type} target IDs are not unique")
        if set(target_ids) != set(audio_ids):
            raise ValueError(f"AudioCaps {query_type} IDs differ from candidates")
        if not np.isfinite(embeddings).all():
            raise ValueError(f"AudioCaps {query_type} embeddings are non-finite")
        protocols.append(
            save_result(
                output_dir,
                {
                    "protocol_id": f"{query_type}_released_uiq",
                    "task": "uiq",
                    "paper_table": {
                        "question": "Table 12",
                        "imperative": "Table 13",
                        "paraphrase": "Table 14",
                        "tagging": "Table 15",
                    }[query_type],
                    "protocol_source": "CODE",
                    "query_selection": "released_uiq_all",
                },
                evaluate_id_retrieval(
                    embeddings, target_ids, audio_embeddings, audio_ids
                ),
            )
        )
        uiq_inputs[query_type] = identity(path)

    summary = {
        "schema_version": 1,
        "status": "complete",
        "model": args.model,
        "dataset": "AudioCaps v2 test",
        "started_at": started_at,
        "finished_at": now(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
        ).strip(),
        "git_status_short": subprocess.check_output(
            ["git", "status", "--short"], cwd=REPOSITORY_ROOT, text=True
        ).strip(),
        "seed": args.seed,
        "candidate_count": args.expected_candidates,
        "caption_count": expected_caption_count,
        "source_usage": {
            "oea_official_source_used": True,
            "official_metric_file": "AudioRetrieval/evaluation/metrics.py",
            "compatibility_layer": "AudioRetrieval/evaluation/canonical.py",
        },
        "inputs": {
            "manifest": identity(args.manifest.resolve()),
            "audio_embeddings": identity(audio_path),
            "caption_embeddings": identity(caption_path),
            "uiq_embeddings": uiq_inputs,
        },
        "protocols": protocols,
    }
    write_json(output_dir / "suite_metrics.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    try:
        report = evaluate(args)
    except BaseException as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            args.output_dir / "suite_metrics.json",
            {
                "schema_version": 1,
                "status": "failed",
                "finished_at": now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    for protocol in report["protocols"]:
        metrics = protocol["metrics"]
        print(
            protocol["protocol_id"],
            f"R@1={metrics['R@1']:.6f}",
            f"R@5={metrics['R@5']:.6f}",
            f"R@10={metrics['R@10']:.6f}",
        )
    print("AUDIOCAPS_MAIN_EVALUATION_STATUS=complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
