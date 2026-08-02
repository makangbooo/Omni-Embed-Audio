#!/usr/bin/env python3
"""Evaluate MECAT public-848 Tables 2/3 with the fixed short-all protocol."""

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

from AudioRetrieval.evaluation.canonical import (
    evaluate_caption_to_caption,
    evaluate_id_retrieval,
)
from scripts.evaluate_official_source_oea_clotho import (
    identity,
    load_npz,
    now,
    save_result,
    strings,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-embedding-dir", type=Path, required=True)
    parser.add_argument("--caption-embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-candidates", type=int, default=848)
    parser.add_argument("--paper-candidates", type=int, default=847)
    parser.add_argument("--captions-per-audio", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def select_one_per_clip(
    caption_ids: list[str], audio_ids: list[str], seed: int
) -> list[int]:
    positions: dict[str, list[int]] = {clip_id: [] for clip_id in audio_ids}
    for index, clip_id in enumerate(caption_ids):
        if clip_id not in positions:
            raise ValueError(f"caption ID absent from candidates: {clip_id}")
        positions[clip_id].append(index)
    rng = random.Random(seed)
    selected = [rng.choice(positions[clip_id]) for clip_id in audio_ids]
    selected.sort()
    return selected


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = now()

    audio_path = args.audio_embedding_dir / "audio_embeddings.npz"
    caption_path = args.caption_embedding_dir / "caption_embeddings.npz"
    audio = load_npz(audio_path)
    captions = load_npz(caption_path)
    audio_embeddings = np.asarray(audio["embeddings"])
    caption_embeddings = np.asarray(captions["embeddings"])
    audio_ids = strings(audio["clip_ids"])
    caption_ids = strings(captions["clip_ids"])
    expected_caption_count = args.expected_candidates * args.captions_per_audio
    if audio_embeddings.ndim != 2 or audio_embeddings.shape[0] != args.expected_candidates:
        raise ValueError(f"unexpected MECAT audio shape: {audio_embeddings.shape}")
    if caption_embeddings.shape != (expected_caption_count, audio_embeddings.shape[1]):
        raise ValueError(f"unexpected MECAT caption shape: {caption_embeddings.shape}")
    if audio_ids != sorted(audio_ids) or len(set(audio_ids)) != len(audio_ids):
        raise ValueError("MECAT audio IDs are not canonical and unique")
    counts = Counter(caption_ids)
    if set(counts) != set(audio_ids) or any(
        counts[clip_id] != args.captions_per_audio for clip_id in audio_ids
    ):
        raise ValueError("MECAT caption ownership is not exactly three per candidate")
    expected_caption_ids = [
        clip_id for clip_id in audio_ids for _ in range(args.captions_per_audio)
    ]
    if caption_ids != expected_caption_ids:
        raise ValueError("MECAT caption IDs are not in canonical grouped order")
    if not np.isfinite(audio_embeddings).all() or not np.isfinite(caption_embeddings).all():
        raise ValueError("MECAT embeddings contain non-finite values")

    seed0_indices = select_one_per_clip(caption_ids, audio_ids, args.seed)
    protocols: list[dict[str, Any]] = []
    definitions = (
        (
            {
                "protocol_id": "t2a_public848_short_all",
                "task": "t2a",
                "paper_table": "Table 2",
                "protocol_source": "CODE",
                "caption_field": "short",
                "query_selection": "all_3_per_candidate",
            },
            evaluate_id_retrieval(
                caption_embeddings, caption_ids, audio_embeddings, audio_ids
            ),
        ),
        (
            {
                "protocol_id": "t2a_public848_short_seed0",
                "task": "t2a",
                "paper_table": "Table 2 sensitivity",
                "protocol_source": "CODE",
                "caption_field": "short",
                "query_selection": "seed0_one_per_candidate",
                "query_selection_seed": args.seed,
            },
            evaluate_id_retrieval(
                caption_embeddings,
                caption_ids,
                audio_embeddings,
                audio_ids,
                query_indices=seed0_indices,
            ),
        ),
        (
            {
                "protocol_id": "t2t_public848_short_seed0",
                "task": "t2t",
                "paper_table": "Table 3",
                "protocol_source": "CODE",
                "caption_field": "short",
                "query_selection": "seed0_one_per_candidate",
                "query_selection_seed": args.seed,
            },
            evaluate_caption_to_caption(
                caption_embeddings, caption_ids, query_indices=seed0_indices
            ),
        ),
        (
            {
                "protocol_id": "t2t_public848_short_all",
                "task": "t2t",
                "paper_table": "Table 3 sensitivity",
                "protocol_source": "CODE",
                "caption_field": "short",
                "query_selection": "all_3_per_candidate",
            },
            evaluate_caption_to_caption(caption_embeddings, caption_ids),
        ),
    )
    for definition, result in definitions:
        protocols.append(
            save_result(
                output_dir,
                {
                    **definition,
                    "public_candidate_count": args.expected_candidates,
                    "paper_candidate_count": args.paper_candidates,
                    "strict_paper_reproduction": False,
                },
                result,
            )
        )

    summary = {
        "schema_version": 1,
        "status": "complete",
        "model": args.model,
        "dataset": "MECAT-Caption 00A/test public release",
        "started_at": started_at,
        "finished_at": now(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
        ).strip(),
        "git_status_short": subprocess.check_output(
            ["git", "status", "--short"], cwd=REPOSITORY_ROOT, text=True
        ).strip(),
        "public_candidate_count": args.expected_candidates,
        "paper_candidate_count": args.paper_candidates,
        "caption_count": expected_caption_count,
        "caption_field": "short",
        "caption_protocol_source": (
            "CODE: AudioRetrieval/scripts/"
            "mine_hard_negatives_laion.py:load_mecat_captions"
        ),
        "strict_paper_reproduction": False,
        "no_post_hoc_protocol_selection": True,
        "claim_boundary": (
            "Public MECAT has 848 candidates and exactly three short captions per "
            "candidate. PAPER reports 847 and does not publish its excluded ID, "
            "caption-field selection, or T2T self/tie protocol."
        ),
        "source_usage": {
            "oea_official_source_used": True,
            "official_metric_file": "AudioRetrieval/evaluation/metrics.py",
            "compatibility_layer": "AudioRetrieval/evaluation/canonical.py",
        },
        "inputs": {
            "audio_embeddings": identity(audio_path),
            "caption_embeddings": identity(caption_path),
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
    print("MECAT_TABLE2_TABLE3_STATUS=complete")
    print(f"PUBLIC_CANDIDATES={report['public_candidate_count']}")
    print(f"PAPER_CANDIDATES={report['paper_candidate_count']}")
    print("STRICT_PAPER_REPRODUCTION=no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
