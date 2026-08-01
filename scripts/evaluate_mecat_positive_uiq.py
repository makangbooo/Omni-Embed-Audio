#!/usr/bin/env python3
"""Evaluate MECAT positive UIQ with the audited public 848-row candidate set."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from AudioRetrieval.evaluation.canonical import evaluate_id_retrieval
from scripts.evaluate_official_source_oea_clotho import (
    QUERY_TYPES,
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
    parser.add_argument("--uiq-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-candidates", type=int, default=848)
    parser.add_argument("--paper-candidates", type=int, default=847)
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = now()

    audio_path = args.audio_embedding_dir / "audio_embeddings.npz"
    audio = load_npz(audio_path)
    audio_embeddings = np.asarray(audio["embeddings"])
    audio_ids = strings(audio["clip_ids"])
    if audio_embeddings.ndim != 2:
        raise ValueError(f"unexpected audio shape: {audio_embeddings.shape}")
    if audio_embeddings.shape[0] != args.expected_candidates:
        raise ValueError(f"unexpected candidate count: {audio_embeddings.shape[0]}")
    if len(audio_ids) != len(set(audio_ids)):
        raise ValueError("MECAT audio candidate IDs are not unique")
    if not np.isfinite(audio_embeddings).all():
        raise ValueError("MECAT audio embeddings contain non-finite values")

    protocol_results: list[dict[str, Any]] = []
    uiq_inputs: dict[str, dict[str, Any]] = {}
    for query_type in QUERY_TYPES:
        path = args.uiq_dir / f"uiq_{query_type}_embeddings.npz"
        data = load_npz(path)
        embeddings = np.asarray(data["embeddings"])
        target_ids = strings(data["clip_ids"])
        expected_shape = (args.expected_candidates, audio_embeddings.shape[1])
        if embeddings.shape != expected_shape:
            raise ValueError(
                f"unexpected {query_type} shape: {embeddings.shape} != {expected_shape}"
            )
        if len(target_ids) != len(set(target_ids)):
            raise ValueError(f"{query_type} target IDs are not unique")
        if set(target_ids) != set(audio_ids):
            raise ValueError(f"{query_type} target IDs differ from candidate IDs")
        if not np.isfinite(embeddings).all():
            raise ValueError(f"{query_type} embeddings contain non-finite values")
        protocol_results.append(
            save_result(
                output_dir,
                {
                    "protocol_id": f"{query_type}_released_uiq_public_848",
                    "task": "uiq",
                    "paper_table": {
                        "question": "Table 12",
                        "imperative": "Table 13",
                        "paraphrase": "Table 14",
                        "tagging": "Table 15",
                    }[query_type],
                    "protocol_source": "CODE",
                    "query_selection": "released_uiq_all_public_848",
                    "paper_candidate_count": args.paper_candidates,
                    "strict_paper_reproduction": False,
                },
                evaluate_id_retrieval(
                    embeddings,
                    target_ids,
                    audio_embeddings,
                    audio_ids,
                ),
            )
        )
        uiq_inputs[query_type] = identity(path)

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
        "protocol_status": "controlled_public_848_row_reproduction",
        "strict_paper_reproduction": False,
        "claim_boundary": (
            "The public MECAT archive and all four released positive UIQ files "
            f"contain {args.expected_candidates} IDs. PAPER reports "
            f"{args.paper_candidates}, but does not publish the excluded ID or rule."
        ),
        "source_usage": {
            "oea_official_source_used": True,
            "official_metric_file": "AudioRetrieval/evaluation/metrics.py",
            "compatibility_layer": "AudioRetrieval/evaluation/canonical.py",
        },
        "inputs": {
            "audio_embeddings": identity(audio_path),
            "uiq_embeddings": uiq_inputs,
        },
        "protocols": protocol_results,
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
    print("MECAT_POSITIVE_UIQ_STATUS=complete")
    print(f"PUBLIC_CANDIDATES={report['public_candidate_count']}")
    print(f"PAPER_CANDIDATES={report['paper_candidate_count']}")
    print("STRICT_PAPER_REPRODUCTION=no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
