#!/usr/bin/env python3
"""Evaluate official-source OEA Clotho NPZ artifacts under explicit protocols."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from AudioRetrieval.evaluation.canonical import (
    CanonicalRetrievalResult,
    evaluate_caption_to_caption,
    evaluate_id_retrieval,
)


QUERY_TYPES = ("question", "imperative", "paraphrase", "tagging")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--uiq-dir", type=Path, required=True)
    parser.add_argument("--captions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="OEA-Nemo3B-AC")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def now() -> str:
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


def write_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


def strings(values: np.ndarray) -> list[str]:
    return [str(value) for value in values.tolist()]


def select_one_per_clip(clip_ids: list[str], ordered_ids: list[str], seed: int) -> list[int]:
    positions: dict[str, list[int]] = {clip_id: [] for clip_id in ordered_ids}
    for index, clip_id in enumerate(clip_ids):
        if clip_id not in positions:
            raise ValueError(f"caption clip ID absent from audio candidates: {clip_id}")
        positions[clip_id].append(index)
    rng = random.Random(seed)
    selected = [rng.choice(positions[clip_id]) for clip_id in ordered_ids]
    selected.sort()
    return selected


def save_result(root: Path, protocol: dict[str, Any], result: CanonicalRetrievalResult) -> dict[str, Any]:
    output = root / "protocols" / protocol["protocol_id"]
    output.mkdir(parents=True, exist_ok=False)
    write_npy(output / "ranks.npy", result.ranks)
    write_npy(output / "rankings.npy", result.rankings)
    write_npy(output / "evaluated_query_indices.npy", result.evaluated_query_indices)
    write_json(output / "positive_indices.json", [list(row) for row in result.positive_indices])
    write_json(output / "ignored_indices.json", [list(row) for row in result.ignored_indices])
    report = {
        **protocol,
        "status": "complete",
        "evaluated_queries": int(result.ranks.shape[0]),
        "candidate_count": int(result.rankings.shape[1]),
        "metrics": result.metrics,
        "tie_policy": result.tie_policy,
    }
    write_json(output / "metrics.json", report)
    report["artifacts"] = {
        name: identity(output / name)
        for name in (
            "metrics.json",
            "ranks.npy",
            "rankings.npy",
            "evaluated_query_indices.npy",
            "positive_indices.json",
            "ignored_indices.json",
        )
    }
    return report


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = now()
    baseline_audio_path = args.baseline_dir / "audio_embeddings.npz"
    baseline_caption_path = args.baseline_dir / "caption_embeddings.npz"
    audio = load_npz(baseline_audio_path)
    captions = load_npz(baseline_caption_path)
    audio_embeddings = np.asarray(audio["embeddings"])
    caption_embeddings = np.asarray(captions["embeddings"])
    audio_ids = strings(audio["clip_ids"])
    caption_ids = strings(captions["clip_ids"])

    if audio_embeddings.shape != (1045, 512):
        raise ValueError(f"unexpected audio shape: {audio_embeddings.shape}")
    if caption_embeddings.shape != (5225, 512):
        raise ValueError(f"unexpected caption shape: {caption_embeddings.shape}")
    if len(set(audio_ids)) != 1045:
        raise ValueError("audio clip IDs are not unique")
    caption_counts = Counter(caption_ids)
    if any(caption_counts[clip_id] != 5 for clip_id in audio_ids):
        raise ValueError("each audio clip must own exactly five captions")

    with args.captions_csv.open(encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != len(audio_ids):
        raise ValueError("caption CSV row count does not match audio embeddings")
    filename_to_audio_id = {
        row["file_name"].strip(): audio_ids[index]
        for index, row in enumerate(csv_rows)
    }
    if len(filename_to_audio_id) != 1045:
        raise ValueError("caption CSV filenames are not unique")

    seed0_indices = select_one_per_clip(caption_ids, audio_ids, args.seed)
    protocol_results: list[dict[str, Any]] = []

    protocol_results.append(
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
                caption_embeddings,
                caption_ids,
                audio_embeddings,
                audio_ids,
            ),
        )
    )
    protocol_results.append(
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
    protocol_results.append(
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
                caption_embeddings,
                caption_ids,
                query_indices=seed0_indices,
            ),
        )
    )
    protocol_results.append(
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
    for query_type in QUERY_TYPES:
        path = args.uiq_dir / f"uiq_{query_type}_embeddings.npz"
        data = load_npz(path)
        embeddings = np.asarray(data["embeddings"])
        raw_ids = strings(data["clip_ids"])
        if embeddings.shape != (1045, 512):
            raise ValueError(f"unexpected {query_type} shape: {embeddings.shape}")
        missing = sorted(set(raw_ids) - set(filename_to_audio_id))
        if missing:
            raise KeyError(f"{query_type} IDs absent from caption CSV: {missing[:10]}")
        target_ids = [filename_to_audio_id[value] for value in raw_ids]
        if len(set(target_ids)) != 1045:
            raise ValueError(f"{query_type} targets are not one-to-one")
        protocol_results.append(
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
        "dataset": "Clotho v2.1 evaluation",
        "started_at": started_at,
        "finished_at": now(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status_short": subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).strip(),
        "seed": args.seed,
        "source_usage": {
            "official_embedding_source": True,
            "official_metric_file": "AudioRetrieval/evaluation/metrics.py",
            "compatibility_layer": "AudioRetrieval/evaluation/canonical.py",
            "reason": "upstream evaluator omits T2T and cannot align released UIQ filename IDs",
        },
        "inputs": {
            "audio_embeddings": identity(baseline_audio_path),
            "caption_embeddings": identity(baseline_caption_path),
            "captions_csv": identity(args.captions_csv),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
