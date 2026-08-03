#!/usr/bin/env python3
"""Evaluate released negative UIQ NPZ embeddings with explicit inferred pairings."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

from AudioRetrieval.evaluation.negative_canonical import (  # noqa: E402
    evaluate_negative_id_retrieval,
)
from scripts.evaluate_embedding_artifacts import (  # noqa: E402
    file_identity,
    load_jsonl_objects,
    write_json,
    write_jsonl,
    write_npy,
)
from scripts.evaluate_negative_embedding_artifacts import per_query_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-npz", type=Path, required=True)
    parser.add_argument("--query-npz", type=Path, required=True)
    parser.add_argument("--pairing-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-candidates", type=int, required=True)
    parser.add_argument("--expected-queries", type=int, required=True)
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def strings(array: np.ndarray, field: str) -> list[str]:
    if array.ndim != 1:
        raise ValueError(f"{field} must be one-dimensional")
    values = [str(value).strip() for value in array.tolist()]
    if any(not value for value in values):
        raise ValueError(f"{field} contains an empty ID")
    if len(set(values)) != len(values):
        raise ValueError(f"{field} contains duplicate IDs")
    return values


def resolve_pairing_candidate_ids(
    requested_ids: Sequence[str],
    candidate_ids: Sequence[str],
    *,
    label: str,
) -> tuple[list[str], dict[str, int]]:
    """Resolve release IDs to candidate IDs using only unique ID aliases."""

    exact = set(candidate_ids)
    aliases: dict[str, list[str]] = defaultdict(list)
    for candidate_id in candidate_ids:
        for alias in dict.fromkeys(
            (candidate_id.casefold(), Path(candidate_id).stem.casefold())
        ):
            aliases[alias].append(candidate_id)

    resolved: list[str] = []
    methods: Counter[str] = Counter()
    for index, requested_id in enumerate(requested_ids):
        if requested_id in exact:
            resolved.append(requested_id)
            methods["exact_candidate_id"] += 1
            continue
        candidates = set(aliases.get(requested_id.casefold(), ()))
        candidates.update(aliases.get(Path(requested_id).stem.casefold(), ()))
        if len(candidates) != 1:
            raise KeyError(
                f"{label} row {index}: candidate alias count={len(candidates)} "
                f"for {requested_id!r}; candidates={sorted(candidates)[:10]}"
            )
        resolved.append(next(iter(candidates)))
        methods["unique_casefold_or_stem_candidate_id"] += 1
    return resolved, dict(sorted(methods.items()))


def run_evaluation(
    *,
    audio_npz: Path,
    query_npz: Path,
    pairing_jsonl: Path,
    output_dir: Path,
    model: str,
    dataset: str,
    expected_candidates: int,
    expected_queries: int,
    require_clean_git: bool = True,
) -> dict[str, Any]:
    audio_npz = audio_npz.resolve()
    query_npz = query_npz.resolve()
    pairing_jsonl = pairing_jsonl.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (audio_npz, query_npz, pairing_jsonl):
        if not path.is_file():
            raise FileNotFoundError(path)

    git_commit = git_output("rev-parse", "HEAD")
    git_status = git_output("status", "--short")
    if require_clean_git and git_status:
        raise RuntimeError(f"formal evaluation requires clean Git: {git_status!r}")

    with np.load(audio_npz, allow_pickle=True) as audio:
        candidate_embeddings = np.asarray(audio["embeddings"], dtype=np.float32)
        candidate_ids = strings(np.asarray(audio["clip_ids"]), "audio clip_ids")
    with np.load(query_npz, allow_pickle=True) as query:
        query_embeddings = np.asarray(query["embeddings"], dtype=np.float32)

    pairings = load_jsonl_objects(pairing_jsonl)
    if candidate_embeddings.shape[0] != expected_candidates:
        raise ValueError("audio candidate count mismatch")
    if len(candidate_ids) != expected_candidates:
        raise ValueError("audio candidate ID count mismatch")
    if query_embeddings.shape[0] != expected_queries:
        raise ValueError("negative query embedding count mismatch")
    if len(pairings) != expected_queries:
        raise ValueError("pairing row count mismatch")
    if candidate_embeddings.ndim != 2 or query_embeddings.ndim != 2:
        raise ValueError("embedding arrays must be two-dimensional")
    if candidate_embeddings.shape[1] != query_embeddings.shape[1]:
        raise ValueError("audio/query embedding dimensions differ")

    query_ids: list[str] = []
    target_ids: list[str] = []
    hard_negative_ids: list[str] = []
    for index, row in enumerate(pairings):
        for field in ("query_id", "target_id", "hard_negative_id"):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"pairing row {index}: invalid {field}")
        query_ids.append(str(row["query_id"]).strip())
        target_ids.append(str(row["target_id"]).strip())
        hard_negative_ids.append(str(row["hard_negative_id"]).strip())
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("pairing query IDs are not unique")

    resolved_target_ids, target_id_resolution = resolve_pairing_candidate_ids(
        target_ids, candidate_ids, label="target_id"
    )
    resolved_hard_negative_ids, hard_negative_id_resolution = (
        resolve_pairing_candidate_ids(
            hard_negative_ids, candidate_ids, label="hard_negative_id"
        )
    )
    equal_rows = [
        index
        for index, (target_id, hard_negative_id) in enumerate(
            zip(resolved_target_ids, resolved_hard_negative_ids)
        )
        if target_id == hard_negative_id
    ]
    if equal_rows:
        raise ValueError(
            "resolved target and hard-negative IDs must differ; rows="
            f"{equal_rows[:10]}"
        )

    result = evaluate_negative_id_retrieval(
        query_embeddings,
        query_ids,
        candidate_embeddings,
        candidate_ids,
        resolved_target_ids,
        resolved_hard_negative_ids,
        normalize=True,
        ks=(1, 5, 10),
    )
    rows = per_query_rows(result, query_ids, candidate_ids, (1, 5, 10))

    write_npy(output_dir / "target_ranks.npy", result.target_ranks)
    write_npy(output_dir / "hard_negative_ranks.npy", result.hard_negative_ranks)
    write_npy(output_dir / "rankings.npy", result.rankings)
    write_npy(output_dir / "evaluated_query_indices.npy", result.evaluated_query_indices)
    write_jsonl(output_dir / "per_query.jsonl", rows)
    write_jsonl(output_dir / "pairing_metadata.jsonl", pairings)

    artifact_names = (
        "target_ranks.npy",
        "hard_negative_ranks.npy",
        "rankings.npy",
        "evaluated_query_indices.npy",
        "per_query.jsonl",
        "pairing_metadata.jsonl",
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "paper_tables": ["Table 4", "Table 17"],
        "model": model,
        "dataset": dataset,
        "git_commit": git_commit,
        "git_status_short": git_status,
        "protocol_source": "INFERRED",
        "strict_paper_reproduction": False,
        "pairing_contract": "deterministic_released_caption_identity",
        "candidate_id_resolution": {
            "target_ids": target_id_resolution,
            "hard_negative_ids": hard_negative_id_resolution,
        },
        "candidate_count": expected_candidates,
        "evaluated_query_count": expected_queries,
        "embedding_dimension": int(candidate_embeddings.shape[1]),
        "normalization_applied": True,
        "tie_policy": result.tie_policy,
        "metrics": result.metrics,
        "inputs": {
            "audio_npz": file_identity(audio_npz),
            "query_npz": file_identity(query_npz),
            "pairing_jsonl": file_identity(pairing_jsonl),
        },
        "artifacts": {
            name: file_identity(output_dir / name) for name in artifact_names
        },
    }
    write_json(output_dir / "metrics.json", report)
    return report


def main() -> int:
    args = parse_args()
    report = run_evaluation(
        audio_npz=args.audio_npz,
        query_npz=args.query_npz,
        pairing_jsonl=args.pairing_jsonl,
        output_dir=args.output_dir,
        model=args.model,
        dataset=args.dataset,
        expected_candidates=args.expected_candidates,
        expected_queries=args.expected_queries,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
