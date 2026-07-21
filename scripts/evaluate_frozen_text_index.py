#!/usr/bin/env python3
"""Evaluate fixed audio-query embeddings against an immutable text index."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.evaluate_embedding_artifacts import (  # noqa: E402
    file_identity,
    load_jsonl_objects,
    resolve_path,
    write_json,
    write_jsonl,
    write_npy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported frozen-index config schema_version")
    required = {
        "candidate_embeddings",
        "candidate_metadata",
        "checkpoint",
        "dataset",
        "experiment_id",
        "model",
        "protocol_label",
        "protocol_source",
        "qrels",
        "query_embeddings",
        "query_metadata",
        "seed",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"frozen-index config missing fields: {missing}")
    if config["protocol_source"] not in {"PAPER", "CODE", "INFERRED"}:
        raise ValueError("protocol_source must be PAPER, CODE, or INFERRED")
    for field in ("checkpoint", "dataset", "experiment_id", "model"):
        if not isinstance(config[field], str) or not config[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool):
        raise ValueError("seed must be an integer")
    cutoffs = config.get("cutoffs", [1, 5, 10])
    if (
        not isinstance(cutoffs, list)
        or not cutoffs
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in cutoffs
        )
        or len(set(cutoffs)) != len(cutoffs)
    ):
        raise ValueError("cutoffs must be a non-empty list of unique positive integers")
    config["cutoffs"] = sorted(cutoffs)
    ranking_depth = config.get("ranking_depth", max(config["cutoffs"]))
    if (
        not isinstance(ranking_depth, int)
        or isinstance(ranking_depth, bool)
        or ranking_depth < max(config["cutoffs"])
    ):
        raise ValueError("ranking_depth must be at least the largest cutoff")
    config["ranking_depth"] = ranking_depth
    query_batch_size = config.get("query_batch_size", 64)
    if (
        not isinstance(query_batch_size, int)
        or isinstance(query_batch_size, bool)
        or query_batch_size <= 0
    ):
        raise ValueError("query_batch_size must be a positive integer")
    config["query_batch_size"] = query_batch_size
    return config


def metadata_ids(
    rows: Sequence[Mapping[str, Any]], field: str, label: str
) -> list[str]:
    values: list[str] = []
    for index, row in enumerate(rows):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} row {index}: {field} must be non-empty")
        values.append(value.strip())
    if len(set(values)) != len(values):
        raise ValueError(f"{label} {field} values must be unique")
    return values


def load_qrels(
    path: Path,
    *,
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
    query_field: str,
    candidate_field: str,
    relevance_field: str,
) -> dict[str, dict[str, float]]:
    query_set = set(query_ids)
    candidate_set = set(candidate_ids)
    qrels: dict[str, dict[str, float]] = {}
    for row_index, row in enumerate(load_jsonl_objects(path)):
        query_id = row.get(query_field)
        candidate_id = row.get(candidate_field)
        relevance = row.get(relevance_field)
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError(f"qrels row {row_index}: invalid query ID")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError(f"qrels row {row_index}: invalid candidate ID")
        if (
            isinstance(relevance, bool)
            or not isinstance(relevance, (int, float))
            or not math.isfinite(float(relevance))
            or float(relevance) <= 0.0
        ):
            raise ValueError(f"qrels row {row_index}: relevance must be finite and > 0")
        query_id = query_id.strip()
        candidate_id = candidate_id.strip()
        if query_id not in query_set:
            raise KeyError(f"qrels query ID is absent from query metadata: {query_id}")
        if candidate_id not in candidate_set:
            raise KeyError(
                f"qrels candidate ID is absent from frozen index: {candidate_id}"
            )
        query_qrels = qrels.setdefault(query_id, {})
        if candidate_id in query_qrels:
            raise ValueError(
                f"duplicate qrel for query={query_id!r}, candidate={candidate_id!r}"
            )
        query_qrels[candidate_id] = float(relevance)
    missing = [query_id for query_id in query_ids if query_id not in qrels]
    if missing:
        raise KeyError(f"queries without positive qrels: {missing[:10]}")
    return qrels


def normalized_embeddings(path: Path, expected_rows: int, label: str) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2 or array.shape[0] != expected_rows or array.shape[1] == 0:
        raise ValueError(
            f"{label} shape mismatch: got {array.shape}, expected rows={expected_rows}"
        )
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain finite numeric values")
    array = array.astype(np.float32, copy=False)
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{label} contains zero-norm rows")
    return array / norms[:, None]


def evaluate(
    query_embeddings: np.ndarray,
    query_ids: Sequence[str],
    candidate_embeddings: np.ndarray,
    candidate_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, float]],
    *,
    cutoffs: Sequence[int],
    ranking_depth: int,
    query_batch_size: int = 64,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if query_embeddings.shape[1] != candidate_embeddings.shape[1]:
        raise ValueError("query and frozen-index embedding dimensions differ")
    candidate_lookup = {
        candidate_id: index for index, candidate_id in enumerate(candidate_ids)
    }
    candidate_order = np.arange(len(candidate_ids), dtype=np.int64)
    depth = min(ranking_depth, len(candidate_ids))
    top_indices = np.empty((len(query_ids), depth), dtype=np.int64)
    top_scores = np.empty((len(query_ids), depth), dtype=np.float32)
    hit_counts = {cutoff: 0 for cutoff in cutoffs}
    reciprocal_ranks: list[float] = []
    ndcg_values: list[float] = []
    per_query: list[dict[str, Any]] = []
    ndcg_cutoff = 10

    for batch_start in range(0, len(query_ids), query_batch_size):
        batch_stop = min(batch_start + query_batch_size, len(query_ids))
        score_block = query_embeddings[batch_start:batch_stop] @ candidate_embeddings.T
        for local_index, scores in enumerate(score_block):
            query_index = batch_start + local_index
            query_id = query_ids[query_index]
            ranking = np.lexsort((candidate_order, -scores))
            top = ranking[:depth]
            top_indices[query_index] = top
            top_scores[query_index] = scores[top]
            relevant = qrels[query_id]
            relevant_indices = {
                candidate_lookup[candidate_id] for candidate_id in relevant
            }
            relevant_ranks = [
                rank
                for rank, candidate_index in enumerate(ranking.tolist(), start=1)
                if candidate_index in relevant_indices
            ]
            best_rank = min(relevant_ranks)
            for cutoff in cutoffs:
                hit_counts[cutoff] += int(best_rank <= cutoff)
            reciprocal_ranks.append(1.0 / best_rank if best_rank <= 10 else 0.0)

            gains = [
                (2.0 ** relevant.get(candidate_ids[index], 0.0)) - 1.0
                for index in ranking[:ndcg_cutoff]
            ]
            dcg = sum(
                gain / math.log2(rank + 1.0)
                for rank, gain in enumerate(gains, 1)
            )
            ideal_gains = sorted(
                ((2.0 ** value) - 1.0 for value in relevant.values()), reverse=True
            )[:ndcg_cutoff]
            idcg = sum(
                gain / math.log2(rank + 1.0)
                for rank, gain in enumerate(ideal_gains, 1)
            )
            ndcg = dcg / idcg if idcg > 0.0 else 0.0
            ndcg_values.append(ndcg)
            per_query.append(
                {
                    "query_index": query_index,
                    "query_id": query_id,
                    "best_relevant_rank": best_rank,
                    "relevant_candidates": [
                        {"candidate_id": candidate_id, "relevance": relevance}
                        for candidate_id, relevance in relevant.items()
                    ],
                    "reciprocal_rank_at_10": reciprocal_ranks[-1],
                    "ndcg_at_10": ndcg,
                    "top_candidate_ids": [candidate_ids[index] for index in top],
                }
            )

    metrics = {
        **{
            f"R@{cutoff}": 100.0 * hit_counts[cutoff] / len(query_ids)
            for cutoff in cutoffs
        },
        "MRR@10": float(np.mean(reciprocal_ranks)),
        "nDCG@10": float(np.mean(ndcg_values)),
    }
    return metrics, top_indices, top_scores, per_query


def run_evaluation(
    config_path: Path, output_dir: Path, argv: Sequence[str] | None = None
) -> dict[str, Any]:
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_preexisting = {
        "environment.txt",
        "stderr.log",
        "stdout.log",
        "wrapper_git_commit.txt",
        "wrapper_git_status.txt",
    }
    unexpected = sorted(
        path.name for path in output_dir.iterdir() if path.name not in allowed_preexisting
    )
    if unexpected:
        raise FileExistsError(
            f"output directory contains prior artifacts: {unexpected}"
        )
    started_at = utc_now()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": started_at,
        "finished_at": None,
        "error": None,
    }
    write_json(output_dir / "metrics.json", report)
    try:
        config = load_config(config_path)
        if output_dir.name != config["experiment_id"]:
            raise ValueError("output directory basename must equal experiment_id")
        git_commit = git_output("rev-parse", "HEAD")
        git_status = git_output("status", "--short")
        if config.get("require_clean_git", True) and git_status:
            raise RuntimeError(f"formal evaluation requires clean Git: {git_status!r}")
        paths = {
            name: resolve_path(config_path, config[name])
            for name in (
                "query_embeddings",
                "query_metadata",
                "candidate_embeddings",
                "candidate_metadata",
                "qrels",
            )
        }
        if any(path is None or not path.is_file() for path in paths.values()):
            missing = [name for name, path in paths.items() if path is None or not path.is_file()]
            raise FileNotFoundError(f"missing frozen-index inputs: {missing}")
        input_identities = {name: file_identity(path) for name, path in paths.items()}
        query_rows = load_jsonl_objects(paths["query_metadata"])
        candidate_rows = load_jsonl_objects(paths["candidate_metadata"])
        query_ids = metadata_ids(
            query_rows, str(config.get("query_id_field", "query_id")), "query metadata"
        )
        candidate_ids = metadata_ids(
            candidate_rows,
            str(config.get("candidate_id_field", "candidate_id")),
            "candidate metadata",
        )
        qrels = load_qrels(
            paths["qrels"],
            query_ids=query_ids,
            candidate_ids=candidate_ids,
            query_field=str(config.get("qrels_query_id_field", "query_id")),
            candidate_field=str(
                config.get("qrels_candidate_id_field", "candidate_id")
            ),
            relevance_field=str(config.get("qrels_relevance_field", "relevance")),
        )
        queries = normalized_embeddings(
            paths["query_embeddings"], len(query_ids), "query_embeddings"
        )
        candidates = normalized_embeddings(
            paths["candidate_embeddings"], len(candidate_ids), "candidate_embeddings"
        )
        metrics, top_indices, top_scores, per_query = evaluate(
            queries,
            query_ids,
            candidates,
            candidate_ids,
            qrels,
            cutoffs=config["cutoffs"],
            ranking_depth=config["ranking_depth"],
            query_batch_size=config["query_batch_size"],
        )
        after_identities = {name: file_identity(path) for name, path in paths.items()}
        if after_identities != input_identities:
            raise RuntimeError("an input changed while the frozen-index evaluation ran")

        write_json(output_dir / "config.yaml", config)
        write_npy(output_dir / "top_ranking_indices.npy", top_indices)
        write_npy(output_dir / "top_ranking_scores.npy", top_scores)
        write_jsonl(output_dir / "per_query.jsonl", per_query)
        write_json(
            output_dir / "frozen_index_identity.json",
            {
                "candidate_embeddings": input_identities["candidate_embeddings"],
                "candidate_metadata": input_identities["candidate_metadata"],
                "candidate_count": len(candidate_ids),
                "embedding_dimension": int(candidates.shape[1]),
                "normalization_at_scoring": "L2",
                "unchanged_during_evaluation": True,
            },
        )
        (output_dir / "git_commit.txt").write_text(git_commit + "\n", encoding="utf-8")
        (output_dir / "git_status.txt").write_text(git_status + "\n", encoding="utf-8")
        command = list(argv if argv is not None else sys.argv)
        (output_dir / "command.sh").write_text(
            " ".join(shlex.quote(value) for value in command) + "\n", encoding="utf-8"
        )
        report.update(
            {
                "status": "complete",
                "finished_at": utc_now(),
                "experiment_id": config["experiment_id"],
                "model": config["model"],
                "checkpoint": config["checkpoint"],
                "dataset": config["dataset"],
                "protocol_label": config["protocol_label"],
                "protocol_source": config["protocol_source"],
                "seed": config["seed"],
                "randomness_used_by_evaluator": False,
                "git_commit": git_commit,
                "git_status_short": git_status,
                "query_count": len(query_ids),
                "candidate_count": len(candidate_ids),
                "embedding_dimension": int(candidates.shape[1]),
                "cutoffs": config["cutoffs"],
                "ranking_depth": int(top_indices.shape[1]),
                "query_batch_size": config["query_batch_size"],
                "qrel_count": sum(len(values) for values in qrels.values()),
                "inputs": input_identities,
                "metrics": metrics,
                "artifacts": {
                    name: file_identity(output_dir / name)
                    for name in (
                        "config.yaml",
                        "frozen_index_identity.json",
                        "per_query.jsonl",
                        "top_ranking_indices.npy",
                        "top_ranking_scores.npy",
                    )
                },
            }
        )
        write_json(output_dir / "metrics.json", report)
        return report
    except BaseException as exc:
        report.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(output_dir / "metrics.json", report)
        raise


def main() -> int:
    args = parse_args()
    report = run_evaluation(args.config, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
