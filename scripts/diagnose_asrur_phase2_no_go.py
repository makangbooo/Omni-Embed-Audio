#!/usr/bin/env python3
"""Diagnose a completed Phase-2 NO-GO without changing model or candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from AudioRetrieval.asr_uncertainty_reranking.aggregation import (  # noqa: E402
    normalized_word_edit_distance,
    pairwise_edit_distance_stats,
)
from AudioRetrieval.asr_uncertainty_reranking.artifacts import (  # noqa: E402
    load_frozen_candidates,
    load_nbest,
    load_unbounded_qrels,
)
from AudioRetrieval.asr_uncertainty_reranking.data import (  # noqa: E402
    load_text_queries,
    normalized_query_text,
    read_jsonl,
)
from AudioRetrieval.asr_uncertainty_reranking.metrics import (  # noqa: E402
    corpus_wer,
)

CONDITIONS = ("clean", "snr_20", "snr_10", "snr_0")
MODES = ("oea", "vanilla")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fiqa-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--example-count", type=int, default=8)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def load_ids(path: Path) -> list[str]:
    result = []
    for expected_index, (line_number, row) in enumerate(read_jsonl(path)):
        if row.get("index") != expected_index:
            raise ValueError(f"non-contiguous index: {path}:{line_number}")
        value = row.get("id")
        if not isinstance(value, str) or not value:
            raise ValueError(f"invalid ID: {path}:{line_number}")
        result.append(value)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"ID artifact must be non-empty and unique: {path}")
    return result


def matrix_geometry(path: Path, *, sample_size: int) -> dict[str, Any]:
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"invalid embedding matrix shape: {path}: {matrix.shape}")
    if matrix.dtype != np.float32:
        raise ValueError(f"embedding matrix must be float32: {path}: {matrix.dtype}")
    norm_min = math.inf
    norm_max = -math.inf
    for start in range(0, int(matrix.shape[0]), 4096):
        block = np.asarray(matrix[start : start + 4096], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError(f"embedding matrix contains non-finite values: {path}")
        norms = np.linalg.norm(block, axis=1)
        norm_min = min(norm_min, float(norms.min()))
        norm_max = max(norm_max, float(norms.max()))
    count = min(sample_size, int(matrix.shape[0]))
    indices = np.linspace(
        0,
        int(matrix.shape[0]) - 1,
        num=count,
        dtype=np.int64,
    )
    sample = np.asarray(matrix[indices], dtype=np.float32)
    similarities = sample @ sample.T
    mask = ~np.eye(count, dtype=bool)
    off_diagonal = similarities[mask]
    return {
        "path": str(path.resolve()),
        "shape": [int(value) for value in matrix.shape],
        "dtype": str(matrix.dtype),
        "norm_min": norm_min,
        "norm_max": norm_max,
        "sample_size": count,
        "sample_centroid_norm": float(np.linalg.norm(sample.mean(axis=0))),
        "sample_off_diagonal_cosine": {
            "mean": float(off_diagonal.mean()),
            "std": float(off_diagonal.std()),
            "minimum": float(off_diagonal.min()),
            "maximum": float(off_diagonal.max()),
            "p01": float(np.quantile(off_diagonal, 0.01)),
            "p50": float(np.quantile(off_diagonal, 0.50)),
            "p99": float(np.quantile(off_diagonal, 0.99)),
        },
    }


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("numeric summary requires a non-empty finite vector")
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "minimum": float(array.min()),
        "p01": float(np.quantile(array, 0.01)),
        "p50": float(np.quantile(array, 0.50)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(array.max()),
    }


def retrieval_geometry(
    *,
    mode: str,
    condition: str,
    cache_root: Path,
    qrels: Mapping[str, Mapping[str, float]],
    sample_size: int,
    document_geometry: Mapping[str, Any],
) -> dict[str, Any]:
    if mode not in MODES or condition not in CONDITIONS:
        raise ValueError("unsupported mode or condition")
    embedding_root = cache_root / mode / condition
    document_root = cache_root / mode / "clean"
    audio_path = embedding_root / "audio_embeddings.npy"
    audio_ids_path = embedding_root / "audio_ids.jsonl"
    document_path = document_root / "document_embeddings.npy"
    document_ids_path = document_root / "document_ids.jsonl"
    ranking_path = cache_root / "rankings" / mode / f"{condition}.jsonl"
    manifest_path = embedding_root / "cache_manifest.json"

    audio_ids = load_ids(audio_ids_path)
    document_ids = load_ids(document_ids_path)
    if set(audio_ids) != set(qrels):
        raise ValueError(f"{mode}/{condition} audio/qrels query sets differ")
    document_positions = {
        document_id: index for index, document_id in enumerate(document_ids)
    }
    audio = np.load(audio_path, mmap_mode="r", allow_pickle=False)
    documents = np.load(document_path, mmap_mode="r", allow_pickle=False)
    if audio.shape[0] != len(audio_ids) or documents.shape[0] != len(document_ids):
        raise ValueError(f"{mode}/{condition} embedding/ID row counts differ")
    if audio.shape[1] != documents.shape[1]:
        raise ValueError(f"{mode}/{condition} audio/document dimensions differ")

    positive_scores = []
    for audio_index, query_id in enumerate(audio_ids):
        relevant_positions = [
            document_positions[document_id]
            for document_id, relevance in qrels[query_id].items()
            if relevance > 0.0
        ]
        if not relevant_positions:
            raise ValueError(f"query has no positive document: {query_id}")
        query = np.asarray(audio[audio_index], dtype=np.float32)
        positives = np.asarray(documents[relevant_positions], dtype=np.float32)
        positive_scores.append(float(np.max(positives @ query)))

    rankings = load_frozen_candidates(ranking_path)
    if set(rankings) != set(qrels):
        raise ValueError(f"{mode}/{condition} ranking/qrels query sets differ")
    top1_documents = [rankings[query_id].candidate_ids[0] for query_id in audio_ids]
    top1_scores = [rankings[query_id].scores[0] for query_id in audio_ids]
    top1_frequency = Counter(top1_documents)
    relevant_in_top100 = []
    positive_ranks = []
    for query_id in audio_ids:
        positive_ids = {
            document_id
            for document_id, relevance in qrels[query_id].items()
            if relevance > 0.0
        }
        ranks = [
            index
            for index, document_id in enumerate(
                rankings[query_id].candidate_ids,
                start=1,
            )
            if document_id in positive_ids
        ]
        relevant_in_top100.append(bool(ranks))
        if ranks:
            positive_ranks.append(min(ranks))
    return {
        "mode": mode,
        "condition": condition,
        "audio_geometry": matrix_geometry(audio_path, sample_size=sample_size),
        "document_geometry": dict(document_geometry),
        "best_positive_cosine": _summary(positive_scores),
        "top1_score": _summary(top1_scores),
        "top1_unique_document_count": len(top1_frequency),
        "top1_most_common_documents": [
            {"document_id": document_id, "count": count}
            for document_id, count in top1_frequency.most_common(10)
        ],
        "queries_with_positive_in_top100": sum(relevant_in_top100),
        "queries_without_positive_in_top100": (
            len(relevant_in_top100) - sum(relevant_in_top100)
        ),
        "best_positive_rank_when_present": (
            _summary(positive_ranks) if positive_ranks else None
        ),
        "provenance": {
            "cache_manifest": file_record(manifest_path),
            "ranking": file_record(ranking_path),
        },
    }


def asr_condition_diagnostic(
    *,
    condition: str,
    nbest_path: Path,
    references: Mapping[str, str],
    example_count: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    nbest = load_nbest(nbest_path, expected_size=4)
    if set(nbest) != set(references):
        raise ValueError(f"{condition} N-best/reference query sets differ")
    top1 = {
        query_id: nbest[query_id].hypotheses[0].text
        for query_id in sorted(references)
    }
    frequency = Counter(top1.values())
    edit_means = []
    edit_maxima = []
    unique_hypothesis_counts = []
    for query_id in sorted(references):
        hypotheses = [
            value.text for value in nbest[query_id].hypotheses
        ]
        edit_mean, edit_max = pairwise_edit_distance_stats(hypotheses)
        edit_means.append(edit_mean)
        edit_maxima.append(edit_max)
        unique_hypothesis_counts.append(len(set(hypotheses)))
    exact = sum(
        normalized_query_text(references[query_id])
        == normalized_query_text(top1[query_id])
        for query_id in references
    )
    examples = [
        {
            "query_id": query_id,
            "reference": references[query_id],
            "top1": top1[query_id],
            "hypotheses": [
                value.text for value in nbest[query_id].hypotheses
            ],
            "top1_normalized_word_edit_distance": (
                normalized_word_edit_distance(
                    references[query_id],
                    top1[query_id],
                )
            ),
        }
        for query_id in sorted(references)[:example_count]
    ]
    return (
        {
            "condition": condition,
            "query_count": len(references),
            "whitespace_casefold_corpus_wer": corpus_wer(
                (references[query_id], top1[query_id])
                for query_id in sorted(references)
            ),
            "exact_normalized_top1_count": exact,
            "empty_top1_count": sum(not value.strip() for value in top1.values()),
            "unique_top1_count": len(frequency),
            "top1_most_common": [
                {"text": text, "count": count}
                for text, count in frequency.most_common(20)
            ],
            "top1_word_count": _summary(
                [len(value.split()) for value in top1.values()]
            ),
            "nbest_unique_hypothesis_count": _summary(
                unique_hypothesis_counts
            ),
            "nbest_pairwise_edit_mean": _summary(edit_means),
            "nbest_pairwise_edit_max": _summary(edit_maxima),
            "examples": examples,
            "provenance": {"nbest": file_record(nbest_path)},
        },
        top1,
    )


def cross_condition_agreement(
    top1_by_condition: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    result = []
    for left, right in combinations(CONDITIONS, 2):
        left_values = top1_by_condition[left]
        right_values = top1_by_condition[right]
        if set(left_values) != set(right_values):
            raise ValueError(f"{left}/{right} top1 query sets differ")
        query_ids = sorted(left_values)
        exact = sum(
            left_values[query_id] == right_values[query_id]
            for query_id in query_ids
        )
        distances = [
            normalized_word_edit_distance(
                left_values[query_id],
                right_values[query_id],
            )
            for query_id in query_ids
        ]
        result.append(
            {
                "left": left,
                "right": right,
                "exact_top1_count": exact,
                "exact_top1_fraction": exact / len(query_ids),
                "normalized_word_edit_distance": _summary(distances),
            }
        )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.sample_size < 2 or args.example_count <= 0:
        raise ValueError("sample-size must be at least 2 and example-count positive")
    fiqa_root = args.fiqa_root.resolve()
    cache_root = args.cache_root.resolve()
    queries_path = fiqa_root / "queries.jsonl"
    qrels_path = fiqa_root / "qrels/test.jsonl"
    all_queries = load_text_queries(queries_path)
    qrels = load_unbounded_qrels(qrels_path)
    missing = sorted(set(qrels) - set(all_queries))
    if missing:
        raise ValueError(f"test qrels reference absent queries: {missing[:20]}")
    references = {
        query_id: all_queries[query_id].text for query_id in sorted(qrels)
    }

    asr = {}
    top1_by_condition = {}
    for condition in CONDITIONS:
        diagnostic, top1 = asr_condition_diagnostic(
            condition=condition,
            nbest_path=cache_root / "whisper" / condition / "nbest.jsonl",
            references=references,
            example_count=args.example_count,
        )
        asr[condition] = diagnostic
        top1_by_condition[condition] = top1

    retrieval = {}
    for mode in MODES:
        document_geometry = matrix_geometry(
            cache_root / mode / "clean" / "document_embeddings.npy",
            sample_size=args.sample_size,
        )
        retrieval[mode] = {
            condition: retrieval_geometry(
                mode=mode,
                condition=condition,
                cache_root=cache_root,
                qrels=qrels,
                sample_size=args.sample_size,
                document_geometry=document_geometry,
            )
            for condition in CONDITIONS
        }
    return {
        "schema_version": 1,
        "status": "complete",
        "stage": "asrur_phase2_no_go_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output(
            "status",
            "--short",
            "--untracked-files=all",
        ),
        "cache_root": str(cache_root),
        "query_count": len(references),
        "asr": asr,
        "asr_cross_condition_agreement": cross_condition_agreement(
            top1_by_condition
        ),
        "retrieval": retrieval,
        "input_provenance": {
            "queries": file_record(queries_path),
            "qrels": file_record(qrels_path),
        },
        "claim_boundary": (
            "This CPU-only diagnostic characterizes completed immutable "
            "artifacts. It does not authorize changing the OEA checkpoint, "
            "candidate generator, ASR protocol, or test-set selection."
        ),
    }


def main() -> int:
    args = parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse output: {args.output}")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "query_count": result["query_count"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
