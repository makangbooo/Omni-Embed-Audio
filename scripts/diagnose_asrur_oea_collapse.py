#!/usr/bin/env python3
"""Attribute the SQuTR-FiQA OEA collapse to LoRA and/or projection heads.

The same deterministic audio/document subset is encoded in four spaces:

* frozen base backbone pooled hidden states;
* LoRA-enabled pooled hidden states;
* frozen base pooled states passed through the OEA projection heads; and
* LoRA-enabled pooled states passed through the OEA projection heads.

Fresh base/full outputs are also compared with the immutable Phase-2 vanilla
and OEA caches.  No checkpoint, formal cache, or candidate ranking is changed.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import heapq
import json
import math
import subprocess
import sys
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from AudioRetrieval.asr_uncertainty_reranking.artifacts import (  # noqa: E402
    load_unbounded_qrels,
)
from AudioRetrieval.asr_uncertainty_reranking.data import (  # noqa: E402
    construct_document_text,
    load_squtr_audio_manifest,
    read_jsonl,
    squtr_subset_name,
)
from scripts.diagnose_asrur_phase2_no_go import load_ids  # noqa: E402

SPACE_NAMES = (
    "base_hidden",
    "lora_hidden",
    "base_plus_oea_heads",
    "lora_plus_oea_heads",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved-model-config", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-root", "--fiqa-root", dest="dataset_root", type=Path, required=True,
        help="MTEB-style dataset root containing corpus.jsonl and qrels/test.jsonl",
    )
    parser.add_argument("--subset", choices=("fiqa", "nq"), default="fiqa")
    parser.add_argument("--audio-manifest", type=Path, required=True)
    parser.add_argument(
        "--exclude-query-ids-json",
        type=Path,
        help="Optional prior diagnostic JSON or JSON list whose query IDs are excluded",
    )
    parser.add_argument(
        "--phase2-cache-root", type=Path,
        help="Optional immutable cache root used only for fresh/cache agreement",
    )
    parser.add_argument("--query-count", type=int, default=256)
    parser.add_argument("--negative-document-count", type=int, default=4096)
    parser.add_argument("--sample-seed", type=int, default=20260805)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260805)
    parser.add_argument("--output", type=Path, required=True)
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


def load_excluded_query_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("query_ids")
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError("excluded query artifact must contain a query_ids string list")
    if len(value) != len(set(value)):
        raise ValueError("excluded query IDs must be unique")
    return set(value)


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def l2_normalize(torch_module: Any, values: Any) -> Any:
    return torch_module.nn.functional.normalize(values.float(), p=2, dim=-1)


def encode_pooled(
    *,
    torch_module: Any,
    adapter: Any,
    model: Any,
    device: Any,
    texts: Sequence[str] | None = None,
    audio_paths: Sequence[Path] | None = None,
    batch_size: int,
    disable_lora: bool,
) -> Any:
    from AudioRetrieval.training.oea.train_omniembed_lora import encode_batch

    if (texts is None) == (audio_paths is None):
        raise ValueError("exactly one of texts or audio_paths must be provided")
    count = len(texts) if texts is not None else len(audio_paths or ())
    if count == 0 or batch_size <= 0:
        raise ValueError("encoding input and batch size must be positive")
    if disable_lora and not hasattr(model, "disable_adapter"):
        raise RuntimeError("loaded PEFT model cannot disable its adapter")
    context = model.disable_adapter() if disable_lora else contextlib.nullcontext()
    chunks = []
    with context, torch_module.inference_mode():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pooled, valid_indices = encode_batch(
                    adapter,
                    model,
                    adapter.processor,
                    list(texts[start:stop]) if texts is not None else None,
                    (
                        [Path(value) for value in audio_paths[start:stop]]
                        if audio_paths is not None
                        else None
                    ),
                    device,
                )
            messages = [str(item.message) for item in caught]
            if any(
                "silence" in message.lower()
                or "failed to load" in message.lower()
                for message in messages
            ):
                raise RuntimeError(f"audio fallback warning detected: {messages}")
            if pooled is None or valid_indices != list(range(stop - start)):
                raise RuntimeError(
                    f"encoder skipped inputs at range {start}:{stop}: {valid_indices}"
                )
            chunks.append(pooled.detach())
    return torch_module.cat(chunks, dim=0)


def off_diagonal_summary(values: np.ndarray) -> dict[str, float | None]:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("embedding matrix must be non-empty and rank two")
    if not np.isfinite(matrix).all():
        raise ValueError("embedding matrix contains non-finite values")
    similarities = matrix @ matrix.T
    if matrix.shape[0] == 1:
        return {"mean": None, "std": None, "minimum": None, "maximum": None}
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    values = similarities[mask]
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def compare_embeddings(fresh: np.ndarray, cached: np.ndarray) -> dict[str, float]:
    left = np.asarray(fresh, dtype=np.float32)
    right = np.asarray(cached, dtype=np.float32)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError(f"fresh/cached shape mismatch: {left.shape}/{right.shape}")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("fresh/cached comparison contains non-finite values")
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    diagonal_cosines = np.sum(left * right, axis=1)
    return {
        "mean_row_cosine": float(diagonal_cosines.mean()),
        "minimum_row_cosine": float(diagonal_cosines.min()),
        "maximum_absolute_difference": float(np.max(np.abs(left - right))),
    }


def evaluate_space(
    *,
    audio: np.ndarray,
    documents: np.ndarray,
    query_ids: Sequence[str],
    document_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    audio_values = np.asarray(audio, dtype=np.float32)
    document_values = np.asarray(documents, dtype=np.float32)
    if (
        audio_values.ndim != 2
        or document_values.ndim != 2
        or audio_values.shape[0] != len(query_ids)
        or document_values.shape[0] != len(document_ids)
        or audio_values.shape[1] != document_values.shape[1]
    ):
        raise ValueError("space matrix/identity dimensions differ")
    scores = audio_values @ document_values.T
    document_positions = {
        document_id: index for index, document_id in enumerate(document_ids)
    }
    positive_scores = []
    positive_ranks = []
    top1 = []
    per_query = []
    recall = {1: 0, 5: 0, 10: 0}
    for query_index, query_id in enumerate(query_ids):
        positive_positions = {
            document_positions[document_id]
            for document_id, relevance in qrels[query_id].items()
            if relevance > 0.0 and document_id in document_positions
        }
        if not positive_positions:
            raise ValueError(f"diagnostic candidate set lost positives: {query_id}")
        ordering = np.argsort(-scores[query_index], kind="stable")
        ranks = [
            rank
            for rank, position in enumerate(ordering, start=1)
            if int(position) in positive_positions
        ]
        best_rank = min(ranks)
        positive_ranks.append(best_rank)
        positive_scores.append(
            max(float(scores[query_index, position]) for position in positive_positions)
        )
        top1.append(document_ids[int(ordering[0])])
        for cutoff in recall:
            recall[cutoff] += int(best_rank <= cutoff)
        per_query.append(
            {
                "query_id": query_id,
                "best_positive_rank": int(best_rank),
                "reciprocal_rank": 1.0 / float(best_rank),
                "positive_score": float(positive_scores[-1]),
                "top1_document_id": top1[-1],
                **{
                    f"hit_at_{cutoff}": int(best_rank <= cutoff)
                    for cutoff in sorted(recall)
                },
            }
        )
    return {
        "dimension": int(audio_values.shape[1]),
        "query_count": len(query_ids),
        "document_count": len(document_ids),
        "audio_off_diagonal_cosine": off_diagonal_summary(audio_values),
        "document_off_diagonal_cosine": off_diagonal_summary(document_values),
        "positive_score_mean": float(np.mean(positive_scores)),
        "positive_score_minimum": float(np.min(positive_scores)),
        "best_positive_rank_mean": float(np.mean(positive_ranks)),
        "best_positive_rank_maximum": int(max(positive_ranks)),
        "unique_top1_document_count": len(set(top1)),
        "top1_most_common": [
            {"document_id": document_id, "count": count}
            for document_id, count in Counter(top1).most_common(10)
        ],
        "query_recall": {
            f"Recall@{cutoff}": recall[cutoff] / len(query_ids)
            for cutoff in sorted(recall)
        },
        "mean_reciprocal_rank": float(
            np.mean([row["reciprocal_rank"] for row in per_query])
        ),
        "per_query": per_query,
    }


def paired_bootstrap_delta(
    method: Sequence[float],
    baseline: Sequence[float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Estimate a paired mean delta without assuming normality."""

    left = np.asarray(method, dtype=np.float64)
    right = np.asarray(baseline, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("paired bootstrap inputs must be equal one-dimensional arrays")
    if left.size < 2 or iterations < 100:
        raise ValueError("paired bootstrap needs at least two rows and 100 iterations")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("paired bootstrap inputs contain non-finite values")
    differences = left - right
    generator = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 1000):
        stop = min(start + 1000, iterations)
        indices = generator.integers(
            0,
            differences.size,
            size=(stop - start, differences.size),
        )
        draws[start:stop] = differences[indices].mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    probability_nonpositive = float(np.mean(draws <= 0.0))
    probability_nonnegative = float(np.mean(draws >= 0.0))
    return {
        "sample_count": int(differences.size),
        "observed_mean_delta": float(differences.mean()),
        "confidence_interval_95": [float(lower), float(upper)],
        "probability_delta_positive": float(np.mean(draws > 0.0)),
        "two_sided_p_value": float(
            min(1.0, 2.0 * min(probability_nonpositive, probability_nonnegative))
        ),
        "iterations": iterations,
        "seed": seed,
    }


def compare_spaces(
    evaluations: Mapping[str, Mapping[str, Any]],
    *,
    baseline: str,
    method: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    baseline_rows = evaluations[baseline]["per_query"]
    method_rows = evaluations[method]["per_query"]
    baseline_ids = [row["query_id"] for row in baseline_rows]
    if baseline_ids != [row["query_id"] for row in method_rows]:
        raise ValueError("space comparison query order differs")
    fields = {
        "Recall@1": "hit_at_1",
        "Recall@5": "hit_at_5",
        "Recall@10": "hit_at_10",
        "MRR": "reciprocal_rank",
    }
    return {
        "baseline": baseline,
        "method": method,
        "metrics": {
            metric: paired_bootstrap_delta(
                [float(row[field]) for row in method_rows],
                [float(row[field]) for row in baseline_rows],
                iterations=iterations,
                seed=seed + index,
            )
            for index, (metric, field) in enumerate(fields.items())
        },
    }


def factorial_component_effects(
    evaluations: Mapping[str, Mapping[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Estimate LoRA, projection-head, and interaction effects in a 2x2 design."""

    fields = {
        "Recall@1": "hit_at_1",
        "Recall@5": "hit_at_5",
        "Recall@10": "hit_at_10",
        "MRR": "reciprocal_rank",
    }
    rows = {
        name: evaluations[name]["per_query"]
        for name in SPACE_NAMES
    }
    query_ids = [row["query_id"] for row in rows["base_hidden"]]
    if any(
        [row["query_id"] for row in rows[name]] != query_ids
        for name in SPACE_NAMES[1:]
    ):
        raise ValueError("factorial attribution query order differs")
    output = {}
    for metric_index, (metric, field) in enumerate(fields.items()):
        base = np.asarray(
            [float(row[field]) for row in rows["base_hidden"]], dtype=np.float64
        )
        lora = np.asarray(
            [float(row[field]) for row in rows["lora_hidden"]], dtype=np.float64
        )
        heads = np.asarray(
            [float(row[field]) for row in rows["base_plus_oea_heads"]],
            dtype=np.float64,
        )
        full = np.asarray(
            [float(row[field]) for row in rows["lora_plus_oea_heads"]],
            dtype=np.float64,
        )
        lora_main = 0.5 * ((lora - base) + (full - heads))
        heads_main = 0.5 * ((heads - base) + (full - lora))
        interaction = full - lora - heads + base
        zero = np.zeros_like(base)
        effect_seed = seed + metric_index * 10
        output[metric] = {
            "lora_main_effect": paired_bootstrap_delta(
                lora_main, zero, iterations=iterations, seed=effect_seed
            ),
            "projection_head_main_effect": paired_bootstrap_delta(
                heads_main, zero, iterations=iterations, seed=effect_seed + 1
            ),
            "lora_projection_interaction": paired_bootstrap_delta(
                interaction, zero, iterations=iterations, seed=effect_seed + 2
            ),
            "lora_minus_projection_head_effect": paired_bootstrap_delta(
                lora_main - heads_main,
                zero,
                iterations=iterations,
                seed=effect_seed + 3,
            ),
        }
    primary = output["MRR"]
    contrast_interval = primary["lora_minus_projection_head_effect"][
        "confidence_interval_95"
    ]
    if contrast_interval[1] < 0.0:
        dominant = "lora"
    elif contrast_interval[0] > 0.0:
        dominant = "projection_heads"
    else:
        dominant = "statistically_unresolved"
    output["predeclared_primary_decision"] = {
        "metric": "MRR",
        "lora_contributes_to_loss": primary["lora_main_effect"][
            "confidence_interval_95"
        ][1]
        < 0.0,
        "projection_heads_contribute_to_loss": primary[
            "projection_head_main_effect"
        ]["confidence_interval_95"][1]
        < 0.0,
        "dominant_component": dominant,
        "decision_rule": (
            "A component contributes when its MRR main-effect 95% CI is below "
            "zero. Dominance requires the LoRA-minus-head MRR effect CI to "
            "exclude zero; otherwise dominance is unresolved."
        ),
    }
    return output


def cached_rows(
    *,
    cache_root: Path,
    mode: str,
    kind: str,
    identifiers: Sequence[str],
) -> np.ndarray:
    if mode not in {"oea", "vanilla"} or kind not in {"audio", "document"}:
        raise ValueError("unsupported cache mode or kind")
    root = cache_root / mode / "clean"
    ids = load_ids(root / f"{kind}_ids.jsonl")
    positions = {identifier: index for index, identifier in enumerate(ids)}
    missing = [identifier for identifier in identifiers if identifier not in positions]
    if missing:
        raise ValueError(f"cache lacks {mode}/{kind} IDs: {missing[:20]}")
    matrix = np.load(
        root / f"{kind}_embeddings.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    return np.asarray(
        matrix[[positions[identifier] for identifier in identifiers]],
        dtype=np.float32,
    )


def select_corpus_rows(
    path: Path,
    *,
    positive_ids: set[str],
    negative_count: int,
    sample_seed: int,
) -> tuple[dict[str, Any], list[str], int]:
    """Select positives plus deterministic negatives without loading a huge corpus."""
    if negative_count <= 0:
        raise ValueError("negative_count must be positive")
    negative_heap: list[tuple[int, str]] = []
    positive_seen: set[str] = set()
    for _, row in read_jsonl(path):
        document_id = str(row.get("_id", "")).strip()
        if not document_id:
            raise ValueError(f"corpus row missing _id: {path}")
        if document_id in positive_ids:
            positive_seen.add(document_id)
        else:
            digest = int.from_bytes(hashlib.sha256(
                f"{sample_seed}:negative:{document_id}".encode("utf-8")
            ).digest(), byteorder="big")
            item = (-digest, document_id)
            if len(negative_heap) < negative_count:
                heapq.heappush(negative_heap, item)
            elif item > negative_heap[0]:
                heapq.heapreplace(negative_heap, item)
    missing = sorted(positive_ids - positive_seen)
    if missing:
        raise ValueError(f"qrels positives absent from corpus: {missing[:20]}")
    negative_ids = [
        document_id
        for negative_digest, document_id in sorted(
            negative_heap, key=lambda item: (-item[0], item[1])
        )
    ]
    selected_ids = sorted(positive_ids) + negative_ids
    selected = set(selected_ids)
    rows: dict[str, Any] = {}
    for line_number, row in read_jsonl(path):
        document_id = str(row.get("_id", "")).strip()
        if document_id not in selected:
            continue
        title = row.get("title", "")
        text = row.get("text", "")
        if not isinstance(title, str) or not isinstance(text, str):
            raise TypeError(f"corpus title/text must be strings: {path}:{line_number}")
        rows[document_id] = construct_document_text(title, text)
    if set(rows) != selected:
        raise ValueError("selected corpus rows changed between scans")
    return rows, negative_ids, len(positive_seen)


def run(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.generate_oea_embeddings import load_model_bundle

    if args.query_count < 2 or args.negative_document_count <= 0:
        raise ValueError("query-count must be at least two and negatives positive")
    if args.bootstrap_iterations < 100:
        raise ValueError("bootstrap-iterations must be at least 100")
    config_path = args.resolved_model_config.resolve()
    model_root = args.model_root.resolve()
    dataset_root = args.dataset_root.resolve()
    audio_manifest_path = args.audio_manifest.resolve()
    exclude_path = (
        args.exclude_query_ids_json.resolve()
        if args.exclude_query_ids_json is not None
        else None
    )
    excluded_query_ids = load_excluded_query_ids(exclude_path)
    cache_root = args.phase2_cache_root.resolve() if args.phase2_cache_root else None
    output = args.output.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    corpus_path = dataset_root / "corpus.jsonl"
    qrels_path = dataset_root / "qrels/test.jsonl"
    qrels = load_unbounded_qrels(qrels_path)
    records = [
        record
        for record in load_squtr_audio_manifest(
            audio_manifest_path,
            require_audio_files=True,
        ).values()
        if squtr_subset_name(record.subset) == args.subset
        and record.condition == "clean"
        and record.query_id in qrels
        and record.query_id not in excluded_query_ids
    ]
    record_by_query = {record.query_id: record for record in records}
    if len(record_by_query) != len(records):
        raise ValueError("duplicate Clean SQuTR-FiQA audio records")
    query_ids = sorted(
        record_by_query,
        key=lambda query_id: hashlib.sha256(
            f"{args.sample_seed}:{query_id}".encode("utf-8")
        ).hexdigest(),
    )[: args.query_count]
    if len(query_ids) != args.query_count:
        raise ValueError("insufficient Clean SQuTR-FiQA audio records")
    positive_ids = {
        document_id
        for query_id in query_ids
        for document_id, relevance in qrels[query_id].items()
        if relevance > 0.0
    }
    corpus, negative_ids, _ = select_corpus_rows(
        corpus_path,
        positive_ids=positive_ids,
        negative_count=args.negative_document_count,
        sample_seed=args.sample_seed,
    )
    document_ids = sorted(positive_ids) + negative_ids
    audio_paths = [Path(record_by_query[query_id].audio_path) for query_id in query_ids]
    document_texts = [corpus[document_id] for document_id in document_ids]

    output.parent.mkdir(parents=True, exist_ok=True)
    model_load_report = {
        "schema_version": 1,
        "status": "loading",
        "metrics_path": str(output.parent / "oea_model_load.json"),
    }
    torch_module, adapter, model, audio_head, text_head, device = load_model_bundle(
        config,
        model_root,
        model_load_report,
    )
    torch_module.cuda.reset_peak_memory_stats(device)

    base_audio = encode_pooled(
        torch_module=torch_module,
        adapter=adapter,
        model=model,
        device=device,
        audio_paths=audio_paths,
        batch_size=1,
        disable_lora=True,
    )
    base_text = encode_pooled(
        torch_module=torch_module,
        adapter=adapter,
        model=model,
        device=device,
        texts=document_texts,
        batch_size=4,
        disable_lora=True,
    )
    lora_audio = encode_pooled(
        torch_module=torch_module,
        adapter=adapter,
        model=model,
        device=device,
        audio_paths=audio_paths,
        batch_size=1,
        disable_lora=False,
    )
    lora_text = encode_pooled(
        torch_module=torch_module,
        adapter=adapter,
        model=model,
        device=device,
        texts=document_texts,
        batch_size=4,
        disable_lora=False,
    )
    with torch_module.inference_mode():
        spaces_torch = {
            "base_hidden": (
                l2_normalize(torch_module, base_audio),
                l2_normalize(torch_module, base_text),
            ),
            "lora_hidden": (
                l2_normalize(torch_module, lora_audio),
                l2_normalize(torch_module, lora_text),
            ),
            "base_plus_oea_heads": (
                audio_head(base_audio.to(dtype=next(audio_head.parameters()).dtype)),
                text_head(base_text.to(dtype=next(text_head.parameters()).dtype)),
            ),
            "lora_plus_oea_heads": (
                audio_head(lora_audio.to(dtype=next(audio_head.parameters()).dtype)),
                text_head(lora_text.to(dtype=next(text_head.parameters()).dtype)),
            ),
        }
    spaces = {
        name: (
            audio.detach().cpu().float().numpy(),
            documents.detach().cpu().float().numpy(),
        )
        for name, (audio, documents) in spaces_torch.items()
    }
    if set(spaces) != set(SPACE_NAMES):
        raise AssertionError("attribution spaces differ from the locked protocol")

    evaluations = {
        name: evaluate_space(
            audio=audio,
            documents=documents,
            query_ids=query_ids,
            document_ids=document_ids,
            qrels=qrels,
        )
        for name, (audio, documents) in spaces.items()
    }
    comparison_specs = (
        ("base_hidden", "lora_hidden"),
        ("base_hidden", "base_plus_oea_heads"),
        ("lora_hidden", "lora_plus_oea_heads"),
        ("base_plus_oea_heads", "lora_plus_oea_heads"),
        ("base_hidden", "lora_plus_oea_heads"),
    )
    component_effects = {
        f"{baseline}_to_{method}": compare_spaces(
            evaluations,
            baseline=baseline,
            method=method,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed + 10 * index,
        )
        for index, (baseline, method) in enumerate(comparison_specs)
    }
    factorial_effects = factorial_component_effects(
        evaluations,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed + 100,
    )
    full_loss = component_effects[
        "base_hidden_to_lora_plus_oea_heads"
    ]["metrics"]
    capability_loss_decision = {
        "primary_metric": "MRR",
        "secondary_metric": "Recall@10",
        "base_capability_loss_supported": (
            full_loss["MRR"]["confidence_interval_95"][1] < 0.0
            and full_loss["Recall@10"]["confidence_interval_95"][1] < 0.0
        ),
        "decision_rule": (
            "Both full-OEA minus base-hidden MRR and Recall@10 paired-bootstrap "
            "95% confidence intervals must be strictly below zero."
        ),
    }
    cache_agreement = None
    if cache_root is not None:
        base_cached_audio = cached_rows(cache_root=cache_root, mode="vanilla", kind="audio", identifiers=query_ids)
        base_cached_documents = cached_rows(cache_root=cache_root, mode="vanilla", kind="document", identifiers=document_ids)
        full_cached_audio = cached_rows(cache_root=cache_root, mode="oea", kind="audio", identifiers=query_ids)
        full_cached_documents = cached_rows(cache_root=cache_root, mode="oea", kind="document", identifiers=document_ids)
        cache_agreement = {
            "fresh_base_audio_vs_vanilla_cache": compare_embeddings(spaces["base_hidden"][0], base_cached_audio),
            "fresh_base_text_vs_vanilla_cache": compare_embeddings(spaces["base_hidden"][1], base_cached_documents),
            "fresh_full_audio_vs_oea_cache": compare_embeddings(spaces["lora_plus_oea_heads"][0], full_cached_audio),
            "fresh_full_text_vs_oea_cache": compare_embeddings(spaces["lora_plus_oea_heads"][1], full_cached_documents),
        }
    peak_allocated = int(torch_module.cuda.max_memory_allocated(device))
    peak_reserved = int(torch_module.cuda.max_memory_reserved(device))
    return {
        "schema_version": 1,
        "status": "complete",
        "stage": "asrur_oea_lora_projection_collapse_attribution",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output(
            "status",
            "--short",
            "--untracked-files=all",
        ),
        "variant": config.get("official_variant_id", config.get("model")),
        "subset": args.subset,
        "query_ids": query_ids,
        "document_ids": document_ids,
        "positive_document_count": len(positive_ids),
        "negative_document_count": len(negative_ids),
        "sampling": {
            "method": "sha256_seeded_query_order",
            "sample_seed": args.sample_seed,
            "excluded_query_count": len(excluded_query_ids),
        },
        "spaces": evaluations,
        "component_effects": component_effects,
        "factorial_component_effects": factorial_effects,
        "capability_loss_decision": capability_loss_decision,
        "cache_agreement": cache_agreement,
        "gpu": {
            "name": torch_module.cuda.get_device_name(device),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "provenance": {
            "resolved_model_config": file_record(config_path),
            "dataset_root": file_record(corpus_path),
            "qrels": file_record(qrels_path),
            "audio_manifest": file_record(audio_manifest_path),
            **(
                {"excluded_query_ids": file_record(exclude_path)}
                if exclude_path is not None
                else {}
            ),
            **({
                "phase2_oea_manifest": file_record(cache_root / "oea/clean/cache_manifest.json"),
                "phase2_vanilla_manifest": file_record(cache_root / "vanilla/clean/cache_manifest.json"),
            } if cache_root is not None else {}),
        },
        "interpretation_rules": {
            "projection_head_failure": (
                "base_hidden/lora_hidden remain discriminative while one or "
                "both projected spaces collapse"
            ),
            "lora_failure": (
                "base_hidden remains discriminative while lora_hidden collapses"
            ),
            "cache_or_runtime_mismatch": (
                "fresh base/full embeddings do not closely agree with their "
                "immutable vanilla/OEA cache rows"
            ),
            "domain_failure": (
                "fresh/cache agreement is high and the degradation appears "
                "only after applying the released OEA adaptation components"
            ),
        },
        "claim_boundary": (
            "This diagnostic attributes the fixed released checkpoint on a "
            "small deterministic subset. It does not select a new checkpoint "
            "or authorize changing formal candidate generation."
        ),
    }


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse output: {output}")
    result = run(args)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
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
    temporary.replace(output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "spaces": list(result["spaces"]),
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
