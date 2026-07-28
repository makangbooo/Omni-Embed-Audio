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
    load_corpus,
    load_squtr_audio_manifest,
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
    parser.add_argument("--fiqa-root", type=Path, required=True)
    parser.add_argument("--audio-manifest", type=Path, required=True)
    parser.add_argument("--phase2-cache-root", type=Path, required=True)
    parser.add_argument("--query-count", type=int, default=8)
    parser.add_argument("--negative-document-count", type=int, default=64)
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
    }


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.generate_oea_embeddings import load_model_bundle

    if args.query_count < 2 or args.negative_document_count <= 0:
        raise ValueError("query-count must be at least two and negatives positive")
    config_path = args.resolved_model_config.resolve()
    model_root = args.model_root.resolve()
    fiqa_root = args.fiqa_root.resolve()
    audio_manifest_path = args.audio_manifest.resolve()
    cache_root = args.phase2_cache_root.resolve()
    output = args.output.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    corpus_path = fiqa_root / "corpus.jsonl"
    qrels_path = fiqa_root / "qrels/test.jsonl"
    corpus = load_corpus(corpus_path)
    qrels = load_unbounded_qrels(qrels_path)
    records = [
        record
        for record in load_squtr_audio_manifest(
            audio_manifest_path,
            require_audio_files=True,
        ).values()
        if squtr_subset_name(record.subset) == "fiqa"
        and record.condition == "clean"
        and record.query_id in qrels
    ]
    record_by_query = {record.query_id: record for record in records}
    if len(record_by_query) != len(records):
        raise ValueError("duplicate Clean SQuTR-FiQA audio records")
    query_ids = sorted(record_by_query)[: args.query_count]
    if len(query_ids) != args.query_count:
        raise ValueError("insufficient Clean SQuTR-FiQA audio records")
    positive_ids = {
        document_id
        for query_id in query_ids
        for document_id, relevance in qrels[query_id].items()
        if relevance > 0.0
    }
    missing_positive = sorted(positive_ids - set(corpus))
    if missing_positive:
        raise ValueError(f"qrels positives absent from corpus: {missing_positive[:20]}")
    negative_ids = [
        document_id
        for document_id in sorted(corpus)
        if document_id not in positive_ids
    ][: args.negative_document_count]
    document_ids = sorted(positive_ids) + negative_ids
    audio_paths = [Path(record_by_query[query_id].audio_path) for query_id in query_ids]
    document_texts = [corpus[document_id].constructed_text for document_id in document_ids]

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
    base_cached_audio = cached_rows(
        cache_root=cache_root,
        mode="vanilla",
        kind="audio",
        identifiers=query_ids,
    )
    base_cached_documents = cached_rows(
        cache_root=cache_root,
        mode="vanilla",
        kind="document",
        identifiers=document_ids,
    )
    full_cached_audio = cached_rows(
        cache_root=cache_root,
        mode="oea",
        kind="audio",
        identifiers=query_ids,
    )
    full_cached_documents = cached_rows(
        cache_root=cache_root,
        mode="oea",
        kind="document",
        identifiers=document_ids,
    )
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
        "query_ids": query_ids,
        "document_ids": document_ids,
        "positive_document_count": len(positive_ids),
        "negative_document_count": len(negative_ids),
        "spaces": evaluations,
        "cache_agreement": {
            "fresh_base_audio_vs_vanilla_cache": compare_embeddings(
                spaces["base_hidden"][0],
                base_cached_audio,
            ),
            "fresh_base_text_vs_vanilla_cache": compare_embeddings(
                spaces["base_hidden"][1],
                base_cached_documents,
            ),
            "fresh_full_audio_vs_oea_cache": compare_embeddings(
                spaces["lora_plus_oea_heads"][0],
                full_cached_audio,
            ),
            "fresh_full_text_vs_oea_cache": compare_embeddings(
                spaces["lora_plus_oea_heads"][1],
                full_cached_documents,
            ),
        },
        "gpu": {
            "name": torch_module.cuda.get_device_name(device),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "provenance": {
            "resolved_model_config": file_record(config_path),
            "corpus": file_record(corpus_path),
            "qrels": file_record(qrels_path),
            "audio_manifest": file_record(audio_manifest_path),
            "phase2_oea_manifest": file_record(
                cache_root / "oea/clean/cache_manifest.json"
            ),
            "phase2_vanilla_manifest": file_record(
                cache_root / "vanilla/clean/cache_manifest.json"
            ),
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
