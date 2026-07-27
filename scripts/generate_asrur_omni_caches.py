#!/usr/bin/env python3
"""Generate resumable FiQA document/audio caches for OEA or its base model."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from AudioRetrieval.asr_uncertainty_reranking.cache_manifest import (  # noqa: E402
    build_cache_manifest,
    file_record,
    write_cache_manifest_once,
)
from AudioRetrieval.asr_uncertainty_reranking.configuration import (  # noqa: E402
    load_main_experiment_config,
)
from AudioRetrieval.asr_uncertainty_reranking.data import (  # noqa: E402
    SQuTR_CONDITIONS,
    load_corpus,
    load_squtr_audio_manifest,
    squtr_subset_name,
)
from scripts.build_official_oea_eval_config import (  # noqa: E402
    verify_official_model_lock_binding,
)
from scripts.build_vanilla_backbone_eval_config import (  # noqa: E402
    verify_vanilla_model_lock_binding,
)
from scripts.generate_asrur_frozen_caches import (  # noqa: E402
    consolidate_embedding_chunks,
    ensure_identity,
    formal_execution_guard,
    git_output,
    load_embedding_chunk,
    ranges,
    save_embedding_chunk,
    write_text_once_or_verify,
)
from scripts.generate_oea_embeddings import (  # noqa: E402
    atomic_write_json,
    load_model_bundle as load_oea_bundle,
    project_batch,
)
from scripts.generate_vanilla_backbone_embeddings import (  # noqa: E402
    encode_base_batch,
    load_model_bundle as load_vanilla_bundle,
    verify_locked_base_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("oea", "original_omni"), required=True)
    parser.add_argument("--main-config", type=Path, required=True)
    parser.add_argument("--resolved-model-config", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--audio-manifest", type=Path, required=True)
    parser.add_argument("--subset", default="fiqa")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=SQuTR_CONDITIONS,
        default=list(SQuTR_CONDITIONS),
    )
    parser.add_argument("--text-batch-size", type=int, required=True)
    parser.add_argument("--audio-batch-size", type=int, required=True)
    parser.add_argument(
        "--content",
        choices=("both", "audio_only"),
        default="both",
        help=(
            "Encode documents and one acoustic condition, or encode only a "
            "later condition while reusing the first run's document cache."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_inputs(
    *,
    corpus_path: Path,
    audio_manifest: Path,
    subset: str,
    conditions: list[str],
) -> tuple[list[object], list[object]]:
    documents = list(load_corpus(corpus_path).values())
    audio = [
        value
        for value in load_squtr_audio_manifest(
            audio_manifest,
            require_audio_files=True,
        ).values()
        if squtr_subset_name(value.subset) == subset.casefold()
        and value.condition in conditions
    ]
    audio.sort(key=lambda value: (value.condition, value.query_id, value.record_id))
    if not documents or not audio:
        raise ValueError("selected FiQA document/audio input is empty")
    return documents, audio


def model_binding(mode: str, config: dict) -> dict:
    if mode == "oea":
        return verify_official_model_lock_binding(config)
    if mode == "original_omni":
        return verify_vanilla_model_lock_binding(config)
    raise AssertionError(mode)


def main() -> int:
    args = parse_args()
    if args.text_batch_size <= 0 or args.audio_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if len(args.conditions) != 1:
        raise ValueError(
            "formal OEA/Omni caches must contain exactly one acoustic condition"
        )
    args.main_config = args.main_config.resolve()
    args.resolved_model_config = args.resolved_model_config.resolve()
    args.model_root = args.model_root.resolve()
    args.corpus = args.corpus.resolve()
    args.audio_manifest = args.audio_manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    formal_execution_guard(device="cuda:0", dry_run=args.dry_run)

    main_config = load_main_experiment_config(args.main_config)
    resolved = json.loads(args.resolved_model_config.read_text(encoding="utf-8"))
    current_commit = git_output("rev-parse", "HEAD")
    if resolved.get("resolution_git_commit") != current_commit:
        raise RuntimeError("resolved model config was not created at the current commit")
    binding = model_binding(args.mode, resolved)
    documents, audio = selected_inputs(
        corpus_path=args.corpus,
        audio_manifest=args.audio_manifest,
        subset=args.subset,
        conditions=args.conditions,
    )
    expected = main_config["datasets"]["fiqa"]
    if len(documents) != expected["documents_expected"]:
        raise ValueError("FiQA document count differs from locked main config")
    expected_audio = expected["test_queries_expected"]
    if len(audio) != expected_audio:
        raise ValueError("SQuTR-FiQA audio count differs from locked main config")
    if {
        value.condition for value in audio
    } != set(args.conditions):
        raise ValueError("one or more requested audio conditions are absent")

    if args.mode == "oea":
        embedding_dimension = int(resolved["model_config"]["projection_dim"]["value"])
        model_name = resolved["checkpoint"]["repo_id"]
        model_revision = resolved["checkpoint"]["revision"]
        model_checkpoint = resolved["checkpoint"]["local_subpath"]
        pooling = "attention_mask_mean_then_modality_projection"
    else:
        base_dir, verified_files, embedding_dimension, dimension_source = (
            verify_locked_base_files(resolved, args.model_root)
        )
        model_name = resolved["base_model"]["repo_id"]
        model_revision = resolved["base_model"]["revision"]
        model_checkpoint = None
        pooling = "attention_mask_mean_no_projection"

    identity = {
        "schema_version": 1,
        "mode": args.mode,
        "git_commit": current_commit,
        "main_config": file_record(args.main_config),
        "resolved_model_config": file_record(args.resolved_model_config),
        "model_binding": binding,
        "model_root": str(args.model_root),
        "corpus": file_record(args.corpus),
        "audio_manifest": file_record(args.audio_manifest),
        "subset": args.subset,
        "conditions": list(args.conditions),
        "document_count": len(documents),
        "audio_count": len(audio),
        "embedding_dimension": embedding_dimension,
        "text_batch_size": args.text_batch_size,
        "audio_batch_size": args.audio_batch_size,
        "content": args.content,
        "audio_protocol": "audio_only_no_text_prefix",
        "text_protocol": "query_prefix",
        "normalization": "l2",
        "normalization_runtime": (
            "adapter_output_then_float32_cache_boundary_l2"
            if args.mode == "original_omni"
            else "oea_projection_float32_l2"
        ),
    }
    ensure_identity(args.output_dir, identity)
    include_documents = args.content == "both"
    document_ids_path = args.output_dir / "document_ids.jsonl"
    audio_ids_path = args.output_dir / "audio_ids.jsonl"
    if include_documents:
        write_text_once_or_verify(
            document_ids_path,
            "".join(
                json.dumps({"index": index, "id": value.document_id}, sort_keys=True)
                + "\n"
                for index, value in enumerate(documents)
            ),
        )
    write_text_once_or_verify(
        audio_ids_path,
        "".join(
            json.dumps(
                {
                    "index": index,
                    "id": value.query_id,
                    "record_id": value.record_id,
                    "source_query_id": value.query_id,
                    "condition": value.condition,
                },
                sort_keys=True,
            )
            + "\n"
            for index, value in enumerate(audio)
        ),
    )
    if args.dry_run:
        print(json.dumps({**identity, "status": "dry_run_complete"}, indent=2))
        return 0

    document_chunks = args.output_dir / "document_chunks"
    audio_chunks = args.output_dir / "audio_chunks"
    document_chunks.mkdir(exist_ok=True)
    audio_chunks.mkdir(exist_ok=True)
    pending_documents = (
        [
            (start, stop)
            for start, stop in ranges(len(documents), args.text_batch_size)
            if load_embedding_chunk(
                document_chunks,
                start,
                stop,
                embedding_dimension,
            )
            is None
        ]
        if include_documents
        else []
    )
    pending_audio = [
        (start, stop)
        for start, stop in ranges(len(audio), args.audio_batch_size)
        if load_embedding_chunk(
            audio_chunks,
            start,
            stop,
            embedding_dimension,
        )
        is None
    ]
    metrics_path = args.output_dir / "generation_metrics.json"
    report = {
        "schema_version": 1,
        "status": "running",
        "metrics_path": str(metrics_path),
        "mode": args.mode,
        "git_commit": current_commit,
        "pending_document_chunks": len(pending_documents),
        "pending_audio_chunks": len(pending_audio),
    }
    if args.mode == "original_omni":
        report["verified_model_files"] = verified_files
        report["embedding_dimension_source"] = dimension_source
        report["normalization_audit"] = {
            "method": "adapter_output_then_float32_cache_boundary_l2",
            "reason": (
                "BF16 adapter normalization may exceed the former "
                "1e-3 float32 norm assertion"
            ),
            "scope": (
                "rows encoded in this process; verified pre-existing chunks "
                "are already post-normalized"
            ),
            "batch_count": 0,
            "row_count": 0,
            "rows_outside_pre_atol_1e3": 0,
            "cached_document_rows_at_start": (
                len(documents)
                - sum(stop - start for start, stop in pending_documents)
            ),
            "cached_audio_rows_at_start": (
                len(audio) - sum(stop - start for start, stop in pending_audio)
            ),
        }
    atomic_write_json(metrics_path, report)

    random.seed(42)
    np.random.seed(42)
    torch = None
    if pending_documents or pending_audio:
        if args.mode == "oea":
            torch, adapter, model, audio_head, text_head, device = load_oea_bundle(
                resolved,
                args.model_root,
                report,
            )
        else:
            torch, adapter, device = load_vanilla_bundle(
                resolved,
                base_dir,
                embedding_dimension,
                report,
            )
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats(device)

    for start, stop in pending_documents:
        texts = [value.constructed_text for value in documents[start:stop]]
        if args.mode == "oea":
            values = project_batch(
                torch,
                adapter,
                model,
                text_head,
                device,
                texts=texts,
            )
        else:
            values = encode_base_batch(
                adapter,
                embedding_dimension,
                texts=texts,
                normalization_audit=report["normalization_audit"],
            )
        save_embedding_chunk(document_chunks, start, stop, values)
        report["pending_document_chunks"] -= 1
        atomic_write_json(metrics_path, report)
        print(f"[PROGRESS] documents {stop}/{len(documents)}", flush=True)

    for start, stop in pending_audio:
        paths = [Path(value.audio_path) for value in audio[start:stop]]
        if args.mode == "oea":
            values = project_batch(
                torch,
                adapter,
                model,
                audio_head,
                device,
                audio_paths=paths,
            )
        else:
            values = encode_base_batch(
                adapter,
                embedding_dimension,
                audio_paths=paths,
                normalization_audit=report["normalization_audit"],
            )
        save_embedding_chunk(audio_chunks, start, stop, values)
        report["pending_audio_chunks"] -= 1
        atomic_write_json(metrics_path, report)
        print(f"[PROGRESS] audio {stop}/{len(audio)}", flush=True)

    document_embeddings = args.output_dir / "document_embeddings.npy"
    audio_embeddings = args.output_dir / "audio_embeddings.npy"
    if include_documents:
        consolidate_embedding_chunks(
            document_chunks,
            total=len(documents),
            batch_size=args.text_batch_size,
            dimension=embedding_dimension,
            destination=document_embeddings,
        )
    consolidate_embedding_chunks(
        audio_chunks,
        total=len(audio),
        batch_size=args.audio_batch_size,
        dimension=embedding_dimension,
        destination=audio_embeddings,
    )
    if torch is not None:
        report["gpu_name"] = torch.cuda.get_device_name(device)
        report["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        report["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    report["status"] = "complete"
    report["outputs"] = {
        "audio_embeddings": file_record(audio_embeddings),
        "audio_ids": file_record(audio_ids_path),
    }
    if include_documents:
        report["outputs"].update(
            {
                "document_embeddings": file_record(document_embeddings),
                "document_ids": file_record(document_ids_path),
            }
        )
    atomic_write_json(metrics_path, report)
    manifest = build_cache_manifest(
        artifact_type=f"{args.mode}_fiqa_embeddings",
        dataset="SQuTR-FiQA",
        split="test_four_conditions",
        inputs=[
            file_record(args.main_config),
            file_record(args.resolved_model_config),
            file_record(args.corpus),
            file_record(args.audio_manifest),
        ],
        model_name=model_name,
        model_revision=model_revision,
        model_checkpoint=model_checkpoint,
        tokenizer={"resolved_model_config": str(args.resolved_model_config)},
        pooling=pooling,
        embedding_dim=embedding_dimension,
        max_length=512,
        dtype="bfloat16",
        normalization="l2",
        seed=42,
        command=sys.argv,
        git_commit=current_commit,
        outputs=[
            *(
                [
                    file_record(document_ids_path),
                    file_record(document_embeddings),
                ]
                if include_documents
                else []
            ),
            file_record(audio_ids_path),
            file_record(audio_embeddings),
            file_record(metrics_path),
        ],
        extra_identity={
            "mode": args.mode,
            "subset": args.subset,
            "conditions": list(args.conditions),
            "content": args.content,
            "model_binding": binding,
            "audio_protocol": "audio_only_no_text_prefix",
            "text_protocol": "query_prefix",
            "normalization_runtime": identity["normalization_runtime"],
        },
    )
    write_cache_manifest_once(args.output_dir / "cache_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
