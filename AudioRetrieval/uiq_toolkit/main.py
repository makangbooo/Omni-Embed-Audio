"""
CLI entry point for the standalone UIQ toolkit.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from uiq_toolkit.generators import create_uiq_generator, resolve_backend
from uiq_toolkit.io_utils import (
    load_examples,
    load_hard_negative_caption_map,
    save_flat_results,
    save_grouped_results,
    write_metadata,
)
from uiq_toolkit.query_types import QueryRecord, QueryType


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Standalone UIQ generation toolkit extracted from the AudioRetrieval repo.",
    )
    parser.add_argument(
        "--backend",
        default="gpt",
        choices=["gpt", "llama", "template"],
        help="Generation backend",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["clotho", "audiocaps", "mecat"],
        help="Dataset type",
    )
    parser.add_argument(
        "--captions-csv",
        type=Path,
        required=True,
        help="Input captions CSV (or metadata directory for MeCAT)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save generated UIQ files",
    )
    parser.add_argument(
        "--query-types",
        nargs="+",
        default=[
            QueryType.QUESTION.value,
            QueryType.IMPERATIVE.value,
            QueryType.PARAPHRASE.value,
            QueryType.TAGGING.value,
        ],
        choices=QueryType.choices(),
        help="Query types to generate",
    )
    parser.add_argument(
        "--output-format",
        default="both",
        choices=["flat", "grouped", "both"],
        help="Save per-query flat JSONL, grouped JSONL, or both",
    )
    parser.add_argument(
        "--hard-neg-jsonl",
        type=Path,
        help="Filtered hard negatives JSONL for negative query generation",
    )
    parser.add_argument(
        "--caption-index",
        type=int,
        default=1,
        help="Caption column to use for Clotho (1-5). Ignored for AudioCaps.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="OpenAI model name or local Hugging Face model name, depending on backend",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key. If omitted, OPENAI_API_KEY is used.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Inference device for the llama backend",
    )
    parser.add_argument(
        "--torch-dtype",
        default="float16",
        help="Torch dtype for the llama backend",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Generation batch size setting",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of generated tokens",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent API call threads (default: 1, sequential)",
    )
    return parser


def _default_model_for_backend(backend: str) -> str | None:
    """Return a sensible default model name for a backend."""
    if backend == "gpt":
        return "gpt-4"
    if backend == "llama":
        return "meta-llama/Llama-2-7b-chat-hf"
    return None


def build_generator(args: argparse.Namespace):
    """Instantiate the selected backend."""
    backend = resolve_backend(args.backend)
    common_kwargs = {
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    if backend == "gpt":
        return create_uiq_generator(
            backend=backend,
            model=args.model or _default_model_for_backend(backend),
            api_key=args.api_key,
            **common_kwargs,
        )
    if backend == "llama":
        return create_uiq_generator(
            backend=backend,
            model_name=args.model or _default_model_for_backend(backend),
            device=args.device,
            torch_dtype=args.torch_dtype,
            **common_kwargs,
        )
    return create_uiq_generator(backend=backend, **common_kwargs)


def main(argv: List[str] | None = None) -> int:
    """Run the standalone UIQ generation CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    examples = load_examples(args.dataset, args.captions_csv, caption_index=args.caption_index)
    if not examples:
        parser.error(f"No examples were loaded from {args.captions_csv}.")

    query_types = [QueryType.from_string(value) for value in args.query_types]
    hard_negative_map: Dict[str, str] | None = None
    if QueryType.NEGATIVE in query_types:
        if not args.hard_neg_jsonl:
            parser.error("--hard-neg-jsonl is required when generating negative queries.")
        hard_negative_map = load_hard_negative_caption_map(args.hard_neg_jsonl)
        if not hard_negative_map:
            parser.error(f"No hard-negative captions were found in {args.hard_neg_jsonl}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generator = build_generator(args)

    all_records: list[QueryRecord] = []
    file_manifest: Dict[str, str] = {}
    query_counts: Dict[str, Dict[str, int]] = {}

    for query_type in query_types:
        print(f"[INFO] Generating {query_type.value} queries for {len(examples)} examples...")
        records = generator.generate(
            examples=examples,
            query_type=query_type,
            hard_negative_captions=hard_negative_map,
            show_progress=not args.no_progress,
            max_workers=args.workers,
        )
        all_records.extend(records)

        non_empty_count = sum(1 for record in records if record.query.strip())
        query_counts[query_type.value] = {
            "total_records": len(records),
            "non_empty_queries": non_empty_count,
        }

        if args.output_format in {"flat", "both"}:
            flat_path = args.output_dir / f"uiq_{query_type.value}.jsonl"
            save_flat_results(records, flat_path)
            file_manifest[f"flat_{query_type.value}"] = str(flat_path)
            print(f"[INFO] Saved flat JSONL to {flat_path}")

    if args.output_format in {"grouped", "both"}:
        grouped_path = args.output_dir / "uiq_grouped.jsonl"
        save_grouped_results(all_records, grouped_path)
        file_manifest["grouped"] = str(grouped_path)
        print(f"[INFO] Saved grouped JSONL to {grouped_path}")

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "dataset": args.dataset,
        "captions_csv": str(args.captions_csv),
        "caption_index": args.caption_index if args.dataset == "clotho" else None,
        "query_types": [query_type.value for query_type in query_types],
        "output_format": args.output_format,
        "model": args.model or _default_model_for_backend(resolve_backend(args.backend)),
        "num_examples": len(examples),
        "counts": query_counts,
        "files": file_manifest,
    }
    metadata_path = args.output_dir / "metadata.json"
    write_metadata(metadata_path, metadata)
    print(f"[INFO] Saved metadata to {metadata_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
