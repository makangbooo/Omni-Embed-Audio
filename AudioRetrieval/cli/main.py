"""
Unified CLI for AudioRetrieval.

Provides subcommands for:
- preprocess: Embedding precomputation and hard negative mining
- train: Model training
- generate-uiq: User intent query generation
- evaluate: Retrieval evaluation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


def add_preprocess_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Add preprocessing subcommands."""
    preprocess = subparsers.add_parser(
        "preprocess",
        help="Preprocessing: embeddings and hard negatives",
    )
    preprocess_sub = preprocess.add_subparsers(dest="preprocess_cmd", help="Preprocessing commands")

    # Embeddings subcommand
    embeddings = preprocess_sub.add_parser(
        "embeddings",
        help="Precompute embeddings for a dataset",
    )
    embeddings.add_argument("--model", required=True, choices=["laion_clap", "robust_clap", "mga_clap", "m2d_clap", "oea"],
                           help="Model to use for embeddings")
    embeddings.add_argument("--audio-dir", type=Path, required=True, help="Audio directory")
    embeddings.add_argument("--captions-csv", type=Path, required=True, help="Captions CSV file")
    embeddings.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    embeddings.add_argument("--dataset", choices=["clotho", "audiocaps"], default="clotho",
                           help="Dataset type")
    embeddings.add_argument("--uiq-jsonl", type=Path, help="Optional UIQ JSONL for text embeddings")
    embeddings.add_argument("--device", default="cuda", help="Device for inference")
    embeddings.add_argument("--batch-size-audio", type=int, default=64)
    embeddings.add_argument("--batch-size-text", type=int, default=256)

    embeddings.add_argument("--laion-ckpt", type=Path)
    embeddings.add_argument("--laion-bert-tokenizer", type=Path)
    embeddings.add_argument("--laion-roberta-tokenizer", type=Path)
    embeddings.add_argument("--laion-bart-tokenizer", type=Path)
    embeddings.add_argument("--robust-ckpt", type=Path)
    embeddings.add_argument("--robust-repo", type=Path)
    embeddings.add_argument("--robust-bert-tokenizer", type=Path)
    embeddings.add_argument("--robust-roberta-tokenizer", type=Path)
    embeddings.add_argument("--robust-bart-tokenizer", type=Path)
    embeddings.add_argument("--robust-bpe-vocab", type=Path)

    # OEA-specific options
    embeddings.add_argument("--checkpoint", type=Path, help="OEA checkpoint path")
    embeddings.add_argument("--repo-id", default="nvidia/omni-embed-nemotron-3b",
                           help="OEA HuggingFace repo ID")
    embeddings.add_argument("--local-path", type=Path,
                           help="Already-downloaded OEA base-model directory")
    embeddings.add_argument("--cache-dir", type=Path,
                           help="Optional Hugging Face cache directory")

    uiq_embeddings = preprocess_sub.add_parser(
        "uiq-embeddings",
        help="Precompute released UIQ text embeddings",
    )
    uiq_embeddings.add_argument(
        "--model",
        choices=["oea", "laion_clap", "robust_clap", "mga_clap", "m2d_clap"],
        default="oea",
    )
    uiq_embeddings.add_argument(
        "--uiq-jsonl",
        type=Path,
        action="append",
        required=True,
        help="Released UIQ JSONL; repeat for multiple query types",
    )
    uiq_embeddings.add_argument("--output-dir", type=Path, required=True)
    uiq_embeddings.add_argument("--dataset", default="unknown")
    uiq_embeddings.add_argument("--device", default="cuda")
    uiq_embeddings.add_argument("--batch-size-text", type=int, default=16)
    uiq_embeddings.add_argument("--checkpoint", type=Path)
    uiq_embeddings.add_argument(
        "--repo-id",
        default="nvidia/omni-embed-nemotron-3b",
    )
    uiq_embeddings.add_argument("--local-path", type=Path)
    uiq_embeddings.add_argument("--laion-ckpt", type=Path)
    uiq_embeddings.add_argument("--laion-bert-tokenizer", type=Path)
    uiq_embeddings.add_argument("--laion-roberta-tokenizer", type=Path)
    uiq_embeddings.add_argument("--laion-bart-tokenizer", type=Path)
    uiq_embeddings.add_argument("--robust-ckpt", type=Path)
    uiq_embeddings.add_argument("--robust-repo", type=Path)
    uiq_embeddings.add_argument("--robust-bert-tokenizer", type=Path)
    uiq_embeddings.add_argument("--robust-roberta-tokenizer", type=Path)
    uiq_embeddings.add_argument("--robust-bart-tokenizer", type=Path)
    uiq_embeddings.add_argument("--robust-bpe-vocab", type=Path)
    uiq_embeddings.add_argument("--mga-repo", type=Path)
    uiq_embeddings.add_argument("--mga-ckpt", type=Path)
    uiq_embeddings.add_argument("--mga-bert-tokenizer", type=Path)
    uiq_embeddings.add_argument("--mga-checkpoint-sha256")
    uiq_embeddings.add_argument("--m2d-ckpt", type=Path)
    uiq_embeddings.add_argument("--m2d-bert-tokenizer", type=Path)

    # MGA-CLAP specific options
    embeddings.add_argument("--mga-repo", type=Path, help="MGA-CLAP repository path")
    embeddings.add_argument("--mga-ckpt", type=Path, help="MGA-CLAP checkpoint path")
    embeddings.add_argument("--mga-bert-tokenizer", type=Path)
    embeddings.add_argument("--mga-checkpoint-sha256")
    embeddings.add_argument("--m2d-ckpt", type=Path, help="M2D-CLAP checkpoint path")
    embeddings.add_argument(
        "--m2d-bert-tokenizer",
        type=Path,
        help="Pinned local BERT-base tokenizer directory",
    )

    # Hard negatives subcommand
    hard_neg = preprocess_sub.add_parser(
        "hard-negatives",
        help="Mine hard negatives for training",
    )
    hard_neg.add_argument("--metadata-csv", type=Path, required=True, help="Dataset metadata CSV")
    hard_neg.add_argument("--audio-dir", type=Path, required=True, help="Audio directory")
    hard_neg.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    hard_neg.add_argument("--dataset", choices=["clotho", "audiocaps"], default="clotho")
    hard_neg.add_argument("--mga-repo", type=Path, required=True, help="MGA-CLAP repository")
    hard_neg.add_argument("--mga-ckpt", type=Path, required=True, help="MGA-CLAP checkpoint")
    hard_neg.add_argument("--topk", type=int, default=50, help="Number of neighbors per audio")
    hard_neg.add_argument("--semantic-threshold", type=float, default=0.7,
                         help="Semantic similarity threshold for filtering")
    hard_neg.add_argument("--device", default="cuda")
    hard_neg.add_argument("--skip-stage1", action="store_true",
                         help="Skip acoustic mining (use existing stage1 output)")
    hard_neg.add_argument("--stage1-output", type=Path,
                         help="Path to existing stage1 output for skip-stage1")


def add_train_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add training subcommand."""
    train = subparsers.add_parser(
        "train",
        help="Train audio-text retrieval models",
    )
    train.add_argument("--config", type=Path, help="Training config file (Hydra YAML)")
    train.add_argument("--dataset", choices=["clotho", "audiocaps"], default="audiocaps")
    train.add_argument("--train-csv", type=Path, help="Training data CSV")
    train.add_argument("--val-csv", type=Path, help="Validation data CSV")
    train.add_argument("--audio-dir", type=Path, help="Audio directory")
    train.add_argument("--output-dir", type=Path, help="Output directory for checkpoints")
    train.add_argument("--hard-neg-json", type=Path, help="Hard negatives JSONL")
    train.add_argument("--init-checkpoint", type=Path, help="Initial checkpoint for fine-tuning")
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--grad-accum", type=int, default=512)
    train.add_argument("--device", default="cuda")
    train.add_argument("--wandb", action="store_true", help="Enable W&B logging")


def add_uiq_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add UIQ generation subcommand."""
    uiq = subparsers.add_parser(
        "generate-uiq",
        help="Generate user intent queries",
    )
    uiq.add_argument("--backend", choices=["gpt", "llama"], default="gpt",
                    help="LLM backend for generation")
    uiq.add_argument("--dataset", choices=["clotho", "audiocaps"], required=True)
    uiq.add_argument("--captions-csv", type=Path, required=True, help="Captions CSV file")
    uiq.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    uiq.add_argument("--query-types", nargs="+",
                    choices=["question", "imperative", "paraphrase", "negative", "tagging"],
                    default=["question", "imperative", "paraphrase", "tagging"],
                    help="Query types to generate")
    uiq.add_argument("--hard-neg-jsonl", type=Path,
                    help="Hard negatives JSONL (required for negative queries)")
    uiq.add_argument("--model", default="gpt-4", help="Model name for generation")
    uiq.add_argument("--batch-size", type=int, default=10)
    uiq.add_argument("--temperature", type=float, default=0.7)


def add_evaluate_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add evaluation subcommand."""
    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate retrieval performance",
    )
    evaluate.add_argument("--mode", choices=["baseline", "uiq", "negative", "precomputed"],
                         default="baseline", help="Evaluation mode")
    evaluate.add_argument("--model", default="laion_clap",
                         choices=["laion_clap", "mga_clap", "oea", "wavcaps"],
                         help="Model to use")
    evaluate.add_argument("--dataset", choices=["clotho", "audiocaps"], default="clotho")
    evaluate.add_argument("--audio-dir", type=Path, help="Audio directory")
    evaluate.add_argument("--captions-csv", type=Path, help="Captions CSV file")
    evaluate.add_argument("--uiq-jsonl", type=Path, help="UIQ JSONL file (for uiq mode)")
    evaluate.add_argument("--negative-jsonl", type=Path,
                         help="Negative queries JSONL (for negative mode)")
    evaluate.add_argument("--embeddings-dir", type=Path,
                         help="Precomputed embeddings directory (for precomputed mode)")
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--batch-size-audio", type=int, default=64)
    evaluate.add_argument("--batch-size-text", type=int, default=256)
    evaluate.add_argument("--output", type=Path, help="Output file for results (JSON)")

    # Model-specific options
    evaluate.add_argument("--checkpoint", type=Path, help="Model checkpoint (for OEA)")


def run_preprocess(args: argparse.Namespace) -> int:
    """Run preprocessing command."""
    if args.preprocess_cmd == "embeddings":
        if args.model == "laion_clap":
            from AudioRetrieval.preprocessing.embeddings import LaionClapEmbeddingPrecomputer
            if not all((
                args.laion_ckpt,
                args.laion_bert_tokenizer,
                args.laion_roberta_tokenizer,
                args.laion_bart_tokenizer,
            )):
                print("Error: LAION-CLAP requires checkpoint and three local tokenizers")
                return 1
            precomputer = LaionClapEmbeddingPrecomputer(
                ckpt_path=str(args.laion_ckpt),
                device=args.device,
                batch_size_audio=args.batch_size_audio,
                batch_size_text=args.batch_size_text,
                bert_tokenizer_path=str(args.laion_bert_tokenizer),
                roberta_tokenizer_path=str(args.laion_roberta_tokenizer),
                bart_tokenizer_path=str(args.laion_bart_tokenizer),
            )
        elif args.model == "mga_clap":
            from AudioRetrieval.preprocessing.embeddings import MGAClapEmbeddingPrecomputer
            if not all((
                args.mga_repo,
                args.mga_ckpt,
                args.mga_bert_tokenizer,
                args.mga_checkpoint_sha256,
            )):
                print("Error: MGA-CLAP requires source, checkpoint, tokenizer, and SHA256")
                return 1
            precomputer = MGAClapEmbeddingPrecomputer(
                repo_path=str(args.mga_repo),
                ckpt_path=str(args.mga_ckpt),
                bert_tokenizer_path=str(args.mga_bert_tokenizer),
                checkpoint_sha256=args.mga_checkpoint_sha256,
                device=args.device,
                batch_size_audio=args.batch_size_audio,
                batch_size_text=args.batch_size_text,
            )
        elif args.model == "robust_clap":
            from AudioRetrieval.preprocessing.embeddings import (
                RobustClapEmbeddingPrecomputer,
            )
            if not all((
                args.robust_ckpt,
                args.robust_repo,
                args.robust_bert_tokenizer,
                args.robust_roberta_tokenizer,
                args.robust_bart_tokenizer,
                args.robust_bpe_vocab,
            )):
                print(
                    "Error: Robust-CLAP requires source, checkpoint, and local "
                    "BERT, RoBERTa, and BART tokenizers plus BPE vocabulary"
                )
                return 1
            precomputer = RobustClapEmbeddingPrecomputer(
                ckpt_path=str(args.robust_ckpt),
                repo_root=str(args.robust_repo),
                bert_tokenizer_path=str(args.robust_bert_tokenizer),
                roberta_tokenizer_path=str(args.robust_roberta_tokenizer),
                bart_tokenizer_path=str(args.robust_bart_tokenizer),
                bpe_vocab_path=str(args.robust_bpe_vocab),
                enable_fusion=False,
                device=args.device,
                batch_size_audio=args.batch_size_audio,
                batch_size_text=args.batch_size_text,
            )
        elif args.model == "m2d_clap":
            from AudioRetrieval.preprocessing.embeddings import M2DClapEmbeddingPrecomputer
            if not args.m2d_ckpt or not args.m2d_bert_tokenizer:
                print("Error: M2D-CLAP requires checkpoint and local BERT tokenizer")
                return 1
            precomputer = M2DClapEmbeddingPrecomputer(
                checkpoint=str(args.m2d_ckpt),
                bert_tokenizer_path=str(args.m2d_bert_tokenizer),
                device=args.device,
                batch_size_audio=args.batch_size_audio,
                batch_size_text=args.batch_size_text,
            )
        elif args.model == "oea":
            from AudioRetrieval.preprocessing.embeddings import OEAEmbeddingPrecomputer
            if not args.checkpoint:
                print("Error: --checkpoint required for OEA model")
                return 1
            precomputer = OEAEmbeddingPrecomputer(
                checkpoint=str(args.checkpoint),
                repo_id=args.repo_id,
                local_path=str(args.local_path) if args.local_path else None,
                cache_dir=str(args.cache_dir) if args.cache_dir else None,
                device=args.device,
                batch_size_audio=args.batch_size_audio,
                batch_size_text=args.batch_size_text,
            )
        else:
            print(f"Unknown model: {args.model}")
            return 1

        precomputer.precompute_full_dataset(
            audio_dir=args.audio_dir,
            captions_csv=args.captions_csv,
            uiq_jsonl=args.uiq_jsonl,
            output_dir=args.output_dir,
            dataset_type=args.dataset,
        )
        return 0

    elif args.preprocess_cmd == "uiq-embeddings":
        import json

        from AudioRetrieval.preprocessing.embeddings import UIQTextEmbeddingPrecomputer

        if args.model == "oea" and not args.checkpoint:
            print("Error: OEA UIQ embeddings require --checkpoint")
            return 1
        if args.model == "laion_clap" and not all(
            (
                args.laion_ckpt,
                args.laion_bert_tokenizer,
                args.laion_roberta_tokenizer,
                args.laion_bart_tokenizer,
            )
        ):
            print(
                "Error: LAION-CLAP UIQ embeddings require checkpoint and "
                "local BERT, RoBERTa, and BART tokenizers"
            )
            return 1
        if args.model == "mga_clap" and not all(
            (
                args.mga_repo,
                args.mga_ckpt,
                args.mga_bert_tokenizer,
                args.mga_checkpoint_sha256,
            )
        ):
            print(
                "Error: MGA-CLAP UIQ embeddings require source, checkpoint, "
                "tokenizer, and SHA256"
            )
            return 1
        if args.model == "m2d_clap" and not all(
            (args.m2d_ckpt, args.m2d_bert_tokenizer)
        ):
            print(
                "Error: M2D-CLAP UIQ embeddings require checkpoint and tokenizer"
            )
            return 1
        if args.model == "robust_clap" and not all(
            (
                args.robust_ckpt,
                args.robust_repo,
                args.robust_bert_tokenizer,
                args.robust_roberta_tokenizer,
                args.robust_bart_tokenizer,
                args.robust_bpe_vocab,
            )
        ):
            print(
                "Error: Robust-CLAP UIQ embeddings require source, checkpoint, "
                "and local BERT, RoBERTa, and BART tokenizers plus BPE vocabulary"
            )
            return 1

        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            print(f"Error: UIQ output directory is not empty: {args.output_dir}")
            return 2

        encoder = UIQTextEmbeddingPrecomputer(
            model=args.model,
            device=args.device,
            batch_size_text=args.batch_size_text,
            oea_checkpoint=str(args.checkpoint),
            oea_repo_id=args.repo_id,
            oea_local_path=str(args.local_path) if args.local_path else None,
            laion_ckpt=(
                str(args.laion_ckpt) if args.laion_ckpt else None
            ),
            laion_bert_tokenizer=(
                str(args.laion_bert_tokenizer)
                if args.laion_bert_tokenizer
                else None
            ),
            laion_roberta_tokenizer=(
                str(args.laion_roberta_tokenizer)
                if args.laion_roberta_tokenizer
                else None
            ),
            laion_bart_tokenizer=(
                str(args.laion_bart_tokenizer)
                if args.laion_bart_tokenizer
                else None
            ),
            mga_repo=str(args.mga_repo) if args.mga_repo else None,
            mga_ckpt=str(args.mga_ckpt) if args.mga_ckpt else None,
            mga_bert_tokenizer=(
                str(args.mga_bert_tokenizer) if args.mga_bert_tokenizer else None
            ),
            mga_checkpoint_sha256=args.mga_checkpoint_sha256,
            m2d_ckpt=str(args.m2d_ckpt) if args.m2d_ckpt else None,
            m2d_bert_tokenizer=(
                str(args.m2d_bert_tokenizer)
                if args.m2d_bert_tokenizer
                else None
            ),
            robust_ckpt=str(args.robust_ckpt) if args.robust_ckpt else None,
            robust_repo=str(args.robust_repo) if args.robust_repo else None,
            robust_bert_tokenizer=(
                str(args.robust_bert_tokenizer)
                if args.robust_bert_tokenizer
                else None
            ),
            robust_roberta_tokenizer=(
                str(args.robust_roberta_tokenizer)
                if args.robust_roberta_tokenizer
                else None
            ),
            robust_bart_tokenizer=(
                str(args.robust_bart_tokenizer)
                if args.robust_bart_tokenizer
                else None
            ),
            robust_bpe_vocab=(
                str(args.robust_bpe_vocab) if args.robust_bpe_vocab else None
            ),
            robust_disable_fusion=True,
        )

        summaries = {}
        for uiq_path in args.uiq_jsonl:
            result = encoder.precompute(
                uiq_jsonl=uiq_path,
                output_dir=args.output_dir,
                dataset=args.dataset,
            )
            summaries[str(uiq_path)] = result

        summary_path = args.output_dir / "uiq_run_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "model": args.model,
                    "inputs": [str(path) for path in args.uiq_jsonl],
                    "summaries": summaries,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"[INFO] UIQ run summary written to {summary_path}")
        return 0

    elif args.preprocess_cmd == "hard-negatives":
        from AudioRetrieval.preprocessing.hard_negatives import HardNegativePipeline

        pipeline = HardNegativePipeline(
            mga_repo=str(args.mga_repo),
            mga_ckpt=str(args.mga_ckpt),
            topk=args.topk,
            semantic_threshold=args.semantic_threshold,
            device=args.device,
        )

        pipeline.run(
            metadata_csv=args.metadata_csv,
            audio_dir=args.audio_dir,
            output_dir=args.output_dir,
            dataset_type=args.dataset,
            skip_stage1=args.skip_stage1,
            stage1_output=args.stage1_output,
        )
        return 0

    else:
        print("Error: Please specify a preprocessing subcommand (embeddings, hard-negatives)")
        return 1


def run_train(args: argparse.Namespace) -> int:
    """Run training command."""
    print("Training command - delegating to training module...")
    print("For full training options, use the training scripts directly:")
    print("  python -m AudioRetrieval.training.oea.train_omniembed_lora --help")
    return 0


def run_uiq(args: argparse.Namespace) -> int:
    """Run UIQ generation command."""
    from AudioRetrieval.uiq_generation import UIQGenerator, QueryType
    from AudioRetrieval.uiq_generation.generators.base import BaseUIQGenerator

    # Load captions
    if args.dataset == "clotho":
        clip_ids, captions = BaseUIQGenerator.load_clotho_captions(args.captions_csv)
    else:
        clip_ids, captions = BaseUIQGenerator.load_audiocaps_captions(args.captions_csv)

    print(f"Loaded {len(captions)} captions from {args.dataset}")

    # Create generator
    generator = UIQGenerator(
        backend=args.backend,
        model=args.model if args.backend == "gpt" else None,
        model_name=args.model if args.backend == "llama" else None,
        batch_size=args.batch_size,
        temperature=args.temperature,
    )

    # Generate for each query type
    query_types = [QueryType.from_string(qt) for qt in args.query_types]

    # Load hard negatives if needed
    hard_neg_captions = None
    if QueryType.NEGATIVE in query_types:
        if not args.hard_neg_jsonl:
            print("Warning: --hard-neg-jsonl required for negative queries, skipping")
            query_types.remove(QueryType.NEGATIVE)
        else:
            # TODO: Load hard negatives mapping
            print("Note: Negative query generation requires hard negatives mapping")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for qt in query_types:
        print(f"\nGenerating {qt.value} queries...")
        results = generator.generate(
            captions=captions,
            query_type=qt,
            clip_ids=clip_ids,
        )

        output_path = args.output_dir / f"uiq_{qt.value}.jsonl"
        generator.save_results(results, output_path)

    print(f"\nUIQ generation complete. Results saved to {args.output_dir}")
    return 0


def run_evaluate(args: argparse.Namespace) -> int:
    """Run evaluation command."""
    import json

    if args.mode == "baseline":
        from AudioRetrieval.evaluation.runners import BaselineRunner

        runner = BaselineRunner(
            model_name=args.model,
            device=args.device,
            batch_size_audio=args.batch_size_audio,
            batch_size_text=args.batch_size_text,
        )

        results = runner.run(
            audio_dir=args.audio_dir,
            captions_csv=args.captions_csv,
            dataset_type=args.dataset,
        )

    elif args.mode == "uiq":
        from AudioRetrieval.evaluation.runners import UIQRunner

        if not args.uiq_jsonl:
            print("Error: --uiq-jsonl required for uiq mode")
            return 1

        runner = UIQRunner(
            model_name=args.model,
            device=args.device,
            batch_size_audio=args.batch_size_audio,
            batch_size_text=args.batch_size_text,
        )

        results = runner.run(
            audio_dir=args.audio_dir,
            captions_csv=args.captions_csv,
            uiq_jsonl=args.uiq_jsonl,
            dataset_type=args.dataset,
        )

    elif args.mode == "negative":
        from AudioRetrieval.evaluation.runners import NegativeQueryRunner

        if not args.negative_jsonl:
            print("Error: --negative-jsonl required for negative mode")
            return 1

        runner = NegativeQueryRunner(
            model_name=args.model,
            device=args.device,
            batch_size_audio=args.batch_size_audio,
            batch_size_text=args.batch_size_text,
        )

        results = runner.run(
            audio_dir=args.audio_dir,
            captions_csv=args.captions_csv,
            negative_queries_jsonl=args.negative_jsonl,
            dataset_type=args.dataset,
        )

    elif args.mode == "precomputed":
        from AudioRetrieval.evaluation.runners import PrecomputedEmbeddingRunner

        if not args.embeddings_dir:
            print("Error: --embeddings-dir required for precomputed mode")
            return 1

        embeddings_dir = Path(args.embeddings_dir)
        runner = PrecomputedEmbeddingRunner(
            audio_embeddings_path=embeddings_dir / "audio_embeddings.npz",
            caption_embeddings_path=embeddings_dir / "caption_embeddings.npz",
            uiq_embeddings_dir=embeddings_dir,
        )

        results = runner.run_all()

    else:
        print(f"Unknown mode: {args.mode}")
        return 1

    # Save results if output specified
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return 0


def main(argv: Optional[list] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="AudioRetrieval",
        description="AudioRetrieval: Audio-Text Retrieval Framework",
    )
    parser.add_argument("--version", action="version", version="AudioRetrieval 1.0.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    add_preprocess_subparsers(subparsers)
    add_train_parser(subparsers)
    add_uiq_parser(subparsers)
    add_evaluate_parser(subparsers)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "preprocess":
        return run_preprocess(args)
    elif args.command == "train":
        return run_train(args)
    elif args.command == "generate-uiq":
        return run_uiq(args)
    elif args.command == "evaluate":
        return run_evaluate(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
