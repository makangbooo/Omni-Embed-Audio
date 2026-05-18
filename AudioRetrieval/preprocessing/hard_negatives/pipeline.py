"""
Combined Hard Negative Mining Pipeline.

Provides a unified interface for the two-stage hard negative mining process:
1. Acoustic mining using MGA-CLAP
2. Semantic filtering using BGE
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from AudioRetrieval.preprocessing.hard_negatives.acoustic_mining import AcousticNegativeMiner
from AudioRetrieval.preprocessing.hard_negatives.semantic_filtering import SemanticFilter


class HardNegativePipeline:
    """
    Combined two-stage hard negative mining pipeline.

    Stage 1: Acoustic mining - Find acoustically similar audio samples
    Stage 2: Semantic filtering - Remove semantically similar samples

    Args:
        mga_repo: Path to MGA-CLAP repository
        mga_ckpt: Path to MGA-CLAP checkpoint
        mga_seconds: Audio duration for MGA-CLAP
        semantic_model: Sentence transformer model for semantic filtering
        semantic_threshold: Maximum semantic similarity to keep
        topk: Number of acoustic neighbors to find
        max_negatives: Maximum negatives to keep after filtering
        device: Device for inference
        batch_size_audio: Batch size for audio encoding
        batch_size_text: Batch size for text encoding

    Example:
        >>> pipeline = HardNegativePipeline(
        ...     mga_repo="models/mga_clap",
        ...     mga_ckpt="checkpoints/mga-clap.pt",
        ...     device="cuda",
        ... )
        >>> results = pipeline.run(
        ...     metadata_csv="clotho/evaluation_meta.csv",
        ...     audio_dir="clotho/evaluation_audio",
        ...     output_dir="hard_negatives/clotho/",
        ...     dataset_type="clotho",
        ... )
    """

    def __init__(
        self,
        mga_repo: str,
        mga_ckpt: str,
        mga_seconds: float = 10.0,
        semantic_model: str = "BAAI/bge-large-en-v1.5",
        semantic_threshold: float = 0.7,
        topk: int = 50,
        max_negatives: int = 50,
        device: str = "cuda",
        batch_size_audio: int = 512,
        batch_size_text: int = 64,
    ):
        self.acoustic_miner = AcousticNegativeMiner(
            mga_repo=mga_repo,
            mga_ckpt=mga_ckpt,
            seconds=mga_seconds,
            device=device,
            batch_size=batch_size_audio,
            topk=topk,
        )

        self.semantic_filter = SemanticFilter(
            model_name=semantic_model,
            device=device,
            batch_size=batch_size_text,
            semantic_threshold=semantic_threshold,
            max_negatives=max_negatives,
        )

    def run(
        self,
        metadata_csv: Path,
        audio_dir: Path,
        output_dir: Path,
        dataset_type: str = "clotho",
        embedding_cache: Optional[Path] = None,
        overwrite_cache: bool = False,
        skip_stage1: bool = False,
        stage1_output: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run the full hard negative mining pipeline.

        Args:
            metadata_csv: Path to dataset metadata CSV
            audio_dir: Directory containing audio files
            output_dir: Output directory for results
            dataset_type: Dataset type (clotho/audiocaps)
            embedding_cache: Optional path to cache audio embeddings
            overwrite_cache: Overwrite existing embedding cache
            skip_stage1: Skip acoustic mining (use existing stage1_output)
            stage1_output: Path to existing Stage 1 output

        Returns:
            Dictionary with pipeline results and statistics
        """
        metadata_csv = Path(metadata_csv)
        audio_dir = Path(audio_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print("Hard Negative Mining Pipeline")
        print("=" * 80)
        print(f"Dataset type : {dataset_type}")
        print(f"Metadata CSV : {metadata_csv}")
        print(f"Audio dir    : {audio_dir}")
        print(f"Output dir   : {output_dir}")
        print("=" * 80)

        results = {
            "dataset_type": dataset_type,
            "metadata_csv": str(metadata_csv),
            "audio_dir": str(audio_dir),
            "output_dir": str(output_dir),
        }

        # Stage 1: Acoustic Mining
        acoustic_output = stage1_output or (output_dir / "stage1_acoustic_neighbors.jsonl")

        if skip_stage1 and acoustic_output.exists():
            print(f"\n[Stage 1] Skipping - using existing: {acoustic_output}")
        else:
            print("\n" + "=" * 40)
            print("[Stage 1] Acoustic Negative Mining")
            print("=" * 40)

            acoustic_results = self.acoustic_miner.mine(
                metadata_csv=metadata_csv,
                audio_dir=audio_dir,
                output_path=acoustic_output,
                dataset_type=dataset_type,
                embedding_cache=embedding_cache,
                overwrite_cache=overwrite_cache,
            )

            results["stage1"] = {
                "output": str(acoustic_output),
                "num_records": len(acoustic_results),
            }

        # Stage 2: Semantic Filtering
        filtered_output = output_dir / "stage2_filtered_negatives.jsonl"

        print("\n" + "=" * 40)
        print("[Stage 2] Semantic Filtering")
        print("=" * 40)

        filtered_results = self.semantic_filter.filter(
            acoustic_negatives=acoustic_output,
            captions_csv=metadata_csv,
            output_path=filtered_output,
            dataset_type=dataset_type,
        )

        results["stage2"] = {
            "output": str(filtered_output),
            "num_records": len(filtered_results),
            "semantic_threshold": self.semantic_filter.semantic_threshold,
        }

        # Final summary
        print("\n" + "=" * 80)
        print("Pipeline Complete")
        print("=" * 80)
        print(f"Stage 1 output: {acoustic_output}")
        print(f"Stage 2 output: {filtered_output}")
        print(f"Total records : {len(filtered_results)}")
        print("=" * 80)

        return results

    def run_stage1_only(
        self,
        metadata_csv: Path,
        audio_dir: Path,
        output_path: Path,
        dataset_type: str = "clotho",
        embedding_cache: Optional[Path] = None,
        overwrite_cache: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Run only Stage 1 (acoustic mining).

        Args:
            metadata_csv: Path to dataset metadata CSV
            audio_dir: Directory containing audio files
            output_path: Output JSONL path
            dataset_type: Dataset type (clotho/audiocaps)
            embedding_cache: Optional path to cache embeddings
            overwrite_cache: Overwrite existing cache

        Returns:
            List of acoustic neighbor records
        """
        return self.acoustic_miner.mine(
            metadata_csv=metadata_csv,
            audio_dir=audio_dir,
            output_path=output_path,
            dataset_type=dataset_type,
            embedding_cache=embedding_cache,
            overwrite_cache=overwrite_cache,
        )

    def run_stage2_only(
        self,
        acoustic_negatives: Path,
        captions_csv: Path,
        output_path: Path,
        dataset_type: str = "clotho",
    ) -> List[Dict[str, Any]]:
        """
        Run only Stage 2 (semantic filtering).

        Args:
            acoustic_negatives: Path to Stage 1 output
            captions_csv: Path to captions CSV
            output_path: Output JSONL path
            dataset_type: Dataset type (clotho/audiocaps)

        Returns:
            List of filtered negative records
        """
        return self.semantic_filter.filter(
            acoustic_negatives=acoustic_negatives,
            captions_csv=captions_csv,
            output_path=output_path,
            dataset_type=dataset_type,
        )
