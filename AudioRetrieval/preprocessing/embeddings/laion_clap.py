"""
LAION-CLAP Embedding Precomputation.

Provides embedding precomputation using the LAION-CLAP model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from AudioRetrieval.preprocessing.embeddings.base import BaseEmbeddingPrecomputer


class LaionClapEmbeddingPrecomputer(BaseEmbeddingPrecomputer):
    """
    LAION-CLAP embedding precomputer.

    Uses the LAION-CLAP model for audio-text embedding extraction.

    Args:
        ckpt_path: Path to LAION-CLAP checkpoint
        amodel: Audio model architecture (default: HTSAT-tiny)
        tmodel: Text model architecture (default: roberta)
        enable_fusion: Enable audio-text fusion (default: False)
        device: Device for inference
        batch_size_audio: Batch size for audio encoding
        batch_size_text: Batch size for text encoding

    Example:
        >>> precomputer = LaionClapEmbeddingPrecomputer(
        ...     ckpt_path="checkpoints/630k-audioset-best.pt",
        ...     device="cuda",
        ... )
        >>> audio_embeds = precomputer.encode_audio(["/path/to/audio.wav"])
        >>> text_embeds = precomputer.encode_text(["A dog barking"])
    """

    def __init__(
        self,
        ckpt_path: str = "venv/.venv-laionclap/lib/python3.10/site-packages/laion_clap/630k-audioset-best.pt",
        amodel: str = "HTSAT-tiny",
        tmodel: str = "roberta",
        enable_fusion: bool = False,
        device: str = "cuda",
        batch_size_audio: int = 64,
        batch_size_text: int = 256,
    ):
        super().__init__(device, batch_size_audio, batch_size_text)
        self.ckpt_path = ckpt_path
        self.amodel = amodel
        self.tmodel = tmodel
        self.enable_fusion = enable_fusion

    def _load_model(self) -> Any:
        """Load LAION-CLAP adapter."""
        from AudioRetrieval.models.laion_clap_adapter import LaionClapAdapter

        return LaionClapAdapter(
            ckpt_path=self.ckpt_path,
            amodel=self.amodel,
            tmodel=self.tmodel,
            enable_fusion=self.enable_fusion,
        )

    def _encode_audio_batch(self, audio_paths: List[str]) -> np.ndarray:
        """Encode a batch of audio files."""
        return self.model.encode_audio(
            audio_paths,
            batch_size=self.batch_size_audio,
            device=self.device,
        )

    def _encode_text_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of text strings."""
        return self.model.encode_text(
            texts,
            batch_size=self.batch_size_text,
            device=self.device,
        )

    def precompute_full_dataset(
        self,
        audio_dir: Path,
        captions_csv: Path,
        uiq_jsonl: Optional[Path],
        output_dir: Path,
        dataset_type: str = "clotho",
        split: str = "eval",
        uiq_categories: Optional[List[str]] = None,
    ) -> None:
        """
        Precompute all embeddings for a dataset.

        Args:
            audio_dir: Directory containing audio files
            captions_csv: Path to captions CSV file
            uiq_jsonl: Optional path to UIQ JSONL file
            output_dir: Output directory for embeddings
            dataset_type: Dataset type (clotho/audiocaps)
            split: Dataset split name
            uiq_categories: UIQ categories to compute (default: all standard)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print(f"LAION-CLAP Embedding Precomputation ({dataset_type})")
        print("=" * 60)
        print(f"Audio directory: {audio_dir}")
        print(f"Captions CSV: {captions_csv}")
        print(f"Output directory: {output_dir}")
        print("=" * 60)

        # Load entries based on dataset type
        if dataset_type.lower() == "clotho":
            entries = self.load_clotho_entries(captions_csv, audio_dir, split)
        elif dataset_type.lower() == "audiocaps":
            entries = self.load_audiocaps_entries(captions_csv, audio_dir, split)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        print(f"Loaded {len(entries)} entries")

        # Precompute audio and caption embeddings
        self.precompute_dataset(entries, output_dir)

        # Precompute UIQ embeddings if provided
        if uiq_jsonl and uiq_jsonl.exists():
            if uiq_categories is None:
                uiq_categories = [
                    "Pragmatic",
                    "Subjective",
                    "Contextual",
                    "Negative/Contrastive",
                    "Cross-domain",
                ]

            uiq_data = self.load_uiq_queries(uiq_jsonl)
            valid_clip_ids = {e.clip_id for e in entries if e.audio_path}

            for category in uiq_categories:
                if category not in uiq_data:
                    print(f"Warning: UIQ category '{category}' not found")
                    continue

                category_queries = uiq_data[category]
                texts = []
                clip_ids = []
                for clip_id, query in sorted(category_queries.items()):
                    if clip_id in valid_clip_ids:
                        texts.append(query)
                        clip_ids.append(clip_id)

                if not texts:
                    print(f"Warning: No UIQ queries for '{category}'")
                    continue

                print(f"Computing UIQ embeddings for '{category}' ({len(texts)} queries)...")
                embeddings = self.encode_text(texts)

                bucket_safe = category.lower().replace("/", "_").replace("-", "_")
                output_path = output_dir / f"uiq_{bucket_safe}_embeddings.npz"
                np.savez_compressed(
                    output_path,
                    embeddings=embeddings,
                    texts=texts,
                    clip_ids=clip_ids,
                    bucket=category,
                )
                print(f"Saved to {output_path}")

        print()
        print("=" * 60)
        print("All embeddings precomputed successfully!")
        print("=" * 60)
