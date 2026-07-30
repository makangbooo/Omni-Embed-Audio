"""
MGA-CLAP Embedding Precomputation.

Provides embedding precomputation using the MGA-CLAP model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from AudioRetrieval.preprocessing.embeddings.base import BaseEmbeddingPrecomputer


class MGAClapEmbeddingPrecomputer(BaseEmbeddingPrecomputer):
    """
    MGA-CLAP embedding precomputer.

    Uses the Multi-Grained Attention CLAP model for audio-text embedding extraction.

    Args:
        repo_path: Path to MGA-CLAP repository
        ckpt_path: Path to MGA-CLAP checkpoint
        seconds: Audio duration in seconds (default: 10.0)
        device: Device for inference
        batch_size_audio: Batch size for audio encoding
        batch_size_text: Batch size for text encoding

    Example:
        >>> precomputer = MGAClapEmbeddingPrecomputer(
        ...     repo_path="AudioRetrieval/models/mga_clap",
        ...     ckpt_path="checkpoints/mga-clap.pt",
        ...     device="cuda",
        ... )
        >>> audio_embeds = precomputer.encode_audio(["/path/to/audio.wav"])
        >>> text_embeds = precomputer.encode_text(["A dog barking"])
    """

    def __init__(
        self,
        repo_path: str = "AudioRetrieval/models/mga_clap",
        ckpt_path: str = "ModelCheckpoint/mga_clap/mga-clap.pt",
        bert_tokenizer_path: str | None = None,
        checkpoint_sha256: str | None = None,
        seconds: float = 10.0,
        device: str = "cuda",
        batch_size_audio: int = 64,
        batch_size_text: int = 256,
    ):
        super().__init__(device, batch_size_audio, batch_size_text)
        self.repo_path = repo_path
        self.ckpt_path = ckpt_path
        self.bert_tokenizer_path = bert_tokenizer_path
        self.checkpoint_sha256 = checkpoint_sha256
        self.seconds = seconds

    def _load_model(self) -> Any:
        """Load MGA-CLAP adapter."""
        from AudioRetrieval.models.mga_clap_adapter import MGAClapAdapter

        return MGAClapAdapter(
            repo_path=self.repo_path,
            ckpt_path=self.ckpt_path,
            seconds=self.seconds,
            device=self.device,
            bert_tokenizer_path=self.bert_tokenizer_path,
            expected_checkpoint_sha256=self.checkpoint_sha256,
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
        for name in ("audio_embeddings.npz", "caption_embeddings.npz"):
            if (output_dir / name).exists():
                raise FileExistsError(f"refusing to overwrite {output_dir / name}")

        print("=" * 60)
        print(f"MGA-CLAP Embedding Precomputation ({dataset_type})")
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

        # Preserve the generic UIQ route; the Table 2/3 wrapper does not use it.
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
                texts = []
                clip_ids = []
                for clip_id, query in sorted(uiq_data[category].items()):
                    if clip_id in valid_clip_ids:
                        texts.append(query)
                        clip_ids.append(clip_id)
                if not texts:
                    print(f"Warning: No UIQ queries for '{category}'")
                    continue
                embeddings = self.encode_text(texts)
                bucket_safe = category.lower().replace("/", "_").replace("-", "_")
                output_path = output_dir / f"uiq_{bucket_safe}_embeddings.npz"
                if output_path.exists():
                    raise FileExistsError(f"refusing to overwrite {output_path}")
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
