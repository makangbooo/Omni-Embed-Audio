"""M2D-CLAP embedding precomputation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from AudioRetrieval.preprocessing.embeddings.base import BaseEmbeddingPrecomputer


class M2DClapEmbeddingPrecomputer(BaseEmbeddingPrecomputer):
    def __init__(
        self,
        checkpoint: str,
        bert_tokenizer_path: str,
        device: str = "cuda",
        batch_size_audio: int = 8,
        batch_size_text: int = 128,
    ):
        super().__init__(device, batch_size_audio, batch_size_text)
        self.checkpoint = checkpoint
        self.bert_tokenizer_path = bert_tokenizer_path

    def _load_model(self) -> Any:
        from AudioRetrieval.models.m2d_clap_adapter import M2DClapAdapter

        return M2DClapAdapter(
            weight_file=self.checkpoint,
            device=self.device,
            bert_tokenizer_path=self.bert_tokenizer_path,
        )

    def _encode_audio_batch(self, audio_paths: List[str]) -> np.ndarray:
        return self.model.encode_audio(
            audio_paths,
            batch_size=self.batch_size_audio,
            device=self.device,
        )

    def _encode_text_batch(self, texts: List[str]) -> np.ndarray:
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
    ) -> None:
        if uiq_jsonl is not None:
            raise ValueError("M2D main-table precomputation does not accept UIQ input")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("audio_embeddings.npz", "caption_embeddings.npz"):
            if (output_dir / name).exists():
                raise FileExistsError(f"refusing to overwrite {output_dir / name}")
        if dataset_type.lower() == "clotho":
            entries = self.load_clotho_entries(captions_csv, audio_dir, split)
        elif dataset_type.lower() == "audiocaps":
            entries = self.load_audiocaps_entries(captions_csv, audio_dir, split)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        print("=" * 60)
        print(f"M2D-CLAP Embedding Precomputation ({dataset_type})")
        print("=" * 60)
        print(f"Loaded {len(entries)} entries")
        self.precompute_dataset(entries, output_dir)
        print("M2D_EMBEDDING_PRECOMPUTATION_STATUS=complete")
