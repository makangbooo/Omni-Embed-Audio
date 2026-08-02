"""Embedding precomputation for the pinned Robust-CLAP controlled binding."""

from __future__ import annotations

from typing import Any, List

import numpy as np

from AudioRetrieval.preprocessing.embeddings.base import BaseEmbeddingPrecomputer
from AudioRetrieval.preprocessing.embeddings.laion_clap import (
    LaionClapEmbeddingPrecomputer,
)


class RobustClapEmbeddingPrecomputer(LaionClapEmbeddingPrecomputer):
    def __init__(
        self,
        ckpt_path: str,
        repo_root: str,
        bert_tokenizer_path: str,
        roberta_tokenizer_path: str,
        bart_tokenizer_path: str,
        bpe_vocab_path: str,
        *,
        enable_fusion: bool = False,
        device: str = "cuda",
        batch_size_audio: int = 32,
        batch_size_text: int = 128,
    ) -> None:
        BaseEmbeddingPrecomputer.__init__(
            self, device, batch_size_audio, batch_size_text
        )
        self.ckpt_path = ckpt_path
        self.repo_root = repo_root
        self.bert_tokenizer_path = bert_tokenizer_path
        self.roberta_tokenizer_path = roberta_tokenizer_path
        self.bart_tokenizer_path = bart_tokenizer_path
        self.bpe_vocab_path = bpe_vocab_path
        self.enable_fusion = enable_fusion

    def _load_model(self) -> Any:
        from AudioRetrieval.models.robust_clap_adapter import RobustClapAdapter

        return RobustClapAdapter(
            ckpt_path=self.ckpt_path,
            repo_root=self.repo_root,
            bert_tokenizer_path=self.bert_tokenizer_path,
            roberta_tokenizer_path=self.roberta_tokenizer_path,
            bart_tokenizer_path=self.bart_tokenizer_path,
            bpe_vocab_path=self.bpe_vocab_path,
            enable_fusion=self.enable_fusion,
        )

    def _encode_audio_batch(self, audio_paths: List[str]) -> np.ndarray:
        return self.model.encode_audio(
            audio_paths, batch_size=self.batch_size_audio, device=self.device
        )

    def _encode_text_batch(self, texts: List[str]) -> np.ndarray:
        return self.model.encode_text(
            texts, batch_size=self.batch_size_text, device=self.device
        )
