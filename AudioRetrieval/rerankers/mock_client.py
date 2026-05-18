"""Mock implementation of a Large Audio Language Model (LALM) client.

This client provides deterministic heuristic scores so that the reranking
pipeline can be exercised without access to proprietary models.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

from .base import CandidateItem, LALMClient


def _stable_score_key(*parts: str) -> float:
    """Return a deterministic pseudo-score in [0, 1)."""
    key = "||".join(parts).encode("utf-8")
    digest = hashlib.md5(key).digest()
    # Convert the first 8 bytes to an integer and normalise.
    return int.from_bytes(digest[:8], "little") / 2**64


class MockLALMClient(LALMClient):
    """
    Deterministic mock that approximates semantic scoring.

    The mock uses hashed combinations of query text, audio stem, and captions
    to produce pseudo-scores. This keeps behaviour reproducible while allowing
    the reranking pipeline to stress mode-specific logic (audio-native vs
    caption-proxy, pointwise vs listwise).
    """

    def score_audio(self, query: str, audio_path: Path) -> float:
        stem = audio_path.stem
        return _stable_score_key("audio", query.lower(), stem)

    def score_caption(self, query: str, caption: str) -> float:
        return _stable_score_key("caption", query.lower(), caption.lower())

    def rank_audio(self, query: str, candidates: List[CandidateItem]) -> List[int]:
        scores = [
            (idx, self.score_audio(query, cand.audio_path))
            for idx, cand in enumerate(candidates)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scores]

    def rank_captions(self, query: str, captions: List[str]) -> List[int]:
        scores = [
            (idx, self.score_caption(query, caption))
            for idx, caption in enumerate(captions)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scores]

    def generate_caption(self, audio_path: Path) -> str:
        stem = audio_path.stem.replace("_", " ")
        # Deterministic pseudo-caption.
        return f"Mock description of audio clip '{stem}'"

