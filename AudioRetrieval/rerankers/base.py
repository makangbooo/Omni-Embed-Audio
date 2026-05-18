"""Base abstractions for reranking retrieved audio candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
from abc import ABC, abstractmethod


@dataclass
class CandidateItem:
    """
    Represents a single candidate produced by the first-stage retriever.

    Attributes
    ----------
    candidate_idx:
        Integer index of this candidate within the dataset / embedding bank.
    clip_id:
        String identifier of the clip (usually the dataset key).
    audio_path:
        Filesystem path to the audio file.
    initial_score:
        Similarity score produced by the first-stage retriever.
    caption:
        Optional text caption already available for the audio clip.
    metadata:
        Arbitrary extra information that downstream rerankers might need.
    """

    candidate_idx: int
    clip_id: str
    audio_path: Path
    initial_score: float
    caption: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankResult:
    """
    Output bundle from a reranker execution.

    Attributes
    ----------
    candidates:
        Candidates sorted according to the reranker's final order.
    latency_seconds:
        Wall-clock time spent inside the reranker (seconds).
    gpu_cost_seconds:
        Optional GPU compute time (seconds). For mocks this is zero.
    extra:
        Additional metadata such as prompt templates, mode identifiers, etc.
    """

    candidates: List[CandidateItem]
    latency_seconds: float
    gpu_cost_seconds: float
    extra: Dict[str, Any] = field(default_factory=dict)


class LALMClient(Protocol):
    """Protocol that Large Audio Language Model clients must satisfy."""

    def score_audio(self, query: str, audio_path: Path) -> float:
        """Pointwise audio-native scoring."""

    def score_caption(self, query: str, caption: str) -> float:
        """Pointwise caption-proxy scoring."""

    def rank_audio(self, query: str, candidates: List[CandidateItem]) -> List[int]:
        """Listwise audio-native ranking; returns candidate indices order."""

    def rank_captions(self, query: str, captions: List[str]) -> List[int]:
        """Listwise caption-proxy ranking; returns indices into ``captions``."""

    def generate_caption(self, audio_path: Path) -> str:
        """Produce a textual caption for the given audio clip."""


class BaseReranker(ABC):
    """Abstract base class for all rerankers."""

    def __init__(
        self,
        name: str,
        top_k: int,
        client: LALMClient,
    ) -> None:
        self.name = name
        self.top_k = top_k
        self.client = client

    @abstractmethod
    def rerank(self, query: str, candidates: List[CandidateItem]) -> RerankResult:
        """Reorder candidates for a text query and return the reranked bundle."""

    def limit_candidates(self, candidates: List[CandidateItem]) -> List[CandidateItem]:
        """Clamp the number of candidates according to ``top_k``."""
        if self.top_k <= 0:
            return candidates
        return candidates[: self.top_k]

