"""Reranking module entry point.

This package exposes utilities for constructing Large Audio Language Model
(LALM) based rerankers that operate on the candidates returned by the
first-stage CLAP-style retriever.

Use :func:`build_reranker` to instantiate a reranker from configuration
and :class:`BaseReranker` as the main interface.
"""

from .base import (
    BaseReranker,
    CandidateItem,
    RerankResult,
)
from .factory import build_reranker
from .late_interaction import LateInteractionClient, LateInteractionReranker
from .xacle_baseline_client import XACLEBaselineClient, XACLEBaselineDependenciesMissing

__all__ = [
    "BaseReranker",
    "CandidateItem",
    "RerankResult",
    "build_reranker",
    "LateInteractionClient",
    "LateInteractionReranker",
    "XACLEBaselineClient",
    "XACLEBaselineDependenciesMissing",
]
