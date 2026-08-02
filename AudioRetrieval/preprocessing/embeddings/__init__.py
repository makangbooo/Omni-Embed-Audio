"""
Embedding Precomputation Module.

Provides unified interfaces for OEA and baseline embedding backends.
"""

from AudioRetrieval.preprocessing.embeddings.base import BaseEmbeddingPrecomputer
from AudioRetrieval.preprocessing.embeddings.laion_clap import (
    LaionClapEmbeddingPrecomputer,
)
from AudioRetrieval.preprocessing.embeddings.m2d_clap import M2DClapEmbeddingPrecomputer
from AudioRetrieval.preprocessing.embeddings.mga_clap import MGAClapEmbeddingPrecomputer
from AudioRetrieval.preprocessing.embeddings.oea import OEAEmbeddingPrecomputer
from AudioRetrieval.preprocessing.embeddings.robust_clap import (
    RobustClapEmbeddingPrecomputer,
)
from AudioRetrieval.preprocessing.embeddings.uiq_text import (
    UIQTextEmbeddingPrecomputer,
)

__all__ = [
    "BaseEmbeddingPrecomputer",
    "OEAEmbeddingPrecomputer",
    "LaionClapEmbeddingPrecomputer",
    "MGAClapEmbeddingPrecomputer",
    "M2DClapEmbeddingPrecomputer",
    "RobustClapEmbeddingPrecomputer",
    "UIQTextEmbeddingPrecomputer",
]
