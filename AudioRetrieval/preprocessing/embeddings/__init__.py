"""
Embedding Precomputation Module.

Provides unified interface for precomputing audio and text embeddings
using various model backends (OEA, LAION-CLAP, MGA-CLAP, etc.).

Classes:
    BaseEmbeddingPrecomputer: Abstract base class for all precomputers
    OEAEmbeddingPrecomputer: Omni-Embed Audio embedding precomputation
    LaionClapEmbeddingPrecomputer: LAION-CLAP embedding precomputation
    MGAClapEmbeddingPrecomputer: MGA-CLAP embedding precomputation
    UIQTextEmbeddingPrecomputer: UIQ text query embedding precomputation
"""

from AudioRetrieval.preprocessing.embeddings.base import BaseEmbeddingPrecomputer
from AudioRetrieval.preprocessing.embeddings.oea import OEAEmbeddingPrecomputer
from AudioRetrieval.preprocessing.embeddings.laion_clap import LaionClapEmbeddingPrecomputer
from AudioRetrieval.preprocessing.embeddings.mga_clap import MGAClapEmbeddingPrecomputer
from AudioRetrieval.preprocessing.embeddings.m2d_clap import M2DClapEmbeddingPrecomputer
from AudioRetrieval.preprocessing.embeddings.uiq_text import UIQTextEmbeddingPrecomputer

__all__ = [
    "BaseEmbeddingPrecomputer",
    "OEAEmbeddingPrecomputer",
    "LaionClapEmbeddingPrecomputer",
    "MGAClapEmbeddingPrecomputer",
    "M2DClapEmbeddingPrecomputer",
    "UIQTextEmbeddingPrecomputer",
]
