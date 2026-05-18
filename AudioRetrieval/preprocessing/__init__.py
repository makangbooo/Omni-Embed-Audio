"""
AudioRetrieval Preprocessing Module.

This module provides tools for:
1. Embedding precomputation (audio and text embeddings)
2. Hard negative mining (acoustic similarity + semantic filtering)
3. Data utilities for dataset processing

Submodules:
    - embeddings: Precompute embeddings for various models
    - hard_negatives: Mine and filter hard negatives for training

Example usage:
    from AudioRetrieval.preprocessing.embeddings import OEAEmbeddingPrecomputer
    from AudioRetrieval.preprocessing.hard_negatives import HardNegativePipeline
"""

from AudioRetrieval.preprocessing.embeddings import (
    BaseEmbeddingPrecomputer,
    OEAEmbeddingPrecomputer,
    LaionClapEmbeddingPrecomputer,
    MGAClapEmbeddingPrecomputer,
    UIQTextEmbeddingPrecomputer,
)

from AudioRetrieval.preprocessing.hard_negatives import (
    AcousticNegativeMiner,
    SemanticFilter,
    HardNegativePipeline,
)

__all__ = [
    # Embedding precomputers
    "BaseEmbeddingPrecomputer",
    "OEAEmbeddingPrecomputer",
    "LaionClapEmbeddingPrecomputer",
    "MGAClapEmbeddingPrecomputer",
    "UIQTextEmbeddingPrecomputer",
    # Hard negative tools
    "AcousticNegativeMiner",
    "SemanticFilter",
    "HardNegativePipeline",
]
