"""
Hard Negative Mining Module.

Provides tools for mining and filtering hard negatives for training:
1. Acoustic mining: Find acoustically similar samples using MGA-CLAP
2. Semantic filtering: Filter out semantically similar samples using BGE

Classes:
    AcousticNegativeMiner: Mine acoustically similar audio samples
    SemanticFilter: Filter negatives by semantic similarity
    HardNegativePipeline: Combined two-stage pipeline
"""

from AudioRetrieval.preprocessing.hard_negatives.acoustic_mining import AcousticNegativeMiner
from AudioRetrieval.preprocessing.hard_negatives.semantic_filtering import SemanticFilter
from AudioRetrieval.preprocessing.hard_negatives.pipeline import HardNegativePipeline

__all__ = [
    "AcousticNegativeMiner",
    "SemanticFilter",
    "HardNegativePipeline",
]
