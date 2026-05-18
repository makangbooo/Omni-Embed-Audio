"""Utilities for working with cached embedding bundles."""

from .cache import CaptionEmbeddingBundle, load_caption_embeddings

__all__ = [
    "CaptionEmbeddingBundle",
    "load_caption_embeddings",
]
