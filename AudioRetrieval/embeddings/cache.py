"""Helpers to load cached caption embeddings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


@dataclass(frozen=True)
class CaptionEmbeddingBundle:
    embeddings: np.ndarray
    index: List[Dict[str, Any]]
    metadata: Dict[str, Any]


def load_caption_embeddings(bundle_dir: Path | str) -> CaptionEmbeddingBundle:
    """
    Load a caption embedding bundle saved by ``precompute_caption_embeddings.py``.

    Parameters
    ----------
    bundle_dir:
        Directory containing ``embeddings.npy``, ``index.json``, and ``metadata.json``.
    """
    bundle_path = Path(bundle_dir)
    embeddings_path = bundle_path / "embeddings.npy"
    index_path = bundle_path / "index.json"
    metadata_path = bundle_path / "metadata.json"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file missing: {embeddings_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Index file missing: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file missing: {metadata_path}")

    embeddings = np.load(embeddings_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return CaptionEmbeddingBundle(embeddings=embeddings, index=index, metadata=metadata)
