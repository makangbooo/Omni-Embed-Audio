"""
Acoustic Hard Negative Mining.

Mines acoustically similar audio samples using MGA-CLAP embeddings.
This is Stage 1 of the two-stage hard negative mining pipeline.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm


@dataclass
class AudioExample:
    """Single audio example with metadata."""
    clip_id: str
    path: Path
    captions: List[str] = None

    def __post_init__(self):
        if self.captions is None:
            self.captions = []


@dataclass
class NeighborResult:
    """Result of nearest neighbor search."""
    audio_id: str
    path: str
    score: float


class AcousticNegativeMiner:
    """
    Mine acoustically similar audio samples using MGA-CLAP.

    This class implements Stage 1 of the hard negative mining pipeline:
    finding audio samples that are acoustically similar but may have
    different semantic content.

    Args:
        mga_repo: Path to MGA-CLAP repository
        mga_ckpt: Path to MGA-CLAP checkpoint
        seconds: Audio duration for MGA-CLAP (default: 10.0)
        device: Device for inference
        batch_size: Batch size for audio encoding
        topk: Number of neighbors to find per audio

    Example:
        >>> miner = AcousticNegativeMiner(
        ...     mga_repo="models/mga_clap",
        ...     mga_ckpt="checkpoints/mga-clap.pt",
        ...     device="cuda",
        ... )
        >>> results = miner.mine(
        ...     metadata_csv="clotho/evaluation_meta.csv",
        ...     audio_dir="clotho/evaluation_audio",
        ...     output_path="hard_negatives/clotho_acoustic.jsonl",
        ... )
    """

    def __init__(
        self,
        mga_repo: str,
        mga_ckpt: str,
        seconds: float = 10.0,
        device: str = "cuda",
        batch_size: int = 512,
        topk: int = 50,
    ):
        self.mga_repo = mga_repo
        self.mga_ckpt = mga_ckpt
        self.seconds = seconds
        self.device = device
        self.batch_size = batch_size
        self.topk = topk
        self._adapter = None

    @property
    def adapter(self):
        """Lazy-load MGA-CLAP adapter."""
        if self._adapter is None:
            from AudioRetrieval.models.mga_clap_adapter import MGAClapAdapter
            self._adapter = MGAClapAdapter(
                repo_path=str(self.mga_repo),
                ckpt_path=str(self.mga_ckpt),
                seconds=self.seconds,
                device=self.device,
            )
        return self._adapter

    def load_clotho_examples(
        self,
        csv_path: Path,
        audio_dir: Path,
    ) -> List[AudioExample]:
        """Load Clotho dataset examples."""
        examples = []
        missing = []

        with open(csv_path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                filename = (row.get("file_name") or "").strip()
                if not filename:
                    continue

                clip_id = Path(filename).stem
                audio_path = audio_dir / filename

                captions = []
                for i in range(1, 6):
                    caption = row.get(f"caption_{i}", "").strip()
                    if caption:
                        captions.append(caption)

                if audio_path.exists():
                    examples.append(AudioExample(
                        clip_id=clip_id,
                        path=audio_path.resolve(),
                        captions=captions,
                    ))
                else:
                    missing.append(filename)

        if missing:
            print(f"[WARN] Missing {len(missing)} audio files (first 5): {missing[:5]}")

        return examples

    def load_audiocaps_examples(
        self,
        csv_path: Path,
        audio_dir: Path,
    ) -> List[AudioExample]:
        """Load AudioCaps dataset examples."""
        audio_to_captions: Dict[str, List[str]] = {}

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                youtube_id = row.get("youtube_id", "").strip()
                start_time = row.get("start_time", "").strip()
                caption = row.get("caption", "").strip()

                if not youtube_id or not start_time:
                    continue

                audio_key = f"{youtube_id}_{start_time}"
                if audio_key not in audio_to_captions:
                    audio_to_captions[audio_key] = []
                if caption:
                    audio_to_captions[audio_key].append(caption)

        examples = []
        missing = []

        for audio_key, captions in sorted(audio_to_captions.items()):
            audio_path = audio_dir / f"{audio_key}.wav"
            if audio_path.exists():
                examples.append(AudioExample(
                    clip_id=audio_key,
                    path=audio_path.resolve(),
                    captions=captions,
                ))
            else:
                missing.append(f"{audio_key}.wav")

        if missing:
            print(f"[WARN] Missing {len(missing)} audio files (first 5): {missing[:5]}")

        return examples

    def encode_examples(
        self,
        examples: Sequence[AudioExample],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode all audio examples to embeddings."""
        clip_paths = [str(ex.path) for ex in examples]
        embeddings_list = []

        iterator = range(0, len(clip_paths), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding audio", unit="batch")

        for i in iterator:
            batch = clip_paths[i:i + self.batch_size]
            batch_embs = self.adapter.encode_audio(
                batch,
                batch_size=self.batch_size,
                device=self.device,
            )
            embeddings_list.append(batch_embs)

        return np.concatenate(embeddings_list, axis=0)

    def compute_topk_neighbors(
        self,
        embeddings: np.ndarray,
        examples: Sequence[AudioExample],
        topk_batch: int = 1024,
        similarity_device: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compute top-K nearest neighbors for each audio.

        Args:
            embeddings: Audio embeddings [N, D]
            examples: List of AudioExample objects
            topk_batch: Batch size for similarity computation
            similarity_device: Device for similarity computation

        Returns:
            List of neighbor records
        """
        assert len(embeddings) == len(examples)
        topk = min(self.topk, len(examples) - 1)

        # L2 normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-6, None)

        sim_device = similarity_device or (self.device if torch.cuda.is_available() else "cpu")
        sims_tensor = torch.from_numpy(embeddings).to(sim_device)

        results = []
        for start in tqdm(range(0, len(examples), topk_batch), desc="Finding neighbors", unit="batch"):
            end = min(start + topk_batch, len(examples))
            queries = sims_tensor[start:end]
            sim_matrix = torch.matmul(queries, sims_tensor.t())

            for i in range(start, end):
                row = sim_matrix[i - start]
                row[i] = -1.0  # exclude self

                scores, indices = torch.topk(row, k=topk, largest=True, sorted=True)

                neighbors = [
                    {
                        "audio_id": examples[j].clip_id,
                        "path": str(examples[j].path),
                        "score": float(scores[k].item()),
                        "captions": examples[j].captions,
                    }
                    for k, j in enumerate(indices.tolist())
                ]

                results.append({
                    "audio_id": examples[i].clip_id,
                    "path": str(examples[i].path),
                    "captions": examples[i].captions,
                    "neighbors": neighbors,
                })

        return results

    def mine(
        self,
        metadata_csv: Path,
        audio_dir: Path,
        output_path: Path,
        dataset_type: str = "clotho",
        embedding_cache: Optional[Path] = None,
        overwrite_cache: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Run acoustic negative mining pipeline.

        Args:
            metadata_csv: Path to dataset metadata CSV
            audio_dir: Directory containing audio files
            output_path: Output JSONL path for neighbors
            dataset_type: Dataset type (clotho/audiocaps)
            embedding_cache: Optional path to cache embeddings
            overwrite_cache: Overwrite existing cache

        Returns:
            List of neighbor records
        """
        metadata_csv = Path(metadata_csv)
        audio_dir = Path(audio_dir)
        output_path = Path(output_path)

        print("=" * 80)
        print("MGA-CLAP Acoustic Negative Mining")
        print("=" * 80)
        print(f"Metadata CSV : {metadata_csv}")
        print(f"Audio dir    : {audio_dir}")
        print(f"Output JSONL : {output_path}")
        print(f"Top-K        : {self.topk}")
        print("=" * 80)

        # Load examples
        if dataset_type.lower() == "clotho":
            examples = self.load_clotho_examples(metadata_csv, audio_dir)
        elif dataset_type.lower() == "audiocaps":
            examples = self.load_audiocaps_examples(metadata_csv, audio_dir)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        if not examples:
            raise RuntimeError("No audio files found")

        print(f"Found {len(examples)} audio clips")

        # Check cache
        embeddings = None
        if embedding_cache and embedding_cache.exists() and not overwrite_cache:
            print(f"[INFO] Loading cached embeddings from {embedding_cache}")
            embeddings = np.load(embedding_cache)
            if embeddings.shape[0] != len(examples):
                print("[WARN] Cache size mismatch, recomputing...")
                embeddings = None

        if embeddings is None:
            embeddings = self.encode_examples(examples)
            if embedding_cache:
                embedding_cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(embedding_cache, embeddings)
                print(f"[INFO] Saved embeddings cache to {embedding_cache}")

        # Compute neighbors
        results = self.compute_topk_neighbors(embeddings, examples)

        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as writer:
            for record in results:
                writer.write(json.dumps(record) + "\n")

        print(f"[INFO] Wrote {len(results)} records to {output_path}")
        return results
