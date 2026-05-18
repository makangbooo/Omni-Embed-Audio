"""
Semantic Filtering for Hard Negatives.

Filters acoustically similar samples by removing semantically similar ones.
This is Stage 2 of the two-stage hard negative mining pipeline.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm


class SemanticFilter:
    """
    Filter hard negatives by semantic similarity.

    Uses BGE or SentenceTransformer models to compute semantic similarity
    and filter out samples that are too semantically similar.

    Args:
        model_name: Name of the sentence transformer model
        device: Device for inference
        batch_size: Batch size for encoding
        semantic_threshold: Maximum semantic similarity to keep (default: 0.7)
        max_negatives: Maximum number of negatives to keep per sample

    Example:
        >>> filter = SemanticFilter(
        ...     model_name="BAAI/bge-large-en-v1.5",
        ...     device="cuda",
        ...     semantic_threshold=0.7,
        ... )
        >>> filtered = filter.filter(
        ...     acoustic_negatives="acoustic_neighbors.jsonl",
        ...     captions_csv="metadata.csv",
        ...     output_path="filtered_negatives.jsonl",
        ... )
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str = "cuda",
        batch_size: int = 64,
        semantic_threshold: float = 0.7,
        max_negatives: int = 50,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.semantic_threshold = semantic_threshold
        self.max_negatives = max_negatives
        self._model = None

    @property
    def model(self):
        """Lazy-load sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode_captions(self, captions: List[str]) -> np.ndarray:
        """Encode captions to embeddings."""
        return self.model.encode(
            captions,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def load_captions_clotho(self, csv_path: Path) -> Dict[str, List[str]]:
        """
        Load Clotho captions from CSV.

        Returns:
            Dictionary mapping audio filename (stem) to list of captions
        """
        caption_dict: Dict[str, List[str]] = {}

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("file_name", "").strip()
                if not filename:
                    continue

                audio_id = Path(filename).stem
                captions = []
                for i in range(1, 6):
                    caption = row.get(f"caption_{i}", "").strip()
                    if caption:
                        captions.append(caption)

                if captions:
                    caption_dict[audio_id] = captions

        return caption_dict

    def load_captions_audiocaps(self, csv_path: Path) -> Dict[str, List[str]]:
        """
        Load AudioCaps captions from CSV.

        Returns:
            Dictionary mapping audio_id (youtube_id_start_time) to list of captions
        """
        caption_dict: Dict[str, List[str]] = {}

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                youtube_id = row.get("youtube_id", "").strip()
                start_time = row.get("start_time", "").strip()
                caption = row.get("caption", "").strip()

                if not youtube_id or not start_time:
                    continue

                audio_id = f"{youtube_id}_{start_time}"
                if audio_id not in caption_dict:
                    caption_dict[audio_id] = []
                if caption:
                    caption_dict[audio_id].append(caption)

        return caption_dict

    def compute_semantic_similarity(
        self,
        anchor_captions: List[str],
        neighbor_captions: List[str],
    ) -> float:
        """
        Compute semantic similarity between caption sets.

        Uses maximum similarity across all caption pairs.

        Args:
            anchor_captions: Captions for anchor audio
            neighbor_captions: Captions for neighbor audio

        Returns:
            Maximum cosine similarity between any caption pair
        """
        if not anchor_captions or not neighbor_captions:
            return 0.0

        anchor_embeds = self.encode_captions(anchor_captions)
        neighbor_embeds = self.encode_captions(neighbor_captions)

        # Compute all pairwise similarities
        sim_matrix = np.dot(anchor_embeds, neighbor_embeds.T)
        return float(np.max(sim_matrix))

    def filter_neighbors(
        self,
        anchor_id: str,
        anchor_captions: List[str],
        neighbors: List[Dict[str, Any]],
        caption_dict: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        """
        Filter neighbors by semantic similarity.

        Args:
            anchor_id: ID of anchor audio
            anchor_captions: Captions for anchor audio
            neighbors: List of neighbor records
            caption_dict: Dictionary mapping audio_id to captions

        Returns:
            Filtered list of neighbors
        """
        if not anchor_captions:
            return []

        # Encode anchor captions once
        anchor_embeds = self.encode_captions(anchor_captions)

        filtered = []
        for neighbor in neighbors:
            neighbor_id = neighbor.get("audio_id", "")
            neighbor_captions = neighbor.get("captions", []) or caption_dict.get(neighbor_id, [])

            if not neighbor_captions:
                continue

            # Compute similarity
            neighbor_embeds = self.encode_captions(neighbor_captions)
            sim_matrix = np.dot(anchor_embeds, neighbor_embeds.T)
            max_sim = float(np.max(sim_matrix))

            # Keep if below threshold
            if max_sim < self.semantic_threshold:
                filtered.append({
                    **neighbor,
                    "semantic_similarity": max_sim,
                })

            if len(filtered) >= self.max_negatives:
                break

        return filtered

    def filter(
        self,
        acoustic_negatives: Path,
        captions_csv: Path,
        output_path: Path,
        dataset_type: str = "clotho",
    ) -> List[Dict[str, Any]]:
        """
        Filter acoustic negatives by semantic similarity.

        Args:
            acoustic_negatives: Path to acoustic neighbors JSONL
            captions_csv: Path to captions CSV
            output_path: Output path for filtered JSONL
            dataset_type: Dataset type (clotho/audiocaps)

        Returns:
            List of filtered negative records
        """
        acoustic_negatives = Path(acoustic_negatives)
        captions_csv = Path(captions_csv)
        output_path = Path(output_path)

        print("=" * 80)
        print("Semantic Filtering for Hard Negatives")
        print("=" * 80)
        print(f"Input (acoustic) : {acoustic_negatives}")
        print(f"Captions CSV     : {captions_csv}")
        print(f"Output           : {output_path}")
        print(f"Threshold        : {self.semantic_threshold}")
        print(f"Max negatives    : {self.max_negatives}")
        print("=" * 80)

        # Load captions
        if dataset_type.lower() == "clotho":
            caption_dict = self.load_captions_clotho(captions_csv)
        elif dataset_type.lower() == "audiocaps":
            caption_dict = self.load_captions_audiocaps(captions_csv)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        print(f"Loaded captions for {len(caption_dict)} audio clips")

        # Load acoustic negatives
        records = []
        with open(acoustic_negatives, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        print(f"Loaded {len(records)} acoustic neighbor records")

        # Filter each record
        filtered_records = []
        for record in tqdm(records, desc="Filtering", unit="record"):
            anchor_id = record.get("audio_id", "")
            anchor_captions = record.get("captions", []) or caption_dict.get(anchor_id, [])
            neighbors = record.get("neighbors", [])

            filtered_neighbors = self.filter_neighbors(
                anchor_id,
                anchor_captions,
                neighbors,
                caption_dict,
            )

            if filtered_neighbors:
                filtered_records.append({
                    "audio_id": anchor_id,
                    "path": record.get("path", ""),
                    "captions": anchor_captions,
                    "hard_negatives": filtered_neighbors,
                    "num_filtered": len(neighbors) - len(filtered_neighbors),
                })

        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as writer:
            for record in filtered_records:
                writer.write(json.dumps(record) + "\n")

        print(f"[INFO] Wrote {len(filtered_records)} filtered records to {output_path}")

        # Statistics
        total_neighbors = sum(len(r.get("neighbors", [])) for r in records)
        total_filtered = sum(len(r.get("hard_negatives", [])) for r in filtered_records)
        print(f"[INFO] Kept {total_filtered}/{total_neighbors} negatives "
              f"({100*total_filtered/max(1,total_neighbors):.1f}%)")

        return filtered_records
