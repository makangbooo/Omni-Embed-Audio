"""
Negative Query Evaluation Runner.

Evaluates retrieval with negative/exclusion queries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

from AudioRetrieval.evaluation.metrics import (
    l2norm,
    cosine_sim,
    ranks_from_scores,
    compute_all_metrics,
    format_metrics,
)
from AudioRetrieval.evaluation.runners.baseline import BaselineRunner


class NegativeQueryRunner(BaselineRunner):
    """
    Negative query evaluation runner.

    Evaluates queries that include exclusion conditions
    (e.g., "Find dog barking but not with music").

    Args:
        model_name: Name of the model to use
        device: Device for inference
        batch_size_audio: Batch size for audio encoding
        batch_size_text: Batch size for text encoding

    Example:
        >>> runner = NegativeQueryRunner(model_name="laion_clap", device="cuda")
        >>> results = runner.run(
        ...     audio_dir=Path("clotho/audio"),
        ...     negative_queries_jsonl=Path("negative_uiq.jsonl"),
        ...     dataset_type="clotho",
        ... )
    """

    def load_negative_queries(
        self,
        jsonl_path: Path,
    ) -> List[Dict[str, Any]]:
        """
        Load negative queries from JSONL file.

        Expected format:
        {
            "target_audio": "clip_id",
            "negative_audio": "negative_clip_id",
            "negative_query": "Find X but not Y",
            ...
        }

        Args:
            jsonl_path: Path to negative queries JSONL

        Returns:
            List of query records
        """
        queries = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                item = json.loads(line)
                target_audio = item.get("target_audio", "")
                negative_audio = item.get("negative_audio", "")
                query = item.get("negative_query", "") or item.get("query", "")

                if target_audio and query:
                    queries.append({
                        "target_audio": target_audio,
                        "negative_audio": negative_audio,
                        "query": query,
                        "metadata": item,
                    })

        return queries

    def evaluate_negative_queries(
        self,
        audio_embeddings: np.ndarray,
        clip_ids: List[str],
        queries: List[Dict[str, Any]],
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate negative queries.

        Args:
            audio_embeddings: Audio embeddings [N_audio, D]
            clip_ids: List of clip IDs
            queries: List of negative query records
            show_progress: Show progress bar

        Returns:
            Dictionary with metrics and detailed results
        """
        clip_id_to_idx = {cid: idx for idx, cid in enumerate(clip_ids)}

        # Filter valid queries
        valid_queries = []
        for q in queries:
            target = q["target_audio"]
            if target in clip_id_to_idx:
                valid_queries.append(q)

        if not valid_queries:
            return {
                "metrics": {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "MRR": 0.0, "DCG": 0.0},
                "num_queries": 0,
            }

        # Encode queries
        query_texts = [q["query"] for q in valid_queries]
        text_embeddings_list = []
        iterator = range(0, len(query_texts), self.batch_size_text)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding queries", unit="batch")

        for i in iterator:
            batch = query_texts[i:i + self.batch_size_text]
            batch_embeds = self.adapter.encode_text(
                batch,
                batch_size=self.batch_size_text,
                device=self.device,
            )
            text_embeddings_list.append(batch_embeds)

        text_embeddings = np.concatenate(text_embeddings_list, axis=0)

        # Compute similarities
        audio_norm = l2norm(audio_embeddings)
        text_norm = l2norm(text_embeddings)
        sim_matrix = cosine_sim(text_norm, audio_norm)

        # Compute ranks
        standard_ranks = []
        adjusted_ranks = []  # Ranks with negative audio excluded

        for i, q in enumerate(valid_queries):
            target_idx = clip_id_to_idx[q["target_audio"]]
            negative_audio = q.get("negative_audio", "")
            negative_idx = clip_id_to_idx.get(negative_audio, None)

            # Standard rank
            rank = ranks_from_scores(sim_matrix[i], target_idx)
            standard_ranks.append(rank)

            # Adjusted rank (exclude negative)
            if negative_idx is not None:
                adj_rank = ranks_from_scores(sim_matrix[i], target_idx, ignore_index=negative_idx)
            else:
                adj_rank = rank
            adjusted_ranks.append(adj_rank)

        standard_metrics = compute_all_metrics(np.array(standard_ranks))
        adjusted_metrics = compute_all_metrics(np.array(adjusted_ranks))

        return {
            "standard_metrics": standard_metrics,
            "adjusted_metrics": adjusted_metrics,
            "num_queries": len(valid_queries),
            "avg_rank_improvement": np.mean(np.array(standard_ranks) - np.array(adjusted_ranks)),
        }

    def run(
        self,
        audio_dir: Path,
        captions_csv: Path,
        negative_queries_jsonl: Path,
        dataset_type: str = "clotho",
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Run negative query evaluation.

        Args:
            audio_dir: Directory containing audio files
            captions_csv: Path to captions CSV
            negative_queries_jsonl: Path to negative queries JSONL
            dataset_type: Dataset type (clotho/audiocaps)
            show_progress: Show progress bars

        Returns:
            Dictionary with evaluation results
        """
        print("=" * 60)
        print(f"Negative Query Evaluation ({self.model_name})")
        print("=" * 60)
        print(f"Dataset: {dataset_type}")
        print(f"Queries: {negative_queries_jsonl}")
        print("=" * 60)

        # Load items
        if dataset_type.lower() == "clotho":
            items = self.load_clotho_items(captions_csv, audio_dir)
        elif dataset_type.lower() == "audiocaps":
            items = self.load_audiocaps_items(captions_csv, audio_dir)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        print(f"Loaded {len(items)} audio items")

        # Encode audio
        audio_paths = [str(item.audio_path) for item in items]
        clip_ids = [item.clip_id for item in items]

        audio_embeddings_list = []
        iterator = range(0, len(audio_paths), self.batch_size_audio)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding audio", unit="batch")

        for i in iterator:
            batch = audio_paths[i:i + self.batch_size_audio]
            batch_embeds = self.adapter.encode_audio(
                batch,
                batch_size=self.batch_size_audio,
                device=self.device,
            )
            audio_embeddings_list.append(batch_embeds)

        audio_embeddings = np.concatenate(audio_embeddings_list, axis=0)
        print(f"Audio embeddings: {audio_embeddings.shape}")

        # Load and evaluate negative queries
        queries = self.load_negative_queries(negative_queries_jsonl)
        print(f"Loaded {len(queries)} negative queries")

        results = self.evaluate_negative_queries(
            audio_embeddings,
            clip_ids,
            queries,
            show_progress,
        )

        print()
        print("Results:")
        print(f"  Standard:  {format_metrics(results['standard_metrics'])}")
        print(f"  Adjusted:  {format_metrics(results['adjusted_metrics'])}")
        print(f"  Queries:   {results['num_queries']}")
        print(f"  Avg rank improvement: {results['avg_rank_improvement']:.2f}")

        return results
