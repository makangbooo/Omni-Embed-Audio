"""
UIQ (User Intent Query) Evaluation Runner.

Evaluates retrieval using user-intent queries instead of captions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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


class UIQRunner(BaselineRunner):
    """
    UIQ (User Intent Query) evaluation runner.

    Evaluates UIQ-to-audio retrieval using precomputed or on-the-fly embeddings.

    Args:
        model_name: Name of the model to use
        device: Device for inference
        uiq_categories: List of UIQ categories to evaluate
        batch_size_audio: Batch size for audio encoding
        batch_size_text: Batch size for text encoding

    Example:
        >>> runner = UIQRunner(model_name="laion_clap", device="cuda")
        >>> results = runner.run(
        ...     audio_dir=Path("clotho/audio"),
        ...     captions_csv=Path("clotho/evaluation.csv"),
        ...     uiq_jsonl=Path("uiq/clotho_eval.jsonl"),
        ...     dataset_type="clotho",
        ... )
    """

    DEFAULT_UIQ_CATEGORIES = [
        "Pragmatic",
        "Subjective",
        "Contextual",
        "Negative/Contrastive",
        "Cross-domain",
    ]

    def __init__(
        self,
        model_name: str = "laion_clap",
        device: str = "cuda",
        uiq_categories: Optional[List[str]] = None,
        batch_size_audio: int = 64,
        batch_size_text: int = 256,
        **model_kwargs,
    ):
        super().__init__(model_name, device, batch_size_audio, batch_size_text, **model_kwargs)
        self.uiq_categories = uiq_categories or self.DEFAULT_UIQ_CATEGORIES

    def load_uiq_queries(
        self,
        jsonl_path: Path,
        valid_clip_ids: Optional[Set[str]] = None,
    ) -> Dict[str, Dict[str, str]]:
        """
        Load UIQ queries from JSONL file.

        Args:
            jsonl_path: Path to UIQ JSONL file
            valid_clip_ids: Optional set of valid clip IDs to filter

        Returns:
            Dictionary {category: {clip_id: query}}
        """
        uiq_data: Dict[str, Dict[str, str]] = {}

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                item = json.loads(line)
                clip_id = item.get("clip_id", "")

                if valid_clip_ids and clip_id not in valid_clip_ids:
                    continue

                for uiq_item in item.get("uiq", []):
                    bucket = uiq_item.get("bucket", "unknown")
                    query = uiq_item.get("query", "")

                    if bucket not in uiq_data:
                        uiq_data[bucket] = {}
                    uiq_data[bucket][clip_id] = query

        return uiq_data

    def evaluate_uiq_category(
        self,
        audio_embeddings: np.ndarray,
        clip_ids: List[str],
        queries: Dict[str, str],
        show_progress: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate a single UIQ category.

        Args:
            audio_embeddings: Audio embeddings [N_audio, D]
            clip_ids: List of clip IDs corresponding to audio embeddings
            queries: Dictionary mapping clip_id to query text
            show_progress: Show progress bar

        Returns:
            Dictionary of metrics
        """
        # Filter to only clips with queries
        clip_id_to_idx = {cid: idx for idx, cid in enumerate(clip_ids)}
        valid_pairs = [(cid, q) for cid, q in queries.items() if cid in clip_id_to_idx]

        if not valid_pairs:
            return {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "MRR": 0.0, "DCG": 0.0}

        query_texts = [q for _, q in valid_pairs]
        gt_indices = [clip_id_to_idx[cid] for cid, _ in valid_pairs]

        # Encode queries
        text_embeddings_list = []
        iterator = range(0, len(query_texts), self.batch_size_text)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding UIQ", unit="batch", leave=False)

        for i in iterator:
            batch = query_texts[i:i + self.batch_size_text]
            batch_embeds = self.adapter.encode_text(
                batch,
                batch_size=self.batch_size_text,
                device=self.device,
            )
            text_embeddings_list.append(batch_embeds)

        text_embeddings = np.concatenate(text_embeddings_list, axis=0)

        # Compute similarities and ranks
        audio_norm = l2norm(audio_embeddings)
        text_norm = l2norm(text_embeddings)
        sim_matrix = cosine_sim(text_norm, audio_norm)

        ranks = []
        for i, gt_idx in enumerate(gt_indices):
            rank = ranks_from_scores(sim_matrix[i], gt_idx)
            ranks.append(rank)

        return compute_all_metrics(np.array(ranks))

    def run(
        self,
        audio_dir: Path,
        captions_csv: Path,
        uiq_jsonl: Path,
        dataset_type: str = "clotho",
        show_progress: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run UIQ evaluation.

        Args:
            audio_dir: Directory containing audio files
            captions_csv: Path to captions CSV
            uiq_jsonl: Path to UIQ JSONL file
            dataset_type: Dataset type (clotho/audiocaps)
            show_progress: Show progress bars

        Returns:
            Dictionary with metrics for each UIQ category
        """
        print("=" * 60)
        print(f"UIQ Retrieval Evaluation ({self.model_name})")
        print("=" * 60)
        print(f"Dataset: {dataset_type}")
        print(f"UIQ JSONL: {uiq_jsonl}")
        print(f"Categories: {self.uiq_categories}")
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

        # Load UIQ queries
        valid_clip_ids = set(clip_ids)
        uiq_data = self.load_uiq_queries(uiq_jsonl, valid_clip_ids)

        # Evaluate each category
        results = {}
        for category in self.uiq_categories:
            if category not in uiq_data:
                print(f"Warning: UIQ category '{category}' not found")
                continue

            queries = uiq_data[category]
            print(f"\nEvaluating '{category}' ({len(queries)} queries)...")

            metrics = self.evaluate_uiq_category(
                audio_embeddings,
                clip_ids,
                queries,
                show_progress,
            )
            results[category] = metrics
            print(f"  {format_metrics(metrics)}")

        # Compute average across categories
        if results:
            avg_metrics = {}
            for key in ["R@1", "R@5", "R@10", "MRR", "DCG"]:
                values = [m[key] for m in results.values() if key in m]
                avg_metrics[key] = sum(values) / len(values) if values else 0.0
            results["Average"] = avg_metrics
            print(f"\nAverage: {format_metrics(avg_metrics)}")

        return results
