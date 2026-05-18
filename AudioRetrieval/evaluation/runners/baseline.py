"""
Baseline Retrieval Evaluation Runner.

Evaluates standard caption-to-audio and audio-to-caption retrieval.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from AudioRetrieval.evaluation.metrics import (
    l2norm,
    cosine_sim,
    ranks_from_scores,
    compute_all_metrics,
    format_metrics,
)


@dataclass
class EvalItem:
    """Single evaluation item."""
    clip_id: str
    audio_path: Path
    captions: List[str]


class BaselineRunner:
    """
    Baseline retrieval evaluation runner.

    Evaluates text-to-audio (T2A) and audio-to-text (A2T) retrieval.

    Args:
        model_name: Name of the model to use
        device: Device for inference
        batch_size_audio: Batch size for audio encoding
        batch_size_text: Batch size for text encoding

    Example:
        >>> runner = BaselineRunner(model_name="laion_clap", device="cuda")
        >>> results = runner.run(
        ...     audio_dir=Path("clotho/audio"),
        ...     captions_csv=Path("clotho/evaluation.csv"),
        ...     dataset_type="clotho",
        ... )
    """

    def __init__(
        self,
        model_name: str = "laion_clap",
        device: str = "cuda",
        batch_size_audio: int = 64,
        batch_size_text: int = 256,
        **model_kwargs,
    ):
        self.model_name = model_name.lower()
        self.device = device
        self.batch_size_audio = batch_size_audio
        self.batch_size_text = batch_size_text
        self.model_kwargs = model_kwargs
        self._adapter = None

    def _load_adapter(self):
        """Lazy-load model adapter."""
        if self.model_name == "laion_clap":
            from AudioRetrieval.models.laion_clap_adapter import LaionClapAdapter
            return LaionClapAdapter(
                ckpt_path=self.model_kwargs.get("ckpt_path"),
                amodel=self.model_kwargs.get("amodel", "HTSAT-tiny"),
                tmodel=self.model_kwargs.get("tmodel", "roberta"),
                enable_fusion=self.model_kwargs.get("enable_fusion", False),
            )
        elif self.model_name == "mga_clap":
            from AudioRetrieval.models.mga_clap_adapter import MGAClapAdapter
            return MGAClapAdapter(
                repo_path=self.model_kwargs.get("repo_path"),
                ckpt_path=self.model_kwargs.get("ckpt_path"),
                seconds=self.model_kwargs.get("seconds", 10.0),
                device=self.device,
            )
        elif self.model_name == "wavcaps":
            from AudioRetrieval.models.wavcaps_adapter import WavCapsAdapter
            return WavCapsAdapter(
                ckpt_path=self.model_kwargs.get("ckpt_path"),
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    @property
    def adapter(self):
        """Get or create model adapter."""
        if self._adapter is None:
            self._adapter = self._load_adapter()
        return self._adapter

    def load_clotho_items(
        self,
        captions_csv: Path,
        audio_dir: Path,
    ) -> List[EvalItem]:
        """Load Clotho evaluation items."""
        items = []
        with open(captions_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("file_name", "").strip()
                if not filename:
                    continue

                audio_path = audio_dir / filename
                if not audio_path.exists():
                    continue

                captions = []
                for i in range(1, 6):
                    caption = row.get(f"caption_{i}", "").strip()
                    if caption:
                        captions.append(caption)

                if captions:
                    items.append(EvalItem(
                        clip_id=Path(filename).stem,
                        audio_path=audio_path,
                        captions=captions,
                    ))

        return items

    def load_audiocaps_items(
        self,
        captions_csv: Path,
        audio_dir: Path,
    ) -> List[EvalItem]:
        """Load AudioCaps evaluation items."""
        audio_to_captions: Dict[str, List[str]] = {}

        with open(captions_csv, "r", encoding="utf-8") as f:
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

        items = []
        for audio_key, captions in sorted(audio_to_captions.items()):
            audio_path = audio_dir / f"{audio_key}.wav"
            if audio_path.exists() and captions:
                items.append(EvalItem(
                    clip_id=audio_key,
                    audio_path=audio_path,
                    captions=captions,
                ))

        return items

    def encode_items(
        self,
        items: List[EvalItem],
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Encode audio and text for all items.

        Returns:
            Tuple of (audio_embeddings, text_embeddings, audio_indices)
            where audio_indices[i] is the audio index for caption i
        """
        # Encode audio
        audio_paths = [str(item.audio_path) for item in items]
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

        # Encode captions
        all_captions = []
        audio_indices = []
        for idx, item in enumerate(items):
            for caption in item.captions:
                all_captions.append(caption)
                audio_indices.append(idx)

        text_embeddings_list = []
        iterator = range(0, len(all_captions), self.batch_size_text)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding text", unit="batch")

        for i in iterator:
            batch = all_captions[i:i + self.batch_size_text]
            batch_embeds = self.adapter.encode_text(
                batch,
                batch_size=self.batch_size_text,
                device=self.device,
            )
            text_embeddings_list.append(batch_embeds)

        text_embeddings = np.concatenate(text_embeddings_list, axis=0)

        return audio_embeddings, text_embeddings, audio_indices

    def evaluate_t2a(
        self,
        audio_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        audio_indices: List[int],
    ) -> Dict[str, float]:
        """
        Evaluate text-to-audio retrieval.

        Args:
            audio_embeddings: Audio embeddings [N_audio, D]
            text_embeddings: Text embeddings [N_text, D]
            audio_indices: Audio index for each caption

        Returns:
            Dictionary of metrics
        """
        # Normalize embeddings
        audio_norm = l2norm(audio_embeddings)
        text_norm = l2norm(text_embeddings)

        # Compute similarities: [N_text, N_audio]
        sim_matrix = cosine_sim(text_norm, audio_norm)

        # Compute ranks
        ranks = []
        for i, gt_idx in enumerate(audio_indices):
            rank = ranks_from_scores(sim_matrix[i], gt_idx)
            ranks.append(rank)

        return compute_all_metrics(np.array(ranks))

    def evaluate_a2t(
        self,
        audio_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        audio_indices: List[int],
        num_captions_per_audio: int = 5,
    ) -> Dict[str, float]:
        """
        Evaluate audio-to-text retrieval.

        Args:
            audio_embeddings: Audio embeddings [N_audio, D]
            text_embeddings: Text embeddings [N_text, D]
            audio_indices: Audio index for each caption
            num_captions_per_audio: Expected captions per audio

        Returns:
            Dictionary of metrics
        """
        # Normalize embeddings
        audio_norm = l2norm(audio_embeddings)
        text_norm = l2norm(text_embeddings)

        # Compute similarities: [N_audio, N_text]
        sim_matrix = cosine_sim(audio_norm, text_norm)

        # For A2T, we need to find which captions belong to each audio
        # and count hits in top-K
        ranks = []
        for audio_idx in range(len(audio_embeddings)):
            # Find all captions for this audio
            gt_caption_indices = [i for i, a in enumerate(audio_indices) if a == audio_idx]
            if not gt_caption_indices:
                continue

            # Get best rank among all ground truth captions
            scores = sim_matrix[audio_idx]
            best_rank = float('inf')
            for gt_idx in gt_caption_indices:
                rank = ranks_from_scores(scores, gt_idx)
                best_rank = min(best_rank, rank)

            ranks.append(best_rank)

        return compute_all_metrics(np.array(ranks))

    def run(
        self,
        audio_dir: Path,
        captions_csv: Path,
        dataset_type: str = "clotho",
        show_progress: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run baseline evaluation.

        Args:
            audio_dir: Directory containing audio files
            captions_csv: Path to captions CSV
            dataset_type: Dataset type (clotho/audiocaps)
            show_progress: Show progress bars

        Returns:
            Dictionary with T2A and A2T metrics
        """
        print("=" * 60)
        print(f"Baseline Retrieval Evaluation ({self.model_name})")
        print("=" * 60)
        print(f"Dataset: {dataset_type}")
        print(f"Audio dir: {audio_dir}")
        print(f"Captions CSV: {captions_csv}")
        print("=" * 60)

        # Load items
        if dataset_type.lower() == "clotho":
            items = self.load_clotho_items(captions_csv, audio_dir)
        elif dataset_type.lower() == "audiocaps":
            items = self.load_audiocaps_items(captions_csv, audio_dir)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        print(f"Loaded {len(items)} items")

        # Encode
        audio_embeds, text_embeds, audio_indices = self.encode_items(items, show_progress)
        print(f"Audio embeddings: {audio_embeds.shape}")
        print(f"Text embeddings: {text_embeds.shape}")

        # Evaluate
        t2a_metrics = self.evaluate_t2a(audio_embeds, text_embeds, audio_indices)
        a2t_metrics = self.evaluate_a2t(audio_embeds, text_embeds, audio_indices)

        print()
        print("Results:")
        print(f"  T2A: {format_metrics(t2a_metrics)}")
        print(f"  A2T: {format_metrics(a2t_metrics)}")

        return {
            "text_to_audio": t2a_metrics,
            "audio_to_text": a2t_metrics,
        }
