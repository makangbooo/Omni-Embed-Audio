"""
Base class for embedding precomputation.

Provides abstract interface and common utilities for all embedding precomputers.
"""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm


@dataclass
class EmbeddingResult:
    """Result of embedding computation."""
    embeddings: np.ndarray
    ids: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetEntry:
    """Single dataset entry with audio and optional captions."""
    clip_id: str
    audio_path: Optional[Path] = None
    captions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseEmbeddingPrecomputer(ABC):
    """
    Abstract base class for embedding precomputation.

    Subclasses must implement:
        - _load_model(): Initialize the embedding model
        - _encode_audio_batch(): Encode a batch of audio files
        - _encode_text_batch(): Encode a batch of text strings

    Attributes:
        device: Device for model inference (cuda/cpu)
        batch_size_audio: Batch size for audio encoding
        batch_size_text: Batch size for text encoding
    """

    def __init__(
        self,
        device: str = "cuda",
        batch_size_audio: int = 32,
        batch_size_text: int = 256,
    ):
        self.device = device
        self.batch_size_audio = batch_size_audio
        self.batch_size_text = batch_size_text
        self._model = None

    @abstractmethod
    def _load_model(self) -> Any:
        """Load and return the embedding model."""
        pass

    @abstractmethod
    def _encode_audio_batch(self, audio_paths: List[str]) -> np.ndarray:
        """Encode a batch of audio files. Returns [batch_size, embed_dim]."""
        pass

    @abstractmethod
    def _encode_text_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of text strings. Returns [batch_size, embed_dim]."""
        pass

    @property
    def model(self) -> Any:
        """Lazy-load and return the model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def encode_audio(
        self,
        audio_paths: Sequence[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode multiple audio files.

        Args:
            audio_paths: List of paths to audio files
            show_progress: Show progress bar

        Returns:
            Numpy array of shape [num_files, embed_dim]
        """
        embeddings_list = []
        iterator = range(0, len(audio_paths), self.batch_size_audio)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding audio", unit="batch")

        for i in iterator:
            batch = list(audio_paths[i:i + self.batch_size_audio])
            batch_embeds = self._encode_audio_batch(batch)
            embeddings_list.append(batch_embeds)

        return np.concatenate(embeddings_list, axis=0)

    def encode_text(
        self,
        texts: Sequence[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode multiple text strings.

        Args:
            texts: List of text strings
            show_progress: Show progress bar

        Returns:
            Numpy array of shape [num_texts, embed_dim]
        """
        embeddings_list = []
        iterator = range(0, len(texts), self.batch_size_text)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding text", unit="batch")

        for i in iterator:
            batch = list(texts[i:i + self.batch_size_text])
            batch_embeds = self._encode_text_batch(batch)
            embeddings_list.append(batch_embeds)

        return np.concatenate(embeddings_list, axis=0)

    def precompute_dataset(
        self,
        entries: Sequence[DatasetEntry],
        output_dir: Path,
        compute_audio: bool = True,
        compute_captions: bool = True,
    ) -> Dict[str, Path]:
        """
        Precompute embeddings for a dataset.

        Args:
            entries: List of DatasetEntry objects
            output_dir: Directory to save embeddings
            compute_audio: Whether to compute audio embeddings
            compute_captions: Whether to compute caption embeddings

        Returns:
            Dictionary mapping output type to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {}

        if compute_audio:
            audio_entries = [e for e in entries if e.audio_path and e.audio_path.exists()]
            if audio_entries:
                audio_paths = [str(e.audio_path) for e in audio_entries]
                clip_ids = [e.clip_id for e in audio_entries]
                filenames = [e.audio_path.name for e in audio_entries]

                print(f"Computing audio embeddings for {len(audio_paths)} files...")
                audio_embeddings = self.encode_audio(audio_paths)

                audio_output = output_dir / "audio_embeddings.npz"
                np.savez_compressed(
                    audio_output,
                    embeddings=audio_embeddings,
                    clip_ids=clip_ids,
                    filenames=filenames,
                )
                outputs["audio"] = audio_output
                print(f"Saved audio embeddings to {audio_output}")

        if compute_captions:
            caption_texts = []
            caption_clip_ids = []
            for e in entries:
                for caption in e.captions:
                    caption_texts.append(caption)
                    caption_clip_ids.append(e.clip_id)

            if caption_texts:
                print(f"Computing caption embeddings for {len(caption_texts)} captions...")
                caption_embeddings = self.encode_text(caption_texts)

                caption_output = output_dir / "caption_embeddings.npz"
                np.savez_compressed(
                    caption_output,
                    embeddings=caption_embeddings,
                    texts=caption_texts,
                    clip_ids=caption_clip_ids,
                )
                outputs["caption"] = caption_output
                print(f"Saved caption embeddings to {caption_output}")

        return outputs

    @staticmethod
    def load_clotho_entries(
        csv_path: Path,
        audio_dir: Path,
        split: str = "eval",
    ) -> List[DatasetEntry]:
        """
        Load Clotho dataset entries from CSV.

        Args:
            csv_path: Path to Clotho metadata CSV
            audio_dir: Directory containing audio files
            split: Dataset split name for clip_id prefix

        Returns:
            List of DatasetEntry objects
        """
        entries = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                filename = row.get("file_name", "").strip()
                if not filename:
                    continue

                audio_path = audio_dir / filename
                captions = []
                for i in range(1, 6):
                    caption = row.get(f"caption_{i}", "").strip()
                    if caption:
                        captions.append(caption)

                entries.append(DatasetEntry(
                    clip_id=f"clotho_{split}_{idx:04d}",
                    audio_path=audio_path if audio_path.exists() else None,
                    captions=captions,
                    metadata={"filename": filename, "index": idx},
                ))

        return entries

    @staticmethod
    def load_audiocaps_entries(
        csv_path: Path,
        audio_dir: Path,
        split: str = "test",
    ) -> List[DatasetEntry]:
        """
        Load AudioCaps dataset entries from CSV.

        Args:
            csv_path: Path to AudioCaps metadata CSV
            audio_dir: Directory containing audio files
            split: Dataset split name for clip_id prefix

        Returns:
            List of DatasetEntry objects (grouped by audio file)
        """
        # Group by audio file since AudioCaps has multiple captions per audio
        audio_to_captions: Dict[str, List[str]] = {}
        audio_to_metadata: Dict[str, Dict] = {}

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
                    audio_to_metadata[audio_key] = {
                        "youtube_id": youtube_id,
                        "start_time": start_time,
                    }

                if caption:
                    audio_to_captions[audio_key].append(caption)

        entries = []
        for audio_key, captions in sorted(audio_to_captions.items()):
            audio_path = audio_dir / f"{audio_key}.wav"
            entries.append(DatasetEntry(
                clip_id=f"audiocaps_{split}_{audio_key}",
                audio_path=audio_path if audio_path.exists() else None,
                captions=captions,
                metadata=audio_to_metadata[audio_key],
            ))

        return entries

    @staticmethod
    def load_uiq_queries(
        jsonl_path: Path,
        clip_id_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict[str, str]]:
        """
        Load UIQ queries from JSONL file.

        Args:
            jsonl_path: Path to UIQ JSONL file
            clip_id_mapping: Optional mapping from old to new clip IDs

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

                # Apply mapping if provided
                if clip_id_mapping and clip_id in clip_id_mapping:
                    clip_id = clip_id_mapping[clip_id]

                for uiq_item in item.get("uiq", []):
                    bucket = uiq_item.get("bucket", "unknown")
                    query = uiq_item.get("query", "")

                    if bucket not in uiq_data:
                        uiq_data[bucket] = {}
                    uiq_data[bucket][clip_id] = query

        return uiq_data
