"""
Base UIQ Generator.

Abstract base class for all UIQ generators.
"""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tqdm import tqdm

from AudioRetrieval.uiq_generation.query_types import QueryType, QueryResult


class BaseUIQGenerator(ABC):
    """
    Abstract base class for UIQ generators.

    Subclasses must implement:
        - _generate_single(): Generate a single query from a caption

    Attributes:
        batch_size: Number of prompts to process at once
        max_tokens: Maximum tokens for generation
        temperature: Sampling temperature
    """

    def __init__(
        self,
        batch_size: int = 10,
        max_tokens: int = 100,
        temperature: float = 0.7,
    ):
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.temperature = temperature

    @abstractmethod
    def _generate_single(
        self,
        caption: str,
        query_type: QueryType,
        hard_negative_caption: Optional[str] = None,
    ) -> str:
        """
        Generate a single query from a caption.

        Args:
            caption: Source caption
            query_type: Type of query to generate
            hard_negative_caption: Optional hard negative (for negative queries)

        Returns:
            Generated query string
        """
        pass

    def generate(
        self,
        captions: Sequence[str],
        query_type: QueryType,
        clip_ids: Optional[Sequence[str]] = None,
        hard_negative_captions: Optional[Sequence[str]] = None,
        show_progress: bool = True,
    ) -> List[QueryResult]:
        """
        Generate queries for multiple captions.

        Args:
            captions: List of source captions
            query_type: Type of query to generate
            clip_ids: Optional list of clip IDs
            hard_negative_captions: Optional list of hard negatives
            show_progress: Show progress bar

        Returns:
            List of QueryResult objects
        """
        if clip_ids is None:
            clip_ids = [f"clip_{i}" for i in range(len(captions))]

        if hard_negative_captions is None:
            hard_negative_captions = [None] * len(captions)

        results = []
        iterator = range(len(captions))
        if show_progress:
            iterator = tqdm(iterator, desc=f"Generating {query_type.value}", unit="query")

        for i in iterator:
            try:
                query = self._generate_single(
                    caption=captions[i],
                    query_type=query_type,
                    hard_negative_caption=hard_negative_captions[i],
                )
                results.append(QueryResult(
                    clip_id=clip_ids[i],
                    original_caption=captions[i],
                    query_type=query_type,
                    query=query.strip(),
                ))
            except Exception as e:
                print(f"[WARN] Failed to generate query for clip {clip_ids[i]}: {e}")
                results.append(QueryResult(
                    clip_id=clip_ids[i],
                    original_caption=captions[i],
                    query_type=query_type,
                    query="",
                    metadata={"error": str(e)},
                ))

        return results

    def generate_all_types(
        self,
        captions: Sequence[str],
        clip_ids: Optional[Sequence[str]] = None,
        query_types: Optional[List[QueryType]] = None,
        hard_negative_captions: Optional[Sequence[str]] = None,
        show_progress: bool = True,
    ) -> Dict[QueryType, List[QueryResult]]:
        """
        Generate queries for all specified types.

        Args:
            captions: List of source captions
            clip_ids: Optional list of clip IDs
            query_types: List of query types (default: all non-negative)
            hard_negative_captions: Optional list of hard negatives
            show_progress: Show progress bar

        Returns:
            Dictionary mapping query type to results
        """
        if query_types is None:
            # Default to all types except negative (requires hard negatives)
            query_types = [
                QueryType.QUESTION,
                QueryType.IMPERATIVE,
                QueryType.PARAPHRASE,
                QueryType.TAGGING,
            ]
            if hard_negative_captions is not None:
                query_types.append(QueryType.NEGATIVE)

        results = {}
        for query_type in query_types:
            if query_type == QueryType.NEGATIVE and hard_negative_captions is None:
                print(f"[WARN] Skipping {query_type.value} - no hard negatives provided")
                continue

            results[query_type] = self.generate(
                captions=captions,
                query_type=query_type,
                clip_ids=clip_ids,
                hard_negative_captions=hard_negative_captions,
                show_progress=show_progress,
            )

        return results

    def save_results(
        self,
        results: List[QueryResult],
        output_path: Path,
        format: str = "jsonl",
    ) -> None:
        """
        Save generation results to file.

        Args:
            results: List of QueryResult objects
            output_path: Output file path
            format: Output format (jsonl or json)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result.to_dict()) + "\n")
        elif format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump([r.to_dict() for r in results], f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")

        print(f"[INFO] Saved {len(results)} results to {output_path}")

    @staticmethod
    def load_clotho_captions(
        csv_path: Path,
        caption_index: int = 1,
    ) -> tuple[List[str], List[str]]:
        """
        Load captions from Clotho CSV.

        Args:
            csv_path: Path to Clotho metadata CSV
            caption_index: Which caption to use (1-5)

        Returns:
            Tuple of (clip_ids, captions)
        """
        clip_ids = []
        captions = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                filename = row.get("file_name", "").strip()
                caption = row.get(f"caption_{caption_index}", "").strip()

                if filename and caption:
                    clip_ids.append(Path(filename).stem)
                    captions.append(caption)

        return clip_ids, captions

    @staticmethod
    def load_audiocaps_captions(
        csv_path: Path,
    ) -> tuple[List[str], List[str]]:
        """
        Load captions from AudioCaps CSV.

        Args:
            csv_path: Path to AudioCaps metadata CSV

        Returns:
            Tuple of (clip_ids, captions)
        """
        clip_ids = []
        captions = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                youtube_id = row.get("youtube_id", "").strip()
                start_time = row.get("start_time", "").strip()
                caption = row.get("caption", "").strip()

                if youtube_id and start_time and caption:
                    clip_ids.append(f"{youtube_id}_{start_time}")
                    captions.append(caption)

        return clip_ids, captions
