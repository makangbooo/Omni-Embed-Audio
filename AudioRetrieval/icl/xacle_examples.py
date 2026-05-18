"""XACLE in-context example selection utilities.

Loads the XACLE dataset metadata and provides a simple bag-of-words
retrieval to select k examples for in-context prompting.

Assumptions
-----------
- Directory layout:
    <root>/wav/{train,validation}/<id>.wav
    <root>/meta_data/{train,validation}_average.csv

- CSV columns for average files: ``wav_file_name,text,average_score``

This module deliberately avoids heavy dependencies and network access.
"""

from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


@dataclass
class XACLEExample:
    audio_path: Path
    text: str
    score: float  # original scale (typically 0..10)


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [tok for tok in text.split() if tok]


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


class XACLEExampleSelector:
    """Lightweight example selector over XACLE metadata.

    Parameters
    ----------
    root : Path
        Root directory of the XACLE dataset.
    split : str
        Either ``"train"`` or ``"validation"``.
    use_average : bool
        If True, read ``*_average.csv`` (recommended). If False, raises for now.
    normalize_to_unit : bool
        Whether to convert scores to [0,1] by dividing by 10.
    seed : int
        RNG seed for deterministic random fallback.
    """

    def __init__(
        self,
        root: Path,
        split: str = "train",
        use_average: bool = True,
        normalize_to_unit: bool = True,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.use_average = bool(use_average)
        self.normalize_to_unit = bool(normalize_to_unit)
        self.rng = random.Random(seed)

        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")
        if not self.use_average:
            raise ValueError("use_average=False not supported; provide averaged CSV.")

        csv_path = self.root / "meta_data" / f"{split}_average.csv"
        audio_dir = self.root / "wav" / split
        if not csv_path.exists():
            raise FileNotFoundError(f"XACLE CSV not found: {csv_path}")
        if not audio_dir.is_dir():
            raise FileNotFoundError(f"XACLE audio dir not found: {audio_dir}")

        self.examples: List[XACLEExample] = []
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get("wav_file_name") or row.get("file") or ""
                text = row.get("text") or row.get("caption") or ""
                score_str = row.get("average_score") or row.get("score") or ""
                try:
                    score = float(score_str)
                except Exception:
                    continue
                apath = audio_dir / fname
                if not apath.exists():
                    # Skip missing audio files; guard against inconsistent dumps
                    continue
                if self.normalize_to_unit:
                    score = max(0.0, min(1.0, score / 10.0))
                self.examples.append(XACLEExample(audio_path=apath, text=text, score=score))

        # Precompute token bags for jaccard
        self._bags: List[List[str]] = [_tokenize(ex.text) for ex in self.examples]

    def __len__(self) -> int:
        return len(self.examples)

    def select(self, query: str, k: int, method: str = "jaccard") -> List[XACLEExample]:
        """Select k examples most relevant to a query.

        Methods supported: ``"jaccard"`` (default) or ``"random"``.
        """
        k = max(0, int(k))
        if k == 0 or not self.examples:
            return []

        if method == "random":
            return self.rng.sample(self.examples, k=min(k, len(self.examples)))

        # jaccard over bag-of-words as a robust, dependency-free baseline
        q_bag = _tokenize(query)
        scored: List[Tuple[float, int]] = []
        for i, bag in enumerate(self._bags):
            scored.append(( _jaccard(q_bag, bag), i ))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_idx = [i for _, i in scored[:k]]
        return [self.examples[i] for i in top_idx]

