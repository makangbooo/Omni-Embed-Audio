"""Strict dependency-light records shared by the ASR reranking pipeline."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple


def nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


@dataclass(frozen=True)
class CorpusDocument:
    document_id: str
    title: str
    text: str
    constructed_text: str

    def __post_init__(self) -> None:
        nonempty_string(self.document_id, field="document_id")
        if not isinstance(self.title, str) or not isinstance(self.text, str):
            raise TypeError("title and text must be strings")
        if not isinstance(self.constructed_text, str):
            raise TypeError("constructed_text must be a string")


@dataclass(frozen=True)
class TextQuery:
    query_id: str
    text: str

    def __post_init__(self) -> None:
        nonempty_string(self.query_id, field="query_id")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")


@dataclass(frozen=True)
class AudioQuery:
    record_id: str
    query_id: str
    dataset: str
    subset: str
    condition: str
    audio_path: str
    original_query_text: str
    sample_rate: Optional[int] = None
    duration_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        for field in (
            "record_id",
            "query_id",
            "dataset",
            "subset",
            "condition",
            "audio_path",
        ):
            nonempty_string(getattr(self, field), field=field)
        if not isinstance(self.original_query_text, str):
            raise TypeError("original_query_text must be a string")
        if self.sample_rate is not None and (
            isinstance(self.sample_rate, bool)
            or not isinstance(self.sample_rate, int)
            or self.sample_rate <= 0
        ):
            raise ValueError("sample_rate must be a positive integer or null")
        if self.duration_seconds is not None:
            duration = finite_float(self.duration_seconds, field="duration_seconds")
            if duration <= 0.0:
                raise ValueError("duration_seconds must be positive")


@dataclass(frozen=True)
class NBestHypothesis:
    rank: int
    text: str
    sequence_score: Optional[float]
    average_token_logprob: float
    valid_token_count: int

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("rank must be a positive integer")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        finite_float(self.average_token_logprob, field="average_token_logprob")
        if self.sequence_score is not None:
            finite_float(self.sequence_score, field="sequence_score")
        if (
            isinstance(self.valid_token_count, bool)
            or not isinstance(self.valid_token_count, int)
            or self.valid_token_count <= 0
        ):
            raise ValueError("valid_token_count must be a positive integer")


@dataclass(frozen=True)
class CandidateFeatureRow:
    query_id: str
    document_id: str
    relevance: float
    oea_score: float
    asr_score: float
    features: Tuple[float, ...]
    feature_names: Tuple[str, ...]

    def __post_init__(self) -> None:
        nonempty_string(self.query_id, field="query_id")
        nonempty_string(self.document_id, field="document_id")
        relevance = finite_float(self.relevance, field="relevance")
        if relevance < 0.0:
            raise ValueError("relevance must be non-negative")
        finite_float(self.oea_score, field="oea_score")
        finite_float(self.asr_score, field="asr_score")
        if len(self.features) != len(self.feature_names) or not self.features:
            raise ValueError("features and feature_names must have the same non-zero length")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique")
        for index, value in enumerate(self.features):
            finite_float(value, field=f"features[{index}]")

    def feature_dict(self) -> Dict[str, float]:
        return dict(zip(self.feature_names, self.features))

    def as_dict(self) -> Dict[str, object]:
        value = asdict(self)
        value["features"] = list(self.features)
        value["feature_names"] = list(self.feature_names)
        return value


def validate_qrels_mapping(
    qrels: Mapping[str, Mapping[str, Real]],
) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for query_id, documents in qrels.items():
        normalized_query_id = nonempty_string(query_id, field="qrels query_id")
        if not isinstance(documents, Mapping) or not documents:
            raise ValueError(f"qrels[{query_id!r}] must be a non-empty mapping")
        normalized_documents: Dict[str, float] = {}
        for document_id, raw_relevance in documents.items():
            normalized_document_id = nonempty_string(
                document_id,
                field=f"qrels[{query_id!r}] document_id",
            )
            relevance = finite_float(
                raw_relevance,
                field=f"qrels[{query_id!r}][{document_id!r}]",
            )
            if relevance < 0.0:
                raise ValueError("qrels relevance must be non-negative")
            if normalized_document_id in normalized_documents:
                raise ValueError("duplicate qrels document ID")
            normalized_documents[normalized_document_id] = relevance
        if not any(value > 0.0 for value in normalized_documents.values()):
            raise ValueError(f"query {query_id!r} has no positive qrel")
        if normalized_query_id in result:
            raise ValueError("duplicate qrels query ID")
        result[normalized_query_id] = normalized_documents
    if not result:
        raise ValueError("qrels must not be empty")
    return result


def require_existing_file(path: Path | str, *, field: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{field} is not a regular file: {resolved}")
    return resolved
