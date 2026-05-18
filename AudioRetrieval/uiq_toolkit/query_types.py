"""
Query type definitions and output containers for the standalone UIQ toolkit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class QueryType(Enum):
    """Supported user-intent query types."""

    QUESTION = "question"
    IMPERATIVE = "imperative"
    PARAPHRASE = "paraphrase"
    NEGATIVE = "negative"
    TAGGING = "tagging"

    @classmethod
    def choices(cls) -> list[str]:
        """Return CLI-friendly string choices."""
        return [item.value for item in cls]

    @classmethod
    def from_string(cls, value: str) -> "QueryType":
        """Parse a query type from a string."""
        normalized = value.lower().strip()
        for item in cls:
            if item.value == normalized or item.name.lower() == normalized:
                return item
        raise ValueError(f"Unknown query type: {value}")


LEGACY_BUCKET_BY_QUERY_TYPE = {
    QueryType.QUESTION: "Pragmatic",
    QueryType.PARAPHRASE: "Subjective",
    QueryType.IMPERATIVE: "Contextual",
    QueryType.NEGATIVE: "Negative/Contrastive",
    QueryType.TAGGING: "Cross-domain",
}


def legacy_bucket_for_query_type(query_type: QueryType) -> str:
    """Return the legacy bucket name used by the existing evaluator."""
    return LEGACY_BUCKET_BY_QUERY_TYPE[query_type]


@dataclass
class QueryRecord:
    """
    A single generated query record.

    Flat JSONL output is written directly from this object, while grouped JSONL
    is assembled by clip ID from multiple records.
    """

    clip_id: str
    original_caption: str
    query_type: QueryType
    query: str
    hard_negative_caption: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to the flat JSONL schema."""
        payload: Dict[str, Any] = {
            "clip_id": self.clip_id,
            "original_caption": self.original_caption,
            "query_type": self.query_type.value,
            "query": self.query,
        }
        if self.hard_negative_caption:
            payload["hard_negative_caption"] = self.hard_negative_caption
        payload.update(self.metadata)
        return payload
