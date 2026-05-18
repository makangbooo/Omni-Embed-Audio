"""
Query Type Definitions.

Defines the different types of user-intent queries and their properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class QueryType(Enum):
    """
    Types of user-intent queries.

    Each type represents a different way users might phrase
    their audio retrieval requests.
    """
    QUESTION = "question"
    IMPERATIVE = "imperative"
    PARAPHRASE = "paraphrase"
    NEGATIVE = "negative"
    TAGGING = "tagging"

    @classmethod
    def all_types(cls) -> List["QueryType"]:
        """Return all query types."""
        return list(cls)

    @classmethod
    def from_string(cls, s: str) -> "QueryType":
        """Parse query type from string."""
        s = s.lower().strip()
        for qt in cls:
            if qt.value == s or qt.name.lower() == s:
                return qt
        raise ValueError(f"Unknown query type: {s}")


@dataclass
class QueryResult:
    """
    Result of query generation for a single audio/caption.

    Attributes:
        clip_id: Unique identifier for the audio clip
        original_caption: Original caption text
        query_type: Type of generated query
        query: Generated query text
        metadata: Additional metadata
    """
    clip_id: str
    original_caption: str
    query_type: QueryType
    query: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "clip_id": self.clip_id,
            "original_caption": self.original_caption,
            "query_type": self.query_type.value,
            "query": self.query,
            **self.metadata,
        }


# Human-readable descriptions for each query type
QUERY_TYPE_DESCRIPTIONS = {
    QueryType.QUESTION: (
        "Polite retrieval requests phrased as questions. "
        "E.g., 'Can you find audio of a dog barking?'"
    ),
    QueryType.IMPERATIVE: (
        "Direct command-driven queries. "
        "E.g., 'Find recordings of rain falling on a roof'"
    ),
    QueryType.PARAPHRASE: (
        "Natural language rephrasing of the original caption. "
        "E.g., 'Audio with sounds of typing on a keyboard'"
    ),
    QueryType.NEGATIVE: (
        "Queries with exclusion conditions. "
        "E.g., 'Find crowd noise but not with music'"
    ),
    QueryType.TAGGING: (
        "Attribute-based queries focusing on audio properties. "
        "E.g., 'High-quality outdoor recording with birds'"
    ),
}


# Legacy bucket names mapping to QueryType
LEGACY_BUCKET_MAPPING = {
    "Pragmatic": QueryType.QUESTION,
    "Subjective": QueryType.PARAPHRASE,
    "Contextual": QueryType.IMPERATIVE,
    "Negative/Contrastive": QueryType.NEGATIVE,
    "Cross-domain": QueryType.TAGGING,
}


def legacy_bucket_to_query_type(bucket: str) -> QueryType:
    """Convert legacy bucket name to QueryType."""
    if bucket in LEGACY_BUCKET_MAPPING:
        return LEGACY_BUCKET_MAPPING[bucket]
    # Try direct matching
    return QueryType.from_string(bucket)
