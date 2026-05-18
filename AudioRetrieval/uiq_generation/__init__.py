"""
UIQ (User Intent Query) Generation Module.

This module provides tools for generating diverse user-intent queries
from audio captions using LLMs. Supports multiple query types:

1. Question queries: Polite retrieval requests in question form
2. Imperative queries: Direct command-driven queries
3. Paraphrase queries: Rephrased natural language queries
4. Negative queries: Queries with exclusion conditions
5. Tagging queries: Attribute-based queries

Submodules:
    - generators: LLM-based query generators (GPT, LLaMA)
    - query_types: Query type definitions and templates
    - cli: Command-line interface

Example usage:
    from AudioRetrieval.uiq_generation import UIQGenerator, QueryType

    generator = UIQGenerator(backend="gpt")
    queries = generator.generate(
        captions=["A dog barking in the park"],
        query_types=[QueryType.QUESTION, QueryType.IMPERATIVE],
    )
"""

from AudioRetrieval.uiq_generation.query_types import QueryType, QueryResult
from AudioRetrieval.uiq_generation.generators import (
    BaseUIQGenerator,
    GPTUIQGenerator,
    LlamaUIQGenerator,
    UIQGenerator,
)

__all__ = [
    "QueryType",
    "QueryResult",
    "BaseUIQGenerator",
    "GPTUIQGenerator",
    "LlamaUIQGenerator",
    "UIQGenerator",
]
