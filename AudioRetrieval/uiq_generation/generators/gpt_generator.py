"""
GPT-based UIQ Generator.

Uses OpenAI GPT models for query generation.
"""

from __future__ import annotations

import os
from typing import Optional

from AudioRetrieval.uiq_generation.query_types import QueryType
from AudioRetrieval.uiq_generation.generators.base import BaseUIQGenerator
from AudioRetrieval.uiq_generation.generators.prompts import format_prompt, GPT_SYSTEM_PROMPT


class GPTUIQGenerator(BaseUIQGenerator):
    """
    GPT-based UIQ generator.

    Uses OpenAI API for query generation.

    Args:
        model: OpenAI model name (default: gpt-4)
        api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
        batch_size: Number of prompts to process at once
        max_tokens: Maximum tokens for generation
        temperature: Sampling temperature

    Example:
        >>> generator = GPTUIQGenerator(model="gpt-4")
        >>> results = generator.generate(
        ...     captions=["A dog barking in the park"],
        ...     query_type=QueryType.QUESTION,
        ... )
    """

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        batch_size: int = 10,
        max_tokens: int = 100,
        temperature: float = 0.7,
    ):
        super().__init__(batch_size, max_tokens, temperature)
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    @property
    def client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
        return self._client

    def _generate_single(
        self,
        caption: str,
        query_type: QueryType,
        hard_negative_caption: Optional[str] = None,
    ) -> str:
        """Generate a single query using GPT."""
        prompt = format_prompt(query_type, caption, hard_negative_caption)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": GPT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            n=1,
        )

        return response.choices[0].message.content.strip()

    def generate_batch(
        self,
        captions: list[str],
        query_type: QueryType,
        hard_negative_captions: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Generate queries for a batch of captions.

        Uses async batch processing for efficiency.

        Args:
            captions: List of source captions
            query_type: Type of query to generate
            hard_negative_captions: Optional list of hard negatives

        Returns:
            List of generated queries
        """
        if hard_negative_captions is None:
            hard_negative_captions = [None] * len(captions)

        results = []
        for caption, hard_neg in zip(captions, hard_negative_captions):
            try:
                query = self._generate_single(caption, query_type, hard_neg)
                results.append(query)
            except Exception as e:
                print(f"[WARN] Generation failed: {e}")
                results.append("")

        return results
