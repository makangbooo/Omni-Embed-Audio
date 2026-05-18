"""
LLaMA-based UIQ Generator.

Uses local LLaMA models via HuggingFace transformers for query generation.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from AudioRetrieval.uiq_generation.query_types import QueryType
from AudioRetrieval.uiq_generation.generators.base import BaseUIQGenerator
from AudioRetrieval.uiq_generation.generators.prompts import format_prompt, LLAMA_SYSTEM_PROMPT


class LlamaUIQGenerator(BaseUIQGenerator):
    """
    LLaMA-based UIQ generator.

    Uses local LLaMA models via HuggingFace transformers.

    Args:
        model_name: HuggingFace model name or local path
        device: Device for inference (cuda/cpu)
        torch_dtype: Torch dtype for model (default: float16)
        batch_size: Number of prompts to process at once
        max_tokens: Maximum tokens for generation
        temperature: Sampling temperature

    Example:
        >>> generator = LlamaUIQGenerator(
        ...     model_name="meta-llama/Llama-2-7b-chat-hf",
        ...     device="cuda",
        ... )
        >>> results = generator.generate(
        ...     captions=["A dog barking in the park"],
        ...     query_type=QueryType.QUESTION,
        ... )
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        device: str = "cuda",
        torch_dtype: str = "float16",
        batch_size: int = 10,
        max_tokens: int = 100,
        temperature: float = 0.7,
    ):
        super().__init__(batch_size, max_tokens, temperature)
        self.model_name = model_name
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype, torch.float16)
        self._pipeline = None

    @property
    def pipeline(self) -> Any:
        """Lazy-load the text generation pipeline."""
        if self._pipeline is None:
            try:
                from transformers import pipeline
                self._pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    torch_dtype=self.torch_dtype,
                    device_map="auto" if self.device == "cuda" else None,
                )
            except ImportError:
                raise ImportError(
                    "transformers package required. Install with: pip install transformers"
                )
        return self._pipeline

    def _generate_single(
        self,
        caption: str,
        query_type: QueryType,
        hard_negative_caption: Optional[str] = None,
    ) -> str:
        """Generate a single query using LLaMA."""
        prompt = format_prompt(query_type, caption, hard_negative_caption)

        # Format for LLaMA chat
        full_prompt = LLAMA_SYSTEM_PROMPT + prompt + " [/INST]"

        outputs = self.pipeline(
            full_prompt,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self.pipeline.tokenizer.eos_token_id,
        )

        # Extract generated text after the prompt
        generated = outputs[0]["generated_text"]
        response = generated[len(full_prompt):].strip()

        # Clean up response (remove any continuation markers)
        if "[/INST]" in response:
            response = response.split("[/INST]")[0]
        if "<s>" in response:
            response = response.split("<s>")[0]

        return response.strip()

    def generate_batch(
        self,
        captions: list[str],
        query_type: QueryType,
        hard_negative_captions: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Generate queries for a batch of captions.

        Args:
            captions: List of source captions
            query_type: Type of query to generate
            hard_negative_captions: Optional list of hard negatives

        Returns:
            List of generated queries
        """
        if hard_negative_captions is None:
            hard_negative_captions = [None] * len(captions)

        # Prepare all prompts
        prompts = []
        for caption, hard_neg in zip(captions, hard_negative_captions):
            prompt = format_prompt(query_type, caption, hard_neg)
            full_prompt = LLAMA_SYSTEM_PROMPT + prompt + " [/INST]"
            prompts.append(full_prompt)

        # Batch generate
        try:
            outputs = self.pipeline(
                prompts,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                batch_size=self.batch_size,
                pad_token_id=self.pipeline.tokenizer.eos_token_id,
            )

            results = []
            for output, prompt in zip(outputs, prompts):
                generated = output[0]["generated_text"]
                response = generated[len(prompt):].strip()

                # Clean up
                if "[/INST]" in response:
                    response = response.split("[/INST]")[0]
                if "<s>" in response:
                    response = response.split("<s>")[0]

                results.append(response.strip())

            return results

        except Exception as e:
            print(f"[WARN] Batch generation failed: {e}")
            # Fall back to individual generation
            return [
                self._generate_single(c, query_type, h)
                for c, h in zip(captions, hard_negative_captions)
            ]
