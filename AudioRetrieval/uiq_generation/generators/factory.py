"""
UIQ Generator Factory.

Provides a unified interface for creating UIQ generators.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from AudioRetrieval.uiq_generation.generators.base import BaseUIQGenerator
from AudioRetrieval.uiq_generation.generators.gpt_generator import GPTUIQGenerator
from AudioRetrieval.uiq_generation.generators.llama_generator import LlamaUIQGenerator


def UIQGenerator(
    backend: str = "gpt",
    **kwargs,
) -> BaseUIQGenerator:
    """
    Factory function to create a UIQ generator.

    Args:
        backend: Generator backend ("gpt" or "llama")
        **kwargs: Backend-specific arguments

    Returns:
        UIQ generator instance

    Examples:
        >>> # Create GPT generator
        >>> generator = UIQGenerator("gpt", model="gpt-4")

        >>> # Create LLaMA generator
        >>> generator = UIQGenerator("llama", model_name="meta-llama/Llama-2-7b-chat-hf")

    Backend-specific arguments:
        GPT:
            - model: OpenAI model name (default: gpt-4)
            - api_key: OpenAI API key

        LLaMA:
            - model_name: HuggingFace model name or path
            - device: Device for inference
            - torch_dtype: Torch dtype

        Common:
            - batch_size: Batch size for generation
            - max_tokens: Maximum tokens
            - temperature: Sampling temperature
    """
    backend = backend.lower().strip()

    if backend == "gpt":
        return GPTUIQGenerator(**kwargs)
    elif backend == "llama":
        return LlamaUIQGenerator(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose from: gpt, llama")


# Convenience aliases
BACKEND_ALIASES = {
    "openai": "gpt",
    "gpt-4": "gpt",
    "gpt4": "gpt",
    "gpt-3.5": "gpt",
    "chatgpt": "gpt",
    "llama2": "llama",
    "llama-2": "llama",
    "llama3": "llama",
    "llama-3": "llama",
}


def resolve_backend(name: str) -> str:
    """Resolve backend alias to canonical name."""
    name = name.lower().strip()
    return BACKEND_ALIASES.get(name, name)
