"""
LLM backends for standalone UIQ generation.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Sequence

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

from uiq_toolkit.io_utils import SourceExample
from uiq_toolkit.prompts import GPT_SYSTEM_PROMPT, LLAMA_SYSTEM_PROMPT, format_prompt
from uiq_toolkit.query_types import QueryRecord, QueryType


class BaseUIQGenerator(ABC):
    """Abstract generator interface shared by all backends."""

    def __init__(
        self,
        batch_size: int = 10,
        max_tokens: int = 100,
        temperature: float = 0.7,
    ) -> None:
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
        """Generate one query from one caption."""

    def _process_one(
        self,
        example: SourceExample,
        query_type: QueryType,
        hard_negative_captions: Optional[Dict[str, str]],
    ) -> QueryRecord:
        """Process a single example (used by both sequential and parallel paths)."""
        hard_negative_caption = None
        if query_type == QueryType.NEGATIVE:
            hard_negative_caption = (hard_negative_captions or {}).get(example.clip_id)
            if not hard_negative_caption:
                return QueryRecord(
                    clip_id=example.clip_id,
                    original_caption=example.caption,
                    query_type=query_type,
                    query="",
                    metadata={"error": "missing_hard_negative_caption"},
                )

        try:
            query = self._generate_single(
                caption=example.caption,
                query_type=query_type,
                hard_negative_caption=hard_negative_caption,
            ).strip()
            return QueryRecord(
                clip_id=example.clip_id,
                original_caption=example.caption,
                query_type=query_type,
                query=query,
                hard_negative_caption=hard_negative_caption,
            )
        except Exception as exc:
            return QueryRecord(
                clip_id=example.clip_id,
                original_caption=example.caption,
                query_type=query_type,
                query="",
                hard_negative_caption=hard_negative_caption,
                metadata={"error": str(exc)},
            )

    def generate(
        self,
        examples: Sequence[SourceExample],
        query_type: QueryType,
        hard_negative_captions: Optional[Dict[str, str]] = None,
        show_progress: bool = True,
        max_workers: int = 1,
    ) -> list[QueryRecord]:
        """Generate one query type for all examples.

        Args:
            max_workers: Number of concurrent threads for API calls.
                         Set >1 to enable parallel generation.
        """
        if max_workers <= 1:
            # Sequential path (original behavior)
            results: list[QueryRecord] = []
            iterator = examples
            if show_progress:
                iterator = tqdm(examples, desc=f"Generating {query_type.value}", unit="query")
            for example in iterator:
                results.append(self._process_one(example, query_type, hard_negative_captions))
            return results

        # Parallel path
        results_indexed: dict[int, QueryRecord] = {}
        pbar = tqdm(total=len(examples), desc=f"Generating {query_type.value}", unit="query") if show_progress else None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._process_one, ex, query_type, hard_negative_captions): idx
                for idx, ex in enumerate(examples)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results_indexed[idx] = future.result()
                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()

        return [results_indexed[i] for i in range(len(examples))]


class GPTUIQGenerator(BaseUIQGenerator):
    """OpenAI-backed generator."""

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        batch_size: int = 10,
        max_tokens: int = 100,
        temperature: float = 0.7,
    ) -> None:
        super().__init__(batch_size=batch_size, max_tokens=max_tokens, temperature=temperature)
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    @property
    def client(self):
        """Lazy-load the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "openai is required for the GPT backend. Install uiq_toolkit/requirements.txt."
                ) from exc
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def _generate_single(
        self,
        caption: str,
        query_type: QueryType,
        hard_negative_caption: Optional[str] = None,
    ) -> str:
        """Generate one query with an OpenAI chat model."""
        prompt = format_prompt(query_type, caption, hard_negative_caption)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": GPT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content or ""
        return content.strip()


class LlamaUIQGenerator(BaseUIQGenerator):
    """Local Hugging Face generator."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        device: str = "cuda",
        torch_dtype: str = "float16",
        batch_size: int = 10,
        max_tokens: int = 100,
        temperature: float = 0.7,
    ) -> None:
        super().__init__(batch_size=batch_size, max_tokens=max_tokens, temperature=temperature)
        self.model_name = model_name
        self.device = device
        self.torch_dtype_name = torch_dtype
        self._pipeline = None

    @property
    def pipeline(self):
        """Lazy-load the text-generation pipeline."""
        if self._pipeline is None:
            try:
                import torch
                from transformers import pipeline
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for the llama backend. "
                    "Install uiq_toolkit/requirements.txt."
                ) from exc

            torch_dtype = getattr(torch, self.torch_dtype_name, torch.float16)
            self._pipeline = pipeline(
                task="text-generation",
                model=self.model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
            )
        return self._pipeline

    def _generate_single(
        self,
        caption: str,
        query_type: QueryType,
        hard_negative_caption: Optional[str] = None,
    ) -> str:
        """Generate one query with a local LLaMA-style model."""
        prompt = format_prompt(query_type, caption, hard_negative_caption)
        full_prompt = LLAMA_SYSTEM_PROMPT + prompt + " [/INST]"

        outputs = self.pipeline(
            full_prompt,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            pad_token_id=self.pipeline.tokenizer.eos_token_id,
        )

        generated = outputs[0]["generated_text"]
        response = generated[len(full_prompt):].strip()
        if "[/INST]" in response:
            response = response.split("[/INST]", 1)[0]
        if "<s>" in response:
            response = response.split("<s>", 1)[0]
        return response.strip()


class TemplateUIQGenerator(BaseUIQGenerator):
    """
    Deterministic backend for smoke tests and prompt-pipeline debugging.

    This backend does not call an LLM. It is useful when collaborators want to
    verify data loading and output formats before spending API credits.
    """

    def _generate_single(
        self,
        caption: str,
        query_type: QueryType,
        hard_negative_caption: Optional[str] = None,
    ) -> str:
        normalized_caption = caption.strip().rstrip(".")
        if query_type == QueryType.QUESTION:
            return f"Can you find audio of {normalized_caption}?"
        if query_type == QueryType.IMPERATIVE:
            return f"Find audio of {normalized_caption}"
        if query_type == QueryType.PARAPHRASE:
            return f"Audio featuring {normalized_caption}"
        if query_type == QueryType.TAGGING:
            return f"Tagged audio of {normalized_caption}"
        if query_type == QueryType.NEGATIVE:
            if not hard_negative_caption:
                raise ValueError("Negative queries require a hard negative caption.")
            normalized_negative = hard_negative_caption.strip().rstrip(".")
            return f"Find audio of {normalized_caption}, not {normalized_negative}"
        raise ValueError(f"Unsupported query type: {query_type}")


BACKEND_ALIASES = {
    "openai": "gpt",
    "chatgpt": "gpt",
    "dummy": "template",
    "mock": "template",
}


def resolve_backend(name: str) -> str:
    """Resolve a backend alias to its canonical name."""
    normalized = name.lower().strip()
    return BACKEND_ALIASES.get(normalized, normalized)


def create_uiq_generator(backend: str = "gpt", **kwargs) -> BaseUIQGenerator:
    """Factory for UIQ generation backends."""
    backend = resolve_backend(backend)
    if backend == "gpt":
        return GPTUIQGenerator(**kwargs)
    if backend == "llama":
        return LlamaUIQGenerator(**kwargs)
    if backend == "template":
        return TemplateUIQGenerator(**kwargs)
    raise ValueError(f"Unknown backend: {backend}. Choose from gpt, llama, template.")
