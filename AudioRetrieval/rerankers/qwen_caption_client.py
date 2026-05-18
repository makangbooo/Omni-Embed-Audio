"""Caption-focused Qwen LLM client supporting pointwise and listwise scoring."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import LALMClient
from .qwen_omni_client import _load_transformer_deps, _resolve_local_model_path


def _extract_float(text: str, default: float) -> float:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not matches:
        return default
    try:
        return float(matches[-1])
    except ValueError:
        return default


@dataclass
class _ListwiseStats:
    total_calls: int = 0
    parse_failures: int = 0
    errors: int = 0
    partial_orders: int = 0


class QwenCaptionLALMClient(LALMClient):
    """
    Qwen2.5-based caption reranking client that uses textual prompts only.

    Provides pointwise scoring (query-caption relevance) and listwise ordering.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        max_new_tokens: int = 64,
        system_prompt: str = (
            "You are an expert judge for audio retrieval captions. "
            "Rate how well a candidate caption matches a natural language query."
        ),
        caption_score_prompt_template: str = (
            "Query: {query}\n"
            "Caption: {caption}\n"
            'Provide a relevance score between 0 and 1 using the format SCORE:<value>.'
        ),
        default_score: float = 0.0,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        listwise_prompt_template: str = (
            "You will be given several captions describing audio clips and a retrieval query.\n"
            "Query: {query}\n\n"
            "Captions:\n{enumerated_captions}\n\n"
            "Respond with their order from most to least relevant using ORDER: i1,i2,... "
            "where each index refers to the caption above."
        ),
        listwise_response_pattern: str = r"ORDER\s*:?\s*\[?([0-9,\s]+)\]?",
    ) -> None:
        self._model_name = model_name or "Qwen/Qwen2.5-Omni-7B"
        (
            torch,
            AutoModelForCausalLM,
            AutoProcessor,
            AutoModel,
        ) = _load_transformer_deps(self._model_name, cache_dir, local_files_only)

        self._torch = torch
        self._dtype = self._resolve_dtype(dtype)
        self._device_map = device_map
        self._max_new_tokens = int(max_new_tokens)
        self._system_prompt = system_prompt
        self._caption_score_prompt_template = caption_score_prompt_template
        self._default_score = float(default_score)
        self._listwise_prompt_template = listwise_prompt_template
        self._listwise_response_regex = re.compile(listwise_response_pattern, re.IGNORECASE)

        model_path = _resolve_local_model_path(self._model_name, cache_dir)
        pretrained_source = str(model_path) if model_path is not None else self._model_name

        from transformers import AutoConfig  # noqa: WPS433

        self._config = AutoConfig.from_pretrained(
            pretrained_source,
            trust_remote_code=True,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        processor_loader = AutoProcessor
        self.processor = processor_loader.from_pretrained(
            pretrained_source,
            trust_remote_code=True,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self._dtype,
            "device_map": self._device_map,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        }

        try:
            self.model = AutoModelForCausalLM.from_pretrained(pretrained_source, **model_kwargs)
        except ValueError as exc:
            if "Unrecognized configuration class" in str(exc):
                warnings.warn(
                    "AutoModelForCausalLM does not recognize config; falling back to AutoModel. "
                    "Generation may fail if the model lacks generate().",
                    RuntimeWarning,
                )
                self.model = AutoModel.from_pretrained(pretrained_source, **model_kwargs)
            else:
                raise

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is None:
            eos_token = getattr(tokenizer, "eos_token", None)
            if eos_token is not None:
                tokenizer.pad_token = eos_token

        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None and pad_token_id is not None:
            generation_config.pad_token_id = pad_token_id
        model_config = getattr(self.model, "config", None)
        if model_config is not None and pad_token_id is not None:
            model_config.pad_token_id = pad_token_id

        if hasattr(self.model, "disable_talker"):
            try:
                self.model.disable_talker()
            except Exception:
                warnings.warn(
                    "Failed to disable Qwen talker module; ignoring.",
                    RuntimeWarning,
                )
        self.model.eval()

        self._pad_token_id = getattr(self.model.generation_config, "pad_token_id", None)
        self._eos_token_id = getattr(self.model.generation_config, "eos_token_id", None)

        self._listwise_stats = _ListwiseStats()
        self._last_rank_info: Optional[Dict[str, Any]] = None

    def _resolve_dtype(self, dtype: str):
        mapping = {
            "float32": self._torch.float32,
            "float16": self._torch.float16,
            "bf16": self._torch.bfloat16,
            "bfloat16": self._torch.bfloat16,
            "fp32": self._torch.float32,
            "fp16": self._torch.float16,
            "auto": self._torch.float32,
        }
        key = dtype.lower()
        if key not in mapping:
            raise ValueError(f"Unsupported dtype '{dtype}'.")
        return mapping[key]

    # ------------------------------------------------------------------ helpers
    def _build_score_messages(self, query: str, caption: str):
        user_text = self._caption_score_prompt_template.format(query=query, caption=caption)
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_text},
        ]

    def _build_listwise_messages(self, query: str, captions: List[str]) -> Tuple[List[Dict], str]:
        lines = [f"{idx}. {cap}" for idx, cap in enumerate(captions, start=1)]
        enumerated = "\n".join(lines)
        prompt_text = self._listwise_prompt_template.format(query=query, enumerated_captions=enumerated)
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt_text},
        ]
        return messages, prompt_text

    def _generate(self, messages: List[Dict]) -> str:
        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(
            text=prompt,
            return_tensors="pt",
            padding=True,
        )
        tensor_inputs = {}
        for key, value in inputs.items():
            tensor = value.to(self.model.device)
            if hasattr(tensor, "dtype") and tensor.dtype.is_floating_point:
                tensor = tensor.to(self._dtype)
            tensor_inputs[key] = tensor
        inputs = tensor_inputs

        with self._torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self._pad_token_id,
                eos_token_id=self._eos_token_id,
            )

        sequences = generation
        if hasattr(generation, "sequences"):
            sequences = generation.sequences
        elif isinstance(generation, tuple):
            sequences = generation[0]

        prompt_length = inputs["input_ids"].shape[-1]
        continuation = sequences[:, prompt_length:]
        response = self.processor.batch_decode(
            continuation,
            skip_special_tokens=True,
        )[0]
        return response

    def _parse_listwise_response(self, response: str, num_items: int) -> List[int]:
        regions = []
        match = self._listwise_response_regex.search(response)
        if match:
            regions.append(match.group(1))
        regions.append(response)
        indices: List[int] = []
        for region in regions:
            for token in re.findall(r"\d+", region):
                try:
                    value = int(token)
                except ValueError:
                    continue
                zero_idx = value - 1
                if 0 <= zero_idx < num_items and zero_idx not in indices:
                    indices.append(zero_idx)
            if indices:
                break
        return indices

    # ------------------------------------------------------------------ protocol
    def score_audio(self, query: str, audio_path) -> float:
        raise NotImplementedError("Audio scoring is not supported by QwenCaptionLALMClient.")

    def score_caption(self, query: str, caption: str) -> float:
        messages = self._build_score_messages(query, caption)
        response = self._generate(messages)
        value = _extract_float(response, self._default_score)
        return max(0.0, min(1.0, value))

    def rank_audio(self, query: str, candidates):
        raise NotImplementedError("Audio ranking is not supported by QwenCaptionLALMClient.")

    def rank_captions(self, query: str, captions: List[str]) -> List[int]:
        self._listwise_stats.total_calls += 1
        messages, prompt_text = self._build_listwise_messages(query, captions)
        try:
            response = self._generate(messages)
        except Exception as exc:
            self._listwise_stats.errors += 1
            warnings.warn(f"Listwise generation failed; falling back to pointwise. Error: {exc}", RuntimeWarning)
            return self._rank_pointwise(query, captions, fallback_reason="generation_error")

        order = self._parse_listwise_response(response, len(captions))
        if not order:
            self._listwise_stats.parse_failures += 1
            return self._rank_pointwise(
                query,
                captions,
                fallback_reason="parse_failure",
                prompt=prompt_text,
                response=response,
            )

        if len(order) < len(captions):
            self._listwise_stats.partial_orders += 1
            pointwise = self._rank_pointwise(query, captions, fallback_reason="partial_listwise")
            seen = set(order)
            for idx in pointwise:
                if idx not in seen:
                    order.append(idx)
        self._last_rank_info = {
            "prompt": prompt_text,
            "response": response,
            "parsed_order": order,
        }
        return order

    def _rank_pointwise(
        self,
        query: str,
        captions: List[str],
        fallback_reason: Optional[str] = None,
        prompt: Optional[str] = None,
        response: Optional[str] = None,
    ) -> List[int]:
        scored: List[Tuple[float, int]] = []
        for idx, caption in enumerate(captions):
            score = self.score_caption(query, caption)
            scored.append((score, idx))
        scored.sort(key=lambda item: item[0], reverse=True)
        order = [idx for _, idx in scored]
        self._last_rank_info = {
            "used_fallback": True,
            "fallback_reason": fallback_reason,
            "scores": {idx: float(score) for score, idx in scored},
        }
        if prompt is not None:
            self._last_rank_info["prompt"] = prompt
        if response is not None:
            self._last_rank_info["response"] = response
        return order

    def generate_caption(self, audio_path: Path) -> str:
        raise NotImplementedError("Caption generation is not supported by QwenCaptionLALMClient.")

    def get_last_rank_info(self) -> Optional[Dict]:
        return self._last_rank_info

    def get_listwise_stats(self) -> Dict[str, int]:
        return {
            "listwise_calls": self._listwise_stats.total_calls,
            "parse_failures": self._listwise_stats.parse_failures,
            "errors": self._listwise_stats.errors,
            "partial_orders": self._listwise_stats.partial_orders,
        }
