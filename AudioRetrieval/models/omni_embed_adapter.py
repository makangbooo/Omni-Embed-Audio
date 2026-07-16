#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapter for NVIDIA Omni-Embed (Nemotron-3B) retrieval model.

This implementation follows the usage described in the official README:
inputs are formatted with the chat template, multimodal features are prepared
via the processor, and the final embedding is obtained by mean pooling the
last hidden-state with the attention mask.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM
try:
    from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
except ImportError:  # pragma: no cover
    Qwen2_5OmniThinkerForConditionalGeneration = None
import logging
from contextlib import contextmanager

from AudioRetrieval.eval_core import BaseRetrievalModel
from AudioRetrieval.models.chat_template_utils import (
    normalize_single_chat_template_output,
)


@dataclass(frozen=True)
class _BatchInputs:
    """Container for processor inputs."""
    text: List[str]
    audio_arrays: Optional[List[np.ndarray]] = None


def _resolve_device(requested: Optional[str]) -> torch.device:
    """Resolve the torch device, falling back to CPU when CUDA is unavailable."""
    raw = (requested or "").strip()
    target = raw.lower()
    if target.startswith("cuda"):
        if not torch.cuda.is_available():
            warnings.warn("CUDA requested but unavailable; falling back to CPU.", RuntimeWarning)
            return torch.device("cpu")
        try:
            _ = torch.cuda.device_count()  # trigger CUDA init early
        except Exception as exc:  # pragma: no cover - environment specific
            warnings.warn(
                f"CUDA requested but not usable ({exc!r}); falling back to CPU.",
                RuntimeWarning,
            )
            return torch.device("cpu")
        return torch.device(raw if raw else "cuda")
    return torch.device("cpu")


def _safe_audio_loader():
    """Return a callable that loads audio into mono float32 arrays."""
    try:
        import soundfile as sf  # type: ignore

        def _load_sf(path: str | Path, target_sr: int) -> np.ndarray:
            wav, sr = sf.read(str(path))
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != target_sr:
                try:
                    import librosa  # type: ignore

                    wav = librosa.resample(
                        wav,
                        orig_sr=sr,
                        target_sr=target_sr,
                        res_type="kaiser_best",
                    )
                except Exception:
                    try:
                        import torchaudio  # type: ignore

                        tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
                        wav = torchaudio.functional.resample(tensor, sr, target_sr)
                        wav = wav.squeeze(0).cpu().numpy()
                    except Exception as exc:
                        raise RuntimeError(
                            "Resampling failed – install librosa or torchaudio."
                        ) from exc
            return wav.astype(np.float32, copy=False)

        return _load_sf
    except Exception:
        pass

    try:
        import torchaudio  # type: ignore

        def _load_ta(path: str | Path, target_sr: int) -> np.ndarray:
            wav, sr = torchaudio.load(str(path))
            wav = wav.mean(dim=0)
            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)
            return wav.cpu().numpy().astype(np.float32, copy=False)

        return _load_ta
    except Exception:
        pass

    try:
        import librosa  # type: ignore

        def _load_lb(path: str | Path, target_sr: int) -> np.ndarray:
            wav, _ = librosa.load(str(path), sr=target_sr, mono=True)
            return wav.astype(np.float32, copy=False)

        return _load_lb
    except Exception as exc:  # pragma: no cover - optional deps
        raise RuntimeError(
            "Unable to load audio. Install one of: soundfile, torchaudio, or librosa."
        ) from exc


class OmniEmbedAdapter(BaseRetrievalModel):
    """
    Adapter exposing encode_text / encode_audio for Omni-Embed Nemotron-3B.
    """

    def __init__(
        self,
        repo_id: str,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        local_path: Optional[str] = None,
        trust_remote_code: bool = True,
        torch_dtype: Optional[str] = None,
        text_max_length: int = 512,
        device_map: Optional[str] = None,
        attn_implementation: Optional[str] = None,
        passage_prefix: str = "passage:",
        query_prefix: str = "query:",
        audio_max_length: Optional[int] = 2048000,  # From paper Table 7
    ):
        self.device = _resolve_device(device)
        self.repo_id = repo_id
        self.cache_dir = cache_dir
        self.local_path = local_path
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.attn_implementation = attn_implementation
        self.passage_prefix = passage_prefix.strip()
        self.query_prefix = query_prefix.strip()
        self.text_max_length = int(text_max_length)
        self.audio_max_length = audio_max_length

        model_path = local_path if local_path else repo_id
        load_kwargs = {
            "trust_remote_code": trust_remote_code,
            # A local directory is authoritative during reproducibility runs.  This
            # also makes an accidentally incomplete model fail instead of reaching
            # out to the Hub and silently changing the resource set.
            "local_files_only": bool(local_path),
        }
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        load_kwargs["low_cpu_mem_usage"] = True

        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        if device_map:
            load_kwargs["device_map"] = device_map
        else:
            if torch_dtype and torch_dtype.lower() != "auto":
                dtype_map = {
                    "bfloat16": torch.bfloat16,
                    "bf16": torch.bfloat16,
                    "float16": torch.float16,
                    "fp16": torch.float16,
                    "half": torch.float16,
                    "float32": torch.float32,
                    "fp32": torch.float32,
                }
                key = torch_dtype.lower()
                if key not in dtype_map:
                    raise ValueError(f"Unsupported torch_dtype '{torch_dtype}'.")
                # Transformers 4.52.4 consumes `torch_dtype` in
                # PreTrainedModel.from_pretrained.  Passing the newer `dtype`
                # spelling leaks into the model constructor in this pinned
                # environment and can fail before any weights are loaded.
                load_kwargs["torch_dtype"] = dtype_map[key]
            else:
                load_kwargs["torch_dtype"] = (
                    torch.bfloat16 if self.device.type == "cuda" else torch.float32
                )

        try:
            try:
                self.processor = AutoProcessor.from_pretrained(
                    model_path,
                    trust_remote_code=trust_remote_code,
                    cache_dir=cache_dir,
                    use_fast=False,
                    local_files_only=bool(local_path),
                )
            except TypeError:
                self.processor = AutoProcessor.from_pretrained(
                    model_path,
                    trust_remote_code=trust_remote_code,
                    cache_dir=cache_dir,
                    local_files_only=bool(local_path),
                )
        except OSError as exc:
            raise RuntimeError(
                f"Failed to load processor for '{model_path}'. "
                "Download the model first or configure HF_HOME for offline mode."
            ) from exc

        def _load_model(kwargs: dict):
            try:
                return AutoModel.from_pretrained(model_path, **kwargs)
            except ValueError as exc:
                if "Qwen2_5Omni" in str(exc):
                    if Qwen2_5OmniThinkerForConditionalGeneration is not None:
                        return Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_path, **kwargs)
                    return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
                raise

        try:
            self.model = _load_model(load_kwargs)
        except ImportError as exc:
            if attn_implementation and "flash_attn" in str(exc).lower():
                warnings.warn(
                    "FlashAttention2 not available; retrying Omni-Embed load without "
                    f"attn_implementation='{attn_implementation}'.",
                    RuntimeWarning,
                )
                retry_kwargs = dict(load_kwargs)
                retry_kwargs.pop("attn_implementation", None)
                self.model = _load_model(retry_kwargs)
                self.attn_implementation = None
            else:
                raise
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and self.device.type == "cuda":
                warnings.warn(
                    "CUDA out of memory while loading Omni-Embed; falling back to CPU. "
                    "Set model.device_map or free GPU memory for faster inference.",
                    RuntimeWarning,
                )
                self.device = torch.device("cpu")
                cpu_kwargs = dict(load_kwargs)
                cpu_kwargs.pop("attn_implementation", None)
                cpu_kwargs.pop("torch_dtype", None)
                cpu_kwargs["low_cpu_mem_usage"] = True
                cpu_kwargs["device_map"] = None
                cpu_kwargs["torch_dtype"] = torch.float32
                self.model = _load_model(cpu_kwargs)
            else:
                raise
        except OSError as exc:
            raise RuntimeError(
                f"Failed to load model weights for '{model_path}'. "
                "Ensure the checkpoint is available locally or accessible via network."
            ) from exc

        if not device_map:
            self.model = self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

        feature_extractor = getattr(self.processor, "feature_extractor", None)
        if feature_extractor is None:
            raise RuntimeError("Processor does not expose a feature_extractor for audio inputs.")

        self.audio_sampling_rate = int(getattr(feature_extractor, "sampling_rate", 16000))
        if self.audio_max_length is None:
            self.audio_max_length = getattr(self.model.config, "audio_max_length", None)

        self._audio_loader = _safe_audio_loader()

    # Convenience for PEFT/LoRA integration and inspection
    def get_underlying_model(self) -> AutoModel:
        """Return the wrapped HF model (for PEFT attachment / inspection)."""
        return self.model

    def set_underlying_model(self, new_model: torch.nn.Module) -> None:
        """Replace the wrapped model (e.g., with a PEFT/LoRA-wrapped version)."""
        self.model = new_model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _chunk_iter(self, seq: Sequence, batch_size: int) -> Iterable[Sequence]:
        for idx in range(0, len(seq), batch_size):
            yield seq[idx : idx + batch_size]

    def _build_text_messages(self, texts: Sequence[str], prefix: str) -> List[List[dict]]:
        prefix = (prefix + " ").strip()
        conversations: List[List[dict]] = []
        for text in texts:
            user_msg = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prefix}{text}".strip(),
                    }
                ],
            }
            conversations.append([user_msg])
        return conversations

    def _build_audio_messages(self, paths: Sequence[str], prefix: str) -> List[List[dict]]:
        """
        Build audio messages for encoding.

        NOTE: According to official nvidia/omni-embed-nemotron-3b documentation,
        audio-only inputs should NOT include a separate text prefix element.
        The prefix parameter is kept for API compatibility but not used for audio-only encoding.
        """
        conversations: List[List[dict]] = []
        for path in paths:
            user_msg = {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio": str(path),
                    },
                ],
            }
            conversations.append([user_msg])
        return conversations

    @contextmanager
    def _suppress_qwen_system_warning(self):
        logger = logging.getLogger()
        class _Filter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                msg = record.getMessage()
                # Qwen2.5 processor emits this warning when system prompt differs
                return "System prompt modified" not in msg
        filt = _Filter()
        logger.addFilter(filt)
        try:
            yield
        finally:
            logger.removeFilter(filt)

    def _apply_chat_template(self, conversations: List[List[dict]]) -> List[str]:
        texts: List[str] = []
        with self._suppress_qwen_system_warning():
            for conv in conversations:
                txt = self.processor.apply_chat_template(
                    conv,
                    add_generation_prompt=False,
                    tokenize=False,
                )
                texts.append(normalize_single_chat_template_output(txt))
        return texts

    def _prepare_batch(
        self,
        batch_inputs: _BatchInputs,
        text_kwargs: Optional[dict] = None,
        audio_kwargs: Optional[dict] = None,
    ) -> dict:
        text_kwargs = text_kwargs or {}
        text_kwargs.setdefault("padding", True)
        text_kwargs.setdefault("truncation", True)
        text_kwargs.setdefault("max_length", self.text_max_length)

        processor_kwargs = {
            "text": batch_inputs.text,
            "text_kwargs": text_kwargs,
            "return_tensors": "pt",
        }

        if batch_inputs.audio_arrays is not None:
            audio_kwargs = audio_kwargs or {}
            audio_kwargs.setdefault("sampling_rate", self.audio_sampling_rate)
            audio_kwargs.setdefault("return_attention_mask", True)
            if self.audio_max_length:
                audio_kwargs.setdefault("max_length", self.audio_max_length)
            processor_kwargs["audio"] = batch_inputs.audio_arrays
            processor_kwargs["audio_kwargs"] = audio_kwargs

        batch = self.processor(**processor_kwargs)
        tensor_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                tensor = value
            else:
                tensor = torch.tensor(value)
            if tensor.dtype == torch.float64:
                tensor = tensor.float()
            if tensor.is_floating_point() and self.device.type == "cuda":
                tensor = tensor.to(dtype=self.model.dtype)
            tensor_batch[key] = tensor.to(self.device)
        return tensor_batch

    @torch.inference_mode()
    def _encode(self, batch: dict) -> np.ndarray:
        outputs = self.model(
            **batch,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1]  # (B, L, D)
        attention_mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        masked = hidden * attention_mask
        summed = masked.sum(dim=1)
        denom = attention_mask.sum(dim=1).clamp(min=1e-6)
        pooled = summed / denom
        pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled.cpu().float().numpy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def encode_text(self, texts: List[str], batch_size: int = 32, device: Optional[str] = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.config.text_config.hidden_size), dtype=np.float32)

        if device and device != str(self.device):
            warnings.warn("Per-call device override is not supported; using adapter device.", RuntimeWarning)

        outputs: List[np.ndarray] = []
        for chunk in self._chunk_iter(texts, max(1, batch_size)):
            conversations = self._build_text_messages(chunk, self.query_prefix)
            chat_texts = self._apply_chat_template(conversations)
            batch_inputs = _BatchInputs(text=chat_texts)
            # For text-only, regular max length is enough
            processor_batch = self._prepare_batch(batch_inputs, text_kwargs={
                "truncation": True,
                "padding": True,
                "max_length": self.text_max_length,
            })
            embeddings = self._encode(processor_batch)
            outputs.append(embeddings)

        return np.concatenate(outputs, axis=0)

    @torch.inference_mode()
    def encode_audio(self, paths: List[str], batch_size: int = 8, device: Optional[str] = None) -> np.ndarray:
        if not paths:
            dim = getattr(self.model.config.text_config, "hidden_size", 2048)
            return np.zeros((0, dim), dtype=np.float32)

        if device and device != str(self.device):
            warnings.warn("Per-call device override is not supported; using adapter device.", RuntimeWarning)

        outputs: List[np.ndarray] = []
        for chunk in self._chunk_iter(paths, max(1, batch_size)):
            audio_arrays: List[np.ndarray] = []
            for path in chunk:
                try:
                    wav = self._audio_loader(path, self.audio_sampling_rate)
                except Exception as exc:
                    warnings.warn(f"Failed to load '{path}': {exc}. Using silence.", RuntimeWarning)
                    wav = np.zeros(int(self.audio_sampling_rate), dtype=np.float32)
                audio_arrays.append(wav)

            conversations = self._build_audio_messages(chunk, self.passage_prefix)
            chat_texts = self._apply_chat_template(conversations)
            batch_inputs = _BatchInputs(text=chat_texts, audio_arrays=audio_arrays)
            # For audio docs, allow much larger tokenized length so that
            # audio placeholder tokens are not truncated.
            processor_batch = self._prepare_batch(batch_inputs, text_kwargs={
                "truncation": True,
                "padding": True,
                "max_length": max(self.text_max_length, 32768),
            })
            embeddings = self._encode(processor_batch)
            outputs.append(embeddings)

        return np.concatenate(outputs, axis=0)
