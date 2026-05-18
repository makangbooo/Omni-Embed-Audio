"""AudioFlamingo-3 client for LALM reranking with dual backends."""

from __future__ import annotations

import contextlib
import io
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from .base import CandidateItem, LALMClient


class AudioFlamingoDependenciesMissing(RuntimeError):
    """Raised when required libraries for AudioFlamingo inference are absent."""


def _load_audio(audio_path: Path, target_sr: int = 16000):
    """
    Load an audio file to a mono waveform tensor at ``target_sr``.

    Uses torchaudio when available; falls back to librosa otherwise.
    """
    try:
        import torchaudio  # type: ignore
    except ImportError:  # pragma: no cover - fallback path
        torchaudio = None

    if torchaudio is not None:
        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        waveform = waveform.squeeze(0)
        return waveform, target_sr

    try:
        import librosa  # type: ignore
    except ImportError as exc:  # pragma: no cover - handled explicitly
        raise AudioFlamingoDependenciesMissing(
            "AudioFlamingoLALMClient requires `torchaudio` or `librosa` to load audio."
        ) from exc

    waveform, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
    return waveform, target_sr


@dataclass
class ParsedScore:
    value: float
    raw_text: str


class _Backend(Protocol):
    """Common interface implemented by concrete Audio Flamingo backends."""

    def score(self, query: str, audio_path: Path) -> ParsedScore:
        ...

    def score_batch(self, query: str, audio_paths: List[Path]) -> List[ParsedScore]:
        ...

    def listwise_rank(self, prompt: str, audio_paths: List[Path]) -> str:
        ...


class _TransformerBackend:
    """Backend backed by the official `audio_flamingo` Hugging Face package."""

    def __init__(
        self,
        *,
        model_name: str,
        dtype: str,
        device_map: Optional[str],
        cache_dir: Optional[str],
        local_files_only: bool,
        max_new_tokens: int,
        default_score: float,
        target_sample_rate: int,
        prompt_builder,
    ) -> None:
        try:
            import torch  # type: ignore
        except ImportError as exc:  # pragma: no cover - handled explicitly
            raise AudioFlamingoDependenciesMissing(
                "AudioFlamingoLALMClient requires the `torch` package. "
                "Install PyTorch to enable this client."
            ) from exc

        try:
            from audio_flamingo import (  # type: ignore
                AudioFlamingoForConditionalGeneration,
                AudioFlamingoProcessor,
            )
        except ImportError as exc:  # pragma: no cover - handled explicitly
            raise AudioFlamingoDependenciesMissing(
                "AudioFlamingoLALMClient requires the `audio_flamingo` package. "
                "Install it via `pip install git+https://github.com/NVIDIA/audio-flamingo@audio_flamingo_3`."
            ) from exc

        self._torch = torch
        self._default_score = default_score
        self._target_sr = target_sample_rate
        self._prompt_builder = prompt_builder
        self._max_new_tokens = max_new_tokens

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "auto": torch.float32,
        }
        if dtype.lower() not in dtype_map:
            raise ValueError(f"Unsupported dtype '{dtype}'.")
        torch_dtype = dtype_map[dtype.lower()]

        self._processor = AudioFlamingoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self._model = AudioFlamingoForConditionalGeneration.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch_dtype,
            torch_dtype=torch_dtype,
            device_map=device_map,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self._model.eval()
        self._max_new_tokens = max_new_tokens

    def _prepare_waveform(self, waveform) -> Any:
        torch = self._torch
        if not isinstance(waveform, torch.Tensor):
            waveform_tensor = torch.tensor(waveform, dtype=torch.float32)
        else:
            waveform_tensor = waveform
        if waveform_tensor.ndim == 1:
            waveform_tensor = waveform_tensor.unsqueeze(0)
        return waveform_tensor

    def score(self, query: str, audio_path: Path) -> ParsedScore:
        return self.score_batch(query, [audio_path])[0]

    def score_batch(self, query: str, audio_paths: List[Path]) -> List[ParsedScore]:
        prompts = [self._prompt_builder(query) for _ in audio_paths]
        waveforms = []
        for path in audio_paths:
            waveform_raw, _ = _load_audio(path, self._target_sr)
            waveforms.append(self._prepare_waveform(waveform_raw))

        processor_inputs = self._processor(
            text=prompts,
            audios=[waveform.squeeze(0) for waveform in waveforms],
            sampling_rate=self._target_sr,
            return_tensors="pt",
            padding=True,
        )
        processor_inputs = {k: v.to(self._model.device) for k, v in processor_inputs.items()}

        attention_mask = processor_inputs.get("attention_mask", None)
        if attention_mask is not None:
            prompt_lengths = attention_mask.sum(dim=1).tolist()
        else:
            prompt_lengths = [processor_inputs["input_ids"].shape[1]] * len(audio_paths)

        with self._torch.inference_mode():
            output_ids = self._model.generate(
                **processor_inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )

        trimmed = []
        for idx, seq in enumerate(output_ids):
            offset = int(prompt_lengths[idx])
            trimmed.append(seq[offset:])

        responses = self._processor.batch_decode(trimmed, skip_special_tokens=True)
        return [_extract_score(response, self._default_score) for response in responses]

    def listwise_rank(self, prompt: str, audio_paths: List[Path]) -> str:
        waveforms = []
        for path in audio_paths:
            waveform_raw, _ = _load_audio(path, self._target_sr)
            waveforms.append(self._prepare_waveform(waveform_raw).squeeze(0))

        processor_inputs = self._processor(
            text=[prompt],
            audios=waveforms,
            sampling_rate=self._target_sr,
            return_tensors="pt",
            padding=True,
        )
        processor_inputs = {k: v.to(self._model.device) for k, v in processor_inputs.items()}

        prompt_tokens = processor_inputs.get("input_ids")
        offset = prompt_tokens.shape[-1] if prompt_tokens is not None else 0

        with self._torch.inference_mode():
            output_ids = self._model.generate(
                **processor_inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )

        generated = output_ids[:, offset:]
        response = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        return response


class _LlavaBackend:
    """Backend that uses the locally packaged `llava` module shipped with AF3."""

    def __init__(
        self,
        *,
        model_name: str,
        device_map: Optional[str],
        cache_dir: Optional[str],
        local_files_only: bool,
        max_new_tokens: int,
        default_score: float,
        prompt_builder,
    ) -> None:
        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise AudioFlamingoDependenciesMissing(
                "AudioFlamingoLALMClient requires the `torch` package. "
                "Install PyTorch to enable this client."
            ) from exc

        try:
            import llava  # type: ignore
        except ImportError as exc:
            raise AudioFlamingoDependenciesMissing(
                "Unable to import the `llava` package from the local Audio Flamingo installation."
            ) from exc

        self._torch = torch
        self._prompt_builder = prompt_builder
        self._default_score = default_score
        self._llava = llava

        load_kwargs = {}
        if device_map:
            load_kwargs["device_map"] = device_map
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        if local_files_only:
            load_kwargs["local_files_only"] = True

        self._model = llava.load(model_name, **load_kwargs)
        generation_config = self._model.default_generation_config
        if hasattr(generation_config, "copy"):
            generation_config = generation_config.copy()
        elif hasattr(generation_config, "clone"):
            generation_config = generation_config.clone()
        elif hasattr(generation_config, "to_dict"):
            generation_config = generation_config.__class__(**generation_config.to_dict())
        if hasattr(generation_config, "max_new_tokens"):
            generation_config.max_new_tokens = max_new_tokens
        else:
            setattr(generation_config, "max_new_tokens", max_new_tokens)
        self._generation_config = generation_config

    def score(self, query: str, audio_path: Path) -> ParsedScore:
        return self.score_batch(query, [audio_path])[0]

    def score_batch(self, query: str, audio_paths: List[Path]) -> List[ParsedScore]:
        prompt = self._prompt_builder(query)
        outputs: List[ParsedScore] = []
        for audio_path in audio_paths:
            sequence = [self._llava.Sound(str(audio_path)), prompt]
            with self._torch.inference_mode():
                response = self._model.generate_content(sequence, generation_config=self._generation_config)
            outputs.append(_extract_score(response, self._default_score))
        return outputs

    def listwise_rank(self, prompt: str, audio_paths: List[Path]) -> str:
        sequence: List[Any] = []
        for idx, audio_path in enumerate(audio_paths, start=1):
            sequence.append(self._llava.Sound(str(audio_path)))
            sequence.append(f"[Clip {idx}]")
        sequence.append(prompt)
        with self._torch.inference_mode():
            response = self._model.generate_content(sequence, generation_config=self._generation_config)
        return response


class _LlavaSpaceBackend:
    """Backend that mirrors the Hugging Face Space loading logic."""

    def __init__(
        self,
        *,
        model_name: str,
        cache_dir: Optional[str],
        local_files_only: bool,
        device_map: Optional[str],
        max_new_tokens: int,
        default_score: float,
        prompt_builder,
    ) -> None:
        try:
            import torch  # type: ignore
        except ImportError as exc:  # pragma: no cover - handled explicitly
            raise AudioFlamingoDependenciesMissing(
                "AudioFlamingoLALMClient requires the `torch` package. "
                "Install PyTorch to enable this client."
            ) from exc

        try:
            from huggingface_hub import snapshot_download  # type: ignore
        except ImportError as exc:  # pragma: no cover - handled explicitly
            raise AudioFlamingoDependenciesMissing(
                "AudioFlamingoLALMClient fallback requires `huggingface_hub`."
            ) from exc

        try:
            space_dir = snapshot_download(
                repo_id=model_name,
                repo_type="space",
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        except Exception as exc:  # pragma: no cover - explicit error
            raise AudioFlamingoDependenciesMissing(
                "Unable to locate the cached Hugging Face Space for Audio Flamingo 3. "
                "Run `huggingface_hub.snapshot_download('nvidia/audio-flamingo-3', repo_type='space')` "
                "in an environment with network access first."
            ) from exc

        if space_dir not in sys.path:
            sys.path.insert(0, space_dir)

        try:
            import llava  # type: ignore
        except ImportError as exc:  # pragma: no cover - handled explicitly
            raise AudioFlamingoDependenciesMissing(
                "Failed to import `llava` from the downloaded Audio Flamingo Space."
            ) from exc

        # Silence verbose loguru logging emitted by llava chat utilities.
        try:
            from loguru import logger as loguru_logger  # type: ignore

            loguru_logger.remove()
            loguru_logger.add(sys.stderr, level="WARNING")
        except Exception:
            pass

        try:
            ckpt_dir = snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        except Exception as exc:  # pragma: no cover - explicit error
            raise AudioFlamingoDependenciesMissing(
                "Unable to locate the cached Audio Flamingo 3 model weights. "
                "Run `huggingface_hub.snapshot_download('nvidia/audio-flamingo-3')` beforehand."
            ) from exc

        device = self._resolve_device(torch, device_map)
        self._model = llava.load(ckpt_dir, model_base=None).to(device)
        self._generation_config = self._model.default_generation_config
        if hasattr(self._generation_config, "max_new_tokens"):
            self._generation_config.max_new_tokens = max_new_tokens

        self._llava = llava
        self._torch = torch
        self._prompt_builder = prompt_builder
        self._default_score = default_score

    @staticmethod
    def _resolve_device(torch_mod, device_map: Optional[str]) -> str:
        if device_map and device_map not in {"auto", "Auto"}:
            return device_map
        return "cuda" if torch_mod.cuda.is_available() else "cpu"

    def score(self, query: str, audio_path: Path) -> ParsedScore:
        return self.score_batch(query, [audio_path])[0]

    @staticmethod
    @contextlib.contextmanager
    def _silence_output():
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield

    def score_batch(self, query: str, audio_paths: List[Path]) -> List[ParsedScore]:
        prompt = self._prompt_builder(query)
        results: List[ParsedScore] = []
        for audio_path in audio_paths:
            sound = self._llava.Sound(str(audio_path))
            with self._silence_output():
                response = self._model.generate_content(
                    [sound, prompt],
                    generation_config=self._generation_config,
                )
            results.append(_extract_score(response, self._default_score))
        return results

    def listwise_rank(self, prompt: str, audio_paths: List[Path]) -> str:
        sequence: List[Any] = []
        for idx, audio_path in enumerate(audio_paths, start=1):
            sequence.append(self._llava.Sound(str(audio_path)))
            sequence.append(f"[Clip {idx}]")
        sequence.append(prompt)
        with self._silence_output():
            response = self._model.generate_content(
                sequence,
                generation_config=self._generation_config,
            )
        return response


def _extract_score(text: str, default_score: float) -> ParsedScore:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    value = default_score
    if matches:
        try:
            parsed = float(matches[-1])
            value = float(max(0.0, min(1.0, parsed)))
        except ValueError:
            value = default_score
    return ParsedScore(value=value, raw_text=text)


class AudioFlamingoLALMClient(LALMClient):
    """
    AudioFlamingo-3 backed client for audio-native reranking.

    Attempts to load the official `audio_flamingo` package first; if that
    dependency is unavailable it falls back to the Hugging Face Space loader
    used in NVIDIA's demos.
    """

    def __init__(
        self,
        model_name: str = "nvidia/audio-flamingo-3",
        device_map: Optional[str] = "auto",
        dtype: str = "bfloat16",
        max_new_tokens: int = 32,
        system_prompt: str = (
            "You are an expert audio retrieval judge. "
            "Rate how well an audio clip matches a natural language query."
        ),
        score_prompt_template: str = (
            "Query: {query}\n"
            "Listen to the audio clip provided and respond with a single "
            "floating point number between 0 and 1 (inclusive) that reflects "
            "how relevant the audio is to the query. Higher is better.\n"
            "Answer format: SCORE:<value>"
        ),
        default_score: float = 0.0,
        target_sample_rate: int = 16000,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        batch_size: int = 1,
        listwise_prompt_template: str = (
            "You will be given several audio clips and a retrieval query.\n"
            "Query: {query}\n\n"
            "Clips:\n{enumerated_clips}\n\n"
            "After listening to every clip, respond with their order from most to "
            "least relevant using the format ORDER: i1,i2,... where each number "
            "refers to the clip index above."
        ),
        listwise_response_pattern: str = r"ORDER\s*:?\s*\[?([0-9,\s]+)\]?",
        # In-context learning (text-only few-shot via XACLE). Examples are prepended as
        # "Example i: Query: ..." and optionally "Answer: SCORE:x.xx".
        icl_enabled: bool = False,
        icl_dataset_dir: Optional[str] = None,
        icl_split: str = "train",
        icl_k: int = 0,
        icl_selection: str = "jaccard",
        icl_include_answers: bool = True,
        icl_use_average: bool = True,
        icl_seed: int = 0,
    ) -> None:
        self._default_score = float(default_score)
        self._target_sr = int(target_sample_rate)
        self._score_prompt_template = score_prompt_template
        self._system_prompt = system_prompt
        self._batch_size = max(1, int(batch_size))
        self._listwise_prompt_template = listwise_prompt_template
        self._listwise_response_regex = re.compile(listwise_response_pattern, re.IGNORECASE)
        self._listwise_stats: Dict[str, int] = {
            "listwise_calls": 0,
            "parse_failures": 0,
            "backend_errors": 0,
            "partial_orders": 0,
        }
        self._last_rank_info: Optional[Dict[str, Any]] = None

        # ICL configuration (text-only few-shot; does not attach example audio)
        self._icl_enabled = bool(icl_enabled)
        self._icl_k = max(0, int(icl_k))
        self._icl_selection = icl_selection
        self._icl_include_answers = bool(icl_include_answers)
        self._icl_selector = None
        if self._icl_enabled and self._icl_k > 0:
            try:
                from ..icl.xacle_examples import XACLEExampleSelector  # lazy import
                ds_dir = icl_dataset_dir or "./XACLE_dataset"
                self._icl_selector = XACLEExampleSelector(
                    Path(ds_dir),
                    split=icl_split,
                    use_average=bool(icl_use_average),
                    normalize_to_unit=True,
                    seed=int(icl_seed),
                )
            except Exception as exc:
                print(f"[WARN] AudioFlamingo ICL disabled: {exc}")
                self._icl_enabled = False

        prompt_builder = self._build_prompt
        backend_errors: List[str] = []
        self._backend: Optional[_Backend] = None

        try:
            self._backend = _TransformerBackend(
                model_name=model_name,
                dtype=dtype,
                device_map=device_map,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                max_new_tokens=max_new_tokens,
                default_score=self._default_score,
                target_sample_rate=self._target_sr,
                prompt_builder=prompt_builder,
            )
        except Exception as exc:  # pragma: no cover - fallback path
            backend_errors.append(str(exc))

        if self._backend is None:
            try:
                self._backend = _LlavaBackend(
                    model_name=model_name,
                    device_map=device_map,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    max_new_tokens=max_new_tokens,
                    default_score=self._default_score,
                    prompt_builder=prompt_builder,
                )
            except Exception as exc:  # pragma: no cover - fallback path
                backend_errors.append(str(exc))

        if self._backend is None:
            try:
                self._backend = _LlavaSpaceBackend(
                    model_name=model_name,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    device_map=device_map,
                    max_new_tokens=max_new_tokens,
                    default_score=self._default_score,
                    prompt_builder=prompt_builder,
                )
            except Exception as exc:  # pragma: no cover - fallback path
                backend_errors.append(str(exc))

        if self._backend is None:
            raise AudioFlamingoDependenciesMissing(
                "AudioFlamingoLALMClient could not initialize any backend.\n"
                + "\n".join(backend_errors)
            )

    # ----------------------------------------------------------------- helpers
    def _build_prompt(self, query: str) -> str:
        # Optional text-only ICL block
        icl_lines: List[str] = []
        if self._icl_enabled and self._icl_selector is not None and self._icl_k > 0:
            try:
                examples = self._icl_selector.select(query, self._icl_k, method=self._icl_selection)
            except Exception:
                examples = []
            for i, ex in enumerate(examples, start=1):
                icl_lines.append(f"Example {i}: Query: {ex.text}")
                if self._icl_include_answers:
                    icl_lines.append(f"Answer: SCORE:{ex.score:.2f}")
            if icl_lines:
                icl_lines.append("Now rate the following query.")

        main = self._score_prompt_template.format(query=query)
        parts: List[str] = []
        if self._system_prompt:
            parts.append(self._system_prompt.strip())
        if icl_lines:
            parts.append("\n".join(icl_lines))
        parts.append(main)
        return "\n\n".join(parts)

    def _build_listwise_prompt(self, query: str, candidates: List[CandidateItem]) -> str:
        lines = [
            f"{idx}. Clip {idx} (id: {cand.clip_id})"
            for idx, cand in enumerate(candidates, start=1)
        ]
        enumerated = "\n".join(lines)
        return self._listwise_prompt_template.format(
            query=query,
            enumerated_clips=enumerated,
        )

    def _parse_listwise_response(self, response: str, num_candidates: int) -> List[int]:
        def extract_indices(text: str) -> List[int]:
            indices: List[int] = []
            for token in re.findall(r"\d+", text):
                try:
                    val = int(token)
                except ValueError:
                    continue
                zero_idx = val - 1
                if 0 <= zero_idx < num_candidates and zero_idx not in indices:
                    indices.append(zero_idx)

            if len(indices) < num_candidates:
                word_map = {
                    "one": 1,
                    "two": 2,
                    "three": 3,
                    "four": 4,
                    "five": 5,
                    "six": 6,
                    "seven": 7,
                    "eight": 8,
                    "nine": 9,
                    "ten": 10,
                }
                for word in re.findall(r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\b", text, flags=re.IGNORECASE):
                    val = word_map[word.lower()]
                    zero_idx = val - 1
                    if 0 <= zero_idx < num_candidates and zero_idx not in indices:
                        indices.append(zero_idx)
            return indices

        search_regions: List[str] = []
        match = self._listwise_response_regex.search(response)
        if match:
            search_regions.append(match.group(1))
        search_regions.append(response)

        for region in search_regions:
            indices = extract_indices(region)
            if indices:
                return indices
        return []

    def _rank_pointwise(
        self,
        query: str,
        candidates: List[CandidateItem],
        fallback_reason: Optional[str] = None,
        context_info: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        if not candidates:
            info: Dict[str, Any] = {
                "used_fallback": True,
                "fallback_reason": fallback_reason or "pointwise_rerank",
                "pointwise_scores": {},
            }
            if context_info:
                info.update(context_info)
            self._last_rank_info = info
            return []
        scores = self.score_audio_batch(query, [cand.audio_path for cand in candidates])
        indexed = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )
        ordered_indices = [idx for idx, _ in indexed]
        info = {
            "used_fallback": True,
            "fallback_reason": fallback_reason or "pointwise_rerank",
            "pointwise_scores": {
                candidates[idx].candidate_idx: float(score) for idx, score in indexed
            },
        }
        if context_info:
            info.update(context_info)
        self._last_rank_info = info
        return ordered_indices

    # ----------------------------------------------------------------- protocol
    def score_audio(self, query: str, audio_path: Path) -> float:
        return self.score_audio_batch(query, [audio_path])[0]

    def score_audio_batch(self, query: str, audio_paths: List[Path]) -> List[float]:
        parsed_scores = self._backend.score_batch(
            query,
            [Path(audio_path) for audio_path in audio_paths],
        )
        return [parsed.value for parsed in parsed_scores]

    def score_caption(self, query: str, caption: str) -> float:
        raise NotImplementedError("Caption scoring is not supported by AudioFlamingoLALMClient.")

    def rank_audio(self, query: str, candidates: List[CandidateItem]) -> List[int]:
        self._listwise_stats["listwise_calls"] += 1
        if not candidates:
            self._last_rank_info = {
                "used_fallback": False,
                "prompt": "",
                "response": "",
                "parsed_order": [],
                "partial_order": False,
            }
            return []

        self._last_rank_info = None

        prompt = self._build_listwise_prompt(query, candidates)
        try:
            response = self._backend.listwise_rank(
                prompt,
                [cand.audio_path for cand in candidates],
            )
        except Exception as exc:
            self._listwise_stats["backend_errors"] += 1
            return self._rank_pointwise(
                query,
                candidates,
                fallback_reason="backend_error",
                context_info={
                    "prompt": prompt,
                    "exception": str(exc),
                },
            )

        order = self._parse_listwise_response(response, len(candidates))
        if not order:
            self._listwise_stats["parse_failures"] += 1
            parsed_ids: List[int] = []
            return self._rank_pointwise(
                query,
                candidates,
                fallback_reason="parse_failure",
                context_info={
                    "prompt": prompt,
                    "response": response,
                    "parsed_order": parsed_ids,
                },
            )

        if len(order) < len(candidates):
            self._listwise_stats["partial_orders"] += 1
            fallback = self._rank_pointwise(query, candidates, fallback_reason="partial_completion")
            seen = set(order)
            for idx in fallback:
                if idx not in seen:
                    order.append(idx)
            parsed_ids = [candidates[idx].candidate_idx for idx in order[: len(seen)]]
            completed_ids = [candidates[idx].candidate_idx for idx in fallback if idx not in seen]
            self._last_rank_info = {
                "used_fallback": False,
                "prompt": prompt,
                "response": response,
                "parsed_order": parsed_ids,
                "partial_order": True,
                "completed_with": completed_ids,
            }
        else:
            parsed_ids = [candidates[idx].candidate_idx for idx in order]
            self._last_rank_info = {
                "used_fallback": False,
                "prompt": prompt,
                "response": response,
                "parsed_order": parsed_ids,
                "partial_order": False,
            }
        return order

    def get_listwise_stats(self) -> Dict[str, int]:
        return dict(self._listwise_stats)

    def get_last_rank_info(self) -> Optional[Dict[str, Any]]:
        if self._last_rank_info is None:
            return None
        return dict(self._last_rank_info)

    def rank_captions(self, query: str, captions: List[str]) -> List[int]:
        raise NotImplementedError("Caption ranking is not supported by AudioFlamingoLALMClient.")

    def generate_caption(self, audio_path: Path) -> str:
        raise NotImplementedError("Caption generation is not supported by AudioFlamingoLALMClient.")

    @property
    def batch_size(self) -> int:
        return self._batch_size
