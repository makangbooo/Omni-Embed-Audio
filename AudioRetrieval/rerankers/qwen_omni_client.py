"""Qwen Omni clients for LALM reranking with offline-friendly loading."""

from __future__ import annotations

import importlib
import importlib.util
import json
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import CandidateItem, LALMClient
try:
    from qwen_omni_utils import process_mm_info  # type: ignore
except ImportError:  # pragma: no cover - handled explicitly
    process_mm_info = None  # type: ignore[assignment]


class QwenOmniDependenciesMissing(RuntimeError):
    """Raised when required libraries for Qwen inference are absent."""


_REGISTERED_MODELS: Dict[str, bool] = {}


def _resolve_local_model_path(
    model_name: str,
    cache_dir: Optional[str],
) -> Optional[Path]:
    """
    Determine if a local directory containing ``config.json`` is available for this model.
    Preference order:
      1. Explicit ``cache_dir`` argument.
      2. ``model_name`` itself pointing to a local directory.
    """
    if cache_dir:
        candidate = Path(cache_dir)
        if (candidate / "config.json").exists():
            return candidate
    candidate = Path(model_name)
    if (candidate / "config.json").exists():
        return candidate
    return None


def _ensure_model_modules(
    model_name: str,
    cache_dir: Optional[str],
    local_files_only: bool,
) -> None:
    """
    Ensure remote custom modules shipped with a Qwen Omni checkpoint are importable.
    """
    if _REGISTERED_MODELS.get(model_name):
        return

    if not model_name:
        raise QwenOmniDependenciesMissing(
            "Model name must be provided to locate cached Qwen Omni files."
        )

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:  # pragma: no cover - handled explicitly
        raise QwenOmniDependenciesMissing(
            "Qwen Omni clients require `huggingface_hub` for offline loading."
        ) from exc

    local_path = _resolve_local_model_path(model_name, cache_dir)
    if local_path is not None:
        snapshot_path = local_path
    else:
        try:
            snapshot_path = snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                allow_patterns=["*.py", "*.json", "*.safetensors", "*.bin", "*.model", "*.tiktoken"],
            )
        except Exception as exc:  # pragma: no cover - explicit guidance
            raise QwenOmniDependenciesMissing(
                "Unable to locate cached assets for Qwen Omni. "
                "Download the repository with network access using:\n"
                f"  from huggingface_hub import snapshot_download\n"
                f"  snapshot_download('{model_name}')"
            ) from exc

    snapshot_root = Path(snapshot_path)
    if str(snapshot_root) not in sys.path:
        sys.path.insert(0, str(snapshot_root))

    config_path = snapshot_root / "config.json"
    if not config_path.exists():
        raise QwenOmniDependenciesMissing(f"config.json not found in snapshot for '{model_name}'.")

    try:
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise QwenOmniDependenciesMissing(f"Failed to parse config.json for '{model_name}'.") from exc

    auto_map = config_data.get("auto_map", {})
    if not auto_map:
        _REGISTERED_MODELS[model_name] = True
        return

    loaded: Dict[str, Any] = {}
    for key, target in auto_map.items():
        if isinstance(target, (list, tuple)):
            target = target[0]
        if not isinstance(target, str):
            continue
        module_name, class_name = target.rsplit(".", 1)
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            module_path = snapshot_root / (module_name.replace(".", "/") + ".py")
            if not module_path.exists():
                raise QwenOmniDependenciesMissing(
                    f"Unable to locate module '{module_name}' required for '{model_name}'."
                )
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise QwenOmniDependenciesMissing(
                    f"Failed to create spec for module '{module_name}' in '{model_name}'."
                )
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
        loaded[key] = getattr(module, class_name)

    try:
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoProcessor
    except ImportError as exc:
        raise QwenOmniDependenciesMissing(
            "Qwen Omni clients require `transformers`. Install it before enabling this client."
        ) from exc

    config_cls = loaded.get("AutoConfig")
    if config_cls is None:
        raise QwenOmniDependenciesMissing(
            f"AutoConfig mapping missing for '{model_name}'. Unable to register configuration."
        )

    model_type = config_data.get("model_type", getattr(config_cls, "model_type", None))
    if not model_type:
        raise QwenOmniDependenciesMissing(
            f"Unable to determine model_type for '{model_name}'."
        )

    AutoConfig.register(model_type, config_cls, exist_ok=True)

    model_for_causal_cls = loaded.get("AutoModelForCausalLM")
    if model_for_causal_cls is not None:
        AutoModelForCausalLM.register(config_cls, model_for_causal_cls, exist_ok=True)

    model_cls = loaded.get("AutoModel")
    if model_cls is not None:
        AutoModel.register(config_cls, model_cls, exist_ok=True)

    processor_cls = loaded.get("AutoProcessor")
    if processor_cls is not None:
        try:
            AutoProcessor.register(config_cls, processor_cls, exist_ok=True)
        except Exception:
            pass

    _REGISTERED_MODELS[model_name] = True


def _load_transformer_deps(
    model_name: str,
    cache_dir: Optional[str],
    local_files_only: bool,
) -> Tuple[Any, Any, Any, Any]:
    """
    Import torch + transformers utilities, downloading/bootstrapping Qwen modules if needed.
    """
    try:
        import torch  # type: ignore
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForCausalLM,
            AutoProcessor,
        )  # type: ignore
    except ImportError as exc:  # pragma: no cover - handled explicitly
        raise QwenOmniDependenciesMissing(
            "Qwen Omni clients require `torch` and `transformers`. "
            "Install them before enabling this client."
        ) from exc

    local_path = _resolve_local_model_path(model_name, cache_dir)

    try:
        AutoConfig.from_pretrained(
            str(local_path) if local_path is not None else model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
    except Exception:
        _ensure_model_modules(model_name, cache_dir, local_files_only)

    return torch, AutoModelForCausalLM, AutoProcessor, AutoModel


def _lazy_imports(  # backwards compatibility with older tooling
    model_name: str = "Qwen/Qwen2.5-Omni-7B",
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
):
    torch, auto_causal, auto_processor, auto_model = _load_transformer_deps(
        model_name, cache_dir, local_files_only
    )
    return torch, auto_causal, auto_processor, auto_model, None


@dataclass
class ParsedScore:
    value: float
    raw_text: str


class BaseQwenOmniClient(LALMClient):
    """Shared logic for Qwen Omni reranker clients."""

    default_model_name: str = "Qwen/Qwen2.5-Omni-7B"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device_map: str = "auto",
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
        attn_implementation: Optional[str] = None,
        listwise_prompt_template: Optional[str] = None,
        listwise_response_pattern: str = r"ORDER\s*:?\s*\[?([0-9,\s]+)\]?",
        # In-context learning (ICL) options
        icl_enabled: bool = False,
        icl_dataset_dir: Optional[str] = None,
        icl_split: str = "train",
        icl_k: int = 4,
        icl_selection: str = "jaccard",
        icl_include_answers: bool = True,
        icl_use_average: bool = True,
        icl_seed: int = 0,
        # Batch scoring
        batch_size: int = 1,
    ) -> None:
        self._model_name = model_name or self.default_model_name
        (
            torch,
            AutoModelForCausalLM,
            AutoProcessor,
            AutoModel,
        ) = _load_transformer_deps(self._model_name, cache_dir, local_files_only)

        self._torch = torch
        self._default_score = float(default_score)
        self._target_sr = int(target_sample_rate)
        self._max_new_tokens = int(max_new_tokens)
        self._system_prompt = system_prompt
        self._score_prompt_template = score_prompt_template
        self._dtype = self._resolve_dtype(dtype)
        self._device_map = device_map
        self._cache_dir = cache_dir
        self._local_files_only = bool(local_files_only)
        self.batch_size = max(1, int(batch_size))

        # ICL configuration
        self._icl_enabled = bool(icl_enabled)
        self._icl_k = int(icl_k)
        self._icl_selection = icl_selection
        self._icl_include_answers = bool(icl_include_answers)
        self._icl_selector = None

        model_path = _resolve_local_model_path(self._model_name, self._cache_dir)
        pretrained_source = str(model_path) if model_path is not None else self._model_name

        from transformers import AutoConfig  # Lazy import to align with dependency checks

        self._config = AutoConfig.from_pretrained(
            pretrained_source,
            trust_remote_code=True,
            cache_dir=self._cache_dir,
            local_files_only=self._local_files_only,
        )

        processor_loader = AutoProcessor
        if getattr(self._config, "model_type", "") == "qwen3_omni_moe":
            try:
                from transformers import Qwen3OmniMoeProcessor

                processor_loader = Qwen3OmniMoeProcessor
            except ImportError:
                warnings.warn(
                    "Qwen3 Omni processor class not found; falling back to AutoProcessor.",
                    RuntimeWarning,
                )

        self.processor = processor_loader.from_pretrained(
            pretrained_source,
            trust_remote_code=True,
            cache_dir=self._cache_dir,
            local_files_only=self._local_files_only,
        )

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self._dtype,
            "device_map": self._device_map,
            "cache_dir": self._cache_dir,
            "local_files_only": self._local_files_only,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        model_loader = AutoModelForCausalLM
        fallback_loader = AutoModel
        if getattr(self._config, "model_type", "") == "qwen3_omni_moe":
            try:
                from transformers import Qwen3OmniMoeForConditionalGeneration

                model_loader = Qwen3OmniMoeForConditionalGeneration
                fallback_loader = Qwen3OmniMoeForConditionalGeneration
            except ImportError:
                warnings.warn(
                    "Qwen3 Omni model class not found; attempting AutoModel fallback which may fail.",
                    RuntimeWarning,
                )

        try:
            self.model = model_loader.from_pretrained(pretrained_source, **model_kwargs)
        except ValueError as exc:
            if "Unrecognized configuration class" in str(exc):
                warnings.warn(
                    "AutoModelForCausalLM does not recognize the Qwen Omni config; "
                    "falling back to AutoModel. Generation will still be available "
                    "if the remote model implements it.",
                    RuntimeWarning,
                )
                self.model = fallback_loader.from_pretrained(pretrained_source, **model_kwargs)
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
            except Exception:  # pragma: no cover - best effort guard
                warnings.warn(
                    "Failed to disable Qwen talker module; audio outputs may be generated.",
                    RuntimeWarning,
                )
        self.model.eval()
        self._listwise_prompt_template = listwise_prompt_template
        self._listwise_response_regex = (
            re.compile(listwise_response_pattern, re.IGNORECASE)
            if listwise_prompt_template
            else None
        )
        self._listwise_stats: Dict[str, int] = {
            "listwise_calls": 0,
            "parse_failures": 0,
            "errors": 0,
            "partial_orders": 0,
        }
        self._last_rank_info: Optional[Dict[str, Any]] = None

        # Build ICL example selector lazily to avoid import if unused
        if self._icl_enabled:
            try:
                from ..icl.xacle_examples import XACLEExampleSelector
            except Exception as exc:  # pragma: no cover - import guard
                warnings.warn(
                    f"ICL requested but XACLE selector unavailable: {exc}", RuntimeWarning
                )
                self._icl_enabled = False
            else:
                ds_dir = icl_dataset_dir or "./XACLE_dataset"
                try:
                    self._icl_selector = XACLEExampleSelector(
                        Path(ds_dir),
                        split=icl_split,
                        use_average=bool(icl_use_average),
                        normalize_to_unit=True,
                        seed=int(icl_seed),
                    )
                except Exception as exc:  # pragma: no cover - robust fallback
                    warnings.warn(
                        f"Failed to initialize XACLE selector at '{ds_dir}': {exc}. "
                        "Proceeding without ICL.",
                        RuntimeWarning,
                    )
                    self._icl_enabled = False

    # ------------------------------------------------------------------ utils
    def _resolve_dtype(self, dtype: str):
        torch = self._torch
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "auto": torch.float32,
        }
        if dtype.lower() not in mapping:
            raise ValueError(f"Unsupported dtype '{dtype}'.")
        return mapping[dtype.lower()]

    def _build_messages(self, query: str, audio_path: Path):
        content: List[Dict[str, Any]] = []

        # Optional few-shot ICL from XACLE examples
        if self._icl_enabled and self._icl_selector is not None and self._icl_k > 0:
            try:
                examples = self._icl_selector.select(query, self._icl_k, method=self._icl_selection)
            except Exception as exc:  # pragma: no cover - resilient to data issues
                warnings.warn(f"ICL selection failed: {exc}", RuntimeWarning)
                examples = []

            for idx, ex in enumerate(examples, start=1):
                content.append({
                    "type": "text",
                    "text": (
                        f"Example {idx}: Query: {ex.text}\n"
                        "Respond strictly with 'SCORE:<value>' between 0 and 1."
                    ),
                })
                content.append({"type": "audio", "audio": str(Path(ex.audio_path).resolve())})
                if self._icl_include_answers:
                    content.append({"type": "text", "text": f"Answer: SCORE:{ex.score:.2f}"})

            if examples:
                content.append({"type": "text", "text": "Now rate the following query."})

        # Target query to score
        content.append({"type": "text", "text": self._score_prompt_template.format(query=query)})
        content.append({"type": "audio", "audio": str(audio_path)})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self._system_prompt}]},
            {"role": "user", "content": content},
        ]
        return messages

    @staticmethod
    def _extract_score(text: str, default_score: float) -> ParsedScore:
        # Prefer explicitly formatted answers like "SCORE: <value>"
        score_matches = re.findall(r"SCORE\s*:\s*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
        value = default_score
        if score_matches:
            try:
                parsed = float(score_matches[-1])
                value = float(max(0.0, min(1.0, parsed)))
                return ParsedScore(value=value, raw_text=text)
            except ValueError:
                pass

        # Fallback to the last float anywhere in the response
        matches = re.findall(r"-?\d+(?:\.\d+)?", text)
        if matches:
            try:
                parsed = float(matches[-1])
                value = float(max(0.0, min(1.0, parsed)))
            except ValueError:
                value = default_score
        return ParsedScore(value=value, raw_text=text)

    def _score_query_audio(self, query: str, audio_path: Path) -> ParsedScore:
        if process_mm_info is None:
            raise QwenOmniDependenciesMissing(
                "Qwen Omni clients require the `qwen-omni-utils` package. "
                "Install it with `pip install qwen-omni-utils`."
            )

        resolved_path = Path(audio_path).resolve()
        messages = self._build_messages(query, resolved_path)

        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        try:
            audio_inputs, image_inputs, video_inputs = process_mm_info(
                messages,
                use_audio_in_video=False,
            )
        except Exception as exc:  # pragma: no cover - convert to user guidance
            raise QwenOmniDependenciesMissing(
                "Failed to preprocess audio for Qwen Omni. Ensure `librosa`/`ffmpeg` are installed."
            ) from exc

        if audio_inputs is None:
            raise QwenOmniDependenciesMissing(
                "No audio content provided to Qwen Omni conversation."
            )

        model_inputs = self.processor(
            text=prompt,
            audio=audio_inputs,
            images=image_inputs,
            videos=video_inputs,
            sampling_rate=self._target_sr,
            return_tensors="pt",
            padding=True,
        )

        model_inputs = model_inputs.to(self.model.device)
        model_inputs = model_inputs.to(self._dtype)

        pad_token_id = getattr(self.model.generation_config, "pad_token_id", None)
        eos_token_id = getattr(self.model.generation_config, "eos_token_id", None)
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id

        with self._torch.inference_mode():
            generation = self.model.generate(
                **model_inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

        sequences = generation
        if hasattr(generation, "sequences"):
            sequences = generation.sequences
        elif isinstance(generation, tuple):
            sequences = generation[0]

        prompt_length = model_inputs["input_ids"].shape[-1]
        continuation = sequences[:, prompt_length:]

        response = self.processor.batch_decode(
            continuation,
            skip_special_tokens=True,
        )[0]
        return self._extract_score(response, self._default_score)

    # ------------------------------------------------------------ batched API
    def score_audio_batch(self, query: str, audio_paths: List[Path]) -> List[float]:
        """Score multiple audios for a single query in one forward pass when possible.

        Falls back to per-item scoring if preprocessing or batching fails.
        """
        if not audio_paths:
            return []
        if process_mm_info is None:
            # Keep identical behavior to single example path
            return [self._score_query_audio(query, Path(p)).value for p in audio_paths]

        messages_list: List[List[Dict[str, Any]]] = []
        prompts: List[str] = []
        audio_inputs_list: List[Any] = []
        image_inputs_list: List[Any] = []
        video_inputs_list: List[Any] = []
        failures: List[int] = []

        for i, path in enumerate(audio_paths):
            try:
                path = Path(path).resolve()
                msgs = self._build_messages(query, path)
                prompt = self.processor.apply_chat_template(
                    msgs,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                a_inputs, i_inputs, v_inputs = process_mm_info(msgs, use_audio_in_video=False)
                if a_inputs is None:
                    failures.append(i)
                    continue
                messages_list.append(msgs)
                prompts.append(prompt)
                audio_inputs_list.append(a_inputs)
                image_inputs_list.append(i_inputs)
                video_inputs_list.append(v_inputs)
            except Exception:
                failures.append(i)

        # Fallback if we cannot batch any
        if not prompts:
            return [self._score_query_audio(query, Path(p)).value for p in audio_paths]

        try:
            # Processor supports list inputs -> BatchEncoding
            model_inputs = self.processor(
                text=prompts,
                audio=audio_inputs_list,
                images=image_inputs_list,
                videos=video_inputs_list,
                sampling_rate=self._target_sr,
                return_tensors="pt",
                padding=True,
            )
            model_inputs = model_inputs.to(self.model.device)
            model_inputs = model_inputs.to(self._dtype)

            pad_token_id = getattr(self.model.generation_config, "pad_token_id", None)
            eos_token_id = getattr(self.model.generation_config, "eos_token_id", None)
            if pad_token_id is None and eos_token_id is not None:
                pad_token_id = eos_token_id

            with self._torch.inference_mode():
                generation = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                )

            sequences = generation
            if hasattr(generation, "sequences"):
                sequences = generation.sequences
            elif isinstance(generation, tuple):
                sequences = generation[0]

            # Compute per-sample prompt lengths using attention_mask
            attn = model_inputs.get("attention_mask", None)
            if attn is None:
                # No mask available; decode whole continuation as best-effort
                responses = self.processor.batch_decode(sequences, skip_special_tokens=True)
            else:
                attn = attn.to("cpu")
                lens = attn.sum(dim=1).tolist()
                conts = []
                for i, L in enumerate(lens):
                    conts.append(sequences[i, int(L):])
                responses = self.processor.batch_decode(conts, skip_special_tokens=True)

            parsed = [self._extract_score(r, self._default_score).value for r in responses]
        except Exception as exc:
            warnings.warn(
                f"Qwen Omni batch scoring failed ({exc}); falling back to sequential.",
                RuntimeWarning,
            )
            return [self._score_query_audio(query, Path(p)).value for p in audio_paths]

        # If some items failed preprocessing, score them individually and merge
        if failures:
            for idx in failures:
                parsed.insert(idx, self._score_query_audio(query, Path(audio_paths[idx])).value)

        # Ensure length equals input list length
        if len(parsed) != len(audio_paths):
            # Best-effort fallback: compute all individually
            return [self._score_query_audio(query, Path(p)).value for p in audio_paths]
        return [float(x) for x in parsed]

    def _build_listwise_messages(
        self,
        query: str,
        candidates: List[CandidateItem],
    ) -> Tuple[List[Dict[str, Any]], str]:
        if self._listwise_prompt_template is None:
            raise QwenOmniDependenciesMissing("Listwise prompt template is not configured.")

        clip_lines: List[str] = []
        content: List[Dict[str, Any]] = []
        for idx, cand in enumerate(candidates, start=1):
            clip_id = cand.clip_id or f"candidate_{cand.candidate_idx}"
            clip_line = f"{idx}. Clip {idx} (id: {clip_id})"
            clip_lines.append(clip_line)
            content.append({"type": "text", "text": clip_line})
            audio_path = Path(cand.audio_path).resolve()
            content.append({"type": "audio", "audio": str(audio_path)})

        prompt_text = self._listwise_prompt_template.format(
            query=query,
            enumerated_clips="\n".join(clip_lines),
        )
        content.append({"type": "text", "text": prompt_text})

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": self._system_prompt}]},
            {"role": "user", "content": content},
        ]
        return messages, prompt_text

    def _parse_listwise_response(self, response: str, num_candidates: int) -> List[int]:
        if not response.strip():
            return []

        regions: List[str] = []
        if self._listwise_response_regex is not None:
            match = self._listwise_response_regex.search(response)
            if match:
                regions.append(match.group(1))
        regions.append(response)

        indices: List[int] = []
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
        spelled_pattern = re.compile(
            r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\b",
            re.IGNORECASE,
        )

        for region in regions:
            for token in re.findall(r"\d+", region):
                try:
                    value = int(token)
                except ValueError:
                    continue
                zero_idx = value - 1
                if 0 <= zero_idx < num_candidates and zero_idx not in indices:
                    indices.append(zero_idx)
            if len(indices) >= num_candidates:
                break
            if len(indices) < num_candidates:
                for word in spelled_pattern.findall(region):
                    value = word_map[word.lower()]
                    zero_idx = value - 1
                    if 0 <= zero_idx < num_candidates and zero_idx not in indices:
                        indices.append(zero_idx)
            if len(indices) >= num_candidates:
                break
        return indices

    def _rank_pointwise(
        self,
        query: str,
        candidates: List[CandidateItem],
        *,
        fallback_reason: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[int], Dict[str, Any]]:
        scores_map: Dict[int, float] = {}
        raw_map: Dict[int, str] = {}
        scored: List[Tuple[int, float]] = []
        for idx, cand in enumerate(candidates):
            parsed = self._score_query_audio(query, Path(cand.audio_path))
            scores_map[cand.candidate_idx] = float(parsed.value)
            raw_map[cand.candidate_idx] = parsed.raw_text
            scored.append((idx, parsed.value))

        scored.sort(key=lambda x: x[1], reverse=True)
        order = [idx for idx, _ in scored]
        info: Dict[str, Any] = {
            "used_listwise": False,
            "pointwise_scores": scores_map,
            "raw_responses": raw_map,
        }
        if fallback_reason:
            info["fallback_reason"] = fallback_reason
        if context:
            info.update(context)
        return order, info

    def _rank_audio_listwise(
        self,
        query: str,
        candidates: List[CandidateItem],
    ) -> Tuple[List[int], Dict[str, Any]]:
        if self._listwise_prompt_template is None:
            raise QwenOmniDependenciesMissing("Listwise prompt template is not configured.")
        if process_mm_info is None:
            raise QwenOmniDependenciesMissing(
                "Qwen Omni listwise ranking requires the `qwen-omni-utils` package."
            )

        self._listwise_stats["listwise_calls"] += 1

        messages, prompt_text = self._build_listwise_messages(query, candidates)
        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        try:
            audio_inputs, image_inputs, video_inputs = process_mm_info(
                messages,
                use_audio_in_video=False,
            )
        except Exception as exc:  # pragma: no cover - propagate more helpful message
            self._listwise_stats["errors"] += 1
            raise QwenOmniDependenciesMissing(
                "Failed to preprocess audio for Qwen Omni listwise ranking. "
                "Ensure `qwen-omni-utils` and its dependencies are installed."
            ) from exc

        if audio_inputs is None:
            self._listwise_stats["errors"] += 1
            raise QwenOmniDependenciesMissing(
                "Listwise ranking did not receive any audio inputs after preprocessing."
            )

        model_inputs = self.processor(
            text=prompt,
            audio=audio_inputs,
            images=image_inputs,
            videos=video_inputs,
            sampling_rate=self._target_sr,
            return_tensors="pt",
            padding=True,
        )
        model_inputs = model_inputs.to(self.model.device)
        model_inputs = model_inputs.to(self._dtype)

        pad_token_id = getattr(self.model.generation_config, "pad_token_id", None)
        eos_token_id = getattr(self.model.generation_config, "eos_token_id", None)
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id

        with self._torch.inference_mode():
            generation = self.model.generate(
                **model_inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

        sequences = generation
        if hasattr(generation, "sequences"):
            sequences = generation.sequences
        elif isinstance(generation, tuple):
            sequences = generation[0]

        prompt_length = model_inputs["input_ids"].shape[-1]
        continuation = sequences[:, prompt_length:]
        response = self.processor.batch_decode(
            continuation,
            skip_special_tokens=True,
        )[0]

        order = self._parse_listwise_response(response, len(candidates))
        if not order:
            self._listwise_stats["parse_failures"] += 1
            context = {
                "listwise_prompt": prompt_text,
                "listwise_response": response,
            }
            fallback_order, fallback_info = self._rank_pointwise(
                query,
                candidates,
                fallback_reason="listwise_parse_failure",
                context=context,
            )
            return fallback_order, fallback_info

        info: Dict[str, Any] = {
            "used_listwise": True,
            "listwise_prompt": prompt_text,
            "listwise_response": response,
            "parsed_order": [candidates[idx].candidate_idx for idx in order],
            "partial_order": False,
        }

        if len(order) < len(candidates):
            self._listwise_stats["partial_orders"] += 1
            fallback_order, fallback_info = self._rank_pointwise(
                query,
                candidates,
                fallback_reason="listwise_partial_completion",
            )
            seen = set(order)
            for idx in fallback_order:
                if idx not in seen:
                    order.append(idx)
            info["partial_order"] = True
            info["completed_with"] = [
                candidates[idx].candidate_idx for idx in fallback_order if idx not in seen
            ]
            info["pointwise_scores"] = fallback_info.get("pointwise_scores")
            info["raw_responses"] = fallback_info.get("raw_responses")

        return order, info

    # ----------------------------------------------------------------- protocol
    def score_audio(self, query: str, audio_path: Path) -> float:
        parsed = self._score_query_audio(query, Path(audio_path))
        return parsed.value

    def score_caption(self, query: str, caption: str) -> float:
        raise NotImplementedError("Caption scoring is not supported by Qwen Omni rerankers.")

    def rank_audio(self, query: str, candidates: List[CandidateItem]) -> List[int]:
        if not candidates:
            self._last_rank_info = {
                "used_listwise": bool(self._listwise_prompt_template),
                "parsed_order": [],
                "partial_order": False,
            }
            return []

        if self._listwise_prompt_template:
            try:
                order, info = self._rank_audio_listwise(query, candidates)
                self._last_rank_info = info
                return order
            except Exception as exc:  # pragma: no cover - fallback robustness
                self._listwise_stats["errors"] += 1
                warnings.warn(
                    f"Falling back to pointwise reranking due to listwise error: {exc}",
                    RuntimeWarning,
                )
                fallback_order, fallback_info = self._rank_pointwise(
                    query,
                    candidates,
                    fallback_reason="listwise_error",
                    context={"exception": str(exc)},
                )
                self._last_rank_info = fallback_info
                return fallback_order

        order, info = self._rank_pointwise(query, candidates)
        self._last_rank_info = info
        return order

    def rank_captions(self, query: str, captions: List[str]) -> List[int]:
        raise NotImplementedError("Caption ranking is not supported by Qwen Omni rerankers.")

    def generate_caption(self, audio_path: Path) -> str:
        raise NotImplementedError("Caption generation is not supported by Qwen Omni rerankers.")

    def get_last_rank_info(self) -> Optional[Dict[str, Any]]:
        if self._last_rank_info is None:
            return None
        return dict(self._last_rank_info)

    def get_listwise_stats(self) -> Dict[str, int]:
        return dict(self._listwise_stats)


class QwenOmniLALMClient(BaseQwenOmniClient):
    """Client targeting Qwen2.5-Omni checkpoints."""

    default_model_name = "Qwen/Qwen2.5-Omni-7B"


class Qwen3OmniLALMClient(BaseQwenOmniClient):
    """Client targeting Qwen3-Omni checkpoints."""

    default_model_name = "Qwen/Qwen3-Omni-7B-Instruct"
