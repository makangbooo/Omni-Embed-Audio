"""Local-only adapters for the frozen BGE and Whisper components.

Heavy libraries are imported only when an adapter is loaded. Importing this
module, validating configuration, or running its pure helpers never loads a
model and never initializes CUDA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .schema import NBestHypothesis


def _nonempty_path(path: Path | str, *, field: str) -> Path:
    value = Path(path).expanduser()
    if not value.is_dir():
        raise FileNotFoundError(f"{field} is not a local model directory: {value}")
    return value.resolve()


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def batched(values: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    """Yield stable, non-empty slices without changing input order."""

    _positive_integer(batch_size, field="batch_size")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    """Return float32 row-normalized embeddings and reject zero/invalid rows."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("embeddings must be a non-empty rank-2 array")
    if not np.isfinite(array).all():
        raise ValueError("embeddings contain non-finite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("cannot normalize a zero embedding row")
    result = array / norms
    if not np.isfinite(result).all():
        raise ValueError("normalized embeddings contain non-finite values")
    return result.astype(np.float32, copy=False)


def scalar_logits(values: np.ndarray, *, expected_rows: int) -> np.ndarray:
    """Validate a single-logit cross-encoder output.

    A two-class head is intentionally rejected instead of guessing which logit
    represents relevance.
    """

    _positive_integer(expected_rows, field="expected_rows")
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or array.shape[0] != expected_rows:
        raise ValueError(
            "reranker must return exactly one scalar logit per query-document pair"
        )
    result = array.astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("reranker logits contain non-finite values")
    return result


def whisper_valid_token_statistics(
    *,
    token_ids: Sequence[int],
    transition_logprobs: Sequence[float],
    ignored_token_ids: Sequence[int],
) -> tuple[float, int]:
    """Compute the reproducible proxy score used for N-best normalization.

    ``transition_logprobs`` must correspond exactly to the generated suffix in
    ``token_ids``. Padding, EOS, and decoder special tokens supplied by the
    caller are excluded. No posterior probability is fabricated here.
    """

    if len(token_ids) != len(transition_logprobs):
        raise ValueError("token_ids and transition_logprobs length mismatch")
    ignored = {int(value) for value in ignored_token_ids}
    retained = [
        float(score)
        for token, score in zip(token_ids, transition_logprobs)
        if int(token) not in ignored
    ]
    if not retained:
        raise ValueError("Whisper hypothesis has no valid generated token")
    if any(not math.isfinite(value) for value in retained):
        raise ValueError("transition log probabilities must be finite")
    return float(sum(retained) / len(retained)), len(retained)


@dataclass(frozen=True)
class FrozenModelIdentity:
    name: str
    revision: str
    local_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("model name must be non-empty")
        if not isinstance(self.revision, str) or len(self.revision) != 40:
            raise ValueError("model revision must be a 40-character commit")
        if any(character not in "0123456789abcdef" for character in self.revision):
            raise ValueError("model revision must be lowercase hexadecimal")


@dataclass(frozen=True)
class BgeDenseSettings:
    identity: FrozenModelIdentity
    max_length: int = 512
    expected_dimension: int = 768
    normalize: bool = True

    def __post_init__(self) -> None:
        _positive_integer(self.max_length, field="max_length")
        _positive_integer(self.expected_dimension, field="expected_dimension")


class BgeDenseEncoder:
    """Frozen CLS-pooled BGE encoder loaded from an audited local directory."""

    def __init__(self, settings: BgeDenseSettings, *, device: str, dtype: str):
        self.settings = settings
        self.device = device
        self.dtype = dtype
        self._torch = None
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        model_path = _nonempty_path(
            self.settings.identity.local_path,
            field="BGE dense local_path",
        )
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised on GPU host
            raise RuntimeError("torch and transformers are required for BGE inference") from exc
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda is unavailable")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("dtype must be float32 or bfloat16")
        torch_dtype = torch.bfloat16 if self.dtype == "bfloat16" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
        )
        model = AutoModel.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch_dtype,
        )
        model.eval()
        model.requires_grad_(False)
        model.to(self.device)
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model

    def encode(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        if self._model is None or self._tokenizer is None or self._torch is None:
            raise RuntimeError("BGE dense encoder must be loaded before encode")
        if not texts:
            raise ValueError("texts must not be empty")
        if any(not isinstance(value, str) for value in texts):
            raise TypeError("all texts must be strings")
        chunks = []
        for values in batched(texts, batch_size):
            encoded = self._tokenizer(
                list(values),
                max_length=self.settings.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(self.device, non_blocking=True)
                for key, value in encoded.items()
            }
            with self._torch.inference_mode():
                output = self._model(**encoded)
                pooled = output.last_hidden_state[:, 0]
            chunks.append(pooled.float().cpu().numpy())
        embeddings = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        expected = (len(texts), self.settings.expected_dimension)
        if embeddings.shape != expected:
            raise ValueError(
                f"unexpected BGE embedding shape {embeddings.shape}; expected {expected}"
            )
        return l2_normalize_rows(embeddings) if self.settings.normalize else embeddings


@dataclass(frozen=True)
class BgeRerankerSettings:
    identity: FrozenModelIdentity
    max_length: int = 512

    def __post_init__(self) -> None:
        _positive_integer(self.max_length, field="max_length")


class BgeCrossEncoder:
    """Frozen BGE v2-M3 single-logit cross-encoder."""

    def __init__(self, settings: BgeRerankerSettings, *, device: str, dtype: str):
        self.settings = settings
        self.device = device
        self.dtype = dtype
        self._torch = None
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        model_path = _nonempty_path(
            self.settings.identity.local_path,
            field="BGE reranker local_path",
        )
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised on GPU host
            raise RuntimeError("torch and transformers are required for CE inference") from exc
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda is unavailable")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("dtype must be float32 or bfloat16")
        torch_dtype = torch.bfloat16 if self.dtype == "bfloat16" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch_dtype,
        )
        model.eval()
        model.requires_grad_(False)
        model.to(self.device)
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model

    def score(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
    ) -> np.ndarray:
        if self._model is None or self._tokenizer is None or self._torch is None:
            raise RuntimeError("BGE cross-encoder must be loaded before score")
        if not pairs:
            raise ValueError("pairs must not be empty")
        if any(
            not isinstance(pair, tuple)
            or len(pair) != 2
            or any(not isinstance(value, str) for value in pair)
            for pair in pairs
        ):
            raise TypeError("pairs must be (query, document) string tuples")
        chunks = []
        for values in batched(pairs, batch_size):
            encoded = self._tokenizer(
                list(values),
                max_length=self.settings.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(self.device, non_blocking=True)
                for key, value in encoded.items()
            }
            with self._torch.inference_mode():
                logits = self._model(**encoded, return_dict=True).logits
            chunks.append(
                scalar_logits(
                    logits.float().cpu().numpy(),
                    expected_rows=len(values),
                )
            )
        result = np.concatenate(chunks).astype(np.float32, copy=False)
        if result.shape != (len(pairs),):
            raise ValueError("cross-encoder output count mismatch")
        return result


@dataclass(frozen=True)
class WhisperSettings:
    identity: FrozenModelIdentity
    num_beams: int = 4
    num_return_sequences: int = 4
    language: str = "en"
    task: str = "transcribe"

    def __post_init__(self) -> None:
        _positive_integer(self.num_beams, field="num_beams")
        _positive_integer(self.num_return_sequences, field="num_return_sequences")
        if self.num_return_sequences > self.num_beams:
            raise ValueError("num_return_sequences cannot exceed num_beams")
        if self.language != "en" or self.task != "transcribe":
            raise ValueError("formal protocol is locked to English transcription")


def load_audio_mono(
    path: Path | str,
    *,
    target_sample_rate: int = 16_000,
) -> np.ndarray:
    """Decode, mix to mono, and explicitly resample an audio query."""

    _positive_integer(target_sample_rate, field="target_sample_rate")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"audio file is missing: {source}")
    try:
        import soundfile
        from scipy.signal import resample_poly
    except ImportError as exc:  # pragma: no cover - exercised on remote host
        raise RuntimeError("soundfile and scipy are required for audio loading") from exc
    waveform, source_rate = soundfile.read(
        str(source),
        dtype="float32",
        always_2d=True,
    )
    if waveform.ndim != 2 or waveform.shape[0] == 0 or waveform.shape[1] == 0:
        raise ValueError(f"decoded audio is empty: {source}")
    if not np.isfinite(waveform).all():
        raise ValueError(f"decoded audio contains non-finite samples: {source}")
    mono = waveform.mean(axis=1, dtype=np.float32)
    if source_rate != target_sample_rate:
        common = gcd(int(source_rate), target_sample_rate)
        mono = resample_poly(
            mono,
            target_sample_rate // common,
            int(source_rate) // common,
        ).astype(np.float32, copy=False)
    if mono.size == 0 or not np.isfinite(mono).all():
        raise ValueError(f"resampled audio is invalid: {source}")
    return mono


def build_whisper_hypotheses(
    *,
    decoded_texts: Sequence[str],
    generated_token_ids: Sequence[Sequence[int]],
    transition_logprobs: Sequence[Sequence[float]],
    sequence_scores: Sequence[float] | None,
    ignored_token_ids: Sequence[int],
) -> tuple[NBestHypothesis, ...]:
    """Convert guarded Whisper beam outputs to the strict artifact schema."""

    count = len(decoded_texts)
    if count == 0:
        raise ValueError("Whisper returned no hypotheses")
    if len(generated_token_ids) != count or len(transition_logprobs) != count:
        raise ValueError("Whisper output field counts differ")
    if sequence_scores is not None and len(sequence_scores) != count:
        raise ValueError("Whisper sequence score count differs")
    result = []
    for index in range(count):
        average, valid_count = whisper_valid_token_statistics(
            token_ids=generated_token_ids[index],
            transition_logprobs=transition_logprobs[index],
            ignored_token_ids=ignored_token_ids,
        )
        sequence_score = (
            None if sequence_scores is None else float(sequence_scores[index])
        )
        result.append(
            NBestHypothesis(
                rank=index + 1,
                text=str(decoded_texts[index]).strip(),
                sequence_score=sequence_score,
                average_token_logprob=average,
                valid_token_count=valid_count,
            )
        )
    return tuple(result)


class WhisperNBestGenerator:
    """Deterministic four-beam Whisper transcription from local weights."""

    def __init__(self, settings: WhisperSettings, *, device: str, dtype: str):
        self.settings = settings
        self.device = device
        self.dtype = dtype
        self._torch = None
        self._processor = None
        self._model = None

    def load(self) -> None:
        model_path = _nonempty_path(
            self.settings.identity.local_path,
            field="Whisper local_path",
        )
        try:
            import torch
            from transformers import AutoProcessor, WhisperForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - exercised on GPU host
            raise RuntimeError(
                "torch and transformers are required for Whisper inference"
            ) from exc
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda is unavailable")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("dtype must be float32 or bfloat16")
        torch_dtype = torch.bfloat16 if self.dtype == "bfloat16" else torch.float32
        processor = AutoProcessor.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch_dtype,
        )
        model.eval()
        model.requires_grad_(False)
        model.to(self.device)
        self._torch = torch
        self._processor = processor
        self._model = model

    def generate(
        self,
        waveform: np.ndarray,
        *,
        sample_rate: int = 16_000,
    ) -> tuple[NBestHypothesis, ...]:
        """Generate N-best hypotheses and token-level proxy scores.

        The returned average token log probabilities are reproducible proxy
        logits, not calibrated ASR posterior probabilities.
        """

        if self._model is None or self._processor is None or self._torch is None:
            raise RuntimeError("Whisper generator must be loaded before generate")
        if sample_rate != 16_000:
            raise ValueError("Whisper input must be explicitly resampled to 16 kHz")
        array = np.asarray(waveform, dtype=np.float32)
        if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
            raise ValueError("waveform must be a non-empty finite mono array")
        processed = self._processor(
            array,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        model_inputs = {
            key: value.to(self.device, non_blocking=True)
            for key, value in processed.items()
            if key in {"input_features", "attention_mask"}
        }
        if "input_features" not in model_inputs:
            raise RuntimeError("Whisper processor did not return input_features")
        with self._torch.inference_mode():
            outputs = self._model.generate(
                **model_inputs,
                language=self.settings.language,
                task=self.settings.task,
                do_sample=False,
                num_beams=self.settings.num_beams,
                num_return_sequences=self.settings.num_return_sequences,
                return_dict_in_generate=True,
                output_scores=True,
                return_timestamps=False,
            )
            transition = self._model.compute_transition_scores(
                outputs.sequences,
                outputs.scores,
                outputs.beam_indices,
                normalize_logits=True,
            )
        if transition.ndim != 2 or transition.shape[0] != self.settings.num_return_sequences:
            raise ValueError(
                f"unexpected Whisper transition-score shape {tuple(transition.shape)}"
            )
        step_count = int(transition.shape[1])
        if step_count <= 0 or outputs.sequences.shape[1] < step_count:
            raise ValueError("Whisper returned no generated transition scores")
        suffix_tokens = outputs.sequences[:, -step_count:]
        decoded = self._processor.batch_decode(
            outputs.sequences,
            skip_special_tokens=True,
        )
        raw_sequence_scores = getattr(outputs, "sequences_scores", None)
        sequence_scores = (
            None
            if raw_sequence_scores is None
            else raw_sequence_scores.float().cpu().tolist()
        )
        rows = list(
            zip(
                decoded,
                suffix_tokens.cpu().tolist(),
                transition.float().cpu().tolist(),
                sequence_scores
                if sequence_scores is not None
                else [None] * len(decoded),
            )
        )
        if sequence_scores is not None:
            rows.sort(key=lambda row: float(row[3]), reverse=True)
        return build_whisper_hypotheses(
            decoded_texts=[row[0] for row in rows],
            generated_token_ids=[row[1] for row in rows],
            transition_logprobs=[row[2] for row in rows],
            sequence_scores=(
                None if sequence_scores is None else [float(row[3]) for row in rows]
            ),
            ignored_token_ids=self._processor.tokenizer.all_special_ids,
        )


def model_identity_from_config(config: Mapping[str, Any], key: str) -> FrozenModelIdentity:
    """Resolve and validate a fixed local model identity from main config."""

    models = config.get("models")
    if not isinstance(models, Mapping) or not isinstance(models.get(key), Mapping):
        raise KeyError(f"missing models.{key}")
    value = models[key]
    local_path = value.get("local_path")
    if key == "oea":
        local_path = value.get("base_path")
    return FrozenModelIdentity(
        name=value["name"],
        revision=value["revision"],
        local_path=Path(local_path),
    )
