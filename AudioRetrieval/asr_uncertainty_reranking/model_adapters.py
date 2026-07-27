"""Local-only adapters for the frozen BGE and Whisper components.

Heavy libraries are imported only when an adapter is loaded. Importing this
module, validating configuration, or running its pure helpers never loads a
model and never initializes CUDA.
"""

from __future__ import annotations

import copy
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


def prepare_whisper_model_inputs(
    processed: Mapping[str, Any],
    *,
    device: str,
    floating_dtype: Any,
) -> dict[str, Any]:
    """Move Whisper inputs and match floating tensors to model precision.

    Whisper feature extraction emits float32 log-mel values. A model loaded
    directly in BF16 does not autocast its first convolution, so floating
    inputs must explicitly match the frozen model dtype. Integer and boolean
    tensors such as attention masks retain their original dtype.
    """

    result: dict[str, Any] = {}
    for key, value in processed.items():
        if key not in {"input_features", "attention_mask"}:
            continue
        move_arguments: dict[str, Any] = {
            "device": device,
            "non_blocking": True,
        }
        if value.is_floating_point():
            move_arguments["dtype"] = floating_dtype
        result[key] = value.to(**move_arguments)
    return result


def whisper_decoder_prompt_tokens(
    *,
    processor: Any,
    decoder_start_token_id: int | None,
    language: str,
    task: str,
) -> tuple[int, ...]:
    """Build an explicit, validated Whisper decoder prompt.

    The Transformers 4.52.4 Whisper wrapper expands ``num_return_sequences``
    before its internal beam search and later re-stacks scores while retaining
    global beam indices. The formal four-best path consequently uses the base
    ``GenerationMixin`` directly. Supplying decoder input IDs explicitly keeps
    the language/task/no-timestamps protocol identical without relying on the
    wrapper's mutable ``forced_decoder_ids`` handling.
    """

    if isinstance(decoder_start_token_id, bool) or not isinstance(
        decoder_start_token_id, int
    ):
        raise ValueError("Whisper decoder_start_token_id must be an integer")
    prompt = processor.get_decoder_prompt_ids(
        language=language,
        task=task,
        no_timestamps=True,
    )
    if not isinstance(prompt, Sequence) or not prompt:
        raise ValueError("Whisper processor returned an empty decoder prompt")
    tokens = [decoder_start_token_id]
    for expected_position, value in enumerate(prompt, start=1):
        if (
            not isinstance(value, Sequence)
            or len(value) != 2
            or value[0] != expected_position
            or isinstance(value[1], bool)
            or not isinstance(value[1], int)
        ):
            raise ValueError(
                "Whisper decoder prompt must contain contiguous integer "
                "(position, token_id) pairs"
            )
        tokens.append(int(value[1]))
    return tuple(tokens)


class WhisperGenerationStageError(RuntimeError):
    """Preserve the exact failing stage of the frozen Whisper pipeline."""

    def __init__(self, stage: str, message: str):
        if not isinstance(stage, str) or not stage:
            raise ValueError("Whisper failure stage must be non-empty")
        super().__init__(f"Whisper stage={stage} failed: {message}")
        self.stage = stage


def whisper_teacher_forced_generated_logprobs(
    *,
    torch_module: Any,
    model: Any,
    model_inputs: Mapping[str, Any],
    sequences: Any,
    prompt_length: int,
) -> tuple[Any, Any]:
    """Score generated suffixes without generation-time beam bookkeeping.

    Transformers generation transition scores include logits-processor state
    and beam ancestry. Whisper's custom multi-return wrapper and the generic
    beam implementation have produced incompatible ancestry/score tensors in
    the pinned 4.52.4 environment. After bypassing that wrapper, the audited
    real-model smoke still returned a non-finite transition score for a
    retained non-special token. Its processor-specific origin is unknown, so
    this protocol does not filter, clamp, or interpret that value.

    The ASR uncertainty proxy therefore uses frozen-model, teacher-forced
    conditional log probabilities for the exact generated token sequences.
    Prompt tokens are excluded, logits are normalized in float32, and the
    caller still retains the beam search ``sequence_score`` separately. This
    is a reproducible proxy likelihood, not a calibrated ASR posterior.
    """

    if isinstance(prompt_length, bool) or not isinstance(prompt_length, int):
        raise ValueError("Whisper prompt_length must be an integer")
    if prompt_length <= 0:
        raise ValueError("Whisper prompt_length must be positive")
    if getattr(sequences, "ndim", None) != 2 or int(sequences.shape[0]) <= 0:
        raise ValueError("Whisper sequences must be a non-empty rank-2 tensor")
    if int(sequences.shape[1]) <= prompt_length:
        raise ValueError("Whisper returned no tokens after the decoder prompt")
    input_features = model_inputs.get("input_features")
    if input_features is None:
        raise ValueError("Whisper model inputs do not contain input_features")

    encoder_arguments: dict[str, Any] = {
        "input_features": input_features,
        "return_dict": True,
    }
    if model_inputs.get("attention_mask") is not None:
        encoder_arguments["attention_mask"] = model_inputs["attention_mask"]
    encoder_outputs = model.get_encoder()(**encoder_arguments)
    encoder_hidden = getattr(encoder_outputs, "last_hidden_state", None)
    if encoder_hidden is None or getattr(encoder_hidden, "ndim", None) != 3:
        raise ValueError("Whisper encoder did not return rank-3 hidden states")

    sequence_count = int(sequences.shape[0])
    if int(encoder_hidden.shape[0]) != 1:
        raise ValueError(
            "formal Whisper scoring expects exactly one encoded audio query"
        )
    expanded_encoder_hidden = encoder_hidden.repeat_interleave(
        sequence_count,
        dim=0,
    )
    decoder_input_ids = sequences[:, :-1]
    target_token_ids = sequences[:, 1:]
    scoring_outputs = model(
        encoder_outputs=(expanded_encoder_hidden,),
        decoder_input_ids=decoder_input_ids,
        use_cache=False,
        return_dict=True,
    )
    logits = getattr(scoring_outputs, "logits", None)
    expected_logits_prefix = (
        sequence_count,
        int(decoder_input_ids.shape[1]),
    )
    if (
        logits is None
        or getattr(logits, "ndim", None) != 3
        or tuple(int(value) for value in logits.shape[:2])
        != expected_logits_prefix
    ):
        observed = (
            None
            if logits is None
            else tuple(int(value) for value in logits.shape)
        )
        raise ValueError(
            "unexpected Whisper teacher-forced logits shape "
            f"{observed}; expected prefix {expected_logits_prefix}"
        )

    token_negative_log_likelihood = torch_module.nn.functional.cross_entropy(
        logits.float().transpose(1, 2),
        target_token_ids,
        reduction="none",
    )
    token_logprobs = -token_negative_log_likelihood
    generated_token_ids = sequences[:, prompt_length:]
    generated_logprobs = token_logprobs[:, prompt_length - 1 :]
    if generated_token_ids.shape != generated_logprobs.shape:
        raise ValueError(
            "Whisper teacher-forced token/log-probability alignment mismatch"
        )
    return generated_token_ids, generated_logprobs


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
        self._base_generate = None

    def load(self) -> None:
        model_path = _nonempty_path(
            self.settings.identity.local_path,
            field="Whisper local_path",
        )
        try:
            import torch
            from transformers import AutoProcessor, WhisperForConditionalGeneration
            from transformers.generation import GenerationMixin
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
        self._base_generate = GenerationMixin.generate.__get__(model, type(model))

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

        if (
            self._model is None
            or self._processor is None
            or self._torch is None
            or self._base_generate is None
        ):
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
            return_attention_mask=True,
        )
        model_inputs = prepare_whisper_model_inputs(
            processed,
            device=self.device,
            floating_dtype=self._model.dtype,
        )
        if "input_features" not in model_inputs:
            raise RuntimeError("Whisper processor did not return input_features")
        generation_config = copy.deepcopy(self._model.generation_config)
        decoder_prompt = whisper_decoder_prompt_tokens(
            processor=self._processor,
            decoder_start_token_id=generation_config.decoder_start_token_id,
            language=self.settings.language,
            task=self.settings.task,
        )
        generation_config.forced_decoder_ids = None
        generation_config.do_sample = False
        generation_config.num_beams = self.settings.num_beams
        generation_config.num_return_sequences = self.settings.num_return_sequences
        generation_config.return_dict_in_generate = True
        generation_config.output_scores = True
        decoder_input_ids = self._torch.tensor(
            [decoder_prompt],
            dtype=self._torch.long,
            device=self.device,
        )
        with self._torch.inference_mode():
            try:
                outputs = self._base_generate(
                    **model_inputs,
                    generation_config=generation_config,
                    decoder_input_ids=decoder_input_ids,
                )
            except Exception as exc:
                raise WhisperGenerationStageError(
                    "four_beam_generation",
                    f"{type(exc).__name__}: {exc}",
                ) from exc
        if (
            getattr(outputs, "sequences", None) is None
            or outputs.sequences.ndim != 2
            or int(outputs.sequences.shape[0])
            != self.settings.num_return_sequences
        ):
            observed = getattr(getattr(outputs, "sequences", None), "shape", None)
            raise WhisperGenerationStageError(
                "four_beam_output_validation",
                "unexpected sequence shape "
                f"{None if observed is None else tuple(observed)}",
            )
        if int(outputs.sequences.shape[1]) <= len(decoder_prompt):
            raise WhisperGenerationStageError(
                "four_beam_output_validation",
                "no token was generated after the explicit decoder prompt",
            )
        expected_prompt = decoder_input_ids.expand(
            self.settings.num_return_sequences,
            -1,
        )
        if not self._torch.equal(
            outputs.sequences[:, : len(decoder_prompt)],
            expected_prompt,
        ):
            raise WhisperGenerationStageError(
                "four_beam_output_validation",
                "generated sequences do not preserve the explicit decoder prompt",
            )
        decoded = self._processor.batch_decode(
            outputs.sequences,
            skip_special_tokens=True,
        )
        raw_sequence_scores = getattr(outputs, "sequences_scores", None)
        if (
            raw_sequence_scores is None
            or raw_sequence_scores.ndim != 1
            or int(raw_sequence_scores.shape[0])
            != self.settings.num_return_sequences
        ):
            raise WhisperGenerationStageError(
                "four_beam_output_validation",
                "beam sequence scores are absent or have the wrong shape",
            )
        sequence_scores = raw_sequence_scores.float().cpu().tolist()
        if any(not math.isfinite(float(value)) for value in sequence_scores):
            raise WhisperGenerationStageError(
                "four_beam_output_validation",
                "beam sequence scores contain non-finite values",
            )
        sequences = outputs.sequences
        del outputs
        try:
            with self._torch.inference_mode():
                suffix_tokens, token_logprobs = (
                    whisper_teacher_forced_generated_logprobs(
                        torch_module=self._torch,
                        model=self._model,
                        model_inputs=model_inputs,
                        sequences=sequences,
                        prompt_length=len(decoder_prompt),
                    )
                )
        except Exception as exc:
            if isinstance(exc, WhisperGenerationStageError):
                raise
            raise WhisperGenerationStageError(
                "teacher_forced_conditional_logprob",
                f"{type(exc).__name__}: {exc}",
            ) from exc
        rows = list(
            zip(
                decoded,
                suffix_tokens.cpu().tolist(),
                token_logprobs.float().cpu().tolist(),
                sequence_scores,
            )
        )
        rows.sort(key=lambda row: float(row[3]), reverse=True)
        try:
            return build_whisper_hypotheses(
                decoded_texts=[row[0] for row in rows],
                generated_token_ids=[row[1] for row in rows],
                transition_logprobs=[row[2] for row in rows],
                sequence_scores=[float(row[3]) for row in rows],
                ignored_token_ids=self._processor.tokenizer.all_special_ids,
            )
        except Exception as exc:
            raise WhisperGenerationStageError(
                "nbest_artifact_validation",
                f"{type(exc).__name__}: {exc}",
            ) from exc


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
