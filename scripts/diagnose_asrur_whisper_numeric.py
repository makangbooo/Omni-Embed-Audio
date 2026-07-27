#!/usr/bin/env python3
"""Diagnose one frozen Whisper N-best numerical failure without changing caches."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from AudioRetrieval.asr_uncertainty_reranking.model_adapters import (  # noqa: E402
    WhisperNBestGenerator,
    WhisperSettings,
    load_audio_mono,
    model_identity_from_config,
    prepare_whisper_model_inputs,
    whisper_decoder_prompt_tokens,
    whisper_teacher_forced_generated_logprobs,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_class(value: float) -> str:
    value = float(value)
    if math.isnan(value):
        return "nan"
    if value == math.inf:
        return "positive_infinity"
    if value == -math.inf:
        return "negative_infinity"
    return "finite"


def summarize_token_scores(
    *,
    method: str,
    token_ids: Sequence[Sequence[int]],
    logprobs: Sequence[Sequence[float]],
    decoded_texts: Sequence[str],
    sequence_scores: Sequence[float],
    tokenizer: Any,
    ignored_token_ids: Sequence[int],
) -> dict[str, Any]:
    if not (
        len(token_ids)
        == len(logprobs)
        == len(decoded_texts)
        == len(sequence_scores)
    ):
        raise ValueError("diagnostic Whisper field counts differ")
    ignored = {int(value) for value in ignored_token_ids}
    hypotheses: list[dict[str, Any]] = []
    total_valid = 0
    total_nonfinite_valid = 0
    for hypothesis_index, (
        hypothesis_tokens,
        hypothesis_scores,
        decoded_text,
        sequence_score,
    ) in enumerate(
        zip(token_ids, logprobs, decoded_texts, sequence_scores),
        start=1,
    ):
        if len(hypothesis_tokens) != len(hypothesis_scores):
            raise ValueError("diagnostic token and score lengths differ")
        rows: list[dict[str, Any]] = []
        valid_values: list[float] = []
        nonfinite_valid = 0
        for position, (token_id, score) in enumerate(
            zip(hypothesis_tokens, hypothesis_scores)
        ):
            token_id = int(token_id)
            score = float(score)
            score_class = numeric_class(score)
            special = token_id in ignored
            if not special:
                total_valid += 1
                if score_class == "finite":
                    valid_values.append(score)
                else:
                    nonfinite_valid += 1
                    total_nonfinite_valid += 1
            token_text = tokenizer.convert_ids_to_tokens(token_id)
            rows.append(
                {
                    "position": position,
                    "token_id": token_id,
                    "token": str(token_text),
                    "special": special,
                    "logprob_class": score_class,
                    "logprob": score if score_class == "finite" else None,
                }
            )
        hypotheses.append(
            {
                "rank": hypothesis_index,
                "decoded_text": str(decoded_text),
                "sequence_score": float(sequence_score),
                "generated_token_count": len(hypothesis_tokens),
                "valid_token_count": len(valid_values) + nonfinite_valid,
                "finite_valid_token_count": len(valid_values),
                "nonfinite_valid_token_count": nonfinite_valid,
                "finite_valid_average_logprob": (
                    sum(valid_values) / len(valid_values) if valid_values else None
                ),
                "tokens": rows,
            }
        )
    return {
        "method": method,
        "hypothesis_count": len(hypotheses),
        "valid_token_count": total_valid,
        "nonfinite_valid_token_count": total_nonfinite_valid,
        "all_valid_token_scores_finite": total_nonfinite_valid == 0,
        "hypotheses": hypotheses,
    }


def load_exact_record(path: Path, record_id: str) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_id") == record_id:
                row["_manifest_line_number"] = line_number
                matches.append(row)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one manifest record for {record_id!r}, "
            f"observed {len(matches)}"
        )
    return matches[0]


def score_sequences(
    *,
    torch_module: Any,
    model: Any,
    model_inputs: Mapping[str, Any],
    sequences: Any,
    prompt_length: int,
    per_hypothesis: bool,
) -> tuple[Any, Any]:
    if not per_hypothesis:
        return whisper_teacher_forced_generated_logprobs(
            torch_module=torch_module,
            model=model,
            model_inputs=model_inputs,
            sequences=sequences,
            prompt_length=prompt_length,
        )
    token_parts = []
    score_parts = []
    for index in range(int(sequences.shape[0])):
        tokens, scores = whisper_teacher_forced_generated_logprobs(
            torch_module=torch_module,
            model=model,
            model_inputs=model_inputs,
            sequences=sequences[index : index + 1],
            prompt_length=prompt_length,
        )
        token_parts.append(tokens)
        score_parts.append(scores)
    return (
        torch_module.cat(token_parts, dim=0),
        torch_module.cat(score_parts, dim=0),
    )


def formal_guard() -> str:
    offline = {
        key: os.environ.get(key)
        for key in (
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "HF_DATASETS_OFFLINE",
        )
    }
    if set(offline.values()) != {"1"}:
        raise RuntimeError(f"strict offline environment is required: {offline}")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("formal diagnostic requires a clean Git worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run(args: argparse.Namespace) -> int:
    git_commit = formal_guard()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    record = load_exact_record(args.manifest, args.record_id)
    audio_path = Path(str(record["audio_path"])).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    waveform = load_audio_mono(audio_path, target_sample_rate=16_000)

    identity = model_identity_from_config(config, "asr")
    settings = WhisperSettings(
        identity=identity,
        num_beams=config["models"]["asr"]["num_beams"],
        num_return_sequences=config["models"]["asr"]["num_return_sequences"],
        language=config["models"]["asr"]["language"],
        task=config["models"]["asr"]["task"],
    )
    generator = WhisperNBestGenerator(
        settings,
        device=args.device,
        dtype="bfloat16",
    )
    generator.load()
    torch = generator._torch
    processor = generator._processor
    model = generator._model
    base_generate = generator._base_generate
    if any(value is None for value in (torch, processor, model, base_generate)):
        raise RuntimeError("Whisper diagnostic model failed to initialize")
    torch.cuda.reset_peak_memory_stats()

    processed = processor(
        np.asarray(waveform, dtype=np.float32),
        sampling_rate=16_000,
        return_tensors="pt",
        return_attention_mask=True,
    )
    bf16_inputs = prepare_whisper_model_inputs(
        processed,
        device=args.device,
        floating_dtype=model.dtype,
    )
    generation_config = copy.deepcopy(model.generation_config)
    decoder_prompt = whisper_decoder_prompt_tokens(
        processor=processor,
        decoder_start_token_id=generation_config.decoder_start_token_id,
        language=settings.language,
        task=settings.task,
    )
    generation_config.forced_decoder_ids = None
    generation_config.do_sample = False
    generation_config.num_beams = settings.num_beams
    generation_config.num_return_sequences = settings.num_return_sequences
    generation_config.return_dict_in_generate = True
    generation_config.output_scores = True
    decoder_input_ids = torch.tensor(
        [decoder_prompt],
        dtype=torch.long,
        device=args.device,
    )
    with torch.inference_mode():
        outputs = base_generate(
            **bf16_inputs,
            generation_config=generation_config,
            decoder_input_ids=decoder_input_ids,
        )
    sequences = outputs.sequences
    raw_sequence_scores = outputs.sequences_scores
    decoded_texts = processor.batch_decode(sequences, skip_special_tokens=True)
    sequence_scores = raw_sequence_scores.float().cpu().tolist()
    ignored_token_ids = processor.tokenizer.all_special_ids
    del outputs

    methods: list[dict[str, Any]] = []
    with torch.inference_mode():
        tokens, scores = score_sequences(
            torch_module=torch,
            model=model,
            model_inputs=bf16_inputs,
            sequences=sequences,
            prompt_length=len(decoder_prompt),
            per_hypothesis=False,
        )
    methods.append(
        summarize_token_scores(
            method="bfloat16_weights_batched_four_hypotheses",
            token_ids=tokens.cpu().tolist(),
            logprobs=scores.float().cpu().tolist(),
            decoded_texts=decoded_texts,
            sequence_scores=sequence_scores,
            tokenizer=processor.tokenizer,
            ignored_token_ids=ignored_token_ids,
        )
    )

    with torch.inference_mode():
        tokens, scores = score_sequences(
            torch_module=torch,
            model=model,
            model_inputs=bf16_inputs,
            sequences=sequences,
            prompt_length=len(decoder_prompt),
            per_hypothesis=True,
        )
    methods.append(
        summarize_token_scores(
            method="bfloat16_weights_one_hypothesis_at_a_time",
            token_ids=tokens.cpu().tolist(),
            logprobs=scores.float().cpu().tolist(),
            decoded_texts=decoded_texts,
            sequence_scores=sequence_scores,
            tokenizer=processor.tokenizer,
            ignored_token_ids=ignored_token_ids,
        )
    )

    model.float()
    fp32_inputs = prepare_whisper_model_inputs(
        processed,
        device=args.device,
        floating_dtype=torch.float32,
    )
    with torch.inference_mode():
        tokens, scores = score_sequences(
            torch_module=torch,
            model=model,
            model_inputs=fp32_inputs,
            sequences=sequences,
            prompt_length=len(decoder_prompt),
            per_hypothesis=False,
        )
    methods.append(
        summarize_token_scores(
            method="float32_upcast_weights_batched_exact_bfloat16_beam_sequences",
            token_ids=tokens.cpu().tolist(),
            logprobs=scores.float().cpu().tolist(),
            decoded_texts=decoded_texts,
            sequence_scores=sequence_scores,
            tokenizer=processor.tokenizer,
            ignored_token_ids=ignored_token_ids,
        )
    )

    payload = {
        "schema_version": 1,
        "status": "complete",
        "diagnostic_only": True,
        "created_at": utc_now(),
        "git_commit": git_commit,
        "record": record,
        "audio": {
            "path": str(audio_path),
            "size_bytes": audio_path.stat().st_size,
            "sha256": sha256_file(audio_path),
            "decoded_samples_at_16khz": int(waveform.shape[0]),
            "finite": bool(np.isfinite(waveform).all()),
            "minimum": float(waveform.min()),
            "maximum": float(waveform.max()),
            "rms": float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64)))),
        },
        "model": {
            "name": identity.name,
            "revision": identity.revision,
            "local_path": str(identity.local_path),
            "generation_dtype": "bfloat16",
            "num_beams": settings.num_beams,
            "num_return_sequences": settings.num_return_sequences,
            "language": settings.language,
            "task": settings.task,
        },
        "generated_sequence_shape": [
            int(value) for value in sequences.shape
        ],
        "decoder_prompt_token_ids": [int(value) for value in decoder_prompt],
        "sequence_scores_all_finite": all(
            math.isfinite(float(value)) for value in sequence_scores
        ),
        "methods": methods,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "cache_mutation_performed": False,
        "training_performed": False,
        "metric_computation_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "status": "complete",
                "record_id": args.record_id,
                "methods": [
                    {
                        "method": value["method"],
                        "nonfinite_valid_token_count": value[
                            "nonfinite_valid_token_count"
                        ],
                        "all_valid_token_scores_finite": value[
                            "all_valid_token_scores_finite"
                        ],
                    }
                    for value in methods
                ],
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.manifest = args.manifest.resolve()
    args.output = args.output.resolve()
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
