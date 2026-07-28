#!/usr/bin/env python3
"""Compare the frozen Whisper protocol with the official generation entrypoint.

This diagnostic is intentionally small and non-mutating.  It transcribes the
same deterministic SQuTR-FiQA audio grid with:

1. the current BF16 four-best generic ``GenerationMixin`` protocol;
2. the official Whisper ``model.generate`` entrypoint in BF16; and
3. the official Whisper ``model.generate`` entrypoint in FP32.

The output separates an entrypoint failure from a precision failure without
silently changing any formal cache.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from AudioRetrieval.asr_uncertainty_reranking.aggregation import (  # noqa: E402
    normalized_word_edit_distance,
)
from AudioRetrieval.asr_uncertainty_reranking.configuration import (  # noqa: E402
    load_main_experiment_config,
)
from AudioRetrieval.asr_uncertainty_reranking.data import (  # noqa: E402
    load_squtr_audio_manifest,
    load_text_queries,
    squtr_subset_name,
)
from AudioRetrieval.asr_uncertainty_reranking.metrics import corpus_wer  # noqa: E402
from AudioRetrieval.asr_uncertainty_reranking.model_adapters import (  # noqa: E402
    WhisperNBestGenerator,
    WhisperSettings,
    load_audio_mono,
    model_identity_from_config,
    prepare_whisper_model_inputs,
)

CONDITIONS = ("clean", "snr_20", "snr_10", "snr_0")
METHODS = (
    "current_generic_bfloat16_four_best",
    "official_bfloat16_one_best",
    "official_float32_one_best",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--query-count", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def waveform_statistics(waveform: np.ndarray, *, sample_rate: int) -> dict[str, Any]:
    array = np.asarray(waveform, dtype=np.float32)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("waveform must be a non-empty finite mono vector")
    if sample_rate <= 0:
        raise ValueError("sample rate must be positive")
    return {
        "sample_count": int(array.size),
        "sample_rate": sample_rate,
        "duration_seconds": float(array.size / sample_rate),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "rms": float(np.sqrt(np.mean(np.square(array, dtype=np.float64)))),
        "nonzero_fraction": float(np.count_nonzero(array) / array.size),
        "pcm_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def select_complete_condition_grid(
    *,
    manifest_path: Path,
    query_count: int,
) -> list[Any]:
    if query_count <= 0:
        raise ValueError("query_count must be positive")
    records = [
        record
        for record in load_squtr_audio_manifest(
            manifest_path,
            require_audio_files=True,
        ).values()
        if squtr_subset_name(record.subset) == "fiqa"
    ]
    by_query: dict[str, dict[str, Any]] = {}
    for record in records:
        condition_records = by_query.setdefault(record.query_id, {})
        if record.condition in condition_records:
            raise ValueError(
                f"duplicate FiQA query/condition: {record.query_id}/{record.condition}"
            )
        condition_records[record.condition] = record
    complete_ids = sorted(
        query_id
        for query_id, values in by_query.items()
        if set(values) == set(CONDITIONS)
    )
    if len(complete_ids) < query_count:
        raise ValueError(
            f"only {len(complete_ids)} complete FiQA condition grids are available"
        )
    selected = [
        by_query[query_id][condition]
        for query_id in complete_ids[:query_count]
        for condition in CONDITIONS
    ]
    if len(selected) != query_count * len(CONDITIONS):
        raise AssertionError("selected grid size is inconsistent")
    return selected


def summarize_method(
    *,
    rows: Sequence[Mapping[str, Any]],
    references: Mapping[str, str],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("method rows must be non-empty")
    texts = [str(row["text"]) for row in rows]
    pairs = [(references[str(row["query_id"])], str(row["text"])) for row in rows]
    frequency = Counter(texts)
    return {
        "record_count": len(rows),
        "unique_transcript_count": len(frequency),
        "empty_transcript_count": sum(not value.strip() for value in texts),
        "most_common_transcripts": [
            {"text": text, "count": count}
            for text, count in frequency.most_common(10)
        ],
        "whitespace_casefold_corpus_wer": corpus_wer(pairs),
        "mean_normalized_word_edit_distance": float(
            np.mean(
                [
                    normalized_word_edit_distance(reference, hypothesis)
                    for reference, hypothesis in pairs
                ]
            )
        ),
    }


def load_official_bundle(
    *,
    local_path: Path,
    device: str,
    dtype_name: str,
) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoProcessor, WhisperForConditionalGeneration
    except ImportError as exc:  # pragma: no cover - remote GPU dependency
        raise RuntimeError("torch and transformers are required") from exc
    if dtype_name == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_name == "float32":
        dtype = torch.float32
    else:
        raise ValueError(f"unsupported dtype: {dtype_name}")
    processor = AutoProcessor.from_pretrained(
        str(local_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        str(local_path),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=dtype,
    )
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    return torch, processor, model


def official_one_best(
    *,
    torch_module: Any,
    processor: Any,
    model: Any,
    waveform: np.ndarray,
    device: str,
) -> str:
    processed = processor(
        waveform,
        sampling_rate=16_000,
        return_tensors="pt",
        return_attention_mask=True,
    )
    model_inputs = prepare_whisper_model_inputs(
        processed,
        device=device,
        floating_dtype=model.dtype,
    )
    with torch_module.inference_mode():
        sequences = model.generate(
            **model_inputs,
            language="en",
            task="transcribe",
            return_timestamps=False,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
        )
    decoded = processor.batch_decode(sequences, skip_special_tokens=True)
    if len(decoded) != 1 or not isinstance(decoded[0], str):
        raise RuntimeError("official Whisper returned an invalid one-best output")
    return decoded[0]


def unload_bundle(torch_module: Any, *objects: Any) -> None:
    del objects
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.device.startswith("cuda"):
        raise ValueError("formal content diagnostic requires a CUDA device")
    config_path = args.config.resolve()
    manifest_path = args.manifest.resolve()
    queries_path = args.queries.resolve()
    config = load_main_experiment_config(config_path)
    identity = model_identity_from_config(config, "asr")
    selected = select_complete_condition_grid(
        manifest_path=manifest_path,
        query_count=args.query_count,
    )
    all_queries = load_text_queries(queries_path)
    query_ids = sorted({record.query_id for record in selected})
    missing = sorted(set(query_ids) - set(all_queries))
    if missing:
        raise ValueError(f"selected query IDs are absent from FiQA queries: {missing}")
    references = {
        query_id: all_queries[query_id].text for query_id in query_ids
    }

    waveforms = {}
    audio = []
    for record in selected:
        waveform = load_audio_mono(record.audio_path, target_sample_rate=16_000)
        waveforms[record.record_id] = waveform
        audio.append(
            {
                "record_id": record.record_id,
                "query_id": record.query_id,
                "condition": record.condition,
                "audio_path": record.audio_path,
                "reference": references[record.query_id],
                "statistics": waveform_statistics(waveform, sample_rate=16_000),
            }
        )

    settings = WhisperSettings(
        identity=identity,
        num_beams=config["models"]["asr"]["num_beams"],
        num_return_sequences=config["models"]["asr"]["num_return_sequences"],
        language=config["models"]["asr"]["language"],
        task=config["models"]["asr"]["task"],
    )
    current = WhisperNBestGenerator(
        settings,
        device=args.device,
        dtype="bfloat16",
    )
    current.load()
    current_rows = []
    for record in selected:
        hypotheses = current.generate(
            waveforms[record.record_id],
            sample_rate=16_000,
        )
        current_rows.append(
            {
                "record_id": record.record_id,
                "query_id": record.query_id,
                "condition": record.condition,
                "text": hypotheses[0].text,
                "nbest": [
                    {
                        "rank": value.rank,
                        "text": value.text,
                        "sequence_score": value.sequence_score,
                        "average_token_logprob": value.average_token_logprob,
                        "valid_token_count": value.valid_token_count,
                    }
                    for value in hypotheses
                ],
            }
        )
    current.unload()

    official_rows = {}
    peak_allocated = {}
    for dtype_name in ("bfloat16", "float32"):
        torch_module, processor, model = load_official_bundle(
            local_path=identity.local_path,
            device=args.device,
            dtype_name=dtype_name,
        )
        torch_module.cuda.reset_peak_memory_stats(args.device)
        rows = []
        for record in selected:
            rows.append(
                {
                    "record_id": record.record_id,
                    "query_id": record.query_id,
                    "condition": record.condition,
                    "text": official_one_best(
                        torch_module=torch_module,
                        processor=processor,
                        model=model,
                        waveform=waveforms[record.record_id],
                        device=args.device,
                    ),
                }
            )
        official_rows[dtype_name] = rows
        peak_allocated[dtype_name] = int(
            torch_module.cuda.max_memory_allocated(args.device)
        )
        del model, processor
        unload_bundle(torch_module)

    rows_by_method = {
        METHODS[0]: current_rows,
        METHODS[1]: official_rows["bfloat16"],
        METHODS[2]: official_rows["float32"],
    }
    summaries = {
        method: summarize_method(rows=rows, references=references)
        for method, rows in rows_by_method.items()
    }
    return {
        "schema_version": 1,
        "status": "complete",
        "stage": "asrur_whisper_content_differential_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output(
            "status",
            "--short",
            "--untracked-files=all",
        ),
        "device": args.device,
        "model": {
            "name": identity.name,
            "revision": identity.revision,
            "local_path": str(identity.local_path),
        },
        "query_ids": query_ids,
        "record_count": len(selected),
        "audio": audio,
        "methods": {
            method: {
                "summary": summaries[method],
                "rows": rows_by_method[method],
            }
            for method in METHODS
        },
        "official_peak_allocated_bytes": peak_allocated,
        "provenance": {
            "config": file_record(config_path),
            "manifest": file_record(manifest_path),
            "queries": file_record(queries_path),
        },
        "claim_boundary": (
            "This diagnostic does not alter formal caches. A better official "
            "result identifies an invalid current ASR protocol but does not "
            "retroactively validate any Phase-2 Whisper metric."
        ),
    }


def main() -> int:
    args = parse_args()
    if args.query_count <= 0:
        raise ValueError("query-count must be positive")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse output: {output}")
    result = run(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "record_count": result["record_count"],
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
