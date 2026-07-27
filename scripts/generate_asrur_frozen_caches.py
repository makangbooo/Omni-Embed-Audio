#!/usr/bin/env python3
"""Generate resumable frozen BGE/Whisper/CE caches and exact Top-K rankings.

Every model is loaded from a pinned local directory with network access
disabled by the caller. The script never trains a model and never changes a
candidate set during reranking.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from AudioRetrieval.asr_uncertainty_reranking.artifacts import (  # noqa: E402
    load_frozen_candidates,
    load_nbest,
    load_unbounded_qrels,
)
from AudioRetrieval.asr_uncertainty_reranking.cache_manifest import (  # noqa: E402
    build_cache_manifest,
    file_record,
    write_cache_manifest_once,
)
from AudioRetrieval.asr_uncertainty_reranking.configuration import (  # noqa: E402
    load_main_experiment_config,
)
from AudioRetrieval.asr_uncertainty_reranking.data import (  # noqa: E402
    load_corpus,
    load_squtr_audio_manifest,
    load_text_queries,
    squtr_subset_name,
)
from AudioRetrieval.asr_uncertainty_reranking.dense import (  # noqa: E402
    exact_chunked_topk,
)
from AudioRetrieval.asr_uncertainty_reranking.model_adapters import (  # noqa: E402
    BgeCrossEncoder,
    BgeDenseEncoder,
    BgeDenseSettings,
    BgeRerankerSettings,
    WhisperGenerationStageError,
    WhisperNBestGenerator,
    WhisperSettings,
    load_audio_mono,
    model_identity_from_config,
)
from AudioRetrieval.asr_uncertainty_reranking.schema import (  # noqa: E402
    NBestHypothesis,
)

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
MAX_CONSECUTIVE_WHISPER_FAILURES = 8
MAX_WHISPER_RECORD_ATTEMPTS = 3
RETRYABLE_WHISPER_NUMERIC_STAGES = frozenset(
    {
        "four_beam_output_validation",
        "teacher_forced_conditional_logprob",
        "nbest_artifact_validation",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def formal_execution_guard(*, device: str, dry_run: bool) -> None:
    """Fail before model loading if formal provenance/offline guards differ."""

    if git_output("status", "--short", "--untracked-files=all"):
        raise RuntimeError("formal cache generation requires a clean Git worktree")
    if dry_run:
        return
    offline = {
        name: os.environ.get(name)
        for name in (
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "HF_DATASETS_OFFLINE",
        )
    }
    if any(value != "1" for value in offline.values()):
        raise RuntimeError(f"formal model execution requires strict offline mode: {offline}")
    if not device.startswith("cuda"):
        raise RuntimeError("formal model cache generation requires an approved CUDA device")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("approved CUDA device is not available to torch")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "formal model cache generation requires exactly one visible GPU"
        )


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            value,
            stream,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_write_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def json_text(value: object) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def strict_json_object(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant {value!r}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact is not an object: {path}")
    return value


def write_text_once_or_verify(path: Path, content: str) -> None:
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"existing immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def ensure_identity(output_dir: Path, value: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text_once_or_verify(output_dir / "run_identity.json", json_text(value))


def ranges(total: int, size: int) -> Iterable[tuple[int, int]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    for start in range(0, total, size):
        yield start, min(start + size, total)


def chunk_paths(root: Path, start: int, stop: int) -> tuple[Path, Path]:
    stem = f"{start:08d}_{stop:08d}"
    return root / f"{stem}.npy", root / f"{stem}.json"


def save_embedding_chunk(
    root: Path,
    start: int,
    stop: int,
    values: np.ndarray,
) -> None:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != stop - start:
        raise ValueError("embedding chunk shape mismatch")
    if not np.isfinite(array).all():
        raise ValueError("embedding chunk contains non-finite values")
    norms = np.linalg.norm(array, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("embedding chunk is not L2 normalized")
    array_path, metadata_path = chunk_paths(root, start, stop)
    if array_path.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite chunk [{start}, {stop})")
    atomic_write_npy(array_path, array)
    atomic_write_json(
        metadata_path,
        {
            "schema_version": 1,
            "start": start,
            "stop": stop,
            "shape": list(array.shape),
            "array": file_record(array_path),
        },
    )


def load_embedding_chunk(
    root: Path,
    start: int,
    stop: int,
    dimension: int,
) -> np.ndarray | None:
    array_path, metadata_path = chunk_paths(root, start, stop)
    if not array_path.exists() and not metadata_path.exists():
        return None
    if not array_path.is_file() or not metadata_path.is_file():
        raise RuntimeError(f"incomplete embedding chunk [{start}, {stop})")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("start") != start or metadata.get("stop") != stop:
        raise RuntimeError("embedding chunk identity mismatch")
    expected_shape = (stop - start, dimension)
    if metadata.get("shape") != list(expected_shape):
        raise RuntimeError("embedding chunk metadata shape mismatch")
    recorded = metadata.get("array")
    actual = file_record(array_path)
    if recorded != actual:
        raise RuntimeError("embedding chunk checksum mismatch")
    values = np.load(array_path, allow_pickle=False)
    if values.shape != expected_shape or values.dtype != np.float32:
        raise RuntimeError("embedding chunk array shape/dtype mismatch")
    if not np.isfinite(values).all():
        raise RuntimeError("embedding chunk contains non-finite values")
    if not np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-3):
        raise RuntimeError("embedding chunk is not L2 normalized")
    return values


def consolidate_embedding_chunks(
    root: Path,
    *,
    total: int,
    batch_size: int,
    dimension: int,
    destination: Path,
) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if destination.exists():
        values = np.load(destination, mmap_mode="r", allow_pickle=False)
        if values.shape != (total, dimension) or values.dtype != np.float32:
            raise RuntimeError("existing consolidated embedding artifact differs")
        return
    if temporary.exists():
        raise RuntimeError(f"stale consolidation temporary file: {temporary}")
    output = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(total, dimension),
    )
    for start, stop in ranges(total, batch_size):
        values = load_embedding_chunk(root, start, stop, dimension)
        if values is None:
            raise RuntimeError(f"missing embedding chunk [{start}, {stop})")
        output[start:stop] = values
        output.flush()
    del output
    temporary.replace(destination)


def record_failure(
    output_dir: Path,
    index: int,
    identifier: str,
    *,
    exception: BaseException | None = None,
    attempt: int | None = None,
    max_attempts: int | None = None,
    will_retry: bool = False,
) -> Path:
    failure_dir = output_dir / "failures"
    failure_dir.mkdir(exist_ok=True)
    failure_stage = getattr(exception, "stage", None)
    destination = failure_dir / f"{index:08d}_{time.time_ns()}.json"
    atomic_write_json(
        destination,
        {
            "schema_version": 1,
            "index": index,
            "identifier": identifier,
            "created_at": utc_now(),
            "exception_type": (
                None if exception is None else type(exception).__name__
            ),
            "failure_stage": failure_stage,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "will_retry": will_retry,
            "error": traceback.format_exc(),
        },
    )
    return destination


def shard_path(output_dir: Path, index: int) -> Path:
    return output_dir / "shards" / f"{index:08d}.json"


def save_json_shard(output_dir: Path, index: int, row: Mapping[str, Any]) -> None:
    path = shard_path(output_dir, index)
    write_text_once_or_verify(path, json_text(dict(row)))


def validate_whisper_shard(
    path: Path,
    *,
    index: int,
    record: Any,
    expected_hypotheses: int,
) -> dict[str, Any]:
    row = strict_json_object(path)
    expected_keys = {
        "query_id",
        "record_id",
        "source_query_id",
        "condition",
        "audio_path",
        "no_speech_probability",
        "no_speech_probability_status",
        "hypotheses",
    }
    if set(row) != expected_keys:
        raise RuntimeError(
            f"Whisper shard {index} fields differ: "
            f"{sorted(set(row) ^ expected_keys)}"
        )
    expected_identity = {
        "query_id": record.query_id,
        "record_id": record.record_id,
        "source_query_id": record.query_id,
        "condition": record.condition,
        "audio_path": record.audio_path,
        "no_speech_probability": None,
        "no_speech_probability_status": (
            "not_reliably_exposed_by_generation_api"
        ),
    }
    observed_identity = {
        key: row.get(key)
        for key in expected_identity
    }
    if observed_identity != expected_identity:
        raise RuntimeError(f"Whisper shard {index} record identity differs")
    raw_hypotheses = row.get("hypotheses")
    if (
        not isinstance(raw_hypotheses, list)
        or len(raw_hypotheses) != expected_hypotheses
    ):
        raise RuntimeError(
            f"Whisper shard {index} must contain "
            f"{expected_hypotheses} hypotheses"
        )
    hypotheses = []
    for raw in raw_hypotheses:
        if not isinstance(raw, dict):
            raise TypeError(f"Whisper shard {index} hypothesis is not an object")
        if set(raw) != {
            "rank",
            "text",
            "sequence_score",
            "average_token_logprob",
            "valid_token_count",
        }:
            raise RuntimeError(f"Whisper shard {index} hypothesis fields differ")
        hypotheses.append(
            NBestHypothesis(
                rank=raw["rank"],
                text=raw["text"],
                sequence_score=raw["sequence_score"],
                average_token_logprob=raw["average_token_logprob"],
                valid_token_count=raw["valid_token_count"],
            )
        )
    if tuple(value.rank for value in hypotheses) != tuple(
        range(1, expected_hypotheses + 1)
    ):
        raise RuntimeError(f"Whisper shard {index} ranks are not contiguous")
    if any(value.sequence_score is None for value in hypotheses):
        raise RuntimeError(f"Whisper shard {index} sequence score is absent")
    sequence_scores = [
        float(value.sequence_score)
        for value in hypotheses
        if value.sequence_score is not None
    ]
    if sequence_scores != sorted(sequence_scores, reverse=True):
        raise RuntimeError(
            f"Whisper shard {index} is not ordered by beam sequence score"
        )
    return row


def whisper_numeric_failure_is_retryable(exception: BaseException) -> bool:
    if not isinstance(exception, WhisperGenerationStageError):
        return False
    if exception.stage not in RETRYABLE_WHISPER_NUMERIC_STAGES:
        return False
    message = str(exception).casefold()
    return "non-finite" in message or "must be finite" in message


def whisper_resume_identity_compatible(
    source: Mapping[str, Any],
    destination: Mapping[str, Any],
    *,
    expected_source_git_commit: str,
) -> None:
    source_value = dict(source)
    destination_value = dict(destination)
    observed_source_commit = source_value.pop("git_commit", None)
    destination_commit = destination_value.pop("git_commit", None)
    if observed_source_commit != expected_source_git_commit:
        raise RuntimeError(
            "Whisper partial resume source commit differs: "
            f"{observed_source_commit!r}"
        )
    if not isinstance(destination_commit, str) or not destination_commit:
        raise RuntimeError("Whisper destination Git commit is absent")
    source_retry = source_value.pop("record_retry", None)
    destination_retry = destination_value.pop("record_retry", None)
    if source_retry is not None:
        raise RuntimeError(
            "Whisper partial resume source unexpectedly has a retry policy"
        )
    if not isinstance(destination_retry, Mapping):
        raise RuntimeError("Whisper destination retry policy is absent")
    if source_value != destination_value:
        differing = sorted(
            key
            for key in set(source_value) | set(destination_value)
            if source_value.get(key) != destination_value.get(key)
        )
        raise RuntimeError(
            "Whisper partial resume identity differs outside the audited "
            f"Git/retry fields: {differing}"
        )


def import_whisper_resume_shards(
    *,
    source_dir: Path,
    destination_dir: Path,
    records: Sequence[Any],
    expected_hypotheses: int,
    expected_source_git_commit: str,
    destination_identity: Mapping[str, Any],
) -> Path:
    source_dir = source_dir.resolve()
    destination_dir = destination_dir.resolve()
    if source_dir == destination_dir:
        raise RuntimeError("Whisper partial resume source and destination match")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Whisper partial resume source is absent: {source_dir}")
    if (source_dir / "cache_manifest.json").exists():
        raise RuntimeError(
            "Whisper partial resume source is already complete; "
            "reuse its immutable cache instead"
        )
    source_identity_path = source_dir / "run_identity.json"
    if not source_identity_path.is_file():
        raise FileNotFoundError(
            f"Whisper partial resume identity is absent: {source_identity_path}"
        )
    source_identity = strict_json_object(source_identity_path)
    whisper_resume_identity_compatible(
        source_identity,
        destination_identity,
        expected_source_git_commit=expected_source_git_commit,
    )
    source_shards = source_dir / "shards"
    imported = []
    for path in sorted(source_shards.glob("*.json")):
        try:
            index = int(path.stem)
        except ValueError as exc:
            raise RuntimeError(f"invalid Whisper shard filename: {path}") from exc
        if index < 0 or index >= len(records):
            raise RuntimeError(f"Whisper resume shard index is out of range: {index}")
        validate_whisper_shard(
            path,
            index=index,
            record=records[index],
            expected_hypotheses=expected_hypotheses,
        )
        destination = shard_path(destination_dir, index)
        write_text_once_or_verify(
            destination,
            path.read_text(encoding="utf-8"),
        )
        imported.append(
            {
                "index": index,
                "source": file_record(path),
                "destination": file_record(destination),
            }
        )
    if not imported:
        raise RuntimeError("Whisper partial resume source contains no valid shards")
    failures = [
        file_record(path)
        for path in sorted((source_dir / "failures").glob("*.json"))
    ]
    provenance = {
        "schema_version": 1,
        "status": "complete",
        "policy": (
            "exact validated shard copy; generation/scoring protocol unchanged"
        ),
        "source_output_dir": str(source_dir),
        "destination_output_dir": str(destination_dir),
        "source_run_identity": file_record(source_identity_path),
        "source_git_commit": expected_source_git_commit,
        "destination_git_commit": destination_identity["git_commit"],
        "imported_shard_count": len(imported),
        "imported_shards": imported,
        "source_failure_records": failures,
    }
    provenance_path = destination_dir / "partial_resume_provenance.json"
    write_text_once_or_verify(provenance_path, json_text(provenance))
    return provenance_path


def finalize_jsonl(
    output_dir: Path,
    *,
    count: int,
    destination_name: str,
) -> Path:
    destination = output_dir / destination_name
    rows = []
    for index in range(count):
        path = shard_path(output_dir, index)
        if not path.is_file():
            raise RuntimeError(f"missing JSON shard {index}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError(f"JSON shard {index} is not an object")
        rows.append(value)
    content = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
        + "\n"
        for row in rows
    )
    write_text_once_or_verify(destination, content)
    return destination


def load_config(path: Path) -> dict[str, Any]:
    return load_main_experiment_config(path)


def model_manifest(
    *,
    artifact_type: str,
    input_paths: Sequence[Path],
    output_paths: Sequence[Path],
    dataset: str,
    split: str,
    model_name: str,
    model_revision: str,
    tokenizer: object,
    pooling: str | None,
    embedding_dim: int | None,
    max_length: int | None,
    dtype: str | None,
    normalization: str | None,
    seed: int | None,
    extra_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return build_cache_manifest(
        artifact_type=artifact_type,
        dataset=dataset,
        split=split,
        inputs=[file_record(path) for path in input_paths],
        model_name=model_name,
        model_revision=model_revision,
        model_checkpoint=None,
        tokenizer=tokenizer,
        pooling=pooling,
        embedding_dim=embedding_dim,
        max_length=max_length,
        dtype=dtype,
        normalization=normalization,
        seed=seed,
        command=sys.argv,
        git_commit=git_output("rev-parse", "HEAD"),
        outputs=[file_record(path) for path in output_paths],
        extra_identity=extra_identity,
    )


def load_bge_inputs(
    path: Path,
    *,
    input_kind: str,
    query_ids: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    if input_kind == "corpus":
        if query_ids is not None:
            raise ValueError("corpus input must not receive query IDs")
        values = list(load_corpus(path).values())
        return (
            [value.document_id for value in values],
            [value.constructed_text for value in values],
        )
    if input_kind == "queries":
        if query_ids is None:
            raise ValueError("query input requires an explicit qrels-derived ID set")
        all_values = load_text_queries(path)
        missing = sorted(query_ids - set(all_values))
        if missing:
            raise ValueError(f"qrels query IDs are absent from query input: {missing[:20]}")
        values = [all_values[query_id] for query_id in sorted(query_ids)]
        return [value.query_id for value in values], [value.text for value in values]
    if input_kind == "nbest1":
        if query_ids is not None:
            raise ValueError("N-best input must not receive query IDs")
        values = list(load_nbest(path).values())
        return (
            [value.query_id for value in values],
            [value.hypotheses[0].text for value in values],
        )
    raise ValueError(f"unsupported BGE input kind: {input_kind}")


def run_bge(args: argparse.Namespace, config: dict[str, Any]) -> int:
    formal_execution_guard(device=args.device, dry_run=args.dry_run)
    identity = model_identity_from_config(config, "dense")
    query_qrels = getattr(args, "query_qrels", None)
    if args.input_kind == "queries":
        if query_qrels is None:
            raise ValueError("BGE query encoding requires --query-qrels")
        qrels = load_unbounded_qrels(query_qrels)
        ids, texts = load_bge_inputs(
            args.input,
            input_kind=args.input_kind,
            query_ids=set(qrels),
        )
    else:
        if query_qrels is not None:
            raise ValueError("--query-qrels is valid only for query input")
        ids, texts = load_bge_inputs(args.input, input_kind=args.input_kind)
    if args.input_kind == "corpus" and args.query_template != "none":
        raise ValueError("BGE corpus documents must not receive a query instruction")
    if args.query_template == "bge_retrieval":
        texts = [BGE_QUERY_INSTRUCTION + value for value in texts]
    elif args.query_template != "none":
        raise ValueError("unsupported BGE query template")
    settings = BgeDenseSettings(
        identity=identity,
        max_length=config["models"]["dense"]["max_length"],
        expected_dimension=config["models"]["dense"]["embedding_dim"],
        normalize=True,
    )
    run_identity = {
        "schema_version": 1,
        "stage": "bge",
        "git_commit": git_output("rev-parse", "HEAD"),
        "config": file_record(args.config),
        "input": file_record(args.input),
        "query_qrels": file_record(query_qrels) if query_qrels is not None else None,
        "input_kind": args.input_kind,
        "query_template": args.query_template,
        "model": {
            "name": identity.name,
            "revision": identity.revision,
            "local_path": str(identity.local_path),
        },
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "row_count": len(ids),
    }
    ensure_identity(args.output_dir, run_identity)
    write_text_once_or_verify(
        args.output_dir / "ids.jsonl",
        "".join(
            json.dumps({"index": index, "id": identifier}, sort_keys=True) + "\n"
            for index, identifier in enumerate(ids)
        ),
    )
    if args.dry_run:
        print(json.dumps({**run_identity, "status": "dry_run_complete"}, indent=2))
        return 0
    encoder = BgeDenseEncoder(settings, device=args.device, dtype=args.dtype)
    chunks = args.output_dir / "chunks"
    chunks.mkdir(exist_ok=True)
    pending = [
        (start, stop)
        for start, stop in ranges(len(ids), args.batch_size)
        if load_embedding_chunk(
            chunks,
            start,
            stop,
            settings.expected_dimension,
        )
        is None
    ]
    if pending:
        encoder.load()
    for start, stop in pending:
        values = encoder.encode(texts[start:stop], batch_size=stop - start)
        save_embedding_chunk(chunks, start, stop, values)
        print(f"[PROGRESS] BGE {stop}/{len(ids)}", flush=True)
    destination = args.output_dir / "embeddings.npy"
    consolidate_embedding_chunks(
        chunks,
        total=len(ids),
        batch_size=args.batch_size,
        dimension=settings.expected_dimension,
        destination=destination,
    )
    manifest = model_manifest(
        artifact_type="bge_dense_embeddings",
        input_paths=[
            args.config,
            args.input,
            *([query_qrels] if query_qrels is not None else []),
        ],
        output_paths=[args.output_dir / "ids.jsonl", destination],
        dataset=args.dataset,
        split=args.split,
        model_name=identity.name,
        model_revision=identity.revision,
        tokenizer={"local_path": str(identity.local_path)},
        pooling="cls",
        embedding_dim=settings.expected_dimension,
        max_length=settings.max_length,
        dtype=args.dtype,
        normalization="l2",
        seed=42,
        extra_identity={
            "input_kind": args.input_kind,
            "query_template": args.query_template,
        },
    )
    write_cache_manifest_once(args.output_dir / "cache_manifest.json", manifest)
    return 0


def selected_audio_records(
    path: Path,
    *,
    subset: str,
    conditions: Sequence[str],
) -> list[Any]:
    records = [
        value
        for value in load_squtr_audio_manifest(
            path,
            require_audio_files=True,
        ).values()
        if squtr_subset_name(value.subset) == subset.casefold()
        and value.condition in conditions
    ]
    records.sort(key=lambda value: (value.condition, value.query_id, value.record_id))
    if not records:
        raise ValueError("audio selection is empty")
    return records


def run_whisper(args: argparse.Namespace, config: dict[str, Any]) -> int:
    formal_execution_guard(device=args.device, dry_run=args.dry_run)
    if len(args.conditions) != 1:
        raise ValueError(
            "formal Whisper caches must contain exactly one acoustic condition"
        )
    if (
        args.resume_shards_from is None
        and args.resume_source_git_commit is not None
    ) or (
        args.resume_shards_from is not None
        and args.resume_source_git_commit is None
    ):
        raise ValueError(
            "--resume-shards-from and --resume-source-git-commit "
            "must be supplied together"
        )
    identity = model_identity_from_config(config, "asr")
    records = selected_audio_records(
        args.input,
        subset=args.subset,
        conditions=args.conditions,
    )
    settings = WhisperSettings(
        identity=identity,
        num_beams=config["models"]["asr"]["num_beams"],
        num_return_sequences=config["models"]["asr"]["num_return_sequences"],
        language=config["models"]["asr"]["language"],
        task=config["models"]["asr"]["task"],
    )
    run_identity = {
        "schema_version": 1,
        "stage": "whisper",
        "git_commit": git_output("rev-parse", "HEAD"),
        "config": file_record(args.config),
        "input": file_record(args.input),
        "subset": args.subset,
        "conditions": list(args.conditions),
        "model": {
            "name": identity.name,
            "revision": identity.revision,
            "local_path": str(identity.local_path),
        },
        "device": args.device,
        "dtype": args.dtype,
        "row_count": len(records),
        "decode": {
            "num_beams": settings.num_beams,
            "num_return_sequences": settings.num_return_sequences,
            "do_sample": False,
            "language": settings.language,
            "task": settings.task,
            "timestamps": False,
            "generation_entrypoint": "base_GenerationMixin_generate",
            "decoder_prompt": "explicit_language_task_no_timestamps",
            "token_score_method": (
                "teacher_forced_conditional_logprob_float32_cross_entropy"
            ),
            "beam_transition_scores_used": False,
            "posterior_status": "proxy_average_token_logprob_not_calibrated",
        },
        "record_retry": {
            "max_attempts": args.max_record_attempts,
            "eligible_failures": (
                "WhisperGenerationStageError in an approved numeric stage "
                "whose message explicitly reports non-finite values"
            ),
            "selection_policy": (
                "first strict finite result; no filtering, clamping, "
                "score replacement, or protocol change"
            ),
            "retry_isolation": "fresh reload of identical pinned local weights",
        },
    }
    ensure_identity(args.output_dir, run_identity)
    resume_provenance = None
    if args.resume_shards_from is not None:
        resume_provenance = import_whisper_resume_shards(
            source_dir=args.resume_shards_from,
            destination_dir=args.output_dir,
            records=records,
            expected_hypotheses=settings.num_return_sequences,
            expected_source_git_commit=args.resume_source_git_commit,
            destination_identity=run_identity,
        )
    for index, record in enumerate(records):
        path = shard_path(args.output_dir, index)
        if path.is_file():
            validate_whisper_shard(
                path,
                index=index,
                record=record,
                expected_hypotheses=settings.num_return_sequences,
            )
    if args.dry_run:
        print(json.dumps({**run_identity, "status": "dry_run_complete"}, indent=2))
        return 0
    generator = WhisperNBestGenerator(settings, device=args.device, dtype=args.dtype)
    pending = [
        (index, record)
        for index, record in enumerate(records)
        if not shard_path(args.output_dir, index).is_file()
    ]
    if pending:
        generator.load()
    failed = []
    consecutive_failures = 0
    for index, record in pending:
        failure_records = []
        recovered_attempt = None
        for attempt in range(1, args.max_record_attempts + 1):
            try:
                if attempt > 1:
                    generator.reload()
                waveform = load_audio_mono(
                    record.audio_path,
                    target_sample_rate=16_000,
                )
                hypotheses = generator.generate(waveform, sample_rate=16_000)
                row = {
                    "query_id": record.query_id,
                    "record_id": record.record_id,
                    "source_query_id": record.query_id,
                    "condition": record.condition,
                    "audio_path": record.audio_path,
                    "no_speech_probability": None,
                    "no_speech_probability_status": (
                        "not_reliably_exposed_by_generation_api"
                    ),
                    "hypotheses": [
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
                save_json_shard(args.output_dir, index, row)
                validate_whisper_shard(
                    shard_path(args.output_dir, index),
                    index=index,
                    record=record,
                    expected_hypotheses=settings.num_return_sequences,
                )
                recovered_attempt = attempt
                consecutive_failures = 0
                print(
                    f"[PROGRESS] Whisper {index + 1}/{len(records)} "
                    f"attempt={attempt}",
                    flush=True,
                )
                break
            except Exception as exc:
                retryable = whisper_numeric_failure_is_retryable(exc)
                will_retry = (
                    retryable and attempt < args.max_record_attempts
                )
                failure_path = record_failure(
                    args.output_dir,
                    index,
                    record.record_id,
                    exception=exc,
                    attempt=attempt,
                    max_attempts=args.max_record_attempts,
                    will_retry=will_retry,
                )
                failure_records.append(file_record(failure_path))
                if will_retry:
                    print(
                        "[WARN] Retrying strict same-protocol Whisper record "
                        f"{record.record_id} after audited numeric failure "
                        f"attempt {attempt}/{args.max_record_attempts}",
                        flush=True,
                    )
                    continue
                failed.append(record.record_id)
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_WHISPER_FAILURES:
                    raise RuntimeError(
                        "Whisper aborted after "
                        f"{consecutive_failures} consecutive failures; "
                        "per-record tracebacks were preserved under "
                        f"{args.output_dir / 'failures'}"
                    ) from None
                break
        if recovered_attempt is not None and recovered_attempt > 1:
            recovery_path = args.output_dir / "numeric_retry_recoveries.jsonl"
            with recovery_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "record_id": record.record_id,
                            "index": index,
                            "successful_attempt": recovered_attempt,
                            "failed_attempt_records": failure_records,
                            "protocol_changed": False,
                        },
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
    if failed:
        raise RuntimeError(f"{len(failed)} Whisper records failed; first={failed[:10]}")
    destination = finalize_jsonl(
        args.output_dir,
        count=len(records),
        destination_name="nbest.jsonl",
    )
    manifest_inputs = [args.config, args.input]
    if resume_provenance is not None:
        manifest_inputs.append(resume_provenance)
    retry_recovery_path = args.output_dir / "numeric_retry_recoveries.jsonl"
    if retry_recovery_path.is_file():
        manifest_inputs.append(retry_recovery_path)
    manifest = model_manifest(
        artifact_type="whisper_nbest",
        input_paths=manifest_inputs,
        output_paths=[destination],
        dataset=args.dataset,
        split=args.split,
        model_name=identity.name,
        model_revision=identity.revision,
        tokenizer={"local_path": str(identity.local_path)},
        pooling=None,
        embedding_dim=None,
        max_length=None,
        dtype=args.dtype,
        normalization="proxy_softmax_deferred_to_fiqa_dev_temperature",
        seed=42,
        extra_identity={
            "subset": args.subset,
            "conditions": list(args.conditions),
            "decode": run_identity["decode"],
            "target_sample_rate": 16_000,
            "record_retry": run_identity["record_retry"],
            "partial_resume_performed": resume_provenance is not None,
        },
    )
    write_cache_manifest_once(args.output_dir / "cache_manifest.json", manifest)
    return 0


def run_ce(args: argparse.Namespace, config: dict[str, Any]) -> int:
    formal_execution_guard(device=args.device, dry_run=args.dry_run)
    identity = model_identity_from_config(config, "reranker")
    candidates = load_frozen_candidates(args.candidates)
    nbest = load_nbest(args.nbest, expected_size=args.expected_hypotheses)
    corpus = load_corpus(args.corpus)
    if set(candidates) != set(nbest):
        raise ValueError("candidate and N-best query sets differ")
    query_ids = sorted(candidates)
    settings = BgeRerankerSettings(
        identity=identity,
        max_length=config["models"]["reranker"]["max_length"],
    )
    run_identity = {
        "schema_version": 1,
        "stage": "cross_encoder",
        "git_commit": git_output("rev-parse", "HEAD"),
        "config": file_record(args.config),
        "candidates": file_record(args.candidates),
        "nbest": file_record(args.nbest),
        "corpus": file_record(args.corpus),
        "model": {
            "name": identity.name,
            "revision": identity.revision,
            "local_path": str(identity.local_path),
        },
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "query_count": len(query_ids),
        "candidate_depth": config["candidate_protocol"]["depth"],
        "hypothesis_count": args.expected_hypotheses,
    }
    ensure_identity(args.output_dir, run_identity)
    for query_id in query_ids:
        missing = [
            document_id
            for document_id in candidates[query_id].candidate_ids
            if document_id not in corpus
        ]
        if missing:
            raise KeyError(f"{query_id}: candidate documents missing: {missing[:10]}")
    if args.dry_run:
        print(json.dumps({**run_identity, "status": "dry_run_complete"}, indent=2))
        return 0
    scorer = BgeCrossEncoder(settings, device=args.device, dtype=args.dtype)
    pending = [
        (index, query_id)
        for index, query_id in enumerate(query_ids)
        if not shard_path(args.output_dir, index).is_file()
    ]
    if pending:
        scorer.load()
    failed = []
    for index, query_id in pending:
        try:
            candidate_ids = candidates[query_id].candidate_ids
            hypotheses = nbest[query_id].hypotheses
            pairs = [
                (hypothesis.text, corpus[document_id].constructed_text)
                for hypothesis in hypotheses
                for document_id in candidate_ids
            ]
            values = scorer.score(pairs, batch_size=args.batch_size)
            matrix = values.reshape(len(hypotheses), len(candidate_ids))
            save_json_shard(
                args.output_dir,
                index,
                {
                    "query_id": query_id,
                    "candidate_ids": list(candidate_ids),
                    "scores": matrix.astype(float).tolist(),
                },
            )
            print(f"[PROGRESS] CE {index + 1}/{len(query_ids)}", flush=True)
        except Exception as exc:
            failed.append(query_id)
            record_failure(
                args.output_dir,
                index,
                query_id,
                exception=exc,
            )
    if failed:
        raise RuntimeError(f"{len(failed)} CE queries failed; first={failed[:10]}")
    destination = finalize_jsonl(
        args.output_dir,
        count=len(query_ids),
        destination_name="cross_encoder_scores.jsonl",
    )
    manifest = model_manifest(
        artifact_type="cross_encoder_scores",
        input_paths=[args.config, args.candidates, args.nbest, args.corpus],
        output_paths=[destination],
        dataset=args.dataset,
        split=args.split,
        model_name=identity.name,
        model_revision=identity.revision,
        tokenizer={"local_path": str(identity.local_path)},
        pooling=None,
        embedding_dim=None,
        max_length=settings.max_length,
        dtype=args.dtype,
        normalization="query_local_zscore_deferred",
        seed=42,
        extra_identity={
            "candidate_depth": config["candidate_protocol"]["depth"],
            "hypothesis_count": args.expected_hypotheses,
        },
    )
    write_cache_manifest_once(args.output_dir / "cache_manifest.json", manifest)
    return 0


def load_id_file(path: Path, *, field: str) -> list[str]:
    result = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = json.loads(line)
            if not isinstance(value, dict) or not isinstance(value.get(field), str):
                raise ValueError(f"invalid ID row: {path}:{line_number}")
            result.append(value[field])
    if not result or len(result) != len(set(result)):
        raise ValueError("ID file must be non-empty and unique")
    return result


def run_dense(args: argparse.Namespace, config: dict[str, Any]) -> int:
    query_ids = load_id_file(args.query_ids, field=args.query_id_field)
    document_ids = load_id_file(args.document_ids, field=args.document_id_field)
    queries = np.load(args.query_embeddings, mmap_mode="r", allow_pickle=False)
    documents = np.load(args.document_embeddings, mmap_mode="r", allow_pickle=False)
    result = exact_chunked_topk(
        queries,
        documents,
        query_ids=query_ids,
        document_ids=document_ids,
        k=config["candidate_protocol"]["depth"],
        query_batch_size=args.query_batch_size,
        document_chunk_size=args.document_chunk_size,
    )
    rows = [
        {
            "query_id": query_id,
            "candidate_ids": result[query_id]["candidate_ids"],
            "scores": result[query_id]["scores"],
        }
        for query_id in query_ids
    ]
    destination = args.output
    write_text_once_or_verify(
        destination,
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True)
            + "\n"
            for row in rows
        ),
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "query_count": len(query_ids),
                "document_count": len(document_ids),
                "depth": config["candidate_protocol"]["depth"],
                "output": str(destination.resolve()),
            },
            indent=2,
        )
    )
    return 0


def add_common_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), required=True)
    parser.add_argument("--dry-run", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    bge = subparsers.add_parser("bge")
    add_common_model_arguments(bge)
    bge.add_argument("--input", type=Path, required=True)
    bge.add_argument(
        "--input-kind",
        choices=("corpus", "queries", "nbest1"),
        required=True,
    )
    bge.add_argument(
        "--query-template",
        choices=("none", "bge_retrieval"),
        required=True,
    )
    bge.add_argument(
        "--query-qrels",
        type=Path,
        help=(
            "Required for query input. The qrels query IDs define the exact "
            "train/dev/test subset and prevent split mixing."
        ),
    )
    bge.add_argument("--batch-size", type=int, required=True)

    whisper = subparsers.add_parser("whisper")
    add_common_model_arguments(whisper)
    whisper.add_argument("--input", type=Path, required=True)
    whisper.add_argument("--subset", required=True)
    whisper.add_argument(
        "--conditions",
        nargs="+",
        choices=("clean", "snr_20", "snr_10", "snr_0"),
        required=True,
    )
    whisper.add_argument(
        "--max-record-attempts",
        type=int,
        choices=tuple(range(1, MAX_WHISPER_RECORD_ATTEMPTS + 1)),
        default=1,
        help=(
            "Bounded retries only for explicitly classified transient "
            "non-finite Whisper failures; every failed attempt is preserved."
        ),
    )
    whisper.add_argument(
        "--resume-shards-from",
        type=Path,
        help=(
            "Import strict validated shards from an incomplete prior cache. "
            "Requires --resume-source-git-commit."
        ),
    )
    whisper.add_argument(
        "--resume-source-git-commit",
        help="Exact producer commit required for --resume-shards-from.",
    )

    ce = subparsers.add_parser("ce")
    add_common_model_arguments(ce)
    ce.add_argument("--candidates", type=Path, required=True)
    ce.add_argument("--nbest", type=Path, required=True)
    ce.add_argument("--corpus", type=Path, required=True)
    ce.add_argument("--batch-size", type=int, required=True)
    ce.add_argument(
        "--expected-hypotheses",
        type=int,
        choices=(1, 4),
        default=4,
        help="Use 4 for Whisper N-best and 1 for the gold-transcript upper bound.",
    )

    dense = subparsers.add_parser("dense")
    dense.add_argument("--config", type=Path, required=True)
    dense.add_argument("--query-embeddings", type=Path, required=True)
    dense.add_argument("--query-ids", type=Path, required=True)
    dense.add_argument("--query-id-field", required=True)
    dense.add_argument("--document-embeddings", type=Path, required=True)
    dense.add_argument("--document-ids", type=Path, required=True)
    dense.add_argument("--document-id-field", required=True)
    dense.add_argument("--query-batch-size", type=int, default=32)
    dense.add_argument("--document-chunk-size", type=int, default=16384)
    dense.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.config = args.config.resolve()
    config = load_config(args.config)
    if args.stage != "dense":
        args.output_dir = args.output_dir.resolve()
        if getattr(args, "batch_size", 1) <= 0:
            raise ValueError("batch_size must be positive")
    if args.stage == "bge":
        args.input = args.input.resolve()
        if args.query_qrels is not None:
            args.query_qrels = args.query_qrels.resolve()
        return run_bge(args, config)
    if args.stage == "whisper":
        args.input = args.input.resolve()
        if args.resume_shards_from is not None:
            args.resume_shards_from = args.resume_shards_from.resolve()
        return run_whisper(args, config)
    if args.stage == "ce":
        args.candidates = args.candidates.resolve()
        args.nbest = args.nbest.resolve()
        args.corpus = args.corpus.resolve()
        return run_ce(args, config)
    if args.stage == "dense":
        args.output = args.output.resolve()
        if args.query_batch_size <= 0 or args.document_chunk_size <= 0:
            raise ValueError("dense chunk sizes must be positive")
        return run_dense(args, config)
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
