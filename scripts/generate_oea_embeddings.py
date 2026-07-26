#!/usr/bin/env python3
"""Generate resumable OEA audio and caption embeddings from a fixed manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.build_official_oea_eval_config import (
    verify_official_model_lock_binding,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attempt-dir", type=Path)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def atomic_write_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported embedding config schema_version")
    required = {
        "audio_batch_size",
        "audio_prompt_protocol",
        "base_model",
        "caption_count_per_audio",
        "checkpoint",
        "dataset",
        "expected_examples",
        "experiment_prefix",
        "model",
        "model_config",
        "official_variant_id",
        "seed",
        "text_batch_size",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"embedding config missing fields: {missing}")
    for field in (
        "audio_batch_size",
        "caption_count_per_audio",
        "expected_examples",
        "text_batch_size",
    ):
        value = config[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool):
        raise ValueError("seed must be an integer")
    for field in (
        "dataset",
        "experiment_prefix",
        "model",
        "official_variant_id",
    ):
        if not isinstance(config[field], str) or not config[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    tagged_value(config, "audio_prompt_protocol", "runtime")
    return config


def tagged_value(config: Mapping[str, Any], section: str, key: str) -> Any:
    value = config[section][key]
    if not isinstance(value, dict) or "value" not in value or "source" not in value:
        raise ValueError(f"{section}.{key} must contain value and source")
    return value["value"]


def tagged_value_or_default(
    config: Mapping[str, Any], section: str, key: str, default: Any
) -> Any:
    values = config.get(section)
    if not isinstance(values, Mapping) or key not in values:
        return default
    return tagged_value(config, section, key)


def load_manifest(
    path: Path, expected_examples: int, caption_count: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank manifest row at line {line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"manifest line {line_number} is not an object")
            sample_id = row.get("sample_id")
            audio_path = row.get("audio_path")
            captions = row.get("captions")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"manifest line {line_number} has invalid sample_id")
            if not isinstance(audio_path, str) or not audio_path.strip():
                raise ValueError(f"manifest line {line_number} has invalid audio_path")
            if not isinstance(captions, list) or len(captions) != caption_count:
                raise ValueError(
                    f"manifest line {line_number} must have {caption_count} captions"
                )
            if any(not isinstance(text, str) or not text.strip() for text in captions):
                raise ValueError(f"manifest line {line_number} has an empty caption")
            resolved_audio = Path(audio_path).expanduser().resolve()
            if not resolved_audio.is_file():
                raise FileNotFoundError(resolved_audio)
            if row.get("file_exists") is not True or row.get("decode_ok") is not True:
                raise ValueError(
                    f"manifest line {line_number} is not marked file_exists/decode_ok"
                )
            rows.append(
                {
                    "sample_id": sample_id.strip(),
                    "audio_path": resolved_audio,
                    "captions": [text.strip() for text in captions],
                }
            )
    if len(rows) != expected_examples:
        raise ValueError(f"manifest count mismatch: {len(rows)} != {expected_examples}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("manifest sample_id values are not unique")
    return rows


def candidate_metadata(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_index": index,
            "candidate_id": row["sample_id"],
            "audio_path": str(row["audio_path"]),
        }
        for index, row in enumerate(rows)
    ]


def query_metadata(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        for caption_index, text in enumerate(row["captions"], start=1):
            result.append(
                {
                    "query_index": len(result),
                    "query_id": f"{row['sample_id']}#caption_{caption_index}",
                    "target_id": row["sample_id"],
                    "clip_id": row["sample_id"],
                    "caption_index": caption_index,
                    "text": text,
                }
            )
    return result


def chunk_stem(kind: str, start: int, stop: int) -> str:
    return f"{kind}_{start:06d}_{stop:06d}"


def save_chunk(
    chunk_root: Path,
    kind: str,
    start: int,
    stop: int,
    embeddings: np.ndarray,
    elapsed_seconds: float,
) -> dict[str, Any]:
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    expected_rows = stop - start
    if embeddings.ndim != 2 or embeddings.shape[0] != expected_rows:
        raise ValueError(
            f"{kind} chunk shape mismatch for [{start}, {stop}): {embeddings.shape}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{kind} chunk contains non-finite embeddings")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(f"{kind} chunk embeddings are not L2 normalized")
    stem = chunk_stem(kind, start, stop)
    array_path = chunk_root / f"{stem}.npy"
    metadata_path = chunk_root / f"{stem}.json"
    if array_path.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite existing chunk: {stem}")
    atomic_write_npy(array_path, embeddings)
    metadata = {
        "schema_version": 1,
        "kind": kind,
        "start": start,
        "stop": stop,
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "elapsed_seconds": elapsed_seconds,
        "array": file_identity(array_path),
        "completed_at": utc_now(),
    }
    atomic_write_json(metadata_path, metadata)
    return metadata


def load_verified_chunk(
    chunk_root: Path,
    kind: str,
    start: int,
    stop: int,
    embedding_dim: int,
) -> np.ndarray | None:
    stem = chunk_stem(kind, start, stop)
    array_path = chunk_root / f"{stem}.npy"
    metadata_path = chunk_root / f"{stem}.json"
    if not array_path.exists() and not metadata_path.exists():
        return None
    if not array_path.is_file() or not metadata_path.is_file():
        raise RuntimeError(f"incomplete chunk artifact pair: {stem}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_shape = [stop - start, embedding_dim]
    if (
        metadata.get("kind") != kind
        or metadata.get("start") != start
        or metadata.get("stop") != stop
        or metadata.get("shape") != expected_shape
    ):
        raise RuntimeError(f"chunk metadata identity mismatch: {stem}")
    identity = file_identity(array_path)
    recorded = metadata.get("array", {})
    if identity["size_bytes"] != recorded.get("size_bytes") or identity["sha256"] != recorded.get("sha256"):
        raise RuntimeError(f"chunk checksum mismatch: {stem}")
    embeddings = np.load(array_path, allow_pickle=False)
    if embeddings.shape != tuple(expected_shape) or embeddings.dtype != np.float32:
        raise RuntimeError(f"chunk array type/shape mismatch: {stem}")
    if not np.isfinite(embeddings).all():
        raise RuntimeError(f"chunk contains non-finite values: {stem}")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise RuntimeError(f"chunk is not L2 normalized: {stem}")
    return embeddings


def ranges(total: int, batch_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, total, batch_size):
        yield start, min(start + batch_size, total)


def consolidate_chunks(
    chunk_root: Path,
    kind: str,
    total: int,
    batch_size: int,
    embedding_dim: int,
    destination: Path,
) -> dict[str, Any]:
    chunks: list[np.ndarray] = []
    for start, stop in ranges(total, batch_size):
        chunk = load_verified_chunk(
            chunk_root, kind, start, stop, embedding_dim
        )
        if chunk is None:
            raise RuntimeError(f"missing completed {kind} chunk [{start}, {stop})")
        chunks.append(chunk)
    combined = np.concatenate(chunks, axis=0)
    if combined.shape != (total, embedding_dim):
        raise RuntimeError(f"consolidated {kind} shape mismatch: {combined.shape}")
    if destination.exists():
        existing = np.load(destination, allow_pickle=False)
        if existing.shape != combined.shape or not np.array_equal(existing, combined):
            raise RuntimeError(f"existing consolidated artifact differs: {destination}")
    else:
        atomic_write_npy(destination, combined)
    return file_identity(destination)


def verify_fixed_file(path: Path, specification: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_size = int(specification["size_bytes"])
    if path.stat().st_size != expected_size:
        raise RuntimeError(f"size mismatch for fixed resource: {path}")
    identity = file_identity(path)
    if identity["sha256"].lower() != str(specification["sha256"]).lower():
        raise RuntimeError(f"SHA256 mismatch for fixed resource: {path}")
    return identity


def ensure_run_identity(output_dir: Path, identity: Mapping[str, Any]) -> None:
    path = output_dir / "run_identity.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != identity:
            raise RuntimeError("resume identity differs from existing run_identity.json")
    else:
        atomic_write_json(path, identity)


def write_text_once_or_verify(path: Path, content: str) -> None:
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"existing immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def immutable_json_text(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def immutable_jsonl_text(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def load_model_bundle(config: Mapping[str, Any], model_root: Path, report: dict[str, Any]):
    import torch

    from AudioRetrieval.models.omni_embed_adapter import OmniEmbedAdapter
    from AudioRetrieval.training.oea.train_omniembed_lora import (
        ProjectionHead,
        attach_lora,
        resolve_text_hidden_size,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("formal embedding generation requires exactly one visible GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("visible GPU does not support BF16")
    device = torch.device("cuda:0")
    base_dir = (model_root / config["base_model"]["local_subdir"]).resolve()
    checkpoint_path = (
        model_root / config["checkpoint"]["local_subpath"]
    ).resolve()
    verified_files = [
        verify_fixed_file(base_dir / relative_path, specification)
        for relative_path, specification in config["base_model"]["files"].items()
    ]
    verified_files.append(verify_fixed_file(checkpoint_path, config["checkpoint"]))
    report["verified_model_files"] = verified_files
    atomic_write_json(Path(report["metrics_path"]), report)

    unsafe_globals = sorted(
        torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint_path)
    )
    if unsafe_globals:
        raise RuntimeError(f"derived checkpoint has unsafe globals: {unsafe_globals}")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=True
    )
    required_sections = {"lora_state_dict", "audio_head", "text_head", "config"}
    missing_sections = sorted(required_sections - set(checkpoint))
    if missing_sections:
        raise RuntimeError(f"checkpoint sections missing: {missing_sections}")

    adapter = OmniEmbedAdapter(
        repo_id=config["base_model"]["repo_id"],
        local_path=str(base_dir),
        device="cuda:0",
        # The NVIDIA Nemotron base declares a custom AutoModel through
        # config.json:auto_map.  Formal Nemo runs may execute it only from the
        # fully enumerated, checksum-verified local model directory bound by the
        # committed model lock.  Existing Qwen protocols remain false by default.
        trust_remote_code=bool(
            tagged_value_or_default(
                config, "model_config", "trust_remote_code", False
            )
        ),
        torch_dtype=tagged_value(config, "model_config", "torch_dtype"),
        passage_prefix=tagged_value(config, "model_config", "passage_prefix"),
        query_prefix=tagged_value(config, "model_config", "query_prefix"),
    )
    lora_config = SimpleNamespace(
        lora_rank=int(tagged_value(config, "model_config", "lora_rank")),
        lora_alpha=int(tagged_value(config, "model_config", "lora_alpha")),
        lora_dropout=float(tagged_value(config, "model_config", "lora_dropout")),
        lora_targets=list(tagged_value(config, "model_config", "lora_targets")),
    )
    peft_model = attach_lora(adapter.get_underlying_model(), lora_config)
    expected_lora_keys = set(checkpoint["lora_state_dict"])
    actual_lora_keys = {
        key for key in peft_model.state_dict() if "lora_" in key.lower()
    }
    if expected_lora_keys != actual_lora_keys:
        raise RuntimeError(
            "LoRA key mismatch: "
            f"missing={sorted(expected_lora_keys - actual_lora_keys)[:10]}, "
            f"extra={sorted(actual_lora_keys - expected_lora_keys)[:10]}"
        )
    incompatible = peft_model.load_state_dict(
        checkpoint["lora_state_dict"], strict=False
    )
    missing_loaded_lora = [
        key for key in incompatible.missing_keys if "lora_" in key.lower()
    ]
    if missing_loaded_lora or incompatible.unexpected_keys:
        raise RuntimeError(
            "LoRA load mismatch: "
            f"missing={missing_loaded_lora[:10]}, "
            f"unexpected={list(incompatible.unexpected_keys)[:10]}"
        )
    adapter.set_underlying_model(peft_model)
    peft_model.eval()

    hidden_size = int(resolve_text_hidden_size(peft_model))
    projection_dim = int(tagged_value(config, "model_config", "projection_dim"))
    dropout = float(tagged_value(config, "model_config", "projection_dropout"))
    audio_head = ProjectionHead(hidden_size, projection_dim, dropout).to(device)
    text_head = ProjectionHead(hidden_size, projection_dim, dropout).to(device)
    audio_head.load_state_dict(checkpoint["audio_head"], strict=True)
    text_head.load_state_dict(checkpoint["text_head"], strict=True)
    audio_head.eval()
    text_head.eval()
    report["model_load"] = {
        "unsafe_globals": unsafe_globals,
        "lora_tensor_count": len(expected_lora_keys),
        "missing_non_lora_base_keys": len(incompatible.missing_keys),
        "hidden_size": hidden_size,
        "projection_dim": projection_dim,
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_memory_bytes": int(
            torch.cuda.get_device_properties(device).total_memory
        ),
    }
    del checkpoint
    return torch, adapter, peft_model, audio_head, text_head, device


def project_batch(
    torch: Any,
    adapter: Any,
    model: Any,
    head: Any,
    device: Any,
    *,
    texts: list[str] | None = None,
    audio_paths: list[Path] | None = None,
) -> np.ndarray:
    from AudioRetrieval.training.oea.train_omniembed_lora import encode_batch

    expected_count = len(texts) if texts is not None else len(audio_paths or [])
    with torch.inference_mode(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pooled, valid_indices = encode_batch(
            adapter,
            model,
            adapter.processor,
            texts,
            audio_paths,
            device,
        )
        if pooled is None or valid_indices != list(range(expected_count)):
            raise RuntimeError(
                "encoder skipped one or more inputs: "
                f"expected={expected_count}, valid_indices={valid_indices}"
            )
        messages = [str(item.message) for item in caught]
        if any("silence" in message.lower() or "failed to load" in message.lower() for message in messages):
            raise RuntimeError(f"audio fallback warning detected: {messages}")
        head_parameter = next(head.parameters())
        projected = head(
            pooled.to(device=head_parameter.device, dtype=head_parameter.dtype)
        )
    return projected.detach().cpu().float().numpy()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    model_root = args.model_root.resolve()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    attempt_dir = args.attempt_dir.resolve() if args.attempt_dir else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "generation_metrics.json"
    started_at = utc_now()
    metrics_preexisting = metrics_path.exists()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": started_at,
        "finished_at": None,
        "metrics_path": str(metrics_path),
        "attempt_dir": str(attempt_dir),
        "error": None,
    }
    identity_validated = False

    try:
        if metrics_preexisting:
            existing_report = json.loads(metrics_path.read_text(encoding="utf-8"))
            if (
                not isinstance(existing_report, dict)
                or existing_report.get("schema_version") != 1
            ):
                raise RuntimeError(
                    f"invalid existing generation metrics: {metrics_path}"
                )
            report = existing_report
            report.update(
                {
                    "status": "running",
                    "finished_at": None,
                    "last_attempt_started_at": started_at,
                    "metrics_path": str(metrics_path),
                    "attempt_dir": str(attempt_dir),
                    "error": None,
                }
            )
            report.pop("traceback", None)
        config = load_config(config_path)
        git_commit = git_output("rev-parse", "HEAD")
        git_status = git_output("status", "--short")
        if git_status:
            raise RuntimeError(f"formal generation requires a clean worktree: {git_status!r}")
        model_lock_binding = verify_official_model_lock_binding(config)
        if config.get("resolution_git_commit") != git_commit:
            raise RuntimeError(
                "resolved embedding config was not generated at the current Git commit"
            )
        if not output_dir.name.startswith(config["experiment_prefix"] + "_"):
            raise ValueError("output directory name must start with experiment_prefix")
        strict_offline = {
            name: os.environ.get(name)
            for name in (
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "HF_DATASETS_OFFLINE",
            )
        }
        if any(value != "1" for value in strict_offline.values()):
            raise RuntimeError("all strict-offline variables must equal 1")
        rows = load_manifest(
            manifest_path,
            int(config["expected_examples"]),
            int(config["caption_count_per_audio"]),
        )
        candidates = candidate_metadata(rows)
        queries = query_metadata(rows)
        identity = {
            "schema_version": 1,
            "experiment_id": output_dir.name,
            "git_commit": git_commit,
            "config": file_identity(config_path),
            "manifest": file_identity(manifest_path),
            "model_root": str(model_root),
            "model": config["model"],
            "official_model_lock": model_lock_binding,
            "dataset": config["dataset"],
            "checkpoint_revision": config["checkpoint"]["revision"],
            "seed": config["seed"],
            "audio_batch_size": config["audio_batch_size"],
            "text_batch_size": config["text_batch_size"],
            "candidate_count": len(candidates),
            "query_count": len(queries),
        }
        ensure_run_identity(output_dir, identity)
        identity_validated = True
        resolved_config = json.loads(json.dumps(config))
        resolved_config["resolved_paths"] = {
            "source_config": str(config_path),
            "model_root": str(model_root),
            "manifest": str(manifest_path),
            "output_dir": str(output_dir),
        }
        write_text_once_or_verify(
            output_dir / "config.yaml", immutable_json_text(resolved_config)
        )
        write_text_once_or_verify(
            output_dir / "candidate_metadata.jsonl",
            immutable_jsonl_text(candidates),
        )
        write_text_once_or_verify(
            output_dir / "query_metadata.jsonl", immutable_jsonl_text(queries)
        )
        report.update(
            {
                "experiment_id": output_dir.name,
                "git_commit": git_commit,
                "git_status_short": git_status,
                "model": config["model"],
                "official_model_lock": model_lock_binding,
                "dataset": config["dataset"],
                "seed": config["seed"],
                "audio_prompt_protocol": config["audio_prompt_protocol"],
                "strict_offline": strict_offline,
                "candidate_count": len(candidates),
                "query_count": len(queries),
                "completed_audio_chunks": 0,
                "completed_text_chunks": 0,
            }
        )
        atomic_write_json(metrics_path, report)

        random.seed(config["seed"])
        np.random.seed(config["seed"])
        embedding_dim = int(tagged_value(config, "model_config", "projection_dim"))
        audio_batch_size = int(config["audio_batch_size"])
        text_batch_size = int(config["text_batch_size"])
        chunk_root = output_dir / "chunks"
        chunk_root.mkdir(exist_ok=True)

        pending_audio = [
            (start, stop)
            for start, stop in ranges(len(candidates), audio_batch_size)
            if load_verified_chunk(chunk_root, "audio", start, stop, embedding_dim)
            is None
        ]
        pending_text = [
            (start, stop)
            for start, stop in ranges(len(queries), text_batch_size)
            if load_verified_chunk(chunk_root, "text", start, stop, embedding_dim)
            is None
        ]
        report["completed_audio_chunks"] = len(
            list(ranges(len(candidates), audio_batch_size))
        ) - len(pending_audio)
        report["completed_text_chunks"] = len(
            list(ranges(len(queries), text_batch_size))
        ) - len(pending_text)
        report["pending_audio_chunks"] = len(pending_audio)
        report["pending_text_chunks"] = len(pending_text)
        atomic_write_json(metrics_path, report)

        if pending_audio or pending_text:
            torch, adapter, model, audio_head, text_head, device = load_model_bundle(
                config, model_root, report
            )
            torch.manual_seed(config["seed"])
            torch.cuda.manual_seed_all(config["seed"])
            torch.cuda.reset_peak_memory_stats(device)
            for start, stop in pending_audio:
                started = time.monotonic()
                embeddings = project_batch(
                    torch,
                    adapter,
                    model,
                    audio_head,
                    device,
                    audio_paths=[rows[index]["audio_path"] for index in range(start, stop)],
                )
                save_chunk(
                    chunk_root,
                    "audio",
                    start,
                    stop,
                    embeddings,
                    time.monotonic() - started,
                )
                report["completed_audio_chunks"] += 1
                report["pending_audio_chunks"] -= 1
                atomic_write_json(metrics_path, report)
            for start, stop in pending_text:
                started = time.monotonic()
                embeddings = project_batch(
                    torch,
                    adapter,
                    model,
                    text_head,
                    device,
                    texts=[queries[index]["text"] for index in range(start, stop)],
                )
                save_chunk(
                    chunk_root,
                    "text",
                    start,
                    stop,
                    embeddings,
                    time.monotonic() - started,
                )
                report["completed_text_chunks"] += 1
                report["pending_text_chunks"] -= 1
                atomic_write_json(metrics_path, report)
            report["gpu_peak_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(device)
            )
            report["gpu_peak_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(device)
            )

        candidate_identity = consolidate_chunks(
            chunk_root,
            "audio",
            len(candidates),
            audio_batch_size,
            embedding_dim,
            output_dir / "candidate_embeddings.npy",
        )
        query_identity = consolidate_chunks(
            chunk_root,
            "text",
            len(queries),
            text_batch_size,
            embedding_dim,
            output_dir / "query_embeddings.npy",
        )
        report.update(
            {
                "status": "complete",
                "finished_at": utc_now(),
                "candidate_embedding_shape": [len(candidates), embedding_dim],
                "query_embedding_shape": [len(queries), embedding_dim],
                "artifacts": {
                    "candidate_embeddings": candidate_identity,
                    "query_embeddings": query_identity,
                    "candidate_metadata": file_identity(
                        output_dir / "candidate_metadata.jsonl"
                    ),
                    "query_metadata": file_identity(
                        output_dir / "query_metadata.jsonl"
                    ),
                },
                "error": None,
            }
        )
        atomic_write_json(metrics_path, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except BaseException as exc:
        report.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        if identity_validated or not metrics_preexisting:
            atomic_write_json(metrics_path, report)
        atomic_write_json(
            attempt_dir / "failure.json",
            {
                "status": "failed",
                "finished_at": report["finished_at"],
                "error": report["error"],
                "traceback": report["traceback"],
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
