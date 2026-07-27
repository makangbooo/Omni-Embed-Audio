#!/usr/bin/env python3
"""Generate resumable base-only audio and caption embeddings from a locked model."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
import sys
import time
import traceback
import warnings
from typing import Any, Mapping

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""} and str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.build_official_oea_eval_config import validate_file_inventory
from scripts.build_vanilla_backbone_eval_config import (
    tagged_value,
    validate_protocol,
    verify_vanilla_model_lock_binding,
)
from scripts.generate_oea_embeddings import (
    atomic_write_json,
    candidate_metadata,
    consolidate_chunks,
    ensure_run_identity,
    file_identity,
    git_output,
    immutable_json_text,
    immutable_jsonl_text,
    load_manifest,
    load_verified_chunk,
    query_metadata,
    ranges,
    save_chunk,
    utc_now,
    verify_fixed_file,
    write_text_once_or_verify,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attempt-dir", type=Path)
    return parser.parse_args()


def positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported vanilla embedding config schema_version")
    required = {
        "audio_batch_size",
        "backbone_id",
        "base_model",
        "caption_count_per_audio",
        "dataset",
        "embedding_output",
        "expected_examples",
        "experiment_prefix",
        "model",
        "model_config",
        "protocol",
        "protocol_config",
        "resolution_git_commit",
        "seed",
        "text_batch_size",
        "vanilla_model_lock",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"vanilla embedding config missing fields: {missing}")
    forbidden = {"checkpoint", "official_model_lock", "official_variant_id"}
    present = sorted(forbidden & set(config))
    if present:
        raise ValueError(f"vanilla embedding config contains OEA-only fields: {present}")
    for field in (
        "audio_batch_size",
        "caption_count_per_audio",
        "expected_examples",
        "text_batch_size",
    ):
        positive_integer(config[field], field)
    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool):
        raise ValueError("seed must be an integer")
    for field in ("backbone_id", "dataset", "experiment_prefix", "model"):
        if not isinstance(config[field], str) or not config[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    validate_protocol(config["protocol"], "protocol")
    base = config.get("base_model")
    if not isinstance(base, dict):
        raise ValueError("base_model must be an object")
    validate_file_inventory(base.get("files"), "base_model.files")
    output = config.get("embedding_output")
    if not isinstance(output, dict) or output.get("projection_head") != "none":
        raise ValueError("vanilla embedding output must explicitly disable projection")
    model_config = config.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("model_config must be an object")
    for key in (
        "audio_max_length",
        "attn_implementation",
        "passage_prefix",
        "query_prefix",
        "text_max_length",
        "torch_dtype",
        "trust_remote_code",
    ):
        tagged_value(model_config, key, "model_config")
    positive_integer(
        tagged_value(model_config, "text_max_length", "model_config"),
        "model_config.text_max_length",
    )
    audio_max_length = tagged_value(
        model_config, "audio_max_length", "model_config"
    )
    if audio_max_length is not None:
        positive_integer(audio_max_length, "model_config.audio_max_length")
    if tagged_value(model_config, "torch_dtype", "model_config") != "bfloat16":
        raise ValueError("formal vanilla generation must use bfloat16")
    if tagged_value(model_config, "trust_remote_code", "model_config") is not True:
        raise ValueError("formal vanilla generation must fix trust_remote_code=true")
    return config


def _local_files(base_dir: Path) -> tuple[set[str], list[str]]:
    files: set[str] = set()
    unsafe: list[str] = []
    for path in sorted(base_dir.rglob("*")):
        relative_path = path.relative_to(base_dir)
        if ".cache" in relative_path.parts:
            continue
        relative = relative_path.as_posix()
        if path.is_symlink():
            unsafe.append(relative)
        elif path.is_file():
            files.add(relative)
    return files, unsafe


def hidden_size_from_locked_config(value: Mapping[str, Any]) -> tuple[int, str]:
    candidates: list[tuple[str, Any]] = []
    text_config = value.get("text_config")
    if isinstance(text_config, dict):
        candidates.append(("text_config.hidden_size", text_config.get("hidden_size")))
    thinker_config = value.get("thinker_config")
    if isinstance(thinker_config, dict) and isinstance(
        thinker_config.get("text_config"), dict
    ):
        candidates.append(
            (
                "thinker_config.text_config.hidden_size",
                thinker_config["text_config"].get("hidden_size"),
            )
        )
    if len(candidates) != 1:
        raise ValueError(
            "locked base config must expose exactly one audited text hidden-size path"
        )
    source, raw_value = candidates[0]
    return positive_integer(raw_value, f"locked base {source}"), source


def hidden_size_from_runtime_config(value: Any) -> tuple[int, str]:
    candidates: list[tuple[str, Any]] = []
    text_config = getattr(value, "text_config", None)
    if text_config is not None:
        candidates.append(
            ("model.config.text_config.hidden_size", getattr(text_config, "hidden_size", None))
        )
    thinker_config = getattr(value, "thinker_config", None)
    thinker_text_config = getattr(thinker_config, "text_config", None)
    if thinker_text_config is not None:
        candidates.append(
            (
                "model.config.thinker_config.text_config.hidden_size",
                getattr(thinker_text_config, "hidden_size", None),
            )
        )
    if len(candidates) != 1:
        raise ValueError(
            "runtime model config must expose exactly one audited text hidden-size path"
        )
    source, raw_value = candidates[0]
    return positive_integer(raw_value, source), source


def verify_locked_base_files(
    config: Mapping[str, Any], model_root: Path
) -> tuple[Path, list[dict[str, Any]], int, str]:
    model_root = model_root.resolve()
    base = config["base_model"]
    base_dir = (model_root / base["local_subdir"]).resolve()
    if not base_dir.is_relative_to(model_root) or base_dir == model_root:
        raise ValueError("base model directory escapes model root")
    if not base_dir.is_dir():
        raise FileNotFoundError(base_dir)
    expected = validate_file_inventory(base["files"], "base_model.files")
    actual, unsafe = _local_files(base_dir)
    if unsafe:
        raise RuntimeError(f"base model contains symlinks: {unsafe[:20]}")
    missing = sorted(set(expected) - actual)
    extra = sorted(actual - set(expected))
    if missing or extra:
        raise RuntimeError(
            f"base model inventory drift: missing={missing[:20]}, extra={extra[:20]}"
        )
    verified = [
        verify_fixed_file(base_dir / relative, specification)
        for relative, specification in expected.items()
    ]
    config_path = base_dir / "config.json"
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    hidden_size, hidden_size_source = hidden_size_from_locked_config(model_config)
    return base_dir, verified, hidden_size, hidden_size_source


def load_model_bundle(
    config: Mapping[str, Any], base_dir: Path, hidden_size: int, report: dict[str, Any]
):
    import torch

    from AudioRetrieval.models.omni_embed_adapter import OmniEmbedAdapter

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("formal vanilla generation requires exactly one visible GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("visible GPU does not support BF16")
    device = torch.device("cuda:0")
    model_config = config["model_config"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapter = OmniEmbedAdapter(
            repo_id=config["base_model"]["repo_id"],
            local_path=str(base_dir),
            device="cuda:0",
            trust_remote_code=bool(
                tagged_value(model_config, "trust_remote_code", "model_config")
            ),
            torch_dtype=str(
                tagged_value(model_config, "torch_dtype", "model_config")
            ),
            text_max_length=int(
                tagged_value(model_config, "text_max_length", "model_config")
            ),
            attn_implementation=tagged_value(
                model_config, "attn_implementation", "model_config"
            ),
            passage_prefix=str(
                tagged_value(model_config, "passage_prefix", "model_config")
            ),
            query_prefix=str(
                tagged_value(model_config, "query_prefix", "model_config")
            ),
            audio_max_length=tagged_value(
                model_config, "audio_max_length", "model_config"
            ),
        )
    load_warnings = [str(item.message) for item in caught]
    if any("falling back to cpu" in message.lower() for message in load_warnings):
        raise RuntimeError(f"model load fell back to CPU: {load_warnings}")
    if adapter.device.type != "cuda":
        raise RuntimeError("vanilla adapter is not resident on the visible GPU")
    model = adapter.get_underlying_model()
    actual_hidden, runtime_hidden_source = hidden_size_from_runtime_config(model.config)
    if actual_hidden != hidden_size:
        raise RuntimeError(
            f"runtime hidden size differs from locked config: {actual_hidden} != {hidden_size}"
        )
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("vanilla base model unexpectedly has trainable parameters")
    report["model_load"] = {
        "hidden_size": hidden_size,
        "hidden_size_runtime_source": runtime_hidden_source,
        "projection_head_loaded": False,
        "lora_loaded": False,
        "oea_checkpoint_loaded": False,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_memory_bytes": int(
            torch.cuda.get_device_properties(device).total_memory
        ),
        "load_warnings": load_warnings,
    }
    return torch, adapter, device


def encode_base_batch(
    adapter: Any,
    expected_dimension: int,
    *,
    texts: list[str] | None = None,
    audio_paths: list[Path] | None = None,
    normalization_audit: dict[str, Any] | None = None,
) -> np.ndarray:
    if (texts is None) == (audio_paths is None):
        raise ValueError("exactly one of texts or audio_paths must be supplied")
    expected_count = len(texts) if texts is not None else len(audio_paths or [])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if texts is not None:
            embeddings = adapter.encode_text(texts, batch_size=max(1, len(texts)))
        else:
            embeddings = adapter.encode_audio(
                [str(path) for path in audio_paths or []],
                batch_size=max(1, len(audio_paths or [])),
            )
    messages = [str(item.message) for item in caught]
    if any(
        "silence" in message.lower() or "failed to load" in message.lower()
        for message in messages
    ):
        raise RuntimeError(f"audio fallback warning detected: {messages}")
    adapter_output_dtype = str(np.asarray(embeddings).dtype)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape != (expected_count, expected_dimension):
        raise RuntimeError(f"vanilla embedding shape mismatch: {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise RuntimeError("vanilla embeddings contain non-finite values")
    pre_norms = np.linalg.norm(embeddings, axis=1)
    if (
        not np.isfinite(pre_norms).all()
        or np.any(pre_norms <= np.finfo(np.float32).tiny)
    ):
        raise RuntimeError(
            "vanilla embeddings have non-finite or zero pre-normalization norms"
        )

    # The public adapter normalizes the pooled hidden state in the model's
    # runtime dtype. With the locked BF16 protocol, casting that result to
    # float32 can expose quantization error larger than the old 1e-3 assertion.
    # Re-normalizing at the cache boundary is deterministic, preserves vector
    # direction, and implements the protocol's required cosine-space output.
    embeddings = embeddings / pre_norms[:, None]
    post_norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(post_norms, 1.0, atol=1e-6):
        raise RuntimeError(
            "vanilla embeddings failed float32 cache-boundary L2 normalization"
        )
    if normalization_audit is not None:
        normalization_audit.setdefault(
            "method", "adapter_output_then_float32_cache_boundary_l2"
        )
        normalization_audit.setdefault(
            "reason",
            (
                "BF16 adapter normalization may exceed the former "
                "1e-3 float32 norm assertion"
            ),
        )
        normalization_audit["batch_count"] = (
            int(normalization_audit.get("batch_count", 0)) + 1
        )
        normalization_audit["row_count"] = (
            int(normalization_audit.get("row_count", 0)) + expected_count
        )
        normalization_audit["rows_outside_pre_atol_1e3"] = int(
            normalization_audit.get("rows_outside_pre_atol_1e3", 0)
        ) + int(np.count_nonzero(np.abs(pre_norms - 1.0) > 1e-3))
        normalization_audit["pre_norm_min"] = min(
            float(normalization_audit.get("pre_norm_min", float("inf"))),
            float(np.min(pre_norms)),
        )
        normalization_audit["pre_norm_max"] = max(
            float(normalization_audit.get("pre_norm_max", float("-inf"))),
            float(np.max(pre_norms)),
        )
        normalization_audit["pre_norm_max_abs_deviation"] = max(
            float(normalization_audit.get("pre_norm_max_abs_deviation", 0.0)),
            float(np.max(np.abs(pre_norms - 1.0))),
        )
        normalization_audit["post_norm_max_abs_deviation"] = max(
            float(normalization_audit.get("post_norm_max_abs_deviation", 0.0)),
            float(np.max(np.abs(post_norms - 1.0))),
        )
        observed_dtypes = set(normalization_audit.get("adapter_output_dtypes", []))
        observed_dtypes.add(adapter_output_dtype)
        normalization_audit["adapter_output_dtypes"] = sorted(observed_dtypes)
    return embeddings


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
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict) or existing.get("schema_version") != 1:
                raise RuntimeError(f"invalid existing generation metrics: {metrics_path}")
            report = existing
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
        lock_binding = verify_vanilla_model_lock_binding(config)
        if config.get("resolution_git_commit") != git_commit:
            raise RuntimeError(
                "resolved vanilla config was not generated at the current Git commit"
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
        (
            base_dir,
            verified_files,
            embedding_dim,
            embedding_dimension_source_path,
        ) = verify_locked_base_files(config, model_root)
        identity = {
            "schema_version": 1,
            "experiment_id": output_dir.name,
            "git_commit": git_commit,
            "config": file_identity(config_path),
            "manifest": file_identity(manifest_path),
            "model_root": str(model_root),
            "model": config["model"],
            "backbone_id": config["backbone_id"],
            "vanilla_model_lock": lock_binding,
            "dataset": config["dataset"],
            "base_revision": config["base_model"]["revision"],
            "embedding_dimension": embedding_dim,
            "embedding_dimension_source_path": embedding_dimension_source_path,
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
                "backbone_id": config["backbone_id"],
                "vanilla_model_lock": lock_binding,
                "dataset": config["dataset"],
                "seed": config["seed"],
                "protocol": config["protocol"],
                "strict_offline": strict_offline,
                "verified_model_files": verified_files,
                "embedding_dimension": embedding_dim,
                "embedding_dimension_source": (
                    f"[CODE] locked base config {embedding_dimension_source_path}"
                ),
                "projection_head_loaded": False,
                "lora_loaded": False,
                "oea_checkpoint_loaded": False,
                "candidate_count": len(candidates),
                "query_count": len(queries),
                "completed_audio_chunks": 0,
                "completed_text_chunks": 0,
            }
        )
        atomic_write_json(metrics_path, report)

        random.seed(config["seed"])
        np.random.seed(config["seed"])
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
        report.setdefault(
            "normalization_audit",
            {
                "method": "adapter_output_then_float32_cache_boundary_l2",
                "reason": (
                    "BF16 adapter normalization may exceed the former "
                    "1e-3 float32 norm assertion"
                ),
                "scope": "cumulative rows encoded by this fixed implementation",
                "batch_count": 0,
                "row_count": 0,
                "rows_outside_pre_atol_1e3": 0,
            },
        )
        atomic_write_json(metrics_path, report)

        if pending_audio or pending_text:
            torch, adapter, device = load_model_bundle(
                config, base_dir, embedding_dim, report
            )
            torch.manual_seed(config["seed"])
            torch.cuda.manual_seed_all(config["seed"])
            torch.cuda.reset_peak_memory_stats(device)
            for start, stop in pending_audio:
                started = time.monotonic()
                embeddings = encode_base_batch(
                    adapter,
                    embedding_dim,
                    audio_paths=[
                        rows[index]["audio_path"] for index in range(start, stop)
                    ],
                    normalization_audit=report["normalization_audit"],
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
                embeddings = encode_base_batch(
                    adapter,
                    embedding_dim,
                    texts=[queries[index]["text"] for index in range(start, stop)],
                    normalization_audit=report["normalization_audit"],
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
    except BaseException as error:
        report.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "error": repr(error),
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
