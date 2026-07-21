#!/usr/bin/env python3
"""Benchmark the lock-bound official OEA audio and text query encoders."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.generate_oea_embeddings import (  # noqa: E402
    atomic_write_json,
    file_identity,
    load_config as load_model_config,
    load_manifest,
    load_model_bundle,
    project_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def load_benchmark_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported benchmark config schema_version")
    required = {
        "claim_scope",
        "dataset",
        "experiment_prefix",
        "hardware",
        "model",
        "model_lock",
        "model_protocol_config",
        "paper_model_label",
        "paper_values",
        "timing_protocol",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"benchmark config missing fields: {missing}")
    for field in ("dataset", "experiment_prefix", "model", "paper_model_label"):
        if not isinstance(config[field], str) or not config[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    hardware = config["hardware"]
    if not isinstance(hardware, dict) or not isinstance(
        hardware.get("required_gpu_name"), str
    ):
        raise ValueError("hardware.required_gpu_name must be a string")
    claim_scope = config["claim_scope"]
    if not isinstance(claim_scope, dict):
        raise ValueError("claim_scope must be an object")
    for field in ("hardware_alignment", "result_label", "allowed_claim"):
        if (
            not isinstance(claim_scope.get(field), str)
            or not claim_scope[field].strip()
        ):
            raise ValueError(f"claim_scope.{field} must be a non-empty string")
    protocol = config["timing_protocol"]
    if not isinstance(protocol, dict):
        raise ValueError("timing_protocol must be an object")
    for field in (
        "audio_measurement_count",
        "batch_size",
        "text_measurement_count",
        "warmup_iterations",
    ):
        value = protocol.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"timing_protocol.{field} must be a positive integer")
    if protocol["batch_size"] != 1:
        raise ValueError("paper-comparison benchmark requires batch_size=1")
    return config


def latency_summary(latencies_ms: Sequence[float]) -> dict[str, float | int]:
    if not latencies_ms:
        raise ValueError("latency list must not be empty")
    values = np.asarray(latencies_ms, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("latencies must be finite and positive")
    return {
        "count": int(values.size),
        "mean_ms": float(np.mean(values)),
        "std_ms": float(np.std(values, ddof=0)),
        "min_ms": float(np.min(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(np.max(values)),
        "throughput_queries_per_second": float(1000.0 / np.mean(values)),
    }


def timed_project(
    torch: Any,
    adapter: Any,
    model: Any,
    head: Any,
    device: Any,
    *,
    text: str | None = None,
    audio_path: Path | None = None,
) -> tuple[np.ndarray, float]:
    if (text is None) == (audio_path is None):
        raise ValueError("exactly one of text or audio_path must be provided")
    torch.cuda.synchronize(device)
    started = time.perf_counter_ns()
    if text is not None:
        embedding = project_batch(
            torch, adapter, model, head, device, texts=[text]
        )
    elif audio_path is not None:
        embedding = project_batch(
            torch, adapter, model, head, device, audio_paths=[audio_path]
        )
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return embedding, elapsed_ms


def trainable_parameter_summary(
    model: Any, audio_head: Any, text_head: Any
) -> dict[str, int | float]:
    lora_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "lora_" in name.lower()
    )
    audio_head_parameters = sum(parameter.numel() for parameter in audio_head.parameters())
    text_head_parameters = sum(parameter.numel() for parameter in text_head.parameters())
    total = lora_parameters + audio_head_parameters + text_head_parameters
    return {
        "lora_parameters": int(lora_parameters),
        "audio_head_parameters": int(audio_head_parameters),
        "text_head_parameters": int(text_head_parameters),
        "total_trainable_parameters": int(total),
        "total_trainable_parameters_m": float(total / 1_000_000.0),
    }


def main() -> int:
    args = parse_args()
    benchmark_config_path = args.benchmark_config.resolve()
    model_config_path = args.model_config.resolve()
    model_root = args.model_root.resolve()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        raise FileExistsError(f"benchmark metrics already exist: {metrics_path}")
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "metrics_path": str(metrics_path),
        "error": None,
    }
    atomic_write_json(metrics_path, report)
    try:
        benchmark_config = load_benchmark_config(benchmark_config_path)
        model_config = load_model_config(model_config_path)
        if not output_dir.name.startswith(benchmark_config["experiment_prefix"] + "_"):
            raise ValueError("output directory name must start with experiment_prefix")
        git_commit = git_output("rev-parse", "HEAD")
        git_status = git_output("status", "--short")
        if git_status:
            raise RuntimeError(f"formal benchmark requires clean Git: {git_status!r}")
        if model_config["model"] != benchmark_config["model"]:
            raise ValueError("benchmark/model config model mismatch")
        if model_config["dataset"] != benchmark_config["dataset"]:
            raise ValueError("benchmark/model config dataset mismatch")
        rows = load_manifest(
            manifest_path,
            int(model_config["expected_examples"]),
            int(model_config["caption_count_per_audio"]),
        )
        texts = [caption for row in rows for caption in row["captions"]]
        protocol = benchmark_config["timing_protocol"]
        audio_count = int(protocol["audio_measurement_count"])
        text_count = int(protocol["text_measurement_count"])
        warmup_count = int(protocol["warmup_iterations"])
        if audio_count != len(rows) or text_count != len(texts):
            raise ValueError(
                "formal benchmark counts must cover the full canonical manifest"
            )
        report.update(
            {
                "experiment_id": output_dir.name,
                "git_commit": git_commit,
                "git_status_short": git_status,
                "model": benchmark_config["model"],
                "paper_model_label": benchmark_config["paper_model_label"],
                "dataset": benchmark_config["dataset"],
                "claim_scope": benchmark_config["claim_scope"],
                "benchmark_config": file_identity(benchmark_config_path),
                "model_config": file_identity(model_config_path),
                "manifest": file_identity(manifest_path),
                "model_root": str(model_root),
                "timing_protocol": protocol,
                "paper_values": benchmark_config["paper_values"],
            }
        )
        atomic_write_json(metrics_path, report)

        random.seed(int(model_config["seed"]))
        np.random.seed(int(model_config["seed"]))
        load_started = time.perf_counter()
        torch, adapter, model, audio_head, text_head, device = load_model_bundle(
            model_config, model_root, report
        )
        torch.manual_seed(int(model_config["seed"]))
        torch.cuda.manual_seed_all(int(model_config["seed"]))
        torch.cuda.synchronize(device)
        report["model_load_seconds"] = time.perf_counter() - load_started
        observed_gpu_name = torch.cuda.get_device_name(device)
        required_gpu_name = benchmark_config["hardware"]["required_gpu_name"]
        if observed_gpu_name != required_gpu_name:
            raise RuntimeError(
                f"benchmark config requires {required_gpu_name!r}, "
                f"observed {observed_gpu_name!r}"
            )
        report["hardware"] = {
            "gpu_name": observed_gpu_name,
            "gpu_total_memory_bytes": int(
                torch.cuda.get_device_properties(device).total_memory
            ),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
        report["parameters"] = trainable_parameter_summary(
            model, audio_head, text_head
        )
        report["model_resident_allocated_bytes"] = int(
            torch.cuda.memory_allocated(device)
        )
        report["model_resident_reserved_bytes"] = int(
            torch.cuda.memory_reserved(device)
        )
        atomic_write_json(metrics_path, report)

        warmup_audio = rows[0]["audio_path"]
        warmup_text = texts[0]
        for _ in range(warmup_count):
            timed_project(
                torch,
                adapter,
                model,
                audio_head,
                device,
                audio_path=warmup_audio,
            )
            timed_project(
                torch, adapter, model, text_head, device, text=warmup_text
            )
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        audio_latencies: list[float] = []
        for row in rows:
            embedding, elapsed_ms = timed_project(
                torch,
                adapter,
                model,
                audio_head,
                device,
                audio_path=row["audio_path"],
            )
            if embedding.shape != (1, 512) or not np.all(np.isfinite(embedding)):
                raise RuntimeError("audio benchmark produced an invalid embedding")
            audio_latencies.append(elapsed_ms)

        text_latencies: list[float] = []
        for text in texts:
            embedding, elapsed_ms = timed_project(
                torch, adapter, model, text_head, device, text=text
            )
            if embedding.shape != (1, 512) or not np.all(np.isfinite(embedding)):
                raise RuntimeError("text benchmark produced an invalid embedding")
            text_latencies.append(elapsed_ms)

        torch.cuda.synchronize(device)
        latency_path = output_dir / "latencies.jsonl"
        with latency_path.open("x", encoding="utf-8") as handle:
            for index, value in enumerate(audio_latencies):
                handle.write(
                    json.dumps(
                        {
                            "modality": "audio",
                            "index": index,
                            "sample_id": rows[index]["sample_id"],
                            "latency_ms": value,
                        }
                    )
                    + "\n"
                )
            for index, value in enumerate(text_latencies):
                handle.write(
                    json.dumps(
                        {"modality": "text", "index": index, "latency_ms": value}
                    )
                    + "\n"
                )
        report.update(
            {
                "status": "complete",
                "finished_at": utc_now(),
                "audio": latency_summary(audio_latencies),
                "text": latency_summary(text_latencies),
                "gpu_peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "gpu_peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                ),
                "artifacts": {"latencies": file_identity(latency_path)},
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
        atomic_write_json(metrics_path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
