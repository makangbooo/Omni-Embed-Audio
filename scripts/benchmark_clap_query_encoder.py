"""Benchmark a checkpoint-pinned public CLAP adapter on canonical Clotho."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_oea_embeddings import (  # noqa: E402
    atomic_write_json,
    file_identity,
    load_manifest,
)


MODEL_IDS = ("laion_clap", "mga_clap", "m2d_clap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_IDS, required=True)
    parser.add_argument("--benchmark-config", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError("latencies must be finite and positive")
    return {
        "count": int(array.size),
        "mean_ms": float(np.mean(array)),
        "std_ms": float(np.std(array)),
        "min_ms": float(np.min(array)),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "max_ms": float(np.max(array)),
        "throughput_queries_per_second": float(1000.0 / np.mean(array)),
    }


def load_adapter(model: str, root: Path) -> Any:
    if model == "laion_clap":
        from AudioRetrieval.models.laion_clap_adapter import LaionClapAdapter

        tokenizers = {
            "bert-base-uncased": str(root / "laion-clap-tokenizers/bert-base-uncased"),
            "roberta-base": str(root / "laion-clap-tokenizers/roberta-base"),
            "facebook/bart-base": str(root / "laion-clap-tokenizers/bart-base"),
        }
        return LaionClapAdapter(
            ckpt_path=root / "laion-clap/630k-audioset-best.pt",
            tokenizer_paths=tokenizers,
        )
    if model == "mga_clap":
        from AudioRetrieval.models.mga_clap_adapter import MGAClapAdapter

        return MGAClapAdapter(
            repo_path=root / "mga-clap/source",
            ckpt_path=root / "mga-clap/pretrained_models/models/model.pt",
            bert_tokenizer_path=root / "laion-clap-tokenizers/bert-base-uncased",
            expected_checkpoint_sha256="8703740b738e973a5b4d8a18a074ad56880e98f8ba21cd618d7d7ee5422d6e26",
        )
    from AudioRetrieval.models.m2d_clap_adapter import M2DClapAdapter

    return M2DClapAdapter(
        weight_file=root / "m2d-clap/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth",
        bert_tokenizer_path=root / "laion-clap-tokenizers/bert-base-uncased",
    )


def checkpoint_path(model: str, root: Path) -> Path:
    paths = {
        "laion_clap": root / "laion-clap/630k-audioset-best.pt",
        "mga_clap": root / "mga-clap/pretrained_models/models/model.pt",
        "m2d_clap": root / "m2d-clap/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth",
    }
    return paths[model]


def module_for(adapter: Any) -> Any:
    import torch

    candidate = getattr(adapter, "model", adapter)
    if isinstance(candidate, torch.nn.Module):
        return candidate
    nested = getattr(candidate, "model", None)
    if isinstance(nested, torch.nn.Module):
        return nested
    return candidate


def timed(torch: Any, fn: Callable[[], Any], device: Any) -> float:
    torch.cuda.synchronize(device)
    start = time.perf_counter_ns()
    value = fn()
    torch.cuda.synchronize(device)
    del value
    return (time.perf_counter_ns() - start) / 1_000_000.0


def main() -> int:
    args = parse_args()
    import torch

    benchmark_config_path = args.benchmark_config.resolve()
    benchmark_config = json.loads(benchmark_config_path.read_text(encoding="utf-8"))
    if benchmark_config.get("schema_version") != 1:
        raise ValueError("unsupported benchmark config schema_version")
    protocol = benchmark_config.get("timing_protocol")
    paper_values = benchmark_config.get("paper_values", {}).get(args.model)
    if not isinstance(protocol, dict) or not isinstance(paper_values, dict):
        raise ValueError("benchmark config lacks model protocol or paper values")
    audio_count = protocol.get("audio_measurement_count")
    text_count = protocol.get("text_measurement_count")
    if protocol.get("batch_size") != 1:
        raise ValueError("efficiency benchmark requires batch_size=1")
    if (
        not isinstance(audio_count, int)
        or isinstance(audio_count, bool)
        or audio_count <= 0
        or not isinstance(text_count, int)
        or isinstance(text_count, bool)
        or text_count <= 0
        or text_count % audio_count != 0
    ):
        raise ValueError("benchmark measurement counts are invalid")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.json"
    if metrics_path.exists():
        raise FileExistsError(metrics_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": now(),
        "finished_at": None,
        "model": args.model,
        "paper_model_label": paper_values["paper_model"],
        "dataset": benchmark_config["dataset"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status_short": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip(),
        "benchmark_config": file_identity(benchmark_config_path),
        "manifest": file_identity(args.manifest.resolve()),
        "checkpoint": file_identity(checkpoint_path(args.model, args.model_root.resolve())),
        "source_usage": {
            "oea_official_source_used": True,
            "adapter": f"AudioRetrieval/models/{args.model}_adapter.py",
        },
        "timing_protocol": protocol,
        "paper_reference": {**paper_values, "source": "PAPER Table 5 / Appendix Table 16", "hardware": "A100-SXM4-80GB"},
        "error": None,
    }
    atomic_write_json(metrics_path, report)
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one CUDA GPU is required")
        device = torch.device("cuda:0")
        if torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 4090":
            raise RuntimeError("benchmark requires NVIDIA GeForce RTX 4090")
        rows = load_manifest(
            args.manifest.resolve(),
            audio_count,
            text_count // audio_count,
        )
        texts = [caption for row in rows for caption in row["captions"]]
        adapter = load_adapter(args.model, args.model_root.resolve())
        module = module_for(adapter)
        all_parameters = sum(parameter.numel() for parameter in module.parameters())
        trainable_parameters = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
        report["hardware"] = {"gpu_name": torch.cuda.get_device_name(device), "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory), "torch": torch.__version__, "cuda": torch.version.cuda}
        report["parameters"] = {"all_parameters": int(all_parameters), "trainable_parameters": int(trainable_parameters), "all_parameters_m": float(all_parameters / 1_000_000.0), "trainable_parameters_m": float(trainable_parameters / 1_000_000.0)}
        for _ in range(int(protocol["warmup_iterations"])):
            timed(torch, lambda: adapter.encode_audio([rows[0]["audio_path"]], batch_size=1, device="cuda"), device)
            timed(torch, lambda: adapter.encode_text([texts[0]], batch_size=1, device="cuda"), device)
        torch.cuda.reset_peak_memory_stats(device)
        audio = [timed(torch, lambda path=row["audio_path"]: adapter.encode_audio([path], batch_size=1, device="cuda"), device) for row in rows]
        text = [timed(torch, lambda value=value: adapter.encode_text([value], batch_size=1, device="cuda"), device) for value in texts]
        latency_path = output / "latencies.jsonl"
        with latency_path.open("x", encoding="utf-8") as stream:
            for index, value in enumerate(audio):
                stream.write(json.dumps({"modality": "audio", "index": index, "latency_ms": value}) + "\n")
            for index, value in enumerate(text):
                stream.write(json.dumps({"modality": "text", "index": index, "latency_ms": value}) + "\n")
        report.update({"status": "complete", "finished_at": now(), "audio": summary(audio), "text": summary(text), "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)), "artifacts": {"latencies": file_identity(latency_path)}})
        atomic_write_json(metrics_path, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except BaseException as exc:
        report.update({"status": "failed", "finished_at": now(), "error": repr(exc)})
        atomic_write_json(metrics_path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
