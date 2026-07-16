#!/usr/bin/env python3
"""Run a strict-offline five-sample OEA-Qwen3B-Cl GPU smoke test."""

from __future__ import annotations

import argparse
import csv
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
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
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


def write_json(path: Path, value: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_size: int, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"size mismatch for {path}: expected {expected_size}, got {actual_size}"
        )
    started = time.monotonic()
    actual_sha256 = sha256_file(path)
    elapsed = time.monotonic() - started
    if actual_sha256.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"SHA256 mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
        )
    return {
        "path": str(path.resolve()),
        "size_bytes": actual_size,
        "sha256": actual_sha256,
        "sha256_elapsed_seconds": elapsed,
    }


def load_samples(manifest: Path, audio_dir: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with manifest.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if not isinstance(entry.get("file"), str):
                raise ValueError(f"manifest line {line_number} has no string 'file'")
            captions = entry.get("captions")
            if not isinstance(captions, list) or not captions:
                raise ValueError(f"manifest line {line_number} has no captions")
            path = (audio_dir / entry["file"]).resolve()
            samples.append(
                {
                    "file": entry["file"],
                    "path": path,
                    "caption": str(captions[0]),
                }
            )
    if len(samples) != 5:
        raise RuntimeError(f"expected exactly 5 bundled samples, found {len(samples)}")
    if len({sample["file"] for sample in samples}) != len(samples):
        raise RuntimeError("bundled sample filenames are not unique")
    return samples


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported smoke config schema")
    if len(config.get("queries", [])) != 5:
        raise ValueError("smoke config must contain exactly five queries")
    return config


def value(config: dict[str, Any], section: str, key: str) -> Any:
    return config[section][key]["value"]


def cuda_memory_snapshot(torch: Any, phase: str) -> dict[str, Any]:
    return {
        "phase": phase,
        "timestamp": utc_now(),
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def synchronize(torch: Any) -> None:
    torch.cuda.synchronize()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    model_root = args.model_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "smoke_passed": False,
        "started_at": utc_now(),
        "finished_at": None,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output("status", "--short"),
        "strict_offline": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE"),
        },
        "not_a_paper_table_result": True,
        "error": None,
    }
    write_json(output_dir / "metrics.json", report)

    # JSON is valid YAML 1.2.  Keeping the exact resolved values here avoids an
    # additional serializer dependency while satisfying the per-run config.yaml
    # audit contract.
    resolved_config = json.loads(json.dumps(config))
    resolved_config["resolved_paths"] = {
        "model_root": str(model_root),
        "output_dir": str(output_dir),
        "source_config": str(config_path),
    }
    write_json(output_dir / "config.yaml", resolved_config)
    memory_snapshots: list[dict[str, Any]] = []

    try:
        import gc

        import numpy as np
        import soundfile as sf
        import torch

        from AudioRetrieval.models.omni_embed_adapter import OmniEmbedAdapter
        from AudioRetrieval.training.oea.train_omniembed_lora import (
            ProjectionHead,
            attach_lora,
            resolve_text_hidden_size,
        )

        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("this smoke test requires one visible CUDA GPU")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("visible CUDA GPU does not report BF16 support")
        if any(value != "1" for value in report["strict_offline"].values()):
            raise RuntimeError("all strict-offline environment variables must equal 1")

        device = torch.device("cuda:0")
        seed = int(value(config, "smoke_config", "seed"))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)

        report["software"] = {
            "python": sys.version,
            "torch": torch.__version__,
        }
        report["gpu"] = {
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": int(
                torch.cuda.get_device_properties(device).total_memory
            ),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "visible_device_count": int(torch.cuda.device_count()),
        }
        memory_snapshots = [cuda_memory_snapshot(torch, "start")]

        base_dir = (model_root / config["base_model"]["local_subdir"]).resolve()
        checkpoint_path = (
            model_root / config["checkpoint"]["local_subpath"]
        ).resolve()
        verified_files: list[dict[str, Any]] = []
        for relative_path, expected in config["base_model"]["files"].items():
            verified_files.append(
                verify_file(
                    base_dir / relative_path,
                    int(expected["size_bytes"]),
                    expected["sha256"],
                )
            )
        verified_files.append(
            verify_file(
                checkpoint_path,
                int(config["checkpoint"]["size_bytes"]),
                config["checkpoint"]["sha256"],
            )
        )
        report["verified_files"] = verified_files
        write_json(output_dir / "metrics.json", report)

        unsafe_globals = sorted(
            torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint_path)
        )
        if unsafe_globals:
            raise RuntimeError(
                f"derived inference checkpoint has unsafe globals: {unsafe_globals}"
            )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        required_sections = {"lora_state_dict", "audio_head", "text_head", "config"}
        missing_sections = sorted(required_sections - set(checkpoint))
        if missing_sections:
            raise RuntimeError(f"checkpoint sections missing: {missing_sections}")

        adapter_started = time.monotonic()
        adapter = OmniEmbedAdapter(
            repo_id=config["base_model"]["repo_id"],
            local_path=str(base_dir),
            device="cuda:0",
            trust_remote_code=False,
            torch_dtype=value(config, "model_config", "torch_dtype"),
            passage_prefix=value(config, "model_config", "passage_prefix"),
            query_prefix=value(config, "model_config", "query_prefix"),
        )
        synchronize(torch)
        report["base_model_load_elapsed_seconds"] = time.monotonic() - adapter_started
        memory_snapshots.append(cuda_memory_snapshot(torch, "base_model_loaded"))

        lora_config = SimpleNamespace(
            lora_rank=int(value(config, "model_config", "lora_rank")),
            lora_alpha=int(value(config, "model_config", "lora_alpha")),
            lora_dropout=float(value(config, "model_config", "lora_dropout")),
            lora_targets=list(value(config, "model_config", "lora_targets")),
        )
        peft_model = attach_lora(adapter.get_underlying_model(), lora_config)
        expected_lora_keys = set(checkpoint["lora_state_dict"])
        actual_lora_keys = {
            key for key in peft_model.state_dict() if "lora_" in key.lower()
        }
        missing_lora_keys = sorted(expected_lora_keys - actual_lora_keys)
        extra_lora_keys = sorted(actual_lora_keys - expected_lora_keys)
        if missing_lora_keys or extra_lora_keys:
            raise RuntimeError(
                "LoRA key mismatch: "
                f"missing={missing_lora_keys[:10]}, extra={extra_lora_keys[:10]}"
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
                f"missing_lora={missing_loaded_lora[:10]}, "
                f"unexpected={incompatible.unexpected_keys[:10]}"
            )
        adapter.set_underlying_model(peft_model)

        hidden_size = int(resolve_text_hidden_size(peft_model))
        projection_dim = int(value(config, "model_config", "projection_dim"))
        projection_dropout = float(
            value(config, "model_config", "projection_dropout")
        )
        audio_head = ProjectionHead(
            hidden_size, projection_dim, projection_dropout
        ).to(device)
        text_head = ProjectionHead(
            hidden_size, projection_dim, projection_dropout
        ).to(device)
        audio_head.load_state_dict(checkpoint["audio_head"], strict=True)
        text_head.load_state_dict(checkpoint["text_head"], strict=True)
        audio_head.eval()
        text_head.eval()
        synchronize(torch)
        memory_snapshots.append(cuda_memory_snapshot(torch, "lora_and_heads_loaded"))

        report["checkpoint_load"] = {
            "unsafe_globals": unsafe_globals,
            "lora_tensor_count": len(expected_lora_keys),
            "model_lora_tensor_count": len(actual_lora_keys),
            "missing_lora_keys": missing_lora_keys,
            "extra_lora_keys": extra_lora_keys,
            "missing_non_lora_base_keys": len(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "hidden_size": hidden_size,
            "projection_dim": projection_dim,
        }
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()

        manifest = (REPOSITORY_ROOT / config["samples"]["manifest"]).resolve()
        audio_dir = (REPOSITORY_ROOT / config["samples"]["audio_dir"]).resolve()
        samples = load_samples(manifest, audio_dir)
        audio_metadata: list[dict[str, Any]] = []
        for sample in samples:
            if not sample["path"].is_file():
                raise FileNotFoundError(sample["path"])
            info = sf.info(str(sample["path"]))
            if info.frames <= 0 or info.samplerate <= 0:
                raise RuntimeError(f"invalid audio metadata: {sample['path']}")
            audio_metadata.append(
                {
                    "file": sample["file"],
                    "path": str(sample["path"]),
                    "caption": sample["caption"],
                    "frames": int(info.frames),
                    "sample_rate": int(info.samplerate),
                    "duration_seconds": float(info.duration),
                    "channels": int(info.channels),
                    "format": info.format,
                    "subtype": info.subtype,
                }
            )
        write_json(output_dir / "audio_manifest.json", audio_metadata)

        audio_raw: list[Any] = []
        audio_embeddings: list[Any] = []
        audio_timings: list[dict[str, Any]] = []
        for sample in samples:
            synchronize(torch)
            started = time.monotonic()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                raw = adapter.encode_audio([str(sample["path"])], batch_size=1)
            messages = [str(item.message) for item in caught]
            if any("Using silence" in message for message in messages):
                raise RuntimeError(
                    f"audio decoder fallback was triggered for {sample['file']}: {messages}"
                )
            with torch.inference_mode():
                projected = audio_head(
                    torch.from_numpy(raw).to(device=device, dtype=torch.float32)
                ).cpu().float().numpy()
            synchronize(torch)
            audio_raw.append(raw[0])
            audio_embeddings.append(projected[0])
            audio_timings.append(
                {
                    "file": sample["file"],
                    "elapsed_seconds": time.monotonic() - started,
                    "warnings": messages,
                }
            )
            memory_snapshots.append(
                cuda_memory_snapshot(torch, f"audio_encoded:{sample['file']}")
            )

        text_raw: list[Any] = []
        text_embeddings: list[Any] = []
        text_timings: list[dict[str, Any]] = []
        for query in config["queries"]:
            synchronize(torch)
            started = time.monotonic()
            raw = adapter.encode_text([query["text"]], batch_size=1)
            with torch.inference_mode():
                projected = text_head(
                    torch.from_numpy(raw).to(device=device, dtype=torch.float32)
                ).cpu().float().numpy()
            synchronize(torch)
            text_raw.append(raw[0])
            text_embeddings.append(projected[0])
            text_timings.append(
                {
                    "type": query["type"],
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
            memory_snapshots.append(
                cuda_memory_snapshot(torch, f"text_encoded:{query['type']}")
            )

        audio_raw_array = np.stack(audio_raw).astype(np.float32, copy=False)
        text_raw_array = np.stack(text_raw).astype(np.float32, copy=False)
        audio_array = np.stack(audio_embeddings).astype(np.float32, copy=False)
        text_array = np.stack(text_embeddings).astype(np.float32, copy=False)
        similarity = text_array @ audio_array.T

        expected_shape = (5, projection_dim)
        if audio_array.shape != expected_shape or text_array.shape != expected_shape:
            raise RuntimeError(
                f"embedding shape mismatch: audio={audio_array.shape}, text={text_array.shape}"
            )
        if not np.isfinite(audio_array).all() or not np.isfinite(text_array).all():
            raise RuntimeError("non-finite projected embedding values detected")
        audio_norms = np.linalg.norm(audio_array, axis=1)
        text_norms = np.linalg.norm(text_array, axis=1)
        if not np.allclose(audio_norms, 1.0, atol=1e-3):
            raise RuntimeError(f"audio embeddings are not L2 normalized: {audio_norms}")
        if not np.allclose(text_norms, 1.0, atol=1e-3):
            raise RuntimeError(f"text embeddings are not L2 normalized: {text_norms}")

        np.save(output_dir / "candidate_embeddings.npy", audio_array)
        np.save(output_dir / "query_embeddings.npy", text_array)
        np.save(output_dir / "candidate_backbone_embeddings.npy", audio_raw_array)
        np.save(output_dir / "query_backbone_embeddings.npy", text_raw_array)

        audio_names = [sample["file"] for sample in samples]
        rankings: list[dict[str, Any]] = []
        target_audio = config["smoke_config"]["target_audio"]
        target_top1_count = 0
        for query, row in zip(config["queries"], similarity):
            order = np.argsort(-row)
            ranked = [
                {
                    "rank": rank,
                    "audio": audio_names[int(index)],
                    "cosine_similarity": float(row[int(index)]),
                }
                for rank, index in enumerate(order, start=1)
            ]
            if ranked[0]["audio"] == target_audio:
                target_top1_count += 1
            rankings.append(
                {
                    "query_type": query["type"],
                    "query": query["text"],
                    "target_audio": target_audio,
                    "target_rank": next(
                        item["rank"] for item in ranked if item["audio"] == target_audio
                    ),
                    "ranking": ranked,
                }
            )
        write_json(output_dir / "rankings.json", rankings)

        with (output_dir / "similarity.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["query_type", "query", *audio_names])
            for query, row in zip(config["queries"], similarity):
                writer.writerow(
                    [query["type"], query["text"], *[float(item) for item in row]]
                )

        memory_snapshots.append(cuda_memory_snapshot(torch, "complete"))
        write_json(output_dir / "gpu_memory.json", memory_snapshots)
        report.update(
            {
                "status": "complete",
                "smoke_passed": True,
                "finished_at": utc_now(),
                "seed": seed,
                "candidate_count": len(samples),
                "query_count": len(config["queries"]),
                "candidate_embedding_shape": list(audio_array.shape),
                "query_embedding_shape": list(text_array.shape),
                "candidate_backbone_embedding_shape": list(audio_raw_array.shape),
                "query_backbone_embedding_shape": list(text_raw_array.shape),
                "candidate_norm_min": float(audio_norms.min()),
                "candidate_norm_max": float(audio_norms.max()),
                "query_norm_min": float(text_norms.min()),
                "query_norm_max": float(text_norms.max()),
                "target_audio": target_audio,
                "target_top1_count": target_top1_count,
                "target_top1_rate": target_top1_count / len(config["queries"]),
                "audio_timings": audio_timings,
                "text_timings": text_timings,
                "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "pass_criteria": {
                    "exact_lora_key_match": True,
                    "strict_projection_head_load": True,
                    "expected_embedding_shapes": True,
                    "all_embeddings_finite": True,
                    "all_projected_embeddings_l2_normalized": True,
                    "retrieval_ranking_is_not_a_pass_criterion": True
                },
            }
        )
        write_json(output_dir / "metrics.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except BaseException as exc:
        if memory_snapshots:
            write_json(output_dir / "gpu_memory.json", memory_snapshots)
        report.update(
            {
                "status": "failed",
                "smoke_passed": False,
                "finished_at": utc_now(),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(output_dir / "metrics.json", report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
