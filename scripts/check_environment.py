#!/usr/bin/env python3
"""Validate the OEA core environment without downloading any model or data."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


REQUIRED_IMPORTS = {
    "accelerate": "accelerate",
    "einops": "einops",
    "huggingface_hub": "huggingface-hub",
    "hydra": "hydra-core",
    "librosa": "librosa",
    "nnAudio": "nnAudio",
    "numpy": "numpy",
    "omegaconf": "omegaconf",
    "pandas": "pandas",
    "peft": "peft",
    "PIL": "Pillow",
    "safetensors": "safetensors",
    "scipy": "scipy",
    "sentencepiece": "sentencepiece",
    "soundfile": "soundfile",
    "timm": "timm",
    "tokenizers": "tokenizers",
    "torch": "torch",
    "torchaudio": "torchaudio",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "wandb": "wandb",
}

REQUIRED_INTERFACES = {
    "peft": ("LoraConfig", "TaskType", "get_peft_model"),
    "transformers.models.qwen2_5_omni": (
        "Qwen2_5OmniThinkerForConditionalGeneration",
    ),
}


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "MISSING"


def command_output(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as exc:
        return f"UNAVAILABLE: {exc!r}"
    return completed.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("cpu", "gpu"),
        default="gpu",
        help="CPU checks imports/audio only; GPU additionally requires CUDA, BF16, and NCCL.",
    )
    return parser.parse_args()


def write_report(output: Path, report: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {
        "python": sys.version,
        "mode": args.mode,
        "executable": sys.executable,
        "platform": platform.platform(),
        "environment": {
            key: os.environ.get(key)
            for key in ("CONDA_DEFAULT_ENV", "CUDA_HOME", "CUDA_VISIBLE_DEVICES")
        },
        "packages": {},
        "checks": {},
    }

    import_errors: dict[str, str] = {}
    for module_name, distribution in REQUIRED_IMPORTS.items():
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - preserve every import failure
            import_errors[module_name] = repr(exc)
        report["packages"][distribution] = package_version(distribution)

    interface_checks: dict[str, str] = {}
    for module_name, attributes in REQUIRED_INTERFACES.items():
        try:
            module = importlib.import_module(module_name)
            missing = [name for name in attributes if not hasattr(module, name)]
            if missing:
                raise AttributeError(f"missing required attributes: {missing}")
            interface_checks[module_name] = f"OK: {', '.join(attributes)}"
        except Exception as exc:  # noqa: BLE001 - preserve every interface failure
            interface_checks[module_name] = repr(exc)
            import_errors[f"{module_name} interface"] = repr(exc)
    report["checks"]["required_interfaces"] = interface_checks

    try:
        importlib.import_module("AudioRetrieval")
        report["checks"]["repository_import"] = "OK"
    except Exception as exc:  # noqa: BLE001 - preserve repository import failure
        report["checks"]["repository_import"] = repr(exc)
        import_errors["AudioRetrieval"] = repr(exc)

    report["checks"]["import_errors"] = import_errors

    essential_modules = {"numpy", "soundfile", "torch", "torchaudio"}
    if essential_modules.intersection(import_errors):
        write_report(args.output, report)
        return 1

    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    nccl_version = None
    if torch.cuda.is_available() and torch.distributed.is_nccl_available():
        raw_nccl_version = torch.cuda.nccl.version()
        nccl_version = (
            list(raw_nccl_version)
            if isinstance(raw_nccl_version, tuple)
            else raw_nccl_version
        )

    report["torch"] = {
        "version": torch.__version__,
        "compiled_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "bf16_supported": bool(
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        ),
        "distributed_available": torch.distributed.is_available(),
        "nccl_available": torch.distributed.is_nccl_available(),
        "nccl_version": nccl_version,
    }

    if not torch.cuda.is_available():
        report["checks"]["cuda"] = (
            "NOT_TESTED: no GPU is visible during the CPU installation check"
            if args.mode == "cpu"
            else "FAILED: torch.cuda.is_available() is false"
        )
    else:
        device = torch.device("cuda:0")
        properties = torch.cuda.get_device_properties(device)
        report["torch"]["device"] = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "capability": list(torch.cuda.get_device_capability(device)),
        }
        left = torch.randn((256, 256), device=device, dtype=torch.bfloat16)
        right = torch.randn((256, 256), device=device, dtype=torch.bfloat16)
        result = left @ right
        torch.cuda.synchronize()
        report["checks"]["cuda"] = {
            "bf16_matmul_shape": list(result.shape),
            "finite": bool(torch.isfinite(result).all().item()),
        }

    waveform = torch.linspace(-1.0, 1.0, 16_000).unsqueeze(0)
    resampled = torchaudio.functional.resample(waveform, 16_000, 8_000)
    report["checks"]["torchaudio_resample"] = {
        "input_shape": list(waveform.shape),
        "output_shape": list(resampled.shape),
    }

    with tempfile.TemporaryDirectory() as temporary_directory:
        audio_path = Path(temporary_directory) / "roundtrip.wav"
        sf.write(audio_path, np.zeros(1_600, dtype=np.float32), 16_000)
        decoded, sample_rate = sf.read(audio_path, dtype="float32")
        report["checks"]["soundfile_roundtrip"] = {
            "samples": int(decoded.shape[0]),
            "sample_rate": int(sample_rate),
        }

    try:
        importlib.import_module("flash_attn")
        report["checks"]["flash_attn"] = package_version("flash-attn")
    except Exception as exc:  # noqa: BLE001 - optional dependency
        report["checks"]["flash_attn"] = f"OPTIONAL_NOT_AVAILABLE: {exc!r}"

    report["nvidia_smi"] = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv",
        ]
    )

    write_report(args.output, report)

    failed = bool(import_errors)
    if args.mode == "gpu":
        failed = failed or not torch.cuda.is_available()
        failed = failed or not torch.cuda.is_bf16_supported()
        failed = failed or not torch.distributed.is_nccl_available()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
