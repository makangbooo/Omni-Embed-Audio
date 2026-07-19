#!/usr/bin/env python3
"""Fail closed unless exactly one visible A100 GPU has at least 79 GiB."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence


MINIMUM_TOTAL_MEMORY_BYTES = 79 * 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_observation(
    *,
    cuda_available: bool,
    bf16_supported: bool,
    devices: Sequence[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    if not cuda_available:
        errors.append("CUDA is not available")
    if len(devices) != 1:
        errors.append(f"expected exactly one visible GPU, found {len(devices)}")
    if not bf16_supported:
        errors.append("BF16 is not supported")
    if len(devices) == 1:
        device = devices[0]
        name = device.get("name")
        total_memory_bytes = device.get("total_memory_bytes")
        if not isinstance(name, str) or "A100" not in name:
            errors.append(f"expected an A100 GPU, found {name!r}")
        if (
            not isinstance(total_memory_bytes, int)
            or isinstance(total_memory_bytes, bool)
            or total_memory_bytes < MINIMUM_TOTAL_MEMORY_BYTES
        ):
            errors.append(
                "expected at least 79 GiB GPU memory, "
                f"found {total_memory_bytes!r} bytes"
            )
    return errors


def write_json_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite GPU preflight: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary GPU preflight exists: {temporary}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    import torch

    cuda_available = bool(torch.cuda.is_available())
    devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
    bf16_supported = bool(
        cuda_available and torch.cuda.is_bf16_supported()
    )
    errors = validate_observation(
        cuda_available=cuda_available,
        bf16_supported=bf16_supported,
        devices=devices,
    )
    report = {
        "schema_version": 1,
        "status": "complete" if not errors else "failed",
        "expected": {
            "visible_gpu_count": 1,
            "name_contains": "A100",
            "minimum_total_memory_bytes": MINIMUM_TOTAL_MEMORY_BYTES,
            "bf16_supported": True,
        },
        "observed": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "compiled_cuda": torch.version.cuda,
            "cuda_available": cuda_available,
            "bf16_supported": bf16_supported,
            "devices": devices,
        },
        "errors": errors,
    }
    write_json_once(args.output.resolve(), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        print("[ERROR] formal evaluation GPU preflight failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
