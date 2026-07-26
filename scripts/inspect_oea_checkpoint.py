#!/usr/bin/env python3
"""Safely inspect an OEA checkpoint without materializing tensor storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.oea_checkpoint_safety import resolve_posix_path_safe_globals


def peak_rss_kib() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-size", type=int)
    parser.add_argument("--expected-sha256")
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


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_mapping_summary(mapping: dict[str, Any], torch: Any) -> dict[str, Any]:
    tensor_items = [(key, value) for key, value in mapping.items() if torch.is_tensor(value)]
    tensor_keys = [key for key, _ in tensor_items]
    lora_keys = [key for key in tensor_keys if "lora_" in key.lower()]
    lora_key_set = set(lora_keys)
    non_lora_keys = [key for key in tensor_keys if key not in lora_key_set]
    dtype_counts: dict[str, int] = {}
    dtype_bytes: dict[str, int] = {}
    lora_dtype_counts: dict[str, int] = {}
    lora_dtype_bytes: dict[str, int] = {}
    total_numel = 0
    total_bytes = 0
    lora_total_numel = 0
    lora_total_bytes = 0
    shapes: dict[str, list[int]] = {}
    for key, tensor in tensor_items:
        dtype = str(tensor.dtype)
        numel = int(tensor.numel())
        size_bytes = numel * int(tensor.element_size())
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        dtype_bytes[dtype] = dtype_bytes.get(dtype, 0) + size_bytes
        total_numel += numel
        total_bytes += size_bytes
        if key in lora_key_set:
            lora_dtype_counts[dtype] = lora_dtype_counts.get(dtype, 0) + 1
            lora_dtype_bytes[dtype] = lora_dtype_bytes.get(dtype, 0) + size_bytes
            lora_total_numel += numel
            lora_total_bytes += size_bytes
        shapes[key] = list(tensor.shape)

    return {
        "mapping_entries": len(mapping),
        "tensor_count": len(tensor_items),
        "non_tensor_count": len(mapping) - len(tensor_items),
        "total_numel": total_numel,
        "estimated_tensor_bytes": total_bytes,
        "dtype_tensor_counts": dtype_counts,
        "dtype_estimated_bytes": dtype_bytes,
        "lora_tensor_count": len(lora_keys),
        "lora_total_numel": lora_total_numel,
        "lora_estimated_tensor_bytes": lora_total_bytes,
        "lora_dtype_tensor_counts": lora_dtype_counts,
        "lora_dtype_estimated_bytes": lora_dtype_bytes,
        "non_lora_tensor_count": len(non_lora_keys),
        "lora_tensor_keys": lora_keys,
        "non_lora_tensor_keys": non_lora_keys,
        "tensor_shapes": shapes,
        "non_tensor_keys": [key for key in mapping if key not in tensor_keys],
    }


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return {"type": type(value).__name__, "repr": repr(value)}


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output("status", "--short"),
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "expected_size_bytes": args.expected_size,
        "expected_sha256": args.expected_sha256,
        "load_policy": {
            "weights_only": True,
            "fake_tensor_mode": True,
            "map_location": None,
            "mmap": True,
        },
    }
    write_json(output, report)

    try:
        import torch

        report["torch_version"] = torch.__version__
        if args.expected_size is not None and checkpoint.stat().st_size != args.expected_size:
            raise RuntimeError(
                f"checkpoint size mismatch: {checkpoint.stat().st_size} != {args.expected_size}"
            )

        hash_started = time.monotonic()
        digest = sha256_file(checkpoint)
        report["checkpoint_sha256"] = digest
        report["sha256_elapsed_seconds"] = time.monotonic() - hash_started
        if args.expected_sha256 and digest.lower() != args.expected_sha256.lower():
            raise RuntimeError(f"checkpoint SHA256 mismatch: {digest}")
        write_json(output, report)

        unsafe_globals_fn = getattr(
            getattr(torch, "serialization", None),
            "get_unsafe_globals_in_checkpoint",
            None,
        )
        if unsafe_globals_fn is None:
            raise RuntimeError(
                "torch.serialization.get_unsafe_globals_in_checkpoint is unavailable"
            )
        unsafe_globals = sorted(unsafe_globals_fn(checkpoint))
        report["unsafe_globals"] = unsafe_globals
        (
            approved_globals,
            unexpected_globals,
            safe_global_objects,
        ) = resolve_posix_path_safe_globals(unsafe_globals)
        report["approved_safe_globals"] = approved_globals
        report["unexpected_unsafe_globals"] = unexpected_globals
        if unexpected_globals:
            raise RuntimeError(
                f"checkpoint contains unapproved globals: {unexpected_globals}"
            )
        write_json(output, report)

        from torch._subclasses.fake_tensor import FakeTensorMode

        load_started = time.monotonic()
        with (
            torch.serialization.safe_globals(safe_global_objects),
            FakeTensorMode(),
        ):
            state = torch.load(
                checkpoint,
                mmap=True,
                weights_only=True,
            )
        report["load_elapsed_seconds"] = time.monotonic() - load_started
        if not isinstance(state, dict):
            raise TypeError(f"checkpoint top level is {type(state).__name__}, expected dict")

        report["top_level_keys"] = list(state.keys())
        report["sections"] = {}
        for key, value in state.items():
            if isinstance(value, dict):
                summary = tensor_mapping_summary(value, torch)
                if summary["tensor_count"] == 0:
                    summary["value"] = json_safe(value)
                report["sections"][key] = summary
            else:
                report["sections"][key] = {
                    "type": type(value).__name__,
                    "value": json_safe(value),
                }

        required = {"lora_state_dict", "audio_head", "text_head"}
        report["missing_required_sections"] = sorted(required - set(state))
        if report["missing_required_sections"]:
            raise RuntimeError(
                f"checkpoint sections missing: {report['missing_required_sections']}"
            )

        report["peak_rss_kib"] = peak_rss_kib()
        report["status"] = "complete"
        report["finished_at"] = utc_now()
        write_json(output, report)
        return 0
    except Exception as exc:  # noqa: BLE001 - preserve exact inspection failure
        report["status"] = "failed"
        report["finished_at"] = utc_now()
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        report["peak_rss_kib"] = peak_rss_kib()
        write_json(output, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
