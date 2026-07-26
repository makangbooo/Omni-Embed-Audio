#!/usr/bin/env python3
"""Extract LoRA and projection weights from a redundant full OEA checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--expected-source-size", type=int, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-lora-tensors", type=int, required=True)
    parser.add_argument("--expected-lora-bytes", type=int, required=True)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help=(
            "Do not write a checkpoint. Require the destination to exist and "
            "compare it exactly with the selected source tensors and metadata."
        ),
    )
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


def is_lora_tensor_key(key: str) -> bool:
    return "lora_" in key.lower()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    raise TypeError(f"unsupported derived-checkpoint metadata type: {type(value).__name__}")


def tensor_stats(mapping: dict[str, Any], torch: Any) -> dict[str, Any]:
    tensors = {key: value for key, value in mapping.items() if torch.is_tensor(value)}
    return {
        "tensor_count": len(tensors),
        "total_numel": sum(int(tensor.numel()) for tensor in tensors.values()),
        "estimated_tensor_bytes": sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in tensors.values()
        ),
        "dtypes": sorted({str(tensor.dtype) for tensor in tensors.values()}),
        "keys": list(tensors),
        "shapes": {key: list(tensor.shape) for key, tensor in tensors.items()},
    }


def assert_tensor_mappings_equal(
    expected: dict[str, Any], actual: dict[str, Any], torch: Any, section: str
) -> None:
    if set(expected) != set(actual):
        raise RuntimeError(f"{section} keys differ after extraction")
    for key in expected:
        if not torch.equal(expected[key], actual[key]):
            raise RuntimeError(f"{section}/{key} differs after extraction")


def assert_derived_state_equal(
    expected: dict[str, Any], actual: Any, torch: Any
) -> None:
    if not isinstance(actual, dict):
        raise TypeError("derived checkpoint top level is not a dict")
    if set(actual) != set(expected):
        raise RuntimeError("derived checkpoint top-level keys differ")
    for section in ("lora_state_dict", "audio_head", "text_head"):
        if not isinstance(actual.get(section), dict):
            raise TypeError(f"derived {section} is not a dict")
        assert_tensor_mappings_equal(
            expected[section], actual[section], torch, section
        )
    for field in (
        "schema_version",
        "source_checkpoint_sha256",
        "config",
        "metrics",
        "global_step",
    ):
        if actual.get(field) != expected[field]:
            raise RuntimeError(f"derived checkpoint {field} differs from source")


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    audit_output = args.audit_output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if args.verify_existing and not destination.is_file():
        raise FileNotFoundError(
            f"existing derived checkpoint is missing: {destination}"
        )
    if not args.verify_existing and destination.exists():
        raise FileExistsError(
            f"refusing to overwrite existing derived checkpoint: {destination}"
        )
    temporary_destination: Path | None = None
    if not args.verify_existing:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_destination = destination.with_name(
            f".{destination.name}.tmp.{os.getpid()}"
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output("status", "--short"),
        "source": str(source),
        "destination": str(destination),
        "operation": (
            "verify_existing_derived" if args.verify_existing else "extract"
        ),
        "source_size_bytes": source.stat().st_size,
        "expected_source_size_bytes": args.expected_source_size,
        "expected_source_sha256": args.expected_source_sha256,
        "load_policy": {
            "weights_only": True,
            "mmap": True,
            "map_location": "cpu",
        },
    }
    write_json(audit_output, report)

    try:
        import resource
        import torch

        report["torch_version"] = torch.__version__
        if source.stat().st_size != args.expected_source_size:
            raise RuntimeError("source checkpoint size mismatch")
        hash_started = time.monotonic()
        source_sha256 = sha256_file(source)
        report["source_sha256"] = source_sha256
        report["source_sha256_elapsed_seconds"] = time.monotonic() - hash_started
        if source_sha256.lower() != args.expected_source_sha256.lower():
            raise RuntimeError("source checkpoint SHA256 mismatch")
        write_json(audit_output, report)

        unsafe_globals = sorted(
            torch.serialization.get_unsafe_globals_in_checkpoint(source)
        )
        report["source_unsafe_globals"] = unsafe_globals
        (
            approved_globals,
            unexpected_globals,
            safe_global_objects,
        ) = resolve_posix_path_safe_globals(unsafe_globals)
        report["source_approved_safe_globals"] = approved_globals
        report["source_unexpected_unsafe_globals"] = unexpected_globals
        if unexpected_globals:
            raise RuntimeError(f"unexpected source globals: {unexpected_globals}")

        load_started = time.monotonic()
        with torch.serialization.safe_globals(safe_global_objects):
            state = torch.load(
                source,
                mmap=True,
                map_location="cpu",
                weights_only=True,
            )
        report["source_load_elapsed_seconds"] = time.monotonic() - load_started
        if not isinstance(state, dict):
            raise TypeError("source checkpoint top level is not a dict")
        required = {"lora_state_dict", "audio_head", "text_head"}
        missing = sorted(required - set(state))
        if missing:
            raise RuntimeError(f"source checkpoint sections missing: {missing}")

        source_lora_state = state["lora_state_dict"]
        lora_state = {
            key: tensor
            for key, tensor in source_lora_state.items()
            if is_lora_tensor_key(key)
        }
        lora_stats = tensor_stats(lora_state, torch)
        excluded_stats = tensor_stats(
            {
                key: tensor
                for key, tensor in source_lora_state.items()
                if not is_lora_tensor_key(key)
            },
            torch,
        )
        audio_head_stats = tensor_stats(state["audio_head"], torch)
        text_head_stats = tensor_stats(state["text_head"], torch)
        report["selected_lora"] = lora_stats
        report["excluded_frozen_backbone"] = excluded_stats
        report["audio_head"] = audio_head_stats
        report["text_head"] = text_head_stats

        if lora_stats["tensor_count"] != args.expected_lora_tensors:
            raise RuntimeError("LoRA tensor count mismatch")
        if lora_stats["estimated_tensor_bytes"] != args.expected_lora_bytes:
            raise RuntimeError("LoRA tensor byte count mismatch")
        if audio_head_stats["shapes"] != text_head_stats["shapes"]:
            raise RuntimeError("audio/text projection head shapes differ")

        derived_state = {
            "schema_version": 1,
            "source_checkpoint_sha256": source_sha256,
            "lora_state_dict": lora_state,
            "audio_head": state["audio_head"],
            "text_head": state["text_head"],
            "config": json_safe(state.get("config", {})),
            "metrics": json_safe(state.get("metrics", {})),
            "global_step": json_safe(state.get("global_step")),
        }
        if args.verify_existing:
            derived_path = destination
        else:
            if temporary_destination is None:
                raise AssertionError("temporary extraction path was not initialized")
            save_started = time.monotonic()
            torch.save(derived_state, temporary_destination)
            report["save_elapsed_seconds"] = time.monotonic() - save_started
            derived_path = temporary_destination

        derived_unsafe_globals = sorted(
            torch.serialization.get_unsafe_globals_in_checkpoint(
                derived_path
            )
        )
        report["derived_unsafe_globals"] = derived_unsafe_globals
        if derived_unsafe_globals:
            raise RuntimeError(
                f"derived checkpoint contains unsafe globals: {derived_unsafe_globals}"
            )
        loaded_derived = torch.load(
            derived_path,
            mmap=True,
            weights_only=True,
        )
        assert_derived_state_equal(derived_state, loaded_derived, torch)

        if temporary_destination is not None:
            temporary_destination.replace(destination)
        report["destination_size_bytes"] = destination.stat().st_size
        report["destination_sha256"] = sha256_file(destination)
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["status"] = "complete"
        report["finished_at"] = utc_now()
        write_json(audit_output, report)
        return 0
    except Exception as exc:  # noqa: BLE001 - retain exact extraction failure
        report["status"] = "failed"
        report["finished_at"] = utc_now()
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        write_json(audit_output, report)
        if temporary_destination is not None and temporary_destination.exists():
            temporary_destination.unlink()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
