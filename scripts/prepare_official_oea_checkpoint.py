#!/usr/bin/env python3
"""Inspect and safely derive inference-only weights for one official OEA variant."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = (
    REPOSITORY_ROOT / "configs/checkpoints/official_oea_checkpoints.json"
)
EXPECTED_VARIANT_IDS = (
    "oea_nemo3b",
    "oea_nemo3b_cl",
    "oea_qwen3b",
    "oea_qwen3b_cl",
    "oea_qwen7b",
    "oea_qwen7b_cl",
)
IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, display_path: str | None = None) -> dict[str, Any]:
    return {
        "path": display_path if display_path is not None else str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def identifier(value: Any, label: str) -> str:
    result = nonempty_string(value, label)
    if IDENTIFIER_PATTERN.fullmatch(result) is None:
        raise ValueError(f"{label} is not a canonical identifier: {result}")
    return result


def positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def safe_repository_path(repository_root: Path, value: Any, label: str) -> Path:
    raw = nonempty_string(value, label)
    if "\\" in raw or ":" in raw:
        raise ValueError(f"{label} must use repository-relative POSIX separators")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{label} is not a safe repository-relative path: {raw}")
    resolved = (repository_root / Path(*relative.parts)).resolve()
    if not resolved.is_relative_to(repository_root.resolve()):
        raise ValueError(f"{label} escapes repository root: {raw}")
    return resolved


def safe_asset_path(value: Any, label: str) -> PurePosixPath:
    raw = nonempty_string(value, label)
    if "\\" in raw or ":" in raw:
        raise ValueError(f"{label} must use POSIX separators")
    result = PurePosixPath(raw)
    if result.is_absolute() or any(part in {"", ".", ".."} for part in result.parts):
        raise ValueError(f"{label} is not a safe relative asset path: {raw}")
    return result


def find_asset(
    reference: Mapping[str, Any], repository_root: Path, label: str
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    manifest_path = safe_repository_path(
        repository_root, reference.get("manifest"), f"{label}.manifest"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(
        manifest.get("assets"), list
    ):
        raise ValueError(f"{label}: invalid resource manifest")
    asset_name = identifier(reference.get("asset_name"), f"{label}.asset_name")
    matches = [
        asset
        for asset in manifest["assets"]
        if isinstance(asset, dict) and asset.get("name") == asset_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{label}: expected one asset named {asset_name}, found {len(matches)}"
        )
    return manifest, matches[0], manifest_path


def validate_asset(asset: Mapping[str, Any], label: str) -> dict[str, Any]:
    repo_id = nonempty_string(asset.get("repo_id"), f"{label}.repo_id")
    revision = nonempty_string(asset.get("revision"), f"{label}.revision")
    if COMMIT_PATTERN.fullmatch(revision) is None:
        raise ValueError(f"{label}.revision must be an immutable commit SHA")
    local_subdir = safe_asset_path(asset.get("local_subdir"), f"{label}.local_subdir")
    return {
        "name": identifier(asset.get("name"), f"{label}.name"),
        "repo_id": repo_id,
        "revision": revision,
        "local_subdir": local_subdir.as_posix(),
    }


def validate_checkpoint_asset(asset: Mapping[str, Any], label: str) -> dict[str, Any]:
    result = validate_asset(asset, label)
    primary = asset.get("expected_primary_file")
    if not isinstance(primary, dict):
        raise ValueError(f"{label}.expected_primary_file must be fixed")
    source_file = safe_asset_path(primary.get("path"), f"{label}.primary.path")
    if source_file.suffix != ".pt":
        raise ValueError(f"{label}.primary.path must be a .pt checkpoint")
    source_size = positive_integer(primary.get("size_bytes"), f"{label}.primary.size")
    source_sha256 = nonempty_string(
        primary.get("lfs_sha256"), f"{label}.primary.lfs_sha256"
    )
    if SHA256_PATTERN.fullmatch(source_sha256) is None:
        raise ValueError(f"{label}.primary.lfs_sha256 must be lowercase SHA256")
    required_files = asset.get("required_files")
    if not isinstance(required_files, list) or source_file.as_posix() not in required_files:
        raise ValueError(f"{label}: primary checkpoint must be a required file")
    result.update(
        {
            "source_file": source_file.as_posix(),
            "source_size_bytes": source_size,
            "source_sha256": source_sha256,
            "derived_file": (
                source_file.with_name(source_file.stem + "_inference_only.pt")
            ).as_posix(),
        }
    )
    return result


def manifest_display_path(path: Path, repository_root: Path) -> str:
    return path.resolve().relative_to(repository_root.resolve()).as_posix()


def load_checkpoint_registry(
    path: Path, repository_root: Path = REPOSITORY_ROOT
) -> dict[str, dict[str, Any]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported official checkpoint registry schema_version")
    if tuple(config.get("expected_variant_ids", [])) != EXPECTED_VARIANT_IDS:
        raise ValueError("expected_variant_ids must be the fixed six-variant order")
    source_tags = config.get("source_tags")
    if not isinstance(source_tags, dict) or any(
        not isinstance(value, str) or not value.startswith("[")
        for value in source_tags.values()
    ):
        raise ValueError("checkpoint registry source_tags must be explicit")
    variants = config.get("variants")
    if not isinstance(variants, list):
        raise ValueError("checkpoint registry variants must be a list")

    output: dict[str, dict[str, Any]] = {}
    checkpoint_paths: set[str] = set()
    for index, row in enumerate(variants):
        label = f"variants[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{label} must be an object")
        variant_id = identifier(row.get("variant_id"), f"{label}.variant_id")
        if variant_id in output:
            raise ValueError(f"duplicate variant_id: {variant_id}")
        paper_model = nonempty_string(row.get("paper_model"), f"{label}.paper_model")
        training_variant = nonempty_string(
            row.get("training_variant"), f"{label}.training_variant"
        )
        if not isinstance(row.get("base_asset"), dict) or not isinstance(
            row.get("checkpoint_asset"), dict
        ):
            raise ValueError(f"{label} asset references must be objects")
        _, base_asset, base_manifest_path = find_asset(
            row["base_asset"], repository_root, f"{label}.base_asset"
        )
        _, checkpoint_asset, checkpoint_manifest_path = find_asset(
            row["checkpoint_asset"], repository_root, f"{label}.checkpoint_asset"
        )
        base = validate_asset(base_asset, f"{label}.base_asset.resolved")
        checkpoint = validate_checkpoint_asset(
            checkpoint_asset, f"{label}.checkpoint_asset.resolved"
        )
        checkpoint_local_path = (
            PurePosixPath(checkpoint["local_subdir"]) / checkpoint["source_file"]
        ).as_posix()
        if checkpoint_local_path in checkpoint_paths:
            raise ValueError(f"duplicate checkpoint local path: {checkpoint_local_path}")
        checkpoint_paths.add(checkpoint_local_path)
        output[variant_id] = {
            "variant_id": variant_id,
            "paper_model": paper_model,
            "training_variant": training_variant,
            "base_asset": base,
            "base_manifest": manifest_display_path(base_manifest_path, repository_root),
            "checkpoint_asset": checkpoint,
            "checkpoint_manifest": manifest_display_path(
                checkpoint_manifest_path, repository_root
            ),
        }
    if tuple(output) != EXPECTED_VARIANT_IDS:
        raise ValueError("variants must match expected_variant_ids exactly and in order")
    return output


def external_model_root(path: Path, repository_root: Path = REPOSITORY_ROOT) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_relative_to(repository_root.resolve()):
        raise ValueError("model root must remain outside the Git repository")
    return resolved


def variant_paths(variant: Mapping[str, Any], model_root: Path) -> tuple[Path, Path]:
    asset = variant["checkpoint_asset"]
    source_relative = PurePosixPath(asset["local_subdir"]) / asset["source_file"]
    derived_relative = PurePosixPath(asset["local_subdir"]) / asset["derived_file"]
    source = (model_root / Path(*source_relative.parts)).resolve()
    destination = (model_root / Path(*derived_relative.parts)).resolve()
    if not source.is_relative_to(model_root) or not destination.is_relative_to(model_root):
        raise ValueError("resolved checkpoint path escapes model root")
    return source, destination


def structure_expectations(
    inspection: Mapping[str, Any], variant: Mapping[str, Any]
) -> dict[str, int]:
    asset = variant["checkpoint_asset"]
    if inspection.get("status") != "complete":
        raise ValueError("checkpoint inspection is not complete")
    if inspection.get("checkpoint_size_bytes") != asset["source_size_bytes"]:
        raise ValueError("inspection checkpoint size differs from fixed registry")
    if inspection.get("checkpoint_sha256") != asset["source_sha256"]:
        raise ValueError("inspection checkpoint SHA256 differs from fixed registry")
    sections = inspection.get("sections")
    if not isinstance(sections, dict):
        raise ValueError("inspection sections are missing")
    lora = sections.get("lora_state_dict")
    audio_head = sections.get("audio_head")
    text_head = sections.get("text_head")
    if not all(isinstance(section, dict) for section in (lora, audio_head, text_head)):
        raise ValueError("inspection required tensor sections are missing")
    lora_count = positive_integer(lora.get("lora_tensor_count"), "LoRA tensor count")
    lora_bytes = positive_integer(
        lora.get("lora_estimated_tensor_bytes"), "LoRA tensor bytes"
    )
    audio_shapes = audio_head.get("tensor_shapes")
    text_shapes = text_head.get("tensor_shapes")
    if not isinstance(audio_shapes, dict) or not audio_shapes:
        raise ValueError("audio projection head tensor shapes are missing")
    if audio_shapes != text_shapes:
        raise ValueError("audio/text projection head tensor shapes differ")
    return {
        "expected_lora_tensors": lora_count,
        "expected_lora_bytes": lora_bytes,
    }


def inspection_command(
    python: str, source: Path, output: Path, variant: Mapping[str, Any]
) -> list[str]:
    asset = variant["checkpoint_asset"]
    return [
        python,
        str(REPOSITORY_ROOT / "scripts/inspect_oea_checkpoint.py"),
        "--checkpoint",
        str(source),
        "--output",
        str(output),
        "--expected-size",
        str(asset["source_size_bytes"]),
        "--expected-sha256",
        str(asset["source_sha256"]),
    ]


def extraction_command(
    python: str,
    source: Path,
    destination: Path,
    output: Path,
    variant: Mapping[str, Any],
    expectations: Mapping[str, int],
) -> list[str]:
    asset = variant["checkpoint_asset"]
    return [
        python,
        str(REPOSITORY_ROOT / "scripts/extract_oea_inference_checkpoint.py"),
        "--source",
        str(source),
        "--destination",
        str(destination),
        "--audit-output",
        str(output),
        "--expected-source-size",
        str(asset["source_size_bytes"]),
        "--expected-source-sha256",
        str(asset["source_sha256"]),
        "--expected-lora-tensors",
        str(expectations["expected_lora_tensors"]),
        "--expected-lora-bytes",
        str(expectations["expected_lora_bytes"]),
    ]


def run_command(arguments: Sequence[str]) -> int:
    completed = subprocess.run(list(arguments), cwd=REPOSITORY_ROOT, check=False)
    return completed.returncode


def git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def main() -> int:
    args = parse_args()
    registry_path = args.registry.resolve()
    output_dir = args.output_dir.resolve()
    report_path = output_dir / "preparation_manifest.json"
    inspection_path = output_dir / "checkpoint_inspection.json"
    extraction_path = output_dir / "extraction_manifest.json"
    if any(path.exists() for path in (report_path, inspection_path, extraction_path)):
        raise FileExistsError("refusing to overwrite checkpoint preparation artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "operation": "inspect_only" if args.inspect_only else "inspect_and_extract",
        "started_at": utc_now(),
        "finished_at": None,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output("status", "--short"),
        "variant_id": args.variant,
    }
    atomic_write_json(report_path, report)

    try:
        registry = load_checkpoint_registry(registry_path)
        if args.variant not in registry:
            raise ValueError(
                f"unknown variant {args.variant}; expected one of {list(registry)}"
            )
        variant = registry[args.variant]
        model_root = external_model_root(args.model_root)
        source, destination = variant_paths(variant, model_root)
        report.update(
            {
                "registry": file_identity(
                    registry_path,
                    registry_path.relative_to(REPOSITORY_ROOT).as_posix(),
                ),
                "variant": variant,
                "source_checkpoint": str(source),
                "derived_checkpoint": None if args.inspect_only else str(destination),
            }
        )
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stat().st_size != variant["checkpoint_asset"]["source_size_bytes"]:
            raise RuntimeError("source checkpoint size differs from fixed registry")
        if not args.inspect_only and destination.exists():
            raise FileExistsError(
                f"refusing to overwrite existing derived checkpoint: {destination}"
            )
        atomic_write_json(report_path, report)

        inspect_arguments = inspection_command(
            sys.executable, source, inspection_path, variant
        )
        report["inspection_command"] = inspect_arguments
        atomic_write_json(report_path, report)
        inspection_exit_code = run_command(inspect_arguments)
        report["inspection_exit_code"] = inspection_exit_code
        if inspection_exit_code != 0:
            raise RuntimeError(
                f"checkpoint inspection exited with code {inspection_exit_code}"
            )
        inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
        expectations = structure_expectations(inspection, variant)
        report["measured_structure_expectations"] = expectations
        if args.inspect_only:
            report["status"] = "complete"
            report["finished_at"] = utc_now()
            atomic_write_json(report_path, report)
            return 0

        extract_arguments = extraction_command(
            sys.executable,
            source,
            destination,
            extraction_path,
            variant,
            expectations,
        )
        report["extraction_command"] = extract_arguments
        atomic_write_json(report_path, report)
        extraction_exit_code = run_command(extract_arguments)
        report["extraction_exit_code"] = extraction_exit_code
        if extraction_exit_code != 0:
            raise RuntimeError(
                f"checkpoint extraction exited with code {extraction_exit_code}"
            )
        extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
        if extraction.get("status") != "complete":
            raise RuntimeError("checkpoint extraction report is not complete")
        if extraction.get("source_sha256") != variant["checkpoint_asset"]["source_sha256"]:
            raise RuntimeError("extraction source SHA256 differs from fixed registry")
        if not destination.is_file():
            raise RuntimeError("derived checkpoint is missing after extraction")
        report["derived_checkpoint_identity"] = {
            "path": str(destination),
            "size_bytes": extraction.get("destination_size_bytes"),
            "sha256": extraction.get("destination_sha256"),
        }
        report["status"] = "complete"
        report["finished_at"] = utc_now()
        atomic_write_json(report_path, report)
        return 0
    except Exception as error:  # noqa: BLE001 - preserve auditable failure
        report["status"] = "failed"
        report["finished_at"] = utc_now()
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        atomic_write_json(report_path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
