#!/usr/bin/env python3
"""Run the CPU-only vanilla resource audit and portable base-lock build."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.audit_vanilla_backbone_resources import resolve_backbone_audit_plan
from scripts.prepare_official_oea_checkpoint import (
    atomic_write_json,
    file_identity,
    utc_now,
)
from scripts.vanilla_backbone_registry import DEFAULT_REGISTRY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lock-output", type=Path, required=True)
    return parser.parse_args()


def git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def validate_paths(backbone_id: str, output_dir: Path, lock_output: Path) -> None:
    logs_root = (REPOSITORY_ROOT / "logs").resolve()
    if not output_dir.is_relative_to(logs_root) or output_dir == logs_root:
        raise ValueError("pipeline output directory must be a child of repository logs/")
    expected = (
        REPOSITORY_ROOT / "results/model_locks" / f"{backbone_id}.json"
    ).resolve()
    if lock_output != expected:
        raise ValueError(f"lock output must use the canonical backbone path: {expected}")


def run_stage(
    *, name: str, command: Sequence[str], report: dict[str, Any], report_path: Path
) -> None:
    stage = {
        "status": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "command": list(command),
        "exit_code": None,
    }
    report["stages"][name] = stage
    atomic_write_json(report_path, report)
    completed = subprocess.run(list(command), cwd=REPOSITORY_ROOT, check=False)
    stage.update(
        {
            "status": "complete" if completed.returncode == 0 else "failed",
            "finished_at": utc_now(),
            "exit_code": completed.returncode,
        }
    )
    atomic_write_json(report_path, report)
    if completed.returncode != 0:
        raise RuntimeError(f"pipeline stage {name} exited with {completed.returncode}")


def main() -> int:
    args = parse_args()
    registry = args.registry.resolve()
    model_root = args.model_root.expanduser().resolve()
    output_dir = args.output_dir.resolve()
    lock_output = args.lock_output.resolve()
    plan = resolve_backbone_audit_plan(args.backbone, registry)
    validate_paths(args.backbone, output_dir, lock_output)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse pipeline output: {output_dir}")
    if lock_output.exists():
        raise FileExistsError(f"refusing to overwrite model lock: {lock_output}")
    git_commit = git_output("rev-parse", "HEAD")
    git_status = git_output("status", "--short")
    if git_status:
        raise RuntimeError("vanilla model pipeline requires a clean Git worktree")

    output_dir.mkdir(parents=True)
    report_path = output_dir / "pipeline_manifest.json"
    audit_path = output_dir / "model_resource_audit.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "git_commit": git_commit,
        "git_status_short": git_status,
        "backbone_id": args.backbone,
        "backbone_plan": plan,
        "model_root": str(model_root),
        "lock_output": lock_output.relative_to(REPOSITORY_ROOT).as_posix(),
        "stages": {},
    }
    atomic_write_json(report_path, report)
    try:
        run_stage(
            name="resource_audit",
            command=[
                sys.executable,
                str(REPOSITORY_ROOT / "scripts/audit_vanilla_backbone_resources.py"),
                "--registry",
                str(registry),
                "--backbone",
                args.backbone,
                "--model-root",
                str(model_root),
                "--output",
                str(audit_path),
            ],
            report=report,
            report_path=report_path,
        )
        run_stage(
            name="model_lock",
            command=[
                sys.executable,
                str(REPOSITORY_ROOT / "scripts/build_vanilla_backbone_lock.py"),
                "--registry",
                str(registry),
                "--backbone",
                args.backbone,
                "--model-resource-audit",
                str(audit_path),
                "--output",
                str(lock_output),
            ],
            report=report,
            report_path=report_path,
        )
        report["model_lock_identity"] = file_identity(
            lock_output, lock_output.relative_to(REPOSITORY_ROOT).as_posix()
        )
        report["status"] = "complete"
        report["finished_at"] = utc_now()
        atomic_write_json(report_path, report)
        return 0
    except Exception as error:  # noqa: BLE001 - preserve failure evidence
        report["status"] = "failed"
        report["finished_at"] = utc_now()
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        atomic_write_json(report_path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
