#!/usr/bin/env python3
"""Plan or execute one explicitly registered reproduction stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPOSITORY_ROOT / "configs/reproduction/stages.json"
ALLOWED_STATUSES = {
    "TODO",
    "IN_PROGRESS",
    "WAITING_USER",
    "RUNNING_REMOTE",
    "COMPLETED",
    "FAILED",
    "BLOCKED",
}
ALLOWED_KINDS = {"executable", "group", "blocked"}
ALLOWED_RESOURCES = {
    "CPU",
    "1xA100-80GB",
    "1xBF16-GPU",
    "CPU -> 1xA100-80GB -> CPU",
    "CPU -> 1xBF16-GPU -> CPU",
    "1xA100-80GB -> CPU",
    "1xBF16-GPU -> CPU",
    "GPU count not yet fixed",
    "CPU/GPU not yet fixed",
    "multiple CPU/GPU server handoffs",
}
OFFICIAL_VARIANTS = (
    "oea_nemo3b",
    "oea_nemo3b_cl",
    "oea_qwen3b",
    "oea_qwen3b_cl",
    "oea_qwen7b",
    "oea_qwen7b_cl",
)
VANILLA_BACKBONES = (
    "vanilla_nemotron_3b",
    "vanilla_qwen2_5_omni_3b",
    "vanilla_qwen2_5_omni_7b",
)
FORBIDDEN_COMMAND_FRAGMENTS = (
    "rm -rf",
    "git reset --hard",
    "git clean -fd",
    "git push --force",
)
REQUIRED_PUBLIC_STAGE_IDS = (
    "official_eval",
    "train_qwen3b",
    "uiq_eval",
    "baselines",
    "all",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-long-operation", action="store_true")
    parser.add_argument(
        "--variant", choices=OFFICIAL_VARIANTS, default="oea_qwen3b_cl"
    )
    parser.add_argument(
        "--backbone", choices=VANILLA_BACKBONES, default="vanilla_nemotron_3b"
    )
    parser.add_argument("--embedding-dir", type=Path)
    parser.add_argument("--caption-embedding-dir", type=Path)
    parser.add_argument("--uiq-embedding-dir", type=Path)
    parser.add_argument("--smoke-metrics", type=Path)
    parser.add_argument("--verify-existing-derived", action="store_true")
    args = parser.parse_args()
    if not args.list and not args.stage:
        parser.error("--stage is required unless --list is used")
    if args.list and args.execute:
        parser.error("--list and --execute cannot be combined")
    return args


def nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def safe_repository_script(value: Any, label: str) -> str:
    text = nonempty_string(value, label)
    if "\\" in text:
        raise ValueError(f"{label} must use POSIX separators")
    parts = text.split("/")
    if any(part in {"", ".", ".."} or ":" in part for part in parts):
        raise ValueError(f"{label} must be a portable relative path")
    path = PurePosixPath(text)
    if path.is_absolute() or path.suffix != ".sh":
        raise ValueError(f"{label} must name a relative shell script")
    resolved = (REPOSITORY_ROOT / Path(*path.parts)).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes the repository") from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return path.as_posix()


def load_stage_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("stage registry has unsupported schema_version")
    stages = document.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("stage registry must contain a non-empty stages list")
    registry: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(stages):
        label = f"stages[{index}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{label} is not an object")
        stage = dict(raw)
        stage_id = nonempty_string(stage.get("stage_id"), f"{label}.stage_id")
        if stage_id in registry:
            raise ValueError(f"duplicate stage_id: {stage_id}")
        kind = stage.get("kind")
        status = stage.get("status")
        if kind not in ALLOWED_KINDS:
            raise ValueError(f"{stage_id}: invalid kind")
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"{stage_id}: invalid status")
        if stage.get("resource") not in ALLOWED_RESOURCES:
            raise ValueError(f"{stage_id}: invalid resource")
        nonempty_string(stage.get("duration"), f"{stage_id}.duration")
        nonempty_string(stage.get("description"), f"{stage_id}.description")
        if not isinstance(stage.get("long_operation"), bool):
            raise ValueError(f"{stage_id}.long_operation must be boolean")
        if status == "BLOCKED":
            nonempty_string(
                stage.get("blocked_reason"), f"{stage_id}.blocked_reason"
            )
        if kind == "executable":
            command = stage.get("command")
            required = stage.get("required_arguments")
            if (
                not isinstance(command, list)
                or len(command) < 2
                or any(not isinstance(token, str) or not token for token in command)
            ):
                raise ValueError(f"{stage_id}: invalid command")
            if command[0] != "bash":
                raise ValueError(f"{stage_id}: executable must call a bash wrapper")
            command[1] = safe_repository_script(command[1], f"{stage_id}.command[1]")
            joined = " ".join(command).lower()
            if any(fragment in joined for fragment in FORBIDDEN_COMMAND_FRAGMENTS):
                raise ValueError(f"{stage_id}: destructive command is forbidden")
            if not isinstance(required, list) or any(
                argument not in {
                    "variant",
                    "backbone",
                    "embedding_dir",
                    "caption_embedding_dir",
                    "uiq_embedding_dir",
                    "smoke_metrics",
                }
                for argument in required
            ):
                raise ValueError(f"{stage_id}: invalid required_arguments")
            placeholders = {
                token[1:-1]
                for token in command
                if token.startswith("{") and token.endswith("}")
            }
            if placeholders != set(required):
                raise ValueError(
                    f"{stage_id}: command placeholders differ from required_arguments"
                )
        elif kind == "group":
            children = stage.get("children")
            if not isinstance(children, list) or not children:
                raise ValueError(f"{stage_id}: group must have children")
            if len(children) != len(set(children)):
                raise ValueError(f"{stage_id}: group children must be unique")
        else:
            if status != "BLOCKED":
                raise ValueError(f"{stage_id}: blocked kind must have BLOCKED status")
        registry[stage_id] = stage

    required_public = document.get("required_public_stage_ids")
    if required_public != list(REQUIRED_PUBLIC_STAGE_IDS):
        raise ValueError(
            "required_public_stage_ids must contain exactly the supported public stages"
        )
    if any(stage_id not in registry for stage_id in required_public):
        raise ValueError("required_public_stage_ids reference unknown stages")
    for stage in registry.values():
        if stage["kind"] == "group":
            missing = [child for child in stage["children"] if child not in registry]
            if missing:
                raise ValueError(
                    f"{stage['stage_id']}: unknown group children: {missing}"
                )
    validate_group_cycles(registry)
    return registry


def validate_group_cycles(registry: Mapping[str, Mapping[str, Any]]) -> None:
    active: set[str] = set()
    complete: set[str] = set()

    def visit(stage_id: str) -> None:
        if stage_id in complete:
            return
        if stage_id in active:
            raise ValueError(f"stage group cycle detected at {stage_id}")
        stage = registry[stage_id]
        if stage["kind"] != "group":
            complete.add(stage_id)
            return
        active.add(stage_id)
        for child in stage["children"]:
            visit(child)
        active.remove(stage_id)
        complete.add(stage_id)

    for stage_id in registry:
        visit(stage_id)


def argument_values(args: argparse.Namespace) -> dict[str, str | None]:
    return {
        "variant": args.variant,
        "backbone": args.backbone,
        "embedding_dir": (
            str(args.embedding_dir.resolve()) if args.embedding_dir else None
        ),
        "caption_embedding_dir": (
            str(args.caption_embedding_dir.resolve())
            if args.caption_embedding_dir
            else None
        ),
        "uiq_embedding_dir": (
            str(args.uiq_embedding_dir.resolve()) if args.uiq_embedding_dir else None
        ),
        "smoke_metrics": (
            str(args.smoke_metrics.resolve()) if args.smoke_metrics else None
        ),
    }


def render_command(
    stage: Mapping[str, Any],
    values: Mapping[str, str | None],
    *,
    allow_missing: bool,
    verify_existing_derived: bool = False,
) -> list[str]:
    if stage["kind"] != "executable":
        raise ValueError("only executable stages have commands")
    command: list[str] = []
    for token in stage["command"]:
        if token.startswith("{") and token.endswith("}"):
            argument = token[1:-1]
            value = values.get(argument)
            if value is None:
                if not allow_missing:
                    raise ValueError(
                        f"{stage['stage_id']} requires --{argument.replace('_', '-')}"
                    )
                value = f"<{argument}>"
            command.append(value)
        else:
            command.append(token)
    if stage["stage_id"] == "official_model_lock" and verify_existing_derived:
        command.append("--verify-existing-derived")
    return command


def flattened_plan(
    stage_id: str, registry: Mapping[str, Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    if stage_id not in registry:
        raise KeyError(f"unknown stage: {stage_id}")
    output: list[Mapping[str, Any]] = []
    seen: set[str] = set()

    def visit(current: str) -> None:
        stage = registry[current]
        if stage["kind"] == "group":
            for child in stage["children"]:
                visit(child)
            return
        if current not in seen:
            output.append(stage)
            seen.add(current)

    visit(stage_id)
    return output


def plan_payload(
    selected: str,
    registry: Mapping[str, Mapping[str, Any]],
    values: Mapping[str, str | None],
    *,
    verify_existing_derived: bool,
) -> dict[str, Any]:
    root = registry[selected]
    steps = []
    for stage in flattened_plan(selected, registry):
        row = {
            key: stage[key]
            for key in (
                "stage_id",
                "kind",
                "status",
                "resource",
                "duration",
                "long_operation",
                "description",
            )
        }
        if stage["kind"] == "executable":
            command = render_command(
                stage,
                values,
                allow_missing=True,
                verify_existing_derived=verify_existing_derived,
            )
            row["command"] = command
            row["shell_command"] = shlex.join(command)
        if stage.get("blocked_reason"):
            row["blocked_reason"] = stage["blocked_reason"]
        steps.append(row)
    payload = {
        "schema_version": 1,
        "selected_stage": selected,
        "selected_kind": root["kind"],
        "selected_status": root["status"],
        "selected_description": root["description"],
        "execution_mode": (
            "plan_only" if root["kind"] != "executable" else "explicit"
        ),
        "steps": steps,
    }
    if root.get("blocked_reason"):
        payload["selected_blocked_reason"] = root["blocked_reason"]
    return payload


def print_human_plan(payload: Mapping[str, Any]) -> None:
    print(
        f"Stage: {payload['selected_stage']} "
        f"({payload['selected_kind']}, {payload['selected_status']})"
    )
    print(str(payload["selected_description"]))
    if payload.get("selected_blocked_reason"):
        print(f"Blocked: {payload['selected_blocked_reason']}")
    for index, step in enumerate(payload["steps"], start=1):
        print(
            f"{index}. {step['stage_id']} | {step['status']} | "
            f"{step['resource']} | {step['duration']}"
        )
        print(f"   {step['description']}")
        if "shell_command" in step:
            print(f"   command: {step['shell_command']}")
        if "blocked_reason" in step:
            print(f"   blocked: {step['blocked_reason']}")


def git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def execute_stage(
    stage: Mapping[str, Any], args: argparse.Namespace
) -> int:
    if stage["kind"] != "executable":
        raise RuntimeError(
            "group and blocked stages are plan-only; execute their registered "
            "CPU/GPU child stages after each required handoff"
        )
    if stage["status"] == "BLOCKED":
        raise RuntimeError(
            f"stage {stage['stage_id']} is BLOCKED; update audited prerequisites first"
        )
    if stage["long_operation"] and not args.acknowledge_long_operation:
        raise RuntimeError(
            "long stage requires --acknowledge-long-operation after the command, "
            "GPU, memory, time, disk, and output report has been reviewed"
        )
    git_status = git_output("status", "--short")
    if git_status:
        raise RuntimeError(
            f"stage execution requires a clean Git worktree: {git_status!r}"
        )
    command = render_command(
        stage,
        argument_values(args),
        allow_missing=False,
        verify_existing_derived=args.verify_existing_derived,
    )
    print(f"Executing: {shlex.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
    return completed.returncode


def stage_listing(registry: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "stage_id": stage_id,
            "kind": stage["kind"],
            "status": stage["status"],
            "resource": stage["resource"],
            "duration": stage["duration"],
        }
        for stage_id, stage in registry.items()
    ]


def main() -> int:
    args = parse_args()
    registry = load_stage_registry(args.registry.resolve())
    if args.list:
        listing = stage_listing(registry)
        if args.json:
            print(json.dumps({"schema_version": 1, "stages": listing}, indent=2))
        else:
            for row in listing:
                print(
                    f"{row['stage_id']}: {row['kind']} | {row['status']} | "
                    f"{row['resource']} | {row['duration']}"
                )
        return 0
    assert args.stage is not None
    if args.stage not in registry:
        raise KeyError(
            f"unknown stage {args.stage}; use --list to inspect registered stages"
        )
    payload = plan_payload(
        args.stage,
        registry,
        argument_values(args),
        verify_existing_derived=args.verify_existing_derived,
    )
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print_human_plan(payload)
    if not args.execute:
        return 0
    return execute_stage(registry[args.stage], args)


if __name__ == "__main__":
    raise SystemExit(main())
