#!/usr/bin/env python3
"""Resolve a committed evaluation protocol against a committed OEA model lock."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def nonnegative_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def sha256_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def safe_relative_path(value: Any, label: str) -> PurePosixPath:
    text = nonempty_string(value, label)
    if "\\" in text:
        raise ValueError(f"{label} must use POSIX separators")
    raw_parts = text.split("/")
    if any(
        part in {"", ".", ".."} or ":" in part for part in raw_parts
    ):
        raise ValueError(f"{label} must be a portable safe relative path")
    path = PurePosixPath(text)
    if path.is_absolute():
        raise ValueError(f"{label} must be a safe relative path")
    return path


def repository_file(
    value: Any, label: str, *, repository_root: Path = REPOSITORY_ROOT
) -> Path:
    relative = safe_relative_path(value, label)
    root = repository_root.resolve()
    resolved = (root / Path(*relative.parts)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes repository root") from exc
    return resolved


def portable_file_identity(path: Path, *, repository_root: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"input must be inside repository: {resolved}") from exc
    return {
        "repository_path": relative,
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def validate_portable_identity(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} identity is not an object")
    return {
        "repository_path": safe_relative_path(
            value.get("repository_path"), f"{label}.repository_path"
        ).as_posix(),
        "size_bytes": positive_integer(
            value.get("size_bytes"), f"{label}.size_bytes"
        ),
        "sha256": sha256_value(value.get("sha256"), f"{label}.sha256"),
    }


def validate_file_inventory(value: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be a non-empty object")
    result: dict[str, dict[str, Any]] = {}
    for raw_path, raw_identity in value.items():
        path = safe_relative_path(raw_path, f"{label} path").as_posix()
        if path in result:
            raise ValueError(f"{label} contains a duplicate path: {path}")
        if not isinstance(raw_identity, dict):
            raise ValueError(f"{label}.{path} is not an object")
        result[path] = {
            "size_bytes": nonnegative_integer(
                raw_identity.get("size_bytes"), f"{label}.{path}.size_bytes"
            ),
            "sha256": sha256_value(
                raw_identity.get("sha256"), f"{label}.{path}.sha256"
            ),
        }
    return dict(sorted(result.items()))


def validate_model_lock(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != 1 or value.get("status") != "locked":
        raise ValueError("model lock is not a locked schema-v1 artifact")
    variant_id = nonempty_string(value.get("variant_id"), "model lock variant_id")
    model = nonempty_string(value.get("model"), "model lock model")
    base = value.get("base_model")
    checkpoint = value.get("checkpoint")
    if not isinstance(base, dict) or not isinstance(checkpoint, dict):
        raise ValueError("model lock must contain base_model and checkpoint objects")
    base_model = {
        "repo_id": nonempty_string(base.get("repo_id"), "base_model.repo_id"),
        "revision": nonempty_string(base.get("revision"), "base_model.revision"),
        "local_subdir": safe_relative_path(
            base.get("local_subdir"), "base_model.local_subdir"
        ).as_posix(),
        "source": nonempty_string(base.get("source"), "base_model.source"),
        "files": validate_file_inventory(base.get("files"), "base_model.files"),
    }
    derived_checkpoint = {
        "repo_id": nonempty_string(
            checkpoint.get("repo_id"), "checkpoint.repo_id"
        ),
        "revision": nonempty_string(
            checkpoint.get("revision"), "checkpoint.revision"
        ),
        "local_subpath": safe_relative_path(
            checkpoint.get("local_subpath"), "checkpoint.local_subpath"
        ).as_posix(),
        "size_bytes": positive_integer(
            checkpoint.get("size_bytes"), "checkpoint.size_bytes"
        ),
        "sha256": sha256_value(checkpoint.get("sha256"), "checkpoint.sha256"),
        "source": nonempty_string(checkpoint.get("source"), "checkpoint.source"),
    }
    return {
        "variant_id": variant_id,
        "model": model,
        "base_model": base_model,
        "checkpoint": derived_checkpoint,
    }


def compare_protocol_to_lock(
    protocol: Mapping[str, Any], locked: Mapping[str, Any]
) -> None:
    if protocol.get("schema_version") != 1:
        raise ValueError("protocol config has unsupported schema_version")
    if protocol.get("official_variant_id") != locked["variant_id"]:
        raise ValueError("protocol official_variant_id differs from model lock")
    if protocol.get("model") != locked["model"]:
        raise ValueError("protocol model differs from model lock")
    protocol_base = protocol.get("base_model")
    protocol_checkpoint = protocol.get("checkpoint")
    if not isinstance(protocol_base, dict) or not isinstance(
        protocol_checkpoint, dict
    ):
        raise ValueError("protocol must contain base_model and checkpoint objects")
    for field in ("repo_id", "revision", "local_subdir"):
        if protocol_base.get(field) != locked["base_model"][field]:
            raise ValueError(f"protocol base_model.{field} differs from model lock")
    protocol_files = validate_file_inventory(
        protocol_base.get("files"), "protocol base_model.files"
    )
    locked_files = locked["base_model"]["files"]
    for path, identity in protocol_files.items():
        if locked_files.get(path) != identity:
            raise ValueError(
                f"protocol base_model file differs from model lock: {path}"
            )
    for field in ("repo_id", "revision", "local_subpath", "size_bytes", "sha256"):
        if protocol_checkpoint.get(field) != locked["checkpoint"][field]:
            raise ValueError(f"protocol checkpoint.{field} differs from model lock")


def build_resolved_config(
    protocol: Mapping[str, Any],
    model_lock: Mapping[str, Any],
    *,
    protocol_identity: Mapping[str, Any],
    model_lock_identity: Mapping[str, Any],
    git_commit: str,
) -> dict[str, Any]:
    locked = validate_model_lock(model_lock)
    compare_protocol_to_lock(protocol, locked)
    if (
        not isinstance(git_commit, str)
        or len(git_commit) != 40
        or any(character not in "0123456789abcdef" for character in git_commit)
    ):
        raise ValueError("resolution git_commit is invalid")
    protocol_identity = validate_portable_identity(
        protocol_identity, "protocol config"
    )
    model_lock_identity = validate_portable_identity(
        model_lock_identity, "official model lock"
    )
    resolved = copy.deepcopy(dict(protocol))
    resolved["base_model"] = copy.deepcopy(locked["base_model"])
    resolved["checkpoint"] = copy.deepcopy(locked["checkpoint"])
    resolved["official_model_lock"] = {
        "variant_id": locked["variant_id"],
        **model_lock_identity,
    }
    resolved["protocol_config"] = protocol_identity
    resolved["resolution_git_commit"] = git_commit
    return resolved


def verify_official_model_lock_binding(
    config: Mapping[str, Any], *, repository_root: Path = REPOSITORY_ROOT
) -> dict[str, Any]:
    binding = config.get("official_model_lock")
    protocol_identity = config.get("protocol_config")
    if not isinstance(binding, dict) or not isinstance(protocol_identity, dict):
        raise ValueError(
            "resolved config must contain official_model_lock and protocol_config"
        )
    lock_path = repository_file(
        binding.get("repository_path"),
        "official_model_lock.repository_path",
        repository_root=repository_root,
    )
    protocol_path = repository_file(
        protocol_identity.get("repository_path"),
        "protocol_config.repository_path",
        repository_root=repository_root,
    )
    for path, identity, label in (
        (lock_path, binding, "official model lock"),
        (protocol_path, protocol_identity, "protocol config"),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != positive_integer(
            identity.get("size_bytes"), f"{label} size_bytes"
        ):
            raise ValueError(f"{label} size differs from resolved config")
        if sha256_file(path) != sha256_value(
            identity.get("sha256"), f"{label} sha256"
        ):
            raise ValueError(f"{label} SHA256 differs from resolved config")
    model_lock = read_json_object(lock_path, "official model lock")
    protocol = read_json_object(protocol_path, "protocol config")
    locked = validate_model_lock(model_lock)
    compare_protocol_to_lock(protocol, locked)
    if binding.get("variant_id") != locked["variant_id"]:
        raise ValueError("resolved binding variant_id differs from model lock")
    if config.get("official_variant_id") != locked["variant_id"]:
        raise ValueError("resolved config variant differs from model lock")
    if config.get("model") != locked["model"]:
        raise ValueError("resolved config model differs from model lock")
    if config.get("base_model") != locked["base_model"]:
        raise ValueError("resolved config base_model differs from model lock")
    if config.get("checkpoint") != locked["checkpoint"]:
        raise ValueError("resolved config checkpoint differs from model lock")
    return {
        "variant_id": locked["variant_id"],
        "model": locked["model"],
        "model_lock": {
            "repository_path": binding["repository_path"],
            "size_bytes": lock_path.stat().st_size,
            "sha256": sha256_file(lock_path),
        },
        "protocol_config": {
            "repository_path": protocol_identity["repository_path"],
            "size_bytes": protocol_path.stat().st_size,
            "sha256": sha256_file(protocol_path),
        },
    }


def write_json_once_or_verify(path: Path, value: Mapping[str, Any]) -> str:
    content = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"existing resolved config differs: {path}")
        return "verified_existing"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)
    return "created"


def require_tracked_clean_input(path: Path, label: str) -> None:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the repository") from exc
    try:
        git_output("ls-files", "--error-unmatch", "--", relative)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{label} is not tracked by Git: {relative}") from exc


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol_config.resolve()
    model_lock_path = args.model_lock.resolve()
    output_path = args.output.resolve()
    if git_output("status", "--short"):
        raise RuntimeError("evaluation config resolution requires a clean Git worktree")
    require_tracked_clean_input(protocol_path, "protocol config")
    require_tracked_clean_input(model_lock_path, "official model lock")
    git_commit = git_output("rev-parse", "HEAD")
    protocol_identity = portable_file_identity(
        protocol_path, repository_root=REPOSITORY_ROOT
    )
    lock_identity = portable_file_identity(
        model_lock_path, repository_root=REPOSITORY_ROOT
    )
    resolved = build_resolved_config(
        read_json_object(protocol_path, "protocol config"),
        read_json_object(model_lock_path, "official model lock"),
        protocol_identity=protocol_identity,
        model_lock_identity=lock_identity,
        git_commit=git_commit,
    )
    operation = write_json_once_or_verify(output_path, resolved)
    print(
        json.dumps(
            {
                "status": "complete",
                "operation": operation,
                "variant_id": resolved["official_variant_id"],
                "output": str(output_path),
                "output_size_bytes": output_path.stat().st_size,
                "output_sha256": sha256_file(output_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
