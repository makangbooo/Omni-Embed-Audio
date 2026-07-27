#!/usr/bin/env python3
"""Resolve a vanilla embedding protocol against a committed base-only lock."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Mapping


if __package__ in {None, ""}:
    repository_root_for_import = Path(__file__).resolve().parents[1]
    if str(repository_root_for_import) not in sys.path:
        sys.path.insert(0, str(repository_root_for_import))

from scripts.build_official_oea_eval_config import (
    REPOSITORY_ROOT,
    git_output,
    nonempty_string,
    portable_file_identity,
    read_json_object,
    repository_file,
    require_tracked_clean_input,
    safe_relative_path,
    sha256_file,
    validate_file_inventory,
    validate_portable_identity,
    write_json_once_or_verify,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def tagged_value(section: Mapping[str, Any], key: str, label: str) -> Any:
    item = section.get(key)
    if (
        not isinstance(item, dict)
        or "value" not in item
        or not isinstance(item.get("source"), str)
        or not item["source"].strip()
    ):
        raise ValueError(f"{label}.{key} must contain value and source")
    return item["value"]


def validate_protocol(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    expected = {
        "text_prompt",
        "audio_prompt",
        "pooling",
        "normalization",
        "projection",
    }
    if set(value) != expected:
        raise ValueError(f"{label} fields are not exact")
    text_prompt = value.get("text_prompt")
    audio_prompt = value.get("audio_prompt")
    if (
        not isinstance(text_prompt, dict)
        or text_prompt.get("value") != "query:"
        or text_prompt.get("runtime_join")
        != "direct concatenation without an inserted separator"
        or text_prompt.get("runtime_form") != "query:<caption>"
    ):
        raise ValueError(f"{label}.text_prompt must fix the exact query:<caption> runtime")
    if (
        not isinstance(audio_prompt, dict)
        or audio_prompt.get("status") != "CONFLICT"
        or audio_prompt.get("runtime_value")
        != "audio-only chat message; passage_prefix parameter is not inserted"
        or audio_prompt.get("paper_value") != "passage:"
    ):
        raise ValueError(f"{label}.audio_prompt must preserve the paper/code conflict")
    if value.get("pooling", {}).get("value") != (
        "attention-mask-aware mean of last hidden states"
    ):
        raise ValueError(f"{label}.pooling differs from public code")
    if value.get("normalization", {}).get("value") != "L2":
        raise ValueError(f"{label}.normalization differs from public code")
    if value.get("projection", {}).get("value") != (
        "none; retain the backbone hidden dimension"
    ):
        raise ValueError(f"{label}.projection must remain base-only")
    return copy.deepcopy(value)


def validate_model_lock(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != 1 or value.get("status") != "locked":
        raise ValueError("vanilla model lock is not a locked schema-v1 artifact")
    if value.get("lock_type") != "vanilla_backbone_base_only":
        raise ValueError("vanilla model lock has an unexpected lock_type")
    if "checkpoint" in value:
        raise ValueError("vanilla base-only lock must not contain a checkpoint")
    backbone_id = nonempty_string(value.get("backbone_id"), "lock backbone_id")
    model = nonempty_string(value.get("model"), "lock model")
    family = nonempty_string(value.get("family"), "lock family")
    base = value.get("base_model")
    if not isinstance(base, dict):
        raise ValueError("vanilla model lock must contain base_model")
    base_model = {
        "repo_id": nonempty_string(base.get("repo_id"), "base_model.repo_id"),
        "revision": nonempty_string(
            base.get("revision"), "base_model.revision"
        ),
        "local_subdir": safe_relative_path(
            base.get("local_subdir"), "base_model.local_subdir"
        ).as_posix(),
        "source": nonempty_string(base.get("source"), "base_model.source"),
        "files": validate_file_inventory(base.get("files"), "base_model.files"),
    }
    embedding = value.get("embedding_output")
    if (
        not isinstance(embedding, dict)
        or embedding.get("projection_head") != "none"
    ):
        raise ValueError("vanilla model lock must explicitly disable projection")
    return {
        "backbone_id": backbone_id,
        "model": model,
        "family": family,
        "protocol": validate_protocol(value.get("protocol"), "lock protocol"),
        "base_model": base_model,
        "embedding_output": copy.deepcopy(embedding),
    }


def compare_protocol_to_lock(
    protocol: Mapping[str, Any], locked: Mapping[str, Any]
) -> None:
    if protocol.get("schema_version") != 1:
        raise ValueError("protocol config has unsupported schema_version")
    if protocol.get("backbone_id") != locked["backbone_id"]:
        raise ValueError("protocol backbone_id differs from vanilla model lock")
    if protocol.get("model") != locked["model"]:
        raise ValueError("protocol model differs from vanilla model lock")
    if validate_protocol(protocol.get("protocol"), "protocol") != locked["protocol"]:
        raise ValueError("protocol prompt/pooling contract differs from model lock")
    base = protocol.get("base_model")
    if not isinstance(base, dict):
        raise ValueError("protocol must contain a base_model object")
    for field in ("repo_id", "revision", "local_subdir"):
        if base.get(field) != locked["base_model"][field]:
            raise ValueError(f"protocol base_model.{field} differs from model lock")
    model_config = protocol.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("protocol must contain model_config")
    if tagged_value(model_config, "query_prefix", "model_config") != "query:":
        raise ValueError("model_config.query_prefix differs from locked protocol")
    if tagged_value(model_config, "passage_prefix", "model_config") != "passage:":
        raise ValueError("model_config.passage_prefix must preserve the unused parameter")
    if tagged_value(model_config, "torch_dtype", "model_config") != "bfloat16":
        raise ValueError("vanilla formal protocol must use bfloat16")
    if tagged_value(model_config, "trust_remote_code", "model_config") is not True:
        raise ValueError("vanilla formal protocol must fix trust_remote_code=true")
    forbidden = {
        "projection_dim",
        "projection_dropout",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "lora_targets",
    }
    present = sorted(forbidden & set(model_config))
    if present:
        raise ValueError(f"vanilla model_config contains OEA-only fields: {present}")


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
        model_lock_identity, "vanilla model lock"
    )
    resolved = copy.deepcopy(dict(protocol))
    resolved["base_model"] = copy.deepcopy(locked["base_model"])
    resolved["embedding_output"] = copy.deepcopy(locked["embedding_output"])
    resolved["vanilla_model_lock"] = {
        "backbone_id": locked["backbone_id"],
        **model_lock_identity,
    }
    resolved["protocol_config"] = protocol_identity
    resolved["resolution_git_commit"] = git_commit
    return resolved


def verify_vanilla_model_lock_binding(
    config: Mapping[str, Any], *, repository_root: Path = REPOSITORY_ROOT
) -> dict[str, Any]:
    binding = config.get("vanilla_model_lock")
    protocol_identity = config.get("protocol_config")
    if not isinstance(binding, dict) or not isinstance(protocol_identity, dict):
        raise ValueError(
            "resolved config must contain vanilla_model_lock and protocol_config"
        )
    lock_path = repository_file(
        binding.get("repository_path"),
        "vanilla_model_lock.repository_path",
        repository_root=repository_root,
    )
    protocol_path = repository_file(
        protocol_identity.get("repository_path"),
        "protocol_config.repository_path",
        repository_root=repository_root,
    )
    for path, identity, label in (
        (lock_path, binding, "vanilla model lock"),
        (protocol_path, protocol_identity, "protocol config"),
    ):
        validated = validate_portable_identity(identity, label)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != validated["size_bytes"]:
            raise ValueError(f"{label} size differs from resolved config")
        if sha256_file(path) != validated["sha256"]:
            raise ValueError(f"{label} SHA256 differs from resolved config")
    model_lock = read_json_object(lock_path, "vanilla model lock")
    protocol = read_json_object(protocol_path, "protocol config")
    locked = validate_model_lock(model_lock)
    compare_protocol_to_lock(protocol, locked)
    if binding.get("backbone_id") != locked["backbone_id"]:
        raise ValueError("resolved binding backbone_id differs from model lock")
    for field in ("backbone_id", "model", "protocol", "base_model", "embedding_output"):
        expected = locked[field]
        if config.get(field) != expected:
            raise ValueError(f"resolved config {field} differs from model lock")
    return {
        "backbone_id": locked["backbone_id"],
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


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol_config.resolve()
    lock_path = args.model_lock.resolve()
    output_path = args.output.resolve()
    if git_output("status", "--short"):
        raise RuntimeError("vanilla config resolution requires a clean Git worktree")
    require_tracked_clean_input(protocol_path, "protocol config")
    require_tracked_clean_input(lock_path, "vanilla model lock")
    git_commit = git_output("rev-parse", "HEAD")
    resolved = build_resolved_config(
        read_json_object(protocol_path, "protocol config"),
        read_json_object(lock_path, "vanilla model lock"),
        protocol_identity=portable_file_identity(
            protocol_path, repository_root=REPOSITORY_ROOT
        ),
        model_lock_identity=portable_file_identity(
            lock_path, repository_root=REPOSITORY_ROOT
        ),
        git_commit=git_commit,
    )
    operation = write_json_once_or_verify(output_path, resolved)
    print(
        json.dumps(
            {
                "status": "complete",
                "operation": operation,
                "backbone_id": resolved["backbone_id"],
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
