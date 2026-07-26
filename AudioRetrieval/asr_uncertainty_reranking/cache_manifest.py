"""Immutable cache manifests and strict compatibility checks."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

SCHEMA_VERSION = 1
REQUIRED_IDENTITY_FIELDS = (
    "dataset",
    "split",
    "inputs",
    "model",
    "tokenizer",
    "pooling",
    "embedding_dim",
    "max_length",
    "dtype",
    "normalization",
    "seed",
)
REQUIRED_PROVENANCE_FIELDS = ("created_at", "command", "git_commit")


class CacheManifestError(ValueError):
    """Base class for malformed or incompatible cache manifests."""


class CacheManifestValidationError(CacheManifestError):
    """Raised when a manifest does not satisfy the schema contract."""


class CacheManifestMismatchError(CacheManifestError):
    """Raised instead of silently reusing an incompatible cache."""

    def __init__(self, differences: Sequence[str]):
        self.differences = tuple(differences)
        super().__init__("cache manifest mismatch:\n- " + "\n- ".join(self.differences))


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a JSON value deterministically for hashes and comparisons."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path | str, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 of a regular file using bounded memory."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"not a regular file: {file_path}")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path | str, *, relative_to: Optional[Path | str] = None) -> Dict[str, Any]:
    """Create a path/size/SHA-256 record for an existing file."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"not a regular file: {file_path}")
    if relative_to is None:
        recorded_path = str(file_path.resolve())
    else:
        recorded_path = file_path.resolve().relative_to(Path(relative_to).resolve()).as_posix()
    return {
        "path": recorded_path,
        "size_bytes": file_path.stat().st_size,
        "sha256": sha256_file(file_path),
    }


def _validate_file_records(records: object, *, field_name: str, allow_empty: bool) -> None:
    if not isinstance(records, list):
        raise CacheManifestValidationError(f"{field_name} must be a list")
    if not records and not allow_empty:
        raise CacheManifestValidationError(f"{field_name} must not be empty")
    seen_paths = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise CacheManifestValidationError(f"{field_name}[{index}] must be an object")
        if set(record) != {"path", "size_bytes", "sha256"}:
            raise CacheManifestValidationError(
                f"{field_name}[{index}] must contain exactly path, size_bytes, sha256"
            )
        path = record["path"]
        size = record["size_bytes"]
        checksum = record["sha256"]
        if not isinstance(path, str) or not path:
            raise CacheManifestValidationError(f"{field_name}[{index}].path is invalid")
        if path in seen_paths:
            raise CacheManifestValidationError(f"duplicate path in {field_name}: {path!r}")
        seen_paths.add(path)
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise CacheManifestValidationError(
                f"{field_name}[{index}].size_bytes must be a non-negative integer"
            )
        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
        ):
            raise CacheManifestValidationError(
                f"{field_name}[{index}].sha256 must be lowercase SHA-256 hex"
            )


def _require_nonempty_string(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CacheManifestValidationError(f"{field_name} must be a non-empty string")


def validate_cache_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the stable cache-manifest schema.

    Fields that are not applicable to a particular artifact must still be
    present in ``identity`` with JSON ``null``.  Presence prevents an omitted
    protocol decision from enabling accidental cache reuse.
    """

    if not isinstance(manifest, Mapping):
        raise CacheManifestValidationError("manifest must be an object")
    required_top_level = {
        "schema_version",
        "artifact_type",
        "identity",
        "provenance",
        "outputs",
    }
    missing = required_top_level - set(manifest)
    if missing:
        raise CacheManifestValidationError(f"missing top-level fields: {sorted(missing)}")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise CacheManifestValidationError(
            f"schema_version must equal {SCHEMA_VERSION}, got {manifest['schema_version']!r}"
        )
    _require_nonempty_string(manifest["artifact_type"], field_name="artifact_type")

    identity = manifest["identity"]
    if not isinstance(identity, Mapping):
        raise CacheManifestValidationError("identity must be an object")
    missing_identity = set(REQUIRED_IDENTITY_FIELDS) - set(identity)
    if missing_identity:
        raise CacheManifestValidationError(
            f"identity is missing fields: {sorted(missing_identity)}"
        )
    _require_nonempty_string(identity["dataset"], field_name="identity.dataset")
    _require_nonempty_string(identity["split"], field_name="identity.split")
    _validate_file_records(identity["inputs"], field_name="identity.inputs", allow_empty=False)
    if not isinstance(identity["model"], Mapping):
        raise CacheManifestValidationError("identity.model must be an object")
    missing_model = {"name", "revision", "checkpoint"} - set(identity["model"])
    if missing_model:
        raise CacheManifestValidationError(
            f"identity.model is missing fields: {sorted(missing_model)}"
        )
    _require_nonempty_string(identity["model"]["name"], field_name="identity.model.name")
    _require_nonempty_string(
        identity["model"]["revision"],
        field_name="identity.model.revision",
    )
    if identity["model"]["checkpoint"] is not None:
        _require_nonempty_string(
            identity["model"]["checkpoint"],
            field_name="identity.model.checkpoint",
        )
    if identity["tokenizer"] is not None and not isinstance(
        identity["tokenizer"], (str, Mapping)
    ):
        raise CacheManifestValidationError("identity.tokenizer must be string, object, or null")
    for integer_field in ("embedding_dim", "max_length", "seed"):
        value = identity[integer_field]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise CacheManifestValidationError(
                f"identity.{integer_field} must be a non-negative integer or null"
            )
    for string_field in ("pooling", "dtype", "normalization"):
        value = identity[string_field]
        if value is not None:
            _require_nonempty_string(value, field_name=f"identity.{string_field}")

    provenance = manifest["provenance"]
    if not isinstance(provenance, Mapping):
        raise CacheManifestValidationError("provenance must be an object")
    missing_provenance = set(REQUIRED_PROVENANCE_FIELDS) - set(provenance)
    if missing_provenance:
        raise CacheManifestValidationError(
            f"provenance is missing fields: {sorted(missing_provenance)}"
        )
    _require_nonempty_string(provenance["created_at"], field_name="provenance.created_at")
    try:
        created_at = datetime.fromisoformat(provenance["created_at"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise CacheManifestValidationError(
            "provenance.created_at must be ISO-8601"
        ) from exc
    if created_at.tzinfo is None:
        raise CacheManifestValidationError("provenance.created_at must include a timezone")
    if not isinstance(provenance["command"], list) or not provenance["command"]:
        raise CacheManifestValidationError("provenance.command must be a non-empty argv list")
    for index, argument in enumerate(provenance["command"]):
        if not isinstance(argument, str):
            raise CacheManifestValidationError(
                f"provenance.command[{index}] must be a string"
            )
    _require_nonempty_string(
        provenance["git_commit"],
        field_name="provenance.git_commit",
    )

    _validate_file_records(manifest["outputs"], field_name="outputs", allow_empty=False)
    try:
        canonical_json_bytes(manifest)
    except (TypeError, ValueError) as exc:
        raise CacheManifestValidationError(f"manifest is not strict JSON: {exc}") from exc


def build_cache_manifest(
    *,
    artifact_type: str,
    dataset: str,
    split: str,
    inputs: Sequence[Mapping[str, Any]],
    model_name: str,
    model_revision: str,
    model_checkpoint: Optional[str],
    tokenizer: Any,
    pooling: Optional[str],
    embedding_dim: Optional[int],
    max_length: Optional[int],
    dtype: Optional[str],
    normalization: Optional[str],
    seed: Optional[int],
    command: Sequence[str],
    git_commit: str,
    outputs: Sequence[Mapping[str, Any]],
    extra_identity: Optional[Mapping[str, Any]] = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build and validate a complete cache manifest."""

    identity: MutableMapping[str, Any] = {
        "dataset": dataset,
        "split": split,
        "inputs": [dict(record) for record in inputs],
        "model": {
            "name": model_name,
            "revision": model_revision,
            "checkpoint": model_checkpoint,
        },
        "tokenizer": deepcopy(tokenizer),
        "pooling": pooling,
        "embedding_dim": embedding_dim,
        "max_length": max_length,
        "dtype": dtype,
        "normalization": normalization,
        "seed": seed,
    }
    if extra_identity:
        collisions = set(extra_identity).intersection(identity)
        if collisions:
            raise CacheManifestValidationError(
                f"extra_identity cannot replace required fields: {sorted(collisions)}"
            )
        identity.update(deepcopy(dict(extra_identity)))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": artifact_type,
        "identity": identity,
        "provenance": {
            "created_at": created_at
            or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": list(command),
            "git_commit": git_commit,
        },
        "outputs": [dict(record) for record in outputs],
    }
    validate_cache_manifest(manifest)
    return manifest


def _compatibility_payload(
    manifest: Mapping[str, Any],
    *,
    include_git_commit: bool,
) -> Dict[str, Any]:
    payload = {
        "schema_version": manifest["schema_version"],
        "artifact_type": manifest["artifact_type"],
        "identity": manifest["identity"],
    }
    if include_git_commit:
        payload["producer_git_commit"] = manifest["provenance"]["git_commit"]
    return payload


def manifest_fingerprint(
    manifest: Mapping[str, Any],
    *,
    include_git_commit: bool = True,
) -> str:
    """Hash the fields that decide whether a cache may be reused."""

    validate_cache_manifest(manifest)
    return hashlib.sha256(
        canonical_json_bytes(
            _compatibility_payload(
                manifest,
                include_git_commit=include_git_commit,
            )
        )
    ).hexdigest()


def _describe_differences(expected: object, actual: object, *, path: str) -> List[str]:
    if type(expected) is not type(actual):
        return [
            f"{path}: expected type {type(expected).__name__}, "
            f"got {type(actual).__name__}"
        ]
    if isinstance(expected, Mapping):
        differences: List[str] = []
        expected_keys = set(expected)
        actual_keys = set(actual)
        for missing in sorted(expected_keys - actual_keys):
            differences.append(f"{path}.{missing}: missing from actual")
        for extra in sorted(actual_keys - expected_keys):
            differences.append(f"{path}.{extra}: unexpected in actual")
        for key in sorted(expected_keys & actual_keys):
            differences.extend(
                _describe_differences(
                    expected[key],
                    actual[key],
                    path=f"{path}.{key}",
                )
            )
        return differences
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [f"{path}: expected length {len(expected)}, got {len(actual)}"]
        differences = []
        for index, (expected_value, actual_value) in enumerate(zip(expected, actual)):
            differences.extend(
                _describe_differences(
                    expected_value,
                    actual_value,
                    path=f"{path}[{index}]",
                )
            )
        return differences
    if expected != actual:
        return [f"{path}: expected {expected!r}, got {actual!r}"]
    return []


def assert_cache_compatible(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    allow_git_commit_mismatch: bool = False,
) -> None:
    """Raise with field-level evidence if an existing cache is incompatible.

    Git commit identity is compared by default.  Cross-commit reuse therefore
    requires an explicit caller decision and should be logged by that caller.
    """

    validate_cache_manifest(expected)
    validate_cache_manifest(actual)
    expected_payload = _compatibility_payload(
        expected,
        include_git_commit=not allow_git_commit_mismatch,
    )
    actual_payload = _compatibility_payload(
        actual,
        include_git_commit=not allow_git_commit_mismatch,
    )
    differences = _describe_differences(expected_payload, actual_payload, path="$")
    if differences:
        raise CacheManifestMismatchError(differences)


def load_cache_manifest(path: Path | str) -> Dict[str, Any]:
    """Load and validate a UTF-8 JSON manifest."""

    manifest_path = Path(path)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_cache_manifest(value)
    return value


def write_cache_manifest_once(path: Path | str, manifest: Mapping[str, Any]) -> bool:
    """Atomically create a manifest without replacing an existing file.

    Returns ``True`` when the file was created and ``False`` when an identical
    manifest already existed.  A different existing file raises
    ``FileExistsError``.  The temporary file and final hard link live in the
    same directory, so successful publication is atomic.
    """

    validate_cache_manifest(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(manifest)

    if destination.exists():
        existing = destination.read_bytes()
        if existing == payload:
            return False
        raise FileExistsError(f"refusing to replace existing manifest: {destination}")

    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.link(temporary_name, destination)
        except FileExistsError:
            existing = destination.read_bytes()
            if existing == payload:
                return False
            raise FileExistsError(f"refusing to replace existing manifest: {destination}")
        return True
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def verify_file_records(
    records: Iterable[Mapping[str, Any]],
    *,
    root: Optional[Path | str] = None,
) -> List[str]:
    """Return human-readable mismatches for recorded files."""

    records_list = [dict(record) for record in records]
    _validate_file_records(records_list, field_name="records", allow_empty=True)
    root_path = Path(root) if root is not None else None
    mismatches: List[str] = []
    for record in records_list:
        path = Path(record["path"])
        if root_path is not None and not path.is_absolute():
            path = root_path / path
        if not path.is_file():
            mismatches.append(f"{record['path']}: missing")
            continue
        actual_size = path.stat().st_size
        if actual_size != record["size_bytes"]:
            mismatches.append(
                f"{record['path']}: size expected {record['size_bytes']}, got {actual_size}"
            )
            continue
        actual_checksum = sha256_file(path)
        if actual_checksum != record["sha256"]:
            mismatches.append(
                f"{record['path']}: sha256 expected {record['sha256']}, "
                f"got {actual_checksum}"
            )
    return mismatches
