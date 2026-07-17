from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_official_oea_eval_config import (
    build_resolved_config,
    portable_file_identity,
    validate_model_lock,
    verify_official_model_lock_binding,
    write_json_once_or_verify,
)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class BuildOfficialOEAEvalConfigTest(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path, dict, dict]:
        protocol_path = root / "configs/eval/protocol.json"
        lock_path = root / "results/model_locks/variant.json"
        protocol_path.parent.mkdir(parents=True)
        lock_path.parent.mkdir(parents=True)
        weights = {"size_bytes": 10, "sha256": "a" * 64}
        checkpoint = {
            "repo_id": "author/checkpoint",
            "revision": "b" * 40,
            "local_subpath": "checkpoint/derived.pt",
            "size_bytes": 20,
            "sha256": "c" * 64,
            "source": "DERIVED_FROM_CODE_CHECKPOINT",
        }
        lock = {
            "schema_version": 1,
            "status": "locked",
            "variant_id": "oea_fixture",
            "model": "OEA-Fixture",
            "base_model": {
                "repo_id": "author/base",
                "revision": "d" * 40,
                "local_subdir": "base",
                "source": "CODE+AUDIT",
                "files": {
                    "model.safetensors": weights,
                    "tokenizer.json": {
                        "size_bytes": 0,
                        "sha256": digest(b""),
                    },
                },
            },
            "checkpoint": checkpoint,
        }
        protocol = {
            "schema_version": 1,
            "official_variant_id": "oea_fixture",
            "model": "OEA-Fixture",
            "base_model": {
                "repo_id": "author/base",
                "revision": "d" * 40,
                "local_subdir": "base",
                "source": "CODE",
                "files": {"model.safetensors": weights},
            },
            "checkpoint": {**checkpoint, "source": "CODE"},
            "dataset": "fixture",
        }
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
        lock_path.write_text(json.dumps(lock), encoding="utf-8")
        return protocol_path, lock_path, protocol, lock

    def resolve(
        self,
        root: Path,
        protocol_path: Path,
        lock_path: Path,
        protocol: dict,
        lock: dict,
    ) -> dict:
        return build_resolved_config(
            protocol,
            lock,
            protocol_identity=portable_file_identity(
                protocol_path, repository_root=root
            ),
            model_lock_identity=portable_file_identity(
                lock_path, repository_root=root
            ),
            git_commit="e" * 40,
        )

    def test_resolved_config_expands_to_complete_locked_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.fixture(root)
            resolved = self.resolve(root, *fixture)
            binding = verify_official_model_lock_binding(
                resolved, repository_root=root
            )
        self.assertEqual(
            set(resolved["base_model"]["files"]),
            {"model.safetensors", "tokenizer.json"},
        )
        self.assertEqual(
            resolved["checkpoint"]["source"],
            "DERIVED_FROM_CODE_CHECKPOINT",
        )
        self.assertEqual(binding["variant_id"], "oea_fixture")
        self.assertEqual(
            binding["model_lock"]["repository_path"],
            "results/model_locks/variant.json",
        )
        self.assertEqual(resolved["resolution_git_commit"], "e" * 40)

    def test_binding_rejects_model_lock_content_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path, lock_path, protocol, lock = self.fixture(root)
            resolved = self.resolve(root, protocol_path, lock_path, protocol, lock)
            lock_path.write_text(
                json.dumps({**lock, "model": "OEA-FixturE"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                verify_official_model_lock_binding(resolved, repository_root=root)

    def test_rejects_variant_checkpoint_and_base_file_mismatches(self) -> None:
        mutations = (
            (
                lambda protocol: protocol.update(official_variant_id="other"),
                "official_variant_id",
            ),
            (
                lambda protocol: protocol["checkpoint"].update(sha256="f" * 64),
                "checkpoint.sha256",
            ),
            (
                lambda protocol: protocol["base_model"]["files"][
                    "model.safetensors"
                ].update(size_bytes=11),
                "base_model file",
            ),
        )
        for mutate, expected_error in mutations:
            with (
                self.subTest(expected_error=expected_error),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                protocol_path, lock_path, protocol, lock = self.fixture(root)
                protocol = copy.deepcopy(protocol)
                mutate(protocol)
                with self.assertRaisesRegex(ValueError, expected_error):
                    self.resolve(root, protocol_path, lock_path, protocol, lock)

    def test_rejects_unsafe_locked_inventory_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, _, lock = self.fixture(root)
            lock = copy.deepcopy(lock)
            lock["base_model"]["files"] = {
                "../outside": {"size_bytes": 1, "sha256": "a" * 64}
            }
            with self.assertRaisesRegex(ValueError, "safe relative path"):
                validate_model_lock(lock)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, _, lock = self.fixture(root)
            lock = copy.deepcopy(lock)
            lock["base_model"]["local_subdir"] = "C:/outside"
            with self.assertRaisesRegex(ValueError, "portable safe relative path"):
                validate_model_lock(lock)

    def test_resolved_output_is_create_or_exact_verify_never_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resolved.json"
            self.assertEqual(write_json_once_or_verify(path, {"value": 1}), "created")
            self.assertEqual(
                write_json_once_or_verify(path, {"value": 1}),
                "verified_existing",
            )
            with self.assertRaisesRegex(RuntimeError, "differs"):
                write_json_once_or_verify(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": 1})


if __name__ == "__main__":
    unittest.main()
