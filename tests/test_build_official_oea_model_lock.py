from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_official_oea_variant_resources import (
    resolve_variant_audit_plan,
)
from scripts.build_official_oea_model_lock import (
    build_model_lock,
    portable_file_identity,
)
from scripts.prepare_official_oea_checkpoint import (
    DEFAULT_REGISTRY,
    load_checkpoint_registry,
    sha256_file,
    variant_paths,
)


EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class BuildOfficialOEAModelLockTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_checkpoint_registry(DEFAULT_REGISTRY)

    def write_fixture(
        self, directory: Path, variant_id: str = "oea_qwen3b_cl"
    ) -> tuple[Path, Path, Path, dict, dict]:
        model_root = directory / "models"
        model_root.mkdir()
        variant = self.registry[variant_id]
        checkpoint_asset = variant["checkpoint_asset"]
        _, derived = variant_paths(variant, model_root)
        derived.parent.mkdir(parents=True)
        derived.write_bytes(b"synthetic derived checkpoint")

        def asset_report(asset: dict, files: list[dict]) -> dict:
            destination = model_root / asset["local_subdir"]
            return {
                "name": asset["name"],
                "repo_id": asset["repo_id"],
                "revision": asset["revision"],
                "destination": str(destination.resolve()),
                "marker": str(
                    (
                        model_root
                        / ".oea_asset_markers"
                        / f"{asset['name']}.json"
                    ).resolve()
                ),
                "marker_identity": {
                    "repo_id": asset["repo_id"],
                    "revision": asset["revision"],
                },
                "status": "complete",
                "errors": [],
                "extra_files": [],
                "incomplete_files": [],
                "missing_files": [],
                "unsafe_symlinks": [],
                "expected_file_count": len(files),
                "verified_local_file_count": len(files),
                "expected_bytes": sum(row["size_bytes"] for row in files),
                "verified_local_bytes": sum(row["size_bytes"] for row in files),
                "local_files": files,
            }

        base_files = [
            {
                "path": "empty-but-valid-file",
                "size_bytes": 0,
                "sha256": EMPTY_SHA256,
                "remote_size_bytes": 0,
                "size_matches_remote": True,
                "remote_lfs_sha256": None,
                "matches_remote_lfs_sha256": None,
                "remote_git_blob_id": "e" * 40,
                "matches_remote_git_blob_id": True,
            }
        ]
        checkpoint_files = [
            {
                "path": checkpoint_asset["source_file"],
                "size_bytes": checkpoint_asset["source_size_bytes"],
                "sha256": checkpoint_asset["source_sha256"],
                "remote_size_bytes": checkpoint_asset["source_size_bytes"],
                "size_matches_remote": True,
                "remote_lfs_sha256": checkpoint_asset["source_sha256"],
                "matches_remote_lfs_sha256": True,
                "remote_git_blob_id": "f" * 40,
                "matches_remote_git_blob_id": None,
            }
        ]
        audit = {
            "schema_version": 1,
            "status": "incomplete",
            "git_commit": "a" * 40,
            "git_status_short": "",
            "model_root": str(model_root.resolve()),
            "fatal_error": None,
            "scope": resolve_variant_audit_plan(variant_id),
            "requested_assets": [
                variant["base_asset"]["name"],
                checkpoint_asset["name"],
            ],
            "assets": [
                asset_report(variant["base_asset"], base_files),
                asset_report(checkpoint_asset, checkpoint_files),
                {
                    "name": "unrelated_asset",
                    "repo_id": "example/unrelated",
                    "revision": "c" * 40,
                    "status": "incomplete",
                    "errors": ["destination directory does not exist"],
                },
            ],
        }
        preparation = {
            "schema_version": 1,
            "status": "complete",
            "operation": "inspect_and_extract",
            "git_commit": "b" * 40,
            "git_status_short": "",
            "variant_id": variant_id,
            "variant": variant,
            "registry": portable_file_identity(DEFAULT_REGISTRY),
            "source_checkpoint": str(variant_paths(variant, model_root)[0]),
            "derived_checkpoint": str(derived),
            "derived_checkpoint_identity": {
                "size_bytes": derived.stat().st_size,
                "sha256": sha256_file(derived),
            },
            "measured_structure_expectations": {
                "expected_lora_tensors": 544,
                "expected_lora_bytes": 50_462_720,
            },
        }
        audit_path = directory / "model_resource_audit.json"
        preparation_path = directory / "preparation_manifest.json"
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        preparation_path.write_text(json.dumps(preparation), encoding="utf-8")
        return audit_path, preparation_path, derived, audit, preparation

    def build(
        self,
        audit_path: Path,
        preparation_path: Path,
        variant_id: str = "oea_qwen3b_cl",
    ) -> dict:
        return build_model_lock(
            variant_id=variant_id,
            registry_path=DEFAULT_REGISTRY,
            resource_audit_path=audit_path,
            preparation_path=preparation_path,
        )

    def test_builds_portable_lock_when_only_unrelated_asset_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            audit_path, preparation_path, _, _, _ = self.write_fixture(directory)
            lock = self.build(audit_path, preparation_path)

        variant = self.registry["oea_qwen3b_cl"]
        self.assertEqual(lock["status"], "locked")
        self.assertEqual(lock["variant_id"], "oea_qwen3b_cl")
        self.assertEqual(
            lock["base_model"]["repo_id"], variant["base_asset"]["repo_id"]
        )
        self.assertEqual(
            lock["base_model"]["files"]["empty-but-valid-file"]["size_bytes"], 0
        )
        self.assertEqual(
            lock["checkpoint"]["local_subpath"],
            "OEA-Qwen3B-Cl/step_40_inference_only.pt",
        )
        self.assertEqual(lock["measured_structure"]["lora_tensor_count"], 544)
        self.assertNotIn(str(directory), json.dumps(lock))

    def test_rejects_derived_checkpoint_content_drift(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            audit_path, preparation_path, derived, _, _ = self.write_fixture(directory)
            derived.write_bytes(b"changed after checkpoint preparation")
            with self.assertRaisesRegex(ValueError, "drifted after preparation"):
                self.build(audit_path, preparation_path)

    def test_rejects_incomplete_selected_resource(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            audit_path, preparation_path, _, audit, _ = self.write_fixture(directory)
            audit["assets"][0]["status"] = "incomplete"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selected asset is not complete"):
                self.build(audit_path, preparation_path)

    def test_rejects_dirty_or_unverified_evidence(self) -> None:
        mutations = (
            (
                "dirty audit",
                lambda audit, prep: audit.update(git_status_short=" M x"),
                "dirty",
            ),
            (
                "dirty preparation",
                lambda audit, prep: prep.update(git_status_short="?? x"),
                "dirty",
            ),
            (
                "unverified base file",
                lambda audit, prep: audit["assets"][0]["local_files"][0].update(
                    matches_remote_git_blob_id=False
                ),
                "Git blob",
            ),
            (
                "wrong official variant scope",
                lambda audit, prep: audit["scope"].update(
                    variant_id="oea_qwen3b"
                ),
                "scope variant_id",
            ),
            (
                "scope and requested assets disagree",
                lambda audit, prep: audit.update(
                    requested_assets=["qwen2_5_omni_3b"]
                ),
                "requested assets",
            ),
        )
        for label, mutate, error in mutations:
            with (
                self.subTest(label=label),
                tempfile.TemporaryDirectory() as raw_directory,
            ):
                directory = Path(raw_directory)
                fixture = self.write_fixture(directory)
                audit_path, preparation_path, _, audit, preparation = fixture
                audit = copy.deepcopy(audit)
                preparation = copy.deepcopy(preparation)
                mutate(audit, preparation)
                audit_path.write_text(json.dumps(audit), encoding="utf-8")
                preparation_path.write_text(json.dumps(preparation), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, error):
                    self.build(audit_path, preparation_path)

    def test_generic_multi_asset_audit_without_scope_remains_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            audit_path, preparation_path, _, audit, _ = self.write_fixture(directory)
            audit.pop("scope")
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            lock = self.build(audit_path, preparation_path)
        self.assertEqual(lock["status"], "locked")

    def test_existing_qwen3b_cl_config_matches_fixed_registry_identity(self) -> None:
        config = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "configs/eval/qwen3b_cl_clotho_embeddings.json"
            ).read_text(encoding="utf-8")
        )
        variant = self.registry["oea_qwen3b_cl"]
        for field in ("repo_id", "revision", "local_subdir"):
            self.assertEqual(config["base_model"][field], variant["base_asset"][field])
        for field in ("repo_id", "revision"):
            self.assertEqual(
                config["checkpoint"][field], variant["checkpoint_asset"][field]
            )


if __name__ == "__main__":
    unittest.main()
