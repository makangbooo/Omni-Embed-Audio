from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_vanilla_backbone_resources import resolve_backbone_audit_plan
from scripts.build_vanilla_backbone_lock import (
    build_vanilla_lock,
    validate_output_path,
)
from scripts.build_vanilla_backbone_eval_config import validate_model_lock
from scripts.vanilla_backbone_registry import (
    DEFAULT_REGISTRY,
    EXPECTED_BACKBONE_IDS,
    load_vanilla_backbone_registry,
)


EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class VanillaBackboneModelLockTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_vanilla_backbone_registry()

    def write_fixture(
        self, directory: Path, backbone_id: str = "vanilla_qwen2_5_omni_3b"
    ) -> tuple[Path, dict]:
        backbone = self.registry[backbone_id]
        model_root = directory / "models"
        destination = model_root / backbone["base_asset"]["local_subdir"]
        destination.mkdir(parents=True)
        report = {
            "name": backbone["base_asset"]["name"],
            "repo_id": backbone["base_asset"]["repo_id"],
            "revision": backbone["base_asset"]["revision"],
            "destination": str(destination.resolve()),
            "marker": str(
                (
                    model_root
                    / ".oea_asset_markers"
                    / f"{backbone['base_asset']['name']}.json"
                ).resolve()
            ),
            "marker_identity": {
                "repo_id": backbone["base_asset"]["repo_id"],
                "revision": backbone["base_asset"]["revision"],
            },
            "status": "complete",
            "errors": [],
            "extra_files": [],
            "incomplete_files": [],
            "missing_files": [],
            "unsafe_symlinks": [],
            "expected_file_count": 1,
            "verified_local_file_count": 1,
            "expected_bytes": 0,
            "verified_local_bytes": 0,
            "local_files": [
                {
                    "path": "empty-but-verified",
                    "size_bytes": 0,
                    "sha256": EMPTY_SHA256,
                    "remote_size_bytes": 0,
                    "size_matches_remote": True,
                    "remote_lfs_sha256": None,
                    "matches_remote_lfs_sha256": None,
                    "remote_git_blob_id": "e" * 40,
                    "matches_remote_git_blob_id": True,
                }
            ],
        }
        audit = {
            "schema_version": 1,
            "status": "complete",
            "git_commit": "a" * 40,
            "git_status_short": "",
            "model_root": str(model_root.resolve()),
            "fatal_error": None,
            "scope": resolve_backbone_audit_plan(backbone_id),
            "requested_assets": [backbone["base_asset"]["name"]],
            "assets": [report],
        }
        path = directory / "resource_audit.json"
        path.write_text(json.dumps(audit), encoding="utf-8")
        return path, audit

    def build(
        self, audit_path: Path, backbone_id: str = "vanilla_qwen2_5_omni_3b"
    ) -> dict:
        return build_vanilla_lock(
            backbone_id=backbone_id,
            registry_path=DEFAULT_REGISTRY,
            resource_audit_path=audit_path,
        )

    def test_registry_resolves_exact_three_base_assets(self) -> None:
        observed = {
            backbone_id: (
                self.registry[backbone_id]["base_manifest"],
                self.registry[backbone_id]["base_asset"]["name"],
            )
            for backbone_id in EXPECTED_BACKBONE_IDS
        }
        self.assertEqual(
            observed,
            {
                "vanilla_nemotron_3b": (
                    "configs/resources/model03_nemo3b.json",
                    "omni_embed_nemotron_3b",
                ),
                "vanilla_qwen2_5_omni_3b": (
                    "configs/resources/model01_qwen3b_cl.json",
                    "qwen2_5_omni_3b",
                ),
                "vanilla_qwen2_5_omni_7b": (
                    "configs/resources/model04_qwen7b.json",
                    "qwen2_5_omni_7b",
                ),
            },
        )

    def test_builds_portable_base_only_lock_with_explicit_protocol_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            audit_path, _ = self.write_fixture(directory)
            lock = self.build(audit_path)

        self.assertEqual(lock["status"], "locked")
        self.assertEqual(lock["lock_type"], "vanilla_backbone_base_only")
        self.assertEqual(lock["protocol"]["audio_prompt"]["status"], "CONFLICT")
        self.assertEqual(lock["protocol"]["text_prompt"]["value"], "query:")
        self.assertEqual(lock["embedding_output"]["projection_head"], "none")
        self.assertNotIn("checkpoint", lock)
        self.assertNotIn(str(directory), json.dumps(lock))

    def test_rejects_dirty_incomplete_or_wrong_scope_evidence(self) -> None:
        mutations = (
            (lambda audit: audit.update(git_status_short=" M x"), "dirty"),
            (lambda audit: audit.update(status="incomplete"), "unacceptable status"),
            (
                lambda audit: audit["scope"].update(
                    backbone_id="vanilla_nemotron_3b"
                ),
                "scope backbone_id",
            ),
            (
                lambda audit: audit["assets"][0].update(status="incomplete"),
                "selected asset is not complete",
            ),
            (
                lambda audit: audit["assets"][0]["local_files"][0].update(
                    matches_remote_git_blob_id=False
                ),
                "Git blob",
            ),
        )
        for mutate, error in mutations:
            with tempfile.TemporaryDirectory() as raw_directory:
                directory = Path(raw_directory)
                audit_path, audit = self.write_fixture(directory)
                changed = copy.deepcopy(audit)
                mutate(changed)
                audit_path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, error):
                    self.build(audit_path)

    def test_unknown_backbone_is_rejected_before_audit(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown backbone"):
            resolve_backbone_audit_plan("vanilla_unknown")

    def test_portable_output_is_scoped_below_logs(self) -> None:
        output = (
            Path(__file__).resolve().parents[1]
            / "logs"
            / "run"
            / "vanilla_nemotron_3b.portable_model_lock.json"
        ).resolve()
        validate_output_path(
            output,
            backbone_id="vanilla_nemotron_3b",
            portable_lock_evidence=True,
        )
        with self.assertRaises(ValueError):
            validate_output_path(
                output.with_name("wrong.json"),
                backbone_id="vanilla_nemotron_3b",
                portable_lock_evidence=True,
            )

    def test_committed_qwen3b_vanilla_lock_inherits_exact_audited_base(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        vanilla_path = (
            repository_root
            / "results/model_locks/vanilla_qwen2_5_omni_3b.json"
        )
        oea_path = repository_root / "results/model_locks/oea_qwen3b.json"
        vanilla = json.loads(vanilla_path.read_text(encoding="utf-8"))
        oea = json.loads(oea_path.read_text(encoding="utf-8"))

        locked = validate_model_lock(vanilla)
        self.assertEqual(locked["base_model"], oea["base_model"])
        self.assertEqual(locked["backbone_id"], "vanilla_qwen2_5_omni_3b")
        self.assertEqual(locked["embedding_output"]["projection_head"], "none")
        self.assertNotIn("checkpoint", vanilla)
        inherited = vanilla["evidence"]["inherited_base_model_lock"]
        self.assertEqual(inherited["size_bytes"], oea_path.stat().st_size)
        self.assertEqual(
            inherited["sha256"], hashlib.sha256(oea_path.read_bytes()).hexdigest()
        )

    def test_committed_qwen7b_vanilla_lock_has_full_audited_base(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        lock_path = (
            repository_root
            / "results/model_locks/vanilla_qwen2_5_omni_7b.json"
        )
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        validated = validate_model_lock(lock)

        self.assertEqual(validated["backbone_id"], "vanilla_qwen2_5_omni_7b")
        self.assertEqual(
            validated["base_model"]["revision"],
            "ae9e1690543ffd5c0221dc27f79834d0294cba00",
        )
        files = validated["base_model"]["files"]
        self.assertEqual(len(files), 20)
        self.assertEqual(
            sum(item["size_bytes"] for item in files.values()),
            22379297323,
        )
        self.assertEqual(validated["embedding_output"]["projection_head"], "none")
        self.assertNotIn("checkpoint", lock)
        self.assertEqual(
            lock["evidence"]["portable_model_lock"]["sha256"],
            "79ab00f180cec1078979c0524a98c254fea298d3cb18a24ed8bdf8162c970014",
        )


if __name__ == "__main__":
    unittest.main()
