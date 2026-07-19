from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


fake_huggingface_hub = types.ModuleType("huggingface_hub")
fake_huggingface_hub.HfApi = object
fake_huggingface_hub.__version__ = "test"
fake_huggingface_hub.snapshot_download = lambda **_: None

with patch.dict(sys.modules, {"huggingface_hub": fake_huggingface_hub}):
    import scripts.audit_model_resources as audit_module
    from scripts.audit_model_resources import (
        audit_local_asset,
        audit_resources,
        combined_status,
        destination_for,
        git_blob_sha1,
        normalize_asset_selection,
        validate_manifest,
    )


class AuditModelResourcesTest(unittest.TestCase):
    ASSET = {
        "name": "fixture",
        "repo_id": "owner/fixture",
        "repo_type": "model",
        "revision": "a" * 40,
        "local_subdir": "fixture",
        "allow_patterns": None,
        "required_files": ["config.json", "weights.bin"],
    }

    def test_exact_asset_selection_rejects_duplicates_and_unsafe_names(self) -> None:
        self.assertEqual(normalize_asset_selection(None), ())
        self.assertEqual(
            normalize_asset_selection(["base_model", "checkpoint-1"]),
            ("base_model", "checkpoint-1"),
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            normalize_asset_selection(["base_model", "base_model"])
        for unsafe in ("../base", "base/model", "", "base model"):
            with self.subTest(value=unsafe), self.assertRaisesRegex(
                ValueError, "unsafe"
            ):
                normalize_asset_selection([unsafe])

    def test_audit_resources_reads_only_exact_requested_assets(self) -> None:
        def asset(name: str) -> dict:
            return {
                "name": name,
                "repo_id": f"owner/{name}",
                "revision": "a" * 40,
                "local_subdir": name,
                "required_files": ["weights.bin"],
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "resources.json"
            output = root / "audit.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "assets": [asset("selected"), asset("unrelated")],
                    }
                ),
                encoding="utf-8",
            )

            audited_names: list[str] = []

            def fake_remote_inventory(api: object, row: dict) -> list[dict]:
                audited_names.append(row["name"])
                return [
                    {
                        "path": "weights.bin",
                        "size_bytes": 1,
                        "lfs_sha256": "b" * 64,
                        "blob_id": None,
                    }
                ]

            def fake_local_audit(**kwargs: object) -> dict:
                row = kwargs["asset"]
                return {"name": row["name"], "status": "complete"}

            with (
                patch.object(audit_module, "HfApi", return_value=object()),
                patch.object(
                    audit_module,
                    "remote_inventory",
                    side_effect=fake_remote_inventory,
                ),
                patch.object(
                    audit_module,
                    "audit_local_asset",
                    side_effect=fake_local_audit,
                ),
            ):
                exit_code = audit_resources(
                    manifest_paths=[manifest],
                    model_root=root / "models",
                    output=output,
                    asset_names=("selected",),
                )

            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(audited_names, ["selected"])
        self.assertEqual(report["requested_assets"], ["selected"])
        self.assertEqual([row["name"] for row in report["assets"]], ["selected"])

    def test_audit_resources_rejects_requested_asset_absent_from_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "resources.json"
            output = root / "audit.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "assets": [
                            {
                                "name": "present",
                                "repo_id": "owner/present",
                                "revision": "a" * 40,
                                "local_subdir": "present",
                                "required_files": ["weights.bin"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(audit_module, "HfApi", return_value=object()):
                exit_code = audit_resources(
                    manifest_paths=[manifest],
                    model_root=root / "models",
                    output=output,
                    asset_names=("absent",),
                )
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 1)
        self.assertEqual(report["status"], "failed")
        self.assertIn("absent", report["fatal_error"])

    @staticmethod
    def remote_files(config: bytes, weights: bytes) -> list[dict[str, object]]:
        config_blob = hashlib.sha1(
            f"blob {len(config)}\0".encode("ascii") + config,
            usedforsecurity=False,
        ).hexdigest()
        return [
            {
                "path": "config.json",
                "size_bytes": len(config),
                "lfs_sha256": None,
                "blob_id": config_blob,
            },
            {
                "path": "weights.bin",
                "size_bytes": len(weights),
                "lfs_sha256": hashlib.sha256(weights).hexdigest(),
                "blob_id": None,
            },
        ]

    def prepare(
        self, root: Path, config: bytes = b"{}", weights: bytes = b"weights"
    ) -> Path:
        destination = root / "fixture"
        destination.mkdir()
        (destination / "config.json").write_bytes(config)
        (destination / "weights.bin").write_bytes(weights)
        marker = root / ".oea_asset_markers" / "fixture.json"
        marker.parent.mkdir()
        marker.write_text(
            json.dumps(
                {"repo_id": "owner/fixture", "revision": "a" * 40}
            ),
            encoding="utf-8",
        )
        return destination

    def test_complete_snapshot_hashes_every_file_and_matches_lfs(self) -> None:
        config = b'{"model":"fixture"}'
        weights = b"immutable weights"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root, config, weights)
            report = audit_local_asset(
                model_root=root,
                asset=dict(self.ASSET),
                remote_files=self.remote_files(config, weights),
            )

        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["missing_files"], [])
        self.assertEqual(report["extra_files"], [])
        self.assertEqual(report["verified_local_file_count"], 2)
        by_path = {item["path"]: item for item in report["local_files"]}
        self.assertTrue(by_path["weights.bin"]["matches_remote_lfs_sha256"])
        self.assertIsNone(by_path["config.json"]["matches_remote_lfs_sha256"])
        self.assertTrue(by_path["config.json"]["matches_remote_git_blob_id"])

    def test_missing_and_incomplete_files_report_incomplete_without_deletion(self) -> None:
        config = b"{}"
        weights = b"weights"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = self.prepare(root, config, weights)
            (destination / "weights.bin").unlink()
            partial = destination / ".cache" / "huggingface" / "download" / "x.incomplete"
            partial.parent.mkdir(parents=True)
            partial.write_bytes(b"partial")
            report = audit_local_asset(
                model_root=root,
                asset=dict(self.ASSET),
                remote_files=self.remote_files(config, weights),
            )

            self.assertTrue(partial.exists())

        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["missing_files"], ["weights.bin"])
        self.assertEqual(
            report["incomplete_files"],
            [".cache/huggingface/download/x.incomplete"],
        )

    def test_extra_file_and_marker_mismatch_are_failed(self) -> None:
        config = b"{}"
        weights = b"weights"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = self.prepare(root, config, weights)
            (destination / "unexpected.txt").write_text("unexpected", encoding="utf-8")
            marker = root / ".oea_asset_markers" / "fixture.json"
            marker.write_text(
                json.dumps({"repo_id": "wrong/repo", "revision": "b" * 40}),
                encoding="utf-8",
            )
            report = audit_local_asset(
                model_root=root,
                asset=dict(self.ASSET),
                remote_files=self.remote_files(config, weights),
            )

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["extra_files"], ["unexpected.txt"])
        self.assertTrue(any("marker mismatch" in error for error in report["errors"]))

    def test_exact_allowlisted_derived_file_is_reported_but_not_rejected(self) -> None:
        config = b"{}"
        weights = b"weights"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = self.prepare(root, config, weights)
            derived = destination / "weights_inference_only.bin"
            derived.write_bytes(b"verified later by checkpoint preparation")
            (destination / "unexpected.txt").write_text(
                "unexpected", encoding="utf-8"
            )
            report = audit_local_asset(
                model_root=root,
                asset=dict(self.ASSET),
                remote_files=self.remote_files(config, weights),
                allowed_extra_files=("weights_inference_only.bin",),
            )

        self.assertEqual(report["status"], "failed")
        self.assertEqual(
            report["allowed_extra_files"], ["weights_inference_only.bin"]
        )
        self.assertEqual(
            report["present_allowed_extra_files"],
            ["weights_inference_only.bin"],
        )
        self.assertEqual(report["extra_files"], ["unexpected.txt"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = self.prepare(root, config, weights)
            (destination / "weights_inference_only.bin").write_bytes(b"derived")
            report = audit_local_asset(
                model_root=root,
                asset=dict(self.ASSET),
                remote_files=self.remote_files(config, weights),
                allowed_extra_files=("weights_inference_only.bin",),
            )

        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["extra_files"], [])
        self.assertEqual(
            report["present_allowed_extra_files"],
            ["weights_inference_only.bin"],
        )

    def test_size_and_lfs_corruption_are_failed(self) -> None:
        config = b"{}"
        expected_weights = b"expected"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root, config, b"corrupt and longer")
            report = audit_local_asset(
                model_root=root,
                asset=dict(self.ASSET),
                remote_files=self.remote_files(config, expected_weights),
            )

        self.assertEqual(report["status"], "failed")
        self.assertTrue(any("size mismatch" in error for error in report["errors"]))
        self.assertTrue(any("LFS SHA256 mismatch" in error for error in report["errors"]))

    def test_same_size_non_lfs_corruption_fails_git_blob_check(self) -> None:
        expected_config = b"good"
        weights = b"weights"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = self.prepare(root, b"evil", weights)
            self.assertNotEqual(
                git_blob_sha1(destination / "config.json"),
                self.remote_files(expected_config, weights)[0]["blob_id"],
            )
            report = audit_local_asset(
                model_root=root,
                asset=dict(self.ASSET),
                remote_files=self.remote_files(expected_config, weights),
            )

        self.assertEqual(report["status"], "failed")
        self.assertTrue(any("Git blob mismatch" in error for error in report["errors"]))

    def test_unsafe_manifest_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            for unsafe in ("../escape", "/absolute", "C:/absolute"):
                manifest = {"schema_version": 1, "assets": [dict(self.ASSET)]}
                manifest["assets"][0]["local_subdir"] = unsafe
                with self.subTest(path=unsafe):
                    with self.assertRaisesRegex(ValueError, "unsafe local_subdir"):
                        validate_manifest(manifest, path)
                    with self.assertRaisesRegex(ValueError, "unsafe local_subdir"):
                        destination_for(Path(directory), manifest["assets"][0])

    def test_combined_status_distinguishes_incomplete_from_failure(self) -> None:
        self.assertEqual(combined_status([{"status": "complete"}]), ("complete", 0))
        self.assertEqual(
            combined_status([{"status": "complete"}, {"status": "incomplete"}]),
            ("incomplete", 2),
        )
        self.assertEqual(
            combined_status([{"status": "incomplete"}, {"status": "failed"}]),
            ("failed", 1),
        )


if __name__ == "__main__":
    unittest.main()
