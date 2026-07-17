from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.audit_baseline_readiness import (
    DEFAULT_REGISTRY,
    EXPECTED_MODEL_IDS,
    REPOSITORY_ROOT,
    audit_registry,
    load_registry,
)


class BaselineReadinessAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_registry(DEFAULT_REGISTRY)
        cls.report = audit_registry(
            cls.registry, REPOSITORY_ROOT, DEFAULT_REGISTRY
        )
        cls.models = {
            model["model_id"]: model for model in cls.report["models"]
        }

    def test_registry_has_exact_paper_scope(self) -> None:
        self.assertEqual(
            [model["model_id"] for model in self.registry["models"]],
            list(EXPECTED_MODEL_IDS),
        )
        self.assertEqual(self.report["summary"]["models"], 7)
        self.assertEqual(self.report["summary"]["clap_baselines"], 4)
        self.assertEqual(self.report["summary"]["vanilla_backbones"], 3)

    def test_static_evidence_matches_committed_repository(self) -> None:
        self.assertEqual(self.report["report_status"], "complete")
        self.assertEqual(self.report["summary"]["evidence_drift"], 0)
        for model in self.report["models"]:
            with self.subTest(model=model["model_id"]):
                self.assertTrue(
                    all(surface["matches_expected"] for surface in model["surfaces"])
                )

    def test_no_model_is_misreported_as_formal_ready(self) -> None:
        self.assertEqual(self.report["summary"]["formal_ready"], 0)
        self.assertTrue(
            all(not model["formal_ready"] for model in self.report["models"])
        )
        self.assertTrue(
            all(model["declared_status"] == "BLOCKED" for model in self.report["models"])
        )

    def test_code_and_resource_gaps_are_distinguished(self) -> None:
        self.assertTrue(self.models["laion_clap"]["code_surface_complete"])
        self.assertTrue(self.models["mga_clap"]["code_surface_complete"])
        self.assertFalse(self.models["robust_clap"]["code_surface_complete"])
        self.assertFalse(self.models["m2d_clap"]["code_surface_complete"])
        self.assertTrue(
            self.models["vanilla_qwen2_5_omni_3b"]["code_surface_complete"]
        )
        self.assertEqual(self.report["summary"]["code_surface_complete"], 5)
        self.assertFalse(
            self.models["laion_clap"]["resource_identity_complete"]
        )
        self.assertTrue(
            self.models["vanilla_nemotron_3b"]["resource_identity_complete"]
        )
        identity = self.models["vanilla_nemotron_3b"]["resource_identity"]
        self.assertTrue(identity["registry_exists"])
        self.assertTrue(identity["asset_found"])
        self.assertEqual(len(identity["revision"]), 40)

    def test_report_paths_are_portable(self) -> None:
        self.assertEqual(self.report["repository_root"], ".")
        self.assertEqual(
            self.report["registry"],
            "configs/baselines/baseline_readiness.json",
        )

    def test_expected_absence_is_evidence_not_a_passing_formal_surface(self) -> None:
        robust_runner = next(
            surface
            for surface in self.models["robust_clap"]["surfaces"]
            if surface["surface_id"] == "baseline_runner"
        )
        self.assertFalse(robust_runner["expected_present"])
        self.assertFalse(robust_runner["actual_present"])
        self.assertTrue(robust_runner["matches_expected"])
        self.assertTrue(robust_runner["required_for_formal"])

    def test_evidence_drift_fails_the_audit(self) -> None:
        registry = copy.deepcopy(self.registry)
        registry["models"][0]["surfaces"][0]["needle"] = "missing sentinel"
        report = audit_registry(registry, REPOSITORY_ROOT, DEFAULT_REGISTRY)
        self.assertEqual(report["report_status"], "failed")
        self.assertEqual(report["summary"]["evidence_drift"], 1)

    def test_invalid_scope_is_rejected(self) -> None:
        document = json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8"))
        document["expected_model_ids"] = document["expected_model_ids"][:-1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected_model_ids"):
                load_registry(path)

    def test_auditor_is_read_only_and_has_no_network_client(self) -> None:
        source = (REPOSITORY_ROOT / "scripts/audit_baseline_readiness.py").read_text(
            encoding="utf-8"
        )
        for fragment in (
            "requests",
            "urllib",
            "snapshot_download",
            "hf_hub_download",
            "shutil.rmtree",
            "rm -rf",
        ):
            self.assertNotIn(fragment, source)
        self.assertIn("refusing to overwrite existing report", source)

    def test_cli_writes_lf_only_json_without_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit.json"
            command = [
                sys.executable,
                str(REPOSITORY_ROOT / "scripts/audit_baseline_readiness.py"),
                "--output",
                str(output),
            ]
            completed = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            content = output.read_bytes()
            self.assertNotIn(b"\r\n", content)
            self.assertTrue(content.endswith(b"\n"))
            repeated = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("refusing to overwrite", repeated.stderr)


if __name__ == "__main__":
    unittest.main()
