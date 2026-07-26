from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_official_oea_model_pipeline import (
    REPOSITORY_ROOT,
    run_stage,
    validate_paths,
)


class RunOfficialOEAModelPipelineTest(unittest.TestCase):
    def test_paths_are_canonical_ignored_logs_and_small_result_lock(self) -> None:
        output = REPOSITORY_ROOT / "logs/test_pipeline/run"
        lock = REPOSITORY_ROOT / "results/model_locks/oea_qwen3b_cl.json"
        validate_paths("oea_qwen3b_cl", output.resolve(), lock.resolve())
        portable_lock = (
            output.parent / "oea_qwen3b_cl.portable_model_lock.json"
        )
        validate_paths(
            "oea_qwen3b_cl",
            output.resolve(),
            portable_lock.resolve(),
            portable_lock_evidence=True,
        )

        invalid = (
            (REPOSITORY_ROOT / "outputs/run", lock),
            (
                output,
                REPOSITORY_ROOT / "results/model_locks/oea_qwen3b.json",
            ),
            (REPOSITORY_ROOT / "logs", lock),
        )
        for output_path, lock_path in invalid:
            with self.subTest(output=output_path, lock=lock_path), self.assertRaises(
                ValueError
            ):
                validate_paths(
                    "oea_qwen3b_cl", output_path.resolve(), lock_path.resolve()
                )

    def test_stage_records_success_and_failure_before_raising(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "pipeline.json"
            report = {"stages": {}}
            with patch(
                "scripts.run_official_oea_model_pipeline.subprocess.run",
                return_value=SimpleNamespace(returncode=0),
            ):
                run_stage(
                    name="success",
                    command=["python", "success.py"],
                    report=report,
                    report_path=report_path,
                )
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["stages"]["success"]["status"], "complete")
            self.assertEqual(saved["stages"]["success"]["exit_code"], 0)

            with patch(
                "scripts.run_official_oea_model_pipeline.subprocess.run",
                return_value=SimpleNamespace(returncode=7),
            ), self.assertRaisesRegex(RuntimeError, "failed exited with 7"):
                run_stage(
                    name="failed",
                    command=["python", "failed.py"],
                    report=report,
                    report_path=report_path,
                )
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["stages"]["failed"]["status"], "failed")
            self.assertEqual(saved["stages"]["failed"]["exit_code"], 7)

    def test_pipeline_order_is_audit_then_prepare_then_lock(self) -> None:
        implementation = (
            REPOSITORY_ROOT / "scripts/run_official_oea_model_pipeline.py"
        ).read_text(encoding="utf-8")
        wrapper = (
            REPOSITORY_ROOT / "scripts/run_official_oea_model_pipeline.sh"
        ).read_text(encoding="utf-8")
        audit_position = implementation.index(
            "scripts/audit_official_oea_variant_resources.py"
        )
        preparation_position = implementation.index(
            "scripts/prepare_official_oea_checkpoint.py"
        )
        lock_position = implementation.index(
            "scripts/build_official_oea_model_lock.py"
        )
        self.assertLess(audit_position, preparation_position)
        self.assertLess(preparation_position, lock_position)
        self.assertIn("--verify-existing-derived", implementation)
        self.assertIn("--allow-registered-derived", implementation)
        self.assertIn("--portable-lock-evidence", implementation)
        self.assertIn("fixed_revision_api_inventory_only", implementation)
        self.assertIn('"content_downloads": False', implementation)
        self.assertIn("--verify-existing-derived", wrapper)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', wrapper)
        self.assertIn("requires a clean Git worktree", wrapper)
        self.assertIn("Refusing to overwrite", wrapper)
        for forbidden in ("rm -rf", "snapshot_download", "git reset"):
            self.assertNotIn(forbidden, implementation)
            self.assertNotIn(forbidden, wrapper)

    def test_asrur_nemo_wrapper_is_cpu_and_metadata_only_with_portable_evidence(
        self,
    ) -> None:
        wrapper = (
            REPOSITORY_ROOT / "scripts/run_asrur_nemo_phase1_audit.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('VARIANT_ID="oea_nemo3b_cl"', wrapper)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', wrapper)
        self.assertIn("unset HF_HUB_OFFLINE", wrapper)
        self.assertIn("unset TRANSFORMERS_OFFLINE", wrapper)
        self.assertIn("unset HF_DATASETS_OFFLINE", wrapper)
        self.assertNotIn('export TRANSFORMERS_OFFLINE="1"', wrapper)
        self.assertIn(
            "fixed-revision Hugging Face metadata API only",
            wrapper,
        )
        self.assertIn(
            'python -m json.tool "${PIPELINE_DIR}/model_resource_audit.json"',
            wrapper,
        )
        self.assertIn("--portable-lock-evidence", wrapper)
        self.assertIn("--verify-existing-derived", wrapper)
        self.assertIn("atomic extraction mode selected", wrapper)
        self.assertNotIn("results/model_locks", wrapper)
        for forbidden in (
            "rm -rf",
            "snapshot_download",
            "hf_hub_download",
            "git reset",
        ):
            self.assertNotIn(forbidden, wrapper)


if __name__ == "__main__":
    unittest.main()
