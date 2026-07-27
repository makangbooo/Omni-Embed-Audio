from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_vanilla_backbone_model_pipeline import (
    REPOSITORY_ROOT,
    run_stage,
    validate_paths,
)


class RunVanillaBackboneModelPipelineTest(unittest.TestCase):
    def test_paths_are_canonical(self) -> None:
        output = (REPOSITORY_ROOT / "logs/test_vanilla_pipeline/run").resolve()
        lock = (
            REPOSITORY_ROOT
            / "results/model_locks/vanilla_qwen2_5_omni_3b.json"
        ).resolve()
        validate_paths("vanilla_qwen2_5_omni_3b", output, lock)
        portable = (
            output.parent
            / "vanilla_qwen2_5_omni_3b.portable_model_lock.json"
        ).resolve()
        validate_paths(
            "vanilla_qwen2_5_omni_3b",
            output,
            portable,
            portable_lock_evidence=True,
        )
        for bad_output, bad_lock in (
            (REPOSITORY_ROOT / "outputs/run", lock),
            (
                output,
                REPOSITORY_ROOT / "results/model_locks/vanilla_nemotron_3b.json",
            ),
            (REPOSITORY_ROOT / "logs", lock),
        ):
            with self.assertRaises(ValueError):
                validate_paths(
                    "vanilla_qwen2_5_omni_3b",
                    bad_output.resolve(),
                    bad_lock.resolve(),
                )

    def test_stage_persists_failure_before_raising(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.json"
            report = {"stages": {}}
            with patch(
                "scripts.run_vanilla_backbone_model_pipeline.subprocess.run",
                return_value=SimpleNamespace(returncode=9),
            ), self.assertRaisesRegex(RuntimeError, "audit exited with 9"):
                run_stage(
                    name="audit",
                    command=["python", "audit.py"],
                    report=report,
                    report_path=path,
                )
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["stages"]["audit"]["status"], "failed")
            self.assertEqual(saved["stages"]["audit"]["exit_code"], 9)

    def test_pipeline_is_cpu_only_non_destructive_and_ordered(self) -> None:
        implementation = (
            REPOSITORY_ROOT / "scripts/run_vanilla_backbone_model_pipeline.py"
        ).read_text(encoding="utf-8")
        wrapper = (
            REPOSITORY_ROOT / "scripts/run_vanilla_backbone_model_pipeline.sh"
        ).read_text(encoding="utf-8")
        self.assertLess(
            implementation.index("audit_vanilla_backbone_resources.py"),
            implementation.index("build_vanilla_backbone_lock.py"),
        )
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', wrapper)
        self.assertIn("requires a clean Git worktree", wrapper)
        self.assertIn("Refusing to overwrite", wrapper)
        asrur_wrapper = (
            REPOSITORY_ROOT / "scripts/run_asrur_vanilla_nemo_phase2_audit.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--portable-lock-evidence", asrur_wrapper)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', asrur_wrapper)
        self.assertIn("fixed-revision Hugging Face metadata API only", asrur_wrapper)
        self.assertIn("unset HF_HUB_OFFLINE", asrur_wrapper)
        self.assertIn("No model or dataset download is performed", asrur_wrapper)
        for forbidden in ("rm -rf", "snapshot_download", "git reset"):
            self.assertNotIn(forbidden, implementation)
            self.assertNotIn(forbidden, wrapper)
            self.assertNotIn(forbidden, asrur_wrapper)


if __name__ == "__main__":
    unittest.main()
