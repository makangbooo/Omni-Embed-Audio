from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/run_takeover_approved_data_tmux.sh"


class TakeoverDataTmuxWrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = WRAPPER.read_text(encoding="utf-8")

    def test_all_three_approved_data_pipelines_are_explicit(self) -> None:
        for pipeline in ("data04", "data06", "data08"):
            self.assertIn(f"{pipeline})", self.source)
        for child in (
            "download_data04_mecat_00a_test.sh",
            "run_data05_mecat_validation.sh",
            "download_data06_audiocaps_v2_metadata.sh",
            "run_data07_audiocaps_v2_metadata_validation.sh",
            "download_data08_wavcaps_metadata.sh",
            "run_data09_wavcaps_metadata_audit.sh",
        ):
            self.assertIn(child, self.source)

    def test_tmux_logging_progress_and_retention_contract(self) -> None:
        self.assertIn('[[ -z "${TMUX:-}" ]]', self.source)
        self.assertIn('exec > >(tee -a "${LOG_FILE}")', self.source)
        for field in (
            "stage_elapsed=",
            "overall_elapsed=",
            "throughput=",
            "stage_remaining=",
            "overall_remaining=",
            "expected_completion=",
            "FINAL_RUN_RC=",
            "COMPLETION_STATUS=",
            "FAILED_STAGE=",
            "ERROR_SUMMARY=",
        ):
            self.assertIn(field, self.source)
        self.assertIn("exec bash -i", self.source)
        self.assertNotIn("nohup", self.source)

    def test_formal_run_is_cpu_only_and_refuses_dirty_worktree(self) -> None:
        self.assertIn('git status --short', self.source)
        self.assertIn("CUDA_VISIBLE_DEVICES is disabled", self.source)
        self.assertNotIn("nvidia-smi", self.source)


if __name__ == "__main__":
    unittest.main()
