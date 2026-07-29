from __future__ import annotations

from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TMUX_WRAPPER = (
    REPOSITORY_ROOT / "scripts/run_official_oea_model_resource_audit_tmux.sh"
)
AUDIT_WRAPPER = REPOSITORY_ROOT / "scripts/run_official_oea_model_resource_audit.sh"


class OfficialModelResourceAuditTmuxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmux_text = TMUX_WRAPPER.read_text(encoding="utf-8")
        cls.audit_text = AUDIT_WRAPPER.read_text(encoding="utf-8")

    def test_long_audit_requires_visible_tmux_and_retains_shell(self) -> None:
        self.assertIn('if [[ -z "${TMUX:-}" ]]', self.tmux_text)
        self.assertIn("exec bash -i", self.tmux_text)
        self.assertNotIn("nohup", self.tmux_text)
        self.assertNotIn("tmux new-session -d", self.tmux_text)
        self.assertIn('tee -a "${LOG_FILE}"', self.tmux_text)

    def test_start_progress_and_final_contract_are_visible(self) -> None:
        required = (
            "EXPERIMENT_NAME=",
            "GIT_COMMIT=",
            "MODEL=",
            "DATASET=",
            "RESOURCES=CPU only; GPU disabled",
            "TOTAL_WORKLOAD=",
            "ESTIMATED_TOTAL_TIME=",
            "ETA_BASIS=",
            "CACHE_DIRECTORY=",
            "RESULT_DIRECTORY=",
            "LOG_FILE=",
            "PROGRESS stage=",
            "assets=",
            "percent=",
            "current_throughput_mib_s=",
            "stage_eta=",
            "overall_eta=",
            "estimated_finish=",
            "FINAL_RUN_RC=",
            "START_TIME=",
            "END_TIME=",
            "TOTAL_ELAPSED=",
            "COMPLETION_STATUS=",
            "LOG_DIRECTORY=",
            "METRICS_PATH=",
            "FAILED_STAGE=",
            "ERROR_SUMMARY=",
        )
        for marker in required:
            with self.subTest(marker=marker):
                self.assertIn(marker, self.tmux_text)
        self.assertIn("sleep 15", self.tmux_text)

    def test_audit_remains_cpu_only_read_only_and_uniquely_addressable(self) -> None:
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', self.tmux_text)
        self.assertIn("no model or data download", self.tmux_text)
        self.assertIn('RUN_ID="${AUDIT_RUN_ID}"', self.tmux_text)
        self.assertIn(
            'RUN_ID="${RUN_ID:-official_model_resource_audit_',
            self.audit_text,
        )
        self.assertIn('Invalid RUN_ID', self.audit_text)
        for forbidden in ("rm -rf", "snapshot_download", "git reset", "git clean"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.tmux_text)

    def test_nemo3b_ac_workload_matches_presence_evidence(self) -> None:
        self.assertIn("EXPECTED_BYTES=18889955098", self.tmux_text)
        self.assertIn('oea_nemo3b)', self.tmux_text)
        self.assertIn('EXPECTED_ASSETS=2', self.tmux_text)


if __name__ == "__main__":
    unittest.main()
