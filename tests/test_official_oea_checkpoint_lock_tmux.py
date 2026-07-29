from __future__ import annotations

from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPOSITORY_ROOT / "scripts/run_official_oea_checkpoint_lock_tmux.sh"


class OfficialOeaCheckpointLockTmuxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = WRAPPER.read_text(encoding="utf-8")

    def test_requires_visible_tmux_and_retains_interactive_shell(self) -> None:
        self.assertIn('if [[ -z "${TMUX:-}" ]]', self.text)
        self.assertIn("exec bash -i", self.text)
        self.assertIn('tee -a "${LOG_FILE}"', self.text)
        self.assertNotIn("nohup", self.text)
        self.assertNotIn("tmux new-session -d", self.text)

    def test_reuses_exact_resource_evidence_without_resource_reaudit(self) -> None:
        self.assertIn('EXPECTED_AUDIT_SHA256="$3"', self.text)
        self.assertIn('ACTUAL_AUDIT_SHA256="$(sha256sum', self.text)
        self.assertIn("prepare_official_oea_checkpoint.py", self.text)
        self.assertIn("build_official_oea_model_lock.py", self.text)
        self.assertNotIn("audit_official_oea_variant_resources.py", self.text)
        self.assertNotIn("run_official_oea_model_pipeline.py", self.text)

    def test_is_cpu_only_offline_non_overwriting_and_portable(self) -> None:
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', self.text)
        self.assertIn("export HF_HUB_OFFLINE=1", self.text)
        self.assertIn("--verify-existing-derived", self.text)
        self.assertIn('PORTABLE_LOCK="${RUN_DIR}/${VARIANT_ID}.portable_model_lock.json"', self.text)
        self.assertNotIn("results/model_locks", self.text)
        for forbidden in (
            "rm -rf",
            "snapshot_download",
            "hf_hub_download",
            "git reset",
            "git clean",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.text)

    def test_progress_and_final_contract_are_complete(self) -> None:
        required = (
            "EXPERIMENT_NAME=",
            "PAPER_EXPERIMENT_STAGE=",
            "GIT_COMMIT=",
            "MODEL=",
            "DATASET=none",
            "RESOURCES=CPU only; GPU disabled; network disabled",
            "CPU_MODEL=",
            "CPU_COUNT=",
            "MEMORY=",
            "TOTAL_WORKLOAD=",
            "ESTIMATED_TOTAL_TIME=",
            "ETA_BASIS=",
            "CACHE_DIRECTORY=",
            "RESULT_DIRECTORY=",
            "LOG_FILE=",
            "PROGRESS stage=",
            "percent=",
            "stage_elapsed=",
            "overall_elapsed=",
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
                self.assertIn(marker, self.text)
        self.assertIn("sleep 15", self.text)
        self.assertIn("ACCOUNTED < LAST_ACCOUNTED", self.text)

    def test_nemo3b_ac_identity_matches_fixed_registry(self) -> None:
        self.assertIn("OEA-Nemo3B-AC/step_400_best.pt", self.text)
        self.assertIn("OEA-Nemo3B-AC/step_400_best_inference_only.pt", self.text)
        self.assertIn("SOURCE_BYTES=9466826153", self.text)


if __name__ == "__main__":
    unittest.main()
