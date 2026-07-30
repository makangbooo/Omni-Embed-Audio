from __future__ import annotations

from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts/run_vanilla_clotho_main_tmux.sh"


class VanillaClothoMainTmuxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCRIPT.read_text(encoding="utf-8")

    def test_scope_is_only_remaining_vanilla_qwen7b_main_tables(self) -> None:
        self.assertIn('"$1" != "vanilla_qwen2_5_omni_7b"', self.source)
        self.assertIn("EXP-10 Table 2 T2A; EXP-11 Table 3 T2T", self.source)
        self.assertNotIn("uiq", self.source.lower())

    def test_self_launches_and_retains_tmux(self) -> None:
        self.assertIn('if [[ -z "${TMUX:-}" ]]', self.source)
        self.assertIn("tmux new-session -d", self.source)
        self.assertIn("ATTACH_COMMAND=tmux attach", self.source)
        self.assertIn("exec bash -i", self.source)

    def test_runs_smoke_full_and_cpu_suite_with_base_only_lock(self) -> None:
        self.assertEqual(
            self.source.count("scripts/run_vanilla_backbone_embeddings.sh"), 2
        )
        self.assertIn("scripts/run_vanilla_clotho_retrieval_suite.sh", self.source)
        self.assertIn("results/model_locks/vanilla_qwen2_5_omni_7b.json", self.source)
        self.assertIn("oea_checkpoint_loaded=false", self.source)
        self.assertIn("OEA_OFFICIAL_SOURCE_USED=yes", self.source)

    def test_final_report_and_artifact_hashes_are_mandatory(self) -> None:
        for field in (
            "FINAL_RUN_RC=",
            "START_TIME=",
            "END_TIME=",
            "TOTAL_ELAPSED_SECONDS=",
            "COMPLETION_STATUS=",
            "FAILED_STAGE=",
            "ERROR_SUMMARY=",
            "RESULT_DIRECTORY=",
            "METRICS_PATH=",
        ):
            self.assertIn(field, self.source)
        self.assertIn('"${SUITE_DIR}/retrieval_summary.csv"', self.source)
        self.assertNotIn("rm -rf", self.source)


if __name__ == "__main__":
    unittest.main()
