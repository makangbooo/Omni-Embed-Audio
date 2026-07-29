from __future__ import annotations

from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts/run_official_source_oea_clotho_main_tmux.sh"


class OfficialSourceOeaClothoMainTmuxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SCRIPT.read_text(encoding="utf-8")

    def test_scope_is_only_remaining_qwen7b_main_table_variants(self) -> None:
        self.assertIn("oea_qwen7b)", self.source)
        self.assertIn("oea_qwen7b_cl)", self.source)
        self.assertNotIn("uiq-embeddings", self.source)
        self.assertIn("EXP-10 Table 2 T2A; EXP-11 Table 3 T2T", self.source)

    def test_uses_official_source_and_local_resources(self) -> None:
        self.assertIn("python examples/encode_example.py", self.source)
        self.assertIn("python -m AudioRetrieval preprocess embeddings", self.source)
        self.assertIn("--local-path \"${BASE_MODEL_DIR}\"", self.source)
        self.assertIn("HF_HUB_OFFLINE=1", self.source)
        self.assertIn("OEA_OFFICIAL_SOURCE_USED=yes", self.source)

    def test_tmux_logging_and_terminal_summary_are_mandatory(self) -> None:
        self.assertIn('if [[ -z "${TMUX:-}" ]]', self.source)
        self.assertIn('tee -a "${LOG_FILE}"', self.source)
        for field in (
            "FINAL_RUN_RC=",
            "START_TIME=",
            "END_TIME=",
            "TOTAL_ELAPSED=",
            "COMPLETION_STATUS=",
            "RESULT_DIRECTORY=",
            "LOG_DIRECTORY=",
            "METRICS_PATH=",
            "FAILED_STAGE=",
            "ERROR_SUMMARY=",
        ):
            self.assertIn(field, self.source)
        self.assertIn("exec bash -i", self.source)

    def test_main_table_finalizer_does_not_pass_uiq(self) -> None:
        finalizer = self.source.split(
            "python scripts/evaluate_official_source_oea_clotho.py", 1
        )[1]
        self.assertNotIn("--uiq-dir", finalizer)
        self.assertIn("--seed 0", finalizer)


if __name__ == "__main__":
    unittest.main()
