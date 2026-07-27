import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts/run_asrur_phase2_attempt4.sh"


class RunASRURPhase2Attempt4Test(unittest.TestCase):
    def test_attempt4_is_tmux_only_and_recovers_exact_attempt3_state(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('[[ -z "${TMUX:-}" ]]', source)
        self.assertIn("tmux new -s asrur_phase2_attempt4", source)
        self.assertIn(
            "bc4450a2bb680313c3cee7085974d246a042d30c",
            source,
        )
        self.assertIn("fiqa_phase2_bc4450a2bb68", source)
        self.assertIn("export PHASE2_REQUIRED_REUSE_COUNT=14", source)
        self.assertIn("export ASRUR_WHISPER_MAX_RECORD_ATTEMPTS=3", source)
        self.assertIn("PHASE2_WHISPER_RESUME_SOURCE_ROOT", source)
        self.assertIn("PHASE2_WHISPER_RESUME_SOURCE_GIT_COMMIT", source)
        self.assertIn("fiqa_phase2_${COMMIT_SHORT}", source)
        self.assertIn(
            "exec bash scripts/run_asrur_phase2_frozen_retrieval.sh --execute",
            source,
        )
        self.assertNotIn("nohup", source)
        self.assertNotIn("rm -", source)
        self.assertIn(
            "prohibited=filtering, clamping, score replacement, protocol fallback",
            source,
        )


if __name__ == "__main__":
    unittest.main()
