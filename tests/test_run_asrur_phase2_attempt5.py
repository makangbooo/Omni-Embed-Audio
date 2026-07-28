import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts/run_asrur_phase2_attempt5.sh"


class RunASRURPhase2Attempt5Test(unittest.TestCase):
    def test_attempt5_reuses_no_whisper_artifact(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('[[ -z "${TMUX:-}" ]]', source)
        self.assertIn("tmux new -s asrur_phase2_attempt5", source)
        self.assertIn("fiqa_phase2_d9baf226c075", source)
        self.assertIn("export PHASE2_REUSE_WHISPER=0", source)
        self.assertIn("export PHASE2_REQUIRED_REUSE_COUNT=11", source)
        self.assertIn("unset PHASE2_WHISPER_RESUME_SOURCE_ROOT", source)
        self.assertIn("unset PHASE2_WHISPER_RESUME_SOURCE_GIT_COMMIT", source)
        self.assertIn("invalid_whisper_policy=preserve_as_failure_evidence_never_reuse", source)
        self.assertIn(
            "exec bash scripts/run_asrur_phase2_frozen_retrieval.sh --execute",
            source,
        )
        self.assertNotIn("nohup", source)
        self.assertNotIn("rm -", source)


if __name__ == "__main__":
    unittest.main()
