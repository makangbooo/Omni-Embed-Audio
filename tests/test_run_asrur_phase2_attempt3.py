import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts/run_asrur_phase2_attempt3.sh"


class RunASRURPhase2Attempt3Test(unittest.TestCase):
    def test_attempt3_is_tmux_only_and_requires_all_reuses(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('[[ -z "${TMUX:-}" ]]', source)
        self.assertIn("tmux new -s asrur_phase2_attempt3", source)
        self.assertIn(
            "/home/jg525/experiment_cache/asr_uncertainty/"
            "fiqa_phase2_ee637721c1f7",
            source,
        )
        self.assertIn("export PHASE2_REQUIRED_REUSE_COUNT=11", source)
        self.assertIn("fiqa_phase2_${COMMIT_SHORT}", source)
        self.assertIn(
            "exec bash scripts/run_asrur_phase2_frozen_retrieval.sh --execute",
            source,
        )
        self.assertNotIn("nohup", source)
        self.assertNotIn("rm -", source)

    def test_attempt3_uses_new_result_identity_and_single_visible_gpu(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"', source)
        self.assertIn("asrur_phase2_fiqa_${COMMIT_SHORT}", source)
        self.assertIn("same commit and same cache root", source)


if __name__ == "__main__":
    unittest.main()
