import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class RunASRURPhase3UnselectedCETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = (
            REPOSITORY_ROOT / "scripts/run_asrur_phase3_unselected_ce.sh"
        ).read_text(encoding="utf-8")

    def test_requires_go_audit_and_never_selects_on_test(self) -> None:
        self.assertIn('value.get("overall_decision") != "GO"', self.text)
        self.assertIn(
            'value.get("candidate_generation_change_authorized") is not False',
            self.text,
        )
        self.assertNotIn("select_asrur_dev_parameters.py", self.text)
        self.assertIn("evaluate_asrur_unselected_ce_baselines.py", self.text)

    def test_scores_one_shared_four_best_matrix_per_condition(self) -> None:
        self.assertIn("CONDITIONS=(clean snr_20 snr_10 snr_0)", self.text)
        self.assertIn("--expected-hypotheses 4", self.text)
        self.assertIn('asr_pair_count=%s', self.text)
        self.assertIn("$((4 * 648 * 100 * 4))", self.text)
        self.assertIn('CE_ROOT="${PHASE3_CACHE_ROOT}/ce/${condition}"', self.text)

    def test_gold_upper_bound_uses_single_hypothesis_same_candidates(self) -> None:
        self.assertIn("build_asrur_gold_nbest.py", self.text)
        self.assertIn("--gold-cross-encoder", self.text)
        self.assertIn("--expected-hypotheses 1", self.text)
        self.assertIn(
            '--candidates "${PHASE2_CACHE_ROOT}/rankings/oea/${condition}.jsonl"',
            self.text,
        )
        self.assertIn('"U2_gold_ce"', self.text)

    def test_gpu_execution_is_separately_guarded_and_resumable(self) -> None:
        self.assertIn("--execute requires a separate user GPU approval", self.text)
        self.assertIn('GPU_NAME}" != *"RTX 4090"*', self.text)
        self.assertIn("caches were preserved", self.text)
        self.assertIn("Reusing existing unselected metrics", self.text)
        self.assertIn("HF_HUB_OFFLINE=1", self.text)


if __name__ == "__main__":
    unittest.main()
