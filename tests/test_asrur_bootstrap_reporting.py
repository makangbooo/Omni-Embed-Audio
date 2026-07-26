import unittest

from AudioRetrieval.asr_uncertainty_reranking.bootstrap import paired_bootstrap
from AudioRetrieval.asr_uncertainty_reranking.reporting import (
    summarize_metric_by_wer_bins,
    summarize_seeds,
)


class BootstrapReportingTest(unittest.TestCase):
    def test_paired_bootstrap_is_deterministic_and_directional(self) -> None:
        method = {f"q{i}": 0.8 for i in range(10)}
        baseline = {f"q{i}": 0.5 for i in range(10)}
        first = paired_bootstrap(method, baseline, iterations=200, seed=9)
        second = paired_bootstrap(method, baseline, iterations=200, seed=9)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean_delta"], 0.3)
        self.assertEqual(first["two_sided_p_value"], 0.0)

    def test_seed_and_wer_bin_summaries(self) -> None:
        summary = summarize_seeds([0.5, 0.6, 0.7])
        self.assertAlmostEqual(summary["mean"], 0.6)
        bins = summarize_metric_by_wer_bins(
            {"q1": 0.05, "q2": 0.3},
            {"q1": 0.8, "q2": 0.4},
        )
        self.assertEqual(sum(bucket["count"] for bucket in bins), 2)


if __name__ == "__main__":
    unittest.main()
