import unittest

from scripts.summarize_cgp_oea_gate import paired_bootstrap


class SummarizeCGPOEAGateTest(unittest.TestCase):
    def test_paired_bootstrap_detects_positive_improvement(self) -> None:
        report = paired_bootstrap(
            [1.0] * 20,
            [0.0] * 20,
            seed=5,
            iterations=200,
        )
        self.assertEqual(report["observed_delta"], 1.0)
        self.assertEqual(report["confidence_interval_95"], [1.0, 1.0])

    def test_paired_bootstrap_is_deterministic(self) -> None:
        first = paired_bootstrap(
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            seed=9,
            iterations=200,
        )
        second = paired_bootstrap(
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            seed=9,
            iterations=200,
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
