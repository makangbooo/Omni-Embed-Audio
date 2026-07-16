from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "AudioRetrieval/evaluation/negative_metrics.py"
)
SPEC = importlib.util.spec_from_file_location("oea_negative_metrics", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
negative_metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(negative_metrics)

compute_delta_rank = negative_metrics.compute_delta_rank
compute_hnsr = negative_metrics.compute_hnsr
compute_hnsr_at_k = negative_metrics.compute_hnsr_at_k
compute_negative_query_metrics = negative_metrics.compute_negative_query_metrics
compute_tfr = negative_metrics.compute_tfr
compute_tfr_hn_at_k = negative_metrics.compute_tfr_hn_at_k


class NegativeMetricsTest(unittest.TestCase):
    def test_exact_batch_formulas(self) -> None:
        target = np.array([1, 2, 6, 4])
        hard_negative = np.array([8, 1, 9, 3])
        metrics = compute_negative_query_metrics(target, hard_negative, ks=(1, 5, 10))

        self.assertEqual(metrics["num_queries"], 4)
        self.assertEqual(metrics["R@1"], 25.0)
        self.assertEqual(metrics["R@5"], 75.0)
        self.assertEqual(metrics["R@10"], 100.0)
        self.assertEqual(metrics["Delta-Rank"], 2.0)
        self.assertEqual(metrics["HNSR"], 50.0)
        self.assertEqual(metrics["HNSR@5"], 25.0)
        self.assertEqual(metrics["TFR"], 25.0)
        self.assertEqual(metrics["TFR-HN@5"], 25.0)

    def test_target_moving_forward_improves_metrics(self) -> None:
        hard_negative = [8]
        before_target = [6]
        after_target = [1]

        self.assertLess(
            compute_delta_rank(before_target, hard_negative),
            compute_delta_rank(after_target, hard_negative),
        )
        self.assertLess(
            compute_hnsr_at_k(before_target, hard_negative, 5),
            compute_hnsr_at_k(after_target, hard_negative, 5),
        )
        self.assertLess(compute_tfr(before_target), compute_tfr(after_target))
        self.assertLess(
            compute_tfr_hn_at_k(before_target, hard_negative, 5),
            compute_tfr_hn_at_k(after_target, hard_negative, 5),
        )

    def test_hard_negative_moving_back_improves_suppression(self) -> None:
        target = [2]
        before_hard_negative = [4]
        after_hard_negative = [8]

        self.assertLess(
            compute_delta_rank(target, before_hard_negative),
            compute_delta_rank(target, after_hard_negative),
        )
        self.assertLess(
            compute_hnsr_at_k(target, before_hard_negative, 5),
            compute_hnsr_at_k(target, after_hard_negative, 5),
        )

    def test_hnsr_at_k_requires_target_inside_top_k(self) -> None:
        # The target is above the hard negative, so unconditioned HNSR succeeds,
        # but neither item is in top-5 and HNSR@5 must remain zero.
        self.assertEqual(compute_hnsr([6], [8]), 100.0)
        self.assertEqual(compute_hnsr_at_k([6], [8], 5), 0.0)

    def test_tfr_hn_at_k_requires_both_conditions(self) -> None:
        self.assertEqual(compute_tfr([1]), 100.0)
        self.assertEqual(compute_tfr_hn_at_k([1], [3], 5), 0.0)
        self.assertEqual(compute_tfr_hn_at_k([1], [7], 5), 100.0)
        self.assertEqual(compute_tfr_hn_at_k([2], [7], 5), 0.0)

    def test_equal_ranks_are_not_suppression(self) -> None:
        self.assertEqual(compute_delta_rank([3], [3]), 0.0)
        self.assertEqual(compute_hnsr([3], [3]), 0.0)

    def test_invalid_ranks_and_cutoffs_are_rejected(self) -> None:
        invalid_calls = (
            lambda: compute_hnsr([], []),
            lambda: compute_hnsr([1, 2], [1]),
            lambda: compute_hnsr([0], [1]),
            lambda: compute_hnsr([1.5], [2]),
            lambda: compute_hnsr([np.nan], [2]),
            lambda: compute_hnsr_at_k([1], [2], 0),
            lambda: compute_negative_query_metrics([1], [2], ks=()),
            lambda: compute_negative_query_metrics([1], [2], ks=(5, 5)),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()


if __name__ == "__main__":
    unittest.main()
