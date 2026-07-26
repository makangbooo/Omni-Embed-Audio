import math
import unittest

from AudioRetrieval.asr_uncertainty_reranking.normalization import (
    rank_normalize_scores,
    sort_scored_items,
    zscore_scores,
)


class QueryLocalNormalizationTest(unittest.TestCase):
    def test_zscore_uses_population_statistics(self) -> None:
        actual = zscore_scores([1.0, 2.0, 3.0])
        scale = math.sqrt(1.5)
        for observed, expected in zip(actual, [-scale, 0.0, scale]):
            self.assertAlmostEqual(observed, expected)
        self.assertAlmostEqual(sum(actual), 0.0)

    def test_degenerate_zscore_route_is_neutral(self) -> None:
        self.assertEqual(zscore_scores([4.0, 4.0, 4.0]), [0.0, 0.0, 0.0])
        self.assertEqual(
            zscore_scores([1.0, 1.0 + 1e-14], epsilon=1e-12),
            [0.0, 0.0],
        )

    def test_rank_normalization_uses_average_ranks_for_ties(self) -> None:
        self.assertEqual(
            rank_normalize_scores([10.0, 5.0, 5.0, 0.0]),
            [1.0, 0.5, 0.5, 0.0],
        )
        self.assertEqual(rank_normalize_scores([7.0]), [1.0])

    def test_rank_normalization_is_invariant_to_tie_order(self) -> None:
        first = rank_normalize_scores([2.0, 2.0, 1.0])
        second = rank_normalize_scores([2.0, 1.0, 2.0])
        self.assertEqual(first, [0.75, 0.75, 0.0])
        self.assertEqual(second, [0.75, 0.0, 0.75])

    def test_scored_item_sort_has_deterministic_tie_break(self) -> None:
        self.assertEqual(
            sort_scored_items([("b", 1.0), ("a", 1.0), ("c", 2.0)]),
            [("c", 2.0), ("a", 1.0), ("b", 1.0)],
        )
        self.assertEqual(
            sort_scored_items([("1", 1.0), (1, 1.0)]),
            [(1, 1.0), ("1", 1.0)],
        )

    def test_invalid_scores_and_duplicate_ids_are_rejected(self) -> None:
        invalid_calls = (
            lambda: zscore_scores([]),
            lambda: zscore_scores([True]),
            lambda: zscore_scores([float("nan")]),
            lambda: zscore_scores([1.0], epsilon=-1.0),
            lambda: rank_normalize_scores([float("inf")]),
            lambda: sort_scored_items([]),
            lambda: sort_scored_items([("a", 1.0), ("a", 2.0)]),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises((TypeError, ValueError)):
                    call()


if __name__ == "__main__":
    unittest.main()
