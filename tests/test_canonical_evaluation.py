from __future__ import annotations

import unittest

import numpy as np

from AudioRetrieval.evaluation.canonical import (
    evaluate_caption_to_caption,
    evaluate_grouped_id_retrieval,
    evaluate_id_retrieval,
    evaluate_query_to_candidates,
)


class CanonicalEvaluationTest(unittest.TestCase):
    def test_a2t_accepts_all_captions_of_the_target_clip(self) -> None:
        audio = np.eye(2, dtype=np.float32)
        captions = np.asarray(
            [[0.8, 0.2], [1.0, 0.0], [0.0, 1.0], [0.2, 0.8]],
            dtype=np.float32,
        )
        result = evaluate_grouped_id_retrieval(
            audio,
            ["a", "b"],
            captions,
            ["a", "a", "b", "b"],
        )

        self.assertEqual(result.ranks.tolist(), [1, 1])
        self.assertEqual(result.positive_indices, ((0, 1), (2, 3)))
        self.assertEqual(result.metrics["R@1"], 100.0)

    def test_a2t_rejects_missing_caption_group(self) -> None:
        with self.assertRaisesRegex(KeyError, "absent from candidate groups"):
            evaluate_grouped_id_retrieval(
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                ["missing"],
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                ["present"],
            )

    def test_t2a_uses_every_explicit_caption_query(self) -> None:
        audio = np.eye(3, dtype=np.float32)
        captions = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.9, 0.1],
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 0.9],
            ],
            dtype=np.float32,
        )
        result = evaluate_id_retrieval(
            captions,
            ["a", "a", "b", "b", "c", "c"],
            audio,
            ["a", "b", "c"],
        )

        np.testing.assert_array_equal(result.ranks, np.ones(6, dtype=np.int64))
        self.assertEqual(result.metrics["R@1"], 100.0)
        self.assertEqual(result.evaluated_query_indices.tolist(), list(range(6)))

    def test_t2t_excludes_self_and_accepts_any_sister_caption(self) -> None:
        captions = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        )
        result = evaluate_caption_to_caption(captions, ["a", "a", "b", "b"])

        np.testing.assert_array_equal(result.ranks, np.ones(4, dtype=np.int64))
        self.assertEqual(result.positive_indices, ((1,), (0,), (3,), (2,)))
        self.assertEqual(result.ignored_indices, ((0,), (1,), (2,), (3,)))
        self.assertEqual(result.rankings[0, 0], 1)

    def test_t2t_query_subset_is_explicit_and_ordered(self) -> None:
        captions = np.array(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
            dtype=np.float32,
        )
        result = evaluate_caption_to_caption(
            captions,
            ["a", "a", "b", "b"],
            query_indices=[2, 0],
        )

        self.assertEqual(result.evaluated_query_indices.tolist(), [2, 0])
        self.assertEqual(result.ranks.tolist(), [1, 1])

    def test_multiple_positives_use_the_best_positive_rank(self) -> None:
        result = evaluate_query_to_candidates(
            np.array([[1.0, 0.0]], dtype=np.float32),
            np.array(
                [[0.8, 0.2], [0.2, 0.8], [1.0, 0.0]], dtype=np.float32
            ),
            [(0, 2)],
        )

        self.assertEqual(result.ranks.tolist(), [1])
        self.assertEqual(result.positive_indices, ((0, 2),))

    def test_optimistic_tie_policy_matches_public_code(self) -> None:
        result = evaluate_query_to_candidates(
            np.array([[1.0, 0.0]], dtype=np.float32),
            np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            [(1,)],
        )

        self.assertEqual(result.ranks.tolist(), [1])
        self.assertEqual(result.tie_policy, "optimistic_strict_greater")
        self.assertEqual(result.rankings.tolist(), [[0, 1]])

    def test_missing_target_id_fails_instead_of_silently_skipping(self) -> None:
        with self.assertRaisesRegex(KeyError, "absent from candidates"):
            evaluate_id_retrieval(
                np.array([[1.0, 0.0]], dtype=np.float32),
                ["missing"],
                np.array([[1.0, 0.0]], dtype=np.float32),
                ["present"],
            )

    def test_duplicate_candidate_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be unique"):
            evaluate_id_retrieval(
                np.array([[1.0, 0.0]], dtype=np.float32),
                ["a"],
                np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                ["a", "a"],
            )

    def test_zero_norm_embeddings_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero-norm"):
            evaluate_query_to_candidates(
                np.array([[0.0, 0.0]], dtype=np.float32),
                np.array([[1.0, 0.0]], dtype=np.float32),
                [(0,)],
            )


if __name__ == "__main__":
    unittest.main()
