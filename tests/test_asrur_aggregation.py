import math
import unittest

from AudioRetrieval.asr_uncertainty_reranking.aggregation import (
    aggregate_nbest_scores,
    build_asr_uncertainty_features,
    logsumexp,
    normalized_entropy,
    normalized_word_edit_distance,
    pairwise_edit_distance_stats,
    softmax_proxy_posteriors,
    word_levenshtein_distance,
)


class ASRProxyPosteriorTest(unittest.TestCase):
    def test_softmax_is_stable_and_temperature_controlled(self) -> None:
        self.assertEqual(softmax_proxy_posteriors([1000.0, 1000.0]), [0.5, 0.5])
        cold = softmax_proxy_posteriors([1.0, 0.0], temperature=0.5)
        warm = softmax_proxy_posteriors([1.0, 0.0], temperature=2.0)
        self.assertGreater(cold[0], warm[0])
        self.assertAlmostEqual(sum(cold), 1.0)
        self.assertAlmostEqual(sum(warm), 1.0)

    def test_normalized_entropy_boundaries(self) -> None:
        self.assertAlmostEqual(normalized_entropy([0.25] * 4), 1.0)
        self.assertEqual(normalized_entropy([1.0, 0.0, 0.0, 0.0]), 0.0)
        self.assertEqual(normalized_entropy([1.0]), 0.0)

    def test_logsumexp_handles_negative_infinity(self) -> None:
        self.assertAlmostEqual(logsumexp([0.0, 0.0]), math.log(2.0))
        self.assertEqual(logsumexp([-math.inf, -math.inf]), -math.inf)
        self.assertEqual(logsumexp([0.0, -math.inf]), 0.0)

    def test_invalid_probability_inputs_are_rejected(self) -> None:
        invalid_calls = (
            lambda: softmax_proxy_posteriors([]),
            lambda: softmax_proxy_posteriors([0.0], temperature=0.0),
            lambda: softmax_proxy_posteriors([float("nan")]),
            lambda: normalized_entropy([0.6, 0.6]),
            lambda: normalized_entropy([-0.1, 1.1]),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises((TypeError, ValueError)):
                    call()


class NBestEditAndFeatureTest(unittest.TestCase):
    def test_word_edit_distance_is_casefolded_and_length_normalized(self) -> None:
        self.assertEqual(word_levenshtein_distance("The CAT", "the cat"), 0)
        self.assertEqual(word_levenshtein_distance("a b c", "a x c"), 1)
        self.assertAlmostEqual(
            normalized_word_edit_distance("a b c", "a x"),
            2.0 / 3.0,
        )
        self.assertEqual(normalized_word_edit_distance("", ""), 0.0)

    def test_pairwise_stats_cover_every_pair(self) -> None:
        mean, maximum = pairwise_edit_distance_stats(["a b", "a c", "x c"])
        expected = [
            normalized_word_edit_distance("a b", "a c"),
            normalized_word_edit_distance("a b", "x c"),
            normalized_word_edit_distance("a c", "x c"),
        ]
        self.assertAlmostEqual(mean, sum(expected) / len(expected))
        self.assertEqual(maximum, max(expected))
        self.assertEqual(pairwise_edit_distance_stats(["only"]), (0.0, 0.0))

    def test_feature_builder_records_proxy_uncertainty_and_missingness(self) -> None:
        features = build_asr_uncertainty_features(
            ["green bond", "green bonds", "clean bond", "green bond"],
            [-0.1, -0.2, -1.0, -0.3],
        )
        self.assertEqual(features.nbest_size, 4)
        self.assertEqual(features.top1_average_token_logprob, -0.1)
        self.assertGreater(features.normalized_nbest_entropy, 0.0)
        self.assertGreater(features.top1_top2_proxy_margin, 0.0)
        self.assertGreater(features.hypothesis_edit_distance_max, 0.0)
        self.assertIsNone(features.no_speech_probability)
        self.assertTrue(features.no_speech_probability_missing)

        with_no_speech = build_asr_uncertainty_features(
            ["query"],
            [-0.2],
            no_speech_probability=0.25,
        )
        self.assertEqual(with_no_speech.top1_top2_proxy_margin, 1.0)
        self.assertEqual(with_no_speech.no_speech_probability, 0.25)
        self.assertFalse(with_no_speech.no_speech_probability_missing)


class NBestAggregationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scores = [[0.0, 2.0], [0.0, 0.0]]

    def test_one_best_equal_mean_and_max(self) -> None:
        self.assertEqual(
            aggregate_nbest_scores(self.scores, mode="one_best"),
            [0.0, 2.0],
        )
        self.assertEqual(
            aggregate_nbest_scores(self.scores, mode="equal_mean"),
            [0.0, 1.0],
        )
        self.assertEqual(
            aggregate_nbest_scores(self.scores, mode="max"),
            [0.0, 2.0],
        )

    def test_proxy_posterior_logsumexp_matches_definition(self) -> None:
        actual = aggregate_nbest_scores(
            self.scores,
            mode="proxy_posterior_logsumexp",
            proxy_posteriors=[0.75, 0.25],
        )
        self.assertAlmostEqual(actual[0], 0.0)
        self.assertAlmostEqual(actual[1], math.log(0.75 * math.exp(2.0) + 0.25))

    def test_zero_probability_hypothesis_is_ignored(self) -> None:
        actual = aggregate_nbest_scores(
            [[1.0], [100.0]],
            mode="proxy_posterior_logsumexp",
            proxy_posteriors=[1.0, 0.0],
        )
        self.assertAlmostEqual(actual[0], 1.0)

    def test_invalid_matrices_modes_and_probabilities_are_rejected(self) -> None:
        invalid_calls = (
            lambda: aggregate_nbest_scores([], mode="one_best"),
            lambda: aggregate_nbest_scores([[]], mode="one_best"),
            lambda: aggregate_nbest_scores([[1.0], [1.0, 2.0]], mode="max"),
            lambda: aggregate_nbest_scores([[float("nan")]], mode="max"),
            lambda: aggregate_nbest_scores([[1.0]], mode="unknown"),
            lambda: aggregate_nbest_scores(
                [[1.0]],
                mode="proxy_posterior_logsumexp",
            ),
            lambda: aggregate_nbest_scores(
                [[1.0], [2.0]],
                mode="proxy_posterior_logsumexp",
                proxy_posteriors=[1.0],
            ),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises((TypeError, ValueError)):
                    call()


if __name__ == "__main__":
    unittest.main()
