import math
import unittest

from AudioRetrieval.asr_uncertainty_reranking.metrics import (
    corpus_wer,
    evaluate_candidate_oracle,
    evaluate_rankings,
    ndcg_at_k,
    noise_degradation,
    oracle_ranking,
    per_query_retrieval_metrics,
    word_error_counts,
)


class RetrievalMetricTest(unittest.TestCase):
    def test_single_query_graded_metrics(self) -> None:
        qrels = {"d1": 2, "d2": 1}
        ranking = ["d2", "x", "d1"]
        metrics = per_query_retrieval_metrics(
            ranking,
            qrels,
            recall_ks=(1, 2, 3),
        )
        ideal = 3.0 + 1.0 / math.log2(3.0)
        actual = 1.0 + 3.0 / math.log2(4.0)
        self.assertAlmostEqual(metrics["nDCG@10"], actual / ideal)
        self.assertEqual(metrics["MRR@10"], 1.0)
        self.assertEqual(metrics["Recall@1"], 0.5)
        self.assertEqual(metrics["Recall@2"], 0.5)
        self.assertEqual(metrics["Recall@3"], 1.0)

    def test_mean_metrics_are_fractions_and_include_per_query_evidence(self) -> None:
        result = evaluate_rankings(
            {"q1": ["a", "b"], "q2": ["x", "y"]},
            {"q1": {"a": 1}, "q2": {"y": 1}},
            recall_ks=(1, 2),
        )
        self.assertEqual(result["num_queries"], 2)
        self.assertEqual(result["scale"], "fraction")
        self.assertEqual(result["mean"]["MRR@10"], 0.75)
        self.assertEqual(result["mean"]["Recall@1"], 0.5)
        self.assertEqual(result["mean"]["Recall@2"], 1.0)
        self.assertEqual(set(result["per_query"]), {"q1", "q2"})

    def test_query_set_mismatch_duplicate_docs_and_empty_qrels_are_rejected(self) -> None:
        invalid_calls = (
            lambda: evaluate_rankings({"q": ["d"]}, {}),
            lambda: evaluate_rankings({"q": ["d", "d"]}, {"q": {"d": 1}}),
            lambda: evaluate_rankings({"q": ["d"]}, {"q": {"d": 0}}),
            lambda: evaluate_rankings(
                {"q": ["d"]},
                {"q": {"d": 1}, "extra": {"x": 1}},
            ),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises((TypeError, ValueError)):
                    call()

    def test_ndcg_uses_global_ideal_relevance(self) -> None:
        # A fixed candidate list missing the grade-2 document cannot reach 1.0.
        self.assertLess(ndcg_at_k(["low"], {"high": 2, "low": 1}, k=10), 1.0)


class CandidateOracleTest(unittest.TestCase):
    def test_oracle_changes_order_but_never_membership(self) -> None:
        candidates = ["n1", "r1", "n2", "r2"]
        qrels = {"r1": 1, "r2": 2}
        oracle = oracle_ranking(candidates, qrels)
        self.assertEqual(oracle, ["r2", "r1", "n1", "n2"])
        self.assertEqual(set(oracle), set(candidates))

    def test_candidate_oracle_reports_recall_ceiling(self) -> None:
        result = evaluate_candidate_oracle(
            {"q": ["r1", "n"]},
            {"q": {"r1": 1, "missing": 1}},
            recall_ks=(1, 2),
        )
        self.assertEqual(result["mean"]["Recall@1"], 0.5)
        self.assertEqual(result["mean"]["Recall@2"], 0.5)
        self.assertLess(result["mean"]["nDCG@10"], 1.0)


class NoiseAndWERTest(unittest.TestCase):
    def test_noise_drop_is_signed_and_relative(self) -> None:
        degradation = noise_degradation(
            {"nDCG@10": 0.4, "Recall@100": 0.0},
            {"nDCG@10": 0.3, "Recall@100": 0.1},
        )
        self.assertAlmostEqual(degradation["nDCG@10"]["absolute_drop"], 0.1)
        self.assertAlmostEqual(degradation["nDCG@10"]["relative_drop"], 0.25)
        self.assertAlmostEqual(degradation["Recall@100"]["absolute_drop"], -0.1)
        self.assertIsNone(degradation["Recall@100"]["relative_drop"])

    def test_word_error_counts_and_micro_corpus_wer(self) -> None:
        counts = word_error_counts("a b c", "a x c d")
        self.assertEqual(counts.substitutions, 1)
        self.assertEqual(counts.insertions, 1)
        self.assertEqual(counts.deletions, 0)
        self.assertEqual(counts.errors, 2)
        self.assertAlmostEqual(counts.wer, 2.0 / 3.0)

        result = corpus_wer([("a b", "a"), ("c", "x")])
        self.assertEqual(result["num_utterances"], 2)
        self.assertEqual(result["errors"], 2)
        self.assertEqual(result["reference_words"], 3)
        self.assertAlmostEqual(result["WER"], 2.0 / 3.0)

    def test_all_empty_corpus_references_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            corpus_wer([("", "")])


if __name__ == "__main__":
    unittest.main()
