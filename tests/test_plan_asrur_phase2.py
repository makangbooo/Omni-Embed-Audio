import unittest

from scripts.plan_asrur_phase2 import phase2_workload


class PlanASRURPhase2Test(unittest.TestCase):
    def test_formal_fiqa_workload_counts_every_condition_and_hypothesis(self) -> None:
        values = phase2_workload(
            document_count=57_638,
            dev_query_count=500,
            test_query_count=648,
            condition_count=4,
            candidate_depth=100,
            nbest_size=4,
        )
        self.assertEqual(values["audio_query_count"], 2_592)
        self.assertEqual(values["whisper_hypothesis_count"], 10_368)
        self.assertEqual(values["phase3_one_best_ce_pair_count"], 259_200)
        self.assertEqual(values["phase3_four_best_ce_pair_count"], 1_036_800)
        self.assertEqual(
            values["oea_dense_dot_product_count"],
            2_592 * 57_638,
        )

    def test_invalid_workload_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            phase2_workload(
                document_count=0,
                dev_query_count=500,
                test_query_count=648,
                condition_count=4,
                candidate_depth=100,
                nbest_size=4,
            )


if __name__ == "__main__":
    unittest.main()
