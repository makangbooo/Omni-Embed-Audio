import unittest
from pathlib import Path

import numpy as np

from scripts.diagnose_asrur_oea_collapse import (
    SPACE_NAMES,
    compare_embeddings,
    compare_spaces,
    evaluate_space,
    factorial_component_effects,
    off_diagonal_summary,
    paired_bootstrap_delta,
)


class DiagnoseASRUROEACollapseTest(unittest.TestCase):
    def test_space_names_lock_component_attribution(self) -> None:
        self.assertEqual(
            SPACE_NAMES,
            (
                "base_hidden",
                "lora_hidden",
                "base_plus_oea_heads",
                "lora_plus_oea_heads",
            ),
        )

    def test_off_diagonal_summary_detects_collapse(self) -> None:
        values = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        report = off_diagonal_summary(values)
        self.assertEqual(report["mean"], 1.0)
        self.assertEqual(report["minimum"], 1.0)

    def test_compare_embeddings_accepts_identical_normalized_rows(self) -> None:
        values = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        report = compare_embeddings(values, values.copy())
        self.assertAlmostEqual(report["mean_row_cosine"], 1.0)
        self.assertAlmostEqual(report["minimum_row_cosine"], 1.0)
        self.assertAlmostEqual(report["maximum_absolute_difference"], 0.0)

    def test_evaluate_space_reports_positive_rank_and_recall(self) -> None:
        audio = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        documents = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            dtype=np.float32,
        )
        report = evaluate_space(
            audio=audio,
            documents=documents,
            query_ids=["q1", "q2"],
            document_ids=["d1", "d2", "d3"],
            qrels={"q1": {"d1": 1.0}, "q2": {"d2": 1.0}},
        )
        self.assertEqual(report["query_recall"]["Recall@1"], 1.0)
        self.assertEqual(report["unique_top1_document_count"], 2)
        self.assertEqual(report["best_positive_rank_maximum"], 1)
        self.assertEqual(len(report["per_query"]), 2)
        self.assertEqual(report["mean_reciprocal_rank"], 1.0)

    def test_paired_bootstrap_delta_is_deterministic(self) -> None:
        first = paired_bootstrap_delta(
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            iterations=500,
            seed=7,
        )
        second = paired_bootstrap_delta(
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            iterations=500,
            seed=7,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["observed_mean_delta"], 0.75)
        self.assertGreaterEqual(first["confidence_interval_95"][0], 0.0)

    def test_compare_spaces_reports_paired_recall_effects(self) -> None:
        audio = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        good_documents = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
        )
        bad_documents = np.asarray(
            [[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]], dtype=np.float32
        )
        kwargs = {
            "audio": audio,
            "query_ids": ["q1", "q2"],
            "document_ids": ["d1", "d2", "d3"],
            "qrels": {"q1": {"d1": 1.0}, "q2": {"d2": 1.0}},
        }
        evaluations = {
            "good": evaluate_space(documents=good_documents, **kwargs),
            "bad": evaluate_space(documents=bad_documents, **kwargs),
        }
        report = compare_spaces(
            evaluations,
            baseline="good",
            method="bad",
            iterations=200,
            seed=11,
        )
        self.assertLess(
            report["metrics"]["Recall@1"]["observed_mean_delta"], 0.0
        )

    def test_factorial_effects_separate_lora_and_heads(self) -> None:
        query_ids = [f"q{index}" for index in range(4)]

        def evaluation(values: list[float]) -> dict:
            return {
                "per_query": [
                    {
                        "query_id": query_id,
                        "hit_at_1": value,
                        "hit_at_5": value,
                        "hit_at_10": value,
                        "reciprocal_rank": value,
                    }
                    for query_id, value in zip(query_ids, values)
                ]
            }

        report = factorial_component_effects(
            {
                "base_hidden": evaluation([1.0] * 4),
                "lora_hidden": evaluation([0.5] * 4),
                "base_plus_oea_heads": evaluation([0.8] * 4),
                "lora_plus_oea_heads": evaluation([0.3] * 4),
            },
            iterations=200,
            seed=9,
        )
        self.assertAlmostEqual(
            report["MRR"]["lora_main_effect"]["observed_mean_delta"], -0.5
        )
        self.assertAlmostEqual(
            report["MRR"]["projection_head_main_effect"]["observed_mean_delta"],
            -0.2,
        )
        self.assertEqual(
            report["predeclared_primary_decision"]["dominant_component"], "lora"
        )


class OEACollapseWrapperTest(unittest.TestCase):
    def test_wrapper_is_tmux_only_offline_and_nonmutating(self) -> None:
        source = Path(
            "scripts/run_asrur_oea_collapse_diagnostic.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('[[ -z "${TMUX:-}" ]]', source)
        self.assertIn("tmux new -s asrur_oea_collapse_diag", source)
        self.assertIn("HF_HUB_OFFLINE=1", source)
        self.assertIn("TRANSFORMERS_OFFLINE=1", source)
        self.assertIn("formal_cache_mutation=disabled", source)
        self.assertIn("checkpoint_selection=disabled", source)
        self.assertIn("training=disabled", source)
        self.assertNotIn("nohup", source)


if __name__ == "__main__":
    unittest.main()
