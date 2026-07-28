import unittest
from pathlib import Path

import numpy as np

from scripts.diagnose_asrur_oea_collapse import (
    SPACE_NAMES,
    compare_embeddings,
    evaluate_space,
    off_diagonal_summary,
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
