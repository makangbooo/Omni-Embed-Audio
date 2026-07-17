from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_paper_metric_registry import (
    DEFAULT_OUTPUT,
    DEFAULT_SOURCE,
    expand_registry,
    registry_text,
)
from scripts.audit_paper_metric_transcription import (
    line_matches_row,
    table_row_label,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class BuildPaperMetricRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(DEFAULT_SOURCE.read_text(encoding="utf-8"))
        cls.rows = expand_registry(cls.config)
        cls.by_id = {row["paper_metric_id"]: row for row in cls.rows}

    def test_all_paper_tables_are_explicitly_classified(self) -> None:
        numeric = {
            int(table["paper_table"].split()[1])
            for table in self.config["numeric_result_tables"]
        }
        non_metric = {
            int(table["paper_table"].split()[1])
            for table in self.config["non_metric_tables"]
        }
        self.assertEqual(numeric | non_metric, set(range(1, 18)))
        self.assertFalse(numeric & non_metric)
        self.assertEqual(non_metric, {6, 8, 9, 10})

    def test_registry_has_expected_table_and_unit_counts(self) -> None:
        self.assertEqual(len(self.rows), 910)
        self.assertEqual(len(self.by_id), 910)
        self.assertEqual(
            Counter(row["paper_table"] for row in self.rows),
            {
                "Table 1": 24,
                "Table 2": 117,
                "Table 3": 117,
                "Table 4": 60,
                "Table 5": 24,
                "Table 7": 36,
                "Table 11": 78,
                "Table 12": 90,
                "Table 13": 90,
                "Table 14": 90,
                "Table 15": 90,
                "Table 16": 24,
                "Table 17": 70,
            },
        )
        self.assertEqual(
            Counter(row["unit"] for row in self.rows),
            {
                "percent": 792,
                "score_1_to_5": 36,
                "score_standard_deviation": 18,
                "milliseconds": 24,
                "gigabytes": 12,
                "million_parameters": 12,
                "count": 6,
                "rank_gap": 10,
            },
        )

    def test_visually_checked_anchor_values_and_pages(self) -> None:
        expected = {
            "table1_human_9_annotators_uiq_validation_question_validity_mean": (
                "4.26",
                4,
            ),
            "table2_m2d_clap_audiocaps_t2a_r5": ("77.13", 6),
            "table3_oea_qwen3b_cl_clotho_t2t_r1": ("64.52", 7),
            "table4_oea_qwen7b_cl_three_dataset_mean_hard_negative_hnsr10": (
                "34.6",
                8,
            ),
            "table5_oea_nemo3b_clotho_a100_inference_efficiency_text_ms_per_query": (
                "2.30",
                9,
            ),
            "table7_human_9_annotators_mecat_overall_validity_mean": (
                "4.37",
                12,
            ),
            "table11_oea_qwen7b_cl_three_dataset_mean_t2t_r10": (
                "73.21",
                15,
            ),
            "table12_oea_qwen3b_cl_clotho_question_r10": ("66.79", 15),
            "table13_oea_qwen3b_cl_clotho_imperative_r5": ("55.69", 16),
            "table14_oea_qwen3b_cl_clotho_paraphrase_r1": ("27.56", 16),
            "table15_oea_qwen3b_cl_clotho_keyphrase_r5": ("57.99", 17),
            "table16_oea_qwen7b_clotho_a100_inference_efficiency_peak_gpu_gb": (
                "18.3",
                17,
            ),
            "table17_oea_qwen7b_cl_three_dataset_mean_hard_negative_hnsr10": (
                "34.6",
                17,
            ),
        }
        self.assertEqual(
            {
                metric_id: (
                    self.by_id[metric_id]["paper_value"],
                    self.by_id[metric_id]["pdf_page"],
                )
                for metric_id in expected
            },
            expected,
        )

    def test_approximate_memory_values_retain_qualifier(self) -> None:
        qualified = [
            row
            for row in self.rows
            if row.get("paper_value_qualifier") == "approximately"
        ]
        self.assertEqual(len(qualified), 4)
        self.assertEqual({row["paper_value"] for row in qualified}, {"0.6"})
        self.assertEqual(
            {row["paper_table"] for row in qualified}, {"Table 5", "Table 16"}
        )

    def test_generated_registry_is_byte_for_byte_current(self) -> None:
        self.assertEqual(
            DEFAULT_OUTPUT.read_text(encoding="utf-8"), registry_text(self.rows)
        )

    def test_duplicate_table_classification_is_rejected(self) -> None:
        config = json.loads(json.dumps(self.config))
        config["non_metric_tables"].append(
            {
                "paper_table": "Table 1",
                "pdf_page": 4,
                "reason": "synthetic duplicate",
            }
        )
        with self.assertRaisesRegex(ValueError, "classified more than once"):
            expand_registry(config)

    def test_value_count_drift_is_rejected(self) -> None:
        config = json.loads(json.dumps(self.config))
        config["numeric_result_tables"][0]["rows"][0]["values"].pop()
        with self.assertRaisesRegex(ValueError, "must match dimension count"):
            expand_registry(config)

    def test_pdf_line_match_ignores_model_numbers_but_preserves_values(self) -> None:
        line = "OEA-Qwen3B(+Cl) 35.96 69.35 81.99 22.87 49.78 63.25"
        self.assertTrue(
            line_matches_row(
                line,
                "OEA-Qwen3B (+Cl)",
                ["35.96", "69.35", "81.99", "22.87", "49.78", "63.25"],
            )
        )
        self.assertFalse(
            line_matches_row(
                line,
                "OEA-Qwen3B (+Cl)",
                ["35.96", "69.35", "81.99", "22.88", "49.78", "63.25"],
            )
        )

    def test_pdf_row_can_retain_a_distinct_displayed_label(self) -> None:
        self.assertEqual(
            table_row_label(
                {
                    "task": "UIQ Overall validity",
                    "displayed_label": "Mean",
                }
            ),
            "Mean",
        )


if __name__ == "__main__":
    unittest.main()
