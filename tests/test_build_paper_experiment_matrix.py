from __future__ import annotations

from collections import Counter
import unittest

from scripts.build_paper_experiment_matrix import (
    DEFAULT_OBSERVATIONS,
    DEFAULT_PAPER_REGISTRY,
    build_matrix,
    read_jsonl,
)


class BuildPaperExperimentMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = build_matrix(
            read_jsonl(DEFAULT_PAPER_REGISTRY), read_jsonl(DEFAULT_OBSERVATIONS)
        )
        cls.by_key = {
            (
                row["inventory_id"],
                row["paper_table"],
                row["model"],
                row["dataset"],
                row["task"],
            ): row
            for row in cls.rows
        }

    def test_every_transcribed_metric_is_covered_once(self) -> None:
        self.assertEqual(sum(row["paper_metric_count"] for row in self.rows), 910)
        self.assertEqual(len(self.rows), len(self.by_key))
        self.assertEqual({row["paper_status"] for row in self.rows}, {"PAPER"})

    def test_committed_close_cells_are_not_hidden_by_strict_blockers(self) -> None:
        for model in ("OEA-Qwen3B", "OEA-Qwen3B (+Cl)"):
            key = ("EXP-10", "Table 2", model, "Clotho", "T2A")
            self.assertEqual(
                self.by_key[key]["reproduction_status"], "REPRODUCED_CLOSE"
            )
        nemo = (
            "EXP-10",
            "Table 2",
            "OEA-Nemo3B (+Cl)",
            "Clotho",
            "T2A",
        )
        self.assertEqual(self.by_key[nemo]["observed_metric_count"], 3)

    def test_nemo_t2t_and_clotho_positive_uiq_are_reproduced(self) -> None:
        t2t = (
            "EXP-11",
            "Table 3",
            "OEA-Nemo3B (+Cl)",
            "Clotho",
            "T2T",
        )
        self.assertEqual(
            self.by_key[t2t]["reproduction_status"], "REPRODUCED_CLOSE"
        )
        self.assertEqual(self.by_key[t2t]["observed_metric_count"], 3)
        uiq_tasks = {
            "EXP-12": ("Table 12", "UIQ Question T2A"),
            "EXP-13": ("Table 13", "UIQ Imperative T2A"),
            "EXP-14": ("Table 14", "UIQ Paraphrase T2A"),
            "EXP-15": ("Table 15", "UIQ Keyphrase T2A"),
        }
        for inventory_id, (paper_table, task) in uiq_tasks.items():
            key = (
                inventory_id,
                paper_table,
                "OEA-Nemo3B (+Cl)",
                "Clotho",
                task,
            )
            self.assertEqual(
                self.by_key[key]["reproduction_status"], "REPRODUCED_CLOSE"
            )
            self.assertEqual(self.by_key[key]["observed_metric_count"], 3)

    def test_efficiency_and_negative_boundaries_are_explicit(self) -> None:
        controlled = (
            "EXP-18",
            "Table 5",
            "OEA-Qwen3B",
            "Clotho (1,045 clips; A100-SXM4-80GB)",
            "Inference efficiency",
        )
        negative = (
            "EXP-17",
            "Table 17",
            "OEA-Qwen3B (+Cl)",
            "Mean across AudioCaps, Clotho, MECAT",
            "Negative UIQ hard-negative discrimination",
        )
        self.assertEqual(
            self.by_key[controlled]["reproduction_status"], "CONTROLLED_ONLY"
        )
        self.assertEqual(self.by_key[negative]["reproduction_status"], "BLOCKED")

    def test_vanilla_qwen3b_clotho_main_cells_are_reproduced(self) -> None:
        for inventory, table, task in (
            ("EXP-10", "Table 2", "T2A"),
            ("EXP-11", "Table 3", "T2T"),
        ):
            key = (
                inventory,
                table,
                "Qwen2.5-Omni-3B",
                "Clotho",
                task,
            )
            self.assertEqual(
                self.by_key[key]["reproduction_status"], "REPRODUCED_CLOSE"
            )
            self.assertEqual(self.by_key[key]["observed_metric_count"], 3)

    def test_vanilla_qwen7b_clotho_main_cells_are_reproduced(self) -> None:
        for inventory, table, task in (
            ("EXP-10", "Table 2", "T2A"),
            ("EXP-11", "Table 3", "T2T"),
        ):
            key = (
                inventory,
                table,
                "Qwen2.5-Omni-7B",
                "Clotho",
                task,
            )
            self.assertEqual(
                self.by_key[key]["reproduction_status"], "REPRODUCED_CLOSE"
            )
            self.assertEqual(self.by_key[key]["observed_metric_count"], 3)

    def test_status_vocabulary_is_closed(self) -> None:
        counts = Counter(row["reproduction_status"] for row in self.rows)
        self.assertEqual(
            set(counts),
            {"REPRODUCED_CLOSE", "CONTROLLED_ONLY", "PARTIAL", "TODO", "BLOCKED"},
        )


if __name__ == "__main__":
    unittest.main()
