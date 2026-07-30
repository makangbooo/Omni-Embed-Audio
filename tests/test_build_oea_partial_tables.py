from __future__ import annotations

from collections import Counter
import unittest

from scripts.build_oea_partial_tables import DEFAULT_INPUT, selected_groups


class BuildOeaPartialTablesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.groups = selected_groups(DEFAULT_INPUT)

    def test_only_requested_paper_tables_are_included(self) -> None:
        self.assertEqual(
            Counter(group["paper_table"] for group in self.groups),
            {
                "Table 2": 16,
                "Table 3": 16,
                "Table 12": 4,
                "Table 13": 4,
                "Table 14": 4,
                "Table 15": 4,
            },
        )

    def test_nemo_t2a_anchor_is_preserved(self) -> None:
        matches = [
            group
            for group in self.groups
            if group["model"] == "OEA-Nemo3B (+Cl)"
            and group["paper_table"] == "Table 2"
            and group["protocol"] == "[CODE] all-caption"
        ]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["paper"], ["21.57", "47.16", "60.36"])
        self.assertEqual(
            matches[0]["reproduced"],
            ["21.72248803827751", "47.119617224880386", "60.44019138755981"],
        )

    def test_nemo_t2t_protocols_remain_separate(self) -> None:
        matches = [
            group
            for group in self.groups
            if group["model"] == "OEA-Nemo3B (+Cl)"
            and group["paper_table"] == "Table 3"
        ]
        self.assertEqual(len(matches), 2)
        self.assertEqual(
            {group["protocol"] for group in matches},
            {
                "[CODE] seed-0 one-caption/self-exclusion",
                "[INFERRED] all-caption sensitivity",
            },
        )
        self.assertEqual({tuple(group["paper"]) for group in matches}, {("63.77", "75.29", "80.11")})

    def test_nemo_positive_uiq_protocols_are_present(self) -> None:
        matches = [
            group
            for group in self.groups
            if group["model"] == "OEA-Nemo3B (+Cl)"
            and group["paper_table"] in {"Table 12", "Table 13", "Table 14", "Table 15"}
        ]
        self.assertEqual(len(matches), 4)
        self.assertEqual(
            {group["paper_table"] for group in matches},
            {"Table 12", "Table 13", "Table 14", "Table 15"},
        )
        self.assertEqual(
            {group["protocol"] for group in matches},
            {"[CODE] released positive UIQ"},
        )

    def test_nemo3b_ac_official_source_cells_are_present(self) -> None:
        matches = [
            group
            for group in self.groups
            if group["model"] == "OEA-Nemo3B"
            and group["dataset"] == "Clotho"
        ]
        self.assertEqual(len(matches), 8)
        self.assertEqual(
            Counter(group["paper_table"] for group in matches),
            {
                "Table 2": 2,
                "Table 3": 2,
                "Table 12": 1,
                "Table 13": 1,
                "Table 14": 1,
                "Table 15": 1,
            },
        )

    def test_qwen7b_official_source_cells_are_present(self) -> None:
        matches = [
            group
            for group in self.groups
            if group["model"] in {"OEA-Qwen7B", "OEA-Qwen7B (+Cl)"}
            and group["dataset"] == "Clotho"
        ]
        self.assertEqual(len(matches), 8)
        self.assertEqual(
            Counter(group["paper_table"] for group in matches),
            {"Table 2": 4, "Table 3": 4},
        )

    def test_vanilla_nemotron_main_table_cells_are_present(self) -> None:
        matches = [
            group
            for group in self.groups
            if group["model"] == "Nemotron-3B"
            and group["dataset"] == "Clotho"
        ]
        self.assertEqual(len(matches), 4)
        self.assertEqual(
            Counter(group["paper_table"] for group in matches),
            {"Table 2": 2, "Table 3": 2},
        )

    def test_vanilla_qwen3b_main_table_cells_are_present(self) -> None:
        matches = [
            group
            for group in self.groups
            if group["model"] == "Qwen2.5-Omni-3B"
            and group["dataset"] == "Clotho"
        ]
        self.assertEqual(len(matches), 4)
        self.assertEqual(
            Counter(group["paper_table"] for group in matches),
            {"Table 2": 2, "Table 3": 2},
        )

    def test_no_extension_tasks_are_present(self) -> None:
        serialized = " ".join(
            f"{group['task']} {group['experiment']}" for group in self.groups
        ).lower()
        for excluded in ("a2t", "fiqa", "squtr", "whisper", "rerank"):
            self.assertNotIn(excluded, serialized)


if __name__ == "__main__":
    unittest.main()
