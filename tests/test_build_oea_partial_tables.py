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
                "Table 2": 6,
                "Table 3": 6,
                "Table 12": 2,
                "Table 13": 2,
                "Table 14": 2,
                "Table 15": 2,
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

    def test_no_extension_tasks_are_present(self) -> None:
        serialized = " ".join(
            f"{group['task']} {group['experiment']}" for group in self.groups
        ).lower()
        for excluded in ("a2t", "fiqa", "squtr", "whisper", "rerank"):
            self.assertNotIn(excluded, serialized)


if __name__ == "__main__":
    unittest.main()
