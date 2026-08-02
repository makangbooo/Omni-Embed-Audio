from __future__ import annotations

import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    REPOSITORY_ROOT
    / "results/audits/audiocaps_remaining_seven_main_eval_20260802.json"
)


class AudioCapsRemainingSevenMainEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
        cls.runs = {row["variant_id"]: row for row in cls.audit["runs"]}

    def test_all_seven_runs_completed_on_the_fixed_commit(self) -> None:
        self.assertEqual(self.audit["status"], "complete")
        self.assertEqual(len(self.runs), 7)
        self.assertEqual(self.audit["execution"]["batch_elapsed_seconds"], 3184)
        self.assertEqual(
            self.audit["git_commit"],
            "c55bf682fc72663f1272e7646090e6b31cc2b82d",
        )
        self.assertTrue(all(row["final_run_rc"] == 0 for row in self.runs.values()))
        self.assertTrue(
            all(row["completion_status"] == "complete" for row in self.runs.values())
        )

    def test_protocol_counts_and_embedding_shapes_are_fixed(self) -> None:
        for variant_id, row in self.runs.items():
            expected_protocols = 4 if variant_id.startswith("vanilla_") else 8
            self.assertEqual(len(row["protocols"]), expected_protocols)
            self.assertEqual(row["embedding_shapes"]["audio"][0], 975)
            self.assertEqual(row["embedding_shapes"]["caption"][0], 4875)

    def test_comparison_boundary_preserves_the_nemotron_t2a_mismatch(self) -> None:
        nemotron = self.runs["vanilla_nemotron_3b"]
        self.assertEqual(
            nemotron["maximum_absolute_delta_percentage_points"]["table2_default"],
            13.454103,
        )
        self.assertEqual(
            nemotron["assessment"]["table2"], "trend_reproduced_not_close"
        )
        self.assertEqual(nemotron["assessment"]["table3"], "close")

    def test_four_oea_models_are_close_and_hashes_are_not_invented(self) -> None:
        oea_runs = [row for key, row in self.runs.items() if key.startswith("oea_")]
        self.assertEqual(len(oea_runs), 4)
        self.assertTrue(all(row["assessment"] == "all_protocols_close" for row in oea_runs))
        self.assertIsNone(self.audit["returned_artifact_hashes"])
        self.assertFalse(self.audit["protocol_boundary"]["strict_paper_reproduction"])
        self.assertTrue(self.audit["protocol_boundary"]["no_post_hoc_selection"])


if __name__ == "__main__":
    unittest.main()
