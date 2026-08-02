from __future__ import annotations

import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    REPOSITORY_ROOT
    / "results/audits/oea_3b_mecat_public848_positive_uiq_eval_20260802.json"
)


class OEA3BMecatPublic848PositiveUIQEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
        cls.runs = {row["variant_id"]: row for row in cls.audit["runs"]}

    def test_all_four_runs_completed_on_the_fixed_commit(self) -> None:
        self.assertEqual(self.audit["status"], "complete")
        self.assertEqual(len(self.runs), 4)
        self.assertEqual(self.audit["execution"]["batch_elapsed_seconds"], 1208)
        self.assertEqual(
            self.audit["git_commit"],
            "925291e8aff98aacbb4a5234fd122062ebd9ab7c",
        )
        self.assertTrue(all(row["final_run_rc"] == 0 for row in self.runs.values()))
        self.assertTrue(all(len(row["protocols"]) == 4 for row in self.runs.values()))

    def test_public_848_and_strict_847_boundary_is_preserved(self) -> None:
        self.assertEqual(self.audit["dataset"]["public_candidate_count"], 848)
        self.assertEqual(self.audit["dataset"]["paper_candidate_count"], 847)
        boundary = self.audit["protocol_boundary"]
        self.assertFalse(boundary["strict_paper_reproduction"])
        self.assertTrue(boundary["no_post_hoc_exclusion"])
        self.assertIn("does not publish the excluded ID", boundary["claim_boundary"])

    def test_nemotron_variants_are_close_but_qwen_variants_are_not(self) -> None:
        self.assertEqual(
            self.runs["oea_nemo3b"]["assessment"],
            "controlled_public848_close",
        )
        self.assertEqual(
            self.runs["oea_nemo3b_cl"]["assessment"],
            "controlled_public848_close",
        )
        self.assertEqual(
            self.runs["oea_qwen3b"]["assessment"],
            "controlled_public848_trend_not_close",
        )
        self.assertEqual(
            self.runs["oea_qwen3b_cl"]["assessment"],
            "controlled_public848_trend_not_close",
        )
        self.assertEqual(
            self.runs["oea_qwen3b_cl"]
            ["maximum_absolute_delta_percentage_points"]["question"],
            7.667170,
        )

    def test_artifact_hashes_are_not_invented(self) -> None:
        self.assertIsNone(self.audit["returned_artifact_hashes"])


if __name__ == "__main__":
    unittest.main()
