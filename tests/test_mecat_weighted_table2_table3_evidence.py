from __future__ import annotations

import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    REPOSITORY_ROOT
    / "results/audits/mecat_weighted_table2_table3_public848_short_eval_20260802.json"
)


class MecatWeightedTable2Table3EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
        cls.runs = {row["variant_id"]: row for row in cls.audit["runs"]}

    def test_all_ten_runs_completed_on_the_fixed_commit(self) -> None:
        self.assertEqual(self.audit["status"], "complete")
        self.assertEqual(len(self.runs), 10)
        self.assertEqual(
            self.audit["git_commit"],
            "163d3e7670f06c6a2cad2ce1abaf0c10ae32bd71",
        )
        execution = self.audit["execution"]
        self.assertEqual(execution["completed_models"], 10)
        self.assertEqual(execution["missing_models"], 0)
        self.assertEqual(execution["failed_models"], 0)
        self.assertEqual(execution["batch_audit_rc"], 0)

    def test_caption_protocol_and_four_protocols_per_model_are_fixed(self) -> None:
        self.assertEqual(self.audit["dataset"]["caption_count"], 2544)
        for row in self.runs.values():
            self.assertEqual(row["caption_shape"][0], 2544)
            self.assertEqual(len(row["comparison"]), 4)
            self.assertRegex(row["suite_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(row["caption_generation_sha256"], r"^[0-9a-f]{64}$")

    def test_comparison_preserves_close_t2a_and_large_t2t_deltas(self) -> None:
        self.assertEqual(
            self.runs["oea_qwen7b"]["comparison"]["t2a_public848_short_all"]
            ["maximum_absolute_delta_percentage_points"],
            0.255409,
        )
        self.assertEqual(
            self.runs["oea_qwen3b_cl"]["comparison"]
            ["t2a_public848_short_seed0"]
            ["maximum_absolute_delta_percentage_points"],
            8.376792,
        )
        self.assertEqual(
            self.runs["mga_clap"]["comparison"]["t2t_public848_short_seed0"]
            ["maximum_absolute_delta_percentage_points"],
            7.974906,
        )

    def test_all_reported_deltas_are_arithmetically_exact(self) -> None:
        registry = [
            json.loads(line)
            for line in (
                REPOSITORY_ROOT / "configs/results/paper_reported_metrics.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        for row in self.runs.values():
            for protocol_id, comparison in row["comparison"].items():
                task = "T2A" if protocol_id.startswith("t2a_") else "T2T"
                paper = {
                    metric["metric"]: float(metric["paper_value"])
                    for metric in registry
                    if metric["dataset"] == "MECAT"
                    and metric["model"] == row["paper_model"]
                    and metric["task"] == task
                }
                expected_paper = [paper[key] for key in ("R@1", "R@5", "R@10")]
                self.assertEqual(comparison["paper"], expected_paper)
                expected_delta = [
                    round(observed - reference, 6)
                    for observed, reference in zip(
                        comparison["reproduced"], expected_paper, strict=True
                    )
                ]
                self.assertEqual(comparison["signed_delta_percentage_points"], expected_delta)
                self.assertEqual(
                    comparison["maximum_absolute_delta_percentage_points"],
                    max(abs(value) for value in expected_delta),
                )

    def test_claim_boundary_forbids_strict_and_post_hoc_claims(self) -> None:
        boundary = self.audit["protocol_boundary"]
        self.assertFalse(boundary["strict_paper_reproduction"])
        self.assertTrue(boundary["no_post_hoc_exclusion"])
        self.assertTrue(boundary["no_post_hoc_protocol_selection"])
        self.assertFalse(self.runs["robust_clap"]["strict_paper_checkpoint_identity"])


if __name__ == "__main__":
    unittest.main()
