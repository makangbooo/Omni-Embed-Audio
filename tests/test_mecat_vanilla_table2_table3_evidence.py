from __future__ import annotations

import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    REPOSITORY_ROOT
    / "results/audits/mecat_vanilla_table2_table3_public848_short_eval_20260803.json"
)


class MecatVanillaTable2Table3EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
        cls.runs = {row["variant_id"]: row for row in cls.audit["runs"]}

    def test_all_three_runs_completed_on_the_fixed_commit(self) -> None:
        self.assertEqual(self.audit["status"], "complete")
        self.assertEqual(
            self.audit["git_commit"],
            "d22fed1078351139975b7ae32fad8d235b44d546",
        )
        self.assertEqual(len(self.runs), 3)
        execution = self.audit["execution"]
        self.assertEqual(execution["completed_models"], 3)
        self.assertEqual(execution["expected_models"], 3)
        self.assertEqual(execution["missing_models"], 0)
        self.assertEqual(execution["failed_models"], 0)
        self.assertEqual(execution["artifact_verify_status"], "complete")

    def test_runs_are_base_only_and_all_stages_succeeded(self) -> None:
        for row in self.runs.values():
            self.assertEqual(row["final_run_rc"], 0)
            self.assertEqual(row["completion_status"], "complete")
            self.assertEqual(row["failed_stage"], "none")
            self.assertTrue(all(rc == 0 for rc in row["stage_rc"].values()))
            self.assertFalse(row["model_checkpoint_loaded"])
            self.assertFalse(row["oea_checkpoint_loaded"])
            self.assertFalse(row["lora_loaded"])
            self.assertFalse(row["projection_head_loaded"])

    def test_fixed_caption_protocol_shapes_and_hashes(self) -> None:
        self.assertEqual(self.audit["dataset"]["caption_count"], 2544)
        self.assertRegex(
            self.audit["dataset"]["projected_manifest_sha256"], r"^[0-9a-f]{64}$"
        )
        for row in self.runs.values():
            self.assertEqual(row["audio_shape"][0], 848)
            self.assertEqual(row["caption_shape"][0], 2544)
            self.assertEqual(len(row["comparison"]), 4)
            self.assertEqual(len(row["artifact_sha256"]), 7)
            for digest in row["artifact_sha256"].values():
                self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_all_reported_deltas_match_the_paper_registry(self) -> None:
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
                self.assertEqual(
                    comparison["signed_delta_percentage_points"], expected_delta
                )
                self.assertEqual(
                    comparison["maximum_absolute_delta_percentage_points"],
                    max(abs(value) for value in expected_delta),
                )

    def test_claim_boundary_forbids_strict_and_post_hoc_claims(self) -> None:
        boundary = self.audit["protocol_boundary"]
        self.assertFalse(boundary["strict_paper_reproduction"])
        self.assertTrue(boundary["no_post_hoc_exclusion"])
        self.assertTrue(boundary["no_post_hoc_protocol_selection"])


if __name__ == "__main__":
    unittest.main()
