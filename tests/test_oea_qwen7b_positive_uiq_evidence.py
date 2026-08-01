from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS_PATH = (
    REPOSITORY_ROOT / "results/observations/reproduction_observations.jsonl"
)
VARIANTS = {
    "qwen7b_ac": {
        "audit": "oea_qwen7b_ac_clotho_positive_uiq_eval_20260802.json",
        "elapsed": 130,
        "maximum_delta": 0.961818,
        "question_r5": 48.421053,
        "tagging_r10": 64.976077,
    },
    "qwen7b_cl": {
        "audit": "oea_qwen7b_cl_clotho_positive_uiq_eval_20260802.json",
        "elapsed": 102,
        "maximum_delta": 1.055933,
        "question_r5": 52.727273,
        "tagging_r10": 68.421053,
    },
}


class OeaQwen7bPositiveUiqEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.observations = [
            json.loads(line)
            for line in OBSERVATIONS_PATH.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def test_both_remote_runs_and_four_protocols_completed(self) -> None:
        for expected in VARIANTS.values():
            path = REPOSITORY_ROOT / "results/audits" / expected["audit"]
            audit = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(audit["status"], "complete")
            self.assertEqual(audit["run"]["final_run_rc"], 0)
            self.assertEqual(audit["run"]["elapsed_seconds"], expected["elapsed"])
            self.assertTrue(audit["run"]["gpu_used"])
            self.assertTrue(audit["official_source"]["oea_official_source_used"])
            self.assertEqual(audit["generation"]["query_count"], 4180)
            self.assertEqual(audit["protocol_count"], 4)

    def test_observed_values_and_close_boundaries_are_fixed(self) -> None:
        for expected in VARIANTS.values():
            path = REPOSITORY_ROOT / "results/audits" / expected["audit"]
            audit = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                audit["protocols"]["question_released_uiq"]["metrics"]["R@5"],
                expected["question_r5"],
            )
            self.assertEqual(
                audit["protocols"]["tagging_released_uiq"]["metrics"]["R@10"],
                expected["tagging_r10"],
            )
            self.assertEqual(
                max(
                    row["maximum_absolute_delta_percentage_points"]
                    for row in audit["comparison"].values()
                ),
                expected["maximum_delta"],
            )
            self.assertTrue(
                all(row["status"] == "close" for row in audit["comparison"].values())
            )
            self.assertFalse(audit["protocol_boundary"]["strict_paper_reproduction"])

    def test_twenty_four_close_observations_bind_to_exact_audits(self) -> None:
        for prefix, expected in VARIANTS.items():
            path = REPOSITORY_ROOT / "results/audits" / expected["audit"]
            relevant = [
                row
                for row in self.observations
                if row["observation_id"].startswith(
                    f"{prefix}_positive_uiq_clotho_"
                )
            ]
            self.assertEqual(len(relevant), 12)
            expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            for row in relevant:
                self.assertEqual(row["status"], "close")
                self.assertEqual(row["seed"], 0)
                self.assertEqual(row["evidence"]["size_bytes"], path.stat().st_size)
                self.assertEqual(row["evidence"]["sha256"], expected_hash)


if __name__ == "__main__":
    unittest.main()
