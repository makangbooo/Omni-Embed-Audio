from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    REPOSITORY_ROOT
    / "results/audits/laion_clap_clotho_positive_uiq_eval_20260802.json"
)
OBSERVATIONS_PATH = (
    REPOSITORY_ROOT / "results/observations/reproduction_observations.jsonl"
)


class LaionClapPositiveUiqEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
        cls.observations = [
            json.loads(line)
            for line in OBSERVATIONS_PATH.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def test_remote_run_and_four_protocols_completed(self) -> None:
        self.assertEqual(self.audit["status"], "complete")
        self.assertEqual(self.audit["run"]["final_run_rc"], 0)
        self.assertEqual(self.audit["run"]["elapsed_seconds"], 82)
        self.assertTrue(self.audit["run"]["gpu_used"])
        self.assertTrue(
            self.audit["official_source"]["oea_official_source_used"]
        )
        self.assertEqual(self.audit["generation"]["query_count"], 4180)
        self.assertEqual(self.audit["protocol_count"], 4)

    def test_observed_values_and_paper_deltas_are_fixed(self) -> None:
        protocols = self.audit["protocols"]
        self.assertEqual(
            protocols["question_released_uiq"]["metrics"]["R@5"],
            43.157895,
        )
        self.assertEqual(
            protocols["tagging_released_uiq"]["metrics"]["R@10"],
            63.349282,
        )
        comparison = self.audit["comparison"]
        self.assertEqual(
            comparison["tagging_released_uiq"]
            ["maximum_absolute_delta_percentage_points"],
            8.329282,
        )
        self.assertFalse(
            self.audit["protocol_boundary"]["strict_paper_reproduction"]
        )

    def test_twelve_trend_observations_bind_to_exact_audit(self) -> None:
        relevant = [
            row
            for row in self.observations
            if row["observation_id"].startswith("laion_clap_clotho_uiq_")
        ]
        self.assertEqual(len(relevant), 12)
        expected_hash = hashlib.sha256(AUDIT_PATH.read_bytes()).hexdigest()
        for row in relevant:
            self.assertEqual(row["status"], "trend_reproduced")
            self.assertEqual(row["seed"], 0)
            self.assertEqual(
                row["evidence"]["size_bytes"], AUDIT_PATH.stat().st_size
            )
            self.assertEqual(row["evidence"]["sha256"], expected_hash)


if __name__ == "__main__":
    unittest.main()
