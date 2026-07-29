from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    REPOSITORY_ROOT
    / "results/audits/oea_nemo3b_ac_clotho_official_source_eval_20260729.json"
)
OBSERVATIONS_PATH = (
    REPOSITORY_ROOT / "results/observations/reproduction_observations.jsonl"
)


class Nemo3bAcOfficialSourceEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
        cls.observations = [
            json.loads(line)
            for line in OBSERVATIONS_PATH.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def test_success_and_failed_attempts_are_both_preserved(self) -> None:
        self.assertEqual(self.audit["status"], "complete")
        self.assertEqual(
            [attempt["status"] for attempt in self.audit["uiq_attempts"]],
            ["failed", "complete"],
        )
        self.assertEqual(
            [attempt["status"] for attempt in self.audit["metric_attempts"]],
            ["failed", "complete"],
        )
        self.assertTrue(self.audit["uiq_attempts"][0]["preserved"])
        self.assertTrue(self.audit["metric_attempts"][0]["preserved"])

    def test_successful_runs_have_fixed_identities(self) -> None:
        uiq = self.audit["uiq_attempts"][1]
        metric = self.audit["metric_attempts"][1]
        self.assertEqual(uiq["final_run_rc"], 0)
        self.assertEqual(uiq["query_count"], 4180)
        self.assertEqual(uiq["embedding_dimension"], 512)
        self.assertEqual(metric["final_run_rc"], 0)
        self.assertEqual(metric["elapsed_seconds"], 7)
        self.assertEqual(
            metric["suite_metrics_sha256"],
            "05cc707af53c0cd86dd8735456eacc3f0fe80c52ad38050518b2f501b70bcc2c",
        )

    def test_all_eight_protocols_are_fixed(self) -> None:
        suite = self.audit["final_suite"]
        self.assertEqual(suite["status"], "complete")
        self.assertEqual(suite["protocol_count"], 8)
        self.assertEqual(len(suite["protocols"]), 8)
        self.assertEqual(
            suite["protocols"]["t2a_public_code_default_joint_all_captions"]
            ["metrics"]["R@5"],
            41.32057416267942,
        )
        self.assertEqual(
            suite["protocols"]["t2t_public_code_default_seed0"]["metrics"]
            ["R@1"],
            63.34928229665072,
        )
        self.assertEqual(
            suite["protocols"]["tagging_released_uiq"]["metrics"]["R@10"],
            60.19138755980862,
        )

    def test_twenty_four_observations_bind_to_exact_audit(self) -> None:
        relevant = [
            row
            for row in self.observations
            if row["observation_id"].startswith("nemo3b_ac_clotho_")
        ]
        self.assertEqual(len(relevant), 24)
        expected_hash = hashlib.sha256(AUDIT_PATH.read_bytes()).hexdigest()
        for row in relevant:
            self.assertEqual(row["status"], "close")
            self.assertEqual(row["evidence"]["size_bytes"], AUDIT_PATH.stat().st_size)
            self.assertEqual(row["evidence"]["sha256"], expected_hash)

    def test_protocol_uncertainty_is_not_erased(self) -> None:
        limitations = " ".join(self.audit["paper_protocol_limitations"])
        self.assertIn("caption-selection", limitations)
        self.assertIn("self-exclusion", limitations)
        self.assertIn("no protocol is selected post hoc", limitations)
        self.assertIn("passage prefix", limitations)


if __name__ == "__main__":
    unittest.main()
