from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPOSITORY_ROOT / "results/audits/oea_nemo3b_clotho_t2t_20260729.json"
OBSERVATIONS_PATH = (
    REPOSITORY_ROOT / "results/observations/reproduction_observations.jsonl"
)


class Nemo3bClT2tEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
        cls.observations = [
            json.loads(line)
            for line in OBSERVATIONS_PATH.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def test_remote_run_completed_cleanly(self) -> None:
        suite = self.audit["retrieval_suite"]
        self.assertEqual(self.audit["status"], "complete")
        self.assertEqual(suite["status"], "complete")
        self.assertEqual(suite["final_run_rc"], 0)
        self.assertEqual(suite["elapsed_seconds"], 47)
        self.assertIsNone(suite["failed_stage"])
        self.assertIsNone(suite["error_summary"])

    def test_both_predeclared_protocols_and_hashes_are_fixed(self) -> None:
        suite = self.audit["retrieval_suite"]
        self.assertEqual(
            suite["protocols"]["t2t_public_code_default_seed0"]["metrics"],
            {
                "R@1": 62.96650717703349,
                "R@5": 75.98086124401914,
                "R@10": 80.86124401913875,
                "MRR": 0.6885831848423701,
                "DCG": 0.7474843659280417,
            },
        )
        self.assertEqual(
            suite["protocols"]["t2t_all_captions_sensitivity"]["metrics"],
            {
                "R@1": 63.693779904306226,
                "R@5": 75.23444976076556,
                "R@10": 80.01913875598086,
                "MRR": 0.6919725083442058,
                "DCG": 0.7493360920246421,
            },
        )
        self.assertEqual(
            suite["artifacts"]["suite_metrics.json"]["sha256"],
            "8c0bf1f370a5c6ed049bb33f908bfe40d798f0ec2c46b810cf195f2c889fe129",
        )
        self.assertEqual(
            suite["protocols"]["t2t_all_captions_sensitivity"]
            ["full_rankings"]["sha256"],
            "c1eb82c420d3ec751ee76b675cdafa3c1e7ccf369128c474f3c9df750f6a9f54",
        )

    def test_paper_protocol_limitation_remains_explicit(self) -> None:
        self.assertEqual(
            self.audit["paper_protocol_limitation"]["status"], "MISSING"
        )
        self.assertEqual(
            self.audit["comparison"]["t2t_public_code_default_seed0"]
            ["maximum_absolute_delta_percentage_points"],
            0.80349282296651,
        )
        self.assertEqual(
            self.audit["comparison"]["t2t_all_captions_sensitivity"]
            ["maximum_absolute_delta_percentage_points"],
            0.09086124401914,
        )

    def test_six_recall_observations_bind_to_this_exact_audit(self) -> None:
        relevant = [
            row
            for row in self.observations
            if row["experiment"]
            in {
                "oea_nemo3b_cl_clotho_t2t_public_code_default_seed0",
                "oea_nemo3b_cl_clotho_t2t_all_captions_sensitivity",
            }
        ]
        self.assertEqual(len(relevant), 6)
        expected_hash = hashlib.sha256(AUDIT_PATH.read_bytes()).hexdigest()
        for row in relevant:
            self.assertEqual(row["evidence"]["size_bytes"], AUDIT_PATH.stat().st_size)
            self.assertEqual(row["evidence"]["sha256"], expected_hash)
            self.assertEqual(row["status"], "close")


if __name__ == "__main__":
    unittest.main()
