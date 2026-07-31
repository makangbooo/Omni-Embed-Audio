from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPOSITORY_ROOT / "results/audits/mga_clap_clotho_main_eval_20260731.json"
OBSERVATIONS_PATH = REPOSITORY_ROOT / "results/observations/reproduction_observations.jsonl"


class MGAClapClothoMainEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
        cls.observations = [
            json.loads(line)
            for line in OBSERVATIONS_PATH.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def test_run_and_embedding_gates_completed(self) -> None:
        self.assertEqual(self.audit["status"], "complete")
        self.assertEqual(self.audit["run"]["final_run_rc"], 0)
        self.assertEqual(self.audit["run"]["elapsed_seconds"], 293)
        self.assertTrue(self.audit["execution"]["gpu_used"])
        self.assertTrue(self.audit["execution"]["oea_official_source_used"])
        self.assertEqual(
            self.audit["generation"]["full"]["audio_embedding_shape"],
            [1045, 1024],
        )
        self.assertEqual(
            self.audit["generation"]["full"]["caption_embedding_shape"],
            [5225, 1024],
        )

    def test_checkpoint_and_four_protocols_are_fixed(self) -> None:
        self.assertEqual(
            self.audit["model"]["checkpoint_sha256"],
            "8703740b738e973a5b4d8a18a074ad56880e98f8ba21cd618d7d7ee5422d6e26",
        )
        self.assertEqual(self.audit["protocol_count"], 4)
        self.assertEqual(
            self.audit["protocols"]["t2a_public_code_default_joint_all_captions"]
            ["metrics"]["R@5"],
            46.947368,
        )
        self.assertEqual(
            self.audit["protocols"]["t2t_all_captions_sensitivity"]["metrics"]
            ["R@10"],
            78.794258,
        )

    def test_twelve_observations_bind_to_exact_audit(self) -> None:
        relevant = [
            row
            for row in self.observations
            if row["observation_id"].startswith("mga_clap_clotho_")
        ]
        self.assertEqual(len(relevant), 12)
        expected_hash = hashlib.sha256(AUDIT_PATH.read_bytes()).hexdigest()
        for row in relevant:
            self.assertEqual(row["status"], "close")
            self.assertEqual(row["seed"], 0)
            self.assertEqual(row["evidence"]["size_bytes"], AUDIT_PATH.stat().st_size)
            self.assertEqual(row["evidence"]["sha256"], expected_hash)

    def test_protocol_uncertainty_is_preserved(self) -> None:
        limitations = " ".join(self.audit["paper_protocol_limitations"])
        self.assertIn("caption-selection", limitations)
        self.assertIn("self-exclusion", limitations)
        self.assertIn("no protocol is selected post hoc", limitations)


if __name__ == "__main__":
    unittest.main()
