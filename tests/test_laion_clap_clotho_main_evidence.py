from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPOSITORY_ROOT / "results/audits/laion_clap_clotho_main_eval_20260730.json"
OBSERVATIONS_PATH = REPOSITORY_ROOT / "results/observations/reproduction_observations.jsonl"


class LaionClapClothoMainEvidenceTests(unittest.TestCase):
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
        self.assertEqual(self.audit["run"]["elapsed_seconds"], 359)
        self.assertIsNone(self.audit["run"]["git_status_short"])
        self.assertEqual(
            self.audit["generation"]["full"]["audio_embedding_shape"],
            [1045, 512],
        )
        self.assertEqual(
            self.audit["generation"]["full"]["caption_embedding_shape"],
            [5225, 512],
        )

    def test_checkpoint_and_four_protocols_are_fixed(self) -> None:
        self.assertEqual(
            self.audit["model"]["checkpoint_sha256"],
            "8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037",
        )
        self.assertEqual(self.audit["protocol_count"], 4)
        self.assertEqual(
            self.audit["protocols"]["t2a_public_code_default_joint_all_captions"]
            ["metrics"]["R@5"],
            37.205742,
        )
        self.assertEqual(
            self.audit["protocols"]["t2t_all_captions_sensitivity"]["metrics"]
            ["R@10"],
            72.669856,
        )

    def test_twelve_observations_bind_to_exact_audit(self) -> None:
        relevant = [
            row
            for row in self.observations
            if row["observation_id"].startswith("laion_clap_clotho_")
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
