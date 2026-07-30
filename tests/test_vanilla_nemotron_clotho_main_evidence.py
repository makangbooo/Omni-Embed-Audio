from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    REPOSITORY_ROOT
    / "results/audits/vanilla_nemotron_3b_clotho_main_eval_20260730.json"
)
OBSERVATIONS_PATH = (
    REPOSITORY_ROOT / "results/observations/reproduction_observations.jsonl"
)


class VanillaNemotronClothoMainEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
        cls.observations = [
            json.loads(line)
            for line in OBSERVATIONS_PATH.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def test_run_is_complete_clean_and_base_only(self) -> None:
        self.assertEqual(self.audit["status"], "complete")
        self.assertEqual(self.audit["run"]["final_run_rc"], 0)
        self.assertEqual(self.audit["run"]["git_status_short"], "")
        self.assertEqual(
            self.audit["run"]["git_commit"],
            "74325cbecb5397f0c80550779b21cb8f834064bd",
        )
        self.assertFalse(self.audit["model"]["projection_head_loaded"])
        self.assertFalse(self.audit["model"]["lora_loaded"])
        self.assertFalse(self.audit["model"]["oea_checkpoint_loaded"])

    def test_smoke_full_and_four_protocols_are_complete(self) -> None:
        self.assertEqual(
            self.audit["generation"]["smoke"]["candidate_embedding_shape"],
            [5, 2048],
        )
        self.assertEqual(
            self.audit["generation"]["full"]["query_embedding_shape"],
            [5225, 2048],
        )
        self.assertEqual(self.audit["protocol_count"], 4)
        self.assertEqual(
            set(self.audit["protocols"]),
            {
                "t2a_public_code_default_joint_all_captions",
                "t2a_public_code_t2a_only_seed0",
                "t2t_public_code_default_seed0",
                "t2t_all_captions_sensitivity",
            },
        )
        self.assertEqual(
            self.audit["protocols"]
            ["t2a_public_code_default_joint_all_captions"]["metrics"]["R@5"],
            21.301435406698566,
        )
        self.assertEqual(
            self.audit["protocols"]
            ["t2t_all_captions_sensitivity"]["metrics"]["R@10"],
            74.20095693779905,
        )

    def test_twelve_observations_bind_to_exact_audit(self) -> None:
        relevant = [
            row
            for row in self.observations
            if row["observation_id"].startswith(
                "vanilla_nemotron_3b_clotho_"
            )
        ]
        self.assertEqual(len(relevant), 12)
        expected_hash = hashlib.sha256(AUDIT_PATH.read_bytes()).hexdigest()
        for row in relevant:
            self.assertEqual(row["status"], "close")
            self.assertEqual(row["seed"], 42)
            self.assertEqual(row["evidence"]["size_bytes"], AUDIT_PATH.stat().st_size)
            self.assertEqual(row["evidence"]["sha256"], expected_hash)

    def test_protocol_uncertainty_and_missing_elapsed_are_preserved(self) -> None:
        limitations = " ".join(self.audit["paper_protocol_limitations"])
        self.assertIn("caption-selection", limitations)
        self.assertIn("self-exclusion", limitations)
        self.assertIn("passage prefix", limitations)
        self.assertIn("no protocol is selected post hoc", limitations)
        self.assertIn("MISSING", self.audit["run"]["elapsed"])


if __name__ == "__main__":
    unittest.main()
