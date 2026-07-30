from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATHS = {
    "qwen7b_ac": REPOSITORY_ROOT
    / "results/audits/oea_qwen7b_ac_clotho_official_source_eval_20260730.json",
    "qwen7b_cl": REPOSITORY_ROOT
    / "results/audits/oea_qwen7b_cl_clotho_official_source_eval_20260730.json",
}
OBSERVATIONS_PATH = (
    REPOSITORY_ROOT / "results/observations/reproduction_observations.jsonl"
)


class Qwen7bClothoOfficialSourceEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audits = {
            name: json.loads(path.read_text(encoding="utf-8"))
            for name, path in AUDIT_PATHS.items()
        }
        cls.observations = [
            json.loads(line)
            for line in OBSERVATIONS_PATH.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def test_both_runs_are_complete_and_clean(self) -> None:
        for audit in self.audits.values():
            self.assertEqual(audit["status"], "complete")
            self.assertEqual(audit["run"]["final_run_rc"], 0)
            self.assertEqual(audit["run"]["failed_stage"], "none")
            self.assertEqual(audit["run"]["git_status_short"], "")
            self.assertEqual(audit["run"]["elapsed"], "00:07:36")
            self.assertEqual(audit["run"]["gpu"], "NVIDIA GeForce RTX 4090")

    def test_checkpoint_and_artifact_identities_are_fixed(self) -> None:
        self.assertIn(
            "cd751e3a71f0b47b9ecbbc0f5a11e4673097f0ebdbd80f610387f5778dd70a46",
            self.audits["qwen7b_ac"]["model"]["checkpoint"],
        )
        self.assertIn(
            "09ce41af7e7ac23106fa74d25e45b7229a046d687774cbd4ab2c00c05097d92e",
            self.audits["qwen7b_cl"]["model"]["checkpoint"],
        )
        for audit in self.audits.values():
            self.assertEqual(len(audit["artifacts"]["suite_metrics.json"]), 64)
            self.assertNotEqual(
                audit["artifacts"]["combined.log.final"],
                audit["artifacts"]["combined.log.pre_finish_manifest_entry"],
            )

    def test_all_eight_protocol_blocks_are_fixed(self) -> None:
        for audit in self.audits.values():
            self.assertEqual(audit["protocol_count"], 4)
            self.assertEqual(
                set(audit["protocols"]),
                {
                    "t2a_public_code_default_joint_all_captions",
                    "t2a_public_code_t2a_only_seed0",
                    "t2t_public_code_default_seed0",
                    "t2t_all_captions_sensitivity",
                },
            )
        self.assertEqual(
            self.audits["qwen7b_ac"]["protocols"]
            ["t2a_public_code_default_joint_all_captions"]["metrics"]["R@5"],
            44.688995215311,
        )
        self.assertEqual(
            self.audits["qwen7b_cl"]["protocols"]
            ["t2t_all_captions_sensitivity"]["metrics"]["R@10"],
            79.6555023923445,
        )

    def test_twenty_four_observations_bind_to_exact_audits(self) -> None:
        for prefix, path in AUDIT_PATHS.items():
            relevant = [
                row
                for row in self.observations
                if row["observation_id"].startswith(prefix + "_clotho_")
            ]
            self.assertEqual(len(relevant), 12)
            expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            for row in relevant:
                self.assertEqual(row["status"], "close")
                self.assertEqual(row["seed"], 0)
                self.assertEqual(row["evidence"]["size_bytes"], path.stat().st_size)
                self.assertEqual(row["evidence"]["sha256"], expected_hash)

    def test_protocol_uncertainty_is_preserved(self) -> None:
        for audit in self.audits.values():
            limitations = " ".join(audit["paper_protocol_limitations"])
            self.assertIn("caption-selection", limitations)
            self.assertIn("self-exclusion", limitations)
            self.assertIn("passage prefix", limitations)
            self.assertIn("no protocol is selected post hoc", limitations)


if __name__ == "__main__":
    unittest.main()
