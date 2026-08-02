from __future__ import annotations

import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    REPOSITORY_ROOT
    / "results/audits/robust_clap_audiocaps_mecat_eval_20260802.json"
)


class RobustClapAudioCapsMecatEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    def test_both_remote_runs_completed(self) -> None:
        self.assertEqual(self.audit["status"], "complete")
        for run in self.audit["runs"].values():
            self.assertEqual(run["final_run_rc"], 0)
            self.assertEqual(run["completion_status"], "complete")
            self.assertEqual(run["gpu_name"], "NVIDIA GeForce RTX 4090")
        self.assertEqual(
            self.audit["model"]["source_revision"],
            "d08d0e3c545fa22df0930fc0d090741aaa9e2cc1",
        )
        self.assertTrue(self.audit["official_source"]["oea_official_source_used"])

    def test_all_predeclared_protocols_and_deltas_are_fixed(self) -> None:
        self.assertEqual(len(self.audit["audiocaps"]["protocols"]), 8)
        self.assertEqual(len(self.audit["mecat_public_848"]["protocols"]), 4)
        self.assertEqual(
            self.audit["audiocaps"]["comparison"]
            ["t2a_public_code_default_joint_all_captions"]
            ["maximum_absolute_delta_percentage_points"],
            0.042821,
        )
        self.assertEqual(
            self.audit["mecat_public_848"]["comparison"]
            ["question_released_uiq_public_848"]
            ["maximum_absolute_delta_percentage_points"],
            1.766415,
        )

    def test_claim_and_evidence_boundaries_are_explicit(self) -> None:
        self.assertFalse(self.audit["model"]["strict_paper_checkpoint_identity"])
        self.assertFalse(self.audit["protocol_boundary"]["strict_paper_reproduction"])
        self.assertTrue(self.audit["protocol_boundary"]["no_post_hoc_selection"])
        self.assertFalse(
            self.audit["returned_artifacts"]["sha256_manifest_returned"]
        )
        self.assertEqual(
            self.audit["runs"]["mecat_public_848"]["public_candidate_count"], 848
        )
        self.assertEqual(
            self.audit["runs"]["mecat_public_848"]["paper_candidate_count"], 847
        )


if __name__ == "__main__":
    unittest.main()
