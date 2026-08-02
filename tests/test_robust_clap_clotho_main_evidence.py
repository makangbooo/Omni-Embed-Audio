from __future__ import annotations

import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    REPOSITORY_ROOT
    / "results/audits/robust_clap_clotho_main_eval_20260802.json"
)


class RobustClapClothoMainEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    def test_run_completed_with_official_source_and_gpu(self) -> None:
        self.assertEqual(self.audit["status"], "complete")
        self.assertEqual(self.audit["run"]["final_run_rc"], 0)
        self.assertEqual(self.audit["run"]["elapsed_seconds"], 339)
        self.assertTrue(self.audit["run"]["gpu_used"])
        self.assertTrue(self.audit["official_source"]["oea_official_source_used"])
        self.assertEqual(self.audit["generation"]["audio_embedding_shape"], [1045, 512])
        self.assertEqual(
            self.audit["generation"]["caption_embedding_shape"], [5225, 512]
        )

    def test_source_checkpoint_and_bpe_are_fixed(self) -> None:
        self.assertEqual(
            self.audit["model"]["source_revision"],
            "d08d0e3c545fa22df0930fc0d090741aaa9e2cc1",
        )
        self.assertEqual(
            self.audit["model"]["checkpoint_sha256"],
            "8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037",
        )
        self.assertEqual(
            self.audit["model"]["bpe_vocab_sha256"],
            "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a",
        )

    def test_all_eight_predeclared_protocols_are_fixed(self) -> None:
        self.assertEqual(len(self.audit["protocols"]), 8)
        self.assertEqual(
            self.audit["protocols"]["t2a_public_code_default_joint_all_captions"]
            ["metrics"]["R@1"],
            14.086124,
        )
        self.assertEqual(
            self.audit["protocols"]["tagging_released_uiq"]["metrics"]["R@10"],
            61.052632,
        )
        maxima = [
            row["maximum_absolute_delta_percentage_points"]
            for row in self.audit["comparison"].values()
        ]
        self.assertEqual(max(maxima), 2.871675)

    def test_artifacts_and_claim_boundary_are_fixed(self) -> None:
        self.assertEqual(
            self.audit["returned_artifacts"]["suite_metrics_sha256"],
            "e3442330bf83c590dba10488d9ebe9969434634ce5b521e5d29b040b72f4e32c",
        )
        self.assertFalse(self.audit["protocol_boundary"]["strict_paper_reproduction"])
        self.assertTrue(self.audit["protocol_boundary"]["no_post_hoc_selection"])
        self.assertIn(
            "does not bind the paper row",
            self.audit["protocol_boundary"]["reason"],
        )


if __name__ == "__main__":
    unittest.main()
