from __future__ import annotations

import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    REPOSITORY_ROOT
    / "results/audits/mga_clap_audiocaps_main_eval_20260802.json"
)


class MGAClapAudioCapsMainEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    def test_run_completed_with_official_source_and_gpu(self) -> None:
        self.assertEqual(self.audit["status"], "complete")
        self.assertEqual(self.audit["run"]["final_run_rc"], 0)
        self.assertEqual(self.audit["run"]["elapsed_seconds"], 142)
        self.assertTrue(self.audit["run"]["gpu_used"])
        self.assertTrue(self.audit["official_source"]["oea_official_source_used"])
        self.assertEqual(self.audit["generation"]["audio_embedding_shape"], [975, 1024])
        self.assertEqual(
            self.audit["generation"]["caption_embedding_shape"], [4875, 1024]
        )

    def test_all_eight_predeclared_protocols_are_fixed(self) -> None:
        self.assertEqual(len(self.audit["protocols"]), 8)
        self.assertEqual(
            self.audit["protocols"]["t2a_public_code_default_joint_all_captions"]
            ["metrics"]["R@1"],
            40.758974,
        )
        self.assertEqual(
            self.audit["protocols"]["tagging_released_uiq"]["metrics"]["R@10"],
            92.307692,
        )
        maxima = [
            row["maximum_absolute_delta_percentage_points"]
            for row in self.audit["comparison"].values()
        ]
        self.assertEqual(max(maxima), 5.944359)
        self.assertEqual(
            self.audit["comparison"]["t2t_public_code_default_seed0"]["status"],
            "close",
        )

    def test_artifacts_and_claim_boundary_are_fixed(self) -> None:
        self.assertEqual(
            self.audit["returned_artifacts"]["suite_metrics_sha256"],
            "77ad25479bb5907f52e1751b5d8f5fcbd805259f7ce874e87669bcad25418ddc",
        )
        self.assertFalse(self.audit["protocol_boundary"]["strict_paper_reproduction"])
        self.assertTrue(self.audit["protocol_boundary"]["no_post_hoc_selection"])


if __name__ == "__main__":
    unittest.main()
