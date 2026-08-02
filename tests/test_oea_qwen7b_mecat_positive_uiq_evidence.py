from __future__ import annotations

import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    REPOSITORY_ROOT
    / "results/audits/oea_qwen7b_mecat_public848_positive_uiq_eval_20260802.json"
)


class OEAQwen7BMecatPositiveUIQEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    def test_a100_run_completed_with_official_source(self) -> None:
        self.assertEqual(self.audit["status"], "complete")
        self.assertEqual(self.audit["run"]["final_run_rc"], 0)
        self.assertEqual(self.audit["run"]["elapsed_seconds"], 408)
        self.assertEqual(self.audit["run"]["gpu_name"], "NVIDIA A100-SXM4-80GB")
        self.assertTrue(self.audit["official_source"]["oea_official_source_used"])
        self.assertEqual(self.audit["generation"]["audio_embedding_shape"], [848, 512])

    def test_four_public_protocols_are_close(self) -> None:
        self.assertEqual(self.audit["protocol_count"], 4)
        self.assertEqual(len(self.audit["protocols"]), 4)
        maxima = [
            row["maximum_absolute_delta_percentage_points"]
            for row in self.audit["comparison"].values()
        ]
        self.assertEqual(max(maxima), 1.532264)
        self.assertTrue(
            all(row["status"] == "close" for row in self.audit["comparison"].values())
        )

    def test_artifacts_and_847_boundary_are_fixed(self) -> None:
        self.assertEqual(
            self.audit["returned_artifacts"]["suite_metrics_sha256"],
            "7bee2ad60e16a7d93f747b614daf63221b5ceb63bb785e13d308984cc4e5d46f",
        )
        self.assertEqual(self.audit["dataset"]["public_candidate_audio_count"], 848)
        self.assertEqual(self.audit["dataset"]["paper_candidate_audio_count"], 847)
        self.assertFalse(self.audit["protocol_boundary"]["strict_paper_reproduction"])
        self.assertTrue(self.audit["protocol_boundary"]["no_post_hoc_selection"])


if __name__ == "__main__":
    unittest.main()
