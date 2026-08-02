from __future__ import annotations

import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    REPOSITORY_ROOT
    / "results/audits/oea_qwen7b_audiocaps_main_eval_20260802.json"
)


class OEAQwen7BAudioCapsMainEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    def test_a100_run_completed_with_official_source_and_gpu(self) -> None:
        self.assertEqual(self.audit["status"], "complete")
        self.assertEqual(self.audit["run"]["final_run_rc"], 0)
        self.assertEqual(self.audit["run"]["elapsed_seconds"], 504)
        self.assertEqual(self.audit["run"]["gpu_name"], "NVIDIA A100-SXM4-80GB")
        self.assertTrue(self.audit["official_source"]["oea_official_source_used"])
        self.assertEqual(self.audit["generation"]["audio_embedding_shape"], [975, 512])
        self.assertEqual(
            self.audit["generation"]["caption_embedding_shape"], [4875, 512]
        )

    def test_failed_4090_attempt_is_retained(self) -> None:
        failed = self.audit["attempts"][0]
        self.assertEqual(failed["gpu_name"], "NVIDIA GeForce RTX 4090")
        self.assertEqual(failed["final_run_rc"], 1)
        self.assertEqual(failed["error_type"], "torch.OutOfMemoryError")
        self.assertTrue(failed["artifacts_retained"])

    def test_all_eight_predeclared_protocols_are_close(self) -> None:
        self.assertEqual(len(self.audit["protocols"]), 8)
        maxima = [
            row["maximum_absolute_delta_percentage_points"]
            for row in self.audit["comparison"].values()
        ]
        self.assertEqual(max(maxima), 2.198718)
        self.assertTrue(
            all("close" in row["status"] for row in self.audit["comparison"].values())
        )

    def test_artifacts_and_claim_boundary_are_fixed(self) -> None:
        self.assertEqual(
            self.audit["returned_artifacts"]["suite_metrics_sha256"],
            "a832f985c8ae617e2c1c440a3f6879e0017a150080b2216f8be8c1989c54d17c",
        )
        self.assertFalse(self.audit["protocol_boundary"]["strict_paper_reproduction"])
        self.assertTrue(self.audit["protocol_boundary"]["no_post_hoc_selection"])


if __name__ == "__main__":
    unittest.main()
