from __future__ import annotations

import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    REPOSITORY_ROOT
    / "results/audits/mga_clap_mecat_public848_positive_uiq_eval_20260802.json"
)


class MGAClapMecatPositiveUiqEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    def test_remote_run_completed_with_official_source_and_gpu(self) -> None:
        self.assertEqual(self.audit["status"], "complete")
        self.assertEqual(self.audit["run"]["final_run_rc"], 0)
        self.assertEqual(self.audit["run"]["elapsed_seconds"], 141)
        self.assertTrue(self.audit["run"]["gpu_used"])
        self.assertTrue(self.audit["official_source"]["oea_official_source_used"])
        self.assertEqual(self.audit["generation"]["audio_embedding_shape"], [848, 1024])
        self.assertEqual(self.audit["generation"]["query_count"], 3392)

    def test_twelve_metrics_and_deltas_are_fixed(self) -> None:
        self.assertEqual(len(self.audit["protocols"]), 4)
        self.assertEqual(
            self.audit["protocols"]["question_released_uiq_public_848"]
            ["metrics"]["R@1"],
            9.080189,
        )
        self.assertEqual(
            self.audit["protocols"]["tagging_released_uiq_public_848"]
            ["metrics"]["R@10"],
            41.155660,
        )
        maxima = [
            row["maximum_absolute_delta_percentage_points"]
            for row in self.audit["comparison"].values()
        ]
        self.assertEqual(max(maxima), 3.305660)
        self.assertTrue(
            all(row["status"] == "close" for row in self.audit["comparison"].values())
        )

    def test_public_848_result_is_not_strict_paper_reproduction(self) -> None:
        self.assertEqual(self.audit["dataset"]["public_candidate_audio_count"], 848)
        self.assertEqual(self.audit["dataset"]["paper_candidate_audio_count"], 847)
        self.assertFalse(self.audit["protocol_boundary"]["strict_paper_reproduction"])


if __name__ == "__main__":
    unittest.main()
