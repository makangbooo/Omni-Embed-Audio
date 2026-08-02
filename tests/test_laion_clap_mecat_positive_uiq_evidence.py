from __future__ import annotations

import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    REPOSITORY_ROOT
    / "results/audits/laion_clap_mecat_public848_positive_uiq_eval_20260802.json"
)


class LaionClapMecatPositiveUiqEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    def test_remote_run_completed_with_official_source_and_gpu(self) -> None:
        self.assertEqual(self.audit["status"], "complete")
        self.assertEqual(self.audit["run"]["final_run_rc"], 0)
        self.assertEqual(self.audit["run"]["elapsed_seconds"], 178)
        self.assertTrue(self.audit["run"]["gpu_used"])
        self.assertTrue(self.audit["official_source"]["oea_official_source_used"])
        self.assertEqual(self.audit["generation"]["audio_embedding_shape"], [848, 512])
        self.assertEqual(self.audit["generation"]["query_count"], 3392)

    def test_twelve_metrics_and_deltas_are_fixed(self) -> None:
        protocols = self.audit["protocols"]
        self.assertEqual(len(protocols), 4)
        self.assertEqual(
            protocols["question_released_uiq_public_848"]["metrics"]["R@1"],
            7.900943,
        )
        self.assertEqual(
            protocols["tagging_released_uiq_public_848"]["metrics"]["R@10"],
            38.561321,
        )
        self.assertEqual(
            self.audit["comparison"]["tagging_released_uiq_public_848"]
            ["maximum_absolute_delta_percentage_points"],
            5.071321,
        )

    def test_public_848_result_is_not_strict_paper_reproduction(self) -> None:
        self.assertEqual(self.audit["dataset"]["public_candidate_audio_count"], 848)
        self.assertEqual(self.audit["dataset"]["paper_candidate_audio_count"], 847)
        self.assertFalse(self.audit["protocol_boundary"]["strict_paper_reproduction"])
        self.assertTrue(self.audit["protocol_boundary"]["no_post_hoc_selection"])


if __name__ == "__main__":
    unittest.main()
