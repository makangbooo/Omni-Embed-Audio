from __future__ import annotations

import json
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    REPOSITORY_ROOT
    / "results/audits/nemo3b_cl_clotho_rtx4090_efficiency_20260803.json"
)


class Nemo3bClEfficiencyEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    def test_run_is_complete_and_clean(self) -> None:
        self.assertEqual(self.audit["status"], "complete")
        self.assertEqual(self.audit["exit_code"], 0)
        self.assertEqual(self.audit["git_status_short"], "")
        self.assertEqual(
            self.audit["hardware"]["gpu_name"], "NVIDIA GeForce RTX 4090"
        )
        self.assertEqual(self.audit["protocol"]["audio_measurement_count"], 1045)
        self.assertEqual(self.audit["protocol"]["text_measurement_count"], 5225)

    def test_measured_values_and_artifact_hashes_are_fixed(self) -> None:
        self.assertEqual(self.audit["results"]["audio"]["count"], 1045)
        self.assertAlmostEqual(
            self.audit["results"]["audio"]["mean_ms"], 530.3152055244019
        )
        self.assertEqual(self.audit["results"]["text"]["count"], 5225)
        self.assertAlmostEqual(
            self.audit["results"]["text"]["mean_ms"], 41.71248486660287
        )
        self.assertEqual(
            self.audit["results"]["parameters"]["total_trainable_parameters"],
            14714880,
        )
        self.assertEqual(
            self.audit["artifacts"]["latencies"]["sha256"],
            "32028787a10210c54a64496770cdbb161027eb4504274e09b818507c8787f36b",
        )
        for artifact in self.audit["artifacts"].values():
            self.assertEqual(len(artifact["sha256"]), 64)

    def test_paper_comparison_remains_non_strict(self) -> None:
        self.assertEqual(self.audit["claim_scope"]["result_label"], "INFERRED")
        self.assertEqual(self.audit["strict_paper_assessment"]["status"], "blocked")
        self.assertEqual(
            self.audit["paper_reference"]["audio_ms_per_clip"], 163.8
        )
        self.assertEqual(self.audit["paper_reference"]["text_ms_per_query"], 2.3)


if __name__ == "__main__":
    unittest.main()
