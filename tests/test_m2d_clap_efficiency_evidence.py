from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results/audits/m2d_clap_clotho_rtx4090_efficiency_20260803.json"


class M2dClapEfficiencyEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(AUDIT.read_text(encoding="utf-8"))

    def test_run_and_counts_are_complete(self) -> None:
        self.assertEqual(self.audit["status"], "complete")
        self.assertEqual(self.audit["exit_code"], 0)
        self.assertEqual(self.audit["git_status_short"], "")
        self.assertEqual(self.audit["results"]["audio"]["count"], 1045)
        self.assertEqual(self.audit["results"]["text"]["count"], 5225)
        self.assertEqual(self.audit["hardware"]["gpu_name"], "NVIDIA GeForce RTX 4090")

    def test_values_and_hashes_are_fixed(self) -> None:
        self.assertAlmostEqual(self.audit["results"]["audio"]["mean_ms"], 87.64511397990431)
        self.assertAlmostEqual(self.audit["results"]["text"]["mean_ms"], 5.66861868861244)
        self.assertEqual(self.audit["results"]["parameters"]["trainable_parameters"], 89041920)
        self.assertEqual(
            self.audit["artifacts"]["latencies"]["sha256"],
            "f3b42a1893a75834ac7ae433e8ca80cb9de3befd2e904bd7d8d29050125995f8",
        )

    def test_strict_paper_boundary_is_preserved(self) -> None:
        self.assertEqual(self.audit["claim_scope"]["result_label"], "INFERRED")
        self.assertEqual(self.audit["strict_paper_assessment"]["status"], "blocked")
        self.assertEqual(self.audit["paper_reference"]["audio_ms_per_clip"], 58.1)


if __name__ == "__main__":
    unittest.main()
