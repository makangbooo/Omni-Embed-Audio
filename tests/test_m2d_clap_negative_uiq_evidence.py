from __future__ import annotations

import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    REPOSITORY_ROOT
    / "results/audits/m2d_clap_negative_uiq_eval_20260803.json"
)


class M2DClapNegativeUIQEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    def test_three_complete_dataset_runs_are_pinned(self) -> None:
        self.assertEqual(self.audit["status"], "complete")
        datasets = self.audit["datasets"]
        self.assertEqual(set(datasets), {"audiocaps", "clotho", "mecat"})
        self.assertEqual(
            {name: row["query_count"] for name, row in datasets.items()},
            {"audiocaps": 630, "clotho": 542, "mecat": 409},
        )
        for row in datasets.values():
            self.assertEqual(len(row["metrics_sha256"]), 64)
            self.assertEqual(len(row["query_npz_sha256"]), 64)
            self.assertEqual(len(row["pairing_sha256"]), 64)

    def test_means_and_deltas_recompute_exactly(self) -> None:
        datasets = self.audit["datasets"].values()
        means = self.audit["three_dataset_mean"]
        paper = self.audit["paper_values"]
        deltas = self.audit["comparison"]["signed_delta"]
        for metric, observed_mean in means.items():
            recomputed = sum(row["metrics"][metric] for row in datasets) / 3
            self.assertAlmostEqual(observed_mean, recomputed, places=12)
            self.assertAlmostEqual(
                deltas[metric], observed_mean - paper[metric], places=12
            )

    def test_claim_boundary_remains_controlled(self) -> None:
        protocol = self.audit["protocol"]
        self.assertFalse(protocol["strict_paper_reproduction"])
        self.assertIn("INFERRED", protocol["source"])
        self.assertEqual(
            self.audit["comparison"]["assessment"],
            "CONTROLLED_COMPLETE_STRICT_PAPER_PROTOCOL_UNAVAILABLE",
        )
        self.assertEqual(
            self.audit["datasets"]["clotho"]["candidate_id_resolution"],
            {
                "target_ids": {"unique_casefold_or_stem_filename": 542},
                "hard_negative_ids": {
                    "unique_casefold_or_stem_filename": 542
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
