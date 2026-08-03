from __future__ import annotations

import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPOSITORY_ROOT / "results/audits/clap_negative_uiq_eval_20260803.json"


class ClapNegativeUIQEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    def test_nine_complete_runs_and_hashes_are_pinned(self) -> None:
        self.assertEqual(self.audit["status"], "complete")
        self.assertEqual(self.audit["execution"]["run_count"], 9)
        self.assertEqual(self.audit["execution"]["complete_run_count"], 9)
        self.assertEqual(
            set(self.audit["models"]),
            {"LAION-CLAP", "MGA-CLAP", "Robust-CLAP"},
        )
        expected_counts = {
            "audiocaps": (975, 630),
            "clotho": (1045, 542),
            "mecat": (848, 409),
        }
        for model in self.audit["models"].values():
            self.assertEqual(set(model["datasets"]), set(expected_counts))
            for name, row in model["datasets"].items():
                self.assertEqual(
                    (row["candidate_count"], row["query_count"]),
                    expected_counts[name],
                )
                for field in (
                    "metrics_sha256",
                    "audio_npz_sha256",
                    "query_npz_sha256",
                    "pairing_sha256",
                ):
                    self.assertEqual(len(row[field]), 64)

    def test_means_deltas_and_maxima_recompute_exactly(self) -> None:
        for model in self.audit["models"].values():
            datasets = model["datasets"].values()
            means = model["three_dataset_mean"]
            deltas = model["comparison"]["signed_delta"]
            for metric, observed_mean in means.items():
                recomputed = sum(row["metrics"][metric] for row in datasets) / 3
                self.assertAlmostEqual(observed_mean, recomputed, places=12)
                self.assertAlmostEqual(
                    deltas[metric],
                    observed_mean - model["paper_values"][metric],
                    places=12,
                )
            max_metric = max(deltas, key=lambda metric: abs(deltas[metric]))
            self.assertEqual(
                model["comparison"]["maximum_absolute_delta_metric"], max_metric
            )
            self.assertAlmostEqual(
                model["comparison"]["maximum_absolute_delta"],
                abs(deltas[max_metric]),
                places=12,
            )

    def test_claim_boundary_and_robust_checkpoint_remain_controlled(self) -> None:
        self.assertFalse(self.audit["protocol"]["strict_paper_reproduction"])
        self.assertIn("INFERRED", self.audit["protocol"]["source"])
        self.assertFalse(
            self.audit["models"]["Robust-CLAP"]["resource_binding"]
            ["strict_paper_checkpoint_identity"]
        )
        for model in self.audit["models"].values():
            self.assertIn("CONTROLLED_COMPLETE", model["comparison"]["assessment"])


if __name__ == "__main__":
    unittest.main()
