from __future__ import annotations

import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPOSITORY_ROOT / "results/audits/oea_negative_uiq_eval_20260803.json"
PAPER_REGISTRY = REPOSITORY_ROOT / "configs/results/paper_reported_metrics.jsonl"
MODELS = {
    "OEA-Nemo3B",
    "OEA-Nemo3B (+Cl)",
    "OEA-Qwen3B",
    "OEA-Qwen3B (+Cl)",
    "OEA-Qwen7B",
    "OEA-Qwen7B (+Cl)",
}
METRICS = {"R@5", "R@10", "Delta-Rank", "HNSR", "HNSR@10", "TFR", "TFR-HN@10"}


class OeaNegativeUIQEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    def test_six_models_and_eighteen_dataset_runs_are_complete(self) -> None:
        self.assertEqual(self.audit["status"], "complete")
        self.assertEqual(set(self.audit["models"]), MODELS)
        self.assertEqual(self.audit["execution"]["model_run_count"], 6)
        self.assertEqual(self.audit["execution"]["dataset_run_count"], 18)
        self.assertEqual(self.audit["execution"]["run_rc"], 0)
        self.assertEqual(self.audit["execution"]["matrix_status"], "complete")
        self.assertTrue(self.audit["execution"]["batch_log"].endswith(".log"))
        expected_counts = {
            "audiocaps": (975, 630),
            "clotho": (1045, 542),
            "mecat": (848, 409),
        }
        for model in self.audit["models"].values():
            self.assertEqual(set(model["datasets"]), set(expected_counts))
            self.assertEqual(len(model["artifact_manifest_sha256"]), 64)
            self.assertEqual(len(model["resource_binding"]["checkpoint_sha256"]), 64)
            for name, row in model["datasets"].items():
                self.assertEqual(
                    (row["candidate_count"], row["query_count"]),
                    expected_counts[name],
                )
                self.assertEqual(set(row["metrics"]), METRICS)
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

    def test_paper_values_match_the_committed_registry(self) -> None:
        expected = {model: {} for model in MODELS}
        for line in PAPER_REGISTRY.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["paper_table"] != "Table 17" or row["model"] not in MODELS:
                continue
            metric = "TFR-HN@10" if row["metric"] == "TFR@10" else row["metric"]
            expected[row["model"]][metric] = float(row["paper_value"])
        for model, evidence in self.audit["models"].items():
            self.assertEqual(evidence["paper_values"], expected[model])

    def test_claim_boundary_remains_controlled(self) -> None:
        self.assertFalse(self.audit["protocol"]["strict_paper_reproduction"])
        self.assertIn("INFERRED", self.audit["protocol"]["source"])
        self.assertIn("848", self.audit["protocol"]["claim_boundary"])
        for model in self.audit["models"].values():
            self.assertEqual(
                model["comparison"]["assessment"],
                "CONTROLLED_COMPLETE_STRICT_PAPER_PROTOCOL_UNAVAILABLE",
            )


if __name__ == "__main__":
    unittest.main()
