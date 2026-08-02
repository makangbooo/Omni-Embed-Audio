from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts/build_table11_public_controlled_means.py"
AUDIT_PATH = (
    REPOSITORY_ROOT
    / "results/audits/table11_public_controlled_means_eval_20260803.json"
)


def load_module():
    spec = importlib.util.spec_from_file_location("table11_builder", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildTable11PublicControlledMeansTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()
        cls.audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    def test_generated_audit_is_current(self) -> None:
        self.assertEqual(self.audit, self.module.build_audit())

    def test_all_models_tasks_and_metrics_are_complete(self) -> None:
        self.assertEqual(self.audit["status"], "complete")
        self.assertEqual(self.audit["model_count"], 13)
        self.assertEqual(self.audit["task_count"], 2)
        self.assertEqual(self.audit["result_count"], 26)
        self.assertEqual(self.audit["metric_count"], 78)
        keys = {(row["model"], row["task"]) for row in self.audit["results"]}
        self.assertEqual(len(keys), 26)
        for row in self.audit["results"]:
            self.assertEqual(set(row["input_values"]), set(self.module.DATASETS))
            self.assertEqual(set(row["reproduced_mean"]), set(self.module.METRICS))

    def test_means_and_deltas_are_arithmetically_exact(self) -> None:
        for row in self.audit["results"]:
            for metric in self.module.METRICS:
                expected_mean = round(
                    sum(
                        row["input_values"][dataset][metric]
                        for dataset in self.module.DATASETS
                    )
                    / 3,
                    6,
                )
                self.assertEqual(row["reproduced_mean"][metric], expected_mean)
                self.assertEqual(
                    row["signed_delta_percentage_points"][metric],
                    round(expected_mean - row["paper_mean"][metric], 6),
                )

    def test_protocol_and_claim_boundaries_are_explicit(self) -> None:
        self.assertTrue(self.audit["protocol"]["no_post_hoc_protocol_selection"])
        self.assertIn("848-row", self.audit["protocol"]["mecat"])
        self.assertFalse(self.audit["claim_boundary"]["strict_paper_reproduction"])


if __name__ == "__main__":
    unittest.main()
