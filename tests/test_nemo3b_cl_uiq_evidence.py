from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_AUDIT_PATH = (
    REPOSITORY_ROOT
    / "results/audits/nemo3b_cl_clotho_positive_uiq_embeddings_20260729.json"
)
EVALUATION_AUDIT_PATH = (
    REPOSITORY_ROOT
    / "results/audits/nemo3b_cl_clotho_positive_uiq_eval_20260729.json"
)
OBSERVATIONS_PATH = (
    REPOSITORY_ROOT / "results/observations/reproduction_observations.jsonl"
)


class Nemo3bClPositiveUiqEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.embedding_audit = json.loads(
            EMBEDDING_AUDIT_PATH.read_text(encoding="utf-8")
        )
        cls.evaluation_audit = json.loads(
            EVALUATION_AUDIT_PATH.read_text(encoding="utf-8")
        )
        cls.observations = [
            json.loads(line)
            for line in OBSERVATIONS_PATH.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def test_embedding_generation_completed_on_the_recorded_gpu(self) -> None:
        run = self.embedding_audit["run"]
        self.assertEqual(self.embedding_audit["status"], "complete")
        self.assertEqual(run["status"], "complete")
        self.assertEqual(run["attempt_exit_code"], 0)
        self.assertIsNone(run["outer_wrapper_final_run_rc"])
        self.assertEqual(run["query_count"], 4180)
        self.assertEqual(run["completed_text_chunks"], 4180)
        self.assertEqual(run["pending_text_chunks"], 0)
        self.assertEqual(run["query_embedding_shape"], [4180, 512])
        self.assertEqual(run["gpu_preflight"]["gpu_name"], "NVIDIA GeForce RTX 4090")
        self.assertEqual(run["gpu_preflight"]["gpu_total_memory_bytes"], 25250627584)
        self.assertTrue(run["gpu_preflight"]["bf16_supported"])
        self.assertEqual(
            self.embedding_audit["execution_evidence"]["all_discovered_exit_code_values"],
            [0, 0, 0, 0, 0, 0],
        )

    def test_embedding_artifact_hashes_are_fixed(self) -> None:
        self.assertEqual(
            self.embedding_audit["run"]["artifacts"],
            {
                "generation_metrics.json": {
                    "size_bytes": 9645,
                    "sha256": "15e7df87be1aa23be065eccb49b256b4d3695209a150e053c864ffc4f2bb7705",
                },
                "run_identity.json": {
                    "size_bytes": 3692,
                    "sha256": "29d2f7300878240b15004a853b44d14019d874bcf01b360347a3b4ad1abcd25a",
                },
                "query_embeddings.npy": {
                    "size_bytes": 8560768,
                    "sha256": "f09134da2eaaf8b4a7e9fd577c41f76bf8321853c172cd4549472e617c12042f",
                },
                "query_metadata.jsonl": {
                    "size_bytes": 1635369,
                    "sha256": "27037a7ba1dc1fcb097bdadcb1a723ee0a41eab0a8fd186bdc2dabb5d4d22d31",
                },
            },
        )

    def test_suite_and_all_four_protocols_completed(self) -> None:
        suite = self.evaluation_audit["suite"]
        self.assertEqual(self.evaluation_audit["status"], "complete")
        self.assertEqual(suite["status"], "complete")
        self.assertEqual(suite["attempt_exit_code"], 0)
        self.assertEqual(suite["protocol_count"], 4)
        self.assertEqual(suite["all_suite_and_protocol_exit_codes"], [0, 0, 0, 0, 0])
        self.assertEqual(
            suite["artifacts"]["suite_metrics.json"]["sha256"],
            "ddd7f3526c69bd12d7b43ef658fea6f34cf3e52ef49ed151d251d9587402060b",
        )
        self.assertEqual(
            suite["artifacts"]["positive_uiq_summary.csv"]["sha256"],
            "02a4ad97610e51ff926c013b77bbbcc70f6acba6dfae32e860fb66c0249428e4",
        )

    def test_four_metric_blocks_and_close_comparisons_are_exact(self) -> None:
        expected = {
            "question_released_uiq": (23.636363636363637, 50.8133971291866, 63.92344497607656, 0.1963636363636354),
            "imperative_released_uiq": (24.019138755980862, 51.100478468899524, 64.97607655502392, 0.28607655502392504),
            "paraphrase_released_uiq": (23.54066985645933, 49.952153110047846, 64.40191387559808, 0.289330143540667),
            "tagging_released_uiq": (25.837320574162682, 52.63157894736842, 66.1244019138756, 0.09440191387560048),
        }
        self.assertEqual(set(self.evaluation_audit["protocols"]), set(expected))
        for protocol_id, (r1, r5, r10, maximum_delta) in expected.items():
            protocol = self.evaluation_audit["protocols"][protocol_id]
            self.assertEqual(protocol["protocol_source"], "CODE")
            self.assertEqual(protocol["evaluated_queries"], 1045)
            self.assertEqual(protocol["metrics"]["R@1"], r1)
            self.assertEqual(protocol["metrics"]["R@5"], r5)
            self.assertEqual(protocol["metrics"]["R@10"], r10)
            comparison = self.evaluation_audit["comparison"][protocol_id]
            self.assertEqual(comparison["status"], "close")
            self.assertEqual(
                comparison["maximum_absolute_delta_percentage_points"],
                maximum_delta,
            )

    def test_twelve_recall_observations_bind_to_this_exact_audit(self) -> None:
        relevant = [
            row
            for row in self.observations
            if row["observation_id"].startswith("nemo3b_cl_clotho_uiq_")
        ]
        self.assertEqual(len(relevant), 12)
        expected_hash = hashlib.sha256(EVALUATION_AUDIT_PATH.read_bytes()).hexdigest()
        for row in relevant:
            self.assertEqual(row["status"], "close")
            self.assertEqual(
                row["evidence"]["size_bytes"], EVALUATION_AUDIT_PATH.stat().st_size
            )
            self.assertEqual(row["evidence"]["sha256"], expected_hash)

    def test_protocol_boundary_is_code_close_not_strict_paper(self) -> None:
        boundary = self.evaluation_audit["protocol_boundary"]
        self.assertEqual(boundary["source"], "CODE")
        self.assertIn("omits the passage: prefix", boundary["note"])
        self.assertIn("rather than strict paper-protocol", boundary["note"])


if __name__ == "__main__":
    unittest.main()
