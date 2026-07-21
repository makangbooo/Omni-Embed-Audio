from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.evaluate_frozen_text_index import evaluate, run_evaluation


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class FrozenTextIndexEvaluationTest(unittest.TestCase):
    def test_graded_qrels_metrics_and_ties_are_deterministic(self) -> None:
        queries = np.asarray([[1.0, 0.0]], dtype=np.float32)
        candidates = np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
        )
        metrics, indices, scores, rows = evaluate(
            queries,
            ["q0"],
            candidates,
            ["d0", "d1", "d2"],
            {"q0": {"d1": 2.0}},
            cutoffs=[1, 5, 10],
            ranking_depth=3,
        )

        self.assertEqual(indices.tolist(), [[0, 1, 2]])
        self.assertEqual(scores[0, 0], scores[0, 1])
        self.assertEqual(rows[0]["best_relevant_rank"], 2)
        self.assertEqual(metrics["R@1"], 0.0)
        self.assertEqual(metrics["R@5"], 100.0)
        self.assertAlmostEqual(metrics["MRR@10"], 0.5)
        self.assertAlmostEqual(metrics["nDCG@10"], 1.0 / np.log2(3.0))

    def test_run_writes_frozen_index_identity_and_complete_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "queries.npy", np.eye(2, dtype=np.float32))
            np.save(root / "candidates.npy", np.eye(2, dtype=np.float32))
            write_jsonl(
                root / "queries.jsonl",
                [{"query_id": "q0"}, {"query_id": "q1"}],
            )
            write_jsonl(
                root / "candidates.jsonl",
                [{"candidate_id": "d0"}, {"candidate_id": "d1"}],
            )
            write_jsonl(
                root / "qrels.jsonl",
                [
                    {"query_id": "q0", "candidate_id": "d0", "relevance": 1},
                    {"query_id": "q1", "candidate_id": "d1", "relevance": 1},
                ],
            )
            config = {
                "schema_version": 1,
                "experiment_id": "synthetic_frozen_index",
                "model": "synthetic-model",
                "checkpoint": "synthetic-checkpoint",
                "dataset": "synthetic",
                "protocol_label": "synthetic-frozen-index",
                "protocol_source": "INFERRED",
                "seed": 42,
                "query_embeddings": "queries.npy",
                "query_metadata": "queries.jsonl",
                "candidate_embeddings": "candidates.npy",
                "candidate_metadata": "candidates.jsonl",
                "qrels": "qrels.jsonl",
                "cutoffs": [1, 5, 10],
                "ranking_depth": 10,
                "require_clean_git": False,
            }
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            output = root / "synthetic_frozen_index"
            with patch(
                "scripts.evaluate_frozen_text_index.git_output",
                side_effect=["abc123", ""],
            ):
                report = run_evaluation(root / "config.json", output)

            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["metrics"]["R@1"], 100.0)
            self.assertEqual(report["metrics"]["MRR@10"], 1.0)
            self.assertEqual(report["metrics"]["nDCG@10"], 1.0)
            self.assertEqual(report["candidate_count"], 2)
            identity = json.loads(
                (output / "frozen_index_identity.json").read_text(encoding="utf-8")
            )
            self.assertTrue(identity["unchanged_during_evaluation"])
            self.assertEqual(identity["candidate_count"], 2)
            self.assertTrue((output / "per_query.jsonl").is_file())
            self.assertTrue((output / "top_ranking_indices.npy").is_file())

    def test_missing_qrel_candidate_is_preserved_as_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "queries.npy", np.eye(1, dtype=np.float32))
            np.save(root / "candidates.npy", np.eye(1, dtype=np.float32))
            write_jsonl(root / "queries.jsonl", [{"query_id": "q0"}])
            write_jsonl(root / "candidates.jsonl", [{"candidate_id": "d0"}])
            write_jsonl(
                root / "qrels.jsonl",
                [{"query_id": "q0", "candidate_id": "missing", "relevance": 1}],
            )
            config = {
                "schema_version": 1,
                "experiment_id": "failed_frozen_index",
                "model": "synthetic-model",
                "checkpoint": "synthetic-checkpoint",
                "dataset": "synthetic",
                "protocol_label": "synthetic",
                "protocol_source": "INFERRED",
                "seed": 42,
                "query_embeddings": "queries.npy",
                "query_metadata": "queries.jsonl",
                "candidate_embeddings": "candidates.npy",
                "candidate_metadata": "candidates.jsonl",
                "qrels": "qrels.jsonl",
                "require_clean_git": False,
            }
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            output = root / "failed_frozen_index"
            with (
                patch(
                    "scripts.evaluate_frozen_text_index.git_output",
                    side_effect=["abc123", ""],
                ),
                self.assertRaisesRegex(KeyError, "absent from frozen index"),
            ):
                run_evaluation(root / "config.json", output)

            report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertIn("absent from frozen index", report["error"])


if __name__ == "__main__":
    unittest.main()
