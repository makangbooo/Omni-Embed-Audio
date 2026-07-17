from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.evaluate_negative_embedding_artifacts import run_evaluation


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def write_fixture(
    root: Path,
    *,
    query_rows: list[dict] | None = None,
    candidate_rows: list[dict] | None = None,
    pairing_rows: list[dict] | None = None,
    queries: np.ndarray | None = None,
    candidates: np.ndarray | None = None,
    experiment_id: str = "synthetic_negative",
) -> Path:
    if queries is None:
        queries = np.asarray(
            [[4.0, 3.0, 2.0, 1.0]] * 3, dtype=np.float32
        )
    if candidates is None:
        candidates = np.eye(4, dtype=np.float32)
    if query_rows is None:
        query_rows = [{"query_id": value} for value in ("q0", "q1", "q2")]
    if candidate_rows is None:
        candidate_rows = [
            {"candidate_id": value} for value in ("a", "b", "c", "d")
        ]
    if pairing_rows is None:
        # Deliberately not in query-metadata order.
        pairing_rows = [
            {"query_id": "q2", "target_id": "d", "hard_negative_id": "a"},
            {"query_id": "q0", "target_id": "a", "hard_negative_id": "d"},
            {"query_id": "q1", "target_id": "b", "hard_negative_id": "d"},
        ]
    np.save(root / "queries.npy", queries)
    np.save(root / "candidates.npy", candidates)
    write_jsonl(root / "queries.jsonl", query_rows)
    write_jsonl(root / "candidates.jsonl", candidate_rows)
    write_jsonl(root / "pairings.jsonl", pairing_rows)
    config = {
        "schema_version": 1,
        "checkpoint": "synthetic-checkpoint",
        "dataset": "synthetic",
        "experiment_id": experiment_id,
        "ks": [1, 3],
        "model": "synthetic-model",
        "pairing_metadata": "pairings.jsonl",
        "paper_table": "test-only",
        "protocol_label": "explicit-synthetic-pairing",
        "protocol_source": "INFERRED",
        "query_embeddings": "queries.npy",
        "candidate_embeddings": "candidates.npy",
        "query_metadata": "queries.jsonl",
        "candidate_metadata": "candidates.jsonl",
        "query_selection": "all",
        "normalize_embeddings": False,
        "require_clean_git": False,
        "seed": 42,
        "task": "negative",
    }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


class EvaluateNegativeEmbeddingArtifactsTest(unittest.TestCase):
    def test_complete_bundle_uses_explicit_pairings_and_exact_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = write_fixture(root)
            output = root / "synthetic_negative"
            with patch(
                "scripts.evaluate_negative_embedding_artifacts.git_output",
                side_effect=["abc123", ""],
            ):
                report = run_evaluation(
                    config_path=config_path,
                    output_dir=output,
                    argv=["python", "negative-evaluate", "--test"],
                )

            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["pairing_inference_used"], False)
            self.assertEqual(report["tie_policy"], "optimistic_strict_greater")
            self.assertEqual(report["evaluated_query_count"], 3)
            self.assertAlmostEqual(report["metrics"]["R@1"], 100.0 / 3.0)
            self.assertAlmostEqual(report["metrics"]["R@3"], 200.0 / 3.0)
            self.assertAlmostEqual(
                report["metrics"]["Delta-Rank"], 2.0 / 3.0
            )
            self.assertAlmostEqual(report["metrics"]["HNSR"], 200.0 / 3.0)
            self.assertAlmostEqual(report["metrics"]["HNSR@1"], 100.0 / 3.0)
            self.assertAlmostEqual(report["metrics"]["HNSR@3"], 200.0 / 3.0)
            self.assertAlmostEqual(report["metrics"]["TFR"], 100.0 / 3.0)
            self.assertAlmostEqual(
                report["metrics"]["TFR-HN@3"], 100.0 / 3.0
            )
            np.testing.assert_array_equal(
                np.load(output / "target_ranks.npy"), [1, 2, 4]
            )
            np.testing.assert_array_equal(
                np.load(output / "hard_negative_ranks.npy"), [4, 4, 1]
            )
            rows = [
                json.loads(line)
                for line in (output / "per_query.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([row["query_id"] for row in rows], ["q0", "q1", "q2"])
            self.assertEqual(rows[0]["target_id"], "a")
            self.assertEqual(rows[0]["hard_negative_id"], "d")
            self.assertTrue(rows[0]["HNSR@3"])
            for name in (
                "config.yaml",
                "command.sh",
                "environment.txt",
                "git_commit.txt",
                "metrics.json",
                "query_embeddings.npy",
                "candidate_embeddings.npy",
                "query_metadata.jsonl",
                "candidate_metadata.jsonl",
                "pairing_metadata.jsonl",
                "target_ranks.npy",
                "hard_negative_ranks.npy",
                "rankings.npy",
                "similarities.npy",
                "target_indices.npy",
                "hard_negative_indices.npy",
                "evaluated_query_indices.npy",
                "per_query.jsonl",
            ):
                self.assertTrue((output / name).is_file(), name)
                if name in report.get("artifacts", {}):
                    self.assertEqual(len(report["artifacts"][name]["sha256"]), 64)

    def test_missing_hard_negative_candidate_fails_without_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = write_fixture(
                root,
                pairing_rows=[
                    {"query_id": "q0", "target_id": "a", "hard_negative_id": "x"},
                    {"query_id": "q1", "target_id": "b", "hard_negative_id": "d"},
                    {"query_id": "q2", "target_id": "d", "hard_negative_id": "a"},
                ],
            )
            output = root / "synthetic_negative"
            with (
                patch(
                    "scripts.evaluate_negative_embedding_artifacts.git_output",
                    side_effect=["abc123", ""],
                ),
                self.assertRaisesRegex(KeyError, "hard-negative IDs"),
            ):
                run_evaluation(config_path=config_path, output_dir=output)

            report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["pairing_inference_used"])

    def test_pairing_coverage_must_exactly_match_queries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = write_fixture(
                root,
                pairing_rows=[
                    {"query_id": "q0", "target_id": "a", "hard_negative_id": "d"},
                    {"query_id": "q1", "target_id": "b", "hard_negative_id": "d"},
                    {"query_id": "extra", "target_id": "c", "hard_negative_id": "d"},
                ],
            )
            with (
                patch(
                    "scripts.evaluate_negative_embedding_artifacts.git_output",
                    side_effect=["abc123", ""],
                ),
                self.assertRaisesRegex(ValueError, "exactly match"),
            ):
                run_evaluation(
                    config_path=config_path, output_dir=root / "synthetic_negative"
                )

    def test_duplicate_pairing_query_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = write_fixture(
                root,
                pairing_rows=[
                    {"query_id": "q0", "target_id": "a", "hard_negative_id": "d"},
                    {"query_id": "q0", "target_id": "b", "hard_negative_id": "d"},
                    {"query_id": "q2", "target_id": "d", "hard_negative_id": "a"},
                ],
            )
            with (
                patch(
                    "scripts.evaluate_negative_embedding_artifacts.git_output",
                    side_effect=["abc123", ""],
                ),
                self.assertRaisesRegex(ValueError, "duplicate pairing"),
            ):
                run_evaluation(
                    config_path=config_path, output_dir=root / "synthetic_negative"
                )

    def test_target_and_hard_negative_must_differ(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = write_fixture(
                root,
                pairing_rows=[
                    {"query_id": "q0", "target_id": "a", "hard_negative_id": "a"},
                    {"query_id": "q1", "target_id": "b", "hard_negative_id": "d"},
                    {"query_id": "q2", "target_id": "d", "hard_negative_id": "a"},
                ],
            )
            with (
                patch(
                    "scripts.evaluate_negative_embedding_artifacts.git_output",
                    side_effect=["abc123", ""],
                ),
                self.assertRaisesRegex(ValueError, "must differ"),
            ):
                run_evaluation(
                    config_path=config_path, output_dir=root / "synthetic_negative"
                )

    def test_ties_use_optimistic_rank_and_stable_full_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = write_fixture(
                root,
                queries=np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32),
                candidates=np.eye(3, dtype=np.float32),
                query_rows=[{"query_id": "q0"}],
                candidate_rows=[
                    {"candidate_id": "a"},
                    {"candidate_id": "b"},
                    {"candidate_id": "c"},
                ],
                pairing_rows=[
                    {"query_id": "q0", "target_id": "b", "hard_negative_id": "a"}
                ],
            )
            with patch(
                "scripts.evaluate_negative_embedding_artifacts.git_output",
                side_effect=["abc123", ""],
            ):
                report = run_evaluation(
                    config_path=config_path, output_dir=root / "synthetic_negative"
                )

            np.testing.assert_array_equal(
                np.load(root / "synthetic_negative" / "target_ranks.npy"), [1]
            )
            np.testing.assert_array_equal(
                np.load(root / "synthetic_negative" / "hard_negative_ranks.npy"), [1]
            )
            np.testing.assert_array_equal(
                np.load(root / "synthetic_negative" / "rankings.npy"), [[0, 1, 2]]
            )
            self.assertEqual(report["metrics"]["HNSR"], 0.0)
            self.assertEqual(report["metrics"]["Delta-Rank"], 0.0)

    def test_existing_artifacts_are_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            (output / "metrics.json").write_text("original\n", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "prior artifacts"):
                run_evaluation(
                    config_path=root / "missing-config.json", output_dir=output
                )
            self.assertEqual(
                (output / "metrics.json").read_text(encoding="utf-8"), "original\n"
            )


if __name__ == "__main__":
    unittest.main()
