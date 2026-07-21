from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.evaluate_embedding_artifacts import run_evaluation


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class EvaluateEmbeddingArtifactsTest(unittest.TestCase):
    def test_a2t_writes_multi_positive_audit_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "audio.npy", np.eye(2, dtype=np.float32))
            np.save(
                root / "captions.npy",
                np.asarray(
                    [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
                    dtype=np.float32,
                ),
            )
            write_jsonl(
                root / "audio.jsonl",
                [{"candidate_id": "a"}, {"candidate_id": "b"}],
            )
            write_jsonl(
                root / "captions.jsonl",
                [
                    {"clip_id": "a"},
                    {"clip_id": "a"},
                    {"clip_id": "b"},
                    {"clip_id": "b"},
                ],
            )
            config = {
                "schema_version": 1,
                "checkpoint": "synthetic-checkpoint",
                "dataset": "synthetic",
                "experiment_id": "synthetic_a2t",
                "model": "synthetic-model",
                "task": "a2t",
                "paper_table": "not-paper-reported",
                "protocol_label": "all-caption-multi-positive",
                "protocol_source": "CODE",
                "query_embeddings": "audio.npy",
                "candidate_embeddings": "captions.npy",
                "query_metadata": "audio.jsonl",
                "candidate_metadata": "captions.jsonl",
                "query_target_id_field": "candidate_id",
                "candidate_group_id_field": "clip_id",
                "query_selection": "all",
                "require_clean_git": False,
                "seed": 42,
            }
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            output = root / "synthetic_a2t"
            with patch(
                "scripts.evaluate_embedding_artifacts.git_output",
                side_effect=["abc123", ""],
            ):
                report = run_evaluation(
                    config_path=root / "config.json", output_dir=output
                )

            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["task"], "a2t")
            self.assertEqual(report["evaluated_query_count"], 2)
            self.assertEqual(report["candidate_count"], 4)
            self.assertEqual(report["metrics"]["R@1"], 100.0)
            self.assertEqual(
                json.loads(
                    (output / "positive_indices.json").read_text(encoding="utf-8")
                ),
                [[0, 1], [2, 3]],
            )

    def test_t2a_writes_complete_audit_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "queries.npy", np.eye(2, dtype=np.float32))
            np.save(root / "candidates.npy", np.eye(2, dtype=np.float32))
            write_jsonl(
                root / "queries.jsonl",
                [
                    {"query_id": "q0", "target_id": "a"},
                    {"query_id": "q1", "target_id": "b"},
                ],
            )
            write_jsonl(
                root / "candidates.jsonl",
                [{"candidate_id": "a"}, {"candidate_id": "b"}],
            )
            config = {
                "schema_version": 1,
                "checkpoint": "synthetic-checkpoint",
                "dataset": "synthetic",
                "experiment_id": "synthetic_t2a",
                "model": "synthetic-model",
                "task": "t2a",
                "paper_table": "test-only",
                "protocol_label": "synthetic",
                "protocol_source": "INFERRED",
                "query_embeddings": "queries.npy",
                "candidate_embeddings": "candidates.npy",
                "query_metadata": "queries.jsonl",
                "candidate_metadata": "candidates.jsonl",
                "query_selection": "all",
                "require_clean_git": False,
                "seed": 42,
            }
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            output = root / "synthetic_t2a"
            with (
                patch(
                    "scripts.evaluate_embedding_artifacts.git_output",
                    side_effect=["abc123", ""],
                ),
            ):
                report = run_evaluation(
                    config_path=root / "config.json",
                    output_dir=output,
                    argv=["python", "evaluate", "--test"],
                )

            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["seed"], 42)
            self.assertFalse(report["randomness_used_by_evaluator"])
            self.assertEqual(report["metrics"]["R@1"], 100.0)
            self.assertEqual(report["evaluated_query_count"], 2)
            for name in (
                "config.yaml",
                "command.sh",
                "environment.txt",
                "git_commit.txt",
                "metrics.json",
                "query_embeddings.npy",
                "candidate_embeddings.npy",
                "ranks.npy",
                "rankings.npy",
                "similarities.npy",
                "positive_indices.json",
                "ignored_indices.json",
                "evaluated_query_indices.json",
            ):
                self.assertTrue((output / name).is_file(), name)
            saved = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["git_commit"], "abc123")
            self.assertEqual(
                saved["artifacts"]["rankings.npy"]["sha256"],
                report["artifacts"]["rankings.npy"]["sha256"],
            )

    def test_t2t_requires_explicit_query_selection_and_saves_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = np.array(
                [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
                dtype=np.float32,
            )
            np.save(root / "captions.npy", embeddings)
            write_jsonl(
                root / "captions.jsonl",
                [
                    {"clip_id": "a"},
                    {"clip_id": "a"},
                    {"clip_id": "b"},
                    {"clip_id": "b"},
                ],
            )
            (root / "indices.json").write_text("[2, 0]\n", encoding="utf-8")
            config = {
                "schema_version": 1,
                "checkpoint": "synthetic-checkpoint",
                "dataset": "synthetic",
                "experiment_id": "synthetic_t2t",
                "model": "synthetic-model",
                "task": "t2t",
                "paper_table": "test-only",
                "protocol_label": "explicit-two-query",
                "protocol_source": "INFERRED",
                "query_embeddings": "captions.npy",
                "query_metadata": "captions.jsonl",
                "query_selection": "indices",
                "query_indices": "indices.json",
                "require_clean_git": False,
                "seed": 42,
            }
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            output = root / "synthetic_t2t"
            with patch(
                "scripts.evaluate_embedding_artifacts.git_output",
                side_effect=["abc123", ""],
            ):
                report = run_evaluation(
                    config_path=root / "config.json", output_dir=output
                )

            self.assertEqual(report["evaluated_query_count"], 2)
            self.assertEqual(
                json.loads(
                    (output / "evaluated_query_indices.json").read_text(
                        encoding="utf-8"
                    )
                ),
                [2, 0],
            )

    def test_failure_is_preserved_in_metrics_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "schema_version": 1,
                "checkpoint": "synthetic-checkpoint",
                "dataset": "synthetic",
                "experiment_id": "failed",
                "model": "synthetic-model",
                "task": "t2a",
                "paper_table": "test-only",
                "protocol_label": "synthetic",
                "protocol_source": "INFERRED",
                "query_embeddings": "missing.npy",
                "query_metadata": "missing.jsonl",
                "candidate_embeddings": "also-missing.npy",
                "candidate_metadata": "also-missing.jsonl",
                "query_selection": "all",
                "require_clean_git": False,
                "seed": 42,
            }
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            output = root / "failed"
            with (
                patch(
                    "scripts.evaluate_embedding_artifacts.git_output",
                    side_effect=["abc123", ""],
                ),
                self.assertRaises(FileNotFoundError),
            ):
                run_evaluation(config_path=root / "config.json", output_dir=output)

            report = json.loads(
                (output / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "failed")
            self.assertIn("FileNotFoundError", report["error"])

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
