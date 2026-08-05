from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.probe_fixed_index_negative_uiq import (
    DatasetBundle,
    apply_adapter,
    assign_component_folds,
    cluster_bootstrap_delta,
    train_low_rank_adapter,
)


def fixture_bundle() -> DatasetBundle:
    queries = np.eye(6, dtype=np.float32)
    return DatasetBundle(
        name="fixture",
        label="fixture",
        model="fixture",
        metrics_path=Path("metrics.json"),
        query_embeddings=queries,
        candidate_embeddings=queries,
        candidate_ids=tuple(f"a{index}" for index in range(6)),
        query_ids=tuple(f"q{index}" for index in range(6)),
        target_ids=tuple(f"a{index}" for index in range(6)),
        hard_negative_ids=tuple(f"b{index}" for index in range(6)),
        target_indices=np.arange(6),
        hard_negative_indices=np.arange(6),
        component_keys=tuple(f"fixture:c{index}" for index in range(6)),
        inputs={},
    )


class FixedIndexNegativeUIQProbeTest(unittest.TestCase):
    def test_script_runs_directly_without_pythonpath(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(repository_root / "scripts/probe_fixed_index_negative_uiq.py"),
                    "--help",
                ],
                cwd=temporary,
                env=environment,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_repository_imports_work_when_script_is_loaded_by_path(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        script = repository_root / "scripts/probe_fixed_index_negative_uiq.py"
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import runpy; runpy.run_path(r'"
                    + str(script)
                    + "'); import scripts.evaluate_negative_uiq_npz"
                ),
            ],
            cwd=Path(tempfile.gettempdir()),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_component_assignment_is_deterministic_and_balanced(self) -> None:
        bundle = fixture_bundle()
        first, audit = assign_component_folds(
            [bundle], folds=3, seed=7, maximum_imbalance=2.0
        )
        second, _ = assign_component_folds(
            [bundle], folds=3, seed=7, maximum_imbalance=2.0
        )
        self.assertEqual(first, second)
        self.assertEqual(audit["fixture"]["query_rows_by_fold"], [2, 2, 2])

    def test_connected_pairs_never_cross_folds(self) -> None:
        bundle = fixture_bundle()
        assignments, _ = assign_component_folds(
            [bundle], folds=3, seed=7, maximum_imbalance=2.0
        )
        for component in bundle.component_keys:
            self.assertIn(assignments[component], (0, 1, 2))

    def test_cluster_bootstrap_reports_positive_paired_gain(self) -> None:
        report = cluster_bootstrap_delta(
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            ["a", "a", "b", "b"],
            iterations=500,
            seed=11,
        )
        self.assertEqual(report["observed_mean_delta"], 1.0)
        self.assertEqual(report["confidence_interval_95"], [1.0, 1.0])

    def test_adapter_training_reduces_predeclared_objective(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("PyTorch is available in the remote oea-repro environment")
        generator = np.random.default_rng(3)
        queries = generator.normal(size=(32, 8)).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        pair_differences = np.roll(queries, 1, axis=1) - queries
        down, up, report = train_low_rank_adapter(
            queries,
            pair_differences,
            rank=4,
            epochs=30,
            learning_rate=1e-2,
            temperature=0.1,
            anchor_weight=0.01,
            weight_decay=0.0,
            seed=5,
            device="cpu",
        )
        adapted = apply_adapter(queries, down, up)
        self.assertEqual(adapted.shape, queries.shape)
        self.assertTrue(np.isfinite(adapted).all())
        self.assertLess(report["final_total_loss"], report["initial_total_loss"])


class FixedIndexNegativeUIQWrapperTest(unittest.TestCase):
    def test_wrapper_locks_frozen_index_protocol(self) -> None:
        source = Path("scripts/run_fixed_index_negative_uiq_probe.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('[[ -z "${TMUX:-}" ]]', source)
        self.assertIn("AUDIO_EMBEDDINGS=frozen and hash-verified", source)
        self.assertIn("AUDIO_INDEX_REBUILT=no", source)
        self.assertIn("CUBLAS_WORKSPACE_CONFIG=:4096:8", source)
        self.assertIn('--folds "${FIXED_INDEX_PROBE_FOLDS:-2}"', source)
        self.assertIn("PY\nfi", source)
        self.assertNotIn("nohup", source)


if __name__ == "__main__":
    unittest.main()
