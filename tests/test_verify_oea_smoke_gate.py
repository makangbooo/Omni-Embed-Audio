from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.build_official_oea_eval_config import (
    build_resolved_config,
    portable_file_identity,
    read_json_object,
    verify_official_model_lock_binding,
)
from scripts.verify_oea_smoke_gate import sha256_file, verify_smoke_gate


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_LOCK_PATH = REPOSITORY_ROOT / "results/model_locks/oea_qwen3b.json"
SMOKE_PROTOCOL_PATH = (
    REPOSITORY_ROOT / "configs/eval/qwen3b_clotho_lock_bound_smoke_embeddings.json"
)
FULL_PROTOCOL_PATH = REPOSITORY_ROOT / "configs/eval/qwen3b_clotho_embeddings.json"


def file_identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


class VerifyOEASmokeGateTest(unittest.TestCase):
    def create_fixture(self, root: Path) -> tuple[Path, dict[str, object]]:
        run_dir = root / "oea_qwen3b_ac_clotho_lock_bound_smoke_seed42_fixture"
        run_dir.mkdir()
        candidate_embeddings = np.zeros((5, 512), dtype=np.float32)
        query_embeddings = np.zeros((25, 512), dtype=np.float32)
        candidate_embeddings[:, 0] = 1.0
        query_embeddings[:, 0] = 1.0
        np.save(run_dir / "candidate_embeddings.npy", candidate_embeddings)
        np.save(run_dir / "query_embeddings.npy", query_embeddings)
        (run_dir / "candidate_metadata.jsonl").write_text(
            "".join(
                json.dumps({"candidate_id": f"candidate-{index}"}) + "\n"
                for index in range(5)
            ),
            encoding="utf-8",
        )
        (run_dir / "query_metadata.jsonl").write_text(
            "".join(
                json.dumps({"query_id": f"query-{index}"}) + "\n"
                for index in range(25)
            ),
            encoding="utf-8",
        )
        lock = read_json_object(MODEL_LOCK_PATH, "model lock")
        smoke_protocol = read_json_object(SMOKE_PROTOCOL_PATH, "smoke protocol")
        smoke_commit = "a" * 40
        smoke_config = build_resolved_config(
            smoke_protocol,
            lock,
            protocol_identity=portable_file_identity(
                SMOKE_PROTOCOL_PATH, repository_root=REPOSITORY_ROOT
            ),
            model_lock_identity=portable_file_identity(
                MODEL_LOCK_PATH, repository_root=REPOSITORY_ROOT
            ),
            git_commit=smoke_commit,
        )
        smoke_config["resolved_paths"] = {"output_dir": str(run_dir.resolve())}
        (run_dir / "config.yaml").write_text(
            json.dumps(smoke_config), encoding="utf-8"
        )
        metrics: dict[str, object] = {
            "schema_version": 1,
            "status": "complete",
            "error": None,
            "experiment_id": run_dir.name,
            "git_commit": smoke_commit,
            "model": "OEA-Qwen3B",
            "dataset": smoke_config["dataset"],
            "seed": smoke_config["seed"],
            "audio_prompt_protocol": smoke_config["audio_prompt_protocol"],
            "official_model_lock": verify_official_model_lock_binding(smoke_config),
            "strict_offline": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            },
            "candidate_count": 5,
            "query_count": 25,
            "completed_audio_chunks": 5,
            "completed_text_chunks": 25,
            "pending_audio_chunks": 0,
            "pending_text_chunks": 0,
            "candidate_embedding_shape": [5, 512],
            "query_embedding_shape": [25, 512],
            "model_load": {
                "lora_tensor_count": 544,
                "projection_dim": 512,
            },
            "artifacts": {
                name: file_identity(run_dir / filename)
                for name, filename in {
                    "candidate_embeddings": "candidate_embeddings.npy",
                    "query_embeddings": "query_embeddings.npy",
                    "candidate_metadata": "candidate_metadata.jsonl",
                    "query_metadata": "query_metadata.jsonl",
                }.items()
            },
        }
        metrics_path = run_dir / "generation_metrics.json"
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        return metrics_path, metrics

    def verify(self, metrics_path: Path, metrics: dict[str, object], **overrides):
        arguments = {
            "metrics_path": metrics_path,
            "model_lock_path": MODEL_LOCK_PATH,
            "full_protocol_path": FULL_PROTOCOL_PATH,
            "current_git_commit": "b" * 40,
            "smoke_commit_is_ancestor": True,
            "changed_critical_files": [],
        }
        arguments.update(overrides)
        return verify_smoke_gate(metrics, **arguments)

    def test_gate_accepts_complete_lock_bound_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics_path, metrics = self.create_fixture(Path(directory))
            result = self.verify(metrics_path, metrics)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["candidate_count"], 5)
        self.assertEqual(result["query_count"], 25)
        self.assertEqual(result["embedding_dimension"], 512)
        self.assertEqual(set(result["verified_artifacts"]), set(metrics["artifacts"]))

    def test_gate_rejects_lock_count_ancestry_and_critical_drift(self) -> None:
        mutations = (
            (
                "lock",
                lambda value: value["official_model_lock"]["model_lock"].update(
                    sha256="0" * 64
                ),
                {},
                "model lock SHA256",
            ),
            (
                "count",
                lambda value: value.update(query_count=24),
                {},
                "query_count",
            ),
            (
                "ancestry",
                lambda value: None,
                {"smoke_commit_is_ancestor": False},
                "not an ancestor",
            ),
            (
                "critical",
                lambda value: None,
                {"changed_critical_files": ["scripts/generate_oea_embeddings.py"]},
                "critical inference files changed",
            ),
        )
        for label, mutate, overrides, message in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                metrics_path, original = self.create_fixture(Path(directory))
                metrics = copy.deepcopy(original)
                mutate(metrics)
                with self.assertRaisesRegex(ValueError, message):
                    self.verify(metrics_path, metrics, **overrides)

    def test_gate_rejects_artifact_and_protocol_core_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics_path, metrics = self.create_fixture(Path(directory))
            with (metrics_path.parent / "query_embeddings.npy").open("ab") as handle:
                handle.write(b"drift")
            with self.assertRaisesRegex(ValueError, "size differs"):
                self.verify(metrics_path, metrics)

        with tempfile.TemporaryDirectory() as directory:
            metrics_path, metrics = self.create_fixture(Path(directory))
            smoke_config_path = metrics_path.parent / "config.yaml"
            smoke_config = json.loads(smoke_config_path.read_text(encoding="utf-8"))
            smoke_config["seed"] = 99
            smoke_config_path.write_text(json.dumps(smoke_config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "core differs at seed"):
                self.verify(metrics_path, metrics)


if __name__ == "__main__":
    unittest.main()
