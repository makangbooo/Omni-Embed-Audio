from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np

from scripts.build_official_oea_eval_config import portable_file_identity
from scripts.build_vanilla_backbone_eval_config import (
    build_resolved_config,
    verify_vanilla_model_lock_binding,
)
from scripts.generate_vanilla_backbone_embeddings import (
    encode_base_batch,
    hidden_size_from_locked_config,
    hidden_size_from_runtime_config,
    load_config,
    verify_locked_base_files,
)
from scripts.generate_oea_embeddings import file_identity, load_manifest
from scripts.vanilla_backbone_registry import (
    EXPECTED_BACKBONE_IDS,
    load_vanilla_backbone_registry,
)
from scripts.verify_vanilla_smoke_gate import verify_smoke_gate


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class VanillaBackboneEmbeddingsTest(unittest.TestCase):
    def protocol_path(
        self, backbone_id: str = "vanilla_qwen2_5_omni_3b"
    ) -> Path:
        return REPOSITORY_ROOT / "configs/eval" / f"{backbone_id}_clotho_embeddings.json"

    def smoke_protocol_path(self, backbone_id: str) -> Path:
        return (
            REPOSITORY_ROOT
            / "configs/eval"
            / f"{backbone_id}_clotho_smoke_embeddings.json"
        )

    def fixture_values(
        self, backbone_id: str = "vanilla_qwen2_5_omni_3b"
    ) -> tuple[dict, dict]:
        protocol = json.loads(self.protocol_path(backbone_id).read_text(encoding="utf-8"))
        registry = load_vanilla_backbone_registry()[backbone_id]
        lock = {
            "schema_version": 1,
            "status": "locked",
            "lock_type": "vanilla_backbone_base_only",
            "backbone_id": backbone_id,
            "model": registry["paper_model"],
            "family": registry["family"],
            "protocol": copy.deepcopy(protocol["protocol"]),
            "base_model": {
                **registry["base_asset"],
                "source": "CODE+AUDIT",
                "files": {
                    "config.json": {
                        "size_bytes": 17,
                        "sha256": "a" * 64,
                    }
                },
            },
            "embedding_output": {
                "projection_head": "none",
                "dimension": "backbone text hidden size resolved at runtime",
                "source": "[CODE] public base adapter returns normalized pooled hidden states",
            },
        }
        lock["base_model"].pop("name")
        return protocol, lock

    def test_three_protocols_match_the_fixed_registry_and_preserve_conflict(self) -> None:
        registry = load_vanilla_backbone_registry()
        observed = []
        for backbone_id in EXPECTED_BACKBONE_IDS:
            protocol = json.loads(
                self.protocol_path(backbone_id).read_text(encoding="utf-8")
            )
            observed.append(protocol["backbone_id"])
            self.assertEqual(protocol["model"], registry[backbone_id]["paper_model"])
            for field in ("repo_id", "revision", "local_subdir"):
                self.assertEqual(
                    protocol["base_model"][field],
                    registry[backbone_id]["base_asset"][field],
                )
            self.assertEqual(protocol["protocol"], registry[backbone_id]["protocol"])
            self.assertEqual(protocol["protocol"]["audio_prompt"]["status"], "CONFLICT")
            self.assertEqual(
                protocol["protocol"]["text_prompt"]["runtime_form"],
                "query:<caption>",
            )
            self.assertNotIn("checkpoint", protocol)
            self.assertNotIn("projection_dim", protocol["model_config"])
        self.assertEqual(tuple(observed), EXPECTED_BACKBONE_IDS)

    def test_smoke_protocols_reuse_exact_full_contract_and_bundled_fixture(self) -> None:
        manifest_path = (
            REPOSITORY_ROOT
            / "configs/eval/fixtures/vanilla_clotho_5_manifest.jsonl"
        )
        rows = load_manifest(manifest_path, 5, 5)
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(row["audio_path"].is_file() for row in rows))
        for backbone_id in EXPECTED_BACKBONE_IDS:
            full = json.loads(
                self.protocol_path(backbone_id).read_text(encoding="utf-8")
            )
            smoke = json.loads(
                self.smoke_protocol_path(backbone_id).read_text(encoding="utf-8")
            )
            for field in ("backbone_id", "model", "base_model", "protocol", "model_config"):
                self.assertEqual(smoke[field], full[field])
            self.assertEqual(smoke["expected_examples"], 5)
            self.assertEqual(smoke["caption_count_per_audio"], 5)
            self.assertIn("_clotho_smoke_seed42", smoke["experiment_prefix"])

    def test_resolved_config_expands_full_lock_without_oea_weights(self) -> None:
        protocol, lock = self.fixture_values()
        resolved = build_resolved_config(
            protocol,
            lock,
            protocol_identity={
                "repository_path": "configs/protocol.json",
                "size_bytes": 1,
                "sha256": "b" * 64,
            },
            model_lock_identity={
                "repository_path": "results/model_locks/vanilla.json",
                "size_bytes": 1,
                "sha256": "c" * 64,
            },
            git_commit="d" * 40,
        )
        self.assertEqual(resolved["base_model"], lock["base_model"])
        self.assertEqual(resolved["embedding_output"], lock["embedding_output"])
        self.assertNotIn("checkpoint", resolved)
        self.assertNotIn("official_model_lock", resolved)
        self.assertEqual(resolved["resolution_git_commit"], "d" * 40)

    def test_binding_rejects_lock_content_drift(self) -> None:
        protocol, lock = self.fixture_values()
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            protocol_path = root / "configs/protocol.json"
            lock_path = root / "results/model_locks/vanilla.json"
            protocol_path.parent.mkdir(parents=True)
            lock_path.parent.mkdir(parents=True)
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            resolved = build_resolved_config(
                protocol,
                lock,
                protocol_identity=portable_file_identity(
                    protocol_path, repository_root=root
                ),
                model_lock_identity=portable_file_identity(
                    lock_path, repository_root=root
                ),
                git_commit="d" * 40,
            )
            binding = verify_vanilla_model_lock_binding(
                resolved, repository_root=root
            )
            self.assertEqual(binding["backbone_id"], protocol["backbone_id"])
            lock["model"] = "drifted"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "size|SHA256"):
                verify_vanilla_model_lock_binding(resolved, repository_root=root)

    def test_locked_base_inventory_and_hidden_dimension_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            base = root / "Qwen2.5-Omni-3B"
            base.mkdir()
            content = json.dumps({"text_config": {"hidden_size": 2048}}).encode()
            config_file = base / "config.json"
            config_file.write_bytes(content)
            config = {
                "base_model": {
                    "local_subdir": base.name,
                    "files": {
                        "config.json": {
                            "size_bytes": len(content),
                            "sha256": hashlib.sha256(content).hexdigest(),
                        }
                    },
                }
            }
            resolved, verified, dimension, source = verify_locked_base_files(config, root)
            self.assertEqual(resolved, base.resolve())
            self.assertEqual(len(verified), 1)
            self.assertEqual(dimension, 2048)
            self.assertEqual(source, "text_config.hidden_size")

            (base / "unexpected.txt").write_text("extra", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "inventory drift"):
                verify_locked_base_files(config, root)

    def test_qwen_and_nemotron_hidden_size_paths_are_explicit(self) -> None:
        self.assertEqual(
            hidden_size_from_locked_config({"text_config": {"hidden_size": 2048}}),
            (2048, "text_config.hidden_size"),
        )
        self.assertEqual(
            hidden_size_from_locked_config(
                {"thinker_config": {"text_config": {"hidden_size": 3584}}}
            ),
            (3584, "thinker_config.text_config.hidden_size"),
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            hidden_size_from_locked_config(
                {
                    "text_config": {"hidden_size": 2048},
                    "thinker_config": {"text_config": {"hidden_size": 2048}},
                }
            )

        class Node:
            def __init__(self, **values):
                self.__dict__.update(values)

        self.assertEqual(
            hidden_size_from_runtime_config(
                Node(thinker_config=Node(text_config=Node(hidden_size=3584)))
            ),
            (3584, "model.config.thinker_config.text_config.hidden_size"),
        )
        self.assertEqual(
            hidden_size_from_runtime_config(Node(text_config=Node(hidden_size=2048))),
            (2048, "model.config.text_config.hidden_size"),
        )

    def test_load_config_rejects_oea_only_fields(self) -> None:
        protocol, lock = self.fixture_values()
        resolved = build_resolved_config(
            protocol,
            lock,
            protocol_identity={
                "repository_path": "configs/protocol.json",
                "size_bytes": 1,
                "sha256": "b" * 64,
            },
            model_lock_identity={
                "repository_path": "results/model_locks/vanilla.json",
                "size_bytes": 1,
                "sha256": "c" * 64,
            },
            git_commit="d" * 40,
        )
        with tempfile.TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "resolved.json"
            path.write_text(json.dumps(resolved), encoding="utf-8")
            loaded = load_config(path)
            self.assertEqual(loaded["embedding_output"]["projection_head"], "none")
            resolved["checkpoint"] = {"forbidden": True}
            path.write_text(json.dumps(resolved), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "OEA-only"):
                load_config(path)

    def test_base_batch_checks_shape_norm_and_audio_fallback(self) -> None:
        class Adapter:
            def encode_text(self, texts, batch_size):
                return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))

            def encode_audio(self, paths, batch_size):
                warnings.warn("Failed to load audio; using silence", RuntimeWarning)
                return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(paths), 1))

        embeddings = encode_base_batch(Adapter(), 2, texts=["one", "two"])
        self.assertEqual(embeddings.shape, (2, 2))
        with self.assertRaisesRegex(RuntimeError, "fallback warning"):
            encode_base_batch(Adapter(), 2, audio_paths=[Path("missing.wav")])

    def create_smoke_gate_fixture(
        self, root: Path, backbone_id: str = "vanilla_qwen2_5_omni_3b"
    ) -> tuple[Path, Path, dict]:
        run_dir = root / "smoke"
        run_dir.mkdir()
        candidate_embeddings = np.tile(
            np.array([[1.0, 0.0]], dtype=np.float32), (5, 1)
        )
        query_embeddings = np.tile(
            np.array([[1.0, 0.0]], dtype=np.float32), (25, 1)
        )
        np.save(run_dir / "candidate_embeddings.npy", candidate_embeddings)
        np.save(run_dir / "query_embeddings.npy", query_embeddings)
        (run_dir / "candidate_metadata.jsonl").write_text(
            "".join(json.dumps({"candidate_id": str(index)}) + "\n" for index in range(5)),
            encoding="utf-8",
        )
        (run_dir / "query_metadata.jsonl").write_text(
            "".join(json.dumps({"query_id": str(index)}) + "\n" for index in range(25)),
            encoding="utf-8",
        )
        lock_path = root / "model_lock.json"
        lock_path.write_text('{"status":"locked"}\n', encoding="utf-8")
        metrics = {
            "schema_version": 1,
            "status": "complete",
            "git_commit": "a" * 40,
            "backbone_id": backbone_id,
            "candidate_count": 5,
            "query_count": 25,
            "embedding_dimension": 2,
            "candidate_embedding_shape": [5, 2],
            "query_embedding_shape": [25, 2],
            "projection_head_loaded": False,
            "lora_loaded": False,
            "oea_checkpoint_loaded": False,
            "model_load": {
                "projection_head_loaded": False,
                "lora_loaded": False,
                "oea_checkpoint_loaded": False,
            },
            "vanilla_model_lock": {
                "model_lock": {
                    "repository_path": "results/model_locks/fixture.json",
                    "size_bytes": lock_path.stat().st_size,
                    "sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
                }
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
        return metrics_path, lock_path, metrics

    def test_smoke_gate_verifies_model_lock_and_all_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            metrics_path, lock_path, metrics = self.create_smoke_gate_fixture(root)
            result = verify_smoke_gate(
                metrics,
                metrics_path=metrics_path,
                backbone=metrics["backbone_id"],
                model_lock=lock_path,
                current_git_commit="a" * 40,
            )
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["candidate_count"], 5)
            self.assertEqual(set(result["verified_artifacts"]), set(metrics["artifacts"]))

    def test_smoke_gate_rejects_commit_flags_counts_and_artifact_drift(self) -> None:
        mutations = (
            ("commit", lambda value: value.update(git_commit="b" * 40), "commit"),
            (
                "flag",
                lambda value: value.update(projection_head_loaded=True),
                "projection_head_loaded",
            ),
            ("count", lambda value: value.update(query_count=24), "25 queries"),
        )
        for label, mutate, message in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as raw_directory:
                root = Path(raw_directory)
                metrics_path, lock_path, metrics = self.create_smoke_gate_fixture(root)
                mutate(metrics)
                with self.assertRaisesRegex(ValueError, message):
                    verify_smoke_gate(
                        metrics,
                        metrics_path=metrics_path,
                        backbone="vanilla_qwen2_5_omni_3b",
                        model_lock=lock_path,
                        current_git_commit="a" * 40,
                    )

        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            metrics_path, lock_path, metrics = self.create_smoke_gate_fixture(root)
            with (metrics_path.parent / "query_embeddings.npy").open("ab") as handle:
                handle.write(b"drift")
            with self.assertRaisesRegex(ValueError, "size differs"):
                verify_smoke_gate(
                    metrics,
                    metrics_path=metrics_path,
                    backbone="vanilla_qwen2_5_omni_3b",
                    model_lock=lock_path,
                    current_git_commit="a" * 40,
                )

    def test_wrapper_is_offline_locked_and_generator_is_base_only(self) -> None:
        wrapper = (REPOSITORY_ROOT / "scripts/run_vanilla_backbone_embeddings.sh").read_text(encoding="utf-8")
        generator = (REPOSITORY_ROOT / "scripts/generate_vanilla_backbone_embeddings.py").read_text(encoding="utf-8")
        self.assertIn("build_vanilla_backbone_eval_config.py", wrapper)
        self.assertIn("generate_vanilla_backbone_embeddings.py", wrapper)
        self.assertIn("MODEL_LOCK", wrapper)
        self.assertIn("HF_HUB_OFFLINE=1", wrapper)
        self.assertIn("requires a clean Git worktree", wrapper)
        self.assertIn("--smoke", wrapper)
        self.assertIn("--full <smoke_metrics.json>", wrapper)
        self.assertIn("SMOKE_METRICS", wrapper)
        self.assertIn("verify_vanilla_smoke_gate.py", wrapper)
        for forbidden in (
            "ProjectionHead",
            "attach_lora",
            "train_omniembed_lora",
            "torch.load(",
        ):
            self.assertNotIn(forbidden, generator)


if __name__ == "__main__":
    unittest.main()
