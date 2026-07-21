from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.evaluate_embedding_artifacts import file_identity, run_evaluation
from scripts.prepare_embedding_evaluation_suite import (
    finalize_suite,
    load_suite_config,
    prepare_suite,
    validate_metadata_and_select_indices,
)


AUDIO_PROMPT_PROTOCOL = {
    "runtime": {
        "value": "audio-only chat message; passage_prefix parameter is not inserted",
        "source": "CODE",
        "note": (
            "The paper states passage:, but public _build_audio_messages() "
            "explicitly ignores it. This run is a public-code protocol, not "
            "a strict paper-protocol claim."
        ),
    }
}
MODEL_LOCK_BINDING = {
    "variant_id": "synthetic-variant",
    "model": "synthetic-model",
    "model_lock": {
        "repository_path": "results/model_locks/synthetic.json",
        "size_bytes": 10,
        "sha256": "c" * 64,
    },
    "protocol_config": {
        "repository_path": "configs/eval/synthetic.json",
        "size_bytes": 10,
        "sha256": "b" * 64,
    },
}
VANILLA_MODEL_LOCK_BINDING = {
    "backbone_id": "synthetic-vanilla",
    "model": "synthetic-model",
    "model_lock": {
        "repository_path": "results/model_locks/synthetic-vanilla.json",
        "size_bytes": 10,
        "sha256": "d" * 64,
    },
    "protocol_config": {
        "repository_path": "configs/eval/synthetic-vanilla.json",
        "size_bytes": 10,
        "sha256": "b" * 64,
    },
}
VANILLA_PROTOCOL = {
    "text_prompt": {
        "value": "query:",
        "source": "[PAPER][CODE]",
        "runtime_join": "direct concatenation without an inserted separator",
        "runtime_form": "query:<caption>",
        "runtime_source": "[CODE] synthetic",
    },
    "audio_prompt": {
        "runtime_value": (
            "audio-only chat message; passage_prefix parameter is not inserted"
        ),
        "runtime_source": "[CODE]",
        "paper_value": "passage:",
        "paper_source": "[PAPER]",
        "status": "CONFLICT",
        "claim_boundary": "synthetic public-code protocol",
    },
    "pooling": {
        "value": "attention-mask-aware mean of last hidden states",
        "source": "[CODE]",
    },
    "normalization": {"value": "L2", "source": "[CODE]"},
    "projection": {
        "value": "none; retain the backbone hidden dimension",
        "source": "[CODE]",
    },
}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def synthetic_suite_config(root: Path) -> Path:
    checkpoint = {
        "repo_id": "synthetic/checkpoint",
        "revision": "checkpoint-revision",
        "local_subpath": "checkpoint.pt",
        "size_bytes": 20,
        "sha256": "a" * 64,
        "source": "DERIVED_FROM_CODE_CHECKPOINT",
    }
    protocols = [
        {
            "protocol_id": "a2t_all",
            "task": "a2t",
            "paper_table": "not-paper-reported",
            "query_selection": "all",
            "protocol_label": "code-a2t-all",
            "protocol_source": "CODE",
            "expected_evaluated_queries": 2,
            "note": "synthetic",
        },
        {
            "protocol_id": "t2a_all",
            "task": "t2a",
            "paper_table": "Table 2",
            "query_selection": "all",
            "protocol_label": "code-all",
            "protocol_source": "CODE",
            "expected_evaluated_queries": 4,
            "note": "synthetic",
        },
        {
            "protocol_id": "t2a_one",
            "task": "t2a",
            "paper_table": "Table 2",
            "query_selection": "public_code_seed0_one_per_clip",
            "query_selection_seed": 0,
            "protocol_label": "code-one",
            "protocol_source": "CODE",
            "expected_evaluated_queries": 2,
            "note": "synthetic",
        },
        {
            "protocol_id": "t2t_one",
            "task": "t2t",
            "paper_table": "Table 3",
            "query_selection": "public_code_seed0_one_per_clip",
            "query_selection_seed": 0,
            "protocol_label": "code-t2t-one",
            "protocol_source": "CODE",
            "expected_evaluated_queries": 2,
            "note": "synthetic",
        },
        {
            "protocol_id": "t2t_all",
            "task": "t2t",
            "paper_table": "Table 3 sensitivity",
            "query_selection": "all",
            "protocol_label": "inferred-all",
            "protocol_source": "INFERRED",
            "expected_evaluated_queries": 4,
            "note": "synthetic",
        },
    ]
    config = {
        "schema_version": 1,
        "suite_prefix": "synthetic_suite",
        "model": "synthetic-model",
        "dataset": "synthetic-dataset",
        "embedding_seed": 42,
        "expected_candidate_count": 2,
        "expected_query_count": 4,
        "expected_embedding_dim": 2,
        "expected_captions_per_clip": 2,
        "expected_generation_protocol_sha256": "b" * 64,
        "official_variant_id": "synthetic-variant",
        "official_model_lock_path": "results/model_locks/synthetic.json",
        "minimum_generator_commit": "minimum-commit",
        "checkpoint": checkpoint,
        "embedding_protocol": {"source": "CODE"},
        "protocols": protocols,
    }
    path = root / "suite_config.json"
    write_json(path, config)
    return path


def synthetic_embedding_directory(root: Path) -> Path:
    embedding_dir = root / "embedding_run"
    embedding_dir.mkdir()
    candidates = np.eye(2, dtype=np.float32)
    queries = np.asarray(
        [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [0.6, 0.8]],
        dtype=np.float32,
    )
    np.save(embedding_dir / "candidate_embeddings.npy", candidates)
    np.save(embedding_dir / "query_embeddings.npy", queries)
    write_jsonl(
        embedding_dir / "candidate_metadata.jsonl",
        [
            {"candidate_index": 0, "candidate_id": "a"},
            {"candidate_index": 1, "candidate_id": "b"},
        ],
    )
    write_jsonl(
        embedding_dir / "query_metadata.jsonl",
        [
            {
                "query_index": 0,
                "query_id": "a#caption_1",
                "target_id": "a",
                "clip_id": "a",
                "caption_index": 1,
                "text": "a one",
            },
            {
                "query_index": 1,
                "query_id": "a#caption_2",
                "target_id": "a",
                "clip_id": "a",
                "caption_index": 2,
                "text": "a two",
            },
            {
                "query_index": 2,
                "query_id": "b#caption_1",
                "target_id": "b",
                "clip_id": "b",
                "caption_index": 1,
                "text": "b one",
            },
            {
                "query_index": 3,
                "query_id": "b#caption_2",
                "target_id": "b",
                "clip_id": "b",
                "caption_index": 2,
                "text": "b two",
            },
        ],
    )
    checkpoint = {
        "repo_id": "synthetic/checkpoint",
        "revision": "checkpoint-revision",
        "local_subpath": "checkpoint.pt",
        "size_bytes": 20,
        "sha256": "a" * 64,
        "source": "DERIVED_FROM_CODE_CHECKPOINT",
    }
    resolved = {
        "checkpoint": checkpoint,
        "audio_prompt_protocol": AUDIO_PROMPT_PROTOCOL,
        "resolution_git_commit": "generation-commit",
    }
    write_json(embedding_dir / "resolved_embedding_config.json", resolved)
    write_json(
        embedding_dir / "config.yaml",
        {
            **resolved,
            "resolved_paths": {"output_dir": str(embedding_dir)},
        },
    )
    write_json(
        embedding_dir / "run_identity.json",
        {
            "git_commit": "generation-commit",
            "config": file_identity(embedding_dir / "resolved_embedding_config.json"),
            "official_model_lock": MODEL_LOCK_BINDING,
            "candidate_count": 2,
            "query_count": 4,
            "checkpoint_revision": "checkpoint-revision",
        },
    )
    artifact_names = (
        "candidate_embeddings",
        "query_embeddings",
        "candidate_metadata",
        "query_metadata",
    )
    suffixes = {
        "candidate_embeddings": ".npy",
        "query_embeddings": ".npy",
        "candidate_metadata": ".jsonl",
        "query_metadata": ".jsonl",
    }
    artifacts = {
        name: file_identity(embedding_dir / f"{name}{suffixes[name]}")
        for name in artifact_names
    }
    write_json(
        embedding_dir / "generation_metrics.json",
        {
            "schema_version": 1,
            "status": "complete",
            "model": "synthetic-model",
            "dataset": "synthetic-dataset",
            "seed": 42,
            "official_model_lock": MODEL_LOCK_BINDING,
            "artifacts": artifacts,
        },
    )
    return embedding_dir


def synthetic_vanilla_suite_config(root: Path) -> Path:
    path = synthetic_suite_config(root)
    config = json.loads(path.read_text(encoding="utf-8"))
    for field in ("checkpoint", "official_model_lock_path", "official_variant_id"):
        config.pop(field)
    config.update(
        {
            "binding_type": "vanilla_backbone",
            "backbone_id": "synthetic-vanilla",
            "vanilla_model_lock_path": (
                "results/model_locks/synthetic-vanilla.json"
            ),
            "base_model": {
                "repo_id": "synthetic/base",
                "revision": "base-revision",
                "local_subdir": "synthetic-base",
            },
            "expected_protocol": VANILLA_PROTOCOL,
        }
    )
    write_json(path, config)
    return path


def synthetic_vanilla_embedding_directory(root: Path) -> Path:
    embedding_dir = synthetic_embedding_directory(root)
    resolved = {
        "backbone_id": "synthetic-vanilla",
        "model": "synthetic-model",
        "base_model": {
            "repo_id": "synthetic/base",
            "revision": "base-revision",
            "local_subdir": "synthetic-base",
        },
        "protocol": VANILLA_PROTOCOL,
        "resolution_git_commit": "generation-commit",
    }
    write_json(embedding_dir / "resolved_embedding_config.json", resolved)
    write_json(
        embedding_dir / "config.yaml",
        {**resolved, "resolved_paths": {"output_dir": str(embedding_dir)}},
    )
    write_json(
        embedding_dir / "run_identity.json",
        {
            "git_commit": "generation-commit",
            "config": file_identity(
                embedding_dir / "resolved_embedding_config.json"
            ),
            "vanilla_model_lock": VANILLA_MODEL_LOCK_BINDING,
            "backbone_id": "synthetic-vanilla",
            "base_revision": "base-revision",
            "embedding_dimension": 2,
            "candidate_count": 2,
            "query_count": 4,
        },
    )
    metrics_path = embedding_dir / "generation_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update(
        {
            "backbone_id": "synthetic-vanilla",
            "vanilla_model_lock": VANILLA_MODEL_LOCK_BINDING,
            "embedding_dimension": 2,
            "projection_head_loaded": False,
            "lora_loaded": False,
            "oea_checkpoint_loaded": False,
        }
    )
    metrics.pop("official_model_lock", None)
    write_json(metrics_path, metrics)
    return embedding_dir


class PrepareEmbeddingEvaluationSuiteTest(unittest.TestCase):
    def test_fixed_a2t_config_is_public_code_extension(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        config = load_suite_config(
            repository_root / "configs/eval/qwen3b_cl_clotho_a2t_suite.json"
        )
        self.assertEqual(config["model"], "OEA-Qwen3B (+Cl)")
        self.assertEqual(len(config["protocols"]), 1)
        protocol = config["protocols"][0]
        self.assertEqual(protocol["task"], "a2t")
        self.assertEqual(protocol["query_selection"], "all")
        self.assertEqual(protocol["protocol_source"], "CODE")
        self.assertIn("not paper-reported", protocol["paper_table"])
        self.assertEqual(protocol["expected_evaluated_queries"], 1045)
        self.assertEqual(
            file_identity(
                repository_root
                / "configs/eval/qwen3b_cl_clotho_embeddings.json"
            )["sha256"],
            config["expected_generation_protocol_sha256"],
        )

    def test_fixed_config_separates_all_public_code_protocols(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        config = load_suite_config(
            repository_root
            / "configs/eval/qwen3b_cl_clotho_retrieval_suite.json"
        )
        protocols = {item["protocol_id"]: item for item in config["protocols"]}
        self.assertEqual(len(protocols), 4)
        self.assertEqual(
            protocols["t2a_public_code_default_joint_all_captions"][
                "expected_evaluated_queries"
            ],
            5225,
        )
        self.assertEqual(
            protocols["t2a_public_code_t2a_only_seed0"]["protocol_source"],
            "CODE",
        )
        self.assertEqual(
            protocols["t2t_public_code_default_seed0"]["query_selection_seed"],
            0,
        )
        self.assertEqual(
            protocols["t2t_all_captions_sensitivity"]["protocol_source"],
            "INFERRED",
        )
        self.assertEqual(
            file_identity(
                repository_root
                / "configs/eval/qwen3b_cl_clotho_embeddings.json"
            )["sha256"],
            config["expected_generation_protocol_sha256"],
        )

    def test_qwen3b_ac_suite_preserves_protocols_and_binds_its_own_lock(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        clotho_tuned = load_suite_config(
            repository_root
            / "configs/eval/qwen3b_cl_clotho_retrieval_suite.json"
        )
        audiocaps_tuned = load_suite_config(
            repository_root / "configs/eval/qwen3b_clotho_retrieval_suite.json"
        )

        self.assertEqual(audiocaps_tuned["model"], "OEA-Qwen3B")
        self.assertEqual(audiocaps_tuned["official_variant_id"], "oea_qwen3b")
        self.assertEqual(
            audiocaps_tuned["official_model_lock_path"],
            "results/model_locks/oea_qwen3b.json",
        )
        self.assertEqual(
            file_identity(
                repository_root / "configs/eval/qwen3b_clotho_embeddings.json"
            )["sha256"],
            audiocaps_tuned["expected_generation_protocol_sha256"],
        )
        self.assertEqual(
            audiocaps_tuned["protocols"], clotho_tuned["protocols"]
        )
        self.assertNotEqual(
            audiocaps_tuned["checkpoint"], clotho_tuned["checkpoint"]
        )

    def test_vanilla_fixed_configs_bind_exact_base_only_protocols(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        expected = {
            "vanilla_nemotron_3b": (
                "vanilla_nemotron_3b_clotho_embeddings.json",
                2048,
            ),
            "vanilla_qwen2_5_omni_3b": (
                "vanilla_qwen2_5_omni_3b_clotho_embeddings.json",
                2048,
            ),
            "vanilla_qwen2_5_omni_7b": (
                "vanilla_qwen2_5_omni_7b_clotho_embeddings.json",
                3584,
            ),
        }
        for backbone_id, (embedding_name, dimension) in expected.items():
            with self.subTest(backbone=backbone_id):
                suite_name = embedding_name.replace(
                    "_clotho_embeddings.json", "_clotho_retrieval_suite.json"
                )
                config = load_suite_config(
                    repository_root / "configs/eval" / suite_name
                )
                self.assertEqual(config["binding_type"], "vanilla_backbone")
                self.assertEqual(config["backbone_id"], backbone_id)
                self.assertEqual(config["expected_embedding_dim"], dimension)
                self.assertNotIn("checkpoint", config)
                generation_config = json.loads(
                    (repository_root / "configs/eval" / embedding_name).read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(config["base_model"], generation_config["base_model"])
                self.assertEqual(
                    config["expected_protocol"], generation_config["protocol"]
                )
                self.assertEqual(
                    file_identity(
                        repository_root / "configs/eval" / embedding_name
                    )["sha256"],
                    config["expected_generation_protocol_sha256"],
                )
                self.assertEqual(len(config["protocols"]), 4)

    def test_seed0_selection_reproduces_one_choice_per_clip(self) -> None:
        candidates = [
            {"candidate_index": 0, "candidate_id": "a"},
            {"candidate_index": 1, "candidate_id": "b"},
        ]
        queries = []
        for clip_id in ("a", "b"):
            for caption_index in (1, 2):
                queries.append(
                    {
                        "query_index": len(queries),
                        "query_id": f"{clip_id}#caption_{caption_index}",
                        "target_id": clip_id,
                        "clip_id": clip_id,
                        "caption_index": caption_index,
                    }
                )
        selected, rows = validate_metadata_and_select_indices(
            candidates,
            queries,
            expected_candidates=2,
            expected_queries=4,
            captions_per_clip=2,
            seed=0,
        )
        self.assertEqual(selected, [1, 3])
        self.assertEqual([row["caption_index"] for row in rows], [2, 2])

    def test_prepare_and_finalize_complete_synthetic_suite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = synthetic_suite_config(root)
            embedding_dir = synthetic_embedding_directory(root)
            suite_dir = root / "synthetic_suite_run"
            with (
                patch(
                    "scripts.prepare_embedding_evaluation_suite.git_output",
                    side_effect=["suite-commit", ""],
                ),
                patch(
                    "scripts.prepare_embedding_evaluation_suite.verify_official_model_lock_binding",
                    return_value=MODEL_LOCK_BINDING,
                ),
                patch("scripts.prepare_embedding_evaluation_suite.subprocess.run"),
                patch("builtins.print"),
            ):
                self.assertEqual(
                    prepare_suite(config_path, embedding_dir, suite_dir), 0
                )

            plan = json.loads(
                (suite_dir / "suite_plan.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(plan["protocols"]), 5)
            self.assertEqual(
                json.loads(
                    (suite_dir / "selections/seed0_indices.json").read_text(
                        encoding="utf-8"
                    )
                ),
                [1, 3],
            )
            for protocol in plan["protocols"]:
                output_dir = Path(protocol["output_dir"])
                output_dir.mkdir(parents=True)
                (output_dir / "command.sh").write_text(
                    "synthetic wrapper\n", encoding="utf-8"
                )
                (output_dir / "environment.txt").write_text(
                    "synthetic environment\n", encoding="utf-8"
                )
                (output_dir / "git_commit.txt").write_text(
                    "suite-commit\n", encoding="utf-8"
                )
                (output_dir / "git_status.txt").write_text("", encoding="utf-8")
                (output_dir / "gpu_info.txt").write_text(
                    "GPU disabled\n", encoding="utf-8"
                )
                (output_dir / "stdout.log").write_text("", encoding="utf-8")
                (output_dir / "stderr.log").write_text("", encoding="utf-8")
                with patch(
                    "scripts.evaluate_embedding_artifacts.git_output",
                    side_effect=["suite-commit", ""],
                ):
                    run_evaluation(
                        config_path=Path(protocol["config"]),
                        output_dir=output_dir,
                        argv=["python", "synthetic-evaluation"],
                    )
                (output_dir / "exit_code.txt").write_text("0\n", encoding="utf-8")
            with patch(
                "scripts.prepare_embedding_evaluation_suite.git_output",
                side_effect=["suite-commit", ""],
            ), patch(
                "scripts.prepare_embedding_evaluation_suite.verify_official_model_lock_binding",
                return_value=MODEL_LOCK_BINDING,
            ), patch("builtins.print"):
                self.assertEqual(
                    finalize_suite(config_path, embedding_dir, suite_dir), 0
                )

            report_path = suite_dir / "suite_metrics.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["protocol_count"], 5)
            summary = (suite_dir / "retrieval_summary.csv").read_text(
                encoding="utf-8"
            )
            self.assertEqual(len(summary.splitlines()), 26)
            original = report_path.read_bytes()
            with patch(
                "scripts.prepare_embedding_evaluation_suite.git_output",
                side_effect=["suite-commit", ""],
            ), patch(
                "scripts.prepare_embedding_evaluation_suite.verify_official_model_lock_binding",
                return_value=MODEL_LOCK_BINDING,
            ), patch("builtins.print"):
                self.assertEqual(
                    finalize_suite(config_path, embedding_dir, suite_dir), 0
                )
            self.assertEqual(report_path.read_bytes(), original)

            ranks_path = Path(plan["protocols"][0]["output_dir"]) / "ranks.npy"
            corrupted_ranks = bytearray(ranks_path.read_bytes())
            corrupted_ranks[-1] ^= 1
            ranks_path.write_bytes(corrupted_ranks)
            with (
                patch(
                    "scripts.prepare_embedding_evaluation_suite.git_output",
                    side_effect=["suite-commit", ""],
                ),
                patch(
                    "scripts.prepare_embedding_evaluation_suite.verify_official_model_lock_binding",
                    return_value=MODEL_LOCK_BINDING,
                ),
                patch("builtins.print"),
                self.assertRaisesRegex(RuntimeError, "recorded SHA256 mismatch"),
            ):
                finalize_suite(config_path, embedding_dir, suite_dir)
            self.assertEqual(report_path.read_bytes(), original)
            self.assertEqual(
                len(list((suite_dir / "failures").glob("failure_*.json"))), 1
            )

    def test_prepare_and_finalize_complete_synthetic_vanilla_suite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = synthetic_vanilla_suite_config(root)
            embedding_dir = synthetic_vanilla_embedding_directory(root)
            suite_dir = root / "synthetic_suite_run"
            with (
                patch(
                    "scripts.prepare_embedding_evaluation_suite.git_output",
                    side_effect=["suite-commit", ""],
                ),
                patch(
                    "scripts.prepare_embedding_evaluation_suite.verify_vanilla_model_lock_binding",
                    return_value=VANILLA_MODEL_LOCK_BINDING,
                ),
                patch("scripts.prepare_embedding_evaluation_suite.subprocess.run"),
                patch("builtins.print"),
            ):
                self.assertEqual(
                    prepare_suite(config_path, embedding_dir, suite_dir), 0
                )

            plan = json.loads(
                (suite_dir / "suite_plan.json").read_text(encoding="utf-8")
            )
            for protocol in plan["protocols"]:
                protocol_config = json.loads(
                    Path(protocol["config"]).read_text(encoding="utf-8")
                )
                self.assertTrue(
                    protocol_config["checkpoint"].startswith(
                        "base-only:synthetic/base@base-revision:synthetic-base"
                    )
                )
                output_dir = Path(protocol["output_dir"])
                output_dir.mkdir(parents=True)
                for name, content in (
                    ("command.sh", "synthetic wrapper\n"),
                    ("environment.txt", "synthetic environment\n"),
                    ("git_commit.txt", "suite-commit\n"),
                    ("git_status.txt", ""),
                    ("gpu_info.txt", "GPU disabled\n"),
                    ("stdout.log", ""),
                    ("stderr.log", ""),
                ):
                    (output_dir / name).write_text(content, encoding="utf-8")
                with patch(
                    "scripts.evaluate_embedding_artifacts.git_output",
                    side_effect=["suite-commit", ""],
                ):
                    run_evaluation(
                        config_path=Path(protocol["config"]),
                        output_dir=output_dir,
                        argv=["python", "synthetic-vanilla-evaluation"],
                    )
                (output_dir / "exit_code.txt").write_text("0\n", encoding="utf-8")

            with (
                patch(
                    "scripts.prepare_embedding_evaluation_suite.git_output",
                    side_effect=["suite-commit", ""],
                ),
                patch(
                    "scripts.prepare_embedding_evaluation_suite.verify_vanilla_model_lock_binding",
                    return_value=VANILLA_MODEL_LOCK_BINDING,
                ),
                patch("builtins.print"),
            ):
                self.assertEqual(
                    finalize_suite(config_path, embedding_dir, suite_dir), 0
                )
            report = json.loads(
                (suite_dir / "suite_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["binding_type"], "vanilla_backbone")
            self.assertEqual(
                report["model_lock_binding"], VANILLA_MODEL_LOCK_BINDING
            )

    def test_vanilla_suite_rejects_loaded_oea_components(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = synthetic_vanilla_suite_config(root)
            embedding_dir = synthetic_vanilla_embedding_directory(root)
            metrics_path = embedding_dir / "generation_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["lora_loaded"] = True
            write_json(metrics_path, metrics)
            with (
                patch(
                    "scripts.prepare_embedding_evaluation_suite.git_output",
                    side_effect=["suite-commit", ""],
                ),
                patch(
                    "scripts.prepare_embedding_evaluation_suite.verify_vanilla_model_lock_binding",
                    return_value=VANILLA_MODEL_LOCK_BINDING,
                ),
                self.assertRaisesRegex(RuntimeError, "lora_loaded=false"),
            ):
                prepare_suite(
                    config_path,
                    embedding_dir,
                    root / "synthetic_suite_run",
                )

    def test_generation_artifact_drift_is_rejected_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = synthetic_suite_config(root)
            embedding_dir = synthetic_embedding_directory(root)
            embedding_path = embedding_dir / "query_embeddings.npy"
            corrupted = bytearray(embedding_path.read_bytes())
            corrupted[-1] ^= 1
            embedding_path.write_bytes(corrupted)
            suite_dir = root / "synthetic_suite_run"
            with (
                patch(
                    "scripts.prepare_embedding_evaluation_suite.git_output",
                    side_effect=["suite-commit", ""],
                ),
                patch(
                    "scripts.prepare_embedding_evaluation_suite.verify_official_model_lock_binding",
                    return_value=MODEL_LOCK_BINDING,
                ),
                patch("scripts.prepare_embedding_evaluation_suite.subprocess.run"),
                self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"),
            ):
                prepare_suite(config_path, embedding_dir, suite_dir)
            report = json.loads(
                (suite_dir / "suite_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "failed")
            self.assertIn("SHA256 mismatch", report["error"])

    def test_existing_complete_suite_metrics_are_not_overwritten_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = synthetic_suite_config(root)
            embedding_dir = synthetic_embedding_directory(root)
            suite_dir = root / "synthetic_suite_run"
            suite_dir.mkdir()
            original = {"schema_version": 1, "status": "complete", "marker": 1}
            write_json(suite_dir / "suite_metrics.json", original)
            embedding_path = embedding_dir / "query_embeddings.npy"
            corrupted = bytearray(embedding_path.read_bytes())
            corrupted[-1] ^= 1
            embedding_path.write_bytes(corrupted)
            with (
                patch(
                    "scripts.prepare_embedding_evaluation_suite.git_output",
                    side_effect=["suite-commit", ""],
                ),
                patch(
                    "scripts.prepare_embedding_evaluation_suite.verify_official_model_lock_binding",
                    return_value=MODEL_LOCK_BINDING,
                ),
                patch("scripts.prepare_embedding_evaluation_suite.subprocess.run"),
                self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"),
            ):
                prepare_suite(config_path, embedding_dir, suite_dir)
            self.assertEqual(
                json.loads(
                    (suite_dir / "suite_metrics.json").read_text(encoding="utf-8")
                ),
                original,
            )
            failures = list((suite_dir / "failures").glob("failure_*.json"))
            self.assertEqual(len(failures), 1)


if __name__ == "__main__":
    unittest.main()
