from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.evaluate_embedding_artifacts import file_identity, run_evaluation
from scripts.prepare_positive_uiq_evaluation_suite import (
    finalize_suite,
    load_suite_config,
    prepare_suite,
    validate_uiq_metadata,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
QUERY_TYPES = ("question", "imperative", "paraphrase", "tagging")
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
        "sha256": "d" * 64,
    },
    "protocol_config": {
        "repository_path": "configs/eval/synthetic.json",
        "size_bytes": 10,
        "sha256": "b" * 64,
    },
}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def synthetic_config(root: Path) -> Path:
    protocols = []
    for index, query_type in enumerate(QUERY_TYPES, start=12):
        protocols.append(
            {
                "protocol_id": f"{query_type}_released_uiq",
                "released_query_type": query_type,
                "paper_query_type": (
                    "Keyphrase" if query_type == "tagging" else query_type.title()
                ),
                "paper_table": f"Table {index}",
                "protocol_label": f"released-{query_type}",
                "protocol_source": "CODE",
                "expected_evaluated_queries": 2,
                "note": "synthetic",
            }
        )
    checkpoint = {
        "repo_id": "synthetic/checkpoint",
        "revision": "checkpoint-revision",
        "local_subpath": "checkpoint.pt",
        "size_bytes": 20,
        "sha256": "a" * 64,
        "source": "DERIVED_FROM_CODE_CHECKPOINT",
    }
    config = {
        "schema_version": 1,
        "suite_prefix": "synthetic_positive_uiq_suite",
        "model": "synthetic-model",
        "caption_dataset": "synthetic-caption-dataset",
        "dataset": "synthetic-uiq-dataset",
        "embedding_seed": 42,
        "expected_candidate_count": 2,
        "expected_caption_query_count": 4,
        "expected_uiq_query_count": 8,
        "expected_queries_per_type": 2,
        "expected_embedding_dim": 2,
        "expected_captions_per_clip": 2,
        "expected_caption_generation_protocol_sha256": "b" * 64,
        "expected_uiq_generation_config_sha256": "c" * 64,
        "official_variant_id": "synthetic-variant",
        "official_model_lock_path": "results/model_locks/synthetic.json",
        "minimum_caption_generator_commit": "caption-minimum",
        "minimum_uiq_generator_commit": "uiq-minimum",
        "checkpoint": checkpoint,
        "embedding_protocol": {"source": "CODE"},
        "protocols": protocols,
    }
    path = root / "suite_config.json"
    write_json(path, config)
    return path


def synthetic_caption_embeddings(root: Path) -> Path:
    directory = root / "caption_embeddings"
    directory.mkdir()
    np.save(directory / "candidate_embeddings.npy", np.eye(2, dtype=np.float32))
    np.save(
        directory / "query_embeddings.npy",
        np.asarray(
            [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [0.6, 0.8]],
            dtype=np.float32,
        ),
    )
    candidates = [
        {"candidate_index": 0, "candidate_id": "a"},
        {"candidate_index": 1, "candidate_id": "b"},
    ]
    queries = []
    for candidate in candidates:
        clip_id = candidate["candidate_id"]
        for caption_index in (1, 2):
            queries.append(
                {
                    "query_index": len(queries),
                    "query_id": f"{clip_id}#caption_{caption_index}",
                    "target_id": clip_id,
                    "clip_id": clip_id,
                    "caption_index": caption_index,
                    "text": f"{clip_id} caption {caption_index}",
                }
            )
    write_jsonl(directory / "candidate_metadata.jsonl", candidates)
    write_jsonl(directory / "query_metadata.jsonl", queries)
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
        "resolution_git_commit": "caption-generation-commit",
    }
    write_json(directory / "resolved_embedding_config.json", resolved)
    write_json(
        directory / "config.yaml",
        {**resolved, "resolved_paths": {"output_dir": str(directory)}},
    )
    write_json(
        directory / "run_identity.json",
        {
            "git_commit": "caption-generation-commit",
            "config": file_identity(directory / "resolved_embedding_config.json"),
            "official_model_lock": MODEL_LOCK_BINDING,
            "candidate_count": 2,
            "query_count": 4,
            "checkpoint_revision": "checkpoint-revision",
        },
    )
    artifacts = {
        name: file_identity(directory / filename)
        for name, filename in {
            "candidate_embeddings": "candidate_embeddings.npy",
            "query_embeddings": "query_embeddings.npy",
            "candidate_metadata": "candidate_metadata.jsonl",
            "query_metadata": "query_metadata.jsonl",
        }.items()
    }
    write_json(
        directory / "generation_metrics.json",
        {
            "schema_version": 1,
            "status": "complete",
            "model": "synthetic-model",
            "dataset": "synthetic-caption-dataset",
            "seed": 42,
            "official_model_lock": MODEL_LOCK_BINDING,
            "artifacts": artifacts,
        },
    )
    return directory


def synthetic_uiq_embeddings(root: Path) -> Path:
    directory = root / "uiq_embeddings"
    directory.mkdir()
    vectors = []
    rows = []
    candidate_vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    paper_names = {
        "question": "Question",
        "imperative": "Imperative",
        "paraphrase": "Paraphrase",
        "tagging": "Keyphrase",
    }
    for query_type in QUERY_TYPES:
        for candidate_id in ("a", "b"):
            vectors.append(candidate_vectors[candidate_id])
            rows.append(
                {
                    "query_index": len(rows),
                    "query_id": f"{candidate_id}#uiq_{query_type}",
                    "target_id": candidate_id,
                    "clip_id": candidate_id,
                    "released_query_type": query_type,
                    "paper_query_type": paper_names[query_type],
                    "text": f"{query_type} {candidate_id}",
                    "release_row_index": len(rows) + 1,
                    "source_model": "gpt-5.1",
                    "regen_model": "gpt-5.1",
                }
            )
    np.save(directory / "query_embeddings.npy", np.asarray(vectors, dtype=np.float32))
    write_jsonl(directory / "query_metadata.jsonl", rows)
    query_sources = [
        {"released_query_type": query_type, "rows": 2}
        for query_type in QUERY_TYPES
    ]
    resolved_base = {
        "resolution_git_commit": "uiq-generation-commit",
    }
    write_json(directory / "resolved_base_embedding_config.json", resolved_base)
    write_json(
        directory / "config.yaml",
        {
            "model": "synthetic-model",
            "dataset": "synthetic-uiq-dataset",
            "seed": 42,
            "expected_total_queries": 8,
            "expected_embedding_dim": 2,
        },
    )
    write_json(
        directory / "run_identity.json",
        {
            "git_commit": "uiq-generation-commit",
            "config": {"sha256": "c" * 64},
            "base_embedding_config": file_identity(
                directory / "resolved_base_embedding_config.json"
            ),
            "official_model_lock": MODEL_LOCK_BINDING,
            "query_count": 8,
            "checkpoint_revision": "checkpoint-revision",
            "query_sources": query_sources,
        },
    )
    artifacts = {
        "query_embeddings": file_identity(directory / "query_embeddings.npy"),
        "query_metadata": file_identity(directory / "query_metadata.jsonl"),
    }
    write_json(
        directory / "generation_metrics.json",
        {
            "schema_version": 1,
            "status": "complete",
            "model": "synthetic-model",
            "dataset": "synthetic-uiq-dataset",
            "seed": 42,
            "base_embedding_config": file_identity(
                directory / "resolved_base_embedding_config.json"
            ),
            "official_model_lock": MODEL_LOCK_BINDING,
            "query_count": 8,
            "query_type_counts": {query_type: 2 for query_type in QUERY_TYPES},
            "query_sources": query_sources,
            "artifacts": artifacts,
        },
    )
    return directory


class PreparePositiveUIQEvaluationSuiteTest(unittest.TestCase):
    def test_direct_script_entrypoint_resolves_repository_packages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(
                        REPOSITORY_ROOT
                        / "scripts/prepare_positive_uiq_evaluation_suite.py"
                    ),
                    "--help",
                ],
                cwd=directory,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("{prepare,finalize}", completed.stdout)

    def test_fixed_config_pins_generators_and_four_paper_tables(self) -> None:
        config = load_suite_config(
            REPOSITORY_ROOT
            / "configs/eval/qwen3b_cl_clotho_positive_uiq_suite.json"
        )
        self.assertEqual(config["expected_candidate_count"], 1045)
        self.assertEqual(config["expected_uiq_query_count"], 4180)
        self.assertEqual(
            [protocol["paper_table"] for protocol in config["protocols"]],
            ["Table 12", "Table 13", "Table 14", "Table 15"],
        )
        self.assertEqual(config["protocols"][-1]["released_query_type"], "tagging")
        self.assertEqual(config["protocols"][-1]["paper_query_type"], "Keyphrase")
        self.assertEqual(
            config["minimum_uiq_generator_commit"],
            "d2b54ba5b61bf8166e070f2a47fc9b7bcd95c337",
        )
        uiq_config = file_identity(
            REPOSITORY_ROOT
            / "configs/eval/qwen3b_cl_clotho_positive_uiq_embeddings.json"
        )
        self.assertEqual(
            uiq_config["sha256"],
            config["expected_uiq_generation_config_sha256"],
        )

    def test_type_major_metadata_selection_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_suite_config(synthetic_config(root))
            caption_dir = synthetic_caption_embeddings(root)
            uiq_dir = synthetic_uiq_embeddings(root)
            candidates = [
                json.loads(line)
                for line in (caption_dir / "candidate_metadata.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            queries = [
                json.loads(line)
                for line in (uiq_dir / "query_metadata.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            indices, rows = validate_uiq_metadata(candidates, queries, config)
        self.assertEqual(indices["question"], [0, 1])
        self.assertEqual(indices["tagging"], [6, 7])
        self.assertEqual(rows["tagging"][0]["paper_query_type"], "Keyphrase")

    def test_prepare_evaluate_and_finalize_synthetic_suite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = synthetic_config(root)
            caption_dir = synthetic_caption_embeddings(root)
            uiq_dir = synthetic_uiq_embeddings(root)
            suite_dir = root / "synthetic_positive_uiq_suite_run"
            with (
                patch(
                    "scripts.prepare_positive_uiq_evaluation_suite.git_output",
                    side_effect=["suite-commit", ""],
                ),
                patch(
                    "scripts.prepare_embedding_evaluation_suite.verify_official_model_lock_binding",
                    return_value=MODEL_LOCK_BINDING,
                ),
                patch(
                    "scripts.prepare_positive_uiq_evaluation_suite.verify_official_model_lock_binding",
                    return_value=MODEL_LOCK_BINDING,
                ),
                patch("scripts.prepare_positive_uiq_evaluation_suite.subprocess.run"),
                patch("builtins.print"),
            ):
                self.assertEqual(
                    prepare_suite(config_path, caption_dir, uiq_dir, suite_dir), 0
                )
            plan = json.loads(
                (suite_dir / "suite_plan.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(plan["protocols"]), 4)
            self.assertEqual(
                json.loads(
                    (suite_dir / "selections/tagging_indices.json").read_text(
                        encoding="utf-8"
                    )
                ),
                [6, 7],
            )
            for protocol in plan["protocols"]:
                output_dir = Path(protocol["output_dir"])
                output_dir.mkdir(parents=True)
                for name, content in {
                    "command.sh": "synthetic wrapper\n",
                    "environment.txt": "synthetic environment\n",
                    "git_commit.txt": "suite-commit\n",
                    "git_status.txt": "",
                    "gpu_info.txt": "GPU disabled\n",
                    "stdout.log": "",
                    "stderr.log": "",
                }.items():
                    (output_dir / name).write_text(content, encoding="utf-8")
                with patch(
                    "scripts.evaluate_embedding_artifacts.git_output",
                    side_effect=["suite-commit", ""],
                ):
                    run_evaluation(
                        config_path=Path(protocol["config"]),
                        output_dir=output_dir,
                        argv=["python", "synthetic-positive-uiq-evaluation"],
                    )
                (output_dir / "exit_code.txt").write_text("0\n", encoding="utf-8")
            with (
                patch(
                    "scripts.prepare_positive_uiq_evaluation_suite.git_output",
                    side_effect=["suite-commit", ""],
                ),
                patch(
                    "scripts.prepare_embedding_evaluation_suite.verify_official_model_lock_binding",
                    return_value=MODEL_LOCK_BINDING,
                ),
                patch(
                    "scripts.prepare_positive_uiq_evaluation_suite.verify_official_model_lock_binding",
                    return_value=MODEL_LOCK_BINDING,
                ),
                patch("builtins.print"),
            ):
                self.assertEqual(
                    finalize_suite(config_path, caption_dir, uiq_dir, suite_dir), 0
                )
            report = json.loads(
                (suite_dir / "suite_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["protocol_count"], 4)
            self.assertEqual(
                len(
                    (suite_dir / "positive_uiq_summary.csv")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ),
                21,
            )

    def test_uiq_artifact_drift_is_rejected_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = synthetic_config(root)
            caption_dir = synthetic_caption_embeddings(root)
            uiq_dir = synthetic_uiq_embeddings(root)
            query_path = uiq_dir / "query_embeddings.npy"
            corrupted = bytearray(query_path.read_bytes())
            corrupted[-1] ^= 1
            query_path.write_bytes(corrupted)
            suite_dir = root / "synthetic_positive_uiq_suite_run"
            with (
                patch(
                    "scripts.prepare_positive_uiq_evaluation_suite.git_output",
                    side_effect=["suite-commit", ""],
                ),
                patch(
                    "scripts.prepare_embedding_evaluation_suite.verify_official_model_lock_binding",
                    return_value=MODEL_LOCK_BINDING,
                ),
                patch(
                    "scripts.prepare_positive_uiq_evaluation_suite.verify_official_model_lock_binding",
                    return_value=MODEL_LOCK_BINDING,
                ),
                patch("scripts.prepare_positive_uiq_evaluation_suite.subprocess.run"),
                self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"),
            ):
                prepare_suite(config_path, caption_dir, uiq_dir, suite_dir)
            report = json.loads(
                (suite_dir / "suite_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "failed")


if __name__ == "__main__":
    unittest.main()
