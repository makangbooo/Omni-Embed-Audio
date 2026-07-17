from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.run_reproduction import (
    ALLOWED_STATUSES,
    DEFAULT_REGISTRY,
    FORBIDDEN_COMMAND_FRAGMENTS,
    REQUIRED_PUBLIC_STAGE_IDS,
    execute_stage,
    flattened_plan,
    load_stage_registry,
    plan_payload,
    render_command,
)


def arguments(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "variant": "oea_qwen3b_cl",
        "backbone": "vanilla_nemotron_3b",
        "embedding_dir": None,
        "caption_embedding_dir": None,
        "uiq_embedding_dir": None,
        "smoke_metrics": None,
        "verify_existing_derived": False,
        "acknowledge_long_operation": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class RunReproductionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_stage_registry(DEFAULT_REGISTRY)

    def test_registry_exposes_required_public_stages_and_allowed_statuses(self) -> None:
        for stage_id in REQUIRED_PUBLIC_STAGE_IDS:
            self.assertIn(stage_id, self.registry)
        self.assertTrue(
            all(
                stage["status"] in ALLOWED_STATUSES
                for stage in self.registry.values()
            )
        )

    def test_registered_commands_are_non_destructive(self) -> None:
        for stage in self.registry.values():
            command = " ".join(stage.get("command", ())).lower()
            for fragment in FORBIDDEN_COMMAND_FRAGMENTS:
                with self.subTest(stage=stage["stage_id"], fragment=fragment):
                    self.assertNotIn(fragment, command)

    def test_all_plan_is_deterministic_and_deduplicates_shared_children(self) -> None:
        stage_ids = [
            stage["stage_id"] for stage in flattened_plan("all", self.registry)
        ]
        self.assertEqual(len(stage_ids), len(set(stage_ids)))
        self.assertEqual(stage_ids[:4], ["data04", "data05", "data06", "data07"])
        self.assertEqual(stage_ids.count("official_eval_embeddings"), 1)
        self.assertIn("train_qwen3b", stage_ids)
        self.assertIn("vanilla_model_lock", stage_ids)
        self.assertIn("vanilla_smoke_embeddings", stage_ids)
        self.assertIn("vanilla_full_embeddings", stage_ids)
        self.assertIn("vanilla_metrics", stage_ids)
        self.assertIn("clap_baselines", stage_ids)

    def test_official_eval_plan_preserves_cpu_gpu_cpu_handoff(self) -> None:
        payload = plan_payload(
            "official_eval",
            self.registry,
            {
                "variant": "oea_qwen3b_cl",
                "backbone": "vanilla_nemotron_3b",
                "embedding_dir": None,
                "caption_embedding_dir": None,
                "uiq_embedding_dir": None,
                "smoke_metrics": None,
            },
            verify_existing_derived=True,
        )
        self.assertEqual(
            [step["resource"] for step in payload["steps"]],
            ["CPU", "1xA100-80GB", "CPU"],
        )
        self.assertIn(
            "--verify-existing-derived", payload["steps"][0]["command"]
        )
        self.assertIn("<embedding_dir>", payload["steps"][2]["command"])

    def test_render_command_requires_runtime_directory_only_on_execute(self) -> None:
        stage = self.registry["official_eval_metrics"]
        values = {
            "variant": "oea_qwen3b_cl",
            "backbone": "vanilla_nemotron_3b",
            "embedding_dir": None,
            "caption_embedding_dir": None,
            "uiq_embedding_dir": None,
            "smoke_metrics": None,
        }
        self.assertIn(
            "<embedding_dir>", render_command(stage, values, allow_missing=True)
        )
        with self.assertRaisesRegex(ValueError, "requires --embedding-dir"):
            render_command(stage, values, allow_missing=False)

    def test_vanilla_plan_preserves_cpu_gpu_gpu_cpu_handoff(self) -> None:
        payload = plan_payload(
            "vanilla_baselines",
            self.registry,
            {
                "variant": "oea_qwen3b_cl",
                "backbone": "vanilla_qwen2_5_omni_3b",
                "embedding_dir": None,
                "caption_embedding_dir": None,
                "uiq_embedding_dir": None,
                "smoke_metrics": None,
            },
            verify_existing_derived=False,
        )
        self.assertEqual(
            [step["resource"] for step in payload["steps"]],
            ["CPU", "1xA100-80GB", "1xA100-80GB", "CPU"],
        )
        self.assertEqual(
            payload["steps"][0]["command"],
            [
                "bash",
                "scripts/run_vanilla_backbone_model_pipeline.sh",
                "vanilla_qwen2_5_omni_3b",
            ],
        )
        self.assertEqual(payload["steps"][1]["command"][-1], "--smoke")
        self.assertEqual(
            payload["steps"][2]["command"][-2:],
            ["--full", "<smoke_metrics>"],
        )

    def test_vanilla_full_requires_explicit_smoke_metrics_on_execute(self) -> None:
        stage = self.registry["vanilla_full_embeddings"]
        values = {
            "variant": "oea_qwen3b_cl",
            "backbone": "vanilla_nemotron_3b",
            "embedding_dir": None,
            "caption_embedding_dir": None,
            "uiq_embedding_dir": None,
            "smoke_metrics": None,
        }
        self.assertIn(
            "<smoke_metrics>", render_command(stage, values, allow_missing=True)
        )
        with self.assertRaisesRegex(ValueError, "requires --smoke-metrics"):
            render_command(stage, values, allow_missing=False)

    def test_group_and_blocked_stages_never_execute(self) -> None:
        for stage_id in ("official_eval", "train_qwen3b", "baselines"):
            with self.subTest(stage=stage_id), self.assertRaisesRegex(
                RuntimeError, "plan-only"
            ):
                execute_stage(self.registry[stage_id], arguments())

    def test_long_stage_requires_explicit_review_acknowledgement(self) -> None:
        stage = copy.deepcopy(self.registry["official_model_lock"])
        stage["status"] = "TODO"
        with self.assertRaisesRegex(RuntimeError, "acknowledge-long-operation"):
            execute_stage(stage, arguments())

    def test_ready_execution_uses_argument_vector_and_repository_cwd(self) -> None:
        completed = Mock(returncode=7)
        with (
            patch("scripts.run_reproduction.git_output", return_value=""),
            patch(
                "scripts.run_reproduction.subprocess.run", return_value=completed
            ) as run,
            patch("builtins.print"),
        ):
            code = execute_stage(self.registry["data04"], arguments())
        self.assertEqual(code, 7)
        run.assert_called_once_with(
            ["bash", "scripts/download_data04_mecat_00a_test.sh"],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
        )

    def test_registry_rejects_group_cycles(self) -> None:
        document = json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8"))
        for stage in document["stages"]:
            if stage["stage_id"] == "all":
                stage["children"] = ["all"]
                break
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stages.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cycle"):
                load_stage_registry(path)


if __name__ == "__main__":
    unittest.main()
