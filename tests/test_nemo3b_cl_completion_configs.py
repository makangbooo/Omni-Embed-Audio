from __future__ import annotations

import hashlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.generate_oea_uiq_embeddings import load_uiq_config, print_progress
from scripts.prepare_embedding_evaluation_suite import load_suite_config as load_retrieval_suite
from scripts.prepare_positive_uiq_evaluation_suite import load_suite_config as load_uiq_suite


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPOSITORY_ROOT / "configs/eval"


class Nemo3BClCompletionConfigsTests(unittest.TestCase):
    def test_retrieval_suite_contains_only_missing_t2t_protocols(self) -> None:
        config = load_retrieval_suite(
            CONFIG_ROOT / "nemo3b_cl_clotho_t2t_suite.json"
        )
        self.assertEqual(config["official_variant_id"], "oea_nemo3b_cl")
        self.assertEqual(len(config["protocols"]), 2)
        self.assertEqual(
            [row["task"] for row in config["protocols"]],
            ["t2t", "t2t"],
        )
        self.assertEqual(
            [row["protocol_source"] for row in config["protocols"]],
            ["CODE", "INFERRED"],
        )
        wrapper = (
            REPOSITORY_ROOT / "scripts/run_nemo3b_cl_clotho_t2t_suite.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("nemo3b_cl_clotho_t2t_suite.json", wrapper)
        self.assertIn("Completed T2A protocols are intentionally not repeated", wrapper)

    def test_nemo_uiq_configs_are_hash_bound(self) -> None:
        embedding_path = CONFIG_ROOT / "nemo3b_cl_clotho_positive_uiq_embeddings.json"
        embedding_config = load_uiq_config(embedding_path)
        suite_config = load_uiq_suite(
            CONFIG_ROOT / "nemo3b_cl_clotho_positive_uiq_suite.json"
        )
        self.assertEqual(embedding_config["model"], "OEA-Nemo3B (+Cl)")
        self.assertEqual(suite_config["official_variant_id"], "oea_nemo3b_cl")
        self.assertEqual(
            suite_config["expected_uiq_generation_config_sha256"],
            hashlib.sha256(embedding_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(len(suite_config["protocols"]), 4)

    def test_progress_output_has_required_eta_fields(self) -> None:
        output = io.StringIO()
        with patch("sys.stdout", output), patch(
            "scripts.generate_oea_uiq_embeddings.time.monotonic", return_value=20.0
        ):
            print_progress(
                completed=50,
                total=100,
                completed_this_attempt=10,
                stage_started=10.0,
                overall_started_at="2026-07-28T00:00:00+00:00",
            )
        line = output.getvalue()
        for field in (
            "stage=uiq_text",
            "current=50",
            "total=100",
            "percent=50.00%",
            "stage_elapsed=",
            "overall_elapsed=",
            "throughput=",
            "stage_remaining=",
            "overall_remaining=",
            "expected_completion=",
        ):
            self.assertIn(field, line)

    def test_tmux_wrapper_retains_pane_and_final_summary(self) -> None:
        source = (
            REPOSITORY_ROOT
            / "scripts/run_nemo3b_cl_clotho_positive_uiq_tmux.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('[[ -z "${TMUX:-}" ]]', source)
        self.assertIn("TOTAL_WORKLOAD=4,180", source)
        self.assertIn("ESTIMATED_TOTAL_TIME=8-25 minutes", source)
        self.assertIn("FINAL_RUN_RC=", source)
        self.assertIn("METRICS_PATH=", source)
        self.assertIn("FAILED_STAGE=", source)
        self.assertIn("exec bash -i", source)
        self.assertIn('tee -a "${LOG_FILE}"', source)
        self.assertNotIn("nohup", source)
        self.assertNotIn("rm -rf", source)

    def test_t2t_tmux_wrapper_retains_pane_and_reports_progress(self) -> None:
        wrapper = (
            REPOSITORY_ROOT / "scripts/run_nemo3b_cl_clotho_t2t_tmux.sh"
        ).read_text(encoding="utf-8")
        retrieval_runner = (
            REPOSITORY_ROOT / "scripts/run_qwen3b_clotho_retrieval_suite.sh"
        ).read_text(encoding="utf-8")
        for field in (
            '[[ -z "${TMUX:-}" ]]',
            "TOTAL_WORKLOAD=2",
            "ESTIMATED_TOTAL_TIME=5-15 minutes",
            "FINAL_RUN_RC=",
            "METRICS_PATH=",
            "FAILED_STAGE=",
            "exec bash -i",
            'tee -a "${LOG_FILE}"',
        ):
            self.assertIn(field, wrapper)
        for field in (
            "stage_elapsed=",
            "overall_elapsed=",
            "throughput=",
            "stage_remaining=",
            "overall_remaining=",
            "expected_completion=",
        ):
            self.assertIn(field, retrieval_runner)
        self.assertNotIn("nohup", wrapper)
        self.assertNotIn("rm -rf", wrapper)


if __name__ == "__main__":
    unittest.main()
