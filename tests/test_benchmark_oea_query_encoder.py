from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.benchmark_oea_query_encoder import (
    latency_summary,
    load_benchmark_config,
)


class OEAQueryEncoderBenchmarkTest(unittest.TestCase):
    def test_fixed_config_records_paper_values_and_missing_protocol(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_benchmark_config(
            root / "configs/eval/qwen3b_cl_clotho_efficiency.json"
        )
        self.assertEqual(
            config["hardware"]["required_gpu_name"], "NVIDIA A100-SXM4-80GB"
        )
        self.assertEqual(config["timing_protocol"]["warmup_iterations"], 10)
        self.assertEqual(config["timing_protocol"]["batch_size"], 1)
        self.assertEqual(config["timing_protocol"]["audio_measurement_count"], 1045)
        self.assertEqual(config["timing_protocol"]["text_measurement_count"], 5225)
        self.assertEqual(config["paper_values"]["audio_ms_per_clip"], 539.3)
        self.assertEqual(config["paper_values"]["text_ms_per_query"], 2.6)
        self.assertIn("does not publish", config["timing_protocol"]["paper_gap"])
        self.assertEqual(
            config["claim_scope"]["hardware_alignment"], "same-hardware-model"
        )

    def test_rtx4090_config_is_explicitly_hardware_mismatched(self) -> None:
        root = Path(__file__).resolve().parents[1]
        a100 = load_benchmark_config(
            root / "configs/eval/qwen3b_cl_clotho_efficiency.json"
        )
        rtx4090 = load_benchmark_config(
            root / "configs/eval/qwen3b_cl_clotho_efficiency_rtx4090.json"
        )
        self.assertEqual(
            rtx4090["hardware"]["required_gpu_name"],
            "NVIDIA GeForce RTX 4090",
        )
        self.assertEqual(
            rtx4090["claim_scope"]["hardware_alignment"], "hardware-mismatched"
        )
        self.assertEqual(rtx4090["claim_scope"]["result_label"], "INFERRED")
        self.assertIn("same-hardware", rtx4090["claim_scope"]["forbidden_claim"])
        self.assertEqual(rtx4090["timing_protocol"], a100["timing_protocol"])
        self.assertEqual(
            rtx4090["paper_values"]["audio_ms_per_clip"],
            a100["paper_values"]["audio_ms_per_clip"],
        )

    def test_rtx4090_wrapper_preserves_a100_entrypoint(self) -> None:
        root = Path(__file__).resolve().parents[1]
        wrapper = (
            root / "scripts/run_qwen3b_cl_clotho_efficiency_rtx4090.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("qwen3b_cl_clotho_efficiency_rtx4090.json", wrapper)
        self.assertIn("oea_qwen3b_clotho_rtx4090_efficiency_", wrapper)
        self.assertIn("HF_HUB_OFFLINE=1", wrapper)
        self.assertIn("TRANSFORMERS_OFFLINE=1", wrapper)
        self.assertIn("git status --short", wrapper)
        self.assertIn('if [[ -e "${OUTPUT_DIR}" ]]', wrapper)
        self.assertNotIn("rm -rf", wrapper)

        a100_wrapper = (
            root / "scripts/run_qwen3b_cl_clotho_efficiency.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("qwen3b_cl_clotho_efficiency.json", a100_wrapper)
        self.assertNotIn("efficiency_rtx4090.json", a100_wrapper)

    def test_latency_summary_is_population_statistic(self) -> None:
        summary = latency_summary([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["mean_ms"], 2.5)
        self.assertAlmostEqual(summary["std_ms"], 1.118033988749895)
        self.assertEqual(summary["p50_ms"], 2.5)
        self.assertEqual(summary["throughput_queries_per_second"], 400.0)

    def test_invalid_nonpositive_warmup_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            root = Path(__file__).resolve().parents[1]
            config = json.loads(
                (root / "configs/eval/qwen3b_cl_clotho_efficiency.json").read_text(
                    encoding="utf-8"
                )
            )
            config["timing_protocol"]["warmup_iterations"] = 0
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "warmup_iterations"):
                load_benchmark_config(path)


if __name__ == "__main__":
    unittest.main()
