from __future__ import annotations

import json
from pathlib import Path
import unittest

from scripts.benchmark_clap_query_encoder import checkpoint_path, summary


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/eval/clap_clotho_efficiency_rtx4090.json"
RUNNER = ROOT / "scripts/run_clap_clotho_efficiency_rtx4090.sh"


class ClapEfficiencyBenchmarkTests(unittest.TestCase):
    def test_latency_summary_is_deterministic(self) -> None:
        result = summary([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["mean_ms"], 2.5)
        self.assertEqual(result["p50_ms"], 2.5)
        self.assertEqual(result["throughput_queries_per_second"], 400.0)

    def test_protocol_and_paper_values_are_fixed(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        protocol = config["timing_protocol"]
        self.assertEqual(protocol["batch_size"], 1)
        self.assertEqual(protocol["warmup_iterations"], 10)
        self.assertEqual(protocol["audio_measurement_count"], 1045)
        self.assertEqual(protocol["text_measurement_count"], 5225)
        self.assertEqual(
            config["paper_values"]["laion_clap"]["trainable_parameters_m"],
            158.0,
        )
        self.assertEqual(
            config["paper_values"]["mga_clap"]["trainable_parameters_m"],
            148.0,
        )
        self.assertEqual(
            config["paper_values"]["m2d_clap"]["audio_ms_per_clip"], 58.1
        )

    def test_runner_exposes_all_three_backends_and_offline_mode(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        for model in ("laion_clap", "mga_clap", "m2d_clap"):
            self.assertIn(model, text)
        self.assertIn("HF_HUB_OFFLINE=1", text)
        self.assertIn("TRANSFORMERS_OFFLINE=1", text)
        self.assertIn("git status --short", text)

    def test_checkpoint_paths_are_explicit(self) -> None:
        root = Path("/models")
        self.assertEqual(
            checkpoint_path("laion_clap", root).as_posix(),
            "/models/laion-clap/630k-audioset-best.pt",
        )
        self.assertEqual(
            checkpoint_path("mga_clap", root).name,
            "model.pt",
        )
        self.assertEqual(
            checkpoint_path("m2d_clap", root).name,
            "checkpoint-30.pth",
        )


if __name__ == "__main__":
    unittest.main()
