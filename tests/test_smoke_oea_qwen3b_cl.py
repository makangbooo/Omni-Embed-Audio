from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.smoke_oea_qwen3b_cl import (
    REPOSITORY_ROOT,
    load_config,
    load_samples,
    verify_file,
)


class SmokeOEAQwen3BClTest(unittest.TestCase):
    def test_fixed_config_has_immutable_resources_and_source_tags(self) -> None:
        config = load_config(
            REPOSITORY_ROOT / "configs/eval/qwen3b_cl_smoke.json"
        )
        self.assertEqual(len(config["base_model"]["revision"]), 40)
        self.assertEqual(len(config["checkpoint"]["revision"]), 40)
        int(config["base_model"]["revision"], 16)
        int(config["checkpoint"]["revision"], 16)
        self.assertEqual(
            config["smoke_config"]["seed"]["source"], "INFERRED"
        )
        for section in (
            "query_prefix",
            "passage_prefix",
            "torch_dtype",
            "projection_dim",
            "projection_dropout",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "lora_targets",
        ):
            self.assertIn("source", config["model_config"][section])

    def test_bundled_sample_manifest_resolves_exactly_five_files(self) -> None:
        config = load_config(
            REPOSITORY_ROOT / "configs/eval/qwen3b_cl_smoke.json"
        )
        samples = load_samples(
            REPOSITORY_ROOT / config["samples"]["manifest"],
            REPOSITORY_ROOT / config["samples"]["audio_dir"],
        )
        self.assertEqual(len(samples), 5)
        self.assertEqual(len({sample["file"] for sample in samples}), 5)
        self.assertTrue(all(sample["path"].is_file() for sample in samples))
        self.assertIn("water_stream.wav", {sample["file"] for sample in samples})

    def test_verify_file_checks_size_and_sha256(self) -> None:
        payload = b"strict offline checkpoint"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            path.write_bytes(payload)
            report = verify_file(
                path,
                len(payload),
                hashlib.sha256(payload).hexdigest(),
            )
            self.assertEqual(report["size_bytes"], len(payload))
            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                verify_file(path, len(payload) + 1, report["sha256"])
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                verify_file(path, len(payload), "0" * 64)


if __name__ == "__main__":
    unittest.main()
