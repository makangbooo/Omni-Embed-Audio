from __future__ import annotations

import unittest
from pathlib import Path

from scripts.extract_oea_inference_checkpoint import (
    assert_derived_state_equal,
    is_lora_tensor_key,
    json_safe,
)


class ExtractOEAInferenceCheckpointTest(unittest.TestCase):
    class FakeTorch:
        @staticmethod
        def equal(left: object, right: object) -> bool:
            return left == right

    def test_lora_key_selection_excludes_base_layer(self) -> None:
        self.assertTrue(is_lora_tensor_key("layer.q_proj.lora_A.default.weight"))
        self.assertTrue(is_lora_tensor_key("layer.q_proj.lora_B.default.weight"))
        self.assertFalse(is_lora_tensor_key("layer.q_proj.base_layer.weight"))

    def test_json_safe_normalizes_paths(self) -> None:
        value = {"path": Path("data/audio"), "items": (1, Path("model.pt"))}
        self.assertEqual(
            json_safe(value),
            {"path": "data/audio", "items": [1, "model.pt"]},
        )

    def test_json_safe_rejects_unknown_objects(self) -> None:
        with self.assertRaisesRegex(TypeError, "unsupported"):
            json_safe(object())

    def test_existing_derived_state_requires_exact_tensors_and_metadata(self) -> None:
        expected = {
            "schema_version": 1,
            "source_checkpoint_sha256": "a" * 64,
            "lora_state_dict": {"layer.lora_A": (1, 2)},
            "audio_head": {"weight": (3, 4)},
            "text_head": {"weight": (5, 6)},
            "config": {"rank": 16},
            "metrics": {"recall": 0.5},
            "global_step": 40,
        }
        actual = {
            key: dict(value) if isinstance(value, dict) else value
            for key, value in expected.items()
        }
        assert_derived_state_equal(expected, actual, self.FakeTorch)

        actual["lora_state_dict"]["layer.lora_A"] = (9, 9)
        with self.assertRaisesRegex(RuntimeError, "lora_state_dict"):
            assert_derived_state_equal(expected, actual, self.FakeTorch)

        actual["lora_state_dict"]["layer.lora_A"] = (1, 2)
        actual["source_checkpoint_sha256"] = "b" * 64
        with self.assertRaisesRegex(RuntimeError, "source_checkpoint_sha256"):
            assert_derived_state_equal(expected, actual, self.FakeTorch)

        actual["source_checkpoint_sha256"] = "a" * 64
        actual["unexpected"] = True
        with self.assertRaisesRegex(RuntimeError, "top-level keys"):
            assert_derived_state_equal(expected, actual, self.FakeTorch)


if __name__ == "__main__":
    unittest.main()
