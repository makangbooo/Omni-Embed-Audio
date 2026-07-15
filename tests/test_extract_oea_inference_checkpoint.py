from __future__ import annotations

import unittest
from pathlib import Path

from scripts.extract_oea_inference_checkpoint import is_lora_tensor_key, json_safe


class ExtractOEAInferenceCheckpointTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
