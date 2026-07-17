from __future__ import annotations

import unittest

from scripts.inspect_oea_checkpoint import tensor_mapping_summary


class FakeTensor:
    def __init__(self, shape: tuple[int, ...], element_size: int, dtype: str) -> None:
        self.shape = shape
        self._element_size = element_size
        self.dtype = dtype

    def numel(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    def element_size(self) -> int:
        return self._element_size


class FakeTorch:
    @staticmethod
    def is_tensor(value: object) -> bool:
        return isinstance(value, FakeTensor)


class InspectOEACheckpointTest(unittest.TestCase):
    def test_lora_subset_counts_and_bytes_exclude_frozen_weights(self) -> None:
        summary = tensor_mapping_summary(
            {
                "layer.lora_A.default.weight": FakeTensor((2, 3), 2, "bfloat16"),
                "layer.lora_B.default.weight": FakeTensor((3, 2), 2, "bfloat16"),
                "layer.base_layer.weight": FakeTensor((4, 4), 4, "float32"),
                "metadata": "ignored",
            },
            FakeTorch,
        )
        self.assertEqual(summary["tensor_count"], 3)
        self.assertEqual(summary["lora_tensor_count"], 2)
        self.assertEqual(summary["lora_total_numel"], 12)
        self.assertEqual(summary["lora_estimated_tensor_bytes"], 24)
        self.assertEqual(summary["lora_dtype_tensor_counts"], {"bfloat16": 2})
        self.assertEqual(summary["non_lora_tensor_count"], 1)


if __name__ == "__main__":
    unittest.main()
