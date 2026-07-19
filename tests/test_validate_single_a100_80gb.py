from __future__ import annotations

import unittest

from scripts.validate_single_a100_80gb import (
    MINIMUM_TOTAL_MEMORY_BYTES,
    validate_observation,
)


class ValidateSingleA10080GBTest(unittest.TestCase):
    def test_accepts_exactly_one_a100_80gb_with_bf16(self) -> None:
        errors = validate_observation(
            cuda_available=True,
            bf16_supported=True,
            devices=[
                {
                    "name": "NVIDIA A100-SXM4-80GB",
                    "total_memory_bytes": 80 * 1024**3,
                }
            ],
        )
        self.assertEqual(errors, [])

    def test_rejects_the_observed_rtx4090(self) -> None:
        errors = validate_observation(
            cuda_available=True,
            bf16_supported=True,
            devices=[
                {
                    "name": "NVIDIA GeForce RTX 4090",
                    "total_memory_bytes": 25250627584,
                }
            ],
        )
        self.assertTrue(any("A100" in error for error in errors))
        self.assertTrue(any("79 GiB" in error for error in errors))

    def test_rejects_zero_or_multiple_visible_devices(self) -> None:
        for devices in (
            [],
            [
                {
                    "name": "NVIDIA A100-SXM4-80GB",
                    "total_memory_bytes": MINIMUM_TOTAL_MEMORY_BYTES,
                },
                {
                    "name": "NVIDIA A100-SXM4-80GB",
                    "total_memory_bytes": MINIMUM_TOTAL_MEMORY_BYTES,
                },
            ],
        ):
            with self.subTest(count=len(devices)):
                errors = validate_observation(
                    cuda_available=bool(devices),
                    bf16_supported=True,
                    devices=devices,
                )
                self.assertTrue(any("exactly one" in error for error in errors))

    def test_rejects_missing_bf16_support(self) -> None:
        errors = validate_observation(
            cuda_available=True,
            bf16_supported=False,
            devices=[
                {
                    "name": "NVIDIA A100-SXM4-80GB",
                    "total_memory_bytes": MINIMUM_TOTAL_MEMORY_BYTES,
                }
            ],
        )
        self.assertIn("BF16 is not supported", errors)


if __name__ == "__main__":
    unittest.main()
