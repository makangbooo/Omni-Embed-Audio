from __future__ import annotations

import unittest

from scripts.validate_single_bf16_gpu import validate_observation


class ValidateSingleBF16GPUTest(unittest.TestCase):
    def test_accepts_a100_and_the_measured_rtx4090(self) -> None:
        devices = (
            {
                "name": "NVIDIA A100-SXM4-80GB",
                "total_memory_bytes": 80 * 1024**3,
            },
            {
                "name": "NVIDIA GeForce RTX 4090",
                "total_memory_bytes": 25250627584,
            },
        )
        for device in devices:
            with self.subTest(name=device["name"]):
                errors = validate_observation(
                    cuda_available=True,
                    bf16_supported=True,
                    devices=[device],
                )
                self.assertEqual(errors, [])

    def test_rejects_zero_or_multiple_visible_devices(self) -> None:
        for devices in (
            [],
            [
                {"name": "GPU one", "total_memory_bytes": 1},
                {"name": "GPU two", "total_memory_bytes": 1},
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
            devices=[{"name": "GPU", "total_memory_bytes": 1}],
        )
        self.assertIn("BF16 is not supported", errors)


if __name__ == "__main__":
    unittest.main()
