from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_official_oea_checkpoint import (
    DEFAULT_REGISTRY,
    EXPECTED_VARIANT_IDS,
    REPOSITORY_ROOT,
    external_model_root,
    extraction_command,
    inspection_command,
    load_checkpoint_registry,
    structure_expectations,
    variant_paths,
)


class PrepareOfficialOEACheckpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_checkpoint_registry(DEFAULT_REGISTRY)

    def test_registry_fixes_all_six_official_checkpoint_identities(self) -> None:
        self.assertEqual(tuple(self.registry), EXPECTED_VARIANT_IDS)
        expected = {
            "oea_nemo3b": (
                "OEA-Nemo3B-AC/step_400_best.pt",
                9466826153,
                "55579dfbd4f6621b5d842c5e731d6a1d37dbfd04b26b1c55bf8cdea980e67d25",
            ),
            "oea_nemo3b_cl": (
                "OEA-Nemo3B-Cl/step_450_best.pt",
                9466826217,
                "c9013285894109487a840a843fe789e8c3ed1c67fc3fbbe9ee7391e57b2ea96d",
            ),
            "oea_qwen3b": (
                "OEA-Qwen3B-AC/step_350.pt",
                9466835918,
                "afb22d02e610016184fa0e2b4314fe0de191ebcc909184e9e128d36abade4b44",
            ),
            "oea_qwen3b_cl": (
                "OEA-Qwen3B-Cl/step_40.pt",
                9466833858,
                "d5f2648c19b07fe5b33c22873098f9b9d749dfe2fb48e5d3dedfcea827c39b5c",
            ),
            "oea_qwen7b": (
                "OEA-Qwen7B-AC/step_300.pt",
                17940602533,
                "cd751e3a71f0b47b9ecbbc0f5a11e4673097f0ebdbd80f610387f5778dd70a46",
            ),
            "oea_qwen7b_cl": (
                "OEA-Qwen7B-Cl/step_330.pt",
                17940602661,
                "09ce41af7e7ac23106fa74d25e45b7229a046d687774cbd4ab2c00c05097d92e",
            ),
        }
        observed = {}
        for variant_id, row in self.registry.items():
            asset = row["checkpoint_asset"]
            observed[variant_id] = (
                f"{asset['local_subdir']}/{asset['source_file']}",
                asset["source_size_bytes"],
                asset["source_sha256"],
            )
        self.assertEqual(observed, expected)

    def test_variant_paths_use_nonofficial_derived_suffix_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, destination = variant_paths(self.registry["oea_qwen7b_cl"], root)
        self.assertEqual(source.name, "step_330.pt")
        self.assertEqual(destination.name, "step_330_inference_only.pt")
        self.assertNotEqual(source, destination)

    def test_structure_expectations_come_from_matching_inspection(self) -> None:
        variant = self.registry["oea_nemo3b"]
        asset = variant["checkpoint_asset"]
        inspection = {
            "status": "complete",
            "checkpoint_size_bytes": asset["source_size_bytes"],
            "checkpoint_sha256": asset["source_sha256"],
            "sections": {
                "lora_state_dict": {
                    "lora_tensor_count": 384,
                    "lora_estimated_tensor_bytes": 123456,
                },
                "audio_head": {"tensor_shapes": {"fc.weight": [512, 2048]}},
                "text_head": {"tensor_shapes": {"fc.weight": [512, 2048]}},
            },
        }
        self.assertEqual(
            structure_expectations(inspection, variant),
            {"expected_lora_tensors": 384, "expected_lora_bytes": 123456},
        )
        inspection["checkpoint_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            structure_expectations(inspection, variant)

    def test_subprocess_commands_pin_identity_and_measured_structure(self) -> None:
        variant = self.registry["oea_qwen3b_cl"]
        source = Path("/models/OEA-Qwen3B-Cl/step_40.pt")
        destination = Path("/models/OEA-Qwen3B-Cl/step_40_inference_only.pt")
        inspection = Path("/logs/inspection.json")
        extraction = Path("/logs/extraction.json")
        inspect_args = inspection_command("python", source, inspection, variant)
        extract_args = extraction_command(
            "python",
            source,
            destination,
            extraction,
            variant,
            {"expected_lora_tensors": 544, "expected_lora_bytes": 50462720},
        )
        self.assertIn(str(variant["checkpoint_asset"]["source_size_bytes"]), inspect_args)
        self.assertIn(variant["checkpoint_asset"]["source_sha256"], inspect_args)
        self.assertIn("544", extract_args)
        self.assertIn("50462720", extract_args)
        self.assertIn(str(destination), extract_args)

    def test_registry_rejects_path_traversal_and_duplicate_variant(self) -> None:
        for unsafe_path in ("../escape.json", "C:/escape.json", "/escape.json"):
            config = json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8"))
            config["variants"][0]["checkpoint_asset"]["manifest"] = unsafe_path
            with self.subTest(path=unsafe_path), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "registry.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "repository-relative"):
                    load_checkpoint_registry(path, REPOSITORY_ROOT)

        config = json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8"))
        config["variants"][1]["variant_id"] = config["variants"][0]["variant_id"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate variant_id"):
                load_checkpoint_registry(path, REPOSITORY_ROOT)

    def test_model_root_inside_repository_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the Git repository"):
            external_model_root(REPOSITORY_ROOT / "models")


if __name__ == "__main__":
    unittest.main()
