from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.download_http_assets import (
    REPOSITORY_ROOT,
    ensure_external_data_root,
    hash_file,
    validate_specification,
)


class DownloadHttpAssetsTest(unittest.TestCase):
    def test_data_root_inside_repository_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the Git repository"):
            ensure_external_data_root(REPOSITORY_ROOT / "datasets")

    def test_hash_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload"
            path.write_bytes(b"clotho")
            self.assertEqual(
                hash_file(path), hashlib.md5(b"clotho", usedforsecurity=False).hexdigest()
            )
            self.assertEqual(hash_file(path, "sha256"), hashlib.sha256(b"clotho").hexdigest())

    def test_manifest_has_exact_official_evaluation_assets(self) -> None:
        manifest = REPOSITORY_ROOT / "configs/resources/data01_clotho_evaluation.json"
        specification = json.loads(manifest.read_text(encoding="utf-8"))
        validate_specification(specification)
        self.assertEqual(specification["version"], "2.1")
        self.assertEqual(specification["split"], "evaluation")
        self.assertEqual(specification["local_subdir"], "clotho_v2.1/source")
        self.assertEqual(specification["expected_examples"], 1045)
        self.assertEqual(
            {asset["name"]: asset["md5"] for asset in specification["files"]},
            {
                "clotho_audio_evaluation.7z": "4569624ccadf96223f19cb59fe4f849f",
                "clotho_captions_evaluation.csv": "1b16b9e57cf7bdb7f13a13802aeb57e2",
                "clotho_metadata_evaluation.csv": "13946f054d4e1bf48079813aac61bf77",
            },
        )

    def test_trainval_manifest_has_exact_official_assets(self) -> None:
        manifest = REPOSITORY_ROOT / "configs/resources/data03_clotho_trainval.json"
        specification = json.loads(manifest.read_text(encoding="utf-8"))
        validate_specification(specification)
        self.assertEqual(specification["version"], "2.1")
        self.assertEqual(specification["split"], "development+validation")
        self.assertEqual(specification["local_subdir"], "clotho_v2.1/source")
        self.assertEqual(specification["expected_examples"], 4884)
        self.assertEqual(
            {asset["name"]: asset["md5"] for asset in specification["files"]},
            {
                "clotho_audio_development.7z": "c8b05bc7acdb13895bb3c6a29608667e",
                "clotho_audio_validation.7z": "7dba730be08bada48bd15dc4e668df59",
                "clotho_captions_development.csv": "d4090b39ce9f2491908eebf4d5b09bae",
                "clotho_captions_validation.csv": "5879e023032b22a2c930aaa0528bead4",
                "clotho_metadata_development.csv": "170d20935ecfdf161ce1bb154118cda5",
                "clotho_metadata_validation.csv": "2e010427c56b1ce6008b0f03f41048ce",
            },
        )
        self.assertEqual(
            {asset["name"]: asset["size_bytes"] for asset in specification["files"]},
            {
                "clotho_audio_development.7z": 4541582263,
                "clotho_audio_validation.7z": 1260701425,
                "clotho_captions_development.csv": 1336762,
                "clotho_captions_validation.csv": 367649,
                "clotho_metadata_development.csv": 830797,
                "clotho_metadata_validation.csv": 224803,
            },
        )
        self.assertEqual(
            {
                asset["name"]: asset.get("sha256")
                for asset in specification["files"]
                if asset["kind"] != "audio_archive"
            },
            {
                "clotho_captions_development.csv": "df2e5b92060b4bb23311f8b3a7f82241d900b9c4f62b0cc467ac2ce5e9c52886",
                "clotho_captions_validation.csv": "fb0365506fe2dfcba9b7299daf7623a795abbd6ab9997a88ab0308e2fdfdbb88",
                "clotho_metadata_development.csv": "b054a8d9d0f88436e7cf6341c82a70e9e975c3b3ddd560d9b04f2cd5fdc75949",
                "clotho_metadata_validation.csv": "066026ae1bc20277614ae9d4fffea085d959f0b5e40120751b8d3e717f5faa97",
            },
        )

    def test_manifest_rejects_path_traversal(self) -> None:
        specification = {
            "schema_version": 1,
            "local_subdir": "clotho_v2.1/source",
            "files": [
                {
                    "name": "../escape",
                    "url": "https://example.invalid/file",
                    "md5": "0" * 32,
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "unsafe"):
            validate_specification(specification)

    def test_optional_sha256_and_size_are_validated(self) -> None:
        specification = {
            "schema_version": 1,
            "local_subdir": "audiocaps_v2/source",
            "files": [
                {
                    "name": "train.csv",
                    "url": "https://example.invalid/train.csv",
                    "md5": "0" * 32,
                    "sha256": "a" * 64,
                    "size_bytes": 123,
                }
            ],
        }
        validate_specification(specification)
        specification["files"][0]["sha256"] = "short"
        with self.assertRaisesRegex(ValueError, "invalid SHA256"):
            validate_specification(specification)


if __name__ == "__main__":
    unittest.main()
