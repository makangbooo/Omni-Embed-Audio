from __future__ import annotations

import hashlib
import json
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.audit_squtr_archive import (
    audit_zip_structure,
    normalized_member_path,
)


def structure_manifest() -> dict:
    return {
        "dataset": "SQuTR-fixture",
        "official_code_repository": "https://example.invalid/code",
        "official_code_revision": "a" * 40,
        "source_tags": {},
        "conditions": [
            {
                "id": "clean",
                "audio_directory": "audio_clean",
                "documented_query_file": "queries_with_audio_clean.jsonl",
            },
            {
                "id": "snr_0",
                "audio_directory": "audio_noise_snr_0",
                "documented_query_file": "queries_with_audio_noise_snr_0.jsonl",
            },
        ],
        "core_relative_paths": [
            "corpus.jsonl",
            "queries.jsonl",
            "qrels/test.jsonl",
        ],
        "subsets": [
            {
                "language": "en",
                "name": "one",
                "relative_path": "en/one",
                "expected_unique_queries": 1,
            },
            {
                "language": "zh",
                "name": "two",
                "relative_path": "zh/two",
                "expected_unique_queries": 1,
            },
        ],
        "expected_audio_instances": 4,
    }


def write_fixture(path: Path, *, omit: str | None = None) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for subset in ("en/one", "zh/two"):
            members = {
                f"source_data/{subset}/corpus.jsonl": "{}\n",
                f"source_data/{subset}/queries.jsonl": "{}\n",
                f"source_data/{subset}/qrels/test.jsonl": "{}\n",
                f"source_data/{subset}/queries_with_audio_clean.jsonl": "{}\n",
                f"source_data/{subset}/queries_with_audio_noise_snr_0.jsonl": "{}\n",
                f"source_data/{subset}/audio_clean/1.wav": "wav",
                f"source_data/{subset}/audio_noise_snr_0/noise_1.wav": "wav",
            }
            for name, value in members.items():
                if name != omit:
                    archive.writestr(name, value)


def resource_manifest(path: Path) -> dict:
    payload = path.read_bytes()
    return {
        "files": [
            {
                "name": path.name,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ]
    }


class AuditSqutrArchiveTest(unittest.TestCase):
    def test_safe_fixture_resolves_root_and_counts_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "source_data.zip"
            write_fixture(archive)
            report = audit_zip_structure(
                archive,
                resource_manifest(archive),
                structure_manifest(),
            )
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["archive_root"], "source_data")
        self.assertEqual(report["condition_wav_files"], 4)
        self.assertEqual(report["violations"], [])
        self.assertFalse(report["extraction_performed"])

    def test_missing_documented_path_is_reported_without_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "source_data.zip"
            missing = "source_data/en/one/queries_with_audio_clean.jsonl"
            write_fixture(archive, omit=missing)
            report = audit_zip_structure(
                archive,
                resource_manifest(archive),
                structure_manifest(),
            )
        self.assertEqual(report["status"], "failed")
        self.assertTrue(any(missing in item for item in report["violations"]))
        self.assertFalse(report["extraction_performed"])

    def test_path_traversal_and_absolute_paths_are_rejected(self) -> None:
        for path in ("../escape.wav", "/absolute.wav", "C:/absolute.wav"):
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "ZIP member path"):
                    normalized_member_path(path)

    def test_symlink_member_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "source_data.zip"
            write_fixture(archive)
            link = zipfile.ZipInfo("source_data/en/one/link")
            link.create_system = 3
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive, "a") as handle:
                handle.writestr(link, "target")
            with self.assertRaisesRegex(ValueError, "symbolic-link"):
                audit_zip_structure(
                    archive,
                    resource_manifest(archive),
                    structure_manifest(),
                )

    def test_official_structure_manifest_is_exactly_pinned(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "configs/resources/data13_squtr_structure.json"
        )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["official_code_revision"],
            "cc3fb31fc0dc44fef3a44c569b344516bbaee79c",
        )
        self.assertEqual(manifest["expected_unique_queries"], 37317)
        self.assertEqual(manifest["expected_audio_instances"], 149268)
        self.assertEqual(
            [item["relative_path"] for item in manifest["subsets"]],
            [
                "en/fiqa",
                "en/hotpotqa",
                "en/nq",
                "zh/DuRetrieval",
                "zh/MedicalRetrieval",
                "zh/T2Retrieval",
            ],
        )


if __name__ == "__main__":
    unittest.main()
