from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.extract_squtr_archive import (
    safe_extract,
    validate_manifests,
    verify_file_against_member,
)


def write_archive(path: Path) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("source_data/en/one/corpus.jsonl", '{"_id":"d1"}\n')
        archive.writestr("source_data/en/one/audio_clean/1.wav", b"RIFF-fixture")
    with zipfile.ZipFile(path) as archive:
        return archive.infolist()


def resource_manifest(path: Path) -> dict:
    payload = path.read_bytes()
    return {
        "files": [
            {
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ]
    }


def structure_manifest(infos: list[zipfile.ZipInfo]) -> dict:
    return {
        "observed_archive_root": "source_data",
        "observed_zip_member_records": len(infos),
        "observed_file_members": sum(not info.is_dir() for info in infos),
        "observed_directory_members": sum(info.is_dir() for info in infos),
        "observed_uncompressed_member_bytes": sum(
            info.file_size for info in infos if not info.is_dir()
        ),
    }


class ExtractSqutrArchiveTest(unittest.TestCase):
    def test_extract_then_verify_existing_is_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source_data.zip"
            infos = write_archive(archive)
            structure = structure_manifest(infos)
            extract_root = root / "extracted"

            first = safe_extract(archive, extract_root, structure, 1)
            second = safe_extract(archive, extract_root, structure, 1)

            self.assertEqual(first["extracted_files"], 2)
            self.assertEqual(first["verified_existing_files"], 0)
            self.assertEqual(second["extracted_files"], 0)
            self.assertEqual(second["verified_existing_files"], 2)
            self.assertTrue(first["full_member_crc_verified"])
            self.assertEqual(
                (
                    extract_root / "source_data/en/one/audio_clean/1.wav"
                ).read_bytes(),
                b"RIFF-fixture",
            )

    def test_mismatched_final_file_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source_data.zip"
            infos = write_archive(archive)
            structure = structure_manifest(infos)
            extract_root = root / "extracted"
            safe_extract(archive, extract_root, structure, 10)
            target = extract_root / "source_data/en/one/corpus.jsonl"
            target.write_bytes(b"x" * target.stat().st_size)

            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                safe_extract(archive, extract_root, structure, 10)
            self.assertEqual(target.read_bytes(), b"x" * target.stat().st_size)

    def test_tool_owned_partial_is_restarted_and_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source_data.zip"
            infos = write_archive(archive)
            structure = structure_manifest(infos)
            extract_root = root / "extracted"
            partial = (
                extract_root
                / "source_data/en/one/corpus.jsonl.data13b.part"
            )
            partial.parent.mkdir(parents=True)
            partial.write_text("interrupted", encoding="utf-8")

            report = safe_extract(archive, extract_root, structure, 10)

            self.assertEqual(report["restarted_partial_files"], 1)
            self.assertFalse(partial.exists())
            self.assertEqual(
                (extract_root / "source_data/en/one/corpus.jsonl").read_text(
                    encoding="utf-8"
                ),
                '{"_id":"d1"}\n',
            )

    def test_regular_file_in_directory_chain_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source_data.zip"
            infos = write_archive(archive)
            structure = structure_manifest(infos)
            extract_root = root / "extracted"
            extract_root.mkdir()
            (extract_root / "source_data").write_text("not a directory")

            with self.assertRaisesRegex(ValueError, "unsafe extraction directory"):
                safe_extract(archive, extract_root, structure, 10)

    def test_archive_identity_is_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source_data.zip"
            infos = write_archive(archive)
            structure = structure_manifest(infos)
            archive_root, size, checksum = validate_manifests(
                archive,
                resource_manifest(archive),
                structure,
            )
            self.assertEqual(archive_root, "source_data")
            self.assertEqual(size, archive.stat().st_size)
            self.assertEqual(checksum, hashlib.sha256(archive.read_bytes()).hexdigest())

            bad = json.loads(json.dumps(resource_manifest(archive)))
            bad["files"][0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                validate_manifests(archive, bad, structure)

    def test_member_crc_verification_detects_same_size_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source_data.zip"
            infos = write_archive(archive)
            corpus_info = next(
                info for info in infos if info.filename.endswith("corpus.jsonl")
            )
            existing = root / "same-size.jsonl"
            existing.write_bytes(b"x" * corpus_info.file_size)
            with self.assertRaisesRegex(ValueError, "CRC mismatch"):
                verify_file_against_member(existing, corpus_info)


if __name__ == "__main__":
    unittest.main()
