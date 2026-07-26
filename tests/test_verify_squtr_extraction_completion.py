from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_squtr_extraction_completion import validate_completion


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def fixture(root: Path) -> tuple[Path, dict, dict]:
    extract_root = root / "extracted"
    structure = {
        "observed_archive_root": "source_data",
        "observed_file_members": 7,
        "observed_uncompressed_member_bytes": 123,
        "core_relative_paths": [
            "corpus.jsonl",
            "queries.jsonl",
            "qrels/test.jsonl",
        ],
        "subsets": [{"relative_path": "en/one"}],
        "conditions": [
            {
                "documented_query_file": "queries_with_audio_clean.jsonl",
                "audio_directory": "audio_clean",
            }
        ],
    }
    resource = {"files": [{"sha256": "a" * 64}]}
    subset = extract_root / "source_data/en/one"
    for relative in (
        "corpus.jsonl",
        "queries.jsonl",
        "qrels/test.jsonl",
        "queries_with_audio_clean.jsonl",
    ):
        path = subset / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    (subset / "audio_clean").mkdir()
    write_json(
        extract_root / ".data13b_extraction_complete.json",
        {
            "archive_sha256": "a" * 64,
            "archive_root": "source_data",
            "expected_file_members": 7,
            "expected_uncompressed_member_bytes": 123,
            "file_members": 7,
            "uncompressed_member_bytes": 123,
            "full_member_crc_verified": True,
        },
    )
    return extract_root, resource, structure


class VerifySqutrExtractionCompletionTest(unittest.TestCase):
    def test_exact_marker_and_core_paths_allow_reuse_without_tree_reread(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            extract_root, resource, structure = fixture(Path(directory))
            report = validate_completion(extract_root, resource, structure)
            self.assertTrue(report["marker_identity_exact"])
            self.assertFalse(report["full_tree_reread_performed"])
            self.assertFalse(report["extraction_performed"])
            self.assertEqual(report["checked_core_paths"], 5)
            self.assertEqual(len(report["marker_sha256"]), 64)

    def test_marker_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            extract_root, resource, structure = fixture(Path(directory))
            resource["files"][0]["sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "completion marker mismatch"):
                validate_completion(extract_root, resource, structure)

    def test_missing_core_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            extract_root, resource, structure = fixture(Path(directory))
            (extract_root / "source_data/en/one/corpus.jsonl").unlink()
            with self.assertRaisesRegex(ValueError, "core file"):
                validate_completion(extract_root, resource, structure)

    def test_recovery_wrapper_is_cpu_only_resumable_and_non_destructive(self) -> None:
        root = Path(__file__).resolve().parents[1]
        text = (
            root / "scripts/run_data13c_squtr_content_recovery.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', text)
        self.assertIn("git status --porcelain --untracked-files=all", text)
        self.assertIn("flock -n 9", text)
        self.assertIn("verify_squtr_extraction_completion.py", text)
        self.assertIn("--probe-cache", text)
        self.assertIn("extraction_reuse_exit_code.txt", text)
        self.assertIn("validation_exit_code.txt", text)
        self.assertIn("wrapper_exit_code.txt", text)
        self.assertNotIn("extract_squtr_archive.py", text)
        self.assertNotIn("rm -", text)
        self.assertNotIn("git reset", text)


if __name__ == "__main__":
    unittest.main()
