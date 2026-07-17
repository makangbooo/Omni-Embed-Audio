from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.validate_mecat_00a_test import (
    archive_sample_index,
    extract_regular_members,
    load_json_object,
    safe_relative_member,
    validate_uiq,
)


class ValidateMecat00ATest(unittest.TestCase):
    def create_archive(self, path: Path, sample_ids: list[str]) -> None:
        metadata = {
            "long": ["long caption"],
            "short": ["short caption"],
            "speech": ["None"],
            "music": ["None"],
            "sound": ["sound caption"],
            "environment": ["environment caption"],
        }
        with tarfile.open(path, mode="w:gz") as handle:
            for sample_id in sample_ids:
                for suffix, payload in (
                    (".flac", b"synthetic-flac"),
                    (".json", json.dumps(metadata).encode("utf-8")),
                ):
                    value = tarfile.TarInfo(f"00A/test/{sample_id}{suffix}")
                    value.size = len(payload)
                    handle.addfile(value, io.BytesIO(payload))

    def create_uiq(self, root: Path, sample_ids: list[str]) -> None:
        root.mkdir()
        for query_type in ("question", "imperative", "paraphrase", "tagging"):
            rows = [
                {
                    "audio_id": sample_id,
                    "dataset": "mecat",
                    "dataset_slug": "mecat",
                    "query_type": query_type,
                    "generated_query": f"{query_type} {sample_id}",
                }
                for sample_id in sample_ids
            ]
            (root / f"mecat_{query_type}_queries.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
        negative = {
            "audio_id": sample_ids[0],
            "dataset": "mecat",
            "dataset_slug": "mecat",
            "query_type": "negative",
            "negative_query": "without a target event",
        }
        (root / "mecat_negative_queries.jsonl").write_text(
            json.dumps(negative) + "\n", encoding="utf-8"
        )

    def test_archive_index_and_manual_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "test.tar.gz"
            self.create_archive(archive, ["sample-a", "sample-b"])
            index = archive_sample_index(archive, expected_examples=2)
            destination = root / "extracted"
            extract_regular_members(archive, destination, index)

            self.assertEqual(set(index), {"sample-a", "sample-b"})
            metadata = load_json_object(destination / index["sample-a"]["json"])
            self.assertEqual(metadata["sound"], ["sound caption"])
            self.assertTrue((destination / index["sample-b"]["flac"]).is_file())

    def test_path_traversal_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsafe archive member path"):
            safe_relative_member("../escape.flac")
        with self.assertRaisesRegex(ValueError, "backslash"):
            safe_relative_member("..\\escape.flac")

    def test_non_regular_archive_member_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "unsafe.tar.gz"
            with tarfile.open(archive, mode="w:gz") as handle:
                link = tarfile.TarInfo("sample.flac")
                link.type = tarfile.SYMTYPE
                link.linkname = "/etc/passwd"
                handle.addfile(link)
            with self.assertRaisesRegex(ValueError, "non-regular member"):
                archive_sample_index(archive, expected_examples=1)

    def test_uiq_exact_id_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uiq = root / "uiq"
            self.create_uiq(uiq, ["sample-a", "sample-b"])
            report = validate_uiq(
                uiq,
                {"sample-a", "sample-b"},
                expected_negative_rows=1,
            )
            self.assertTrue(report["question"]["exact_archive_id_set_match"])
            self.assertTrue(report["negative"]["all_audio_ids_in_archive"])

    def test_uiq_unknown_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uiq = root / "uiq"
            self.create_uiq(uiq, ["sample-a", "sample-b"])
            path = uiq / "mecat_question_queries.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["audio_id"] = "unknown"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "positive UIQ ID mismatch"):
                validate_uiq(
                    uiq,
                    {"sample-a", "sample-b"},
                    expected_negative_rows=1,
                )

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            path.write_text(
                '{"long": [], "long": [], "short": [], "speech": [], '
                '"music": [], "sound": [], "environment": []}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                load_json_object(path)


if __name__ == "__main__":
    unittest.main()
