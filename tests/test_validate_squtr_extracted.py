from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_squtr_extracted import validate_dataset


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def structure_manifest() -> dict:
    return {
        "observed_archive_root": "source_data",
        "expected_audio_instances": 2,
        "conditions": [
            {
                "id": "clean",
                "audio_directory": "audio_clean",
                "documented_query_file": "queries_with_audio_clean.jsonl",
                "snr_db": None,
            },
            {
                "id": "snr_0",
                "audio_directory": "audio_noise_snr_0",
                "documented_query_file": "queries_with_audio_noise_snr_0.jsonl",
                "snr_db": 0,
            },
        ],
        "subsets": [
            {
                "language": "en",
                "name": "one",
                "relative_path": "en/one",
                "expected_unique_queries": 1,
            }
        ],
    }


def write_extracted_fixture(root: Path, *, corpus_id: str = "d1") -> None:
    subset = root / "source_data/en/one"
    write_jsonl(
        subset / "corpus.jsonl",
        [{"_id": "d1", "title": "Title", "text": "Document"}],
    )
    write_jsonl(subset / "queries.jsonl", [{"_id": "q1", "text": "question"}])
    write_jsonl(
        subset / "qrels/test.jsonl",
        [{"query-id": "q1", "corpus-id": corpus_id, "score": 1}],
    )
    write_jsonl(
        subset / "queries_with_audio_clean.jsonl",
        [{"_id": "q1", "text": "question", "audio": "q1.wav"}],
    )
    write_jsonl(
        subset / "queries_with_audio_noise_snr_0.jsonl",
        [
            {
                "_id": "q1",
                "text": "normalized question",
                "audio": "noise_q1.wav",
                "snr_db": 0,
                "noise_id": "fixture-noise",
            }
        ],
    )
    for path in (
        subset / "audio_clean/q1.wav",
        subset / "audio_noise_snr_0/noise_q1.wav",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-wav")


def fake_probe(path: Path) -> dict:
    if not path.read_bytes():
        raise ValueError("empty fixture")
    return {
        "sample_rate": 16000,
        "channels": 1,
        "frames": 16000,
        "duration_seconds": 1.0,
        "format": "WAV",
        "subtype": "PCM_16",
        "decode_ok": True,
        "decode_scope": "fixture",
    }


class ValidateSqutrExtractedTest(unittest.TestCase):
    def test_schema_id_qrels_and_audio_sets_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extract_root = root / "extracted"
            write_extracted_fixture(extract_root)
            manifest = root / "manifests/squtr.jsonl"

            report, manifest_status, checksum = validate_dataset(
                extract_root,
                structure_manifest(),
                manifest,
                1,
                audio_probe=fake_probe,
            )

            rows = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(manifest_status, "created")
            self.assertEqual(len(checksum), 64)
            self.assertEqual(len(rows), 2)
            self.assertEqual(report["validated_audio_files"], 2)
            self.assertEqual(report["total_duration_seconds"], 2.0)
            self.assertTrue(
                report["subsets"][0]["qrels"]["all_corpus_ids_resolved"]
            )
            self.assertEqual(
                report["subsets"][0]["conditions"]["snr_0"][
                    "query_text_mismatches"
                ],
                1,
            )
            self.assertFalse(rows[1]["query_text_exact_match"])
            self.assertFalse(rows[0]["include_in_training"])

            _, second_status, second_checksum = validate_dataset(
                extract_root,
                structure_manifest(),
                manifest,
                10,
                audio_probe=fake_probe,
            )
            self.assertEqual(second_status, "verified_existing")
            self.assertEqual(second_checksum, checksum)

    def test_unknown_corpus_reference_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extract_root = root / "extracted"
            write_extracted_fixture(extract_root, corpus_id="missing")
            with self.assertRaisesRegex(ValueError, "unknown corpus ID"):
                validate_dataset(
                    extract_root,
                    structure_manifest(),
                    root / "manifest.jsonl",
                    10,
                    audio_probe=fake_probe,
                )

    def test_unsafe_audio_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extract_root = root / "extracted"
            write_extracted_fixture(extract_root)
            metadata = (
                extract_root
                / "source_data/en/one/queries_with_audio_clean.jsonl"
            )
            write_jsonl(
                metadata,
                [{"_id": "q1", "text": "question", "audio": "../q1.wav"}],
            )
            with self.assertRaisesRegex(ValueError, "unsafe audio filename"):
                validate_dataset(
                    extract_root,
                    structure_manifest(),
                    root / "manifest.jsonl",
                    10,
                    audio_probe=fake_probe,
                )

    def test_boolean_qrels_score_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extract_root = root / "extracted"
            write_extracted_fixture(extract_root)
            qrels = extract_root / "source_data/en/one/qrels/test.jsonl"
            write_jsonl(
                qrels,
                [{"query-id": "q1", "corpus-id": "d1", "score": True}],
            )
            with self.assertRaisesRegex(ValueError, "invalid qrels schema"):
                validate_dataset(
                    extract_root,
                    structure_manifest(),
                    root / "manifest.jsonl",
                    10,
                    audio_probe=fake_probe,
                )


if __name__ == "__main__":
    unittest.main()
