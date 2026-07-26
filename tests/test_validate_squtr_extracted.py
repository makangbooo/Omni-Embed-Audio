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


def write_completion_marker(extract_root: Path) -> None:
    marker = extract_root / ".data13b_extraction_complete.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "archive_sha256": "a" * 64,
                "archive_root": "source_data",
                "expected_file_members": 7,
                "expected_uncompressed_member_bytes": 123,
                "file_members": 7,
                "uncompressed_member_bytes": 123,
                "full_member_crc_verified": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )


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
            self.assertEqual(
                report["subsets"][0]["corpus"]["fully_empty_rows"],
                0,
            )

            _, second_status, second_checksum = validate_dataset(
                extract_root,
                structure_manifest(),
                manifest,
                10,
                audio_probe=fake_probe,
            )
            self.assertEqual(second_status, "verified_existing")
            self.assertEqual(second_checksum, checksum)

    def test_fully_empty_corpus_row_is_preserved_and_audited(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extract_root = root / "extracted"
            write_extracted_fixture(extract_root)
            corpus = extract_root / "source_data/en/one/corpus.jsonl"
            write_jsonl(
                corpus,
                [{"_id": "d1", "title": "", "text": ""}],
            )

            report, manifest_status, _ = validate_dataset(
                extract_root,
                structure_manifest(),
                root / "manifest.jsonl",
                10,
                audio_probe=fake_probe,
            )

            corpus_report = report["subsets"][0]["corpus"]
            self.assertEqual(manifest_status, "created")
            self.assertEqual(corpus_report["rows"], 1)
            self.assertEqual(corpus_report["empty_text_rows"], 1)
            self.assertEqual(corpus_report["fully_empty_rows"], 1)
            self.assertEqual(
                corpus_report["fully_empty_examples"],
                [{"document_id": "d1", "line_number": 1}],
            )
            self.assertEqual(
                len(corpus_report["fully_empty_document_ids_sha256"]),
                64,
            )
            self.assertIn("preserve row", corpus_report["fully_empty_document_policy"])

    def test_probe_cache_resumes_after_an_interrupted_audio_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extract_root = root / "extracted"
            write_extracted_fixture(extract_root)
            write_completion_marker(extract_root)
            probe_cache = root / "manifests/probes.jsonl"
            calls: list[str] = []

            def interrupted_probe(path: Path) -> dict:
                calls.append(path.name)
                if path.name == "noise_q1.wav":
                    raise RuntimeError("simulated interruption")
                return fake_probe(path)

            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                validate_dataset(
                    extract_root,
                    structure_manifest(),
                    root / "manifest.jsonl",
                    1,
                    audio_probe=interrupted_probe,
                    probe_cache=probe_cache,
                    probe_cache_git_commit="fixture-commit",
                )
            self.assertEqual(calls, ["q1.wav", "noise_q1.wav"])
            self.assertTrue(probe_cache.is_file())

            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                validate_dataset(
                    extract_root,
                    structure_manifest(),
                    root / "manifest.jsonl",
                    1,
                    audio_probe=fake_probe,
                    probe_cache=probe_cache,
                    probe_cache_git_commit="different-commit",
                )

            resumed_calls: list[str] = []

            def resumed_probe(path: Path) -> dict:
                resumed_calls.append(path.name)
                return fake_probe(path)

            report, status, _ = validate_dataset(
                extract_root,
                structure_manifest(),
                root / "manifest.jsonl",
                1,
                audio_probe=resumed_probe,
                probe_cache=probe_cache,
                probe_cache_git_commit="fixture-commit",
            )
            self.assertEqual(status, "created")
            self.assertEqual(resumed_calls, ["noise_q1.wav"])
            self.assertEqual(report["probe_cache"]["hits"], 1)
            self.assertEqual(report["probe_cache"]["misses"], 1)
            self.assertEqual(report["probe_cache"]["records_after_run"], 2)

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
            with self.assertRaisesRegex(ValueError, "invalid qrels score"):
                validate_dataset(
                    extract_root,
                    structure_manifest(),
                    root / "manifest.jsonl",
                    10,
                    audio_probe=fake_probe,
                )

    def test_integer_string_qrels_score_matches_pinned_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extract_root = root / "extracted"
            write_extracted_fixture(extract_root)
            qrels = extract_root / "source_data/en/one/qrels/test.jsonl"
            write_jsonl(
                qrels,
                [{"query-id": "q1", "corpus-id": "d1", "score": "1"}],
            )

            report, status, _ = validate_dataset(
                extract_root,
                structure_manifest(),
                root / "manifest.jsonl",
                10,
                audio_probe=fake_probe,
            )

            qrels_report = report["subsets"][0]["qrels"]
            self.assertEqual(status, "created")
            self.assertEqual(qrels_report["score_counts"], {"1": 1})
            self.assertEqual(qrels_report["raw_score_type_counts"], {"str": 1})
            self.assertEqual(qrels_report["integer_string_scores_coerced"], 1)
            self.assertEqual(len(qrels_report["source_sha256"]), 64)
            self.assertEqual(qrels_report["source_size_bytes"], qrels.stat().st_size)

    def test_non_integral_qrels_scores_are_rejected(self) -> None:
        for score in ("1.5", 1.5):
            with self.subTest(score=score), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                extract_root = root / "extracted"
                write_extracted_fixture(extract_root)
                qrels = extract_root / "source_data/en/one/qrels/test.jsonl"
                write_jsonl(
                    qrels,
                    [{"query-id": "q1", "corpus-id": "d1", "score": score}],
                )
                with self.assertRaisesRegex(ValueError, "qrels score"):
                    validate_dataset(
                        extract_root,
                        structure_manifest(),
                        root / "manifest.jsonl",
                        10,
                        audio_probe=fake_probe,
                    )


if __name__ == "__main__":
    unittest.main()
