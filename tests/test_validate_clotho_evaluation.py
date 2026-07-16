from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_clotho_evaluation import (
    CAPTION_COLUMNS,
    METADATA_COLUMNS,
    discover_audio,
    validate_text_and_uiq,
)


class ValidateClothoEvaluationTest(unittest.TestCase):
    def create_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        names = ["First.wav", "Second.wav"]
        captions = root / "captions.csv"
        with captions.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CAPTION_COLUMNS)
            writer.writeheader()
            for name in names:
                writer.writerow(
                    {"file_name": name, **{f"caption_{i}": f"caption {i}" for i in range(1, 6)}}
                )

        metadata = root / "metadata.csv"
        with metadata.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS)
            writer.writeheader()
            for index, name in enumerate(names):
                writer.writerow(
                    {
                        "file_name": name,
                        "keywords": "test",
                        "sound_id": str(index),
                        "sound_link": "https://example.invalid",
                        "start_end_samples": "",
                        "manufacturer": "test",
                        "license": "test",
                    }
                )

        uiq = root / "uiq"
        uiq.mkdir()
        for query_type in ("question", "imperative", "paraphrase", "tagging"):
            rows = [
                {
                    "audio_id": name,
                    "dataset": "clotho",
                    "dataset_slug": "clotho_evaluation",
                    "query_type": query_type,
                    "generated_query": f"query for {name}",
                }
                for name in names
            ]
            path = uiq / f"clotho_evaluation_{query_type}_queries.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
        negative = {
            "audio_id": "First",
            "dataset": "clotho",
            "dataset_slug": "clotho_evaluation",
            "query_type": "negative",
            "negative_query": "without a second sound",
        }
        (uiq / "clotho_evaluation_negative_queries.jsonl").write_text(
            json.dumps(negative) + "\n", encoding="utf-8"
        )
        return captions, metadata, uiq

    def test_text_and_uiq_alignment_including_suffix_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            captions, metadata, uiq = self.create_fixture(Path(directory))
            captions_by_name, report = validate_text_and_uiq(
                captions, metadata, uiq, expected_examples=2, expected_negative_rows=1
            )
        self.assertEqual(set(captions_by_name), {"First.wav", "Second.wav"})
        self.assertTrue(report["question"]["exact_audio_id_set_match"])
        self.assertEqual(report["negative"]["raw_exact_matches"], 0)
        self.assertEqual(report["negative"]["inferred_append_wav_matches"], 1)

    def test_positive_uiq_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            captions, metadata, uiq = self.create_fixture(Path(directory))
            path = uiq / "clotho_evaluation_question_queries.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["audio_id"] = "Missing.wav"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "positive UIQ audio_id mismatch"):
                validate_text_and_uiq(
                    captions, metadata, uiq, expected_examples=2, expected_negative_rows=1
                )

    def test_duplicate_audio_basenames_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").mkdir()
            (root / "b").mkdir()
            (root / "a" / "same.wav").write_bytes(b"a")
            (root / "b" / "same.wav").write_bytes(b"b")
            with self.assertRaisesRegex(ValueError, "duplicate WAV basenames"):
                discover_audio(root)


if __name__ == "__main__":
    unittest.main()
