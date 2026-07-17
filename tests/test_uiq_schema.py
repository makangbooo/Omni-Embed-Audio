from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from AudioRetrieval.evaluation.uiq_schema import (
    POSITIVE_QUERY_TYPES,
    load_released_uiq,
    parse_released_uiq_row,
    released_uiq_summary,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def positive_row(**overrides):
    row = {
        "audio_id": "clip.wav",
        "dataset": "clotho",
        "dataset_slug": "clotho_evaluation",
        "query_type": "question",
        "generated_query": "Can you find a bell ringing?",
        "original_captions": ["A bell rings."],
        "metadata": {"split": "evaluation"},
        "source_model": "gpt-5.1",
        "regen_model": "gpt-5.1",
    }
    row.update(overrides)
    return row


def negative_row(**overrides):
    row = positive_row(
        query_type="negative",
        negative_query="A bell without speech.",
        negative_captions=["A person speaks."],
    )
    del row["generated_query"]
    row.update(overrides)
    return row


class ReleasedUIQSchemaTest(unittest.TestCase):
    def test_positive_and_negative_query_fields_are_adapted(self) -> None:
        positive = parse_released_uiq_row(positive_row(), row_index=1)
        negative = parse_released_uiq_row(
            negative_row(),
            row_index=2,
        )

        self.assertEqual(positive.query, "Can you find a bell ringing?")
        self.assertEqual(positive.negative_captions, ())
        self.assertEqual(negative.query, "A bell without speech.")
        self.assertEqual(negative.negative_captions, ("A person speaks.",))

    def test_positive_duplicate_audio_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.jsonl"
            rows = [positive_row(), positive_row(generated_query="Find a bell.")]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "duplicate audio IDs"):
                load_released_uiq(path)

    def test_negative_duplicate_audio_ids_are_retained(self) -> None:
        row = negative_row()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "negative.jsonl"
            path.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n",
                encoding="utf-8",
            )
            loaded = load_released_uiq(path)

        self.assertEqual(len(loaded), 2)
        self.assertEqual(released_uiq_summary(loaded)["unique_audio_ids"], 1)
        self.assertFalse(
            released_uiq_summary(loaded)["hard_negative_audio_id_available"]
        )

    def test_repository_release_inventory_matches_published_files(self) -> None:
        expected_counts = {
            "audiocaps": {"positive": 975, "negative": 630},
            "clotho": {"positive": 1045, "negative": 542},
            "mecat": {"positive": 848, "negative": 409},
        }
        total_rows = 0
        for dataset, counts in expected_counts.items():
            paths = sorted((REPOSITORY_ROOT / "data" / "UIQ" / dataset).glob("*.jsonl"))
            self.assertEqual(len(paths), 5)
            for path in paths:
                query_type = next(
                    query_type
                    for query_type in (*sorted(POSITIVE_QUERY_TYPES), "negative")
                    if f"_{query_type}_queries" in path.name
                )
                rows = load_released_uiq(
                    path,
                    expected_dataset=dataset,
                    expected_query_type=query_type,
                )
                expected = (
                    counts["negative"]
                    if query_type == "negative"
                    else counts["positive"]
                )
                self.assertEqual(len(rows), expected)
                total_rows += len(rows)
        self.assertEqual(total_rows, 13053)

    def test_missing_required_field_is_rejected(self) -> None:
        row = positive_row()
        del row["audio_id"]
        with self.assertRaisesRegex(ValueError, "audio_id"):
            parse_released_uiq_row(row, row_index=7)


if __name__ == "__main__":
    unittest.main()
