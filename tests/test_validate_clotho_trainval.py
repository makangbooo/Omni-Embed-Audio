from __future__ import annotations

import csv
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from scripts.validate_clotho_trainval import (
    CAPTION_COLUMNS,
    METADATA_COLUMNS,
    cross_split_audit,
    serialize_jsonl,
    validate_audio_split,
    validate_split_text,
    write_reproducible,
)


class ValidateClothoTrainvalTest(unittest.TestCase):
    @staticmethod
    def decode_fixture(path: Path) -> dict[str, object]:
        with wave.open(str(path), "rb") as audio:
            frames = audio.getnframes()
            sample_rate = audio.getframerate()
            channels = audio.getnchannels()
        return {
            "duration_seconds": frames / sample_rate,
            "sample_rate": sample_rate,
            "channels": channels,
            "frames": frames,
            "format": "WAV",
            "subtype": "PCM_16",
            "decode_ok": True,
            "all_samples_finite": True,
        }

    def write_split(
        self,
        root: Path,
        split: str,
        rows: list[tuple[str, str]],
    ) -> tuple[Path, Path, Path]:
        captions = root / f"captions_{split}.csv"
        with captions.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CAPTION_COLUMNS)
            writer.writeheader()
            for name, _sound_id in rows:
                writer.writerow(
                    {
                        "file_name": name,
                        **{
                            f"caption_{index}": f"{split} caption {index} for {name}"
                            for index in range(1, 6)
                        },
                    }
                )

        metadata = root / f"metadata_{split}.csv"
        with metadata.open("w", encoding="iso-8859-1", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS)
            writer.writeheader()
            for name, sound_id in rows:
                writer.writerow(
                    {
                        "file_name": name,
                        "keywords": "test",
                        "sound_id": sound_id,
                        "sound_link": "https://example.invalid",
                        "start_end_samples": "",
                        "manufacturer": "Oscar de Ávila",
                        "license": "test",
                    }
                )

        audio_root = root / f"audio_{split}"
        audio_root.mkdir()
        for name, _sound_id in rows:
            with wave.open(str(audio_root / name), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(8000)
                audio.writeframes(b"\x00\x00" * 80)
        return captions, metadata, audio_root

    def test_strict_latin1_metadata_and_cross_split_overlap_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dev_csv, dev_meta, dev_audio = self.write_split(
                root, "development", [("Shared.wav", "100"), ("Dev.wav", "200")]
            )
            val_csv, val_meta, val_audio = self.write_split(
                root, "validation", [("Shared.wav", "100"), ("Val.wav", "300")]
            )
            dev_captions, dev_metadata, dev_text = validate_split_text(
                "development", dev_csv, dev_meta, expected_examples=2
            )
            val_captions, val_metadata, _val_text = validate_split_text(
                "validation", val_csv, val_meta, expected_examples=2
            )
            with patch(
                "scripts.validate_clotho_trainval.decode_audio",
                side_effect=self.decode_fixture,
            ):
                dev_rows, _ = validate_audio_split(
                    "development", dev_audio, dev_captions, dev_metadata, True
                )
                val_rows, _ = validate_audio_split(
                    "validation", val_audio, val_captions, val_metadata, False
                )
            cross = cross_split_audit(dev_rows, val_rows)

        self.assertFalse(dev_text["metadata_encoding"]["utf8_decodable"])
        self.assertEqual(
            dev_text["metadata_encoding"]["non_ascii_codepoint_counts"],
            {"U+00C1": 2},
        )
        self.assertEqual(cross["file_name_overlap_count"], 1)
        self.assertEqual(cross["valid_sound_id_overlap_count"], 1)
        self.assertTrue(cross["file_name_overlaps"][0]["exact_audio_bytes_match"])
        self.assertFalse(cross["file_name_overlaps"][0]["caption_lists_match"])
        self.assertTrue(all(row["include_in_training"] for row in dev_rows))
        self.assertTrue(all(not row["include_in_training"] for row in val_rows))

    def test_audio_set_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            captions_csv, metadata_csv, audio_root = self.write_split(
                root, "development", [("One.wav", "1"), ("Two.wav", "2")]
            )
            (audio_root / "Two.wav").unlink()
            captions, metadata, _ = validate_split_text(
                "development", captions_csv, metadata_csv, expected_examples=2
            )
            with self.assertRaisesRegex(ValueError, "audio/caption mismatch"):
                validate_audio_split(
                    "development", audio_root, captions, metadata, True
                )

    def test_canonical_manifest_is_reused_and_drift_is_rejected(self) -> None:
        rows = [{"sample_id": "one", "captions": ["caption"]}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            content = serialize_jsonl(rows)
            self.assertEqual(write_reproducible(path, content), "created")
            self.assertEqual(write_reproducible(path, content), "verified_existing")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                write_reproducible(path, serialize_jsonl([{"sample_id": "two"}]))


if __name__ == "__main__":
    unittest.main()
