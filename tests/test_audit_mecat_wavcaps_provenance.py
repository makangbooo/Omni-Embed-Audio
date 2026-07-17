from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_mecat_wavcaps_provenance import (
    derive_source_video_overlaps,
    mecat_youtube_id,
    validate_mecat_manifest,
)


def mecat_row(sample_id: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "audio_relpath": f"00A/test/{sample_id}.flac",
        "split": "test",
        "include_in_training": False,
        "file_exists": True,
        "decode_ok": True,
    }


class AuditMecatWavcapsProvenanceTest(unittest.TestCase):
    def test_strict_mecat_source_id_extraction(self) -> None:
        self.assertEqual(
            mecat_youtube_id("02YV3omdtfs_83_46000000000001_93_46"),
            "02YV3omdtfs",
        )
        for invalid in ("short", "02YV3omdtfs", "02YV3omdtfs_", "bad!id00000_suffix"):
            with self.subTest(sample_id=invalid):
                with self.assertRaisesRegex(ValueError, "sample_id structure"):
                    mecat_youtube_id(invalid)

    def test_source_video_overlap_retains_all_mecat_segments(self) -> None:
        rows = [
            mecat_row("abcdefghijk_0_10"),
            mecat_row("abcdefghijk_10_20"),
            mecat_row("zyxwvutsrqp_0_10"),
        ]
        wavcaps = [
            {"id": "Yabcdefghijk.wav", "caption": "one", "duration": 10},
            {"id": "Y00000000000.wav", "caption": "two", "duration": 10},
        ]
        report = derive_source_video_overlaps(rows, wavcaps)
        self.assertEqual(report["mecat_examples"], 3)
        self.assertEqual(report["mecat_unique_source_videos"], 2)
        self.assertEqual(report["mecat_videos_with_multiple_segments"], 1)
        self.assertEqual(report["source_video_overlap_ids"], ["abcdefghijk"])
        self.assertEqual(report["mecat_overlap_sample_count"], 2)
        self.assertEqual(
            report["overlap_rows"][0]["mecat_sample_ids"],
            ["abcdefghijk_0_10", "abcdefghijk_10_20"],
        )
        self.assertEqual(
            report["overlap_rows"][0]["blocklist_status"],
            "NOT_APPLIED_SOURCE_VIDEO_CANDIDATE_ONLY",
        )

    def test_manifest_count_duplicates_and_alignment_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            row = mecat_row("abcdefghijk_0_10")
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertEqual(validate_mecat_manifest(path, 1), [row])

            path.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate MECAT"):
                validate_mecat_manifest(path, 2)

            invalid = dict(row)
            invalid["audio_relpath"] = "00A/test/different.flac"
            path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "audio_relpath/sample_id"):
                validate_mecat_manifest(path, 1)


if __name__ == "__main__":
    unittest.main()
