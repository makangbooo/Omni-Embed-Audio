from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_wavcaps_metadata import (
    REPOSITORY_ROOT,
    derive_overlaps,
    duration_policy_counts,
    manifest_rows,
    normalize_wavcaps_audioset_id,
    write_jsonl_without_overwriting_mismatch,
)


class ValidateWavCapsMetadataTest(unittest.TestCase):
    def sources(self) -> dict[str, list[dict[str, object]]]:
        return {
            "AudioSet_SL": [
                {
                    "id": "Yabcdefghijk.wav",
                    "caption": "test audio",
                    "duration": 10.0,
                },
                {"id": "Y12345678901.wav", "caption": "zero", "duration": 0.0},
            ],
            "BBC_Sound_Effects": [
                {"id": "bbc-1", "caption": "exactly 31", "duration": 31.0}
            ],
            "FreeSound": [
                {
                    "id": "10",
                    "file_name": "Same.wav",
                    "caption": "first",
                    "duration": 4.0,
                },
                {
                    "id": "11",
                    "file_name": "same.WAV",
                    "caption": "second",
                    "duration": 40.0,
                },
            ],
            "SoundBible": [
                {"id": "sb-1", "caption": "short", "duration": 30.999}
            ],
        }

    def test_duration_policy_distinguishes_written_and_count_equivalent_rules(self) -> None:
        counts = duration_policy_counts(self.sources())
        self.assertEqual(counts["all_rows"], 6)
        self.assertEqual(counts["duration_lt_31"], 4)
        self.assertEqual(counts["duration_le_31"], 5)
        self.assertEqual(counts["positive_duration_lt_31"], 3)
        self.assertEqual(counts["positive_duration_le_31"], 4)
        self.assertEqual(counts["non_positive_duration"], 1)
        self.assertEqual(counts["duration_equal_31"], 1)

    def test_overlap_derivation_preserves_ambiguous_filename_candidates(self) -> None:
        sources = self.sources()
        audiocaps = [{"youtube_id": "abcdefghijk"}]
        clotho = [{"file_name": "SAME.wav", "sound_id": "10"}]
        overlaps = derive_overlaps(sources, audiocaps, clotho)

        self.assertEqual(overlaps["audiocaps_overlap_ids"], ["abcdefghijk"])
        self.assertEqual(overlaps["audiocaps_blocked_ids"], {"Yabcdefghijk.wav"})
        self.assertEqual(overlaps["clotho_candidate_ids"], {"10", "11"})
        self.assertEqual(overlaps["clotho_confirmed_ids"], {"10"})
        self.assertEqual(overlaps["clotho_ambiguous_filenames"], 1)
        self.assertEqual(
            overlaps["clotho_matches"][0]["wavcaps_candidate_ids"], ["10", "11"]
        )

        rows = list(
            manifest_rows(
                sources,
                audiocaps_blocked_ids=overlaps["audiocaps_blocked_ids"],
                clotho_candidate_ids=overlaps["clotho_candidate_ids"],
            )
        )
        by_sample = {row["sample_id"]: row for row in rows}
        self.assertFalse(by_sample["AudioSet_SL:Yabcdefghijk"]["include_in_training"])
        self.assertIn(
            "AUDIOCAPS_TEST_OVERLAP_EXACT_YOUTUBE_ID",
            by_sample["AudioSet_SL:Yabcdefghijk"]["filter_reasons"],
        )
        self.assertFalse(by_sample["FreeSound:10"]["include_in_training"])
        self.assertIn(
            "CLOTHO_EVALUATION_FILENAME_MATCH_CONSERVATIVE",
            by_sample["FreeSound:10"]["filter_reasons"],
        )
        self.assertTrue(by_sample["SoundBible:sb-1"]["include_in_training"])

    def test_audioset_id_normalization_is_strict(self) -> None:
        self.assertEqual(
            normalize_wavcaps_audioset_id("Yabcdefghijk.wav"), "abcdefghijk"
        )
        with self.assertRaisesRegex(ValueError, "unexpected WavCaps AudioSet_SL ID"):
            normalize_wavcaps_audioset_id("abcdefghijk.wav")

    def test_generated_jsonl_reuses_identical_and_refuses_different_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.jsonl"
            created = write_jsonl_without_overwriting_mismatch(path, [{"id": 1}])
            reused = write_jsonl_without_overwriting_mismatch(path, [{"id": 1}])
            self.assertEqual(created["status"], "created")
            self.assertEqual(reused["status"], "verified_existing")
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                write_jsonl_without_overwriting_mismatch(path, [{"id": 2}])

    def test_resource_manifest_excludes_audio_archives_and_pins_all_files(self) -> None:
        path = REPOSITORY_ROOT / "configs/resources/data08_wavcaps_metadata.json"
        specification = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(specification["resource_id"], "DATA-08")
        asset = specification["assets"][0]
        self.assertEqual(
            asset["revision"], "0930ec11ded28fa0eaa910fde2f6fc3538acbeac"
        )
        self.assertEqual(len(asset["expected_files"]), 8)
        self.assertEqual(set(asset["allow_patterns"]), set(asset["required_files"]))
        self.assertTrue(
            all(not item.startswith("Zip_files/") for item in asset["allow_patterns"])
        )
        self.assertEqual(
            sum(item["size_bytes"] for item in asset["expected_files"]),
            specification["expected_download_bytes"],
        )


if __name__ == "__main__":
    unittest.main()
