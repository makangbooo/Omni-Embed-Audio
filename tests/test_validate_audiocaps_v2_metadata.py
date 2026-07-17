from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_audiocaps_v2_metadata import (
    REPOSITORY_ROOT,
    build_manifest_rows,
    normalize_start_time,
    order_preserving_casefold_unique,
    parse_source_csv,
    validate_uiq,
)


class ValidateAudioCapsV2MetadataTest(unittest.TestCase):
    def test_bare_cr_fragment_is_audited_and_optionally_repaired(self) -> None:
        payload = (
            b"audiocap_id,youtube_id,start_time,caption\n"
            b"1,video-a,20,first part\rorphan tail\n"
            b'2,video-b,30,"quoted\r\nmultiline"\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            path.write_bytes(payload)
            source = parse_source_csv(path, "train")
            public = build_manifest_rows(source, repair_bare_cr=False)
            repaired = build_manifest_rows(source, repair_bare_cr=True)

        self.assertEqual(source["parsed_rows"], 3)
        self.assertEqual(source["valid_rows"], 2)
        self.assertEqual(source["bare_cr_count"], 1)
        self.assertEqual(len(source["multiline_rows"]), 1)
        self.assertEqual(
            source["malformed_rows"][0]["orphan_fragment"], "orphan tail"
        )
        self.assertEqual(public[0]["captions"], ["first part"])
        self.assertEqual(repaired[0]["captions"], ["first part orphan tail"])
        self.assertEqual(
            public[0]["caption_integrity_status"],
            ["TRUNCATED_BY_UNQUOTED_BARE_CR_IN_SOURCE"],
        )

    def test_start_time_normalization_is_decimal_not_float_based(self) -> None:
        self.assertEqual(normalize_start_time("20.0"), "20")
        self.assertEqual(normalize_start_time("0.1250"), "0.125")
        with self.assertRaisesRegex(ValueError, "invalid AudioCaps start_time"):
            normalize_start_time("NaN")

    def test_order_preserving_caption_deduplication(self) -> None:
        self.assertEqual(
            order_preserving_casefold_unique(["a", "B", "A", "c", "b"]),
            ["a", "B", "c"],
        )

    def create_uiq_fixture(self, root: Path) -> dict[str, object]:
        groups = {
            "video-a_20": [
                {"caption": "caption one"},
                {"caption": "caption one"},
                {"caption": "caption two"},
            ],
            "video-b_30": [
                {"caption": "caption three"},
            ],
        }
        source: dict[str, object] = {"groups": groups}
        root.mkdir()
        for query_type in ("question", "imperative", "paraphrase", "tagging"):
            rows = [
                {
                    "audio_id": audio_id,
                    "dataset": "audiocaps",
                    "dataset_slug": "audiocaps_test",
                    "query_type": query_type,
                    "generated_query": f"query {audio_id}",
                    "original_captions": order_preserving_casefold_unique(
                        [record["caption"] for record in records]
                    ),
                }
                for audio_id, records in groups.items()
            ]
            (root / f"audiocaps_test_{query_type}_queries.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
        negative = {
            "audio_id": "video-b_30",
            "dataset": "audiocaps",
            "dataset_slug": "audiocaps_test",
            "query_type": "negative",
            "negative_query": "without a distractor",
            "original_captions": ["caption three"],
        }
        (root / "audiocaps_test_negative_queries.jsonl").write_text(
            json.dumps(negative) + "\n", encoding="utf-8"
        )
        return source

    def test_uiq_alignment_uses_deduplicated_positive_captions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            uiq_root = Path(directory) / "uiq"
            source = self.create_uiq_fixture(uiq_root)
            report = validate_uiq(
                uiq_root,
                source,
                expected_positive_rows=2,
                expected_negative_rows=1,
                expected_negative_unique_ids=1,
            )
        self.assertTrue(report["question"]["exact_csv_id_set_match"])
        self.assertTrue(report["negative"]["original_captions_exact_match"])

    def test_resource_manifest_pins_official_commit_and_hashes(self) -> None:
        manifest = json.loads(
            (
                REPOSITORY_ROOT
                / "configs/resources/data06_audiocaps_v2_metadata.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["resource_id"], "DATA-06")
        self.assertIn("d004db3ea1b01cf4fd0347dd8d27db90cadc8809", manifest["source_record"])
        files = {item["name"]: item for item in manifest["files"]}
        self.assertEqual(files["train.csv"]["size_bytes"], 6311901)
        self.assertEqual(
            files["train.csv"]["sha256"],
            "25659eee0ff887972b6a8a74008dfffe582343930e89db71519d66e9dd014f6e",
        )


if __name__ == "__main__":
    unittest.main()
