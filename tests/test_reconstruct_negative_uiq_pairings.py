from __future__ import annotations

import unittest

from scripts.reconstruct_negative_uiq_pairings import reconstruct_pairings


class ReconstructNegativeUIQPairingsTest(unittest.TestCase):
    def test_exact_caption_pairing_and_stem_target_mapping(self) -> None:
        positive = [
            {"audio_id": "target.wav", "original_captions": ["Target sound."]},
            {"audio_id": "negative.wav", "original_captions": ["Other sound."]},
        ]
        negative = [
            {
                "_release_line_number": 7,
                "audio_id": "target",
                "negative_query": "Target sound without the other sound.",
                "negative_captions": ["Other sound."],
            }
        ]

        report = reconstruct_pairings(negative, positive, dataset="clotho")

        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["full_coverage"])
        self.assertEqual(report["pairings"][0]["target_id"], "target.wav")
        self.assertEqual(
            report["pairings"][0]["hard_negative_id"], "negative.wav"
        )
        self.assertEqual(report["queries"][0]["release_line_number"], 7)

    def test_ambiguous_caption_pairing_fails_closed(self) -> None:
        positive = [
            {"audio_id": "target", "original_captions": ["Target sound."]},
            {"audio_id": "negative-a", "original_captions": ["Duplicate."]},
            {"audio_id": "negative-b", "original_captions": ["Duplicate."]},
        ]
        negative = [
            {
                "audio_id": "target",
                "negative_query": "Target without duplicate.",
                "negative_captions": ["Duplicate."],
            }
        ]

        report = reconstruct_pairings(negative, positive, dataset="audiocaps")

        self.assertEqual(report["status"], "incomplete")
        self.assertFalse(report["full_coverage"])
        self.assertEqual(report["matched_pairing_count"], 0)
        self.assertIn(
            "exact_hard_negative_candidate_count=2",
            report["failures"][0]["reasons"],
        )

    def test_target_cannot_equal_hard_negative(self) -> None:
        positive = [
            {"audio_id": "same", "original_captions": ["Same sound."]},
        ]
        negative = [
            {
                "audio_id": "same",
                "negative_query": "Same without same.",
                "negative_captions": ["Same sound."],
            }
        ]

        report = reconstruct_pairings(negative, positive, dataset="mecat")

        self.assertEqual(report["status"], "incomplete")
        self.assertIn("target_equals_hard_negative", report["failures"][0]["reasons"])


if __name__ == "__main__":
    unittest.main()
