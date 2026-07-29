import random
import unittest

from scripts.evaluate_official_source_oea_clotho import select_one_per_clip


class EvaluateOfficialSourceOeaClothoTests(unittest.TestCase):
    def test_seed_zero_selection_matches_public_code_choice_order(self):
        ordered_ids = ["clip_a", "clip_b", "clip_c"]
        clip_ids = [clip_id for clip_id in ordered_ids for _ in range(5)]

        selected = select_one_per_clip(clip_ids, ordered_ids, seed=0)

        rng = random.Random(0)
        expected = sorted(
            rng.choice(list(range(offset, offset + 5)))
            for offset in (0, 5, 10)
        )
        self.assertEqual(selected, expected)

    def test_selection_rejects_unknown_caption_owner(self):
        with self.assertRaisesRegex(ValueError, "absent from audio candidates"):
            select_one_per_clip(["unknown"], ["known"], seed=0)


if __name__ == "__main__":
    unittest.main()
