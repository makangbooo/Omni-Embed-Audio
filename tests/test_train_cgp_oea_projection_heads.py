import unittest

from scripts.train_cgp_oea_projection_heads import make_unique_batches


class TrainCGPOEATest(unittest.TestCase):
    def test_batches_do_not_contain_duplicate_clips(self) -> None:
        entries = [
            {"clip_id": "a", "caption": "a1"},
            {"clip_id": "a", "caption": "a2"},
            {"clip_id": "b", "caption": "b1"},
            {"clip_id": "b", "caption": "b2"},
            {"clip_id": "c", "caption": "c1"},
        ]
        batches = make_unique_batches(entries, 2, seed=7)
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(5)))
        for batch in batches:
            self.assertEqual(
                len({entries[index]["clip_id"] for index in batch}), len(batch)
            )

    def test_batches_are_deterministic_for_seed(self) -> None:
        entries = [{"clip_id": str(index // 2)} for index in range(8)]
        self.assertEqual(
            make_unique_batches(entries, 3, seed=11),
            make_unique_batches(entries, 3, seed=11),
        )


if __name__ == "__main__":
    unittest.main()
