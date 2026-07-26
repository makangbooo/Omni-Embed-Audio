from __future__ import annotations

import unittest

from AudioRetrieval.asr_uncertainty_reranking.data import squtr_subset_name


class AuditAsrurFiqaDataTest(unittest.TestCase):
    def test_subset_name_accepts_manifest_relative_paths(self) -> None:
        self.assertEqual(squtr_subset_name("en/fiqa"), "fiqa")
        self.assertEqual(squtr_subset_name("fiqa"), "fiqa")
        self.assertEqual(squtr_subset_name(r"en\FiQA"), "fiqa")

    def test_empty_subset_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid SQuTR subset"):
            squtr_subset_name("")


if __name__ == "__main__":
    unittest.main()
