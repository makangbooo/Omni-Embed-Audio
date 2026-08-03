from __future__ import annotations

import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPOSITORY_ROOT / "scripts/run_clap_negative_uiq.sh"


class ClapNegativeUIQRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_three_models_and_datasets_are_explicit(self) -> None:
        for model in ("laion_clap", "mga_clap", "robust_clap"):
            self.assertIn(f"  {model})", self.source)
        for dataset in ("clotho", "audiocaps", "mecat"):
            self.assertIn(f"  {dataset})", self.source)

    def test_checkpoint_and_pairing_hashes_are_pinned(self) -> None:
        for digest in (
            "8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037",
            "8703740b738e973a5b4d8a18a074ad56880e98f8ba21cd618d7d7ee5422d6e26",
            "c05101e76d6343a1446a0132bb16b041c03dcca7e9d96a62d61710177c9fb177",
            "8a773c5cd8519214baf1ebd715df89dd1b75ee685e563ba4bf9c0c8cf1e7effa",
            "e7c7311281681190f544c5d9d0eed348fd3378bbc4ce9c31358096da36bab8aa",
        ):
            self.assertIn(digest, self.source)

    def test_runner_uses_canonical_negative_evaluator(self) -> None:
        self.assertIn("scripts/evaluate_negative_uiq_npz.py", self.source)
        self.assertIn("uiq_negative_embeddings.npz", self.source)
        self.assertIn("DOWNLOADS_REQUIRED=no", self.source)
        self.assertIn("strict paper checkpoint identity not claimed", self.source)


if __name__ == "__main__":
    unittest.main()
