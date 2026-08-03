from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts.evaluate_negative_uiq_npz import (
    resolve_pairing_candidate_ids,
    run_evaluation,
)


class EvaluateNegativeUIQNPZTest(unittest.TestCase):
    def test_exact_metrics_from_npz_and_pairings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_path = root / "audio.npz"
            query_path = root / "query.npz"
            pairing_path = root / "pairing.jsonl"
            output = root / "output"
            np.savez_compressed(
                audio_path,
                embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
                clip_ids=np.asarray(["target", "hard-negative", "other"], dtype=object),
            )
            np.savez_compressed(
                query_path,
                embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
                clip_ids=np.asarray(["target"], dtype=object),
            )
            pairing_path.write_text(
                json.dumps(
                    {
                        "query_id": "q0",
                        "target_id": "target",
                        "hard_negative_id": "hard-negative",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with mock.patch(
                "scripts.evaluate_negative_uiq_npz.git_output",
                side_effect=["deadbeef", ""],
            ):
                report = run_evaluation(
                    audio_npz=audio_path,
                    query_npz=query_path,
                    pairing_jsonl=pairing_path,
                    output_dir=output,
                    model="fixture",
                    dataset="fixture",
                    expected_candidates=3,
                    expected_queries=1,
                )

            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["metrics"]["R@1"], 100.0)
            self.assertEqual(report["metrics"]["HNSR"], 100.0)
            self.assertEqual(report["metrics"]["HNSR@1"], 100.0)
            self.assertEqual(report["metrics"]["TFR-HN@1"], 100.0)
            self.assertTrue((output / "metrics.json").is_file())
            self.assertTrue((output / "per_query.jsonl").is_file())

    def test_pairing_count_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.savez_compressed(
                root / "audio.npz",
                embeddings=np.eye(2, dtype=np.float32),
                clip_ids=np.asarray(["a", "b"], dtype=object),
            )
            np.savez_compressed(
                root / "query.npz",
                embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
            )
            (root / "pairing.jsonl").write_text("", encoding="utf-8")

            with mock.patch(
                "scripts.evaluate_negative_uiq_npz.git_output",
                side_effect=["deadbeef", ""],
            ):
                with self.assertRaises(ValueError):
                    run_evaluation(
                        audio_npz=root / "audio.npz",
                        query_npz=root / "query.npz",
                        pairing_jsonl=root / "pairing.jsonl",
                        output_dir=root / "output",
                        model="fixture",
                        dataset="fixture",
                        expected_candidates=2,
                        expected_queries=1,
                    )

    def test_unique_suffix_alias_resolves_to_candidate_stem(self) -> None:
        resolved, methods = resolve_pairing_candidate_ids(
            ["target.wav", "negative.wav"],
            ["target", "negative", "other"],
            label="fixture",
        )

        self.assertEqual(resolved, ["target", "negative"])
        self.assertEqual(
            methods, {"unique_casefold_or_stem_candidate_id": 2}
        )

    def test_ambiguous_stem_alias_fails_closed(self) -> None:
        with self.assertRaises(KeyError):
            resolve_pairing_candidate_ids(
                ["duplicate"],
                ["duplicate.wav", "duplicate.mp3"],
                label="fixture",
            )


if __name__ == "__main__":
    unittest.main()
