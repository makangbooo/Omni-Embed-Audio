import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.diagnose_asrur_phase2_no_go import (
    asr_condition_diagnostic,
    cross_condition_agreement,
    load_ids,
    matrix_geometry,
)


def nbest_row(query_id: str, top1: str) -> dict:
    return {
        "query_id": query_id,
        "no_speech_probability": None,
        "hypotheses": [
            {
                "rank": rank,
                "text": top1 if rank == 1 else f"alternative {rank}",
                "sequence_score": -float(rank),
                "average_token_logprob": -0.1 * rank,
                "valid_token_count": rank,
            }
            for rank in range(1, 5)
        ],
    }


class DiagnoseASRURPhase2NoGoTest(unittest.TestCase):
    def test_load_ids_requires_contiguous_unique_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ids.jsonl"
            path.write_text(
                '{"id":"q1","index":0}\n{"id":"q2","index":1}\n',
                encoding="utf-8",
            )
            self.assertEqual(load_ids(path), ["q1", "q2"])
            path.write_text(
                '{"id":"q1","index":1}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-contiguous"):
                load_ids(path)

    def test_matrix_geometry_reports_normalized_collapse_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "embeddings.npy"
            np.save(
                path,
                np.asarray(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [-1.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
            )
            report = matrix_geometry(path, sample_size=3)
            self.assertEqual(report["shape"], [3, 2])
            self.assertAlmostEqual(report["norm_min"], 1.0)
            self.assertAlmostEqual(report["norm_max"], 1.0)
            self.assertEqual(report["sample_size"], 3)

    def test_asr_diagnostic_reports_wer_and_hypothesis_diversity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nbest.jsonl"
            rows = [
                nbest_row("q1", "hello world"),
                nbest_row("q2", "wrong text"),
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            report, top1 = asr_condition_diagnostic(
                condition="clean",
                nbest_path=path,
                references={"q1": "hello world", "q2": "right text"},
                example_count=2,
            )
            self.assertEqual(report["query_count"], 2)
            self.assertEqual(report["exact_normalized_top1_count"], 1)
            self.assertEqual(report["unique_top1_count"], 2)
            self.assertEqual(top1["q1"], "hello world")
            self.assertGreater(
                report["whitespace_casefold_corpus_wer"]["WER"],
                0.0,
            )

    def test_cross_condition_agreement_is_exact_and_symmetric(self) -> None:
        values = {
            "clean": {"q1": "alpha", "q2": "beta"},
            "snr_20": {"q1": "alpha", "q2": "gamma"},
            "snr_10": {"q1": "alpha", "q2": "beta"},
            "snr_0": {"q1": "delta", "q2": "gamma"},
        }
        result = cross_condition_agreement(values)
        self.assertEqual(len(result), 6)
        clean_snr10 = next(
            value
            for value in result
            if value["left"] == "clean" and value["right"] == "snr_10"
        )
        self.assertEqual(clean_snr10["exact_top1_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
