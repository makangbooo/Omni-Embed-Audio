from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.analyze_asrur_phase2_wer import (
    CONDITIONS,
    analyze,
    write_outputs,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class AnalyzeASRURPhase2WERTest(unittest.TestCase):
    def make_inputs(self, root: Path) -> tuple[Path, Path, Path]:
        cache_root = root / "cache"
        result_root = root / "results"
        queries = root / "queries.jsonl"
        write_jsonl(
            queries,
            [
                {"_id": "q1", "text": "alpha beta"},
                {"_id": "q2", "text": "gamma delta"},
            ],
        )
        for condition in CONDITIONS:
            write_jsonl(
                cache_root / "whisper" / condition / "nbest.jsonl",
                [
                    {
                        "query_id": query_id,
                        "no_speech_probability": None,
                        "hypotheses": [
                            {
                                "rank": rank,
                                "text": (
                                    reference
                                    if query_id == "q1"
                                    else "gamma wrong"
                                ),
                                "sequence_score": -float(rank),
                                "average_token_logprob": -float(rank),
                                "valid_token_count": 2,
                            }
                            for rank in range(1, 5)
                        ],
                    }
                    for query_id, reference in (
                        ("q1", "alpha beta"),
                        ("q2", "gamma delta"),
                    )
                ],
            )
            metrics = {
                "evaluations": {
                    "B1_whisper_1best_bge_dense": {
                        "per_query": {
                            "q1": {
                                "nDCG@10": 1.0,
                                "MRR@10": 1.0,
                                "Recall@10": 1.0,
                                "Recall@20": 1.0,
                                "Recall@50": 1.0,
                                "Recall@100": 1.0,
                            },
                            "q2": {
                                "nDCG@10": 0.0,
                                "MRR@10": 0.0,
                                "Recall@10": 0.0,
                                "Recall@20": 0.0,
                                "Recall@50": 0.0,
                                "Recall@100": 0.0,
                            },
                        }
                    }
                }
            }
            metrics_path = result_root / condition / "metrics.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        return cache_root, result_root, queries

    @mock.patch(
        "scripts.analyze_asrur_phase2_wer.git_output",
        return_value="test",
    )
    def test_analyzes_all_conditions_and_writes_json_safe_bins(
        self,
        _git: object,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_root, result_root, queries = self.make_inputs(root)
            report, buckets, per_query = analyze(
                cache_root=cache_root,
                result_root=result_root,
                queries_path=queries,
                method="B1_whisper_1best_bge_dense",
            )
            self.assertEqual(report["status"], "complete")
            self.assertEqual(set(report["conditions"]), set(CONDITIONS))
            self.assertEqual(len(per_query), 8)
            self.assertEqual(
                report["conditions"]["clean"]["corpus_wer"]["WER"],
                0.25,
            )
            self.assertEqual(
                report["conditions"]["clean"]["per_query_wer"]["maximum"],
                0.5,
            )
            self.assertTrue(
                any(row["upper_exclusive"] is None for row in buckets)
            )
            self.assertFalse(
                any(
                    isinstance(value, float) and not math.isfinite(value)
                    for row in buckets
                    for value in row.values()
                )
            )

            output = root / "output"
            write_outputs(output, report, buckets, per_query)
            json.loads((output / "wer_analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(
                len((output / "per_query.jsonl").read_text().splitlines()),
                8,
            )
            self.assertTrue((output / "wer_bins.csv").is_file())

            with self.assertRaises(FileExistsError):
                write_outputs(output, report, buckets, per_query)

    @mock.patch(
        "scripts.analyze_asrur_phase2_wer.git_output",
        return_value="test",
    )
    def test_rejects_query_set_mismatch(self, _git: object) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_root, result_root, queries = self.make_inputs(root)
            metrics_path = result_root / "clean" / "metrics.json"
            value = json.loads(metrics_path.read_text(encoding="utf-8"))
            del value["evaluations"]["B1_whisper_1best_bge_dense"]["per_query"]["q2"]
            metrics_path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "query sets differ"):
                analyze(
                    cache_root=cache_root,
                    result_root=result_root,
                    queries_path=queries,
                    method="B1_whisper_1best_bge_dense",
                )


if __name__ == "__main__":
    unittest.main()
