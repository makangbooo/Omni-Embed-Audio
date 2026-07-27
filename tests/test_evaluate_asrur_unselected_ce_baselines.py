import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.evaluate_asrur_unselected_ce_baselines import (
    UNSELECTED_METHODS,
    evaluate_unselected,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class EvaluateUnselectedCEBaselinesTest(unittest.TestCase):
    def make_inputs(self, root: Path):
        candidates = root / "candidates.jsonl"
        nbest = root / "nbest.jsonl"
        ce = root / "ce.jsonl"
        qrels = root / "qrels.jsonl"
        query_ids = [f"q{index:04d}" for index in range(648)]
        write_jsonl(
            candidates,
            [
                {
                    "query_id": query_id,
                    "candidate_ids": ["d1", "d2"],
                    "scores": [0.8, 0.2],
                }
                for query_id in query_ids
            ],
        )
        write_jsonl(
            nbest,
            [
                {
                    "query_id": query_id,
                    "no_speech_probability": None,
                    "hypotheses": [
                        {
                            "rank": rank,
                            "text": f"hypothesis {rank}",
                            "sequence_score": -float(rank),
                            "average_token_logprob": -float(rank) / 10.0,
                            "valid_token_count": 2,
                        }
                        for rank in range(1, 5)
                    ],
                }
                for query_id in query_ids
            ],
        )
        write_jsonl(
            ce,
            [
                {
                    "query_id": query_id,
                    "candidate_ids": ["d1", "d2"],
                    "scores": [
                        [1.0, 0.0],
                        [0.9, 0.1],
                        [0.8, 0.2],
                        [0.7, 0.3],
                    ],
                }
                for query_id in query_ids
            ],
        )
        write_jsonl(
            qrels,
            [
                {"query-id": query_id, "corpus-id": "d1", "score": 1}
                for query_id in query_ids
            ],
        )
        return candidates, nbest, ce, qrels

    def test_reports_only_methods_without_dev_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_inputs(Path(temporary))
            with mock.patch(
                "scripts.evaluate_asrur_unselected_ce_baselines.git_output",
                return_value="clean",
            ):
                result = evaluate_unselected(
                    config_path=REPOSITORY_ROOT
                    / "configs/asr_uncertainty_reranking/main_experiment.json",
                    top100_path=paths[0],
                    nbest_path=paths[1],
                    cross_encoder_path=paths[2],
                    gold_cross_encoder_path=None,
                    qrels_path=paths[3],
                )
        self.assertEqual(tuple(result["reported_methods"]), UNSELECTED_METHODS)
        self.assertEqual(set(result["evaluations"]), set(UNSELECTED_METHODS))
        self.assertNotIn("B5_fixed_fusion", result["evaluations"])
        self.assertNotIn("B6_rrf", result["evaluations"])
        self.assertNotIn("B7c_4best_proxy", result["evaluations"])
        self.assertFalse(
            result["provenance"]["test_qrels_used_for_parameter_selection"]
        )

    def test_rejects_non_test_query_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = list(self.make_inputs(Path(temporary)))
            rows = paths[3].read_text(encoding="utf-8").splitlines()
            paths[3].write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "query sets"):
                evaluate_unselected(
                    config_path=REPOSITORY_ROOT
                    / "configs/asr_uncertainty_reranking/main_experiment.json",
                    top100_path=paths[0],
                    nbest_path=paths[1],
                    cross_encoder_path=paths[2],
                    gold_cross_encoder_path=None,
                    qrels_path=paths[3],
                )

    def test_reports_gold_upper_bound_from_single_score_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.make_inputs(root)
            gold = root / "gold.jsonl"
            write_jsonl(
                gold,
                [
                    {
                        "query_id": f"q{index:04d}",
                        "candidate_ids": ["d1", "d2"],
                        "scores": [[1.0, 0.0]],
                    }
                    for index in range(648)
                ],
            )
            with mock.patch(
                "scripts.evaluate_asrur_unselected_ce_baselines.git_output",
                return_value="clean",
            ):
                result = evaluate_unselected(
                    config_path=REPOSITORY_ROOT
                    / "configs/asr_uncertainty_reranking/main_experiment.json",
                    top100_path=paths[0],
                    nbest_path=paths[1],
                    cross_encoder_path=paths[2],
                    gold_cross_encoder_path=gold,
                    qrels_path=paths[3],
                )
        self.assertIn("U2_gold_ce", result["reported_methods"])
        self.assertEqual(
            result["evaluations"]["U2_gold_ce"]["num_queries"],
            648,
        )


if __name__ == "__main__":
    unittest.main()
