import json
import tempfile
import unittest
from pathlib import Path

from AudioRetrieval.asr_uncertainty_reranking.artifacts import (
    assemble_rerank_inputs,
    load_cross_encoder_scores,
    load_feature_rows,
    load_frozen_candidates,
    load_nbest,
    load_unbounded_qrels,
    write_feature_rows_once,
)
from AudioRetrieval.asr_uncertainty_reranking.schema import CandidateFeatureRow


def write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class ASRURArtifactTest(unittest.TestCase):
    def test_cached_inputs_are_joined_by_exact_query_and_candidate_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates_path = root / "top100.jsonl"
            nbest_path = root / "nbest.jsonl"
            ce_path = root / "ce.jsonl"
            qrels_path = root / "qrels.jsonl"
            write_jsonl(
                candidates_path,
                [{"query_id": "q1", "candidate_ids": ["d1", "d2"], "scores": [0.8, 0.2]}],
            )
            write_jsonl(
                nbest_path,
                [
                    {
                        "query_id": "q1",
                        "no_speech_probability": None,
                        "hypotheses": [
                            {
                                "rank": rank,
                                "text": f"hypothesis {rank}",
                                "sequence_score": -rank,
                                "average_token_logprob": -rank / 10,
                                "valid_token_count": 3,
                            }
                            for rank in range(1, 5)
                        ],
                    }
                ],
            )
            write_jsonl(
                ce_path,
                [
                    {
                        "query_id": "q1",
                        "candidate_ids": ["d1", "d2"],
                        "scores": [[rank, -rank] for rank in range(1, 5)],
                    }
                ],
            )
            write_jsonl(
                qrels_path,
                [{"query-id": "q1", "corpus-id": "outside_top100", "score": "1"}],
            )
            inputs = assemble_rerank_inputs(
                load_frozen_candidates(candidates_path),
                load_nbest(nbest_path),
                load_cross_encoder_scores(ce_path),
            )
            self.assertEqual(inputs["q1"].candidate_ids, ("d1", "d2"))
            self.assertEqual(len(inputs["q1"].hypotheses), 4)
            self.assertEqual(
                load_unbounded_qrels(qrels_path),
                {"q1": {"outside_top100": 1.0}},
            )

            write_jsonl(
                ce_path,
                [
                    {
                        "query_id": "q1",
                        "candidate_ids": ["d2", "d1"],
                        "scores": [[rank, -rank] for rank in range(1, 5)],
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "candidate order differs"):
                assemble_rerank_inputs(
                    load_frozen_candidates(candidates_path),
                    load_nbest(nbest_path),
                    load_cross_encoder_scores(ce_path),
                )

    def test_feature_rows_round_trip_and_refuse_overwrite(self) -> None:
        row = CandidateFeatureRow(
            query_id="q",
            document_id="d",
            relevance=1.0,
            oea_score=0.2,
            asr_score=0.8,
            features=(1.0, 2.0),
            feature_names=("left", "right"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.jsonl"
            write_feature_rows_once(path, [row])
            self.assertEqual(load_feature_rows(path), [row])
            with self.assertRaises(FileExistsError):
                write_feature_rows_once(path, [row])


if __name__ == "__main__":
    unittest.main()
