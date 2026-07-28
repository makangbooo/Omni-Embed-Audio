from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.audit_asrur_whisper_cache import audit_cache


class AuditASRURWhisperCacheTest(unittest.TestCase):
    def write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def build_inputs(self, root: Path, transcripts: list[str]) -> tuple[Path, Path, Path]:
        nbest = root / "nbest.jsonl"
        queries = root / "queries.jsonl"
        qrels = root / "qrels.jsonl"
        self.write_jsonl(
            queries,
            [
                {"_id": str(index), "text": f"reference query {index}"}
                for index in range(len(transcripts))
            ],
        )
        self.write_jsonl(
            qrels,
            [
                {
                    "query-id": str(index),
                    "corpus-id": f"d{index}",
                    "score": 1,
                }
                for index in range(len(transcripts))
            ],
        )
        self.write_jsonl(
            nbest,
            [
                {
                    "query_id": str(index),
                    "record_id": f"clean:{index}",
                    "source_query_id": str(index),
                    "condition": "clean",
                    "audio_path": f"/audio/{index}.wav",
                    "no_speech_probability": None,
                    "no_speech_probability_status": "not_exposed",
                    "hypotheses": [
                        {
                            "rank": rank,
                            "text": text if rank == 1 else f"{text} variant {rank}",
                            "sequence_score": -float(rank),
                            "average_token_logprob": -float(rank),
                            "valid_token_count": 3,
                        }
                        for rank in range(1, 5)
                    ],
                }
                for index, text in enumerate(transcripts)
            ],
        )
        return nbest, queries, qrels

    @patch(
        "scripts.audit_asrur_whisper_cache.git_output",
        return_value="test",
    )
    def test_accepts_diverse_finite_complete_cache(self, _git: object) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            nbest, queries, qrels = self.build_inputs(
                Path(temporary),
                [f"reference query {index}" for index in range(10)],
            )
            report = audit_cache(
                nbest_path=nbest,
                queries_path=queries,
                qrels_path=qrels,
                condition="clean",
                expected_hypotheses=4,
                min_unique_top1=8,
                max_mode_fraction=0.95,
                max_corpus_wer=0.95,
            )
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["violations"], [])
            self.assertEqual(report["top1"]["unique_normalized_count"], 10)

    @patch(
        "scripts.audit_asrur_whisper_cache.git_output",
        return_value="test",
    )
    def test_rejects_universal_thank_you_cache(self, _git: object) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            nbest, queries, qrels = self.build_inputs(
                Path(temporary),
                ["Thank you."] * 10,
            )
            report = audit_cache(
                nbest_path=nbest,
                queries_path=queries,
                qrels_path=qrels,
                condition="clean",
                expected_hypotheses=4,
                min_unique_top1=8,
                max_mode_fraction=0.95,
                max_corpus_wer=0.95,
            )
            self.assertEqual(report["status"], "failed")
            self.assertEqual(
                {value["code"] for value in report["violations"]},
                {
                    "top1_diversity_below_integrity_floor",
                    "single_transcript_mode_exceeds_integrity_ceiling",
                    "corpus_wer_exceeds_integrity_ceiling",
                },
            )


if __name__ == "__main__":
    unittest.main()
