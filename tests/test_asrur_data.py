import json
import tempfile
import unittest
from pathlib import Path

from AudioRetrieval.asr_uncertainty_reranking.data import (
    audit_query_split_leakage,
    construct_document_text,
    load_corpus,
    load_qrels,
    load_text_queries,
    normalized_query_text,
)


def write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class ASRURDataTest(unittest.TestCase):
    def test_document_construction_matches_squtr(self) -> None:
        self.assertEqual(construct_document_text("Title", "Body"), "Title\nBody")
        self.assertEqual(construct_document_text("", "Body"), "Body")

    def test_loaders_and_string_qrels_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path = root / "corpus.jsonl"
            query_path = root / "queries.jsonl"
            qrels_path = root / "qrels.jsonl"
            write_jsonl(
                corpus_path,
                [{"_id": "d1", "title": "T", "text": "B"}],
            )
            write_jsonl(query_path, [{"_id": "q1", "text": "query"}])
            write_jsonl(
                qrels_path,
                [{"query-id": "q1", "corpus-id": "d1", "score": "1"}],
            )
            corpus = load_corpus(corpus_path)
            queries = load_text_queries(query_path)
            qrels = load_qrels(
                qrels_path,
                query_ids=queries,
                document_ids=corpus,
            )
            self.assertEqual(corpus["d1"].constructed_text, "T\nB")
            self.assertEqual(qrels, {"q1": {"d1": 1.0}})

    def test_leakage_audit_catches_id_and_normalized_text_overlap(self) -> None:
        result = audit_query_split_leakage(
            {
                "train": {"q1": "  Café   price "},
                "dev": {"q1": "different"},
                "test": {"q9": "CAFÉ PRICE"},
            }
        )
        self.assertTrue(result["has_query_id_overlap"])
        self.assertTrue(result["has_normalized_text_overlap"])
        self.assertEqual(normalized_query_text("  Café   PRICE "), "café price")


if __name__ == "__main__":
    unittest.main()
