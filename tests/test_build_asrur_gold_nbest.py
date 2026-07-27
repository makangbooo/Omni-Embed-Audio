import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.build_asrur_gold_nbest import build_rows, deterministic_jsonl, main


def write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class BuildASRURGoldNBestTest(unittest.TestCase):
    def test_builds_exact_qrels_subset_without_fabricating_posterior(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queries = root / "queries.jsonl"
            qrels = root / "qrels.jsonl"
            write_jsonl(
                queries,
                [
                    {"_id": "q-extra", "text": "not in test"},
                    {"_id": "q2", "text": "second query"},
                    {"_id": "q1", "text": "first query"},
                ],
            )
            write_jsonl(
                qrels,
                [
                    {"query-id": "q2", "corpus-id": "d2", "score": 1},
                    {"query-id": "q1", "corpus-id": "d1", "score": 1},
                ],
            )
            rows = build_rows(
                queries_path=queries,
                qrels_path=qrels,
                expected_query_count=2,
            )
        self.assertEqual([row["query_id"] for row in rows], ["q1", "q2"])
        self.assertIsNone(rows[0]["hypotheses"][0]["sequence_score"])
        self.assertEqual(rows[0]["hypotheses"][0]["text"], "first query")
        self.assertIn("not_inference_input", rows[0]["source"])
        self.assertEqual(
            deterministic_jsonl(rows),
            deterministic_jsonl(rows),
        )

    def test_rejects_query_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queries = root / "queries.jsonl"
            qrels = root / "qrels.jsonl"
            write_jsonl(queries, [{"_id": "q1", "text": "first query"}])
            write_jsonl(
                qrels,
                [{"query-id": "q1", "corpus-id": "d1", "score": 1}],
            )
            with self.assertRaisesRegex(ValueError, "expected 2"):
                build_rows(
                    queries_path=queries,
                    qrels_path=qrels,
                    expected_query_count=2,
                )

    def test_cli_reuses_identical_immutable_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queries = root / "queries.jsonl"
            qrels = root / "qrels.jsonl"
            output = root / "output"
            write_jsonl(queries, [{"_id": "q1", "text": "first query"}])
            write_jsonl(
                qrels,
                [{"query-id": "q1", "corpus-id": "d1", "score": 1}],
            )
            arguments = [
                "build_asrur_gold_nbest.py",
                "--queries",
                str(queries),
                "--qrels",
                str(qrels),
                "--output-dir",
                str(output),
                "--dataset",
                "FiQA",
                "--split",
                "test",
                "--expected-query-count",
                "1",
            ]
            with (
                mock.patch("sys.argv", arguments),
                mock.patch(
                    "scripts.build_asrur_gold_nbest.git_output",
                    return_value="a" * 40,
                ),
            ):
                self.assertEqual(main(), 0)
                self.assertEqual(main(), 0)
            self.assertTrue((output / "nbest.jsonl").is_file())
            self.assertTrue((output / "cache_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
