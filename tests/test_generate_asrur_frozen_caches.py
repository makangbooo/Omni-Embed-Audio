import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.generate_asrur_frozen_caches import (
    consolidate_embedding_chunks,
    finalize_jsonl,
    load_bge_inputs,
    load_embedding_chunk,
    load_id_file,
    save_embedding_chunk,
    save_json_shard,
)


def write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class GenerateASRURFrozenCachesTest(unittest.TestCase):
    def test_formal_scripts_keep_conditions_separate_and_use_qrels_ids(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "generate_asrur_frozen_caches.py"
        ).read_text(encoding="utf-8")
        self.assertIn("len(args.conditions) != 1", source)
        self.assertIn('"query_id": record.query_id', source)
        self.assertIn('"record_id": record.record_id', source)
        self.assertIn("--expected-hypotheses", source)

    def test_embedding_chunks_resume_and_consolidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks = root / "chunks"
            chunks.mkdir()
            first = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            second = np.asarray([[-1.0, 0.0]], dtype=np.float32)
            save_embedding_chunk(chunks, 0, 2, first)
            save_embedding_chunk(chunks, 2, 3, second)
            np.testing.assert_array_equal(
                load_embedding_chunk(chunks, 0, 2, 2),
                first,
            )
            output = root / "embeddings.npy"
            consolidate_embedding_chunks(
                chunks,
                total=3,
                batch_size=2,
                dimension=2,
                destination=output,
            )
            np.testing.assert_array_equal(
                np.load(output, allow_pickle=False),
                np.concatenate([first, second]),
            )
            with self.assertRaises(FileExistsError):
                save_embedding_chunk(chunks, 0, 2, first)

    def test_json_shards_finalize_in_index_order_and_refuse_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_json_shard(root, 1, {"query_id": "q2"})
            save_json_shard(root, 0, {"query_id": "q1"})
            output = finalize_jsonl(
                root,
                count=2,
                destination_name="values.jsonl",
            )
            self.assertEqual(
                [json.loads(line)["query_id"] for line in output.read_text().splitlines()],
                ["q1", "q2"],
            )
            with self.assertRaisesRegex(RuntimeError, "differs"):
                save_json_shard(root, 0, {"query_id": "changed"})

    def test_bge_inputs_use_pinned_document_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            write_jsonl(
                path,
                [
                    {"_id": "d1", "title": "Title", "text": "Body"},
                    {"_id": "d2", "title": "", "text": ""},
                ],
            )
            ids, texts = load_bge_inputs(path, input_kind="corpus")
            self.assertEqual(ids, ["d1", "d2"])
            self.assertEqual(texts, ["Title\nBody", ""])

    def test_bge_query_inputs_require_and_apply_qrels_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.jsonl"
            write_jsonl(
                path,
                [
                    {"_id": "test-b", "text": "B"},
                    {"_id": "train-a", "text": "leak"},
                    {"_id": "test-a", "text": "A"},
                ],
            )
            with self.assertRaisesRegex(ValueError, "qrels-derived"):
                load_bge_inputs(path, input_kind="queries")
            ids, texts = load_bge_inputs(
                path,
                input_kind="queries",
                query_ids={"test-a", "test-b"},
            )
            self.assertEqual(ids, ["test-a", "test-b"])
            self.assertEqual(texts, ["A", "B"])
            with self.assertRaisesRegex(ValueError, "absent"):
                load_bge_inputs(
                    path,
                    input_kind="queries",
                    query_ids={"missing"},
                )

    def test_id_loader_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ids.jsonl"
            write_jsonl(path, [{"id": "a"}, {"id": "b"}])
            self.assertEqual(load_id_file(path, field="id"), ["a", "b"])
            write_jsonl(path, [{"id": "a"}, {"id": "a"}])
            with self.assertRaisesRegex(ValueError, "unique"):
                load_id_file(path, field="id")


if __name__ == "__main__":
    unittest.main()
