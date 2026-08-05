import json
import tempfile
import unittest
from pathlib import Path

from scripts.train_cgp_oea_projection_heads import (
    _load_replay_entries,
    make_replay_batches,
    make_unique_batches,
)


class TrainCGPOEATest(unittest.TestCase):
    def test_batches_do_not_contain_duplicate_clips(self) -> None:
        entries = [
            {"clip_id": "a", "caption": "a1"},
            {"clip_id": "a", "caption": "a2"},
            {"clip_id": "b", "caption": "b1"},
            {"clip_id": "b", "caption": "b2"},
            {"clip_id": "c", "caption": "c1"},
        ]
        batches = make_unique_batches(entries, 2, seed=7)
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(5)))
        for batch in batches:
            self.assertEqual(
                len({entries[index]["clip_id"] for index in batch}), len(batch)
            )

    def test_batches_are_deterministic_for_seed(self) -> None:
        entries = [{"clip_id": str(index // 2)} for index in range(8)]
        self.assertEqual(
            make_unique_batches(entries, 3, seed=11),
            make_unique_batches(entries, 3, seed=11),
        )

    def test_audio_path_prevents_audiocaps_false_negatives(self) -> None:
        entries = [
            {"clip_id": "caption-1", "audio_path": "shared.wav"},
            {"clip_id": "caption-2", "audio_path": "shared.wav"},
            {"clip_id": "caption-3", "audio_path": "other.wav"},
        ]
        for batch in make_unique_batches(entries, 3, seed=3):
            paths = [entries[index]["audio_path"] for index in batch]
            self.assertEqual(len(paths), len(set(paths)))

    def test_replay_batches_do_not_duplicate_documents(self) -> None:
        entries = [
            {"query_id": "q1", "document_id": "d1"},
            {"query_id": "q2", "document_id": "d1"},
            {"query_id": "q3", "document_id": "d2"},
            {"query_id": "q4", "document_id": "d3"},
        ]
        batches = make_replay_batches(entries, 3, seed=7)
        selected = [entries[index] for batch in batches for index in batch]
        self.assertEqual(
            len({entry["document_id"] for entry in selected}), len(selected)
        )

    def test_replay_batches_are_deterministic(self) -> None:
        entries = [
            {"query_id": str(index), "document_id": str(index // 2)}
            for index in range(8)
        ]
        self.assertEqual(
            make_replay_batches(entries, 3, seed=11),
            make_replay_batches(entries, 3, seed=11),
        )

    def test_replay_batches_avoid_cross_query_false_negatives(self) -> None:
        entries = [
            {
                "query_id": "q1",
                "document_id": "d1",
                "positive_document_ids": ["d1", "d2"],
            },
            {
                "query_id": "q2",
                "document_id": "d2",
                "positive_document_ids": ["d2"],
            },
        ]
        batches = make_replay_batches(entries, 2, seed=3)
        self.assertEqual(sorted(len(batch) for batch in batches), [1, 1])

    def test_replay_loader_uses_requested_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "qrels").mkdir()
            (root / "queries.jsonl").write_text(
                '\n'.join(
                    json.dumps({"_id": value, "text": f"query {value}"})
                    for value in ("q1", "q2")
                ) + '\n',
                encoding="utf-8",
            )
            (root / "corpus.jsonl").write_text(
                '\n'.join(
                    json.dumps({"_id": value, "title": "", "text": f"doc {value}"})
                    for value in ("d1", "d2")
                ) + '\n',
                encoding="utf-8",
            )
            (root / "qrels" / "train.jsonl").write_text(
                '\n'.join(
                    json.dumps({"query-id": query, "corpus-id": document, "score": 1})
                    for query, document in (("q1", "d1"), ("q2", "d2"))
                ) + '\n',
                encoding="utf-8",
            )
            entries = _load_replay_entries(root, "train", 0, seed=5)
            self.assertEqual(
                [(row["query_id"], row["document_id"]) for row in entries],
                [("q1", "d1"), ("q2", "d2")],
            )


if __name__ == "__main__":
    unittest.main()
