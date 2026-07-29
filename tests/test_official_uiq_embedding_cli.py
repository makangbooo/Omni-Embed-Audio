import argparse
import unittest
from pathlib import Path

from AudioRetrieval.cli.main import add_preprocess_subparsers


class OfficialUiqEmbeddingCliTests(unittest.TestCase):
    def test_multiple_released_uiq_files_are_accepted(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_preprocess_subparsers(subparsers)

        args = parser.parse_args(
            [
                "preprocess",
                "uiq-embeddings",
                "--uiq-jsonl",
                "question.jsonl",
                "--uiq-jsonl",
                "imperative.jsonl",
                "--output-dir",
                "output",
                "--checkpoint",
                "checkpoint.pt",
                "--local-path",
                "base-model",
            ]
        )

        self.assertEqual(args.preprocess_cmd, "uiq-embeddings")
        self.assertEqual(
            args.uiq_jsonl,
            [Path("question.jsonl"), Path("imperative.jsonl")],
        )
        self.assertEqual(args.output_dir, Path("output"))
        self.assertEqual(args.checkpoint, Path("checkpoint.pt"))
        self.assertEqual(args.local_path, Path("base-model"))
        self.assertEqual(args.batch_size_text, 16)


if __name__ == "__main__":
    unittest.main()
