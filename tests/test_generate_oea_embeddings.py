from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.generate_oea_embeddings import (
    candidate_metadata,
    consolidate_chunks,
    ensure_run_identity,
    load_config,
    load_manifest,
    load_verified_chunk,
    query_metadata,
    save_chunk,
    write_text_once_or_verify,
)


class GenerateOEAEmbeddingsTest(unittest.TestCase):
    def test_fixed_config_pins_resources_and_public_code_prompt_protocol(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        config = load_config(
            repository_root / "configs/eval/qwen3b_cl_clotho_embeddings.json"
        )
        self.assertEqual(config["expected_examples"], 1045)
        self.assertEqual(config["caption_count_per_audio"], 5)
        self.assertEqual(config["audio_batch_size"], 1)
        self.assertEqual(config["text_batch_size"], 1)
        self.assertEqual(config["official_variant_id"], "oea_qwen3b_cl")
        self.assertEqual(
            config["checkpoint"]["revision"],
            "54ccd008d4a1340d2a1f8edcd5dd0e82c61367a4",
        )
        self.assertEqual(
            config["audio_prompt_protocol"]["runtime"]["source"], "CODE"
        )
        self.assertIn(
            "not inserted",
            config["audio_prompt_protocol"]["runtime"]["value"],
        )

    def test_lock_bound_smoke_matches_formal_model_protocol(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        formal = load_config(
            repository_root / "configs/eval/qwen3b_cl_clotho_embeddings.json"
        )
        smoke = load_config(
            repository_root
            / "configs/eval/qwen3b_cl_clotho_lock_bound_smoke_embeddings.json"
        )
        rows = load_manifest(
            repository_root
            / "configs/eval/fixtures/vanilla_clotho_5_manifest.jsonl",
            expected_examples=5,
            caption_count=5,
        )

        self.assertEqual(smoke["expected_examples"], 5)
        self.assertEqual(len(rows), 5)
        self.assertEqual(sum(len(row["captions"]) for row in rows), 25)
        for field in (
            "official_variant_id",
            "model",
            "base_model",
            "checkpoint",
            "model_config",
            "audio_prompt_protocol",
            "audio_batch_size",
            "text_batch_size",
        ):
            with self.subTest(field=field):
                self.assertEqual(smoke[field], formal[field])

    def create_manifest(self, root: Path) -> Path:
        rows = []
        for index in range(2):
            audio = root / f"sample-{index}.wav"
            audio.write_bytes(b"fixture")
            rows.append(
                {
                    "sample_id": audio.name,
                    "audio_path": str(audio),
                    "captions": [f"caption {index}-{number}" for number in range(2)],
                    "file_exists": True,
                    "decode_ok": True,
                }
            )
        manifest = root / "manifest.jsonl"
        manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return manifest

    def test_manifest_creates_exact_candidate_and_query_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = load_manifest(
                self.create_manifest(root), expected_examples=2, caption_count=2
            )
            candidates = candidate_metadata(rows)
            queries = query_metadata(rows)

        self.assertEqual(
            [row["candidate_id"] for row in candidates],
            ["sample-0.wav", "sample-1.wav"],
        )
        self.assertEqual(
            [row["query_id"] for row in queries],
            [
                "sample-0.wav#caption_1",
                "sample-0.wav#caption_2",
                "sample-1.wav#caption_1",
                "sample-1.wav#caption_2",
            ],
        )
        self.assertEqual([row["target_id"] for row in queries], [
            "sample-0.wav", "sample-0.wav", "sample-1.wav", "sample-1.wav"
        ])

    def test_chunks_are_verified_resumable_and_consolidated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks = root / "chunks"
            first = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            second = np.array([[1.0, 0.0]], dtype=np.float32)
            save_chunk(chunks, "text", 0, 2, first, 0.1)
            save_chunk(chunks, "text", 2, 3, second, 0.1)

            resumed = load_verified_chunk(chunks, "text", 0, 2, 2)
            self.assertTrue(np.array_equal(resumed, first))
            identity = consolidate_chunks(
                chunks,
                "text",
                total=3,
                batch_size=2,
                embedding_dim=2,
                destination=root / "query_embeddings.npy",
            )
            self.assertEqual(identity["size_bytes"], (root / "query_embeddings.npy").stat().st_size)
            combined = np.load(root / "query_embeddings.npy", allow_pickle=False)
            self.assertTrue(np.array_equal(combined, np.concatenate([first, second])))

    def test_chunk_checksum_corruption_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_chunk(
                root,
                "audio",
                0,
                1,
                np.array([[1.0, 0.0]], dtype=np.float32),
                0.1,
            )
            with (root / "audio_000000_000001.npy").open("ab") as handle:
                handle.write(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                load_verified_chunk(root, "audio", 0, 1, 2)

    def test_resume_identity_must_match_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ensure_run_identity(root, {"git_commit": "one"})
            ensure_run_identity(root, {"git_commit": "one"})
            with self.assertRaisesRegex(RuntimeError, "resume identity differs"):
                ensure_run_identity(root, {"git_commit": "two"})

    def test_immutable_text_artifact_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.jsonl"
            write_text_once_or_verify(path, "first\n")
            write_text_once_or_verify(path, "first\n")
            with self.assertRaisesRegex(RuntimeError, "immutable artifact differs"):
                write_text_once_or_verify(path, "second\n")
            self.assertEqual(path.read_text(encoding="utf-8"), "first\n")


if __name__ == "__main__":
    unittest.main()
