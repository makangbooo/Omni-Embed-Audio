import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from AudioRetrieval.asr_uncertainty_reranking.model_adapters import (
    WhisperGenerationStageError,
)
from scripts.generate_asrur_frozen_caches import (
    consolidate_embedding_chunks,
    finalize_jsonl,
    import_whisper_resume_shards,
    load_bge_inputs,
    load_embedding_chunk,
    load_id_file,
    save_embedding_chunk,
    save_json_shard,
    strict_json_object,
    validate_whisper_shard,
    whisper_numeric_failure_is_retryable,
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
        self.assertIn("MAX_CONSECUTIVE_WHISPER_FAILURES = 8", source)
        self.assertIn("consecutive failures", source)
        self.assertIn('"generation_entrypoint": "base_GenerationMixin_generate"', source)
        self.assertIn(
            '"decoder_prompt": "explicit_language_task_no_timestamps"',
            source,
        )
        self.assertIn(
            '"teacher_forced_conditional_logprob_float32_cross_entropy"',
            source,
        )
        self.assertIn('"beam_transition_scores_used": False', source)
        self.assertIn('"failure_stage": failure_stage', source)
        self.assertIn("--max-record-attempts", source)
        self.assertIn("--resume-shards-from", source)
        self.assertIn("partial_resume_provenance.json", source)
        self.assertIn("numeric_retry_recoveries.jsonl", source)

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

    def test_numeric_retry_is_limited_to_explicit_whisper_failures(self) -> None:
        self.assertTrue(
            whisper_numeric_failure_is_retryable(
                WhisperGenerationStageError(
                    "nbest_artifact_validation",
                    "ValueError: transition log probabilities must be finite",
                )
            )
        )
        self.assertTrue(
            whisper_numeric_failure_is_retryable(
                WhisperGenerationStageError(
                    "four_beam_output_validation",
                    "beam sequence scores contain non-finite values",
                )
            )
        )
        self.assertFalse(
            whisper_numeric_failure_is_retryable(
                WhisperGenerationStageError(
                    "four_beam_generation",
                    "CUDA out of memory",
                )
            )
        )
        self.assertFalse(
            whisper_numeric_failure_is_retryable(
                ValueError("transition log probabilities must be finite")
            )
        )

    def test_strict_json_rejects_nan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            path.write_text('{"value": NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-standard"):
                strict_json_object(path)

    def test_partial_whisper_resume_validates_and_copies_exact_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "destination"
            (source / "shards").mkdir(parents=True)
            (source / "failures").mkdir()
            destination.mkdir()
            records = [
                SimpleNamespace(
                    query_id="q1",
                    record_id="en/fiqa:snr_0:q1",
                    condition="snr_0",
                    audio_path="/audio/q1.wav",
                ),
                SimpleNamespace(
                    query_id="q2",
                    record_id="en/fiqa:snr_0:q2",
                    condition="snr_0",
                    audio_path="/audio/q2.wav",
                ),
            ]
            source_identity = {
                "schema_version": 1,
                "stage": "whisper",
                "git_commit": "a" * 40,
                "config": {"sha256": "config"},
                "input": {"sha256": "input"},
                "subset": "fiqa",
                "conditions": ["snr_0"],
                "model": {"name": "whisper"},
                "device": "cuda:0",
                "dtype": "bfloat16",
                "row_count": 2,
                "decode": {"num_beams": 4},
            }
            destination_identity = {
                **source_identity,
                "git_commit": "b" * 40,
                "record_retry": {
                    "max_attempts": 3,
                    "eligible_failures": "numeric",
                    "selection_policy": "first strict finite result",
                },
            }
            (source / "run_identity.json").write_text(
                json.dumps(source_identity, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            row = {
                "query_id": "q1",
                "record_id": "en/fiqa:snr_0:q1",
                "source_query_id": "q1",
                "condition": "snr_0",
                "audio_path": "/audio/q1.wav",
                "no_speech_probability": None,
                "no_speech_probability_status": (
                    "not_reliably_exposed_by_generation_api"
                ),
                "hypotheses": [
                    {
                        "rank": rank,
                        "text": f"text {rank}",
                        "sequence_score": -float(rank),
                        "average_token_logprob": -0.1 * rank,
                        "valid_token_count": rank,
                    }
                    for rank in range(1, 5)
                ],
            }
            save_json_shard(source, 0, row)
            (source / "failures" / "failure.json").write_text(
                '{"status":"failed"}\n',
                encoding="utf-8",
            )

            provenance = import_whisper_resume_shards(
                source_dir=source,
                destination_dir=destination,
                records=records,
                expected_hypotheses=4,
                expected_source_git_commit="a" * 40,
                destination_identity=destination_identity,
            )

            self.assertTrue(provenance.is_file())
            self.assertEqual(
                strict_json_object(provenance)["imported_shard_count"],
                1,
            )
            copied = destination / "shards" / "00000000.json"
            self.assertEqual(
                copied.read_text(encoding="utf-8"),
                (source / "shards" / "00000000.json").read_text(
                    encoding="utf-8"
                ),
            )
            validate_whisper_shard(
                copied,
                index=0,
                record=records[0],
                expected_hypotheses=4,
            )
            second = import_whisper_resume_shards(
                source_dir=source,
                destination_dir=destination,
                records=records,
                expected_hypotheses=4,
                expected_source_git_commit="a" * 40,
                destination_identity=destination_identity,
            )
            self.assertEqual(second, provenance)


if __name__ == "__main__":
    unittest.main()
