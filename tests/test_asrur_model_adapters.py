import tempfile
import unittest
from pathlib import Path

import numpy as np

from AudioRetrieval.asr_uncertainty_reranking.model_adapters import (
    BgeDenseEncoder,
    BgeDenseSettings,
    FrozenModelIdentity,
    WhisperNBestGenerator,
    WhisperSettings,
    batched,
    build_whisper_hypotheses,
    l2_normalize_rows,
    model_identity_from_config,
    prepare_whisper_model_inputs,
    scalar_logits,
    whisper_valid_token_statistics,
)


REVISION = "a" * 40


class FakeTensor:
    def __init__(self, *, floating: bool):
        self.floating = floating
        self.calls = []

    def is_floating_point(self):
        return self.floating

    def to(self, **kwargs):
        self.calls.append(kwargs)
        return self


class ASRURModelAdapterTest(unittest.TestCase):
    def test_batching_and_l2_normalization_are_stable(self) -> None:
        self.assertEqual(
            list(batched(["a", "b", "c", "d", "e"], 2)),
            [["a", "b"], ["c", "d"], ["e"]],
        )
        normalized = l2_normalize_rows(
            np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float64)
        )
        np.testing.assert_allclose(
            normalized,
            np.asarray([[0.6, 0.8], [0.0, 1.0]], dtype=np.float32),
        )
        self.assertEqual(normalized.dtype, np.float32)
        with self.assertRaisesRegex(ValueError, "zero"):
            l2_normalize_rows(np.asarray([[0.0, 0.0]]))

    def test_cross_encoder_requires_one_scalar_per_pair(self) -> None:
        np.testing.assert_array_equal(
            scalar_logits(np.asarray([[1.0], [2.0]]), expected_rows=2),
            np.asarray([1.0, 2.0], dtype=np.float32),
        )
        with self.assertRaisesRegex(ValueError, "one scalar"):
            scalar_logits(np.asarray([[0.1, 0.9]]), expected_rows=1)

    def test_proxy_score_excludes_declared_special_tokens(self) -> None:
        average, count = whisper_valid_token_statistics(
            token_ids=[10, 11, 2, 0],
            transition_logprobs=[-0.1, -0.3, -4.0, -5.0],
            ignored_token_ids=[0, 2],
        )
        self.assertAlmostEqual(average, -0.2)
        self.assertEqual(count, 2)

    def test_whisper_inputs_match_model_dtype_without_casting_masks(self) -> None:
        features = FakeTensor(floating=True)
        attention_mask = FakeTensor(floating=False)
        ignored = FakeTensor(floating=True)
        result = prepare_whisper_model_inputs(
            {
                "input_features": features,
                "attention_mask": attention_mask,
                "input_values": ignored,
            },
            device="cuda:0",
            floating_dtype="bfloat16",
        )
        self.assertEqual(set(result), {"input_features", "attention_mask"})
        self.assertEqual(
            features.calls,
            [
                {
                    "device": "cuda:0",
                    "non_blocking": True,
                    "dtype": "bfloat16",
                }
            ],
        )
        self.assertEqual(
            attention_mask.calls,
            [{"device": "cuda:0", "non_blocking": True}],
        )
        self.assertEqual(ignored.calls, [])

    def test_whisper_hypothesis_builder_preserves_beam_order(self) -> None:
        hypotheses = build_whisper_hypotheses(
            decoded_texts=["first", "second", "third", "fourth"],
            generated_token_ids=[
                [10, 2],
                [11, 2],
                [12, 2],
                [13, 2],
            ],
            transition_logprobs=[
                [-0.1, -9.0],
                [-0.2, -9.0],
                [-0.3, -9.0],
                [-0.4, -9.0],
            ],
            sequence_scores=[-0.1, -0.2, -0.3, -0.4],
            ignored_token_ids=[2],
        )
        self.assertEqual([value.rank for value in hypotheses], [1, 2, 3, 4])
        self.assertEqual([value.text for value in hypotheses], ["first", "second", "third", "fourth"])
        self.assertEqual(hypotheses[0].valid_token_count, 1)
        self.assertAlmostEqual(hypotheses[3].average_token_logprob, -0.4)

    def test_model_identity_and_settings_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            identity = FrozenModelIdentity(
                name="BAAI/test",
                revision=REVISION,
                local_path=Path(directory),
            )
            settings = BgeDenseSettings(identity=identity)
            encoder = BgeDenseEncoder(settings, device="cpu", dtype="float32")
            with self.assertRaisesRegex(RuntimeError, "loaded"):
                encoder.encode(["query"], batch_size=1)
            WhisperSettings(
                identity=identity,
                num_beams=4,
                num_return_sequences=4,
            )
            generator = WhisperNBestGenerator(
                WhisperSettings(identity=identity),
                device="cpu",
                dtype="float32",
            )
            with self.assertRaisesRegex(RuntimeError, "loaded"):
                generator.generate(np.ones(16_000, dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "cannot exceed"):
                WhisperSettings(
                    identity=identity,
                    num_beams=3,
                    num_return_sequences=4,
                )

    def test_config_identity_uses_exact_local_path_and_revision(self) -> None:
        identity = model_identity_from_config(
            {
                "models": {
                    "dense": {
                        "name": "BAAI/bge-base-en-v1.5",
                        "revision": REVISION,
                        "local_path": "/models/bge",
                    }
                }
            },
            "dense",
        )
        self.assertEqual(identity.local_path, Path("/models/bge"))
        self.assertEqual(identity.revision, REVISION)


if __name__ == "__main__":
    unittest.main()
