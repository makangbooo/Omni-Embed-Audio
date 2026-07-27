import tempfile
import unittest
from pathlib import Path

import numpy as np

from AudioRetrieval.asr_uncertainty_reranking.model_adapters import (
    BgeDenseEncoder,
    BgeDenseSettings,
    FrozenModelIdentity,
    WhisperGenerationStageError,
    WhisperNBestGenerator,
    WhisperSettings,
    batched,
    build_whisper_hypotheses,
    l2_normalize_rows,
    model_identity_from_config,
    prepare_whisper_model_inputs,
    scalar_logits,
    whisper_decoder_prompt_tokens,
    whisper_teacher_forced_generated_logprobs,
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


class FakeWhisperProcessor:
    def __init__(self, prompt):
        self.prompt = prompt
        self.calls = []

    def get_decoder_prompt_ids(self, **kwargs):
        self.calls.append(kwargs)
        return self.prompt


class FakeNumericTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    @property
    def ndim(self):
        return self.values.ndim

    @property
    def shape(self):
        return self.values.shape

    def __getitem__(self, key):
        return FakeNumericTensor(self.values[key])

    def __neg__(self):
        return FakeNumericTensor(-self.values)

    def float(self):
        return self

    def transpose(self, left, right):
        return FakeNumericTensor(np.swapaxes(self.values, left, right))

    def repeat_interleave(self, repeats, *, dim):
        return FakeNumericTensor(np.repeat(self.values, repeats, axis=dim))


class FakeTeacherForcedModel:
    def __init__(self):
        self.encoder_arguments = None
        self.scoring_arguments = None

    def get_encoder(self):
        def encode(**kwargs):
            self.encoder_arguments = kwargs
            return type(
                "EncoderOutput",
                (),
                {
                    "last_hidden_state": FakeNumericTensor(
                        np.zeros((1, 3, 2), dtype=np.float32)
                    )
                },
            )()

        return encode

    def __call__(self, **kwargs):
        self.scoring_arguments = kwargs
        decoder = kwargs["decoder_input_ids"]
        batch, length = decoder.shape
        return type(
            "ScoringOutput",
            (),
            {
                "logits": FakeNumericTensor(
                    np.zeros((batch, length, 20), dtype=np.float32)
                )
            },
        )()


class FakeTorchModule:
    class nn:
        class functional:
            @staticmethod
            def cross_entropy(logits, targets, *, reduction):
                if reduction != "none":
                    raise AssertionError(reduction)
                if logits.shape != (2, 20, 6):
                    raise AssertionError(logits.shape)
                if targets.shape != (2, 6):
                    raise AssertionError(targets.shape)
                return FakeNumericTensor(
                    [
                        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                        [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
                    ]
                )


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

    def test_whisper_decoder_prompt_is_explicit_and_fail_closed(self) -> None:
        processor = FakeWhisperProcessor(
            [(1, 50_259), (2, 50_360), (3, 50_364)]
        )
        result = whisper_decoder_prompt_tokens(
            processor=processor,
            decoder_start_token_id=50_258,
            language="en",
            task="transcribe",
        )
        self.assertEqual(result, (50_258, 50_259, 50_360, 50_364))
        self.assertEqual(
            processor.calls,
            [
                {
                    "language": "en",
                    "task": "transcribe",
                    "no_timestamps": True,
                }
            ],
        )
        with self.assertRaisesRegex(ValueError, "contiguous"):
            whisper_decoder_prompt_tokens(
                processor=FakeWhisperProcessor([(2, 50_259)]),
                decoder_start_token_id=50_258,
                language="en",
                task="transcribe",
            )
        with self.assertRaisesRegex(ValueError, "contiguous"):
            whisper_decoder_prompt_tokens(
                processor=FakeWhisperProcessor([(1, None)]),
                decoder_start_token_id=50_258,
                language="en",
                task="transcribe",
            )

    def test_whisper_runtime_failure_preserves_exact_stage(self) -> None:
        failure = WhisperGenerationStageError(
            "teacher_forced_conditional_logprob",
            "non-finite score",
        )
        self.assertEqual(
            failure.stage,
            "teacher_forced_conditional_logprob",
        )
        self.assertIn(
            "stage=teacher_forced_conditional_logprob",
            str(failure),
        )

    def test_teacher_forced_helper_declares_frozen_score_protocol(self) -> None:
        self.assertIn(
            "teacher_forced",
            whisper_teacher_forced_generated_logprobs.__name__,
        )
        self.assertIn(
            "not a calibrated ASR posterior",
            whisper_teacher_forced_generated_logprobs.__doc__,
        )

    def test_teacher_forced_scores_align_only_generated_suffix(self) -> None:
        model = FakeTeacherForcedModel()
        sequences = FakeNumericTensor(
            [
                [1, 2, 3, 10, 11, 12, 13],
                [1, 2, 3, 14, 15, 16, 17],
            ]
        )
        tokens, logprobs = whisper_teacher_forced_generated_logprobs(
            torch_module=FakeTorchModule,
            model=model,
            model_inputs={
                "input_features": FakeNumericTensor(
                    np.zeros((1, 80, 30), dtype=np.float32)
                ),
                "attention_mask": FakeNumericTensor(
                    np.ones((1, 30), dtype=np.int64)
                ),
            },
            sequences=sequences,
            prompt_length=3,
        )
        np.testing.assert_array_equal(
            tokens.values,
            np.asarray(
                [
                    [10, 11, 12, 13],
                    [14, 15, 16, 17],
                ]
            ),
        )
        np.testing.assert_allclose(
            logprobs.values,
            np.asarray(
                [
                    [-0.3, -0.4, -0.5, -0.6],
                    [-1.3, -1.4, -1.5, -1.6],
                ]
            ),
        )
        self.assertEqual(
            model.encoder_arguments["input_features"].shape,
            (1, 80, 30),
        )
        self.assertEqual(
            model.scoring_arguments["encoder_outputs"][0].shape,
            (2, 3, 2),
        )
        self.assertFalse(model.scoring_arguments["use_cache"])

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
