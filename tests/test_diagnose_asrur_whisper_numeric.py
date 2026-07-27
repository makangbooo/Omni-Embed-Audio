import json
import math
import unittest

from scripts.diagnose_asrur_whisper_numeric import (
    numeric_class,
    summarize_token_scores,
)


class FakeTokenizer:
    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"token-{token_id}"


class DiagnoseASRURWhisperNumericTest(unittest.TestCase):
    def test_numeric_class_never_serializes_nonfinite_value(self) -> None:
        self.assertEqual(numeric_class(0.0), "finite")
        self.assertEqual(numeric_class(math.nan), "nan")
        self.assertEqual(numeric_class(math.inf), "positive_infinity")
        self.assertEqual(numeric_class(-math.inf), "negative_infinity")

    def test_summary_distinguishes_special_and_valid_nonfinite_scores(self) -> None:
        value = summarize_token_scores(
            method="test",
            token_ids=[[10, 99, 11], [12, 13, 99]],
            logprobs=[[-0.1, -math.inf, -math.inf], [-0.2, -0.3, math.nan]],
            decoded_texts=["first", "second"],
            sequence_scores=[-1.0, -2.0],
            tokenizer=FakeTokenizer(),
            ignored_token_ids=[99],
        )
        self.assertEqual(value["valid_token_count"], 4)
        self.assertEqual(value["nonfinite_valid_token_count"], 1)
        self.assertFalse(value["all_valid_token_scores_finite"])
        first = value["hypotheses"][0]
        self.assertEqual(first["nonfinite_valid_token_count"], 1)
        self.assertIsNone(first["tokens"][2]["logprob"])
        self.assertEqual(first["tokens"][2]["logprob_class"], "negative_infinity")
        self.assertTrue(first["tokens"][1]["special"])

    def test_summary_accepts_all_finite_valid_tokens(self) -> None:
        value = summarize_token_scores(
            method="test",
            token_ids=[[10, 99]],
            logprobs=[[-0.2, -math.inf]],
            decoded_texts=["ok"],
            sequence_scores=[-1.0],
            tokenizer=FakeTokenizer(),
            ignored_token_ids=[99],
        )
        self.assertTrue(value["all_valid_token_scores_finite"])
        self.assertEqual(value["nonfinite_valid_token_count"], 0)
        self.assertAlmostEqual(
            value["hypotheses"][0]["finite_valid_average_logprob"],
            -0.2,
        )

    def test_summary_strictly_serializes_nonfinite_sequence_scores(self) -> None:
        value = summarize_token_scores(
            method="test",
            token_ids=[[10], [11], [12]],
            logprobs=[[-0.1], [-0.2], [-0.3]],
            decoded_texts=["nan", "positive", "negative"],
            sequence_scores=[math.nan, math.inf, -math.inf],
            tokenizer=FakeTokenizer(),
            ignored_token_ids=[],
        )

        hypotheses = value["hypotheses"]
        self.assertIsNone(hypotheses[0]["sequence_score"])
        self.assertEqual(hypotheses[0]["sequence_score_class"], "nan")
        self.assertIsNone(hypotheses[1]["sequence_score"])
        self.assertEqual(
            hypotheses[1]["sequence_score_class"],
            "positive_infinity",
        )
        self.assertIsNone(hypotheses[2]["sequence_score"])
        self.assertEqual(
            hypotheses[2]["sequence_score_class"],
            "negative_infinity",
        )
        json.dumps(value, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
