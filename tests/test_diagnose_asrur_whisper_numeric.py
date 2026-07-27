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


if __name__ == "__main__":
    unittest.main()
