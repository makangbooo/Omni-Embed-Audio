import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.diagnose_asrur_whisper_content import (
    METHODS,
    summarize_method,
    waveform_statistics,
)


class DiagnoseASRURWhisperContentTest(unittest.TestCase):
    def test_waveform_statistics_are_finite_and_auditable(self) -> None:
        report = waveform_statistics(
            np.asarray([-0.5, 0.0, 0.5, 1.0], dtype=np.float32),
            sample_rate=2,
        )
        self.assertEqual(report["sample_count"], 4)
        self.assertEqual(report["duration_seconds"], 2.0)
        self.assertEqual(report["minimum"], -0.5)
        self.assertEqual(report["maximum"], 1.0)
        self.assertGreater(report["rms"], 0.0)
        self.assertEqual(len(report["pcm_sha256"]), 64)

    def test_waveform_statistics_reject_nonfinite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty finite"):
            waveform_statistics(
                np.asarray([0.0, np.nan], dtype=np.float32),
                sample_rate=16_000,
            )

    def test_method_summary_exposes_universal_hallucination(self) -> None:
        rows = [
            {"query_id": "q1", "text": "Thank you."},
            {"query_id": "q2", "text": "Thank you."},
        ]
        report = summarize_method(
            rows=rows,
            references={"q1": "alpha beta", "q2": "gamma delta"},
        )
        self.assertEqual(report["record_count"], 2)
        self.assertEqual(report["unique_transcript_count"], 1)
        self.assertEqual(
            report["most_common_transcripts"][0],
            {"text": "Thank you.", "count": 2},
        )
        self.assertGreaterEqual(
            report["whitespace_casefold_corpus_wer"]["WER"],
            1.0,
        )

    def test_method_names_lock_the_three_way_comparison(self) -> None:
        self.assertEqual(
            METHODS,
            (
                "current_generic_bfloat16_four_best",
                "official_bfloat16_one_best",
                "official_float32_one_best",
            ),
        )


class WhisperContentWrapperTest(unittest.TestCase):
    def test_wrapper_is_tmux_only_offline_and_nonmutating(self) -> None:
        source = Path(
            "scripts/run_asrur_whisper_content_diagnostic.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('[[ -z "${TMUX:-}" ]]', source)
        self.assertIn("tmux new -s asrur_whisper_content_diag", source)
        self.assertIn("HF_HUB_OFFLINE=1", source)
        self.assertIn("TRANSFORMERS_OFFLINE=1", source)
        self.assertIn("formal_cache_mutation=disabled", source)
        self.assertIn("training=disabled", source)
        self.assertNotIn("nohup", source)
        self.assertNotIn("snapshot_download", source)


if __name__ == "__main__":
    unittest.main()
