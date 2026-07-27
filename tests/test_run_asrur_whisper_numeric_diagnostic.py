import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts/run_asrur_whisper_numeric_diagnostic.sh"


class RunASRURWhisperNumericDiagnosticTest(unittest.TestCase):
    def test_wrapper_is_tmux_only_offline_and_non_mutating(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('[[ -z "${TMUX:-}" ]]', source)
        self.assertIn("tmux new -s asrur_whisper_numeric_diag", source)
        self.assertIn("HF_HUB_OFFLINE=1", source)
        self.assertIn("TRANSFORMERS_OFFLINE=1", source)
        self.assertIn("HF_DATASETS_OFFLINE=1", source)
        self.assertIn("verify_asrur_model_assets.py", source)
        self.assertIn("diagnose_asrur_whisper_numeric.py", source)
        self.assertIn("en/fiqa:snr_0:10639", source)
        self.assertIn("cache_mutation=disabled", source)
        self.assertIn('GPU_NAME}" != *"RTX 4090"*', source)
        self.assertNotIn("nohup", source)
        self.assertNotIn("rm -", source)


if __name__ == "__main__":
    unittest.main()
