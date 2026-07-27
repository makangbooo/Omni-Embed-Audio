import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts/run_asrur_whisper_dtype_smoke.sh"


class RunASRURWhisperDtypeSmokeTest(unittest.TestCase):
    def test_smoke_is_one_record_offline_lock_bound_and_auditable(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"$1" != "--execute"', source)
        self.assertIn("validate_single_bf16_gpu.py", source)
        self.assertIn('GPU_NAME}" != *"RTX 4090"*', source)
        self.assertIn("verify_asrur_model_assets.py", source)
        self.assertIn("HF_HUB_OFFLINE=1", source)
        self.assertIn("TRANSFORMERS_OFFLINE=1", source)
        self.assertIn("HF_DATASETS_OFFLINE=1", source)
        self.assertIn("test_clean_dtype_smoke", source)
        self.assertIn("expected one Whisper row", source)
        self.assertIn("exactly four hypotheses", source)
        self.assertIn("proxy_scores_finite", source)
        self.assertIn("elapsed_seconds.txt", source)
        adapter_source = (
            REPOSITORY_ROOT
            / "AudioRetrieval/asr_uncertainty_reranking/model_adapters.py"
        ).read_text(encoding="utf-8")
        self.assertIn("GenerationMixin.generate.__get__", adapter_source)
        self.assertNotIn("outputs = self._model.generate(", adapter_source)
        self.assertIn("return_attention_mask=True", adapter_source)
        self.assertIn("generation_config.forced_decoder_ids = None", adapter_source)
        self.assertIn("decoder_input_ids=decoder_input_ids", adapter_source)
        self.assertNotIn("nohup", source)
        self.assertNotIn("rm -", source)
        self.assertNotIn("git reset", source)
        self.assertNotIn("git clean", source)


if __name__ == "__main__":
    unittest.main()
