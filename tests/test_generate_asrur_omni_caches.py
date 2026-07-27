import ast
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "generate_asrur_omni_caches.py"


class GenerateASRUROmniCachesTest(unittest.TestCase):
    def test_runner_is_offline_guarded_resumable_and_supports_both_models(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        ast.parse(text)
        self.assertIn('choices=("oea", "original_omni")', text)
        self.assertIn("formal_execution_guard", text)
        self.assertIn("verify_official_model_lock_binding", text)
        self.assertIn("verify_vanilla_model_lock_binding", text)
        self.assertIn("load_embedding_chunk", text)
        self.assertIn("save_embedding_chunk", text)
        self.assertIn("write_cache_manifest_once", text)
        self.assertNotIn("from_pretrained(", text)
        self.assertNotIn("requests.", text)
        self.assertNotIn("rm -", text)

    def test_audio_and_text_protocols_are_explicit(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"audio_protocol": "audio_only_no_text_prefix"', text)
        self.assertIn('"text_protocol": "query_prefix"', text)
        self.assertIn('"normalization": "l2"', text)
        self.assertIn("len(args.conditions) != 1", text)
        self.assertIn('"id": value.query_id', text)
        self.assertIn('"record_id": value.record_id', text)
        self.assertIn('choices=("both", "audio_only")', text)
        self.assertIn("if include_documents", text)


if __name__ == "__main__":
    unittest.main()
