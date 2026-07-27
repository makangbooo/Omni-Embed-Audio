import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts/run_asrur_phase2_frozen_retrieval.sh"


class RunASRURPhase2FrozenRetrievalTest(unittest.TestCase):
    def test_wrapper_is_explicit_resumable_and_protocol_locked(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('--dry-run|--execute', source)
        self.assertIn('GPU_NAME}" != *"RTX 4090"*', source)
        self.assertIn("validate_single_bf16_gpu.py", source)
        self.assertIn("verify_asrur_model_assets.py", source)
        self.assertIn("build_official_oea_eval_config.py", source)
        self.assertIn("build_vanilla_backbone_eval_config.py", source)
        self.assertIn("generate_asrur_omni_caches.py", source)
        self.assertIn("generate_asrur_frozen_caches.py whisper", source)
        self.assertIn("--query-qrels", source)
        self.assertIn("select_asrur_bge_query_template.py", source)
        self.assertIn("evaluate_asrur_dense_rankings.py", source)
        self.assertIn("content=\"audio_only\"", source)
        self.assertIn('content="both"', source)
        self.assertIn("HF_HUB_OFFLINE=1", source)
        self.assertIn("TRANSFORMERS_OFFLINE=1", source)
        self.assertIn("HF_DATASETS_OFFLINE=1", source)
        self.assertIn("PHASE2_REUSE_CACHE_ROOT", source)
        self.assertIn("reuse_complete_cache_dir", source)
        self.assertIn("run_cache_step", source)
        self.assertIn("reused_cache_dirs.jsonl", source)
        self.assertIn('ln -s "${source}" "${destination}"', source)
        self.assertIn("verify_file_records", source)
        self.assertIn("refusing cross-commit cache reuse", source)
        self.assertNotIn("rm -", source)
        self.assertNotIn("git reset", source)
        self.assertNotIn("git clean", source)

    def test_wrapper_uses_one_shared_document_cache_per_model(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            '"${CACHE_ROOT}/oea/clean/document_embeddings.npy"',
            source,
        )
        self.assertIn(
            '"${CACHE_ROOT}/vanilla/clean/document_embeddings.npy"',
            source,
        )
        self.assertIn(
            '"${CACHE_ROOT}/bge/corpus/embeddings.npy"',
            source,
        )


if __name__ == "__main__":
    unittest.main()
