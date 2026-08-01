from __future__ import annotations

from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPOSITORY_ROOT / "scripts/run_oea_qwen7b_clotho_positive_uiq.sh"
UIQ_PRECOMPUTER = (
    REPOSITORY_ROOT
    / "AudioRetrieval/preprocessing/embeddings/uiq_text.py"
)


class OeaQwen7bPositiveUiqPipelineTests(unittest.TestCase):
    def test_wrapper_supports_both_pinned_variants(self) -> None:
        source = WRAPPER.read_text(encoding="utf-8")
        for marker in (
            "oea_qwen7b)",
            "oea_qwen7b_cl)",
            "OEA-Qwen7B-AC",
            "step_300.pt",
            "17940602533",
            "OEA-Qwen7B-Cl",
            "step_330.pt",
            "17940602661",
            "official_source_oea_qwen7b_clotho_main_20260730_002924",
            "official_source_oea_qwen7b_cl_clotho_main_20260730_004037",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, source)

    def test_wrapper_is_direct_and_reuses_main_embeddings(self) -> None:
        source = WRAPPER.read_text(encoding="utf-8")
        for marker in (
            "Clotho positive UIQ Tables 12-15",
            "GPU_USED=yes",
            "OEA_OFFICIAL_SOURCE_USED=yes",
            "gpu_uiq_embeddings",
            "cpu_table12_table15_metrics",
            "--model oea",
            "--checkpoint",
            "--repo-id Qwen/Qwen2.5-Omni-7B",
            "--local-path",
            "--baseline-dir",
            "--uiq-dir",
            "reuses completed audio embeddings",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, source)
        for forbidden in ("tmux", "exec bash -i", "rm -rf", "git clean"):
            self.assertNotIn(forbidden, source)

    def test_uiq_precomputer_uses_official_oea_precomputer(self) -> None:
        source = UIQ_PRECOMPUTER.read_text(encoding="utf-8")
        for marker in (
            'elif self.model_name == "oea"',
            "OEAEmbeddingPrecomputer",
            "oea_checkpoint",
            "oea_repo_id",
            "oea_local_path",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, source)


if __name__ == "__main__":
    unittest.main()
