from __future__ import annotations

import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ShellEnvironmentGuardsTest(unittest.TestCase):
    def test_conda_environment_checks_do_not_use_quiet_grep_with_pipefail(self) -> None:
        affected_scripts = (
            "download_model01.sh",
            "download_model02.sh",
            "download_data01_clotho_evaluation.sh",
            "resume_environment.sh",
            "setup_environment.sh",
            "validate_gpu_environment.sh",
        )
        for filename in affected_scripts:
            with self.subTest(script=filename):
                source = (REPOSITORY_ROOT / "scripts" / filename).read_text(
                    encoding="utf-8"
                )
                self.assertNotIn("grep -Fxq", source)
                self.assertIn('grep -Fx "${ENV_NAME}" >/dev/null', source)

    def test_wrappers_resolve_the_initializing_conda_independently_of_path(self) -> None:
        wrappers = (
            "validate_gpu_environment.sh",
            "setup_environment.sh",
            "install_data_tools.sh",
            "run_qwen3b_cl_smoke.sh",
            "download_model01.sh",
            "download_model02.sh",
            "run_model_resource_download.sh",
            "download_data01_clotho_evaluation.sh",
            "download_data03_clotho_trainval.sh",
            "run_checkpoint_extraction.sh",
            "run_checkpoint_inspection.sh",
            "resume_environment.sh",
            "run_data02_clotho_validation.sh",
            "run_embedding_evaluation.sh",
            "run_qwen3b_cl_clotho_lock_bound_smoke.sh",
            "run_qwen3b_cl_clotho_embeddings.sh",
            "download_data04_mecat_00a_test.sh",
            "run_data05_mecat_validation.sh",
            "download_data06_audiocaps_v2_metadata.sh",
            "run_data07_audiocaps_v2_metadata_validation.sh",
            "download_data08_wavcaps_metadata.sh",
            "run_data09_wavcaps_metadata_audit.sh",
            "run_data10_clotho_trainval_validation.sh",
            "run_model03_model04_audit.sh",
            "run_data11_mecat_wavcaps_provenance.sh",
            "run_negative_embedding_evaluation.sh",
            "run_qwen3b_clotho_retrieval_suite.sh",
            "run_qwen3b_cl_clotho_positive_uiq_embeddings.sh",
            "run_qwen3b_clotho_positive_uiq_suite.sh",
            "run_official_checkpoint_preparation.sh",
            "run_reproduction.sh",
        )
        for filename in wrappers:
            with self.subTest(script=filename):
                source = (REPOSITORY_ROOT / "scripts" / filename).read_text(
                    encoding="utf-8"
                )
                self.assertIn('source "${ROOT_DIR}/scripts/lib/conda.sh"', source)
                self.assertIn('CONDA_BASE="$(resolve_conda_base)"', source)
                self.assertNotIn('CONDA_BASE="$(conda info --base)"', source)

        resolver = (REPOSITORY_ROOT / "scripts/lib/conda.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('${CONDA_EXE:-}', resolver)
        self.assertIn('${CONDA_PREFIX:-}', resolver)
        self.assertIn("command -v conda", resolver)

    def test_embedding_evaluation_wrapper_refuses_nonempty_output(self) -> None:
        wrappers = (
            "run_embedding_evaluation.sh",
            "run_negative_embedding_evaluation.sh",
        )
        for filename in wrappers:
            with self.subTest(script=filename):
                source = (REPOSITORY_ROOT / "scripts" / filename).read_text(
                    encoding="utf-8"
                )
                self.assertIn('[[ -d "${OUTPUT_DIR}" ]]', source)
                self.assertIn('Output directory is not empty', source)
                self.assertIn('> >(tee "${OUTPUT_DIR}/stdout.log")', source)
                self.assertIn('2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)', source)
                self.assertNotIn("rm -rf", source)

    def test_large_raw_result_tree_is_ignored(self) -> None:
        source = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("/results/raw/", source)

    def test_clotho_trainval_wrapper_audits_before_isolated_extraction(self) -> None:
        source = (
            REPOSITORY_ROOT / "scripts/run_data10_clotho_trainval_validation.sh"
        ).read_text(encoding="utf-8")
        listing_audit = source.index("python scripts/audit_7z_listing.py")
        extraction = source.index('"${EXTRACTOR}" x "${archive}"')
        self.assertLess(listing_audit, extraction)
        self.assertIn('EXTRACT_ROOT="${DATASET_ROOT}/extracted_trainval"', source)
        self.assertIn("refusing to overwrite", source)
        self.assertNotIn("rm -rf", source)

    def test_model_resource_audit_is_read_only(self) -> None:
        source = (
            REPOSITORY_ROOT / "scripts/run_model03_model04_audit.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("scripts/audit_model_resources.py", source)
        self.assertNotIn("download_model_assets.py", source)
        self.assertNotIn("snapshot_download", source)
        self.assertNotIn("rm -rf", source)
        self.assertNotIn("Remove-Item", source)

    def test_official_checkpoint_preparation_is_cpu_only_and_non_overwriting(self) -> None:
        wrapper = (
            REPOSITORY_ROOT / "scripts/run_official_checkpoint_preparation.sh"
        ).read_text(encoding="utf-8")
        implementation = (
            REPOSITORY_ROOT / "scripts/prepare_official_oea_checkpoint.py"
        ).read_text(encoding="utf-8")
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', wrapper)
        self.assertIn("prepare_official_oea_checkpoint.py", wrapper)
        self.assertIn("--inspect-only", wrapper)
        self.assertIn("refusing to overwrite", implementation)
        self.assertNotIn("rm -rf", wrapper)
        self.assertNotIn("rm -rf", implementation)

    def test_mecat_wavcaps_provenance_is_metadata_only(self) -> None:
        source = (
            REPOSITORY_ROOT / "scripts/run_data11_mecat_wavcaps_provenance.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("audit_mecat_wavcaps_provenance.py", source)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', source)
        self.assertNotIn("snapshot_download", source)
        self.assertNotIn("rm -rf", source)

    def test_qwen3b_retrieval_suite_is_cpu_only_and_non_overwriting(self) -> None:
        source = (
            REPOSITORY_ROOT / "scripts/run_qwen3b_clotho_retrieval_suite.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', source)
        self.assertIn("prepare_embedding_evaluation_suite.py prepare", source)
        self.assertIn("run_embedding_evaluation.sh", source)
        self.assertIn("prepare_embedding_evaluation_suite.py finalize", source)
        self.assertIn("Use a new SUITE_ID", source)
        self.assertNotIn("rm -rf", source)

    def test_positive_uiq_suite_is_cpu_only_and_non_overwriting(self) -> None:
        source = (
            REPOSITORY_ROOT / "scripts/run_qwen3b_clotho_positive_uiq_suite.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', source)
        self.assertIn("prepare_positive_uiq_evaluation_suite.py prepare", source)
        self.assertIn("run_embedding_evaluation.sh", source)
        self.assertIn("prepare_positive_uiq_evaluation_suite.py finalize", source)
        self.assertIn("Use a new SUITE_ID", source)
        self.assertNotIn("rm -rf", source)

    def test_formal_embedding_wrappers_resolve_the_committed_model_lock(self) -> None:
        wrappers = (
            "run_qwen3b_cl_clotho_lock_bound_smoke.sh",
            "run_qwen3b_cl_clotho_embeddings.sh",
            "run_qwen3b_cl_clotho_positive_uiq_embeddings.sh",
        )
        for filename in wrappers:
            with self.subTest(script=filename):
                source = (REPOSITORY_ROOT / "scripts" / filename).read_text(
                    encoding="utf-8"
                )
                self.assertIn("results/model_locks/oea_qwen3b_cl.json", source)
                self.assertIn("build_official_oea_eval_config.py", source)
                self.assertIn("config_resolution.json", source)
                self.assertIn("validate_single_a100_80gb.py", source)
                self.assertIn("gpu_preflight.json", source)
                self.assertNotIn("rm -rf", source)

        caption = (
            REPOSITORY_ROOT / "scripts/run_qwen3b_cl_clotho_embeddings.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('--config "${RESOLVED_CONFIG}"', caption)
        smoke = (
            REPOSITORY_ROOT
            / "scripts/run_qwen3b_cl_clotho_lock_bound_smoke.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("vanilla_clotho_5_manifest.jsonl", smoke)
        self.assertIn("5 audio candidates and 25 caption queries", smoke)
        self.assertIn('--config "${RESOLVED_CONFIG}"', smoke)
        uiq = (
            REPOSITORY_ROOT
            / "scripts/run_qwen3b_cl_clotho_positive_uiq_embeddings.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('--base-embedding-config "${RESOLVED_BASE_CONFIG}"', uiq)

    def test_unified_reproduction_entry_is_plan_first_and_non_destructive(self) -> None:
        wrapper = (REPOSITORY_ROOT / "scripts/run_reproduction.sh").read_text(
            encoding="utf-8"
        )
        runner = (REPOSITORY_ROOT / "scripts/run_reproduction.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('exec python scripts/run_reproduction.py "$@"', wrapper)
        self.assertIn("--acknowledge-long-operation", runner)
        self.assertIn("if not args.execute", runner)
        for fragment in (
            "rm -rf",
            "git reset --hard",
            "git clean -fd",
            "git push --force",
        ):
            self.assertNotIn(fragment, wrapper)


if __name__ == "__main__":
    unittest.main()
