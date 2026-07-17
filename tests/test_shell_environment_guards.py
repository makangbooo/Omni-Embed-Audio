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
            "run_qwen3b_cl_clotho_embeddings.sh",
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
        source = (
            REPOSITORY_ROOT / "scripts/run_embedding_evaluation.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('[[ -d "${OUTPUT_DIR}" ]]', source)
        self.assertIn('Output directory is not empty', source)
        self.assertIn('> >(tee "${OUTPUT_DIR}/stdout.log")', source)
        self.assertIn('2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)', source)
        self.assertNotIn("rm -rf", source)

    def test_large_raw_result_tree_is_ignored(self) -> None:
        source = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("/results/raw/", source)


if __name__ == "__main__":
    unittest.main()
