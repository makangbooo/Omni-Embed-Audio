from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from scripts.build_mga_clap_portable_lock import EXPECTED_SOURCE_REVISION
from scripts.validate_mga_clap_embeddings import load_embeddings


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MAIN_WRAPPER = REPOSITORY_ROOT / "scripts/run_mga_clap_clotho_main.sh"
UIQ_WRAPPER = REPOSITORY_ROOT / "scripts/run_mga_clap_clotho_positive_uiq.sh"
DEPENDENCY_INSTALLER = (
    REPOSITORY_ROOT / "scripts/install_mga_clap_runtime_dependencies.sh"
)
DEPENDENCY_REQUIREMENTS = (
    REPOSITORY_ROOT / "configs/resources/mga_clap_runtime_overlay.requirements.txt"
)
CLI = REPOSITORY_ROOT / "AudioRetrieval/cli/main.py"
ADAPTER = REPOSITORY_ROOT / "AudioRetrieval/models/mga_clap_adapter.py"
PRECOMPUTER = REPOSITORY_ROOT / "AudioRetrieval/preprocessing/embeddings/mga_clap.py"
FAILURE_AUDIT = (
    REPOSITORY_ROOT
    / "results/audits/mga_clap_clotho_smoke_failure_20260731.json"
)
RUAMEL_FAILURE_AUDIT = (
    REPOSITORY_ROOT
    / "results/audits/mga_clap_clotho_ruamel_failure_20260731.json"
)


class MGAClapPipelineTests(unittest.TestCase):
    def test_source_revision_is_pinned(self) -> None:
        self.assertEqual(
            EXPECTED_SOURCE_REVISION,
            "48ca5a5cd22cd34427e118bd8cf332090ec54770",
        )

    def test_cli_requires_all_formal_resources(self) -> None:
        source = CLI.read_text(encoding="utf-8")
        for marker in (
            "--mga-repo",
            "--mga-ckpt",
            "--mga-bert-tokenizer",
            "--mga-checkpoint-sha256",
        ):
            self.assertIn(marker, source)

    def test_adapter_is_external_offline_strict_and_hash_bound(self) -> None:
        source = ADAPTER.read_text(encoding="utf-8")
        for marker in (
            'importlib.import_module("models.ase_model")',
            "local_files_only=True",
            "expected_checkpoint_sha256",
            "strict=True",
            'device="cpu"',
            "resolved outside the lock",
        ):
            self.assertIn(marker, source)
        self.assertNotIn("models.mga_clap.models.ase_model", source)

    def test_main_wrapper_is_direct_and_table2_table3_only(self) -> None:
        source = MAIN_WRAPPER.read_text(encoding="utf-8")
        for marker in (
            "MGA-CLAP Clotho main Tables 2 and 3",
            "resource_lock",
            "gpu_smoke_embeddings",
            "gpu_full_embeddings",
            "cpu_table2_table3_metrics",
            "--expected-audio 1045",
            "--expected-captions 5225",
            "evaluate_official_source_oea_clotho.py",
            "MGA_RUNTIME_OVERLAY",
            "from torchlibrosa.augmentation import SpecAugmentation",
            "--runtime-overlay",
            "--runtime-requirements",
            "--dependency-overlay",
            "--dependency-requirements",
            "RUAMEL_YAML_VERSION=0.18.10",
        ):
            self.assertIn(marker, source)
        for forbidden in ("tmux", "exec bash -i", "--uiq-dir", "positive_uiq"):
            self.assertNotIn(forbidden, source)

    def test_runtime_dependency_is_lock_bound(self) -> None:
        lock_builder = (
            REPOSITORY_ROOT / "scripts/build_mga_clap_portable_lock.py"
        ).read_text(encoding="utf-8")
        for marker in (
            "--runtime-overlay",
            "--runtime-requirements",
            "--dependency-overlay",
            "--dependency-requirements",
            "MGA runtime overlay marker mismatch",
            "MGA dependency overlay marker mismatch",
            "torchlibrosa-0.1.0.dist-info",
            '"torchlibrosa_version": "0.1.0"',
            "ruamel.yaml-0.18.10.dist-info",
            '"ruamel_yaml_version": "0.18.10"',
        ):
            self.assertIn(marker, lock_builder)

    def test_first_remote_failure_is_preserved(self) -> None:
        audit = json.loads(FAILURE_AUDIT.read_text(encoding="utf-8"))
        self.assertEqual(audit["status"], "failed")
        self.assertEqual(audit["run"]["failed_stage"], "gpu_smoke_embeddings")
        self.assertTrue(audit["execution"]["gpu_used"])
        self.assertTrue(audit["execution"]["oea_official_source_used"])
        self.assertEqual(audit["failure"]["missing_module"], "torchlibrosa")
        self.assertIsNone(audit["results"]["retrieval_metrics"])
        self.assertFalse(audit["preservation"]["checkpoint_redownload_required"])

        ruamel_audit = json.loads(
            RUAMEL_FAILURE_AUDIT.read_text(encoding="utf-8")
        )
        self.assertEqual(ruamel_audit["failure"]["missing_module"], "ruamel")
        self.assertTrue(ruamel_audit["execution"]["gpu_used"])
        self.assertTrue(ruamel_audit["execution"]["oea_official_source_used"])
        self.assertIsNone(ruamel_audit["results"]["retrieval_metrics"])

    def test_dependency_installer_is_pinned_direct_and_cpu_only(self) -> None:
        source = DEPENDENCY_INSTALLER.read_text(encoding="utf-8")
        for marker in (
            "MGA-CLAP runtime dependency installation",
            "GPU_USED=no",
            "OEA_OFFICIAL_SOURCE_USED=yes",
            "--no-deps --require-hashes",
            "RUAMEL_YAML_IMPORT=complete",
            "COMPLETION_STATUS=complete",
        ):
            self.assertIn(marker, source)
        for forbidden in ("tmux", "exec bash -i", "CUDA_VISIBLE_DEVICES=0"):
            self.assertNotIn(forbidden, source)
        self.assertEqual(
            DEPENDENCY_REQUIREMENTS.read_text(encoding="utf-8").strip(),
            "ruamel.yaml==0.18.10 "
            "--hash=sha256:30f22513ab2301b3d2b577adc121c6471f28734d3d9728581245f1e76468b4f1",
        )

    def test_precomputer_refuses_main_artifact_overwrite(self) -> None:
        source = PRECOMPUTER.read_text(encoding="utf-8")
        self.assertIn("refusing to overwrite", source)
        self.assertIn("Preserve the generic UIQ route", source)

    def test_positive_uiq_wrapper_is_direct_and_reuses_main_embeddings(self) -> None:
        source = UIQ_WRAPPER.read_text(encoding="utf-8")
        for marker in (
            "MGA-CLAP Clotho positive UIQ Tables 12-15",
            "GPU_USED=yes",
            "OEA_OFFICIAL_SOURCE_USED=yes",
            "gpu_uiq_embeddings",
            "cpu_table12_table15_metrics",
            "--model mga_clap",
            "--mga-bert-tokenizer",
            "--mga-checkpoint-sha256",
            "--baseline-dir",
            "--uiq-dir",
        ):
            self.assertIn(marker, source)
        for forbidden in ("tmux", "exec bash -i"):
            self.assertNotIn(forbidden, source)

    def test_uiq_cli_passes_pinned_mga_resources(self) -> None:
        source = CLI.read_text(encoding="utf-8")
        precomputer = (
            REPOSITORY_ROOT
            / "AudioRetrieval/preprocessing/embeddings/uiq_text.py"
        ).read_text(encoding="utf-8")
        for marker in (
            'choices=["oea", "mga_clap"]',
            "mga_bert_tokenizer",
            "mga_checkpoint_sha256",
        ):
            self.assertIn(marker, source)
        self.assertIn("bert_tokenizer_path", precomputer)
        self.assertIn("expected_checkpoint_sha256", precomputer)

    def test_validator_accepts_shared_dynamic_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.npz"
            values = np.zeros((2, 9), dtype=np.float32)
            values[:, 0] = 1.0
            np.savez_compressed(
                path,
                embeddings=values,
                clip_ids=np.array(["a", "b"]),
            )
            report, dimension = load_embeddings(path, 2)
        self.assertEqual(dimension, 9)
        self.assertEqual(report["shape"], [2, 9])


if __name__ == "__main__":
    unittest.main()
