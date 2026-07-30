from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.build_mga_clap_portable_lock import EXPECTED_SOURCE_REVISION
from scripts.validate_mga_clap_embeddings import load_embeddings


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MAIN_WRAPPER = REPOSITORY_ROOT / "scripts/run_mga_clap_clotho_main.sh"
CLI = REPOSITORY_ROOT / "AudioRetrieval/cli/main.py"
ADAPTER = REPOSITORY_ROOT / "AudioRetrieval/models/mga_clap_adapter.py"
PRECOMPUTER = REPOSITORY_ROOT / "AudioRetrieval/preprocessing/embeddings/mga_clap.py"


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
        ):
            self.assertIn(marker, source)
        for forbidden in ("tmux", "exec bash -i", "--uiq-dir", "positive_uiq"):
            self.assertNotIn(forbidden, source)

    def test_precomputer_refuses_main_artifact_overwrite(self) -> None:
        source = PRECOMPUTER.read_text(encoding="utf-8")
        self.assertIn("refusing to overwrite", source)
        self.assertIn("Preserve the generic UIQ route", source)

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
