from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.build_m2d_clap_portable_lock import (
    EXPECTED_ARCHIVE_SHA256,
    EXPECTED_ARCHIVE_SIZE,
    EXPECTED_SOURCE_REVISION,
    partition_git_status as partition_lock_git_status,
)
from scripts.validate_m2d_clap_embeddings import (
    load_embeddings,
    partition_git_status as partition_validation_git_status,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MAIN_WRAPPER = REPOSITORY_ROOT / "scripts/run_m2d_clap_clotho_main.sh"
UIQ_WRAPPER = (
    REPOSITORY_ROOT / "scripts/run_m2d_clap_clotho_positive_uiq.sh"
)
CLI = REPOSITORY_ROOT / "AudioRetrieval/cli/main.py"
UIQ_PRECOMPUTER = (
    REPOSITORY_ROOT
    / "AudioRetrieval/preprocessing/embeddings/uiq_text.py"
)
PORTABLE_MODEL = REPOSITORY_ROOT / "AudioRetrieval/models/portable_m2d.py"
EVALUATOR = REPOSITORY_ROOT / "scripts/evaluate_official_source_oea_clotho.py"


class M2DClapPipelineTests(unittest.TestCase):
    def test_resource_identity_is_pinned_to_release_archive(self) -> None:
        self.assertEqual(len(EXPECTED_SOURCE_REVISION), 40)
        self.assertEqual(EXPECTED_ARCHIVE_SIZE, 1469703420)
        self.assertEqual(
            EXPECTED_ARCHIVE_SHA256,
            "fd193ae591720df7f1e27ed728ce127e0309b8bd427f0f4b3e5cd17d7ee5e1e1",
        )

    def test_embedding_validator_accepts_shared_dynamic_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.npz"
            values = np.zeros((3, 7), dtype=np.float32)
            values[:, 0] = 1.0
            np.savez_compressed(
                path,
                embeddings=values,
                clip_ids=np.array(["a", "b", "c"]),
            )
            report, dimension = load_embeddings(path, 3)
        self.assertEqual(dimension, 7)
        self.assertEqual(report["shape"], [3, 7])

    def test_cli_requires_explicit_m2d_resources(self) -> None:
        source = CLI.read_text(encoding="utf-8")
        for fragment in ("m2d_clap", "--m2d-ckpt", "--m2d-bert-tokenizer"):
            self.assertIn(fragment, source)

    def test_local_bert_path_avoids_pretrained_weight_download(self) -> None:
        source = PORTABLE_MODEL.read_text(encoding="utf-8")
        self.assertIn("M2D_CLAP_BERT_TOKENIZER_PATH", source)
        self.assertIn("BertModel(BertConfig())", source)
        self.assertIn("local_files_only=True", source)

    def test_main_wrapper_is_direct_and_table2_table3_only(self) -> None:
        source = MAIN_WRAPPER.read_text(encoding="utf-8")
        for marker in (
            "M2D-CLAP Clotho main Tables 2 and 3",
            "resource_revalidation",
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

    def test_positive_uiq_wrapper_is_direct_and_reuses_main_embeddings(self) -> None:
        source = UIQ_WRAPPER.read_text(encoding="utf-8")
        for marker in (
            "M2D-CLAP Clotho positive UIQ Tables 12-15",
            "GPU_USED=yes",
            "OEA_OFFICIAL_SOURCE_USED=yes",
            "gpu_uiq_embeddings",
            "cpu_table12_table15_metrics",
            "--model m2d_clap",
            "--m2d-ckpt",
            "--m2d-bert-tokenizer",
            "--baseline-dir",
            "--uiq-dir",
        ):
            self.assertIn(marker, source)
        for forbidden in ("tmux", "exec bash -i"):
            self.assertNotIn(forbidden, source)

    def test_uiq_precomputer_binds_pinned_m2d_resources(self) -> None:
        source = UIQ_PRECOMPUTER.read_text(encoding="utf-8")
        for marker in (
            '"m2d_clap"',
            "M2DClapAdapter",
            "weight_file=checkpoint",
            "bert_tokenizer_path=tokenizer",
        ):
            self.assertIn(marker, source)

    def test_only_ephemeral_dpc_status_is_ignored(self) -> None:
        raw = "\n".join(
            (
                "?? scripts/.__dpc00000000a6e18d11000003ce",
                " M AudioRetrieval/cli/main.py",
                "?? real-untracked.txt",
            )
        )
        for partition in (partition_lock_git_status, partition_validation_git_status):
            meaningful, ignored = partition(raw)
            self.assertEqual(
                meaningful,
                " M AudioRetrieval/cli/main.py\n?? real-untracked.txt",
            )
            self.assertEqual(ignored, "?? scripts/.__dpc00000000a6e18d11000003ce")
        wrapper = MAIN_WRAPPER.read_text(encoding="utf-8")
        self.assertIn("git_status_ignored.txt", wrapper)
        self.assertIn("IGNORED_EPHEMERAL_GIT_STATUS", wrapper)

    def test_evaluator_requires_matching_shared_dimension(self) -> None:
        source = EVALUATOR.read_text(encoding="utf-8")
        self.assertIn("caption_embeddings.shape[1] != audio_embeddings.shape[1]", source)


if __name__ == "__main__":
    unittest.main()
