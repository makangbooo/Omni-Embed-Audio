from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from AudioRetrieval.models.laion_clap_tokenizers import local_tokenizer_redirect
from scripts.build_laion_clap_portable_lock import (
    EXPECTED_DISTRIBUTIONS,
    write_new_json,
)
from scripts.select_clotho_smoke_csv import select_rows


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPOSITORY_ROOT / "configs/resources/model05_laion_clap.json"
REQUIREMENTS = (
    REPOSITORY_ROOT
    / "configs/resources/laion_clap_1_1_6_overlay.requirements.txt"
)
WRAPPER = REPOSITORY_ROOT / "scripts/run_laion_clap_resource_pipeline.sh"
MAIN_WRAPPER = REPOSITORY_ROOT / "scripts/run_laion_clap_clotho_main.sh"


class LaionClapResourcePipelineTests(unittest.TestCase):
    def test_manifest_fixes_checkpoint_and_all_eager_tokenizers(self) -> None:
        document = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(document["resource_id"], "MODEL-05")
        assets = {asset["name"]: asset for asset in document["assets"]}
        self.assertEqual(len(assets), 4)
        checkpoint = assets["laion_clap_630k_audioset_best"]
        self.assertEqual(
            checkpoint["revision"],
            "b3708341862f581175dba5c356a4ebf74a9b6651",
        )
        self.assertEqual(
            checkpoint["expected_primary_file"],
            {
                "path": "630k-audioset-best.pt",
                "size_bytes": 1863587645,
                "lfs_sha256": (
                    "8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037"
                ),
            },
        )
        self.assertEqual(
            {asset["repo_id"] for asset in document["assets"][1:]},
            {
                "google-bert/bert-base-uncased",
                "FacebookAI/roberta-base",
                "facebook/bart-base",
            },
        )
        self.assertTrue(
            all(len(asset["revision"]) == 40 for asset in document["assets"])
        )

    def test_overlay_requirements_are_exact_and_hash_locked(self) -> None:
        lines = [
            line.strip()
            for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(lines), len(EXPECTED_DISTRIBUTIONS))
        for name, version in EXPECTED_DISTRIBUTIONS.items():
            requirement_name = "laion-clap" if name == "laion_clap" else name
            matches = [
                line
                for line in lines
                if line.startswith(f"{requirement_name}=={version} ")
            ]
            self.assertEqual(len(matches), 1, name)
            self.assertRegex(matches[0], r"--hash=sha256:[0-9a-f]{64}$")

    def test_tokenizer_redirect_is_local_and_restores_classes(self) -> None:
        calls = []

        class BaseTokenizer:
            @classmethod
            def from_pretrained(cls, requested, *args, **kwargs):
                calls.append((cls.__name__, requested, kwargs))
                return requested

        class BertTokenizer(BaseTokenizer):
            pass

        class RobertaTokenizer(BaseTokenizer):
            pass

        class BartTokenizer(BaseTokenizer):
            pass

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.BertTokenizer = BertTokenizer
        fake_transformers.RobertaTokenizer = RobertaTokenizer
        fake_transformers.BartTokenizer = BartTokenizer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for model_id in (
                "bert-base-uncased",
                "roberta-base",
                "facebook/bart-base",
            ):
                path = root / model_id.replace("/", "_")
                path.mkdir()
                paths[model_id] = str(path)
            with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
                with local_tokenizer_redirect(paths):
                    self.assertEqual(
                        BertTokenizer.from_pretrained("bert-base-uncased"),
                        str(Path(paths["bert-base-uncased"]).resolve()),
                    )
                    self.assertEqual(
                        RobertaTokenizer.from_pretrained("roberta-base"),
                        str(Path(paths["roberta-base"]).resolve()),
                    )
                    self.assertEqual(
                        BartTokenizer.from_pretrained("facebook/bart-base"),
                        str(Path(paths["facebook/bart-base"]).resolve()),
                    )
                self.assertNotIn("from_pretrained", BertTokenizer.__dict__)
                self.assertEqual(BertTokenizer.from_pretrained("original"), "original")
        self.assertTrue(all(call[2].get("local_files_only") for call in calls[:3]))
        self.assertNotIn("local_files_only", calls[-1][2])

    def test_portable_lock_refuses_to_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "lock.json"
            write_new_json(output, {"status": "complete"})
            with self.assertRaises(FileExistsError):
                write_new_json(output, {"status": "replaced"})

    def test_wrapper_is_cpu_only_logged_and_non_destructive(self) -> None:
        source = WRAPPER.read_text(encoding="utf-8")
        for marker in (
            "GPU_USED=no",
            "OEA_OFFICIAL_SOURCE_USED=yes",
            "ESTIMATED_TOTAL_TIME=",
            "STAGE_START=package_overlay",
            "STAGE_START=resource_download",
            "STAGE_START=portable_model_lock",
            "FINAL_RUN_RC=",
            "COMPLETION_STATUS=",
            "METRICS_PATH=",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, source)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', source)
        self.assertIn("--require-hashes", source)
        for forbidden in (
            "rm -rf",
            "git reset",
            "git clean",
            "--upgrade",
            "tmux",
            "exec bash -i",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_official_cli_requires_explicit_laion_resources(self) -> None:
        source = (REPOSITORY_ROOT / "AudioRetrieval/cli/main.py").read_text(
            encoding="utf-8"
        )
        for option in (
            "--laion-ckpt",
            "--laion-bert-tokenizer",
            "--laion-roberta-tokenizer",
            "--laion-bart-tokenizer",
        ):
            self.assertIn(option, source)

    def test_smoke_csv_selection_is_fixed_and_non_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "captions.csv"
            output = root / "smoke.csv"
            header = ["file_name", *(f"caption_{index}" for index in range(1, 6))]
            lines = [",".join(header)]
            for row in range(7):
                lines.append(
                    ",".join([f"audio-{row}.wav", *(f"caption-{row}-{i}" for i in range(5))])
                )
            source.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.assertEqual(select_rows(source, output, 5), 5)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 6)
            with self.assertRaises(FileExistsError):
                select_rows(source, output, 5)

    def test_main_wrapper_is_only_clotho_table2_table3(self) -> None:
        source = MAIN_WRAPPER.read_text(encoding="utf-8")
        for marker in (
            "LAION-CLAP Clotho main Tables 2 and 3",
            "gpu_smoke_embeddings",
            "gpu_full_embeddings",
            "cpu_table2_table3_metrics",
            "--expected-audio 1045",
            "--expected-captions 5225",
            "evaluate_official_source_oea_clotho.py",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, source)
        self.assertNotIn("--uiq-dir", source)
        self.assertNotIn("positive_uiq", source)
        self.assertIn("torch.cuda.device_count() == 1", source)
        self.assertNotIn("tmux", source)
        self.assertNotIn("exec bash -i", source)

    def test_embedding_gate_requires_shape_finite_norm_and_new_output(self) -> None:
        source = (
            REPOSITORY_ROOT / "scripts/validate_laion_clap_embeddings.py"
        ).read_text(encoding="utf-8")
        for marker in (
            "expected_shape = (expected_rows, 512)",
            "np.isfinite",
            "np.allclose(norms, 1.0, atol=1e-5)",
            'with path.open("x"',
            '"protocol_status": "controlled_public_code_reproduction"',
        ):
            self.assertIn(marker, source)


if __name__ == "__main__":
    unittest.main()
