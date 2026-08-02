from __future__ import annotations

import json
import hashlib
import tempfile
import sys
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np

from scripts.build_vanilla_backbone_eval_config import build_resolved_config
from scripts.export_vanilla_audiocaps_npz import export
from AudioRetrieval.models.robust_clap_adapter import (
    _local_robust_tokenizer_redirect,
)
from scripts import validate_robust_clap_source


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class VanillaAudioCapsExpansionTests(unittest.TestCase):
    def test_all_three_protocol_configs_are_audio_caps_base_only(self) -> None:
        for backbone in (
            "vanilla_nemotron_3b",
            "vanilla_qwen2_5_omni_3b",
            "vanilla_qwen2_5_omni_7b",
        ):
            path = (
                REPOSITORY_ROOT
                / "configs/eval"
                / f"{backbone}_audiocaps_embeddings.json"
            )
            protocol = json.loads(path.read_text(encoding="utf-8"))
            lock = json.loads(
                (REPOSITORY_ROOT / "results/model_locks" / f"{backbone}.json").read_text(
                    encoding="utf-8"
                )
            )
            config = build_resolved_config(
                protocol,
                lock,
                protocol_identity={
                    "repository_path": path.relative_to(REPOSITORY_ROOT).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": "a" * 64,
                },
                model_lock_identity={
                    "repository_path": f"results/model_locks/{backbone}.json",
                    "size_bytes": 1,
                    "sha256": "b" * 64,
                },
                git_commit="c" * 40,
            )
            self.assertEqual(config["dataset"], "AudioCaps v2 test")
            self.assertEqual(config["expected_examples"], 975)
            self.assertEqual(config["caption_count_per_audio"], 5)
            self.assertNotIn("checkpoint", config)

    def test_export_preserves_candidate_and_caption_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            (source / "generation_metrics.json").write_text(
                json.dumps({"schema_version": 1, "status": "complete"}),
                encoding="utf-8",
            )
            candidates = [
                {"candidate_index": 0, "candidate_id": "a"},
                {"candidate_index": 1, "candidate_id": "b"},
            ]
            queries = [
                {"query_index": index, "target_id": target, "text": f"text-{index}"}
                for index, target in enumerate(("a", "a", "b", "b"))
            ]
            (source / "candidate_metadata.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in candidates),
                encoding="utf-8",
            )
            (source / "query_metadata.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in queries),
                encoding="utf-8",
            )
            np.save(source / "candidate_embeddings.npy", np.eye(2, dtype=np.float32))
            np.save(
                source / "query_embeddings.npy",
                np.repeat(np.eye(2, dtype=np.float32), 2, axis=0),
            )
            report = export(
                Namespace(
                    generation_dir=source,
                    output_dir=output,
                    expected_candidates=2,
                    captions_per_audio=2,
                )
            )
            self.assertEqual(report["status"], "complete")
            with np.load(output / "caption_embeddings.npz", allow_pickle=True) as data:
                self.assertEqual(data["clip_ids"].tolist(), ["a", "a", "b", "b"])

    def test_vanilla_runner_declares_no_checkpoint_or_overwrite(self) -> None:
        source = (
            REPOSITORY_ROOT / "scripts/run_audiocaps_vanilla_table2_table3.sh"
        ).read_text(encoding="utf-8")
        for fragment in (
            "MODEL_CHECKPOINT_LOADED=no",
            "DOWNLOADS_REQUIRED=no",
            "OVERWRITE_DELETE_RISK=none",
            "--skip-uiq",
        ):
            self.assertIn(fragment, source)
        for forbidden in ("curl ", "wget ", "gdown ", "rm ", "tmux "):
            self.assertNotIn(forbidden, source)


class RobustClapExpansionTests(unittest.TestCase):
    def test_archive_source_identity_uses_marker_and_file_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".source_revision").write_text("fixed-revision\n", encoding="utf-8")
            source_file = root / "src/laion_clap/hook.py"
            source_file.parent.mkdir(parents=True)
            source_file.write_bytes(b"pinned source")
            expected = {
                "src/laion_clap/hook.py": (
                    len(b"pinned source"),
                    hashlib.sha256(b"pinned source").hexdigest(),
                )
            }
            with mock.patch.object(
                validate_robust_clap_source,
                "EXPECTED_SOURCE_REVISION",
                "fixed-revision",
            ), mock.patch.object(
                validate_robust_clap_source, "EXPECTED_FILES", expected
            ):
                report = validate_robust_clap_source.validate_source(root)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["revision"], "fixed-revision")

    def test_flan_t5_request_is_redirected_without_network_access(self) -> None:
        calls = []

        class BaseTokenizer:
            @classmethod
            def from_pretrained(cls, requested, *args, **kwargs):
                calls.append((cls.__name__, requested, kwargs))
                return f"local:{requested}"

        class BertTokenizer(BaseTokenizer):
            pass

        class RobertaTokenizer(BaseTokenizer):
            pass

        class BartTokenizer(BaseTokenizer):
            pass

        class T5Tokenizer(BaseTokenizer):
            pass

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.BertTokenizer = BertTokenizer
        fake_transformers.RobertaTokenizer = RobertaTokenizer
        fake_transformers.BartTokenizer = BartTokenizer
        fake_transformers.T5Tokenizer = T5Tokenizer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("bert", "roberta", "bart")]
            for path in paths:
                path.mkdir()
            with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
                with _local_robust_tokenizer_redirect(*paths) as local_roberta:
                    self.assertEqual(
                        T5Tokenizer.from_pretrained("google/flan-t5-large"),
                        local_roberta,
                    )
                    with self.assertRaisesRegex(RuntimeError, "unexpected"):
                        T5Tokenizer.from_pretrained("unapproved/remote-tokenizer")

        t5_network_calls = [call for call in calls if call[0] == "T5Tokenizer"]
        self.assertEqual(t5_network_calls, [])
        self.assertTrue(calls[0][2]["local_files_only"])

    def test_adapter_enforces_strict_resource_binding(self) -> None:
        source = (
            REPOSITORY_ROOT / "AudioRetrieval/models/robust_clap_adapter.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037",
            source,
        )
        self.assertIn("local_files_only=True", source)
        self.assertNotIn("strict=False", source)

    def test_clotho_runner_has_smoke_and_claim_boundary(self) -> None:
        source = (
            REPOSITORY_ROOT / "scripts/run_robust_clap_clotho_main.sh"
        ).read_text(encoding="utf-8")
        for fragment in (
            "d08d0e3c545fa22df0930fc0d090741aaa9e2cc1",
            "gpu_smoke_embeddings",
            "--expected-audio 5 --expected-captions 25",
            "strict Robust-specific paper checkpoint identity not claimed",
            "DOWNLOADS_REQUIRED=no",
            "OVERWRITE_DELETE_RISK=none",
            "--robust-bert-tokenizer",
            "--robust-roberta-tokenizer",
            "--robust-bart-tokenizer",
        ):
            self.assertIn(fragment, source)
        for forbidden in ("curl ", "wget ", "gdown ", "rm ", "tmux "):
            self.assertNotIn(forbidden, source)
        self.assertIn("validate_robust_clap_source.py", source)
        self.assertNotIn('git -C "${SOURCE_DIR}"', source)


if __name__ == "__main__":
    unittest.main()
