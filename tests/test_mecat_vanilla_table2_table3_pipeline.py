from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.build_vanilla_backbone_eval_config import build_resolved_config
from scripts.prepare_vanilla_mecat_manifest import project


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKBONES = (
    "vanilla_nemotron_3b",
    "vanilla_qwen2_5_omni_3b",
    "vanilla_qwen2_5_omni_7b",
)


class MecatVanillaTable2Table3PipelineTests(unittest.TestCase):
    def test_manifest_projection_fixes_short_all_and_audio_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "sample.flac"
            audio.write_bytes(b"audited audio")
            row = {
                "sample_id": "sample",
                "audio_path": str(audio),
                "audio_size_bytes": audio.stat().st_size,
                "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                "caption_fields": {"short": ["one", "two", "three"]},
                "file_exists": True,
                "decode_ok": True,
            }
            source = root / "source.jsonl"
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")
            output = root / "projected.jsonl"
            report = project(
                Namespace(
                    source_manifest=source,
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    output=output,
                    expected_candidates=1,
                    captions_per_candidate=3,
                )
            )
            self.assertEqual(report["caption_count"], 3)
            projected = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(projected["captions"], ["one", "two", "three"])
            self.assertEqual(projected["audio_path"], str(audio.resolve()))

    def test_manifest_projection_refuses_audio_content_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "sample.flac"
            audio.write_bytes(b"changed")
            row = {
                "sample_id": "sample",
                "audio_path": str(audio),
                "audio_size_bytes": len(b"changed"),
                "audio_sha256": "0" * 64,
                "caption_fields": {"short": ["one", "two", "three"]},
                "file_exists": True,
                "decode_ok": True,
            }
            source = root / "source.jsonl"
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "audio SHA256 mismatch"):
                project(
                    Namespace(
                        source_manifest=source,
                        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        output=root / "projected.jsonl",
                        expected_candidates=1,
                        captions_per_candidate=3,
                    )
                )

    def test_all_three_configs_are_base_only_mecat_short_all(self) -> None:
        for backbone in BACKBONES:
            path = REPOSITORY_ROOT / "configs/eval" / f"{backbone}_mecat_embeddings.json"
            protocol = json.loads(path.read_text(encoding="utf-8"))
            lock = json.loads(
                (REPOSITORY_ROOT / "results/model_locks" / f"{backbone}.json").read_text(
                    encoding="utf-8"
                )
            )
            resolved = build_resolved_config(
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
            self.assertEqual(resolved["expected_examples"], 848)
            self.assertEqual(resolved["caption_count_per_audio"], 3)
            self.assertNotIn("checkpoint", resolved)

    def test_runner_declares_protocol_and_non_destructive_boundary(self) -> None:
        source = (
            REPOSITORY_ROOT / "scripts/run_mecat_vanilla_table2_table3.sh"
        ).read_text(encoding="utf-8")
        for fragment in (
            "MODEL_CHECKPOINT_LOADED=no",
            "CAPTION_PROTOCOL=short_all",
            "STRICT_PAPER_REPRODUCTION=no",
            "DOWNLOADS_REQUIRED=no",
            "OVERWRITE_DELETE_RISK=none",
            "prepare_vanilla_mecat_manifest.py",
            "evaluate_mecat_table2_table3.py",
        ):
            self.assertIn(fragment, source)
        for forbidden in ("curl ", "wget ", "gdown ", "rm ", "tmux "):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
