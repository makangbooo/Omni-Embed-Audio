from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from scripts.evaluate_mecat_table2_table3 import evaluate
from scripts.precompute_mecat_audio_embeddings import load_manifest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPOSITORY_ROOT / "scripts/run_mecat_positive_uiq.sh"
WRAPPER = REPOSITORY_ROOT / "scripts/run_mecat_table2_table3.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MecatCaptionManifestTests(unittest.TestCase):
    def test_short_all_is_loaded_in_canonical_grouped_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for sample_id in ("a", "b"):
                audio = root / f"{sample_id}.flac"
                audio.write_bytes(f"audio-{sample_id}".encode())
                rows.append(
                    {
                        "sample_id": sample_id,
                        "audio_path": str(audio),
                        "audio_size_bytes": audio.stat().st_size,
                        "audio_sha256": sha256(audio),
                        "decode_ok": True,
                        "file_exists": True,
                        "caption_fields": {
                            "short": [f"{sample_id}-1", f"{sample_id}-2", f"{sample_id}-3"]
                        },
                    }
                )
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            entries, _ = load_manifest(
                manifest,
                sha256(manifest),
                2,
                caption_field="short",
                captions_per_example=3,
            )
        self.assertEqual([entry.clip_id for entry in entries], ["a", "b"])
        self.assertEqual(entries[0].captions, ["a-1", "a-2", "a-3"])


class MecatTable2Table3EvaluatorTests(unittest.TestCase):
    def test_four_protocols_complete_with_explicit_public_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_dir = root / "audio"
            caption_dir = root / "captions"
            audio_dir.mkdir()
            caption_dir.mkdir()
            ids = np.asarray(["a", "b", "c"], dtype=object)
            audio = np.eye(3, dtype=np.float32)
            caption_ids = np.repeat(ids, 3)
            captions = np.repeat(audio, 3, axis=0)
            np.savez_compressed(
                audio_dir / "audio_embeddings.npz",
                embeddings=audio,
                clip_ids=ids,
            )
            np.savez_compressed(
                caption_dir / "caption_embeddings.npz",
                embeddings=captions,
                clip_ids=caption_ids,
            )
            report = evaluate(
                Namespace(
                    audio_embedding_dir=audio_dir,
                    caption_embedding_dir=caption_dir,
                    output_dir=root / "metrics",
                    model="synthetic",
                    expected_candidates=3,
                    paper_candidates=2,
                    captions_per_audio=3,
                    seed=0,
                )
            )
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(report["protocols"]), 4)
        self.assertEqual(report["caption_field"], "short")
        self.assertFalse(report["strict_paper_reproduction"])
        self.assertTrue(report["no_post_hoc_protocol_selection"])
        for protocol in report["protocols"]:
            self.assertEqual(protocol["metrics"]["R@1"], 100.0)

    def test_caption_ownership_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_dir = root / "audio"
            caption_dir = root / "captions"
            audio_dir.mkdir()
            caption_dir.mkdir()
            np.savez_compressed(
                audio_dir / "audio_embeddings.npz",
                embeddings=np.eye(2, dtype=np.float32),
                clip_ids=np.asarray(["a", "b"], dtype=object),
            )
            np.savez_compressed(
                caption_dir / "caption_embeddings.npz",
                embeddings=np.repeat(np.eye(2, dtype=np.float32), 3, axis=0),
                clip_ids=np.asarray(["a", "a", "a", "a", "b", "b"], dtype=object),
            )
            with self.assertRaisesRegex(ValueError, "exactly three"):
                evaluate(
                    Namespace(
                        audio_embedding_dir=audio_dir,
                        caption_embedding_dir=caption_dir,
                        output_dir=root / "metrics",
                        model="synthetic",
                        expected_candidates=2,
                        paper_candidates=1,
                        captions_per_audio=3,
                        seed=0,
                    )
                )


class MecatTable2Table3RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = RUNNER.read_text(encoding="utf-8")
        cls.wrapper = WRAPPER.read_text(encoding="utf-8")

    def test_runner_reuses_audio_and_fixes_short_protocol(self) -> None:
        for fragment in (
            'RUN_MODE="table2_table3"',
            "REUSED_AUDIO_RESULT",
            "--skip-audio --compute-captions",
            "--caption-field short --captions-per-example 3",
            "evaluate_mecat_table2_table3.py",
            "STRICT_PAPER_REPRODUCTION=no",
        ):
            self.assertIn(fragment, self.runner)
        self.assertIn("table2_table3", self.wrapper)

    def test_runner_has_no_download_or_background_session(self) -> None:
        for source in (self.runner, self.wrapper):
            for forbidden in ("tmux ", "curl ", "wget ", "gdown "):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
