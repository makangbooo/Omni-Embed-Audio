from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from scripts.evaluate_mecat_positive_uiq import evaluate
from scripts.precompute_mecat_audio_embeddings import load_manifest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPOSITORY_ROOT / "scripts/run_mecat_positive_uiq.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MecatManifestTests(unittest.TestCase):
    def create_manifest(self, root: Path, sample_ids: list[str]) -> Path:
        rows = []
        for sample_id in sample_ids:
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
                }
            )
        manifest = root / "manifest.jsonl"
        manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        return manifest

    def test_manifest_accepts_audited_sorted_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.create_manifest(Path(directory), ["a", "b"])
            entries, rows = load_manifest(manifest, sha256(manifest), 2)
            self.assertEqual([entry.clip_id for entry in entries], ["a", "b"])
            self.assertEqual(len(rows), 2)

    def test_manifest_rejects_identity_and_order_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.create_manifest(Path(directory), ["b", "a"])
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_manifest(manifest, "0" * 64, 2)
            with self.assertRaisesRegex(ValueError, "canonical sample_id order"):
                load_manifest(manifest, sha256(manifest), 2)

    def test_manifest_rejects_audio_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.create_manifest(root, ["a"])
            (root / "a.flac").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "audio SHA256 mismatch"):
                load_manifest(manifest, sha256(manifest), 1)

    def test_manifest_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.create_manifest(Path(directory), ["a", "a"])
            with self.assertRaisesRegex(ValueError, "not unique"):
                load_manifest(manifest, sha256(manifest), 2)


class MecatPositiveUiqEvaluatorTests(unittest.TestCase):
    def create_inputs(self, root: Path) -> tuple[Path, Path]:
        audio_dir = root / "audio"
        uiq_dir = root / "uiq"
        audio_dir.mkdir()
        uiq_dir.mkdir()
        ids = np.array(["a", "b", "c"], dtype=object)
        embeddings = np.eye(3, dtype=np.float32)
        np.savez_compressed(
            audio_dir / "audio_embeddings.npz",
            embeddings=embeddings,
            clip_ids=ids,
        )
        for query_type in ("question", "imperative", "paraphrase", "tagging"):
            np.savez_compressed(
                uiq_dir / f"uiq_{query_type}_embeddings.npz",
                embeddings=embeddings,
                clip_ids=ids,
            )
        return audio_dir, uiq_dir

    def args(self, root: Path, audio_dir: Path, uiq_dir: Path) -> Namespace:
        return Namespace(
            audio_embedding_dir=audio_dir,
            uiq_dir=uiq_dir,
            output_dir=root / "metrics",
            model="synthetic",
            expected_candidates=3,
            paper_candidates=2,
        )

    def test_four_protocols_complete_with_explicit_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_dir, uiq_dir = self.create_inputs(root)
            report = evaluate(self.args(root, audio_dir, uiq_dir))
            self.assertEqual(report["status"], "complete")
            self.assertEqual(len(report["protocols"]), 4)
            self.assertEqual(report["public_candidate_count"], 3)
            self.assertEqual(report["paper_candidate_count"], 2)
            self.assertFalse(report["strict_paper_reproduction"])
            for protocol in report["protocols"]:
                self.assertEqual(protocol["metrics"]["R@1"], 100.0)

    def test_query_id_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_dir, uiq_dir = self.create_inputs(root)
            path = uiq_dir / "uiq_question_embeddings.npz"
            np.savez_compressed(
                path,
                embeddings=np.eye(3, dtype=np.float32),
                clip_ids=np.array(["a", "b", "unknown"], dtype=object),
            )
            with self.assertRaisesRegex(ValueError, "target IDs differ"):
                evaluate(self.args(root, audio_dir, uiq_dir))

    def test_query_shape_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_dir, uiq_dir = self.create_inputs(root)
            path = uiq_dir / "uiq_question_embeddings.npz"
            np.savez_compressed(
                path,
                embeddings=np.eye(2, 3, dtype=np.float32),
                clip_ids=np.array(["a", "b"], dtype=object),
            )
            with self.assertRaisesRegex(ValueError, "unexpected question shape"):
                evaluate(self.args(root, audio_dir, uiq_dir))


class MecatPositiveUiqRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_runner_has_five_variants_and_protocol_boundary(self) -> None:
        for variant in (
            "laion_clap",
            "mga_clap",
            "m2d_clap",
            "oea_qwen7b",
            "oea_qwen7b_cl",
        ):
            self.assertIn(variant, self.source)
        for fragment in (
            "PUBLIC_CANDIDATES",
            "--expected-candidates 848",
            "--paper-candidates 847",
            "STRICT_PAPER_REPRODUCTION=no",
            "GPU_USED=yes",
            "OEA_OFFICIAL_SOURCE_USED=yes",
        ):
            self.assertIn(fragment, self.source)

    def test_runner_is_foreground_without_download_or_tmux(self) -> None:
        for forbidden in ("tmux ", "curl ", "wget ", "gdown "):
            self.assertNotIn(forbidden, self.source)
        self.assertIn("exec > >(tee -a", self.source)
        self.assertIn("DOWNLOADS_REQUIRED=no", self.source)
        self.assertIn("OVERWRITE_DELETE_RISK=none", self.source)


if __name__ == "__main__":
    unittest.main()
