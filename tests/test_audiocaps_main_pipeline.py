from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from scripts.evaluate_audiocaps_main import evaluate
from scripts.precompute_audiocaps_embeddings import load_manifest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPOSITORY_ROOT / "scripts/run_audiocaps_main.sh"
VALIDATION_AUDIT = (
    REPOSITORY_ROOT
    / "results/audits/audiocaps_v2_test_audio_validation_20260802.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(
    root: Path,
    sample_ids: list[str],
    captions_per_audio: int,
) -> Path:
    rows = []
    for sample_id in sample_ids:
        audio = root / f"{sample_id}.wav"
        audio.write_bytes(f"audio-{sample_id}".encode())
        rows.append(
            {
                "sample_id": sample_id,
                "audio_path": str(audio),
                "audio_size_bytes": audio.stat().st_size,
                "audio_sha256": sha256(audio),
                "captions": [
                    f"caption {index} for {sample_id}"
                    for index in range(captions_per_audio)
                ],
                "decode_ok": True,
                "file_exists": True,
            }
        )
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return manifest


class AudioCapsMainManifestTests(unittest.TestCase):
    def test_remote_validation_evidence_authorizes_main_evaluation(self) -> None:
        audit = json.loads(VALIDATION_AUDIT.read_text(encoding="utf-8"))
        self.assertEqual(audit["status"], "complete")
        self.assertEqual(audit["run"]["final_run_rc"], 0)
        self.assertEqual(audit["test_selection"]["candidate_count"], 975)
        self.assertEqual(audit["test_selection"]["caption_count"], 4875)
        self.assertTrue(
            audit["test_selection"]["all_finite_and_fully_decoded"]
        )
        self.assertTrue(audit["claim_boundary"]["main_experiment_authorized"])
        self.assertEqual(
            audit["output_manifest"]["sha256"],
            "a341c9dfcb1b2cbe69675e8f1338edb502ce03f8984bf5882dc4d107b72144b4",
        )

    def test_manifest_accepts_bound_audio_and_five_captions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_manifest(root, ["a", "b"], 5)
            entries, rows = load_manifest(manifest, sha256(manifest), 2, 10)
            self.assertEqual([entry.clip_id for entry in entries], ["a", "b"])
            self.assertEqual(len(rows), 2)
            self.assertEqual(sum(len(entry.captions) for entry in entries), 10)

    def test_manifest_rejects_hash_order_and_caption_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_manifest(root, ["b", "a"], 5)
            with self.assertRaisesRegex(ValueError, "manifest SHA256 mismatch"):
                load_manifest(manifest, "0" * 64, 2, 10)
            with self.assertRaisesRegex(ValueError, "sorted and unique"):
                load_manifest(manifest, sha256(manifest), 2, 10)

            manifest = write_manifest(root, ["a", "b"], 5)
            (root / "a.wav").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "audio (size|SHA256) mismatch"):
                load_manifest(manifest, sha256(manifest), 2, 10)


class AudioCapsMainEvaluatorTests(unittest.TestCase):
    def create_inputs(self, root: Path) -> tuple[Path, Path, Path]:
        ids = ["a", "b", "c"]
        manifest = write_manifest(root, ids, 2)
        embedding_dir = root / "embeddings"
        uiq_dir = root / "uiq"
        embedding_dir.mkdir()
        uiq_dir.mkdir()
        basis = np.eye(3, dtype=np.float32)
        np.savez_compressed(
            embedding_dir / "audio_embeddings.npz",
            embeddings=basis,
            clip_ids=np.array(ids, dtype=object),
        )
        caption_ids = [sample_id for sample_id in ids for _ in range(2)]
        caption_embeddings = np.repeat(basis, 2, axis=0)
        np.savez_compressed(
            embedding_dir / "caption_embeddings.npz",
            embeddings=caption_embeddings,
            clip_ids=np.array(caption_ids, dtype=object),
            texts=np.array([f"text-{index}" for index in range(6)], dtype=object),
        )
        for query_type in ("question", "imperative", "paraphrase", "tagging"):
            np.savez_compressed(
                uiq_dir / f"uiq_{query_type}_embeddings.npz",
                embeddings=basis,
                clip_ids=np.array(ids, dtype=object),
            )
        return manifest, embedding_dir, uiq_dir

    def args(
        self,
        root: Path,
        manifest: Path,
        embedding_dir: Path,
        uiq_dir: Path,
    ) -> Namespace:
        return Namespace(
            embedding_dir=embedding_dir,
            uiq_dir=uiq_dir,
            manifest=manifest,
            manifest_sha256=sha256(manifest),
            output_dir=root / "metrics",
            model="synthetic",
            expected_candidates=3,
            captions_per_audio=2,
            seed=0,
        )

    def test_all_eight_protocols_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, embedding_dir, uiq_dir = self.create_inputs(root)
            report = evaluate(
                self.args(root, manifest, embedding_dir, uiq_dir)
            )
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["candidate_count"], 3)
            self.assertEqual(report["caption_count"], 6)
            self.assertEqual(len(report["protocols"]), 8)
            self.assertTrue(report["source_usage"]["oea_official_source_used"])
            for protocol in report["protocols"]:
                self.assertEqual(protocol["metrics"]["R@1"], 100.0)

    def test_uiq_id_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, embedding_dir, uiq_dir = self.create_inputs(root)
            np.savez_compressed(
                uiq_dir / "uiq_question_embeddings.npz",
                embeddings=np.eye(3, dtype=np.float32),
                clip_ids=np.array(["a", "b", "unknown"], dtype=object),
            )
            with self.assertRaisesRegex(ValueError, "IDs differ from candidates"):
                evaluate(self.args(root, manifest, embedding_dir, uiq_dir))


class AudioCapsMainRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = RUNNER.read_text(encoding="utf-8")

    def test_runner_has_five_models_and_all_main_tables(self) -> None:
        for variant in (
            "laion_clap",
            "mga_clap",
            "m2d_clap",
            "oea_qwen7b",
            "oea_qwen7b_cl",
        ):
            self.assertIn(variant, self.source)
        for fragment in (
            "EXP-10 Table 2",
            "EXP-11 Table 3",
            "EXP-12-15 positive UIQ",
            "--expected-examples 975",
            "--expected-captions 4875",
            "GPU_USED=yes",
            "OEA_OFFICIAL_SOURCE_USED=yes",
        ):
            self.assertIn(fragment, self.source)

    def test_runner_is_foreground_offline_and_non_destructive(self) -> None:
        for forbidden in ("tmux ", "curl ", "wget ", "gdown ", "rm "):
            self.assertNotIn(forbidden, self.source)
        self.assertIn("exec > >(tee -a", self.source)
        self.assertIn("DOWNLOADS_REQUIRED=no", self.source)
        self.assertIn("OVERWRITE_DELETE_RISK=none", self.source)
        self.assertIn("HF_HUB_OFFLINE=1", self.source)
        self.assertIn("artifact_sha256.txt", self.source)


if __name__ == "__main__":
    unittest.main()
