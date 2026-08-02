from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_audiocaps_raw_audio import (
    index_audio_files,
    resolve_audio_path,
    write_jsonl_idempotent,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPOSITORY_ROOT / "scripts/run_audiocaps_raw_audio_validation.sh"


class ValidateAudioCapsRawAudioTests(unittest.TestCase):
    def test_runner_is_foreground_cpu_only_and_non_destructive(self) -> None:
        runner = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("tmux", runner)
        self.assertNotIn("curl ", runner)
        self.assertNotIn("wget ", runner)
        self.assertNotIn("rm ", runner)
        self.assertIn("GPU_USED=no", runner)
        self.assertIn("OEA_OFFICIAL_SOURCE_USED=no", runner)
        self.assertIn("expected 975 audio and 4,875 captions", runner)
        self.assertIn("OVERWRITE_DELETE_RISK=none", runner)

    def test_recursive_index_and_exact_filename_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "test" / "abcdefghijk_10.0.wav"
            audio.parent.mkdir()
            audio.write_bytes(b"wav")
            index, inventory = index_audio_files(root)
            selected, aliases = resolve_audio_path(
                {"expected_audio_basename": audio.name}, index
            )
            self.assertEqual(selected, audio.resolve())
            self.assertEqual(aliases, [])
            self.assertEqual(inventory["total_files"], 1)

    def test_missing_expected_audio_is_rejected(self) -> None:
        with self.assertRaises(FileNotFoundError):
            resolve_audio_path(
                {"expected_audio_basename": "missing_0.0.wav"}, {}
            )

    def test_nonidentical_duplicate_stems_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a" / "sample_0.0.wav"
            second = root / "b" / "sample_0.0.wav"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"one")
            second.write_bytes(b"two")
            index, _ = index_audio_files(root)
            with self.assertRaisesRegex(ValueError, "ambiguous non-identical"):
                resolve_audio_path(
                    {"expected_audio_basename": "sample_0.0.wav"}, index
                )

    def test_identical_duplicate_stems_are_audited(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a" / "sample_0.0.wav"
            second = root / "b" / "sample_0.0.wav"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"same")
            second.write_bytes(b"same")
            index, _ = index_audio_files(root)
            selected, aliases = resolve_audio_path(
                {"expected_audio_basename": "sample_0.0.wav"}, index
            )
            self.assertEqual(selected.name, "sample_0.0.wav")
            self.assertEqual(len(aliases), 1)

    def test_output_manifest_is_idempotent_but_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            rows = [{"sample_id": "a", "captions": ["one"]}]
            first = write_jsonl_idempotent(path, rows)
            second = write_jsonl_idempotent(path, rows)
            self.assertEqual(first, second)
            with self.assertRaisesRegex(FileExistsError, "differs"):
                write_jsonl_idempotent(path, [{"sample_id": "b"}])
            self.assertEqual(json.loads(path.read_text()), rows[0])


if __name__ == "__main__":
    unittest.main()
