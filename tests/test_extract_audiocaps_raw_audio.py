from __future__ import annotations

import tempfile
import unittest
import warnings
import zipfile
from argparse import Namespace
from pathlib import Path

from scripts.extract_audiocaps_raw_audio import (
    extract_archive,
    plan_members,
    safe_member_parts,
)


class ExtractAudioCapsRawAudioTests(unittest.TestCase):
    def args(self, archive: Path, output: Path) -> Namespace:
        return Namespace(
            archive=archive,
            output_dir=output,
            report=None,
            progress_files=1,
            minimum_free_margin_bytes=0,
            started_at="test",
        )

    def test_single_wrapper_directory_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "audio.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("audiocaps_raw_audio/test/a.wav", b"audio-a")
                handle.writestr("audiocaps_raw_audio/train/b.wav", b"audio-b")
            output = root / "audiocaps_raw_audio"
            report = extract_archive(self.args(archive, output))
            self.assertEqual(
                report["stripped_wrapper_name"],
                "audiocaps_raw_audio",
            )
            self.assertEqual(report["stripped_wrapper_levels"], 1)
            self.assertEqual((output / "test/a.wav").read_bytes(), b"audio-a")
            self.assertFalse((output / "audiocaps_raw_audio").exists())

    def test_archive_without_wrapper_keeps_top_level_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "audio.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("test/a.wav", b"audio-a")
                handle.writestr("train/b.wav", b"audio-b")
            output = root / "audiocaps_raw_audio"
            report = extract_archive(self.args(archive, output))
            self.assertIsNone(report["stripped_wrapper_name"])
            self.assertTrue((output / "test/a.wav").is_file())

    def test_multiple_duplicate_wrapper_levels_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "audio.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr(
                    "audiocaps_raw_audio/audiocaps_raw_audio/test/a.wav",
                    b"audio-a",
                )
            output = root / "audiocaps_raw_audio"
            report = extract_archive(self.args(archive, output))
            self.assertEqual(report["stripped_wrapper_levels"], 2)
            self.assertTrue((output / "test/a.wav").is_file())

    def test_existing_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "audio.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("test/a.wav", b"audio-a")
            output = root / "audiocaps_raw_audio"
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "refusing overwrite"):
                extract_archive(self.args(archive, output))

    def test_unsafe_paths_are_rejected(self) -> None:
        for value in ("../escape.wav", "/absolute.wav", "C:/drive.wav"):
            with self.assertRaisesRegex(ValueError, "unsafe ZIP member"):
                safe_member_parts(value)

    def test_flattening_collision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "collision.zip"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(archive, "w") as handle:
                    handle.writestr("wrapper/a.wav", b"one")
                    handle.writestr("wrapper/a.wav", b"two")
            with zipfile.ZipFile(archive) as handle:
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    plan_members(handle.infolist(), "wrapper")


if __name__ == "__main__":
    unittest.main()
