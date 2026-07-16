from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


fake_huggingface_hub = types.ModuleType("huggingface_hub")
fake_huggingface_hub.HfApi = object
fake_huggingface_hub.__version__ = "test"
fake_huggingface_hub.snapshot_download = lambda **_: None

with patch.dict(sys.modules, {"huggingface_hub": fake_huggingface_hub}):
    import scripts.download_model_assets as download_module

    from scripts.download_model_assets import (
        REPOSITORY_ROOT,
        download_snapshot_with_retries,
        ensure_external_model_root,
        is_retryable_network_error,
        local_files,
        selected,
        sha256_file,
    )


class DownloadModelAssetsTest(unittest.TestCase):
    def test_chunked_encoding_error_is_retryable(self) -> None:
        chunked_error_type = type(
            "ChunkedEncodingError", (Exception,), {"__module__": "requests.exceptions"}
        )
        self.assertTrue(is_retryable_network_error(chunked_error_type("broken")))

    def test_snapshot_download_resumes_after_network_error(self) -> None:
        chunked_error_type = type(
            "ChunkedEncodingError", (Exception,), {"__module__": "requests.exceptions"}
        )
        report = {"status": "running", "assets": []}
        asset_report = {"download_attempts": []}
        report["assets"].append(asset_report)
        downloader = Mock(side_effect=[chunked_error_type("broken"), None])

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.json"
            with (
                patch.object(download_module, "snapshot_download", downloader),
                patch.object(download_module.time, "sleep") as sleep,
            ):
                download_snapshot_with_retries(
                    download_kwargs={"repo_id": "owner/model"},
                    asset_report=asset_report,
                    report=report,
                    output=output,
                    max_attempts=3,
                    backoff_seconds=30,
                )

        self.assertEqual(downloader.call_count, 2)
        sleep.assert_called_once_with(30)
        self.assertEqual(
            [attempt["status"] for attempt in asset_report["download_attempts"]],
            ["retryable_error", "complete"],
        )

    def test_model_root_inside_repository_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the Git repository"):
            ensure_external_model_root(REPOSITORY_ROOT / "models")

    def test_pattern_selection(self) -> None:
        self.assertTrue(selected("step_40.pt", ["*.pt"]))
        self.assertFalse(selected("config.json", ["*.pt"]))
        self.assertTrue(selected("any/file", None))

    def test_hash_and_local_file_inventory_skip_hf_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "step_40.pt"
            payload.write_bytes(b"checkpoint")
            cache_file = root / ".cache" / "huggingface" / "metadata"
            cache_file.parent.mkdir(parents=True)
            cache_file.write_text("ignored", encoding="utf-8")

            self.assertEqual(local_files(root), [payload])
            self.assertEqual(
                sha256_file(payload), hashlib.sha256(b"checkpoint").hexdigest()
            )

    def test_first_download_manifest_uses_immutable_revisions(self) -> None:
        manifest = REPOSITORY_ROOT / "configs/resources/model01_qwen3b_cl.json"
        specification = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(len(specification["assets"]), 2)
        for asset in specification["assets"]:
            revision = asset["revision"]
            self.assertEqual(len(revision), 40)
            int(revision, 16)

    def test_second_download_manifest_pins_qwen3b_ac_checkpoint(self) -> None:
        manifest = REPOSITORY_ROOT / "configs/resources/model02_qwen3b_ac.json"
        specification = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(len(specification["assets"]), 1)
        asset = specification["assets"][0]
        self.assertEqual(
            asset["revision"], "f6c4b3b86385fd7ecbe3bacf45548a2259af8db4"
        )
        self.assertEqual(
            asset["expected_primary_file"],
            {
                "path": "step_350.pt",
                "size_bytes": 9466835918,
                "lfs_sha256": "afb22d02e610016184fa0e2b4314fe0de191ebcc909184e9e128d36abade4b44",
            },
        )


if __name__ == "__main__":
    unittest.main()
