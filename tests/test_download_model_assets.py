from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


fake_huggingface_hub = types.ModuleType("huggingface_hub")
fake_huggingface_hub.HfApi = object
fake_huggingface_hub.__version__ = "test"
fake_huggingface_hub.snapshot_download = lambda **_: None

with patch.dict(sys.modules, {"huggingface_hub": fake_huggingface_hub}):
    from scripts.download_model_assets import (
        REPOSITORY_ROOT,
        ensure_external_model_root,
        local_files,
        selected,
        sha256_file,
    )


class DownloadModelAssetsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
