from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.inspect_official_oea_model_presence import (
    REPOSITORY_ROOT,
    inspect_asset,
    load_assets,
    write_new_json,
)


class InspectOfficialOeaModelPresenceTests(unittest.TestCase):
    def test_loads_nine_unique_assets_for_six_variants(self) -> None:
        assets = load_assets(
            REPOSITORY_ROOT / "configs/checkpoints/official_oea_checkpoints.json"
        )
        self.assertEqual(len(assets), 9)
        self.assertEqual(
            {asset["name"] for asset in assets},
            {
                "omni_embed_nemotron_3b",
                "oea_nemo3b_ac",
                "oea_nemo3b_cl",
                "qwen2_5_omni_3b",
                "oea_qwen3b_ac",
                "oea_qwen3b_cl",
                "qwen2_5_omni_7b",
                "oea_qwen7b_ac",
                "oea_qwen7b_cl",
            },
        )

    def test_presence_candidate_requires_marker_required_files_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = {
                "name": "checkpoint",
                "repo_id": "owner/repo",
                "revision": "a" * 40,
                "local_subdir": "checkpoint",
                "required_files": ["README.md", "weights.pt"],
                "expected_primary_file": {
                    "path": "weights.pt",
                    "size_bytes": 4,
                    "lfs_sha256": "b" * 64,
                },
                "variant_consumers": ["variant"],
            }
            destination = root / "checkpoint"
            destination.mkdir()
            (destination / "README.md").write_text("readme", encoding="utf-8")
            (destination / "weights.pt").write_bytes(b"1234")
            marker = root / ".oea_asset_markers/checkpoint.json"
            marker.parent.mkdir()
            marker.write_text(
                json.dumps({"repo_id": "owner/repo", "revision": "a" * 40}),
                encoding="utf-8",
            )

            report = inspect_asset(root, asset)
            self.assertTrue(report["presence_candidate"])
            self.assertTrue(report["expected_primary_file"]["size_matches"])
            self.assertFalse(report["expected_primary_file"]["content_hash_checked"])
            self.assertIn("not checked", report["claim_boundary"])

            (destination / ".cache").mkdir()
            (destination / ".cache/weights.pt.incomplete").write_bytes(b"partial")
            report = inspect_asset(root, asset)
            self.assertFalse(report["presence_candidate"])
            self.assertEqual(report["incomplete_files"], [".cache/weights.pt.incomplete"])

    def test_output_is_non_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            write_new_json(output, {"status": "complete"})
            with self.assertRaises(FileExistsError):
                write_new_json(output, {"status": "different"})

    def test_direct_entrypoint_scans_without_network_or_model_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "report.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "scripts/inspect_official_oea_model_presence.py"),
                    "--model-root",
                    str(root / "models"),
                    "--output",
                    str(output),
                ],
                cwd=REPOSITORY_ROOT,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("MODEL_PRESENCE_SCAN_STATUS=complete", completed.stdout)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["asset_count"], 9)
            self.assertEqual(report["summary"]["presence_candidate_count"], 0)
            self.assertIn("no hashing, network, or downloads", report["operation"])

    def test_wrapper_is_cpu_only_short_and_non_destructive(self) -> None:
        wrapper = (
            REPOSITORY_ROOT / "scripts/run_official_oea_model_presence_scan.sh"
        ).read_text(encoding="utf-8")
        implementation = (
            REPOSITORY_ROOT / "scripts/inspect_official_oea_model_presence.py"
        ).read_text(encoding="utf-8")
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', wrapper)
        self.assertIn("Estimated time: 1-3 minutes", wrapper)
        self.assertIn("requires a clean Git worktree", wrapper)
        self.assertIn("no hashing, network, or downloads", implementation)
        for forbidden in ("snapshot_download", "sha256_file", "rm -rf", "unlink("):
            self.assertNotIn(forbidden, wrapper)
            self.assertNotIn(forbidden, implementation)


if __name__ == "__main__":
    unittest.main()
