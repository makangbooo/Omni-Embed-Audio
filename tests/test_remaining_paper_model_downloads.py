from __future__ import annotations

import json
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPOSITORY_ROOT / "configs/resources/remaining_paper_models.json"
TOOLS_REQUIREMENTS = (
    REPOSITORY_ROOT
    / "configs/resources/paper_model_download_tools.requirements.txt"
)
WRAPPER = REPOSITORY_ROOT / "scripts/run_remaining_paper_model_downloads.sh"
CHECKER = REPOSITORY_ROOT / "scripts/check_remaining_paper_models.sh"
MGA_CHECKPOINT_WRAPPER = REPOSITORY_ROOT / "scripts/download_mga_clap_checkpoint.sh"


class RemainingPaperModelDownloadsTests(unittest.TestCase):
    def test_checker_is_read_only_and_covers_all_six_assets(self) -> None:
        source = CHECKER.read_text(encoding="utf-8")
        for asset in (
            "robust_source",
            "mga_source",
            "mga_checkpoint",
            "m2d_source",
            "m2d_checkpoint",
            "bge_snapshot",
        ):
            self.assertIn(asset, source)
        self.assertIn('echo "COMPLETE=${COMPLETE}/6"', source)
        for forbidden in ("tmux", "curl", "snapshot_download", "mv ", "rm "):
            self.assertNotIn(forbidden, source)

    def test_google_drive_tool_has_complete_hash_locked_runtime(self) -> None:
        source = TOOLS_REQUIREMENTS.read_text(encoding="utf-8")
        for distribution in ("beautifulsoup4", "gdown", "soupsieve"):
            self.assertRegex(
                source,
                rf"(?m)^{distribution}==[^ ]+ --hash=sha256:[0-9a-f]{{64}}$",
            )

    def test_manifest_pins_every_public_source(self) -> None:
        document = json.loads(MANIFEST.read_text(encoding="utf-8"))
        assets = document["assets"]
        self.assertEqual(len(assets), 6)
        for asset in assets:
            if asset["kind"] in {"source", "huggingface_snapshot"}:
                self.assertRegex(asset["revision"], r"^[0-9a-f]{40}$")
        m2d = next(asset for asset in assets if asset["kind"] == "checkpoint_archive")
        self.assertEqual(m2d["size_bytes"], 1469703420)
        self.assertEqual(
            m2d["sha256"],
            "fd193ae591720df7f1e27ed728ce127e0309b8bd427f0f4b3e5cd17d7ee5e1e1",
        )
        self.assertEqual(document["known_blockers"][0]["model"], "Robust-CLAP")

    def test_wrapper_is_parallel_resumable_and_auditable(self) -> None:
        source = WRAPPER.read_text(encoding="utf-8")
        for marker in (
            "PARALLEL_DOWNLOADS=6",
            "--continue-at -",
            "snapshot_download(",
            "python -m gdown",
            "ROBUST_CHECKPOINT_STATUS=blocked_not_published",
            "declare -A PIDS",
            "FINAL_RUN_RC=",
            "COMPLETION_STATUS=",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, source)
        for forbidden in (
            "rm -rf",
            "git reset",
            "git clean",
            "git checkout",
            "tmux",
            "exec bash -i",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_dedicated_mga_checkpoint_download_is_simple_and_non_destructive(self) -> None:
        source = MGA_CHECKPOINT_WRAPPER.read_text(encoding="utf-8")
        self.assertIn("python -m gdown", source)
        self.assertIn("MGA_CHECKPOINT_STATUS=complete", source)
        self.assertIn("checkpoint_sha256.txt", source)
        self.assertIn("failed_finalize", source)
        self.assertIn("failed_checkpoint_hash", source)
        for forbidden in ("tmux", "exec bash -i", "rm -rf", "git reset", "git clean"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
