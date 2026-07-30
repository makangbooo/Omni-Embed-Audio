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
WRAPPER = REPOSITORY_ROOT / "scripts/run_remaining_paper_model_downloads_tmux.sh"


class RemainingPaperModelDownloadsTests(unittest.TestCase):
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
            "tmux new-session -d",
            "PARALLEL_DOWNLOADS=6",
            "--continue-at -",
            "snapshot_download(",
            "python -m gdown",
            "ROBUST_CHECKPOINT_STATUS=blocked_not_published",
            "declare -A PIDS",
            "FINAL_RUN_RC=",
            "COMPLETION_STATUS=",
            "TMUX_RETAINED_SHELL=yes",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, source)
        for forbidden in ("rm -rf", "git reset", "git clean", "git checkout"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
