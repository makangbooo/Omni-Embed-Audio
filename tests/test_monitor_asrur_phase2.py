from __future__ import annotations

import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class MonitorAsrurPhase2Test(unittest.TestCase):
    def test_monitor_is_read_only_and_supports_live_and_once_modes(self) -> None:
        source = (
            REPOSITORY_ROOT / "scripts/monitor_asrur_phase2.sh"
        ).read_text(encoding="utf-8")
        for required in (
            "--run-dir",
            "--pid",
            "--interval",
            "--once",
            "wrapper_exit_code.txt",
            "completion_manifest.json",
            "latest_progress=",
            "nvidia-smi",
            "Press Ctrl-C",
        ):
            self.assertIn(required, source)
        for forbidden in (
            "nohup",
            "kill -9",
            "rm -rf",
            "git pull",
            "git checkout",
        ):
            self.assertNotIn(forbidden, source)

    def test_background_execution_policy_requires_tmux(self) -> None:
        plan = (
            REPOSITORY_ROOT
            / "docs/asr_uncertainty_reranking/EXPERIMENT_PLAN.md"
        ).read_text(encoding="utf-8")
        self.assertIn("tmux new-session -d -s", plan)
        self.assertIn("禁止使用 `nohup`", plan)
        self.assertIn("中断、重启或丢弃缓存", plan)


if __name__ == "__main__":
    unittest.main()
