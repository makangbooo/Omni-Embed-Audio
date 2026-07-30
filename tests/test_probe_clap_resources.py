from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.audit_baseline_readiness import DEFAULT_REGISTRY, load_registry
from scripts.probe_clap_resources import (
    CLAP_MODEL_IDS,
    REPOSITORY_ROOT,
    probe_registry,
    write_new_json,
)


class ClapResourceProbeTests(unittest.TestCase):
    def test_exactly_four_clap_models_are_covered(self) -> None:
        registry = load_registry(DEFAULT_REGISTRY)
        with mock.patch(
            "scripts.probe_clap_resources.inspect_laion_package",
            return_value={"installed": False},
        ):
            report = probe_registry(registry, REPOSITORY_ROOT)
        self.assertEqual(
            [model["model_id"] for model in report["models"]],
            list(CLAP_MODEL_IDS),
        )
        self.assertEqual(report["summary"]["models"], 4)
        self.assertEqual(report["summary"]["formal_reproduction_authorized"], 0)

    def test_probe_is_local_read_only_and_does_not_load_models(self) -> None:
        source = (REPOSITORY_ROOT / "scripts/probe_clap_resources.py").read_text(
            encoding="utf-8"
        )
        for fragment in (
            "requests",
            "urllib",
            "snapshot_download",
            "hf_hub_download",
            "torch.load",
            "from_pretrained",
            "rm -rf",
            "shutil.rmtree",
        ):
            self.assertNotIn(fragment, source)
        self.assertIn('with path.open("x"', source)

    def test_new_json_refuses_to_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "probe.json"
            write_new_json(output, {"status": "complete"})
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))["status"],
                "complete",
            )
            with self.assertRaises(FileExistsError):
                write_new_json(output, {"status": "replaced"})

    def test_wrapper_has_compact_mandatory_summary(self) -> None:
        source = (REPOSITORY_ROOT / "scripts/run_clap_resource_probe.sh").read_text(
            encoding="utf-8"
        )
        for marker in (
            "EXPERIMENT_NAME=",
            "GIT_COMMIT=",
            "MODEL=",
            "DATASET=none",
            "GPU_USED=no",
            "OEA_OFFICIAL_SOURCE_USED=no",
            "ESTIMATED_TOTAL_TIME=",
            "FINAL_RUN_RC=",
            "COMPLETION_STATUS=",
            "METRICS_PATH=",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, source)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', source)
        self.assertNotIn("tmux", source)

        probe_source = (
            REPOSITORY_ROOT / "scripts/probe_clap_resources.py"
        ).read_text(encoding="utf-8")
        for marker in ("LAION_PACKAGE ", "EXECUTION_CANDIDATES=", "git_head="):
            with self.subTest(probe_marker=marker):
                self.assertIn(marker, probe_source)


if __name__ == "__main__":
    unittest.main()
