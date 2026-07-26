import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "run_asrur_cpu.py"


class RunASRURCPUTest(unittest.TestCase):
    def test_plan_and_synthetic_smoke_are_cpu_only_and_auditable(self) -> None:
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = ""
        plan = subprocess.run(
            [sys.executable, str(SCRIPT), "--stage", "plan"],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        self.assertEqual(json.loads(plan.stdout)["ablation_count"], 9)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--stage",
                    "synthetic_smoke",
                    "--output-dir",
                    str(output),
                ],
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            self.assertIn("synthetic_smoke_only", completed.stdout)
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(
                metrics["status"],
                "synthetic_smoke_only_not_a_research_result",
            )
            self.assertTrue((output / "gate_models.json").is_file())
            self.assertTrue((output / "run_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
