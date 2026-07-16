from __future__ import annotations

import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ShellEnvironmentGuardsTest(unittest.TestCase):
    def test_conda_environment_checks_do_not_use_quiet_grep_with_pipefail(self) -> None:
        affected_scripts = (
            "download_model01.sh",
            "download_model02.sh",
            "download_data01_clotho_evaluation.sh",
            "resume_environment.sh",
            "setup_environment.sh",
            "validate_gpu_environment.sh",
        )
        for filename in affected_scripts:
            with self.subTest(script=filename):
                source = (REPOSITORY_ROOT / "scripts" / filename).read_text(
                    encoding="utf-8"
                )
                self.assertNotIn("grep -Fxq", source)
                self.assertIn('grep -Fx "${ENV_NAME}" >/dev/null', source)


if __name__ == "__main__":
    unittest.main()
