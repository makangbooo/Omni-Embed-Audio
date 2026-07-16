from __future__ import annotations

import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class DataToolEnvironmentTest(unittest.TestCase):
    def test_environment_pins_modern_7zip(self) -> None:
        environment = (REPOSITORY_ROOT / "environment.yml").read_text(encoding="utf-8")
        self.assertIn("- 7zip=26.02", environment)

    def test_installer_defaults_to_tsinghua_conda_forge_mirror(self) -> None:
        installer = (REPOSITORY_ROOT / "scripts/install_data_tools.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge",
            installer,
        )
        self.assertIn('SEVEN_ZIP_SPEC="7zip=26.02"', installer)
        self.assertIn("[d]ownload_model_assets.py", installer)


if __name__ == "__main__":
    unittest.main()
