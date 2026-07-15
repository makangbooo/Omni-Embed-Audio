from __future__ import annotations

import unittest
import sys

from scripts.check_environment import REPOSITORY_ROOT, command_output


class CommandOutputTest(unittest.TestCase):
    def test_unavailable_command_is_recorded(self) -> None:
        result = command_output(["oea-command-that-does-not-exist"])
        self.assertTrue(result.startswith("UNAVAILABLE: "), result)

    def test_repository_root_is_available_to_script_imports(self) -> None:
        self.assertIn(str(REPOSITORY_ROOT), sys.path)


if __name__ == "__main__":
    unittest.main()
