from __future__ import annotations

import unittest

from scripts.check_environment import command_output


class CommandOutputTest(unittest.TestCase):
    def test_unavailable_command_is_recorded(self) -> None:
        result = command_output(["oea-command-that-does-not-exist"])
        self.assertTrue(result.startswith("UNAVAILABLE: "), result)


if __name__ == "__main__":
    unittest.main()
