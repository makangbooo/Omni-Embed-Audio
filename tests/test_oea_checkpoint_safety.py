from __future__ import annotations

import unittest
from pathlib import PosixPath

from scripts.oea_checkpoint_safety import resolve_posix_path_safe_globals


class OEACheckpointSafetyTest(unittest.TestCase):
    def test_accepts_both_exact_posix_path_serialization_names(self) -> None:
        approved, unexpected, aliases = resolve_posix_path_safe_globals(
            ["pathlib._local.PosixPath", "pathlib.PosixPath"]
        )
        self.assertEqual(
            approved,
            ["pathlib.PosixPath", "pathlib._local.PosixPath"],
        )
        self.assertEqual(unexpected, [])
        self.assertEqual(
            aliases,
            [
                (PosixPath, "pathlib.PosixPath"),
                (PosixPath, "pathlib._local.PosixPath"),
            ],
        )

    def test_rejects_every_other_pickle_global(self) -> None:
        approved, unexpected, aliases = resolve_posix_path_safe_globals(
            ["builtins.eval", "pathlib._local.PosixPath"]
        )
        self.assertEqual(approved, ["pathlib._local.PosixPath"])
        self.assertEqual(unexpected, ["builtins.eval"])
        self.assertEqual(
            aliases,
            [(PosixPath, "pathlib._local.PosixPath")],
        )


if __name__ == "__main__":
    unittest.main()
