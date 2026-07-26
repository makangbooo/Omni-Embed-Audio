#!/usr/bin/env python3
"""Exact safe-global policy for official OEA checkpoint metadata."""

from __future__ import annotations

from pathlib import PosixPath
from typing import Iterable


APPROVED_POSIX_PATH_GLOBAL_NAMES = frozenset(
    {
        "pathlib.PosixPath",
        "pathlib._local.PosixPath",
    }
)


def resolve_posix_path_safe_globals(
    names: Iterable[str],
) -> tuple[list[str], list[str], list[tuple[object, str]]]:
    """Return approved names, rejected names, and explicit torch aliases.

    Python 3.13 moved pathlib implementations into ``pathlib._local``. Some
    official checkpoints consequently store ``pathlib._local.PosixPath`` while
    older checkpoints store ``pathlib.PosixPath``. Both exact names represent
    the same inert path metadata class. No other pickle global is accepted.
    """

    observed = sorted(set(names))
    approved = sorted(set(observed) & APPROVED_POSIX_PATH_GLOBAL_NAMES)
    unexpected = sorted(set(observed) - APPROVED_POSIX_PATH_GLOBAL_NAMES)
    aliases = [(PosixPath, name) for name in approved]
    return approved, unexpected, aliases
