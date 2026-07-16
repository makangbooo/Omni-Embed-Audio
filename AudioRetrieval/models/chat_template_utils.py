"""Small compatibility helpers for multimodal processor chat templates."""

from __future__ import annotations

from typing import Any


def normalize_single_chat_template_output(rendered: Any) -> str:
    """Normalize a processor result for one conversation to one string.

    Transformers 4.52.4's Qwen2.5-Omni processor wraps a single conversation
    as a batch before delegating to ``ProcessorMixin.apply_chat_template`` and
    therefore returns ``list[str]``.  Other processors return ``str`` for the
    same call.  The retrieval adapter processes one conversation at a time, so
    only a one-item string batch is valid here.
    """

    if isinstance(rendered, str):
        return rendered
    if (
        isinstance(rendered, list)
        and len(rendered) == 1
        and isinstance(rendered[0], str)
    ):
        return rendered[0]
    raise TypeError(
        "expected chat template output to be str or a one-item list[str], "
        f"got {type(rendered).__name__}: {rendered!r}"
    )
