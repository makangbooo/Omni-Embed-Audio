"""Offline tokenizer redirection for the eager LAION-CLAP 1.1.6 import path."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Dict, Optional


@contextlib.contextmanager
def local_tokenizer_redirect(tokenizer_paths: Optional[Dict[str, str]]):
    """Redirect the three eager LAION-CLAP tokenizers to audited local trees."""
    if tokenizer_paths is None:
        yield
        return

    expected = {
        "bert-base-uncased": "BertTokenizer",
        "roberta-base": "RobertaTokenizer",
        "facebook/bart-base": "BartTokenizer",
    }
    if set(tokenizer_paths) != set(expected):
        raise ValueError(
            "tokenizer_paths must cover bert-base-uncased, roberta-base, and "
            "facebook/bart-base"
        )

    import transformers

    sentinel = object()
    previous = {}
    resolved_paths = {
        model_id: Path(tokenizer_paths[model_id]).expanduser().resolve()
        for model_id in expected
    }
    for local_path in resolved_paths.values():
        if not local_path.is_dir():
            raise FileNotFoundError(local_path)
    for model_id, class_name in expected.items():
        tokenizer_class = getattr(transformers, class_name)
        local_path = resolved_paths[model_id]
        previous[tokenizer_class] = tokenizer_class.__dict__.get(
            "from_pretrained", sentinel
        )
        original = tokenizer_class.from_pretrained

        def redirected(
            requested,
            *args,
            _model_id=model_id,
            _local_path=local_path,
            _original=original,
            **kwargs,
        ):
            if requested != _model_id:
                raise RuntimeError(
                    f"unexpected LAION-CLAP tokenizer request: {requested!r}"
                )
            kwargs["local_files_only"] = True
            return _original(str(_local_path), *args, **kwargs)

        setattr(tokenizer_class, "from_pretrained", staticmethod(redirected))

    try:
        yield
    finally:
        for tokenizer_class, old_value in previous.items():
            if old_value is sentinel:
                delattr(tokenizer_class, "from_pretrained")
            else:
                setattr(tokenizer_class, "from_pretrained", old_value)
