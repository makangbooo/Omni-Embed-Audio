"""Adapter for the controlled Robust-CLAP standard-checkpoint binding."""

from __future__ import annotations

import contextlib
import hashlib
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np

from AudioRetrieval.eval_core import BaseRetrievalModel, l2norm


TRUSTED_CHECKPOINT_SHA256 = (
    "8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _add_repo_to_path(repo_root: str | Path) -> str:
    root = Path(repo_root).expanduser().resolve()
    for candidate in (root / "src", root):
        if candidate.is_dir():
            value = str(candidate)
            if value not in sys.path:
                sys.path.insert(0, value)
            return value
    raise FileNotFoundError(root)


def _create_roberta_tokenizer(tokenizer_path: str | Path):
    from transformers import RobertaTokenizer

    tokenizer = RobertaTokenizer.from_pretrained(
        str(Path(tokenizer_path).expanduser().resolve()), local_files_only=True
    )

    def tokenize_fn(texts):
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )

    return tokenize_fn


@contextlib.contextmanager
def _local_robust_tokenizer_redirect(
    bert_tokenizer_path: str | Path,
    roberta_tokenizer_path: str | Path,
    bart_tokenizer_path: str | Path,
):
    """Keep all eager upstream tokenizer construction offline and pinned."""
    from transformers import RobertaTokenizer, T5Tokenizer

    roberta_path = Path(roberta_tokenizer_path).expanduser().resolve()
    local_roberta = RobertaTokenizer.from_pretrained(
        str(roberta_path), local_files_only=True
    )
    tokenizer_paths = {
        "bert-base-uncased": str(Path(bert_tokenizer_path).expanduser().resolve()),
        "roberta-base": str(roberta_path),
        "facebook/bart-base": str(Path(bart_tokenizer_path).expanduser().resolve()),
    }

    from AudioRetrieval.models.laion_clap_tokenizers import local_tokenizer_redirect

    sentinel = object()
    previous = T5Tokenizer.__dict__.get("from_pretrained", sentinel)

    def redirect_t5(requested, *args, **kwargs):
        if requested != "google/flan-t5-large":
            raise RuntimeError(
                f"unexpected Robust-CLAP T5 tokenizer request: {requested!r}"
            )
        return local_roberta

    with local_tokenizer_redirect(tokenizer_paths):
        setattr(T5Tokenizer, "from_pretrained", staticmethod(redirect_t5))
        try:
            yield local_roberta
        finally:
            if previous is sentinel:
                delattr(T5Tokenizer, "from_pretrained")
            else:
                setattr(T5Tokenizer, "from_pretrained", previous)


class RobustClapAdapter(BaseRetrievalModel):
    """Load the pinned upstream source with an explicit standard checkpoint."""

    def __init__(
        self,
        ckpt_path: str,
        amodel: str = "HTSAT-tiny",
        tmodel: str = "roberta",
        enable_fusion: bool = False,
        repo_root: Optional[str | Path] = None,
        bert_tokenizer_path: Optional[str | Path] = None,
        roberta_tokenizer_path: Optional[str | Path] = None,
        bart_tokenizer_path: Optional[str | Path] = None,
    ) -> None:
        if repo_root is None:
            raise ValueError("Robust-CLAP requires a pinned upstream source tree")
        if not all(
            (bert_tokenizer_path, roberta_tokenizer_path, bart_tokenizer_path)
        ):
            raise ValueError(
                "Robust-CLAP requires pinned local BERT, RoBERTa, and BART "
                "tokenizers"
            )
        self.repo_path = _add_repo_to_path(repo_root)

        with _local_robust_tokenizer_redirect(
            bert_tokenizer_path,
            roberta_tokenizer_path,
            bart_tokenizer_path,
        ) as local_roberta:
            from laion_clap import CLAP_Module

            self.model = CLAP_Module(
                enable_fusion=enable_fusion, amodel=amodel, tmodel=tmodel
            )
        self.model.tokenize = local_roberta
        self.model.tokenizer = _create_roberta_tokenizer(roberta_tokenizer_path)

        checkpoint = Path(ckpt_path).expanduser().resolve()
        if _sha256_file(checkpoint) != TRUSTED_CHECKPOINT_SHA256:
            raise RuntimeError("Robust-CLAP checkpoint differs from the pinned artifact")
        from AudioRetrieval.models.laion_clap_adapter import _load_checkpoint

        _load_checkpoint(self.model, checkpoint)
        self.model.eval()

    def encode_audio(
        self, paths: List[str], batch_size: int = 64, device: str = "cuda"
    ) -> np.ndarray:
        self.model.to(device)
        embeddings = []
        for index in range(0, len(paths), batch_size):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                value = self.model.get_audio_embedding_from_filelist(
                    paths[index : index + batch_size], use_tensor=False
                )
            embeddings.append(np.asarray(value, dtype=np.float32))
        return l2norm(np.concatenate(embeddings, axis=0))

    def encode_text(
        self, texts: List[str], batch_size: int = 256, device: str = "cuda"
    ) -> np.ndarray:
        self.model.to(device)
        embeddings = []
        for index in range(0, len(texts), batch_size):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                value = self.model.get_text_embedding(
                    texts[index : index + batch_size], use_tensor=False
                )
            embeddings.append(np.asarray(value, dtype=np.float32))
        return l2norm(np.concatenate(embeddings, axis=0))
