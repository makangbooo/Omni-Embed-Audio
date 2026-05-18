# -*- coding: utf-8 -*-
"""Adapter for the linguistic robust-CLAP checkpoint (fusion) we downloaded locally.

IMPORTANT FIXES (2024-12):
1. Custom torchlibrosa: The checkpoint was trained with an old torchlibrosa version
   that used conv-based STFT keys (stft.conv_real.weight, stft.conv_imag.weight, melW).
   Modern torchlibrosa uses transform.window and mel.fb. We inject our custom
   torchlibrosa/stft.py before importing laion_clap to match checkpoint keys.

2. RoBERTa tokenizer: The hook.py in robust-clap uses T5Tokenizer but the text model
   is RoBERTa. This produces completely wrong token IDs. We replace with RobertaTokenizer.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np

from AudioRetrieval.eval_core import BaseRetrievalModel, l2norm
from AudioRetrieval.models.laion_clap_adapter import _center_crop_or_pad, _safe_import_sound_loader

# Default to the checked-out repo if present (same level as this project root).
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2] / "_linguistic_robust_clap-master"
# Custom torchlibrosa with old key format (conv-based STFT)
CUSTOM_TORCHLIBROSA_ROOT = Path(__file__).resolve().parents[2]


def _inject_custom_torchlibrosa():
    """Inject custom torchlibrosa BEFORE any laion_clap imports.

    The robust-clap checkpoint was trained with an old torchlibrosa that used:
    - stft.conv_real.weight, stft.conv_imag.weight (conv-based STFT)
    - melW (mel filter bank)

    Modern torchlibrosa uses:
    - transform.window (transform-based STFT)
    - mel.fb (mel filter bank)

    Our custom torchlibrosa/stft.py matches the checkpoint's key structure.
    """
    custom_path = str(CUSTOM_TORCHLIBROSA_ROOT.resolve())
    torchlibrosa_path = CUSTOM_TORCHLIBROSA_ROOT / "torchlibrosa"

    if torchlibrosa_path.exists() and custom_path not in sys.path:
        # Insert at the very beginning to override installed torchlibrosa
        sys.path.insert(0, custom_path)


def _maybe_add_repo_to_path(repo_root: Optional[str | Path]) -> Optional[str]:
    """Prepend the robust-clap repo (or its src dir) to sys.path if it exists."""
    if repo_root is None:
        return None
    root = Path(repo_root).expanduser()
    for candidate in (root / "src", root):
        if candidate.exists() and candidate.is_dir():
            path_str = str(candidate.resolve())
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            return path_str
    return None


def _create_roberta_tokenizer():
    """Create proper RoBERTa tokenizer (instead of T5 used in buggy hook.py)."""
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def tokenize_fn(texts):
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
    return tokenize_fn


class RobustClapAdapter(BaseRetrievalModel):
    """
    Thin wrapper around the robust CLAP checkpoint; mirrors the LaionClapAdapter API.

    IMPORTANT: This adapter includes critical fixes for the robust-CLAP checkpoint:
    1. Custom torchlibrosa with matching key names (conv-based STFT)
    2. RoBERTa tokenizer instead of T5 (bug fix for hook.py)

    Without these fixes, retrieval R@1 is ~0-2% instead of expected 25-30%.
    """

    def __init__(
        self,
        ckpt_path: str,
        amodel: str = "HTSAT-tiny",
        tmodel: str = "roberta",
        enable_fusion: bool = False,  # Changed default: use non-fusion for 630k-audioset-best.pt
        repo_root: Optional[str | Path] = None,
        resample_sr: int = 48000,
        audio_duration_sec: float = 10.0,
        audio_crop: str = "center",
    ):
        # FIX 1: Inject custom torchlibrosa BEFORE importing laion_clap
        # This ensures checkpoint keys match (stft.conv_real.weight vs transform.window)
        _inject_custom_torchlibrosa()

        # Prefer the local robust-clap repo if present; otherwise fall back to any installed laion_clap.
        repo_root = repo_root or (DEFAULT_REPO_ROOT if DEFAULT_REPO_ROOT.exists() else None)
        self.repo_path = _maybe_add_repo_to_path(repo_root)

        try:
            from laion_clap import CLAP_Module
        except ImportError as exc:
            raise ImportError(
                "Could not import laion_clap for robust-CLAP. "
                "Install dependencies or point repo_root to _linguistic_robust_clap-master."
            ) from exc

        self.model = CLAP_Module(enable_fusion=enable_fusion, amodel=amodel, tmodel=tmodel)

        # FIX 2: Replace T5 tokenizer with RoBERTa tokenizer
        # The hook.py uses T5Tokenizer but text model is RoBERTa - completely wrong token IDs!
        self.model.tokenizer = _create_roberta_tokenizer()

        # Load checkpoint with relaxed strictness to accommodate key mismatches between fusion/non-fusion variants.
        try:
            from laion_clap.clap_module.factory import load_state_dict as _robust_load_state_dict
            state = _robust_load_state_dict(ckpt_path, skip_params=True)
            # Note: self.model is CLAP_Module, self.model.model is the actual CLAP model
            result = self.model.model.load_state_dict(state, strict=False)
            if result.missing_keys or result.unexpected_keys:
                print(f"[RobustClapAdapter] Load state dict: "
                      f"missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}")
        except Exception:
            # Fallback to the module's own loader
            self.model.load_ckpt(ckpt_path)
        self.model.eval()

        self.target_sr = int(resample_sr)
        self.target_len = int(self.target_sr * float(audio_duration_sec))
        self.audio_crop = audio_crop
        self._loader = _safe_import_sound_loader()

    def encode_audio(self, paths: List[str], batch_size: int = 64, device: str = "cuda"):
        try:
            import torch

            self.model.to(device)
        except Exception:
            pass

        # Use get_audio_embedding_from_filelist to let CLAP handle audio preprocessing
        # This avoids double-preprocessing (pre-cropping + internal random truncation)
        embs = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                e = self.model.get_audio_embedding_from_filelist(batch_paths, use_tensor=False)
            embs.append(np.asarray(e, dtype=np.float32))
        return l2norm(np.concatenate(embs, axis=0))

    def encode_text(self, texts: List[str], batch_size: int = 256, device: str = "cuda"):
        try:
            import torch

            self.model.to(device)
        except Exception:
            pass
        embs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                e = self.model.get_text_embedding(chunk, use_tensor=False)
            embs.append(np.asarray(e, dtype=np.float32))
        return l2norm(np.concatenate(embs, axis=0))
