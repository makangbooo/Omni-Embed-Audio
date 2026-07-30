# -*- coding: utf-8 -*-
import hashlib
import pickle
import warnings
from pathlib import Path

import numpy as np
from typing import List, Optional
from AudioRetrieval.eval_core import BaseRetrievalModel, l2norm
from AudioRetrieval.models.laion_clap_tokenizers import local_tokenizer_redirect


_TRUSTED_LAION_CLAP_CHECKPOINT_SHA256 = (
    "8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_official_state_dict(state_dict):
    """Mirror laion-clap 1.1.6 checkpoint key normalization."""
    if state_dict and next(iter(state_dict)).startswith("module"):
        state_dict = {key[7:]: value for key, value in state_dict.items()}

    import transformers

    if transformers.__version__ >= "4.31.0":
        state_dict.pop("text_branch.embeddings.position_ids", None)
    return state_dict


def _load_trusted_checkpoint_compat(clap_module, checkpoint_path) -> None:
    """Load the pinned official checkpoint under PyTorch 2.6+ semantics."""
    path = Path(checkpoint_path).expanduser().resolve()
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != _TRUSTED_LAION_CLAP_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Refusing unsafe LAION-CLAP compatibility load: checkpoint SHA256 "
            f"{actual_sha256} does not match the pinned official artifact"
        )

    import torch

    warnings.warn(
        "Using the PyTorch 2.6 compatibility loader for the SHA256-verified "
        "official LAION-CLAP checkpoint.",
        RuntimeWarning,
    )
    checkpoint = torch.load(
        str(path), map_location="cpu", weights_only=False
    )
    state_dict = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint
        else checkpoint
    )
    clap_module.model.load_state_dict(_normalize_official_state_dict(state_dict))


def _load_checkpoint(clap_module, checkpoint_path) -> None:
    try:
        clap_module.load_ckpt(checkpoint_path)
    except pickle.UnpicklingError as exc:
        if "Weights only load failed" not in str(exc):
            raise
        _load_trusted_checkpoint_compat(clap_module, checkpoint_path)

def _safe_import_sound_loader():
    try:
        import soundfile as sf, numpy as _np
        def _sf_loader(path, target_sr):
            wav, sr = sf.read(path)
            if wav.ndim > 1: wav = wav.mean(axis=1)
            if sr != target_sr:
                try:
                    import librosa; wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best")
                except Exception:
                    try:
                        import torchaudio, torch
                        t = torch.tensor(wav).float().unsqueeze(0)
                        wav = torchaudio.functional.resample(t, sr, target_sr).squeeze(0).numpy()
                    except Exception:
                        raise RuntimeError("Resampling failed: please install librosa or torchaudio.")
            return _np.asarray(wav, dtype=_np.float32)
        return _sf_loader
    except Exception: pass
    try:
        import librosa, numpy as _np
        def _lb_loader(path, target_sr):
            wav, _ = librosa.load(path, sr=target_sr, mono=True)
            return _np.asarray(wav, dtype=_np.float32)
        return _lb_loader
    except Exception: pass
    try:
        import torchaudio, torch
        def _ta_loader(path, target_sr):
            wav, sr = torchaudio.load(path)
            wav = wav.mean(dim=0, keepdim=False)
            if sr != target_sr: wav = torchaudio.functional.resample(wav, sr, target_sr)
            return wav.numpy().astype(np.float32)
        return _ta_loader
    except Exception: pass
    raise RuntimeError("Please install one of: soundfile, librosa, or torchaudio to load audio.")

def _center_crop_or_pad(wav: np.ndarray, target_len: int) -> np.ndarray:
    T = wav.shape[0]
    if T == target_len: return wav
    if T > target_len:
        start = (T - target_len) // 2
        return wav[start:start+target_len]
    pad_total = target_len - T
    left = pad_total // 2
    right = pad_total - left
    return np.pad(wav, (left, right), mode="constant")


class LaionClapAdapter(BaseRetrievalModel):
    def __init__(self, ckpt_path=None, amodel='HTSAT-tiny', tmodel='roberta',
                 enable_fusion=False, resample_sr=48000, audio_duration_sec=10.0,
                 audio_crop="center", tokenizer_paths=None):
        with local_tokenizer_redirect(tokenizer_paths):
            from laion_clap import CLAP_Module
            self.model = CLAP_Module(
                enable_fusion=enable_fusion, amodel=amodel, tmodel=tmodel
            )
        _load_checkpoint(self.model, ckpt_path)
        self.model.eval()
        self.target_sr = int(resample_sr)
        self.target_len = int(self.target_sr * float(audio_duration_sec))
        self.audio_crop = audio_crop
        self._loader = _safe_import_sound_loader()

    def encode_audio(self, paths: List[str], batch_size: int = 64, device: str = "cuda"):
        try:
            import torch; self.model.to(device)
        except Exception: pass
        embs = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i+batch_size]
            wavs = []
            for p in batch_paths:
                w = self._loader(p, self.target_sr)
                w = _center_crop_or_pad(w, self.target_len)
                wavs.append(w)
            x = np.stack(wavs, axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                e = self.model.get_audio_embedding_from_data(x, use_tensor=False)
            embs.append(np.asarray(e, dtype=np.float32))
        return l2norm(np.concatenate(embs, axis=0))

    def encode_text(self, texts: List[str], batch_size: int = 256, device: str = "cuda"):
        try:
            import torch; self.model.to(device)
        except Exception: pass
        embs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i+batch_size]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                e = self.model.get_text_embedding(chunk, use_tensor=False)
            embs.append(np.asarray(e, dtype=np.float32))
        return l2norm(np.concatenate(embs, axis=0))
