# -*- coding: utf-8 -*-
import warnings, numpy as np
from typing import List, Optional
from AudioRetrieval.eval_core import BaseRetrievalModel, l2norm

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
                 enable_fusion=False, resample_sr=48000, audio_duration_sec=10.0, audio_crop="center"):
        from laion_clap import CLAP_Module
        self.model = CLAP_Module(enable_fusion=enable_fusion, amodel=amodel, tmodel=tmodel)
        self.model.load_ckpt(ckpt_path); self.model.eval()
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
