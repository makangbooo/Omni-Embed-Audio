# -*- coding: utf-8 -*-
from typing import List
import numpy as np

from eval_core import BaseRetrievalModel, l2norm
from models.laion_clap_adapter import _safe_import_sound_loader, _center_crop_or_pad  # reuse loaders

class GlapAdapter(BaseRetrievalModel):
    """
    GLAP adapter using the official glap_model API.
    - install: pip install glap_model
    - API: from glap_model import glap_inference; model.encode_text(list[str]); model.encode_audio(torch.Tensor)
      (See README: 'Embedding extraction' / 'Batched scoring'.)
    """
    def __init__(self, resample_sr: int = 16000, audio_duration_sec: float = 10.0):
        # The official examples show 10s=160k samples at 16kHz; we follow that default.
        # https://github.com/xiaomi-research/dasheng-glap (README)
        from glap_model import glap_inference  # installs from PyPI per README
        self.model = glap_inference()
        self.target_sr = int(resample_sr)
        self.target_len = int(self.target_sr * float(audio_duration_sec))
        self._loader = _safe_import_sound_loader()

    def encode_audio(self, paths: List[str], batch_size: int = 64, device: str = "cuda") -> np.ndarray:
        import torch
        embs = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i+batch_size]
            wavs = []
            for p in batch_paths:
                w = self._loader(p, self.target_sr)                      # mono float32 @16k
                w = _center_crop_or_pad(w, self.target_len)              # 10s center crop/pad
                wavs.append(w)
            x = torch.from_numpy(np.stack(wavs, axis=0))                 # [B,T]
            with torch.no_grad():
                e = self.model.encode_audio(x)                           # [B,D]
                e = e.detach().cpu().numpy().astype(np.float32)
            embs.append(e)
        return l2norm(np.concatenate(embs, axis=0))

    def encode_text(self, texts: List[str], batch_size: int = 256, device: str = "cuda") -> np.ndarray:
        import torch
        embs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i+batch_size]
            with torch.no_grad():
                e = self.model.encode_text(chunk)                        # [b,D]
                e = e.detach().cpu().numpy().astype(np.float32)
            embs.append(e)
        return l2norm(np.concatenate(embs, axis=0))
