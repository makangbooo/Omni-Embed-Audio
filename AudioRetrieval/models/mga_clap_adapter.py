# -*- coding: utf-8 -*-
# models/mga_clap_adapter.py
# Adapter for MGA-CLAP (PyTorch). Uses the repo's ASE model and codebook pooling.
# Repo shows: ASE(config); load ckpt; model.encode_text -> word_embeds (+attn_mask)
# then text_embeds = model.msc(word_embeds, model.codebook, attn_mask)
# and audio: model.encode_audio(wavs) -> frame_embeds; then msc -> clip_embeds.
# (See example.py in their repo.)  # :contentReference[oaicite:1]{index=1}

from __future__ import annotations
import importlib
import sys
import warnings
from pathlib import Path
from typing import List
import numpy as np
import hashlib

import torch

def _extract_state_dict(obj):
    """
    Accept a variety of checkpoint formats and return a plain state_dict (dict[str, Tensor]).
    """
    if isinstance(obj, dict):
        # common keys in various training scripts
        for k in ["state_dict", "model", "module", "model_state_dict", "ema_state_dict"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        # already a state_dict (all tensors)
        if all(torch.is_tensor(v) for v in obj.values()):
            return obj
        # Some trainers nest more deeply
        for k, v in obj.items():
            if isinstance(v, dict) and all(torch.is_tensor(x) for x in v.values()):
                return v
    # Fallback to .state_dict() if it's a module/Lightning wrapper
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    raise RuntimeError("Could not locate a state_dict in checkpoint")

def _strip_prefixes(state_dict, prefixes=("module.", "model.", "encoder.")):
    """
    Remove known DDP/Module prefixes if present.
    """
    if not state_dict:
        return state_dict
    # Detect by sampling a few keys
    keys = list(state_dict.keys())
    if not keys:
        return state_dict
    new_sd = {}
    for k, v in state_dict.items():
        new_key = k
        for p in prefixes:
            if new_key.startswith(p):
                new_key = new_key[len(p):]
        new_sd[new_key] = v
    return new_sd


def _resolve_device(requested: str) -> torch.device:
    """
    Resolve the desired torch.device, gracefully falling back to CPU when CUDA
    is requested but unavailable (common on sandboxed environments).
    """
    req = (requested or "cpu").lower()
    if req.startswith("cuda"):
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")
            # Touching device_count triggers initialization and surfaces driver issues.
            _ = torch.cuda.device_count()
            return torch.device(requested)
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            warnings.warn(
                f"Requested CUDA device '{requested}' but it is not usable ({exc}); falling back to CPU.",
                RuntimeWarning,
            )
            return torch.device("cpu")
    return torch.device("cpu")

def _checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mga_ckpt(ckpt_path: str | Path, device="cpu"):
    """
    Robust loader for MGA-CLAP checkpoints across safetensors / pickle formats,
    and across PyTorch 2.6+ safe load changes.
    """
    ckpt_path = Path(ckpt_path)
    # 1) Prefer safetensors if the provided file is .safetensors or a sibling exists
    try:
        if ckpt_path.suffix == ".safetensors":
            from safetensors.torch import load_file as safe_load_file
            sd = safe_load_file(str(ckpt_path), device=str(device))
            return _strip_prefixes(sd)
        # sibling named 'model.safetensors' or same stem
        for cand in [ckpt_path.with_suffix(".safetensors"),
                     ckpt_path.parent / "model.safetensors"]:
            if cand.exists():
                from safetensors.torch import load_file as safe_load_file
                sd = safe_load_file(str(cand), device=str(device))
                return _strip_prefixes(sd)
    except Exception:
        # fall through to torch.load path
        pass

    # 2) PyTorch 2.6+ safe load with allow-listed globals
    try:
        import numpy as np
        from torch.serialization import safe_globals
        with safe_globals([np.core.multiarray.scalar]):  # allowlist the needed NumPy scalar
            obj = torch.load(str(ckpt_path), map_location=device)  # weights_only=True default
        sd = _extract_state_dict(obj)
        return _strip_prefixes(sd)
    except Exception as e_safe:
        last_safe_err = e_safe

    # 3) As a last resort, allow full unpickling (only if you trust the source!)
    try:
        obj = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        sd = _extract_state_dict(obj)
        return _strip_prefixes(sd)
    except Exception as e_unpick:
        # Surface both errors to aid debugging
        raise RuntimeError(
            "Failed to load checkpoint with safe globals and with weights_only=False.\n"
            f"Safe-load error: {last_safe_err}\n"
            f"Unpickle-load error: {e_unpick}"
        )


# Robust YAML loader that works with ruamel.yaml (new API) or PyYAML
def _get_yaml_loader():
    try:
        # ruamel.yaml >= 0.18: no safe_load, use YAML(typ='safe')
        from ruamel.yaml import YAML  # type: ignore
        _yaml = YAML(typ='safe', pure=True)
        def _load(stream):
            return _yaml.load(stream)
        return _load
    except Exception:
        # Fallback to PyYAML (has safe_load)
        import yaml  # type: ignore
        def _load(stream):
            return yaml.safe_load(stream)
        return _load

yaml_load = _get_yaml_loader()

TARGET_SR = 32000  # repo example resamples to 32k via torchaudio  # :contentReference[oaicite:2]{index=2}

def _safe_import_sound_loader():
    # Prefer torchaudio since the official example uses it.  # :contentReference[oaicite:3]{index=3}
    try:
        import torchaudio, torch
        def _ta_loader(path, target_sr):
            wav, sr = torchaudio.load(path)  # [C, T]
            wav = wav.mean(dim=0)            # mono
            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)
            return wav
        return _ta_loader
    except Exception:
        pass
    try:
        import soundfile as sf, torch, librosa
        def _sf_loader(path, target_sr):
            wav, sr = sf.read(path)
            if wav.ndim > 1: wav = wav.mean(axis=1)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best")
            return torch.tensor(wav, dtype=torch.float32)
        return _sf_loader
    except Exception:
        pass
    raise RuntimeError("Install torchaudio (preferred) or soundfile+librosa to load audio.")

import contextlib


class MGAClapAdapter:
    """
    Minimal interface expected by eval_clotho: encode_text() / encode_audio().
    Requirements:
      - MGA_CLAP_REPO: path to cloned MGA-CLAP repo (contains models/ASE etc.)
      - MGA_CLAP_CKPT: path to the provided checkpoint (Google Drive link in README)
    README: checkpoint goes under pretrained_models/models; example demonstrates encode & pooling.  # :contentReference[oaicite:4]{index=4}
    """
    def __init__(
        self,
        repo_path: str,
        ckpt_path: str,
        seconds: float = 10.0,
        device: str = "cuda",
        amp: bool = False,
        bert_tokenizer_path: str | None = None,
        expected_checkpoint_sha256: str | None = None,
    ):
        self.repo_root = Path(repo_path).expanduser().resolve()
        self.ckpt_path = str(Path(ckpt_path).expanduser().resolve())
        if not Path(self.ckpt_path).exists():
            raise FileNotFoundError(f"MGA-CLAP checkpoint not found: {self.ckpt_path}")

        for required in (
            self.repo_root / "models/ase_model.py",
            self.repo_root / "settings/inference_example.yaml",
        ):
            if not required.is_file():
                raise FileNotFoundError(f"MGA-CLAP source file not found: {required}")
        tokenizer_path = Path(bert_tokenizer_path or "").expanduser().resolve()
        if not tokenizer_path.is_dir():
            raise FileNotFoundError(f"MGA-CLAP BERT tokenizer not found: {tokenizer_path}")
        actual_sha256 = _checkpoint_sha256(self.ckpt_path)
        if not expected_checkpoint_sha256 or actual_sha256 != expected_checkpoint_sha256:
            raise RuntimeError("MGA-CLAP checkpoint does not match the trusted SHA256")

        source_path = str(self.repo_root)
        if source_path not in sys.path:
            sys.path.insert(0, source_path)

        ase_module = importlib.import_module("models.ase_model")
        text_encoder_module = importlib.import_module("models.text_encoder")
        ase_source = Path(ase_module.__file__).resolve()
        if not ase_source.is_relative_to(self.repo_root):
            raise RuntimeError(f"MGA source module resolved outside the lock: {ase_source}")
        ASE = ase_module.ASE
        from transformers import BertConfig, BertModel, BertTokenizer

        # Use their inference config shape; device + ckpt come from args
        # If you keep a YAML, you can load it here; we only need minimal fields.
        self.device = _resolve_device(str(device))

        # Build a tiny config dict the ASE ctor expects (mirrors repo usage).  # :contentReference[oaicite:5]{index=5}
        # If your repo's ASE needs more keys, read settings/inference_example.yaml and supply them here.
        cfg_path = self.repo_root / "settings" / "inference_example.yaml"
        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                self.cfg = yaml_load(f)
            # override device & ckpt from CLI
            self.cfg["device"] = str(self.device)
            self.cfg.setdefault("eval", {})["ckpt"] = self.ckpt_path
        else:
            self.cfg = {"device": str(self.device), "eval": {"ckpt": self.ckpt_path}}

        original_bert = text_encoder_module.MODELS["bert-base-uncased"]

        class LocalBertTokenizer:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return BertTokenizer.from_pretrained(
                    tokenizer_path, local_files_only=True
                )

        class CheckpointInitializedBertModel:
            @classmethod
            def from_pretrained(cls, *_args, **kwargs):
                return BertModel(
                    BertConfig(),
                    add_pooling_layer=kwargs.get("add_pooling_layer", True),
                )

        text_encoder_module.MODELS["bert-base-uncased"] = (
            CheckpointInitializedBertModel,
            LocalBertTokenizer,
            768,
        )
        try:
            self.model = ASE(self.cfg)
        finally:
            text_encoder_module.MODELS["bert-base-uncased"] = original_bert
        state_dict = _load_mga_ckpt(
            self.ckpt_path,
            device="cpu",
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(self.device).eval()
        print(f"[INFO] MGA-CLAP checkpoint SHA256={actual_sha256}")
        print(f"[INFO] MGA-CLAP source={self.repo_root}")
        print(f"[INFO] MGA-CLAP local BERT tokenizer={tokenizer_path}")

        self._loader = _safe_import_sound_loader()
        self.target_sr = TARGET_SR
        self.target_len = int(self.target_sr * float(seconds))

        # utility
        import torch.nn.functional as F
        self.F = F
        self.torch = torch
        # Mixed precision control
        self.amp = bool(amp)

    # -------- helpers --------
    def _center_crop_or_pad(self, wav_t: "torch.Tensor") -> "torch.Tensor":
        T = wav_t.shape[-1]
        if T == self.target_len:
            return wav_t
        if T > self.target_len:
            s = (T - self.target_len) // 2
            return wav_t[..., s:s+self.target_len]
        pad = self.target_len - T
        L = pad // 2
        R = pad - L
        return self.torch.nn.functional.pad(wav_t, (L, R))

    def _l2norm_np(self, x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.clip(n, 1e-9, None)

    # -------- public API --------
    def encode_text(self, texts: List[str], batch_size: int = 256, device: str = "cuda") -> np.ndarray:
        torch, F = self.torch, self.F
        embs = []
        with torch.no_grad():
            ctx = (
                torch.autocast(device_type=str(self.device).split(":")[0], dtype=torch.float16)
                if self.amp and self.device.type in ("cuda", "cuda:0", "cuda:1")
                else contextlib.nullcontext()
            )
            
            with ctx:
                for i in range(0, len(texts), batch_size):
                    chunk = texts[i:i+batch_size]
                    # word-level embeds + attn mask  # :contentReference[oaicite:6]{index=6}
                    _, word_embeds, attn_mask = self.model.encode_text(chunk)
                    # aggregate to sentence-level via shared codebook  # :contentReference[oaicite:7]{index=7}
                    sent = self.model.msc(word_embeds, self.model.codebook, attn_mask)
                    sent = F.normalize(sent, dim=-1)  # cosine space  # :contentReference[oaicite:8]{index=8}
                    embs.append(sent.detach().cpu().numpy().astype(np.float32))
        return self._l2norm_np(np.concatenate(embs, 0))

    def encode_audio(self, paths: List[str], batch_size: int = 64, device: str = "cuda") -> np.ndarray:
        torch, F = self.torch, self.F
        A = []
        with torch.no_grad():
            ctx = (
                torch.autocast(device_type=str(self.device).split(":")[0], dtype=torch.float16)
                if self.amp and self.device.type in ("cuda", "cuda:0", "cuda:1")
                else contextlib.nullcontext()
            )
            # batch waveforms into [B, T]
            for i in range(0, len(paths), batch_size):
                batch = []
                for p in paths[i:i+batch_size]:
                    w = self._loader(p, self.target_sr)  # torch tensor [T]
                    w = self._center_crop_or_pad(w)
                    batch.append(w)
                x = torch.stack(batch, 0).to(self.device)  # [B, T]
                with ctx:
                    # frame-level encodings, then codebook pooling to clip-level  # :contentReference[oaicite:9]{index=9}
                    _, frame_embeds = self.model.encode_audio(x)
                    clip = self.model.msc(frame_embeds, self.model.codebook)
                    clip = F.normalize(clip, dim=-1)
                A.append(clip.detach().cpu().numpy().astype(np.float32))
        return self._l2norm_np(np.concatenate(A, 0))

    # -------- token/frame accessors for late interaction --------
    def encode_text_tokens(
        self,
        texts: List[str],
        batch_size: int = 128,
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Return token-level text embeddings and attention masks (no pooling).
        """
        torch = self.torch
        tokens_out: List[np.ndarray] = []
        masks_out: List[np.ndarray] = []

        with torch.no_grad():
            ctx = (
                torch.autocast(device_type=str(self.device).split(":")[0], dtype=torch.float16)
                if self.amp and self.device.type.startswith("cuda")
                else contextlib.nullcontext()
            )

            with ctx:
                for i in range(0, len(texts), batch_size):
                    chunk = texts[i : i + batch_size]
                    _, word_embeds, attn_mask = self.model.encode_text(chunk)
                    word_np = word_embeds.detach().cpu().numpy().astype(np.float32)
                    mask_np = attn_mask.detach().cpu().numpy().astype(np.int32)
                    for j in range(word_np.shape[0]):
                        tokens_out.append(word_np[j])
                        masks_out.append(mask_np[j])

        return tokens_out, masks_out

    def encode_audio_frames(
        self,
        paths: List[str],
        batch_size: int = 32,
    ) -> List[np.ndarray]:
        """
        Return frame-level audio embeddings (no pooling).
        """
        torch = self.torch
        frames_out: List[np.ndarray] = []

        with torch.no_grad():
            ctx = (
                torch.autocast(device_type=str(self.device).split(":")[0], dtype=torch.float16)
                if self.amp and self.device.type.startswith("cuda")
                else contextlib.nullcontext()
            )

            for i in range(0, len(paths), batch_size):
                batch_paths = paths[i : i + batch_size]
                wave_batch = []
                for p in batch_paths:
                    wav = self._loader(p, self.target_sr)
                    wav = self._center_crop_or_pad(wav)
                    wave_batch.append(wav)
                x = torch.stack(wave_batch, 0).to(self.device)
                with ctx:
                    _, frame_embeds = self.model.encode_audio(x)
                frame_np = frame_embeds.detach().cpu().numpy().astype(np.float32)
                for j in range(frame_np.shape[0]):
                    frames_out.append(frame_np[j])

        return frames_out
