# -*- coding: utf-8 -*-
# models/cacophony_adapter.py
# JAX/Flax adapter for Cacophony that plugs into your BaseRetrievalModel
# Assumes the Cacophony repo is cloned locally and provides retrieval_utils.{build_inference_fns,...}
# Fallback: we try to auto-discover working encode functions; if not found, we print a tiny helper signature.

from __future__ import annotations
import os, sys, warnings
from pathlib import Path
from typing import List, Callable, Tuple, Optional
import numpy as np

# ---- simple audio utilities (16 kHz, mono, center-crop/pad) ----
def _safe_import_sound_loader():
    try:
        import soundfile as sf
        def _sf_loader(path, target_sr):
            wav, sr = sf.read(path)
            if wav.ndim > 1: wav = wav.mean(axis=1)
            if sr != target_sr:
                try:
                    import librosa
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best")
                except Exception:
                    try:
                        import torchaudio, torch
                        t = torch.tensor(wav).float().unsqueeze(0)
                        wav = torchaudio.functional.resample(t, sr, target_sr).squeeze(0).numpy()
                    except Exception:
                        raise RuntimeError("Resampling failed: please install librosa or torchaudio.")
            return wav.astype(np.float32)
        return _sf_loader
    except Exception:
        pass
    try:
        import librosa
        def _lb_loader(path, target_sr):
            wav, _ = librosa.load(path, sr=target_sr, mono=True)
            return wav.astype(np.float32)
        return _lb_loader
    except Exception:
        pass
    try:
        import torchaudio, torch
        def _ta_loader(path, target_sr):
            wav, sr = torchaudio.load(path)
            wav = wav.mean(dim=0)
            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)
            return wav.numpy().astype(np.float32)
        return _ta_loader
    except Exception:
        pass
    raise RuntimeError("Please install one of: soundfile, librosa, or torchaudio.")

def _center_crop_or_pad(wav: np.ndarray, target_len: int) -> np.ndarray:
    T = wav.shape[0]
    if T == target_len: return wav
    if T > target_len:
        s = (T - target_len) // 2
        return wav[s:s+target_len]
    pad = target_len - T
    L = pad // 2
    return np.pad(wav, (L, pad - L), mode="constant")

def _l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, eps, None)

# ---- the adapter ----
class CacophonyAdapter:
    """
    Uses functions from the Cacophony repo to produce embeddings:
      encode_audio_fn(wavs_np: [B, T@16k]) -> [B, D]
      encode_text_fn(list[str]) -> [B, D]
    Repo: https://github.com/gzhu06/Cacophony (eval_caco.py shows retrieval on AudioCaps/Clotho). :contentReference[oaicite:1]{index=1}
    """
    def __init__(self, repo_path: str, ckpt_path: str, seconds: float = 10.0):
        self.repo_root = Path(repo_path).expanduser().resolve()
        self.ckpt_path = str(Path(ckpt_path).expanduser().resolve())
        if not self.repo_root.exists():
            raise FileNotFoundError(f"Cacophony repo not found: {self.repo_root}")
        if not Path(self.ckpt_path).exists():
            raise FileNotFoundError(f"Cacophony ckpt not found: {self.ckpt_path}")
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))

        # Cacophony is trained/evaluated at 16 kHz (see README & eval script). :contentReference[oaicite:2]{index=2}
        self.target_sr = 16000
        self.target_len = int(self.target_sr * float(seconds))
        self._loader = _safe_import_sound_loader()

        self._ea: Optional[Callable[[np.ndarray], np.ndarray]] = None
        self._et: Optional[Callable[[List[str]], np.ndarray]] = None
        self._ensure_inference_fns()

    def _ensure_inference_fns(self) -> None:
        """
        Wire up two callables:
          self._ea(wavs_np[B, T@16k]) -> np.float32[B, D]
          self._et(list[str])         -> np.float32[B, D]

        Priority:
          1) <repo>/retrieval_utils.build_inference_fns(model_path)
          2) Fallback: auto-discover builder+encoders in <repo>/retrieval_utils.py
          3) Fallback: eval_caco with ABSEIL flags seeded (CLI-friendly path)
        """
        import os, sys, importlib, importlib.util, warnings, inspect
        import numpy as np

        # ---------- helpers ----------
        def _wrap_with_state(state, enc_audio, enc_text):
            def _ea(wavs_np: np.ndarray) -> np.ndarray:
                return np.asarray(enc_audio(state, wavs_np), dtype=np.float32)
            def _et(texts: list[str]) -> np.ndarray:
                return np.asarray(enc_text(state, texts), dtype=np.float32)
            return _ea, _et

        def _accepts_1arg_or_var(fn):
            try:
                sig = inspect.signature(fn)
                pos = [p for p in sig.parameters.values()
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
                return len(pos) == 1 or any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values())
            except Exception:
                return True  # be permissive

        # ---------- 1) Load retrieval_utils.py from the repo by absolute path ----------
        ru = None
        ru_path = os.path.join(str(self.repo_root), "retrieval_utils.py")
        if os.path.isfile(ru_path):
            try:
                spec = importlib.util.spec_from_file_location("ru_caco", ru_path)
                ru = importlib.util.module_from_spec(spec)
                sys.modules["ru_caco"] = ru
                assert spec.loader is not None
                spec.loader.exec_module(ru)
            except Exception as e:
                warnings.warn(f"[Cacophony] failed to import retrieval_utils at {ru_path}: {e!r}")
                ru = None
        else:
            warnings.warn(f"[Cacophony] retrieval_utils.py not found at {ru_path}")

        # 1a) Preferred explicit entrypoint
        if ru is not None and hasattr(ru, "build_inference_fns"):
            try:
                self._ea, self._et = ru.build_inference_fns(self.ckpt_path)
                return
            except Exception as e:
                warnings.warn(f"[Cacophony] retrieval_utils.build_inference_fns failed: {e!r}")

        # ---------- 2) Auto-discover in retrieval_utils ----------
        if ru is not None:
            names = dir(ru)

            builders = [n for n in names
                        if any(k in n.lower() for k in ("load", "build", "init", "setup", "make", "create"))
                        and callable(getattr(ru, n))]
            texters  = [n for n in names
                        if ("text" in n.lower()) and (("encode" in n.lower()) or ("embed" in n.lower()))
                        and callable(getattr(ru, n))]
            audios   = [n for n in names
                        if ("audio" in n.lower()) and (("encode" in n.lower()) or ("embed" in n.lower()))
                        and callable(getattr(ru, n))]

            tried = []
            for bname in builders:
                bfn = getattr(ru, bname)
                if not _accepts_1arg_or_var(bfn):
                    continue
                try:
                    state = bfn(self.ckpt_path)
                except Exception as e:
                    tried.append((bname, f"builder error: {e!r}"))
                    continue

                # probe encoders
                _et = _ea = None
                for tname in texters:
                    tfn = getattr(ru, tname)
                    try:
                        tfn(state, [])  # dry-run
                        _et = tfn
                        break
                    except Exception:
                        pass
                for aname in audios:
                    afn = getattr(ru, aname)
                    try:
                        dummy = np.zeros((0, self.target_len), dtype=np.float32)
                        afn(state, dummy)  # dry-run
                        _ea = afn
                        break
                    except Exception:
                        pass
                if _ea is not None and _et is not None:
                    self._ea, self._et = _wrap_with_state(state, _ea, _et)
                    return

            if tried:
                warnings.warn("[Cacophony] auto-discovery attempts:\n  " +
                              "\n  ".join(f"{n}: {msg}" for n, msg in tried))
            else:
                # Helpful breadcrumb: what symbols exist in ru?
                syms = ", ".join(sorted([n for n in names if not n.startswith("_")][:30]))
                warnings.warn(f"[Cacophony] retrieval_utils has no obvious builder/encoders. Symbols: {syms}")

        # ---------- 3) LAST RESORT: use eval_caco but seed ABSEIL FLAGS first ----------
        # The official entrypoint is a CLI `python eval_caco.py --task ar --model_path <ckpt>`.
        # We emulate that by setting FLAGS before importing/using its helpers.
        try:
            from absl import flags
            FLAGS = flags.FLAGS
            # Define only if not already defined
            if "task" not in FLAGS:
                flags.DEFINE_string("task", None, "Task to run (ar|zs|caption)")
            if "model_path" not in FLAGS:
                flags.DEFINE_string("model_path", None, "Path to checkpoint")
            # Parse minimal argv to satisfy eval_caco expectations
            if not FLAGS.is_parsed():
                FLAGS(["prog", f"--task=ar", f"--model_path={self.ckpt_path}"])
        except Exception as e:
            warnings.warn(f"[Cacophony] ABSEIL flags init failed (eval_caco path may break): {e!r}")

        def _wrap_eval_caco(eval_mod):
            # Find a loader that can work with FLAGS or takes (model_path)
            cand_loaders = [getattr(eval_mod, n) for n in dir(eval_mod)
                            if ("load" in n.lower() or "init" in n.lower())
                            and callable(getattr(eval_mod, n))]
            last_err = None
            state = None
            for lf in cand_loaders:
                try:
                    # Try with explicit path first
                    if _accepts_1arg_or_var(lf):
                        state = lf(self.ckpt_path)
                    else:
                        # Some versions read FLAGS internally; call with no args
                        state = lf()
                    if state is not None:
                        break
                except Exception as e:
                    last_err = e
                    continue
            if state is None:
                raise RuntimeError(f"no usable loader in eval_caco; last_err={last_err!r}")

            # Find audio/text encoders
            def _pick(kind):
                keys = ("audio", "text")[kind == "text"]
                for n in dir(eval_mod):
                    if ("encode" in n.lower() or "embed" in n.lower()) and \
                      (("audio" in n.lower()) if kind=="audio" else ("text" in n.lower())):
                        fn = getattr(eval_mod, n)
                        if callable(fn):
                            return fn
                raise RuntimeError(f"no encode_{kind} function in eval_caco")

            enc_audio = _pick("audio")
            enc_text  = _pick("text")
            return _wrap_with_state(state, enc_audio, enc_text)

        try:
            ec = importlib.import_module("eval_caco")
            self._ea, self._et = _wrap_eval_caco(ec)
            return
        except Exception as e:
            warnings.warn(f"[Cacophony] eval_caco wiring failed: {e!r}")

        # ---------- If we’re here, nothing worked ----------
        ru_syms = ""
        if ru is not None:
            ru_syms = "Found retrieval_utils symbols: " + ", ".join(
                sorted([n for n in dir(ru) if not n.startswith("_")][:30])
            )
        raise RuntimeError(
            "Could not locate Cacophony encode functions.\n"
            "Expected either:\n"
            " - retrieval_utils.build_inference_fns(model_path) in your repo, or\n"
            " - retrieval_utils to contain a builder (load/build/init/setup) and encoders (encode_audio/encode_text), or\n"
            " - eval_caco.{load_model, encode_audio, encode_text} (with FLAGS seeded)\n"
            f"{ru_syms}\n"
            "If needed, add this minimal shim to retrieval_utils.py:\n"
            "  def build_inference_fns(model_path):\n"
            "      from absl import flags\n"
            "      FLAGS = flags.FLAGS\n"
            "      if 'task' not in FLAGS: flags.DEFINE_string('task', None, '')\n"
            "      if 'model_path' not in FLAGS: flags.DEFINE_string('model_path', None, '')\n"
            "      if not FLAGS.is_parsed(): FLAGS(['prog', f'--task=ar', f'--model_path={model_path}'])\n"
            "      import eval_caco as ec\n"
            "      state = ec.load_model(model_path) if getattr(ec, 'load_model', None) else ec.load()\n"
            "      def ea(wavs_np): return ec.encode_audio(state, wavs_np)\n"
            "      def et(texts):   return ec.encode_text(state, texts)\n"
            "      return ea, et\n"
        )
    
    # ---- evaluator API ----
    def encode_audio(self, paths: List[str], batch_size: int = 32, device: str = "cuda") -> np.ndarray:
        wavs = []
        for p in paths:
            w = self._loader(p, self.target_sr)
            w = _center_crop_or_pad(w, self.target_len)
            wavs.append(w)
        embs = []
        for i in range(0, len(wavs), batch_size):
            chunk = np.stack(wavs[i:i+batch_size], axis=0)
            e = self._ea(chunk)  # expects [B,T@16k] -> [B,D]
            embs.append(np.asarray(e, dtype=np.float32))
        return _l2norm(np.concatenate(embs, axis=0))

    def encode_text(self, texts: List[str], batch_size: int = 256, device: str = "cuda") -> np.ndarray:
        embs = []
        for i in range(0, len(texts), batch_size):
            e = self._et(texts[i:i+batch_size])  # expects list[str] -> [b,D]
            embs.append(np.asarray(e, dtype=np.float32))
        return _l2norm(np.concatenate(embs, axis=0))
