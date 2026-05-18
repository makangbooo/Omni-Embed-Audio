# -*- coding: utf-8 -*-
# models/m2d_clap_adapter.py
# Adapter for M2D-CLAP (2025) - A SOTA Audio-Language Representation Model
# Paper: "M2D-CLAP: Exploring General-purpose Audio-Language Representations Beyond CLAP"
# Repo: https://github.com/nttcslab/m2d

from __future__ import annotations
import sys
import warnings
from pathlib import Path
from typing import List
import numpy as np
import torch
from tqdm import tqdm


def _resolve_device(requested: str) -> torch.device:
    """
    Resolve the desired torch.device, gracefully falling back to CPU when CUDA
    is requested but unavailable.
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


def _safe_import_audio_loader():
    """
    Import librosa for loading audio files at 16kHz.
    M2D-CLAP expects 16kHz audio input.
    """
    try:
        import librosa
        import soundfile as sf

        def _librosa_loader(path, target_sr=16000):
            audio, sr = librosa.load(path, sr=target_sr, mono=True)
            return torch.tensor(audio, dtype=torch.float32)

        return _librosa_loader
    except ImportError:
        raise RuntimeError(
            "Install librosa and soundfile to load audio: pip install librosa soundfile"
        )


class M2DClapAdapter:
    """
    Adapter for M2D-CLAP (2025) model for audio retrieval evaluation.

    The M2D-CLAP model is a state-of-the-art audio-language representation model
    that jointly learns general-purpose audio features and CLAP features.

    Requirements:
      - PortableM2D module (from nttcslab/m2d repository)
      - M2D-CLAP checkpoint file

    Expected checkpoint location:
      - ModelCheckpoint/M2D2/m2d_clap/checkpoint-30.pth

    The model expects:
      - Audio: 16kHz sample rate, 10 seconds duration (padded/cropped as needed)
      - Text: Plain text captions
    """

    def __init__(
        self,
        weight_file: str,
        seconds: float = 10.0,
        device: str = "cuda",
        amp: bool = False,
    ):
        """
        Initialize M2D-CLAP adapter.

        Args:
            weight_file: Path to M2D-CLAP checkpoint file
            seconds: Audio duration in seconds (default: 10.0)
            device: Device to run model on (default: "cuda")
            amp: Enable automatic mixed precision (default: False)
        """
        # Don't resolve symlinks - the parent folder name is used for model config parsing
        self.weight_file = Path(weight_file).expanduser().absolute()
        if not self.weight_file.exists():
            raise FileNotFoundError(f"M2D-CLAP checkpoint not found: {self.weight_file}")

        self.device = _resolve_device(str(device))
        self.amp = bool(amp)
        self.target_sr = 16000  # M2D-CLAP expects 16kHz audio
        self.target_len = int(self.target_sr * float(seconds))

        # Import PortableM2D
        try:
            # Add models directory to path if needed
            models_dir = Path(__file__).parent
            if str(models_dir) not in sys.path:
                sys.path.insert(0, str(models_dir))

            from portable_m2d import PortableM2D

            # Initialize model with CLAP features
            print(f"[INFO] Loading M2D-CLAP from {self.weight_file}")
            self.model = PortableM2D(
                weight_file=str(self.weight_file),
                flat_features=True  # Use CLAP features
            )
            self.model = self.model.to(self.device).eval()

        except Exception as e:
            raise RuntimeError(
                f"Failed to load PortableM2D. Make sure portable_m2d.py is in the models directory. Error: {e}"
            )

        # Set up audio loader
        self._loader = _safe_import_audio_loader()

        # Store torch module for utilities
        self.torch = torch

        print(f"[INFO] M2D-CLAP initialized on {self.device}")
        print(f"[INFO] Audio target: {self.target_sr}Hz, {seconds}s")

    def _prepare_audio(self, wav_t: torch.Tensor) -> torch.Tensor:
        """
        Pad or crop audio to target length (center crop/pad).

        Args:
            wav_t: Audio waveform tensor [T]

        Returns:
            Processed audio tensor [target_len]
        """
        T = wav_t.shape[-1]
        if T == self.target_len:
            return wav_t
        if T > self.target_len:
            # Center crop
            start = (T - self.target_len) // 2
            return wav_t[..., start : start + self.target_len]
        # Pad
        pad_total = self.target_len - T
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return torch.nn.functional.pad(wav_t, (pad_left, pad_right))

    def _l2norm_np(self, x: np.ndarray) -> np.ndarray:
        """L2 normalize numpy array along last dimension."""
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.clip(norm, 1e-9, None)

    def encode_text(
        self, texts: List[str], batch_size: int = 256, device: str = "cuda"
    ) -> np.ndarray:
        """
        Encode text captions to embeddings.

        Args:
            texts: List of text captions
            batch_size: Batch size for encoding (default: 256)
            device: Device string (ignored, uses self.device)

        Returns:
            Text embeddings as numpy array [N, D]
        """
        embeddings = []
        num_batches = (len(texts) + batch_size - 1) // batch_size

        with torch.no_grad():
            pbar = tqdm(
                range(0, len(texts), batch_size),
                total=num_batches,
                desc="Encoding text",
                unit="batch"
            )
            for i in pbar:
                batch = texts[i : i + batch_size]
                # M2D-CLAP encode_clap_text expects list of strings
                batch_emb = self.model.encode_clap_text(batch)

                # Convert to numpy
                if isinstance(batch_emb, torch.Tensor):
                    batch_emb = batch_emb.detach().cpu().numpy()

                embeddings.append(batch_emb.astype(np.float32))

                # Update progress bar with batch info
                pbar.set_postfix({"texts": f"{min(i + batch_size, len(texts))}/{len(texts)}"})

        result = np.concatenate(embeddings, axis=0)
        return self._l2norm_np(result)

    def encode_audio(
        self, paths: List[str], batch_size: int = 64, device: str = "cuda"
    ) -> np.ndarray:
        """
        Encode audio files to embeddings.

        Args:
            paths: List of audio file paths
            batch_size: Batch size for encoding (default: 64)
            device: Device string (ignored, uses self.device)

        Returns:
            Audio embeddings as numpy array [N, D]
        """
        embeddings = []
        num_batches = (len(paths) + batch_size - 1) // batch_size

        with torch.no_grad():
            pbar = tqdm(
                range(0, len(paths), batch_size),
                total=num_batches,
                desc="Encoding audio",
                unit="batch"
            )
            for i in pbar:
                batch_paths = paths[i : i + batch_size]

                # Load and prepare audio waveforms
                waveforms = []
                for path in batch_paths:
                    wav = self._loader(path, self.target_sr)
                    wav = self._prepare_audio(wav)
                    waveforms.append(wav)

                # Stack to batch tensor [B, T]
                batch_wav = torch.stack(waveforms, dim=0)

                # Ensure audio is on correct device
                batch_wav = batch_wav.to(self.device)

                # Encode using M2D-CLAP
                batch_emb = self.model.encode_clap_audio(batch_wav)

                # Convert to numpy
                if isinstance(batch_emb, torch.Tensor):
                    batch_emb = batch_emb.detach().cpu().numpy()

                embeddings.append(batch_emb.astype(np.float32))

                # Update progress bar with file info
                pbar.set_postfix({"files": f"{min(i + batch_size, len(paths))}/{len(paths)}"})

        result = np.concatenate(embeddings, axis=0)
        return self._l2norm_np(result)

    def encode_text_tokens(
        self, texts: List[str], batch_size: int = 128
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Encode text to token-level embeddings (for late interaction).

        Note: M2D-CLAP may not expose token-level features in the same way
        as other models. This is a placeholder implementation.

        Args:
            texts: List of text captions
            batch_size: Batch size for encoding

        Returns:
            Tuple of (token embeddings list, attention masks list)
        """
        warnings.warn(
            "M2D-CLAP token-level encoding not fully implemented. "
            "Using sentence-level embeddings as fallback.",
            UserWarning
        )

        # Fallback: return sentence embeddings as single-token sequences
        sentence_embs = self.encode_text(texts, batch_size=batch_size)
        tokens_out = [emb[np.newaxis, :] for emb in sentence_embs]  # [1, D] per text
        masks_out = [np.ones(1, dtype=np.int32) for _ in sentence_embs]

        return tokens_out, masks_out

    def encode_audio_frames(
        self, paths: List[str], batch_size: int = 32
    ) -> List[np.ndarray]:
        """
        Encode audio to frame-level embeddings (for late interaction).

        Note: M2D-CLAP may not expose frame-level features in the same way
        as other models. This is a placeholder implementation.

        Args:
            paths: List of audio file paths
            batch_size: Batch size for encoding

        Returns:
            List of frame-level embeddings
        """
        warnings.warn(
            "M2D-CLAP frame-level encoding not fully implemented. "
            "Using clip-level embeddings as fallback.",
            UserWarning
        )

        # Fallback: return clip embeddings as single-frame sequences
        clip_embs = self.encode_audio(paths, batch_size=batch_size)
        frames_out = [emb[np.newaxis, :] for emb in clip_embs]  # [1, D] per audio

        return frames_out
