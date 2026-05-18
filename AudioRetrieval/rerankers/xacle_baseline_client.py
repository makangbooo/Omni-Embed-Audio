"""XACLE baseline alignment scorer client.

This module integrates the official XACLE baseline MOS prediction model as a
pointwise audio reranker. The implementation mirrors the original training
code published under the MIT license in the XACLE challenge repository while
keeping the amount of vendored code minimal and focused on inference.
"""

from __future__ import annotations

import csv
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .base import LALMClient


class XACLEBaselineDependenciesMissing(RuntimeError):
    """Raised when optional dependencies for the XACLE baseline are absent."""


def _ensure_package(name: str) -> None:
    try:
        __import__(name)
    except Exception as exc:  # pragma: no cover - runtime guard
        raise XACLEBaselineDependenciesMissing(
            f"The XACLE baseline client requires the '{name}' package. "
            f"Install it with `pip install {name}`."
        ) from exc


def _resolve_audio_path(root: Path, relative: str) -> Optional[Path]:
    candidate = (root / relative.lstrip("/")).resolve()
    return candidate if candidate.exists() else None


@dataclass
class XACLEBaselineConfig:
    checkpoint_path: Path
    dataset_root: Path
    tokenizer_name: str
    tokenizer_cache_dir: Optional[Path]
    tokenizer_local_files_only: bool
    max_audio_seconds: float
    sample_rate: int
    normalizer_split: str
    normalizer_cache: Optional[Path]
    normalizer_max_files: Optional[int]
    device: str
    dtype: str
    batch_size: int


class PrecomputedNorm(nn.Module):
    """Simple normalization module wrapping pre-computed mean and std."""

    def __init__(self, mean: Tensor, std: Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - passthrough
        eps = torch.finfo(x.dtype).eps
        return (x - self.mean) / self.std.clamp_min(eps)


class AudioNTT2022(nn.Module):
    """CNN encoder used inside the XACLE baseline (BYOL-A backbone)."""

    def __init__(
        self,
        n_mels: int,
        feature_dim: int = 3072,
        base_channels: int = 64,
        mlp_hidden: int = 2048,
        conv_layers: int = 2,
    ) -> None:
        super().__init__()
        convs: List[nn.Module] = [
            nn.Conv2d(1, base_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        ]
        for _ in range(1, conv_layers):
            convs.extend(
                [
                    nn.Conv2d(base_channels, base_channels, 3, stride=1, padding=1),
                    nn.BatchNorm2d(base_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(2, stride=2),
                ]
            )
        self.features = nn.Sequential(*convs)
        conv_out = base_channels * (n_mels // (2 ** conv_layers))
        self.fc = nn.Sequential(
            nn.Linear(conv_out, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(mlp_hidden, feature_dim - conv_out),
            nn.ReLU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        feats = self.features(inputs)  # (B, C, F, T)
        feats = feats.permute(0, 3, 2, 1)  # (B, T, F, C)
        batch, frames, freq, channels = feats.shape
        feats = feats.reshape(batch, frames, freq * channels)
        fc = self.fc(feats)
        stacked = torch.hstack([feats.transpose(1, 2), fc.transpose(1, 2)]).transpose(1, 2)
        return stacked


class TextEncoder(nn.Module):
    """Wrapper around Hugging Face RoBERTa producing CLS embeddings."""

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        cache_dir: Optional[Path],
        local_files_only: bool,
    ) -> None:
        super().__init__()
        from transformers import RobertaModel  # local import to defer dependency

        kwargs = {
            "cache_dir": str(cache_dir) if cache_dir else None,
            "local_files_only": local_files_only,
        }
        self.model = RobertaModel.from_pretrained(model_name, **kwargs)
        self.model.to(device)

    def forward(self, tokens: Dict[str, Tensor]) -> Tensor:
        outputs = self.model(**tokens)
        return outputs.last_hidden_state[:, 0, :]


class AudioEncoder(nn.Module):
    """Log-mel + AudioNTT2022 feature extractor."""

    def __init__(
        self,
        n_mels: int,
        feature_dim: int,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        fmin: int,
        fmax: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        _ensure_package("nnAudio")
        from nnAudio.features import MelSpectrogram  # type: ignore

        self.model = AudioNTT2022(n_mels=n_mels, feature_dim=feature_dim)
        self.model.to(device)
        self.to_melspec = MelSpectrogram(
            sr=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            center=True,
            power=2,
            verbose=False,
        )
        self._device = device

    def forward(self, wavs: Tensor, normalizer: PrecomputedNorm) -> Tensor:
        eps = torch.finfo(torch.float32).eps
        lms = self.to_melspec(wavs)
        lms = torch.log(lms + eps)
        lms = normalizer(lms)
        lms = lms.to(self._device)
        lms = lms.unsqueeze(1)
        return self.model(lms)


class LDConditioner(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, audio_emb: Tensor, text_emb: Tensor) -> Tensor:
        text_expand = text_emb.unsqueeze(1).expand(-1, audio_emb.size(1), -1)
        feat = torch.cat([audio_emb, text_expand], dim=2)
        out, _ = self.rnn(feat)
        first = out[:, 0, :]
        last = out[:, -1, :]
        return (first + last) / 2.0


class Projection(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        activation: str = "ReLU",
        range_clipping: bool = False,
    ) -> None:
        super().__init__()
        act = getattr(nn, activation)() if isinstance(activation, str) else activation
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._range = range_clipping
        if self._range:
            self._out_act = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        if self._range:
            x = self._out_act(x)
        return x.squeeze(-1)


class XACLEAlignmentModel(nn.Module):
    """Full baseline network wrapping text/audio encoders and projection."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        device: torch.device,
        cache_dir: Optional[Path],
        local_files_only: bool,
    ) -> None:
        super().__init__()
        audio_cfg = cfg["audio_encoder"]
        model_cfg = cfg["model"]
        self.text_encoder = TextEncoder(
            model_name=cfg["text_encoder"]["pretrained_model"],
            device=device,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.audio_encoder = AudioEncoder(
            n_mels=audio_cfg["n_mels"],
            feature_dim=audio_cfg["feature_d"],
            sample_rate=audio_cfg["sample_rate"],
            n_fft=audio_cfg["n_fft"],
            win_length=audio_cfg["win_length"],
            hop_length=audio_cfg["hop_length"],
            fmin=audio_cfg["f_min"],
            fmax=audio_cfg["f_max"],
            device=device,
        )
        self.ldconditioner = LDConditioner(
            input_dim=model_cfg["conditioner"]["input_dim"],
            hidden_size=model_cfg["conditioner"]["rnn_hidden_size"],
            num_layers=model_cfg["conditioner"]["rnn_num_layers"],
        )
        self.projection = Projection(
            input_dim=model_cfg["projection"]["input_dim"],
            hidden_dim=model_cfg["projection"]["hidden_dim"],
            activation=model_cfg["projection"]["activation"],
            range_clipping=model_cfg["projection"]["range_clipping"],
            dropout=model_cfg["projection"]["dropout"],
        )
        self.to(device)

    def forward(self, batch: Dict[str, Any], normalizer: PrecomputedNorm) -> Tensor:
        text_emb = self.text_encoder(batch["caption_tokens"])
        audio_emb = self.audio_encoder(batch["wavs"], normalizer)
        cond = self.ldconditioner(audio_emb, text_emb)
        return self.projection(cond)


class XACLEBaselineClient(LALMClient):
    """Pointwise audio scorer that wraps the XACLE baseline model."""

    def __init__(
        self,
        checkpoint_dir: str = "./xacle_baseline_checkpoint",
        dataset_root: str = "./XACLE_dataset",
        tokenizer_name: str = "roberta-large",
        tokenizer_cache_dir: Optional[str] = None,
        tokenizer_local_files_only: bool = False,
        cache_dir: Optional[str] = None,
        local_files_only: Optional[bool] = None,
        normalizer_split: str = "train",
        normalizer_cache: Optional[str] = None,
        normalizer_max_files: Optional[int] = None,
        max_audio_seconds: float = 10.0,
        sample_rate: int = 16_000,
        device: str = "cuda:0",
        dtype: str = "float32",
        batch_size: int = 1,
    ) -> None:
        _ensure_package("torch")
        _ensure_package("transformers")
        _ensure_package("torchaudio")

        cache_override = Path(cache_dir) if cache_dir else None
        local_override = bool(local_files_only) if local_files_only is not None else None

        cfg = XACLEBaselineConfig(
            checkpoint_path=Path(checkpoint_dir) / "best_model.pt",
            dataset_root=Path(dataset_root),
            tokenizer_name=tokenizer_name,
            tokenizer_cache_dir=(
                Path(tokenizer_cache_dir)
                if tokenizer_cache_dir
                else cache_override
            ),
            tokenizer_local_files_only=(
                tokenizer_local_files_only
                if local_override is None
                else local_override
            ),
            max_audio_seconds=float(max_audio_seconds),
            sample_rate=int(sample_rate),
            normalizer_split=normalizer_split,
            normalizer_cache=Path(normalizer_cache) if normalizer_cache else None,
            normalizer_max_files=int(normalizer_max_files) if normalizer_max_files else None,
            device=device,
            dtype=dtype,
            batch_size=max(1, int(batch_size)),
        )
        self._config = cfg

        if not cfg.checkpoint_path.exists():
            raise XACLEBaselineDependenciesMissing(
                f"Checkpoint not found at {cfg.checkpoint_path}")
        cfg_path = cfg.checkpoint_path.parent / "config.json"
        if not cfg_path.exists():
            raise XACLEBaselineDependenciesMissing(
                f"Config file not found at {cfg_path}")

        with cfg_path.open("r", encoding="utf-8") as f:
            raw_cfg = json.load(f)

        self._device = torch.device(cfg.device)
        self.batch_size = cfg.batch_size
        self._dtype = self._resolve_dtype(cfg.dtype)
        self._max_audio_samples = int(cfg.max_audio_seconds * cfg.sample_rate)

        self._tokenizer = self._load_tokenizer()
        raw_cfg["device"] = cfg.device
        model = XACLEAlignmentModel(
            raw_cfg,
            device=self._device,
            cache_dir=cfg.tokenizer_cache_dir,
            local_files_only=cfg.tokenizer_local_files_only,
        )
        state = torch.load(cfg.checkpoint_path, map_location="cpu")
        removed_buffers = [k for k in list(state.keys()) if k.endswith("position_ids")]
        for key in removed_buffers:
            state.pop(key, None)
        if removed_buffers:
            warnings.warn(
                "Dropped position_ids entries from XACLE checkpoint to match current RoBERTa implementation.",
                RuntimeWarning,
            )
        model.load_state_dict(state, strict=True)
        model.to(self._device)
        model.eval()
        self.model = model
        self._normalizer = self._prepare_normalizer(raw_cfg)
        self._last_scores: Optional[List[float]] = None

    # ------------------------------------------------------------------ helpers
    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        cache_dir = self._config.tokenizer_cache_dir
        kwargs = {
            "cache_dir": str(cache_dir) if cache_dir else None,
            "use_fast": True,
            "local_files_only": self._config.tokenizer_local_files_only,
        }
        try:
            return AutoTokenizer.from_pretrained(self._config.tokenizer_name, **kwargs)
        except Exception as exc:  # pragma: no cover - runtime guard
            raise XACLEBaselineDependenciesMissing(
                "Failed to load tokenizer. Download it first or set "
                "`tokenizer_local_files_only=false`."
            ) from exc

    def _resolve_dtype(self, dtype: str) -> torch.dtype:
        mapping = {
            "float32": torch.float32,
            "float": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
        }
        key = dtype.lower()
        if key not in mapping:
            raise ValueError(f"Unsupported dtype '{dtype}' for XACLE baseline client")
        return mapping[key]

    def _prepare_normalizer(self, cfg: Dict[str, Any]) -> PrecomputedNorm:
        cache_path = self._config.normalizer_cache
        if cache_path and cache_path.exists():
            data = torch.load(cache_path, map_location="cpu")
            mean = torch.tensor(data["mean"], dtype=torch.float32)
            std = torch.tensor(data["std"], dtype=torch.float32)
            return PrecomputedNorm(mean, std)

        split = self._config.normalizer_split.lower()
        csv_path = self._config.dataset_root / "meta_data" / f"{split}_average.csv"
        audio_root = self._config.dataset_root / "wav" / split
        if not csv_path.exists() or not audio_root.exists():
            raise XACLEBaselineDependenciesMissing(
                f"Missing XACLE dataset split '{split}'. Expected {csv_path} and {audio_root}.")
        mean, std = self._compute_logmel_stats(csv_path, audio_root)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"mean": mean.item(), "std": std.item()}, cache_path)
        return PrecomputedNorm(mean, std)

    def _compute_logmel_stats(self, csv_path: Path, audio_root: Path) -> Tuple[Tensor, Tensor]:
        import torchaudio

        mel = self.model.audio_encoder.to_melspec
        eps = torch.finfo(torch.float32).eps
        max_files = self._config.normalizer_max_files
        total_sum = 0.0
        total_sq = 0.0
        total_count = 0
        processed = 0

        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("wav_file_name") or row.get("file") or row.get("path")
                if not name:
                    continue
                path = _resolve_audio_path(audio_root, name)
                if path is None:
                    continue
                waveform, sr = torchaudio.load(path)
                waveform = self._prepare_waveform_tensor(waveform, sr)
                waveform = waveform.to(self._device, dtype=self._dtype)
                with torch.no_grad():
                    lms = mel(waveform)
                    log_lms = torch.log(lms + eps).cpu()
                total_sum += float(log_lms.sum())
                total_sq += float((log_lms ** 2).sum())
                total_count += log_lms.numel()
                processed += 1
                if max_files and processed >= max_files:
                    break

        if total_count == 0:
            raise XACLEBaselineDependenciesMissing(
                f"Failed to compute log-mel stats from {csv_path}; no audio processed.")

        mean = total_sum / total_count
        variance = (total_sq / total_count) - (mean ** 2)
        std = math.sqrt(max(variance, 1e-6))
        return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)

    def _prepare_waveform_tensor(self, waveform: Tensor, sample_rate: int) -> Tensor:
        import torchaudio

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self._config.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self._config.sample_rate)
        waveform = waveform.to(torch.float32)
        if waveform.shape[1] < self._max_audio_samples:
            pad = self._max_audio_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, : self._max_audio_samples]
        return waveform

    def _prepare_batch(self, query: str, audio_paths: Sequence[Path]) -> Dict[str, Any]:
        import torchaudio

        waveforms: List[Tensor] = []
        resolved: List[Path] = []
        for path in audio_paths:
            waveform, sr = torchaudio.load(str(path))
            waveform = self._prepare_waveform_tensor(waveform, sr)
            waveforms.append(waveform)
            resolved.append(Path(path))
        if not waveforms:
            raise XACLEBaselineDependenciesMissing("No valid audio paths provided for scoring.")

        wav_batch = torch.stack(waveforms).to(self._device, dtype=self._dtype)
        tokens = self._tokenizer([query] * len(resolved), padding=True, truncation=True, return_tensors="pt")
        tokens = {k: v.to(self._device) for k, v in tokens.items()}
        return {
            "wavs": wav_batch,
            "caption_tokens": tokens,
            "wav_paths": [str(p) for p in resolved],
        }

    def _postprocess_scores(self, scores: Tensor) -> List[float]:
        mos = scores.float() * 5.0 + 5.0
        normalized = torch.clamp(mos / 10.0, min=0.0, max=1.0)
        return normalized.cpu().tolist()

    # ---------------------------------------------------------------- client API
    def score_audio(self, query: str, audio_path: Path) -> float:
        batch = self._prepare_batch(query, [Path(audio_path)])
        with torch.no_grad():
            preds = self.model(batch, self._normalizer)
        scores = self._postprocess_scores(preds)
        self._last_scores = scores
        return scores[0]

    def score_audio_batch(self, query: str, audio_paths: List[Path]) -> List[float]:
        if not audio_paths:
            return []
        batch = self._prepare_batch(query, [Path(p) for p in audio_paths])
        with torch.no_grad():
            preds = self.model(batch, self._normalizer)
        scores = self._postprocess_scores(preds)
        self._last_scores = scores
        return scores

    def get_last_scores(self) -> Optional[List[float]]:
        return self._last_scores
