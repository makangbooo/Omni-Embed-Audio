#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Omni-Embed (Nemotron-3B) with LoRA + projection heads for audio-text retrieval.

Features
--------
- Loads Omni-Embed via HuggingFace AutoModel / AutoProcessor (trust_remote_code=True).
- Attaches PEFT LoRA modules to attention projections (configurable).
- Pools per-modality embeddings (text queries vs. audio documents).
- Trains lightweight projection heads with symmetric contrastive loss (InfoNCE).
- Supports offline hard-negative injection (JSONL mined captions -> extra audio docs).
- Gradient accumulation for large effective batch sizes.
- Validation with Recall@{1,5,10}; saves best checkpoint + LoRA weights.
- Optional Weights & Biases logging.

Example (AudioCaps pre-alignment):
    python scripts/train_omniembed_lora_retrieval.py \\
        --dataset audiocaps \\
        --train-csv AudioCaps/meta_data/train.csv \\
        --val-csv AudioCaps/v2_meta_data/val.csv \\
        --audio-dir AudioCaps/audiocaps_raw_audio \\
        --hard-neg-json results/hard_negatives/audiocaps_train_mga_top50_unique.jsonl \\
        --repo-id omniengineering/omni-embed-nemotron-3b \\
        --device cuda \\
        --epochs 3 \\
        --batch-size 2 \\
        --grad-accum 512 \\
        --output-dir outputs/omniembed_lora/audiocaps_ft

Stage-2 (Clotho fine-tune) would pass --init-checkpoint pointing to stage-1 best.pt.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]  # AudioRetrieval/training/oea -> AudioRetrieval/training -> AudioRetrieval
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError as exc:  # pragma: no cover - dependency
    raise SystemExit("peft is required. Install with `pip install peft`.") from exc

from transformers import AutoModel, AutoProcessor

from AudioRetrieval.models.omni_embed_adapter import (
    OmniEmbedAdapter,
    _BatchInputs,
    _safe_audio_loader,
)

def _lazy_load_wandb():
    try:
        import wandb as _wandb
    except ImportError:  # pragma: no cover - optional dependency
        return None

    if not hasattr(_wandb, "init"):
        warnings.warn(
            "Weights & Biases import succeeded but missing `wandb.init`. "
            "Telemetry logging will be disabled. Reinstall wandb>=0.15 if logging is required.",
            RuntimeWarning,
        )
        return None
    return _wandb


wandb = _lazy_load_wandb()


def _patch_pathlib_local():
    """
    Some checkpoints were saved in environments that registered a synthetic module
    named `pathlib._local`. If torch.load later tries to import it, standard
    Python raises `ModuleNotFoundError` because pathlib is a module, not a package.
    We synthesise a best-effort shim to satisfy the pickle loader.
    """
    if "pathlib._local" in sys.modules:
        return
    import pathlib as _pathlib
    shim = types.ModuleType("pathlib._local")
    # Expose common classes the pickle is likely expecting.
    for attr in ("Path", "PosixPath", "WindowsPath", "PurePath", "PurePosixPath", "PureWindowsPath"):
        if hasattr(_pathlib, attr):
            setattr(shim, attr, getattr(_pathlib, attr))
    # Some pickles expect module-level variables like _local. Mirror namespace.
    shim.__dict__.update({k: v for k, v in _pathlib.__dict__.items() if k.startswith("_")})
    sys.modules["pathlib._local"] = shim


def safe_torch_load(path: Path, map_location: torch.device) -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except ModuleNotFoundError as exc:
        if "pathlib._local" in str(exc):
            print(f"[WARN] Missing {exc.name}. Injecting compatibility shim for pathlib._local.")
            _patch_pathlib_local()
            return torch.load(path, map_location=map_location, weights_only=False)
        raise


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def resolve_text_hidden_size(model: torch.nn.Module) -> int:
    """
    Return the hidden size of the text tower for models that may not expose
    `config.text_config.hidden_size` directly (e.g., Qwen2.5-Omni).
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        if hasattr(model, "get_input_embeddings"):
            return model.get_input_embeddings().embedding_dim
        raise ValueError("Model configuration missing and embeddings unavailable.")

    text_cfg = None
    if hasattr(cfg, "text_config"):
        text_cfg = cfg.text_config
    elif hasattr(cfg, "get_text_config"):
        text_cfg = cfg.get_text_config()

    if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
        return text_cfg.hidden_size

    if hasattr(cfg, "hidden_size"):
        return cfg.hidden_size

    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings().embedding_dim

    raise ValueError("Unable to infer hidden size from model configuration.")


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

def _find_audio_caps_audio(
    audio_dir: Path,
    youtube_id: str,
    start_time: str,
) -> Optional[Path]:
    """
    Resolve AudioCaps audio filename. We check several plausible patterns.
    """
    stem_candidates = [
        f"{youtube_id}_{start_time}",
        f"{youtube_id}_{float(start_time):.0f}",
        f"Y{youtube_id}_{start_time}",
        f"{youtube_id}",
        f"Y{youtube_id}",
    ]
    extensions = [".wav", ".flac", ".mp3", ".ogg", ".m4a"]
    for stem in stem_candidates:
        for ext in extensions:
            candidate = audio_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    return None


def _list_clotho_entries(csv_path: Path, audio_dir: Path) -> List[dict]:
    entries: List[dict] = []
    missing_files = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = ["file_name", "caption_1", "caption_2", "caption_3", "caption_4", "caption_5"]
        for col in required:
            if col not in reader.fieldnames:
                raise RuntimeError(f"Clotho CSV missing column '{col}'. Header={reader.fieldnames}")
        for row in reader:
            file_name = row["file_name"].strip()
            clip_id = Path(file_name).stem

            # Try multiple filename variations to handle inconsistent naming in Clotho v2.0
            audio_path = audio_dir / file_name
            if not audio_path.exists():
                # Try with leading space (common issue in Clotho v2.0)
                audio_path = audio_dir / f" {file_name}"
            if not audio_path.exists():
                # Try without extension and re-add
                stem = Path(file_name).stem
                for ext in [".wav", ".flac", ".mp3"]:
                    candidate = audio_dir / f"{stem}{ext}"
                    if candidate.exists():
                        audio_path = candidate
                        break
                    # Also try with leading space
                    candidate = audio_dir / f" {stem}{ext}"
                    if candidate.exists():
                        audio_path = candidate
                        break

            if not audio_path.exists():
                missing_files.append(str(audio_path))
                continue  # Skip this entry instead of crashing

            for idx in range(5):
                cap = row[f"caption_{idx+1}"].strip()
                if not cap:
                    continue
                caption_key = f"{clip_id}_caption_{idx:02d}"
                entries.append(
                    {
                        "clip_id": clip_id,
                        "caption": cap,
                        "caption_key": caption_key,
                        "audio_path": audio_path,
                    }
                )

    if missing_files:
        print(f"[WARN] Skipped {len(missing_files)} Clotho entries with missing audio files")
        if len(missing_files) <= 5:
            for f in missing_files:
                print(f"  - {f}")

    if not entries:
        raise RuntimeError(f"No Clotho entries found from {csv_path}. Check audio dir {audio_dir}.")

    return entries


def _list_audiocaps_entries(csv_path: Path, audio_dir: Path) -> List[dict]:
    entries: List[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = ["audiocap_id", "youtube_id", "start_time", "caption"]
        for col in required:
            if col not in reader.fieldnames:
                raise RuntimeError(f"AudioCaps CSV missing column '{col}'. Header={reader.fieldnames}")
        seen_per_clip: Dict[str, int] = {}
        missing_audio = 0
        for row in reader:
            # Handle malformed rows where fields might be None
            caption = row.get("caption")
            if caption is None or not caption.strip():
                continue
            caption = caption.strip()

            clip_id = row.get("audiocap_id", "")
            if clip_id:
                clip_id = clip_id.strip()
            if not clip_id:
                youtube_id = row.get("youtube_id", "")
                start_time = row.get("start_time", "")
                if not youtube_id or not start_time:
                    continue
                clip_id = f"{youtube_id}_{start_time}"
            count = seen_per_clip.get(clip_id, 0)
            seen_per_clip[clip_id] = count + 1
            caption_key = f"{clip_id}_caption_{count:02d}"

            youtube_id = row.get("youtube_id", "")
            start_time = row.get("start_time", "")
            if not youtube_id or not start_time:
                continue
            path = _find_audio_caps_audio(audio_dir, youtube_id.strip(), start_time.strip())
            if path is None:
                missing_audio += 1
                continue
            entries.append(
                {
                    "clip_id": clip_id,
                    "caption": caption,
                    "caption_key": caption_key,
                    "audio_path": path,
                }
            )
        if missing_audio:
            print(f"[WARN] {missing_audio} AudioCaps entries missing audio files under {audio_dir}")
    if not entries:
        raise RuntimeError(f"No AudioCaps entries built from {csv_path}. Check audio dir {audio_dir}.")
    return entries


def _list_wavcaps_entries(csv_path: Path, audio_dir: Path) -> List[dict]:
    """Load WavCaps manifest CSV created by build_wavcaps_manifests.py"""
    entries: List[dict] = []
    missing_files = []
    
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = ["clip_id", "audio_relpath", "caption"]
        for col in required:
            if col not in reader.fieldnames:
                raise RuntimeError(f"WavCaps CSV missing column '{col}'. Header={reader.fieldnames}")
        
        for row in reader:
            clip_id = row["clip_id"].strip()
            caption = row["caption"].strip()
            if not caption:
                continue
            
            # audio_relpath is relative to audio_dir
            audio_relpath = row["audio_relpath"].strip()
            audio_path = audio_dir / audio_relpath
            
            if not audio_path.exists():
                missing_files.append(str(audio_path))
                continue
            
            # WavCaps uses caption_index from CSV (usually 0)
            caption_idx = int(row.get("caption_index", 0))
            caption_key = f"{clip_id}_caption_{caption_idx:02d}"
            
            entries.append({
                "clip_id": clip_id,
                "caption": caption,
                "caption_key": caption_key,
                "audio_path": audio_path,
            })
    
    if missing_files:
        print(f"[WARN] Skipped {len(missing_files)} WavCaps entries with missing audio files")
        if len(missing_files) <= 5:
            for f in missing_files:
                print(f"  - {f}")
    
    if not entries:
        raise RuntimeError(f"No WavCaps entries found from {csv_path}. Check audio dir {audio_dir}.")
    
    return entries


def load_corrupted_files(path: Optional[Path]) -> set[str]:
    """Load list of corrupted audio files to skip during training."""
    if path is None or not path.exists():
        return set()
    corrupted = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                corrupted.add(str(Path(line)))
    if corrupted:
        print(f"[INFO] Loaded {len(corrupted)} corrupted audio files from {path}")
    return corrupted


def load_blacklist_pairs(path: Optional[Path]) -> set[Tuple[str, str]]:
    if path is None:
        return set()
    if not path.exists():
        print(f"[WARN] Blacklist JSON not found: {path}")
        return set()
    data = json.loads(path.read_text())
    pairs = data.get("pairs", [])
    result = {tuple(sorted((entry["audio_a"], entry["audio_b"]))) for entry in pairs}
    print(f"[INFO] Loaded blacklist with {len(result)} pairs from {path}")
    return result


def _load_hard_negative_cache(
    jsonl_path: Optional[Path],
    caption_key_to_index: Dict[str, int],
    audio_stem_to_indices: Dict[str, List[int]],
    negatives_per_anchor: int,
) -> Dict[int, List[int]]:
    """
    Load hard negatives JSONL and map audio stems to dataset caption indices.
    If the file doesn't exist, returns empty dict (no hard negatives).

    The hard negatives file contains audio file stems (e.g., 'youtube_id_start_time'),
    but we need to map them to caption indices since each audio can have multiple captions.
    """
    if jsonl_path is None:
        return {}
    if not jsonl_path.exists():
        print(f"[WARNING] Hard-negative JSONL not found: {jsonl_path}. Training without hard negatives.")
        return {}

    mapping: Dict[int, List[int]] = {}
    missing_anchor = 0
    missing_neg = 0
    total_anchor_audios = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # Get the audio stem from the hard negatives file
            anchor_audio_stem = Path(record.get("audio", record.get("text", ""))).stem
            neg_audio_stems = [Path(n).stem for n in record.get("negatives", [])]

            total_anchor_audios += 1

            # Get all caption indices for this anchor audio
            anchor_indices = audio_stem_to_indices.get(anchor_audio_stem, [])
            if not anchor_indices:
                missing_anchor += 1
                continue

            # Collect all negative caption indices
            all_neg_indices: List[int] = []
            for neg_audio_stem in neg_audio_stems:
                neg_indices = audio_stem_to_indices.get(neg_audio_stem, [])
                if not neg_indices:
                    missing_neg += 1
                    continue
                all_neg_indices.extend(neg_indices)

            # Assign the same hard negatives to all captions of this anchor audio
            if all_neg_indices:
                if negatives_per_anchor > 0:
                    all_neg_indices = all_neg_indices[:negatives_per_anchor]
                for anchor_idx in anchor_indices:
                    mapping[anchor_idx] = all_neg_indices

    print(
        f"[INFO] Hard negatives loaded: {len(mapping)}/{len(caption_key_to_index)} caption anchors "
        f"from {total_anchor_audios} audio files "
        f"(missing {missing_anchor} audio anchors, {missing_neg} audio negatives)."
    )
    return mapping


class CaptionAudioDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        train_csv: Path,
        audio_dir: Path,
        hard_neg_json: Optional[Path],
        hard_negatives_per_anchor: int,
    ) -> None:
        if dataset not in {"clotho", "audiocaps", "wavcaps"}:
            raise ValueError("dataset must be 'clotho', 'audiocaps', or 'wavcaps'")
        self.dataset = dataset
        if dataset == "clotho":
            self.entries = _list_clotho_entries(train_csv, audio_dir)
        elif dataset == "audiocaps":
            self.entries = _list_audiocaps_entries(train_csv, audio_dir)
        else:  # wavcaps
            self.entries = _list_wavcaps_entries(train_csv, audio_dir)

        self.caption_key_to_index = {entry["caption_key"]: idx for idx, entry in enumerate(self.entries)}

        # Build audio stem to caption indices mapping for hard negatives
        # Hard negatives are indexed by audio file stems, but we need caption indices
        audio_stem_to_indices: Dict[str, List[int]] = {}
        for idx, entry in enumerate(self.entries):
            audio_stem = Path(entry["audio_path"]).stem
            if audio_stem not in audio_stem_to_indices:
                audio_stem_to_indices[audio_stem] = []
            audio_stem_to_indices[audio_stem].append(idx)

        self.hard_negatives = _load_hard_negative_cache(
            hard_neg_json, self.caption_key_to_index, audio_stem_to_indices, hard_negatives_per_anchor
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        neg_indices = self.hard_negatives.get(idx, [])
        return {
            "index": idx,
            "caption": entry["caption"],
            "audio_path": entry["audio_path"],
            "clip_id": entry["clip_id"],
            "caption_key": entry["caption_key"],
            "neg_indices": neg_indices,
        }


def collate_batch(batch: Sequence[dict], dataset: CaptionAudioDataset, corrupted_files: Optional[set[str]] = None) -> Optional[dict]:
    """Collate anchors and gather doc audio paths (positives first).
    Returns None if batch contains corrupted files.
    """
    captions = [item["caption"] for item in batch]
    pos_audio_paths = [Path(item["audio_path"]) for item in batch]
    pos_indices = list(range(len(batch)))
    query_clip_ids = [item.get("clip_id", "") for item in batch]
    
    # Pre-check for corrupted files to skip batch early
    if corrupted_files:
        for p in pos_audio_paths:
            if str(p) in corrupted_files:
                print(f"[WARN] Skipping batch containing corrupted file: {p}")
                return None

    doc_paths: List[Path] = []
    doc_clip_ids: List[str] = []
    seen: set[str] = set()
    for p, item in zip(pos_audio_paths, batch):
        doc_paths.append(p)
        doc_clip_ids.append(item.get("clip_id", ""))
        seen.add(str(p))

    for item in batch:
        for neg_idx in item["neg_indices"]:
            neg_path = Path(dataset.entries[neg_idx]["audio_path"])
            neg_clip = dataset.entries[neg_idx].get("clip_id", "")
            key = str(neg_path)
            if key not in seen:
                doc_paths.append(neg_path)
                doc_clip_ids.append(neg_clip)
                seen.add(key)

    return {
        "captions": captions,
        "doc_paths": doc_paths,
        "positive_doc_indices": pos_indices,
        "anchor_indices": [item["index"] for item in batch],
        "query_clip_ids": query_clip_ids,
        "doc_clip_ids": doc_clip_ids,
    }


def build_blacklist_mask(
    query_clip_ids: Sequence[str],
    doc_clip_ids: Sequence[str],
    blacklist_pairs: set[Tuple[str, str]],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if not blacklist_pairs:
        return None
    mask = torch.zeros((len(query_clip_ids), len(doc_clip_ids)), dtype=torch.bool, device=device)
    any_mask = False
    for qi, q_id in enumerate(query_clip_ids):
        if not q_id:
            continue
        for dj, d_id in enumerate(doc_clip_ids):
            if qi == dj:  # skip positives
                continue
            if not d_id:
                continue
            key = tuple(sorted((q_id, d_id)))
            if key in blacklist_pairs:
                mask[qi, dj] = True
                any_mask = True
    if any_mask:
        return mask
    return None


# ---------------------------------------------------------------------------
# Projection heads
# ---------------------------------------------------------------------------


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, proj_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.dropout(x)
        x = self.norm(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


class PredictionHead(nn.Module):
    """Lightweight predictor used by the SLAP loss (maps projections -> queries)."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1 - F.cosine_similarity(a, b, dim=-1).mean()


def compute_slap_loss(
    q_text: torch.Tensor,
    q_audio: torch.Tensor,
    z_audio: torch.Tensor,
    z_text: torch.Tensor,
    lambda_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Balanced combination of inter/intra-modal cosine losses."""
    q_text = F.normalize(q_text, p=2, dim=-1)
    q_audio = F.normalize(q_audio, p=2, dim=-1)
    z_audio = F.normalize(z_audio, p=2, dim=-1)
    z_text = F.normalize(z_text, p=2, dim=-1)

    inter_loss = 0.5 * (cosine_distance(q_audio, z_text) + cosine_distance(q_text, z_audio))
    intra_loss = 0.5 * (cosine_distance(q_audio, z_audio) + cosine_distance(q_text, z_text))
    total = lambda_weight * inter_loss + (1 - lambda_weight) * intra_loss
    logs = {
        "slap/inter_loss": inter_loss.detach().item(),
        "slap/intra_loss": intra_loss.detach().item(),
    }
    return total, logs


def update_moving_average(target: nn.Module, source: nn.Module, momentum: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(momentum).add_(source_param.data, alpha=1.0 - momentum)


# ---------------------------------------------------------------------------
# Omni-Embed LoRA trainer
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    dataset: str
    train_csv: Path
    val_csv: Path
    audio_dir: Path
    val_audio_dir: Optional[Path]
    repo_id: Optional[str]
    local_path: Optional[str]
    device: str
    query_prefix: str
    passage_prefix: str
    lora_targets: List[str]
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    projection_dim: int
    projection_dropout: float
    hard_neg_json: Optional[Path]
    hard_negatives_per_anchor: int
    blacklist_json: Optional[Path]
    temperature: float
    symmetric_loss: bool
    loss_type: str
    slap_lambda: float
    slap_ema_momentum: float
    slap_predictor_hidden_dim: int
    slap_predictor_dropout: float
    batch_size: int
    grad_accum: int
    epochs: int
    eval_every: int
    learning_rate: float
    weight_decay: float
    output_dir: Path
    init_checkpoint: Optional[Path]
    early_stop_patience: int
    save_every_steps: int
    wandb_project: Optional[str]
    wandb_entity: Optional[str]
    wandb_group: Optional[str]
    wandb_run_name: Optional[str]
    wandb_tags: Optional[List[str]]
    use_amp: bool = True


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser("Train Omni-Embed with LoRA for audio-text retrieval")

    parser.add_argument("--dataset", choices=["clotho", "audiocaps", "wavcaps"], required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True, help="Audio directory for training data")
    parser.add_argument("--val-audio-dir", type=Path, default=None, help="Audio directory for validation data (defaults to --audio-dir if not specified)")
    parser.add_argument("--hard-neg-json", type=Path, default=None)
    parser.add_argument("--hard-negatives-per-anchor", type=int, default=4)
    parser.add_argument("--blacklist-json", type=Path, default=None, help="JSON containing high-similarity audio pairs to ignore")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--repo-id", type=str, default=None)
    group.add_argument("--local-path", type=str, default=None)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--query-prefix", type=str, default="query:")
    parser.add_argument("--passage-prefix", type=str, default="passage:")

    parser.add_argument("--lora-targets", type=str, default="q_proj,k_proj,v_proj,o_proj,qkv,out_proj")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--projection-dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--symmetric-loss", action="store_true", default=True)
    parser.add_argument("--loss-type", type=str, choices=["infonce", "slap"], default="infonce",
                        help="InfoNCE (contrastive) or SLAP-style positive-only loss.")
    parser.add_argument("--slap-lambda", type=float, default=0.5,
                        help="Balance between inter- and intra-modal SLAP losses (lambda in paper).")
    parser.add_argument("--slap-ema-momentum", type=float, default=0.996,
                        help="EMA momentum for target heads when using SLAP.")
    parser.add_argument("--slap-predictor-hidden-dim", type=int, default=0,
                        help="Hidden dimension for SLAP predictor MLP (0 = use projection dim).")
    parser.add_argument("--slap-predictor-dropout", type=float, default=0.1,
                        help="Dropout inside the SLAP predictor MLP.")

    parser.add_argument("--batch-size", type=int, default=2, help="Micro batch size (per gradient step).")
    parser.add_argument("--grad-accum", type=int, default=512, help="Gradient accumulation steps.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (autoselects per dataset if omitted).")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, default=None)

    parser.add_argument("--early-stop-patience", type=int, default=10, help="Early stopping patience (evaluation steps)")
    parser.add_argument("--save-every-steps", type=int, default=100, help="Periodic checkpoint interval (0 disables)")
    parser.add_argument("--wandb-project", type=str, default="omni-embed-audio-retrieval")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated tags")
    parser.add_argument("--no-amp", action="store_true")

    args = parser.parse_args()

    wandb_tags = [t.strip() for t in args.wandb_tags.split(",")] if args.wandb_tags else None
    lora_targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]

    # Default epochs: AudioCaps=5, Clotho=10, WavCaps=15
    if args.dataset == "audiocaps":
        epochs = 5
    elif args.dataset == "wavcaps":
        epochs = 15
    else:  # clotho
        epochs = 10
    epochs = args.epochs if args.epochs is not None else epochs
    slap_hidden = args.slap_predictor_hidden_dim or args.projection_dim

    return TrainConfig(
        dataset=args.dataset,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        audio_dir=args.audio_dir,
        val_audio_dir=args.val_audio_dir if args.val_audio_dir else args.audio_dir,
        repo_id=args.repo_id,
        local_path=args.local_path,
        device=args.device,
        query_prefix=args.query_prefix,
        passage_prefix=args.passage_prefix,
        lora_targets=lora_targets,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        projection_dim=args.projection_dim,
        projection_dropout=args.projection_dropout,
        hard_neg_json=args.hard_neg_json,
        hard_negatives_per_anchor=args.hard_negatives_per_anchor,
        blacklist_json=args.blacklist_json,
        temperature=args.temperature,
        symmetric_loss=args.symmetric_loss,
        loss_type=args.loss_type,
        slap_lambda=args.slap_lambda,
        slap_ema_momentum=args.slap_ema_momentum,
        slap_predictor_hidden_dim=slap_hidden,
        slap_predictor_dropout=args.slap_predictor_dropout,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=epochs,
        eval_every=args.eval_every,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir,
        init_checkpoint=args.init_checkpoint,
        early_stop_patience=args.early_stop_patience,
        save_every_steps=max(0, args.save_every_steps),
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=wandb_tags,
        use_amp=not args.no_amp,
    )


def setup_wandb(cfg: TrainConfig) -> Optional[str]:
    if cfg.wandb_project is None:
        return None
    if wandb is None:
        print(
            "[WARN] Weights & Biases is unavailable or incomplete; "
            "set WANDB_PROJECT='' to silence this warning if logging is unnecessary.",
            flush=True,
        )
        return None
    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        group=cfg.wandb_group,
        name=cfg.wandb_run_name,
        tags=cfg.wandb_tags,
        config=asdict(cfg),
    )
    return run.id


def build_adapter_and_model(cfg: TrainConfig) -> Tuple[OmniEmbedAdapter, torch.nn.Module, AutoProcessor]:
    adapter = OmniEmbedAdapter(
        repo_id=cfg.repo_id or "",
        local_path=cfg.local_path,
        device=cfg.device,
        cache_dir=None,
        trust_remote_code=True,
        text_max_length=512,
        query_prefix=cfg.query_prefix,
        passage_prefix=cfg.passage_prefix,
    )
    model = adapter.get_underlying_model()
    processor = adapter.processor
    return adapter, model, processor


def attach_lora(model: torch.nn.Module, cfg: TrainConfig) -> torch.nn.Module:
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_targets,
        bias="none",
    )
    return get_peft_model(model, lora_cfg)


def build_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader, CaptionAudioDataset, CaptionAudioDataset, set[str]]:
    # Load corrupted files list
    corrupted_files_path = Path("WavCaps/corrupted_files.txt") if cfg.dataset == "wavcaps" else None
    corrupted_files = load_corrupted_files(corrupted_files_path)
    
    train_dataset = CaptionAudioDataset(
        dataset=cfg.dataset,
        train_csv=cfg.train_csv,
        audio_dir=cfg.audio_dir,
        hard_neg_json=cfg.hard_neg_json,
        hard_negatives_per_anchor=cfg.hard_negatives_per_anchor,
    )
    val_dataset = CaptionAudioDataset(
        dataset=cfg.dataset,
        train_csv=cfg.val_csv,
        audio_dir=cfg.val_audio_dir,
        hard_neg_json=None,
        hard_negatives_per_anchor=0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda batch: collate_batch(batch, train_dataset, corrupted_files),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda batch: collate_batch(batch, val_dataset),
    )
    return train_loader, val_loader, train_dataset, val_dataset, corrupted_files


def mean_pool_last_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    masked = hidden * mask
    summed = masked.sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    pooled = summed / denom
    return pooled


def encode_batch(
    adapter: OmniEmbedAdapter,
    model: torch.nn.Module,
    processor: AutoProcessor,
    texts: Optional[List[str]],
    audio_paths: Optional[List[Path]],
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[List[int]]]:
    """
    Encode either text-only or text+audio batch (grad enabled).
    Returns (None, None) if all audio files in the batch are corrupted.
    The second return value contains the indices of inputs that produced valid embeddings.
    """
    if audio_paths is None:
        assert texts is not None
        conversations = adapter._build_text_messages(texts, adapter.query_prefix)
        chat_texts = adapter._apply_chat_template(conversations)
        batch_inputs = _BatchInputs(text=chat_texts)
        processor_batch = adapter._prepare_batch(
            batch_inputs,
            text_kwargs={"truncation": True, "padding": True, "max_length": adapter.text_max_length},
        )
        valid_indices = list(range(len(texts)))
    else:
        conversations = adapter._build_audio_messages([str(p) for p in audio_paths], adapter.passage_prefix)
        chat_texts = adapter._apply_chat_template(conversations)
        loader = adapter._audio_loader
        audio_arrays = []
        valid_indices = []
        
        for idx, path in enumerate(audio_paths):
            try:
                wav = loader(path, adapter.audio_sampling_rate)
                audio_arrays.append(wav)
                valid_indices.append(idx)
            except Exception as e:
                print(f"[WARN] Skipping corrupted audio file {path}: {e}")
                continue

        # If all files are corrupted, return None
        if not audio_arrays:
            print(f"[ERROR] All {len(audio_paths)} audio files in batch are corrupted. Skipping batch.")
            return None, None

        # Filter conversations and chat_texts to only include valid files
        if len(valid_indices) < len(audio_paths):
            conversations = [conversations[i] for i in valid_indices]
            chat_texts = adapter._apply_chat_template(conversations)
        
        batch_inputs = _BatchInputs(text=chat_texts, audio_arrays=audio_arrays)
        processor_batch = adapter._prepare_batch(
            batch_inputs,
            text_kwargs={"truncation": True, "padding": True, "max_length": max(adapter.text_max_length, 32768)},
        )


    outputs = model(
        **processor_batch,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    hidden = outputs.hidden_states[-1]
    attention_mask = processor_batch["attention_mask"]
    pooled = mean_pool_last_hidden(hidden, attention_mask)
    return pooled, valid_indices


def compute_recall_at_k(sim_matrix: torch.Tensor, ks: Sequence[int], ground_truth_indices: Optional[torch.Tensor] = None) -> Dict[int, float]:
    """
    sim_matrix: [num_queries, num_docs]
    ground_truth_indices: [num_queries] - the index of the positive document for each query
                          If None, assumes positives are on the diagonal (query i -> doc i).
    """
    recalls = {}
    sorted_indices = torch.argsort(sim_matrix, dim=1, descending=True)

    if ground_truth_indices is None:
        # Default: assume diagonal positives (one-to-one mapping)
        positives = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    else:
        positives = ground_truth_indices

    for k in ks:
        topk = sorted_indices[:, :k]
        correct = (topk == positives.unsqueeze(1)).any(dim=1).float().mean().item()
        recalls[k] = correct
    return recalls


def evaluate(
    cfg: TrainConfig,
    adapter: OmniEmbedAdapter,
    model: torch.nn.Module,
    processor: AutoProcessor,
    text_head: nn.Module,
    audio_head: nn.Module,
    dataset: CaptionAudioDataset,
    batch_size: int,
) -> Dict[str, float]:
    """
    Evaluate with full candidate set (all validation samples).
    Handles multiple captions per audio file correctly.
    This provides realistic retrieval metrics matching test-time conditions.
    """
    model.eval()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Collect all text queries and unique audio documents
    all_texts: List[str] = []
    all_audio_paths: List[Path] = []
    query_to_doc_index: List[int] = []  # Maps query index to its ground-truth doc index

    # First, collect unique audio files and build the mapping
    audio_path_to_index: Dict[str, int] = {}
    unique_audio_paths: List[Path] = []

    for entry in dataset.entries:
        all_texts.append(entry["caption"])
        audio_path = Path(entry["audio_path"])
        audio_key = str(audio_path)

        # Get or assign index for this audio file
        if audio_key not in audio_path_to_index:
            audio_path_to_index[audio_key] = len(unique_audio_paths)
            unique_audio_paths.append(audio_path)

        # Map this query to its corresponding audio document index
        query_to_doc_index.append(audio_path_to_index[audio_key])

    print(f"[EVAL] Dataset: {len(all_texts)} text queries, {len(unique_audio_paths)} unique audio files")
    print(f"[EVAL] Encoding {len(all_texts)} text queries...")
    text_embeddings: List[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i + batch_size]
            text_emb, _ = encode_batch(adapter, model, processor, batch_texts, None, device)
            text_embeddings.append(text_emb.detach().cpu())

    print(f"[EVAL] Encoding {len(unique_audio_paths)} unique audio documents...")
    doc_embeddings: List[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(unique_audio_paths), batch_size):
            batch_paths = unique_audio_paths[i:i + batch_size]
            doc_emb, _ = encode_batch(adapter, model, processor, None, batch_paths, device)
            doc_embeddings.append(doc_emb.detach().cpu())

    model.train()

    # Concatenate and normalize all embeddings
    text_emb = torch.cat(text_embeddings, dim=0)
    doc_emb = torch.cat(doc_embeddings, dim=0)

    head_device = next(text_head.parameters()).device
    head_dtype = next(text_head.parameters()).dtype
    with torch.no_grad():
        text_proj = text_head(text_emb.to(device=head_device, dtype=head_dtype))
        doc_proj = audio_head(doc_emb.to(device=head_device, dtype=head_dtype))

    text_proj = F.normalize(text_proj, p=2, dim=-1)
    doc_proj = F.normalize(doc_proj, p=2, dim=-1)

    # Compute similarity over ALL unique audio candidates
    print(f"[EVAL] Computing similarity matrix: {text_proj.shape[0]} queries x {doc_proj.shape[0]} documents...")
    sim = text_proj @ doc_proj.T

    # Convert ground truth mapping to tensor
    gt_indices = torch.tensor(query_to_doc_index, dtype=torch.long, device=sim.device)

    logits = sim / cfg.temperature
    val_loss = F.cross_entropy(logits, gt_indices)

    recalls = compute_recall_at_k(sim, ks=[1, 5, 10], ground_truth_indices=gt_indices)
    print(
        f"[EVAL] Loss: {val_loss.item():.4f}, "
        f"Recall@1: {recalls[1]:.4f}, Recall@5: {recalls[5]:.4f}, Recall@10: {recalls[10]:.4f}"
    )
    metrics = {"loss": val_loss.item()}
    metrics.update({f"R@{k}": v for k, v in recalls.items()})

    model.train()
    text_head.train()
    audio_head.train()
    return metrics


def save_checkpoint(
    path: Path,
    text_head: nn.Module,
    audio_head: nn.Module,
    peft_model: torch.nn.Module,
    cfg: TrainConfig,
    global_step: int,
    metrics: Dict[str, float],
    extra_state: Optional[Dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "text_head": text_head.state_dict(),
        "audio_head": audio_head.state_dict(),
        "lora_state_dict": peft_model.state_dict(),
        "config": asdict(cfg),
        "metrics": metrics,
        "global_step": global_step,
    }
    if extra_state:
        state.update(extra_state)
    torch.save(state, path)


def train(cfg: TrainConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = setup_wandb(cfg)

    adapter, base_model, processor = build_adapter_and_model(cfg)
    base_model.train()
    peft_model = attach_lora(base_model, cfg)
    adapter.set_underlying_model(peft_model)
    peft_model = adapter.get_underlying_model()
    peft_model.train()

    embed_dim = resolve_text_hidden_size(peft_model)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    text_head = ProjectionHead(embed_dim, cfg.projection_dim, cfg.projection_dropout).to(device)
    audio_head = ProjectionHead(embed_dim, cfg.projection_dim, cfg.projection_dropout).to(device)
    text_head_ema: Optional[ProjectionHead] = None
    audio_head_ema: Optional[ProjectionHead] = None
    text_predictor: Optional[PredictionHead] = None
    audio_predictor: Optional[PredictionHead] = None

    if cfg.loss_type == "slap":
        text_head_ema = copy.deepcopy(text_head).to(device)
        audio_head_ema = copy.deepcopy(audio_head).to(device)
        for module in (text_head_ema, audio_head_ema):
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = False
        text_predictor = PredictionHead(
            cfg.projection_dim,
            cfg.slap_predictor_hidden_dim,
            cfg.slap_predictor_dropout,
        ).to(device)
        audio_predictor = PredictionHead(
            cfg.projection_dim,
            cfg.slap_predictor_hidden_dim,
            cfg.slap_predictor_dropout,
        ).to(device)

    def current_slap_state() -> Optional[Dict[str, Any]]:
        if cfg.loss_type != "slap":
            return None
        if any(
            module is None
            for module in (text_head_ema, audio_head_ema, text_predictor, audio_predictor)
        ):
            return None
        return {
            "text_head_ema": text_head_ema.state_dict(),
            "audio_head_ema": audio_head_ema.state_dict(),
            "text_predictor": text_predictor.state_dict(),
            "audio_predictor": audio_predictor.state_dict(),
        }

    # Load checkpoint if specified
    loaded_ckpt = None
    if cfg.init_checkpoint is not None and cfg.init_checkpoint.exists():
        print(f"[INFO] Loading checkpoint from {cfg.init_checkpoint}")
        loaded_ckpt = safe_torch_load(cfg.init_checkpoint, map_location=device)
        text_head.load_state_dict(loaded_ckpt['text_head'])
        audio_head.load_state_dict(loaded_ckpt['audio_head'])
        peft_model.load_state_dict(loaded_ckpt['lora_state_dict'], strict=False)
        if cfg.loss_type == "slap":
            if text_head_ema is not None:
                if "text_head_ema" in loaded_ckpt:
                    text_head_ema.load_state_dict(loaded_ckpt["text_head_ema"])
                else:
                    text_head_ema.load_state_dict(text_head.state_dict())
            if audio_head_ema is not None:
                if "audio_head_ema" in loaded_ckpt:
                    audio_head_ema.load_state_dict(loaded_ckpt["audio_head_ema"])
                else:
                    audio_head_ema.load_state_dict(audio_head.state_dict())
            if text_predictor is not None and "text_predictor" in loaded_ckpt:
                text_predictor.load_state_dict(loaded_ckpt["text_predictor"])
            if audio_predictor is not None and "audio_predictor" in loaded_ckpt:
                audio_predictor.load_state_dict(loaded_ckpt["audio_predictor"])
        print(f"[INFO] Checkpoint loaded successfully. Previous metrics: {loaded_ckpt.get('metrics', {})}")
        print(f"[INFO] Checkpoint was saved at global step: {loaded_ckpt.get('global_step', 'N/A')}")
    elif cfg.init_checkpoint is not None:
        print(f"[WARN] Checkpoint path specified but not found: {cfg.init_checkpoint}")

    params = list(text_head.parameters()) + list(audio_head.parameters())
    if cfg.loss_type == "slap" and text_predictor is not None and audio_predictor is not None:
        params.extend(list(text_predictor.parameters()))
        params.extend(list(audio_predictor.parameters()))
    params.extend(p for p in peft_model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    use_amp = cfg.use_amp and amp_device == "cuda"
    scaler = torch.amp.GradScaler(amp_device, enabled=use_amp)

    train_loader, val_loader, train_dataset, val_dataset, corrupted_files = build_dataloaders(cfg)
    blacklist_pairs = load_blacklist_pairs(cfg.blacklist_json)
    total_steps = cfg.epochs * math.ceil(len(train_dataset) / (cfg.batch_size))

    # Initialize best metrics from loaded checkpoint if available
    best_val_loss = float("inf")
    best_recall = 0.0
    if loaded_ckpt is not None:
        ckpt_metrics = loaded_ckpt.get('metrics', {})
        if 'loss' in ckpt_metrics:
            best_val_loss = ckpt_metrics['loss']
            print(f"[INFO] Initialized best_val_loss from checkpoint: {best_val_loss:.4f}")
        if 'R@10' in ckpt_metrics:
            best_recall = ckpt_metrics['R@10']
            print(f"[INFO] Initialized best_recall from checkpoint: {best_recall:.4f}")

    best_path = cfg.output_dir / "best.pt"
    early_stop_counter = 0
    early_stopped = False

    global_step = 0
    accum_steps = 0
    peft_model.train()

    for epoch in range(cfg.epochs):
        if early_stopped:
            break
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}", leave=True)
        for batch in progress:
            # Skip None batches (corrupted files filtered by collate_fn)
            if batch is None:
                continue
                
            captions = batch["captions"]
            doc_paths = [Path(p) for p in batch["doc_paths"]]
            query_clip_ids = batch.get("query_clip_ids", [])
            doc_clip_ids = batch.get("doc_clip_ids", [])
            pos_doc_count = len(batch["positive_doc_indices"])
            doc_paths_forward = doc_paths if cfg.loss_type == "infonce" else doc_paths[:pos_doc_count]
            doc_clip_ids_forward = doc_clip_ids if cfg.loss_type == "infonce" else doc_clip_ids[:pos_doc_count]
            extra_logs: Dict[str, float] = {}

            with torch.amp.autocast(amp_device, enabled=use_amp):
                text_embeddings, _ = encode_batch(adapter, peft_model, processor, captions, None, device)
                doc_embeddings, valid_doc_indices = encode_batch(
                    adapter, peft_model, processor, None, doc_paths_forward, device
                )

                # Skip batch if all audio files are corrupted
                if doc_embeddings is None or text_embeddings is None:
                    print(f"[WARN] Skipping batch due to corrupted files")
                    continue

                original_pos_count = pos_doc_count
                if valid_doc_indices is None:
                    valid_doc_indices = list(range(len(doc_paths_forward)))
                total_docs_expected = len(doc_paths_forward)
                if len(valid_doc_indices) != total_docs_expected:
                    valid_doc_set = set(valid_doc_indices)
                    missing_pos_indices = [idx for idx in range(original_pos_count) if idx not in valid_doc_set]
                    if missing_pos_indices:
                        missing_paths = [str(doc_paths_forward[idx]) for idx in missing_pos_indices]
                        print(
                            "[WARN] Dropping corrupted positive audios: "
                            + ", ".join(missing_paths)
                        )
                        if corrupted_files is not None:
                            corrupted_files.update(missing_paths)

                    keep_pos_indices = [idx for idx in range(original_pos_count) if idx in valid_doc_set]
                    if not keep_pos_indices:
                        print("[WARN] No valid positives left in batch after filtering; skipping batch.")
                        continue

                    if len(keep_pos_indices) != original_pos_count:
                        keep_pos_tensor = torch.tensor(keep_pos_indices, device=text_embeddings.device)
                        text_embeddings = text_embeddings.index_select(0, keep_pos_tensor)
                        captions = [captions[idx] for idx in keep_pos_indices]
                        query_clip_ids = [query_clip_ids[idx] for idx in keep_pos_indices]
                        pos_doc_count = len(keep_pos_indices)

                    idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(valid_doc_indices)}
                    pos_tensor = torch.tensor([idx_map[idx] for idx in keep_pos_indices], device=doc_embeddings.device)
                    doc_pos_embeddings = doc_embeddings.index_select(0, pos_tensor)

                    neg_orig_indices = [idx for idx in valid_doc_indices if idx >= original_pos_count]
                    if neg_orig_indices:
                        neg_tensor = torch.tensor(
                            [idx_map[idx] for idx in neg_orig_indices],
                            device=doc_embeddings.device,
                        )
                        doc_neg_embeddings = doc_embeddings.index_select(0, neg_tensor)
                        doc_embeddings = torch.cat([doc_pos_embeddings, doc_neg_embeddings], dim=0)
                        doc_clip_ids_forward = [doc_clip_ids_forward[idx] for idx in keep_pos_indices] + [
                            doc_clip_ids_forward[idx] for idx in neg_orig_indices
                        ]
                    else:
                        doc_embeddings = doc_pos_embeddings
                        doc_clip_ids_forward = [doc_clip_ids_forward[idx] for idx in keep_pos_indices]
                else:
                    keep_pos_indices = list(range(pos_doc_count))

                query_proj = text_head(text_embeddings)
                doc_proj = audio_head(doc_embeddings)

                if cfg.loss_type == "slap" and text_predictor is not None and audio_predictor is not None \
                        and text_head_ema is not None and audio_head_ema is not None:
                    text_pred = text_predictor(query_proj)
                    audio_pred = audio_predictor(doc_proj)
                    with torch.no_grad():
                        text_target = text_head_ema(text_embeddings.detach())
                        audio_target = audio_head_ema(doc_embeddings.detach())
                    loss, extra_logs = compute_slap_loss(
                        text_pred,
                        audio_pred,
                        audio_target,
                        text_target,
                        cfg.slap_lambda,
                    )
                else:
                    logits = (query_proj @ doc_proj.T) / cfg.temperature
                    blacklist_mask = build_blacklist_mask(
                        query_clip_ids, doc_clip_ids_forward, blacklist_pairs, device
                    )
                    if blacklist_mask is not None:
                        logits = logits.masked_fill(blacklist_mask, float("-inf"))
                    labels = torch.arange(pos_doc_count, device=device)
                    loss_q = F.cross_entropy(logits[:pos_doc_count], labels)

                    if cfg.symmetric_loss:
                        logits_doc = (doc_proj[:pos_doc_count] @ query_proj.T) / cfg.temperature
                        if blacklist_mask is not None:
                            mask_doc = blacklist_mask[:pos_doc_count, :pos_doc_count]
                            if mask_doc.numel() > 0:
                                logits_doc = logits_doc.masked_fill(mask_doc.T, float("-inf"))
                        loss_d = F.cross_entropy(logits_doc, labels)
                        loss = 0.5 * (loss_q + loss_d)
                    else:
                        loss = loss_q



            loss = loss / cfg.grad_accum
            scaler.scale(loss).backward()
            accum_steps += 1

            if accum_steps >= cfg.grad_accum:
                scaler.step(optimizer)
                scaler.update()
                if cfg.loss_type == "slap" and text_head_ema is not None and audio_head_ema is not None:
                    update_moving_average(text_head_ema, text_head, cfg.slap_ema_momentum)
                    update_moving_average(audio_head_ema, audio_head, cfg.slap_ema_momentum)
                optimizer.zero_grad(set_to_none=True)
                accum_steps = 0
                global_step += 1

                train_loss = loss.item() * cfg.grad_accum
                if wandb is not None and run_id is not None:
                    log_payload = {"train/loss": train_loss}
                    if extra_logs:
                        log_payload.update({f"train/{k}": v for k, v in extra_logs.items()})
                    wandb.log(log_payload, step=global_step)

                if cfg.save_every_steps and global_step % cfg.save_every_steps == 0:
                    ckpt_path = cfg.output_dir / "checkpoints" / f"step_{global_step}.pt"
                    save_checkpoint(
                        ckpt_path,
                        text_head,
                        audio_head,
                        peft_model,
                        cfg,
                        global_step,
                        {"train_loss": train_loss},
                        extra_state=current_slap_state(),
                    )
                    print(f"[INFO] Saved periodic checkpoint: {ckpt_path}")

                if global_step % cfg.eval_every == 0:
                    metrics = evaluate(
                        cfg,
                        adapter,
                        peft_model,
                        processor,
                        text_head,
                        audio_head,
                        val_dataset,
                        cfg.batch_size,
                    )
                    val_loss = metrics["loss"]
                    recall10 = metrics.get("R@10", 0.0)
                    if wandb is not None and run_id is not None:
                        wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=global_step)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_recall = recall10
                        early_stop_counter = 0
                        save_checkpoint(
                            best_path,
                            text_head,
                            audio_head,
                            peft_model,
                            cfg,
                            global_step,
                            metrics,
                            extra_state=current_slap_state(),
                        )
                        print(
                            f"[INFO] Saved new best checkpoint to {best_path} "
                            f"(val_loss={val_loss:.4f}, R@10={recall10:.4f})"
                        )
                    else:
                        early_stop_counter += 1
                        print(
                            f"[INFO] No val-loss improvement "
                            f"({early_stop_counter}/{cfg.early_stop_patience}); "
                            f"current={val_loss:.4f} best={best_val_loss:.4f}"
                        )
                        if early_stop_counter >= cfg.early_stop_patience:
                            print(f"[INFO] Early stopping triggered after {early_stop_counter} evaluations without improvement")
                            early_stopped = True
                            break

        # end epoch

    # final evaluation
    if best_path.exists():
        print(
            f"[INFO] Training complete. Best checkpoint: {best_path} "
            f"(val_loss={best_val_loss:.4f}, R@10={best_recall:.4f})"
        )
    else:
        save_checkpoint(
            best_path,
            text_head,
            audio_head,
            peft_model,
            cfg,
            global_step,
            {"note": "final"},
            extra_state=current_slap_state(),
        )
        print(f"[INFO] Training complete. Saved last checkpoint to {best_path}")

    if wandb is not None and run_id is not None:
        wandb.finish()


if __name__ == "__main__":  # pragma: no cover
    train(parse_args())
