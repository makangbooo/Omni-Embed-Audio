#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate a saved Omni-Embed checkpoint on a chosen dataset.

This script mirrors the evaluation path used during training so we can
measure Recall@K / loss for checkpoints saved from the OEA (Omni-Embed Audio)
pipeline directly from the git-tracked training package.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]  # Add repository root to import path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AudioRetrieval.training.oea.train_omniembed_lora import (  # noqa: E402
    CaptionAudioDataset,
    OmniEmbedAdapter,
    ProjectionHead,
    evaluate,
    safe_torch_load,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Validate Omni-Embed checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint file")
    parser.add_argument(
        "--dataset",
        choices=["clotho", "audiocaps", "wavcaps"],
        default=None,
        help="Dataset name. Defaults to value saved in checkpoint config.",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=None,
        help="Validation CSV/manifest. Defaults to config['val_csv'] if omitted.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Directory containing validation audio. Defaults to config['val_audio_dir'] or config['audio_dir'].",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device string passed to torch / adapter")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation encodes")
    return parser.parse_args()


def _prepare_paths(
    cfg_dict: Dict[str, object],
    dataset: Optional[str],
    val_csv: Optional[Path],
    audio_dir: Optional[Path],
) -> Tuple[str, Path, Path]:
    dataset_name = dataset or cfg_dict.get("dataset")
    if not isinstance(dataset_name, str):
        raise ValueError("Dataset name must be provided via --dataset or stored in checkpoint config.")

    csv_path = val_csv or cfg_dict.get("val_csv")
    if not csv_path:
        raise ValueError(
            "Validation CSV is missing. Provide --val-csv or ensure 'val_csv' is stored in checkpoint config."
        )
    csv_path = Path(csv_path)

    audio_path = audio_dir or cfg_dict.get("val_audio_dir") or cfg_dict.get("audio_dir")
    if not audio_path:
        raise ValueError(
            "Validation audio directory is missing. Provide --audio-dir or ensure it exists in checkpoint config."
        )
    audio_path = Path(audio_path)

    return dataset_name, csv_path, audio_path


def run_validation(
    checkpoint_path: Path,
    dataset: Optional[str] = None,
    val_csv: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
    device: str = "cuda",
    batch_size: int = 4,
) -> Tuple[Dict[str, float], Dict[str, object], Dict[str, object]]:
    """
    Load the specified checkpoint, rebuild the adapter/heads, and run evaluation.
    Returns (metrics, checkpoint_state, checkpoint_config).
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    ckpt = safe_torch_load(checkpoint_path, map_location=torch_device)

    cfg_dict = ckpt.get("config", {}) or {}
    dataset_name, csv_path, audio_path = _prepare_paths(cfg_dict, dataset, val_csv, audio_dir)

    print(f"[INFO] Dataset: {dataset_name}")
    print(f"[INFO] Validation CSV: {csv_path}")
    print(f"[INFO] Validation audio dir: {audio_path}")

    adapter = OmniEmbedAdapter(
        repo_id=cfg_dict.get("repo_id") or "",
        local_path=cfg_dict.get("local_path"),
        device=device,
        cache_dir=None,
        trust_remote_code=True,
        text_max_length=cfg_dict.get("text_max_length", 512),
        query_prefix=cfg_dict.get("query_prefix", "query:"),
        passage_prefix=cfg_dict.get("passage_prefix", "passage:"),
    )
    model = adapter.get_underlying_model()
    processor = adapter.processor

    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=cfg_dict.get("lora_rank", 16),
        lora_alpha=cfg_dict.get("lora_alpha", 32),
        lora_dropout=cfg_dict.get("lora_dropout", 0.05),
        target_modules=cfg_dict.get("lora_targets", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias="none",
    )
    peft_model = get_peft_model(model, lora_cfg)
    adapter.set_underlying_model(peft_model)
    peft_model = adapter.get_underlying_model()

    embed_dim = peft_model.config.text_config.hidden_size
    projection_dim = cfg_dict.get("projection_dim", 512)
    projection_dropout = cfg_dict.get("projection_dropout", 0.1)
    text_head = ProjectionHead(embed_dim, projection_dim, projection_dropout).to(torch_device)
    audio_head = ProjectionHead(embed_dim, projection_dim, projection_dropout).to(torch_device)

    print("[INFO] Loading checkpoint weights into heads + LoRA adapters...")
    text_head.load_state_dict(ckpt["text_head"])
    audio_head.load_state_dict(ckpt["audio_head"])
    peft_model.load_state_dict(ckpt["lora_state_dict"], strict=False)

    val_dataset = CaptionAudioDataset(
        dataset=dataset_name,
        train_csv=csv_path,
        audio_dir=audio_path,
        hard_neg_json=None,
        hard_negatives_per_anchor=0,
    )
    print(f"[INFO] Validation dataset size: {len(val_dataset)} samples")

    eval_cfg = SimpleNamespace(device=device, temperature=cfg_dict.get("temperature", 0.07))

    metrics = evaluate(
        eval_cfg,
        adapter,
        peft_model,
        processor,
        text_head,
        audio_head,
        val_dataset,
        batch_size,
    )

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Validation Loss: {metrics['loss']:.4f}")
    print(f"Recall@1:  {metrics.get('R@1', 0.0):.4f}")
    print(f"Recall@5:  {metrics.get('R@5', 0.0):.4f}")
    print(f"Recall@10: {metrics.get('R@10', 0.0):.4f}")
    print("=" * 60)

    return metrics, ckpt, cfg_dict


def main() -> None:
    args = parse_args()
    run_validation(
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        val_csv=args.val_csv,
        audio_dir=args.audio_dir,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
