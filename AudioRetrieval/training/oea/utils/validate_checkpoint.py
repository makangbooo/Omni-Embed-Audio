#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate a saved Omni-Embed checkpoint on validation data.
This script loads a checkpoint and runs evaluation without training.

Usage:
    python scripts/training/validate_checkpoint.py \\
        --checkpoint outputs/omniembed_lora/audiocaps_stage1/checkpoints/step_400.pt \\
        --dataset audiocaps \\
        --val-csv AudioCaps/v2_meta_data/val.csv \\
        --audio-dir AudioCaps/audiocaps_raw_audio \\
        --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AudioRetrieval.training.oea.train_omniembed_lora import (
    CaptionAudioDataset,
    OmniEmbedAdapter,
    ProjectionHead,
    evaluate,
)
from transformers import AutoProcessor



def parse_args():
    parser = argparse.ArgumentParser("Validate Omni-Embed checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint file")
    parser.add_argument("--dataset", choices=["clotho", "audiocaps", "wavcaps"], required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    print(f"[INFO] Checkpoint keys: {list(ckpt.keys())}")
    if 'metrics' in ckpt:
        print(f"[INFO] Saved metrics: {ckpt['metrics']}")
    if 'global_step' in ckpt:
        print(f"[INFO] Global step: {ckpt['global_step']}")

    # Extract config from checkpoint
    cfg_dict = ckpt.get('config', {})
    print(f"\n[INFO] Checkpoint configuration:")
    print(f"  - Dataset: {cfg_dict.get('dataset', 'N/A')}")
    print(f"  - Projection dim: {cfg_dict.get('projection_dim', 'N/A')}")
    print(f"  - LoRA rank: {cfg_dict.get('lora_rank', 'N/A')}")
    print(f"  - Query prefix: {cfg_dict.get('query_prefix', 'N/A')}")
    print(f"  - Passage prefix: {cfg_dict.get('passage_prefix', 'N/A')}")

    # Build model and adapter
    print("\n[INFO] Building model and adapter...")
    repo_id = cfg_dict.get('repo_id')
    local_path = cfg_dict.get('local_path')

    adapter = OmniEmbedAdapter(
        repo_id=repo_id or "",
        local_path=local_path,
        device=args.device,
        cache_dir=None,
        trust_remote_code=True,
        text_max_length=512,
        query_prefix=cfg_dict.get('query_prefix', 'query:'),
        passage_prefix=cfg_dict.get('passage_prefix', 'passage:'),
    )

    model = adapter.get_underlying_model()
    processor = adapter.processor

    # Attach LoRA
    from peft import LoraConfig, TaskType, get_peft_model

    lora_targets = cfg_dict.get('lora_targets', ['q_proj', 'k_proj', 'v_proj', 'o_proj'])
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=cfg_dict.get('lora_rank', 16),
        lora_alpha=cfg_dict.get('lora_alpha', 32),
        lora_dropout=cfg_dict.get('lora_dropout', 0.05),
        target_modules=lora_targets,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_cfg)
    adapter.set_underlying_model(peft_model)
    peft_model = adapter.get_underlying_model()

    # Create projection heads
    embed_dim = peft_model.config.text_config.hidden_size
    projection_dim = cfg_dict.get('projection_dim', 512)
    projection_dropout = cfg_dict.get('projection_dropout', 0.1)

    text_head = ProjectionHead(embed_dim, projection_dim, projection_dropout).to(device)
    audio_head = ProjectionHead(embed_dim, projection_dim, projection_dropout).to(device)

    # Load checkpoint weights
    print("\n[INFO] Loading checkpoint weights...")
    text_head.load_state_dict(ckpt['text_head'])
    audio_head.load_state_dict(ckpt['audio_head'])
    peft_model.load_state_dict(ckpt['lora_state_dict'], strict=False)
    print("[INFO] Checkpoint weights loaded successfully")

    # Create validation dataset
    print(f"\n[INFO] Loading validation dataset from {args.val_csv}...")
    val_dataset = CaptionAudioDataset(
        dataset=args.dataset,
        train_csv=args.val_csv,
        audio_dir=args.audio_dir,
        hard_neg_json=None,
        hard_negatives_per_anchor=0,
    )
    print(f"[INFO] Validation dataset size: {len(val_dataset)} samples")

    # Create a minimal config object for evaluation
    class EvalConfig:
        def __init__(self, device, temperature):
            self.device = device
            self.temperature = temperature
            self.batch_size = args.batch_size

    eval_cfg = EvalConfig(
        device=args.device,
        temperature=cfg_dict.get('temperature', 0.07)
    )

    # Run evaluation
    print("\n[INFO] Running validation...")
    peft_model.eval()
    text_head.eval()
    audio_head.eval()

    metrics = evaluate(
        eval_cfg,
        adapter,
        peft_model,
        processor,
        text_head,
        audio_head,
        val_dataset,
        args.batch_size,
    )

    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Validation Loss: {metrics['loss']:.4f}")
    print(f"Recall@1:  {metrics.get('R@1', 0.0):.4f}")
    print(f"Recall@5:  {metrics.get('R@5', 0.0):.4f}")
    print(f"Recall@10: {metrics.get('R@10', 0.0):.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
