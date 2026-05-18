#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-run validation for a checkpoint and update the stored metrics.

Useful when the checkpoint was saved with placeholder metrics or when the
validation set changed (e.g., switching from AudioCaps to WavCaps manifests).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AudioRetrieval.training.oea.validate_checkpoint import run_validation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Recompute and store checkpoint validation metrics")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint to fix")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination for corrected checkpoint (defaults to in-place overwrite)",
    )
    parser.add_argument(
        "--dataset",
        choices=["clotho", "audiocaps", "wavcaps"],
        default=None,
        help="Dataset name override. Defaults to value stored in checkpoint config.",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=None,
        help="Validation CSV/manifest override.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Validation audio directory override.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics, ckpt, cfg_dict = run_validation(
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        val_csv=args.val_csv,
        audio_dir=args.audio_dir,
        device=args.device,
        batch_size=args.batch_size,
    )

    ckpt["metrics"] = metrics
    cfg = ckpt.get("config", cfg_dict)
    if args.dataset:
        cfg["dataset"] = args.dataset
    if args.val_csv:
        cfg["val_csv"] = str(args.val_csv)
    if args.audio_dir:
        cfg["val_audio_dir"] = str(args.audio_dir)
    ckpt["config"] = cfg

    output_path = args.output or args.checkpoint
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(ckpt, output_path)
    print(f"[INFO] Saved checkpoint with refreshed metrics to: {output_path}")


if __name__ == "__main__":
    main()
