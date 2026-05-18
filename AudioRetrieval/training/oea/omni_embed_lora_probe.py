#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick probe to verify LoRA attachment on Omni-Embed (Nemotron-3B) via PEFT.

The script is now located inside the git-tracked OEA training package so we
can keep the probing utilities versioned alongside the main training entrypoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
except Exception as exc:  # pragma: no cover - dependency
    raise SystemExit("PEFT is required: pip install peft") from exc

from AudioRetrieval.models.omni_embed_adapter import OmniEmbedAdapter


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Probe LoRA attachment for Omni-Embed")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--repo-id", type=str, default=None)
    group.add_argument("--local-path", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument(
        "--targets",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,qkv,out_proj",
        help="Comma-separated leaf module names to target with LoRA",
    )
    ap.add_argument("--list-only", action="store_true", help="Only list candidate modules and exit")
    ap.add_argument("--dry-run-forward", action="store_true", help="Run a tiny encode_text forward pass")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    adapter = OmniEmbedAdapter(
        repo_id=args.repo_id or "",
        local_path=args.local_path,
        device=args.device,
        trust_remote_code=True,
    )
    model = adapter.get_underlying_model()

    leaf_to_full: dict[str, List[str]] = {}
    for name, _ in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        leaf_to_full.setdefault(leaf, []).append(name)

    print("[INFO] Discovered leaf module names (first 20 entries):")
    for leaf, names in list(leaf_to_full.items())[:20]:
        print(f"  - {leaf}: {len(names)} matches (e.g., {names[0]})")

    if args.list_only:
        return

    target_leafs = [leaf.strip() for leaf in args.targets.split(",") if leaf.strip()]
    target_modules = [leaf for leaf in target_leafs if leaf in leaf_to_full]
    if not target_modules:
        print(f"[WARN] None of the requested targets exist: {target_leafs}. Aborting.")
        sys.exit(1)

    print(f"[INFO] Attaching LoRA to leaf modules: {target_modules}")
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=target_modules,
        inference_mode=False,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"[INFO] Trainable params: {trainable:,} / {total:,} ({trainable / total:.4%})")

    adapter.set_underlying_model(peft_model)

    if args.dry_run_forward:
        texts = [
            "query: a person speaking while a car engine idles",
            "query: birds chirping in a forest with footsteps",
        ]
        emb = adapter.encode_text(texts, batch_size=2, device=args.device)
        print(f"[INFO] Dry-run encode_text OK: shape={tuple(emb.shape)}")

    print("[DONE] LoRA probe finished.")


if __name__ == "__main__":  # pragma: no cover
    main()
