#!/usr/bin/env python3
"""
Fix checkpoint validation metrics by re-running validation on the full dataset.
"""
import sys
import torch
from pathlib import Path

# Add project root to path
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
from peft import LoraConfig, TaskType, get_peft_model


def fix_checkpoint_metrics(checkpoint_path: Path, output_path: Path):
    """Load checkpoint, re-validate, and save with correct metrics."""

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    print(f"[INFO] Current (corrupted) metrics: {ckpt.get('metrics', {})}")
    print(f"[INFO] Global step: {ckpt.get('global_step', 'N/A')}")

    # Extract config
    cfg_dict = ckpt['config']

    # Build model and adapter
    print("\n[INFO] Building model and adapter...")
    adapter = OmniEmbedAdapter(
        repo_id=cfg_dict.get('repo_id', ''),
        local_path=cfg_dict.get('local_path'),
        device=str(device),
        cache_dir=None,
        trust_remote_code=True,
        text_max_length=512,
        query_prefix=cfg_dict.get('query_prefix', 'query:'),
        passage_prefix=cfg_dict.get('passage_prefix', 'passage:'),
    )

    model = adapter.get_underlying_model()
    processor = adapter.processor

    # Attach LoRA
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=cfg_dict.get('lora_rank', 16),
        lora_alpha=cfg_dict.get('lora_alpha', 32),
        lora_dropout=cfg_dict.get('lora_dropout', 0.05),
        target_modules=cfg_dict.get('lora_targets', ['q_proj', 'k_proj', 'v_proj', 'o_proj']),
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

    # Load weights
    print("[INFO] Loading checkpoint weights...")
    text_head.load_state_dict(ckpt['text_head'])
    audio_head.load_state_dict(ckpt['audio_head'])
    peft_model.load_state_dict(ckpt['lora_state_dict'], strict=False)

    # Load validation dataset with CORRECT audio directory
    print(f"\n[INFO] Loading validation dataset...")
    val_csv = Path(cfg_dict['val_csv'])
    val_audio_dir = Path(cfg_dict.get('val_audio_dir', cfg_dict['audio_dir']))

    print(f"  Val CSV: {val_csv}")
    print(f"  Val audio dir: {val_audio_dir}")

    val_dataset = CaptionAudioDataset(
        dataset=cfg_dict['dataset'],
        train_csv=val_csv,
        audio_dir=val_audio_dir,
        hard_neg_json=None,
        hard_negatives_per_anchor=0,
    )

    print(f"[INFO] Validation dataset size: {len(val_dataset)} samples")

    # Create minimal config for evaluation
    class EvalConfig:
        def __init__(self, device, temperature):
            self.device = str(device)
            self.temperature = temperature
            self.batch_size = 4

    eval_cfg = EvalConfig(device, cfg_dict.get('temperature', 0.07))

    # Run proper validation
    print("\n[INFO] Running validation on FULL dataset...")
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
        batch_size=4,
    )

    print("\n" + "="*70)
    print("CORRECTED VALIDATION RESULTS")
    print("="*70)
    print(f"Validation Loss: {metrics['loss']:.4f}")
    print(f"Recall@1:  {metrics.get('R@1', 0.0):.4f}")
    print(f"Recall@5:  {metrics.get('R@5', 0.0):.4f}")
    print(f"Recall@10: {metrics.get('R@10', 0.0):.4f}")
    print("="*70)

    # Update checkpoint with correct metrics
    ckpt['metrics'] = metrics

    # Save corrected checkpoint
    print(f"\n[INFO] Saving corrected checkpoint to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output_path)

    print(f"[INFO] Checkpoint corrected successfully!")
    print(f"\nOld metrics: loss={0.0:.4f}, R@1={1.0:.4f}, R@10={1.0:.4f}")
    print(f"New metrics: loss={metrics['loss']:.4f}, R@1={metrics.get('R@1', 0.0):.4f}, R@10={metrics.get('R@10', 0.0):.4f}")

    return metrics


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="Recompute metrics stored inside an OEA checkpoint.")
    p.add_argument("checkpoint", type=Path, help="Path to OEA .pt checkpoint")
    p.add_argument("--output", type=Path, default=None, help="Output path (default: overwrite input)")
    args = p.parse_args()
    fix_checkpoint_metrics(args.checkpoint, args.output or args.checkpoint)
