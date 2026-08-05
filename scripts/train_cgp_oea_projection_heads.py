#!/usr/bin/env python3
"""Train the CGP-OEA MVP with only the two OEA projection heads trainable.

The official OEA backbone and its LoRA adapter are frozen.  Each positive
audio-caption pair is encoded once in the base (teacher) and LoRA (student)
hidden spaces.  Head training combines symmetric InfoNCE with KL distillation
of the teacher's cross-modal similarity distribution.  No UIQ, FiQA, or NQ
examples are used for training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("audiocaps", "clotho"), required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--val-audio-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--head-init-checkpoint", type=Path)
    parser.add_argument("--max-train-examples", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--text-encode-batch-size", type=int, default=8)
    parser.add_argument("--audio-encode-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--distill-temperature", type=float, default=0.07)
    parser.add_argument("--geometry-lambda", type=float, default=0.5)
    parser.add_argument("--pca-anchor-lambda", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def make_unique_batches(entries: Sequence[dict[str, Any]], batch_size: int, *, seed: int) -> list[list[int]]:
    """Return batches with at most one caption per clip, avoiding false negatives."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    groups: dict[str, list[int]] = {}
    for index, entry in enumerate(entries):
        # AudioCaps audiocap_id identifies a caption row, so audio_path is the
        # reliable positive-group identity when multiple captions share audio.
        group_id = str(entry.get("audio_path") or entry["clip_id"])
        groups.setdefault(group_id, []).append(index)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    order = list(groups)
    rng.shuffle(order)
    batches: list[list[int]] = []
    remaining = True
    round_index = 0
    while remaining:
        remaining = False
        round_items: list[int] = []
        for clip_id in order:
            values = groups[clip_id]
            if round_index < len(values):
                round_items.append(values[round_index])
                remaining = True
        rng.shuffle(round_items)
        batches.extend(round_items[start : start + batch_size] for start in range(0, len(round_items), batch_size))
        round_index += 1
    return batches


def symmetric_infonce(student_text: Any, student_audio: Any, temperature: float) -> Any:
    import torch
    import torch.nn.functional as F

    if student_text.shape != student_audio.shape or student_text.ndim != 2:
        raise ValueError("student text/audio shapes must match and be rank two")
    labels = torch.arange(student_text.shape[0], device=student_text.device)
    logits = (F.normalize(student_text, dim=-1) @ F.normalize(student_audio, dim=-1).T) / temperature
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def relational_geometry_kl(
    student_text: Any,
    student_audio: Any,
    teacher_text: Any,
    teacher_audio: Any,
    temperature: float,
) -> Any:
    import torch.nn.functional as F

    student = F.normalize(student_text, dim=-1) @ F.normalize(student_audio, dim=-1).T
    with __import__("torch").no_grad():
        teacher = F.normalize(teacher_text, dim=-1) @ F.normalize(teacher_audio, dim=-1).T
    tau = max(float(temperature), 1e-4)
    row = F.kl_div(F.log_softmax(student / tau, dim=-1), F.softmax(teacher / tau, dim=-1), reduction="batchmean") * tau * tau
    col = F.kl_div(F.log_softmax(student.T / tau, dim=-1), F.softmax(teacher.T / tau, dim=-1), reduction="batchmean") * tau * tau
    return 0.5 * (row + col)


def fit_base_pca_anchor(base_text: Any, base_audio: Any, projection_dim: int) -> tuple[Any, Any]:
    """Fit a frozen shared coordinate system from base hidden states only."""
    import torch
    import torch.nn.functional as F

    values = torch.cat((base_text.float().cpu(), base_audio.float().cpu()), dim=0)
    mean = values.mean(dim=0)
    centered = values - mean
    covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
    _, components = torch.linalg.eigh(covariance)
    components = components[:, -projection_dim:]
    # PCA signs are arbitrary; the fixed matrix is persisted for audit only.
    components = F.normalize(components, dim=0)
    return mean, components


def apply_base_pca_anchor(values: Any, mean: Any, components: Any) -> Any:
    import torch.nn.functional as F

    return F.normalize((values.float().cpu() - mean) @ components, dim=-1)


def _entry_loader(dataset: str, csv_path: Path, audio_dir: Path) -> list[dict[str, Any]]:
    from AudioRetrieval.training.oea.train_omniembed_lora import (
        _list_audiocaps_entries,
        _list_clotho_entries,
    )
    if dataset == "audiocaps":
        return _list_audiocaps_entries(csv_path, audio_dir)
    return _list_clotho_entries(csv_path, audio_dir)


def _encode_hidden(adapter: Any, model: Any, entries: Sequence[dict[str, Any]], *, device: Any, texts: bool, disable_lora: bool, batch_size: int, label: str) -> Any:
    import contextlib
    import torch
    from AudioRetrieval.training.oea.train_omniembed_lora import encode_batch

    values: list[Any] = []
    context = model.disable_adapter() if disable_lora else contextlib.nullcontext()
    inputs = [entry["caption"] for entry in entries] if texts else [Path(entry["audio_path"]) for entry in entries]
    total_batches = (len(inputs) + batch_size - 1) // batch_size
    print(f"CGP_ENCODING_START label={label} rows={len(inputs)} batches={total_batches}", flush=True)
    with context, torch.inference_mode():
        for batch_index, start in enumerate(range(0, len(inputs), batch_size), start=1):
            batch = inputs[start : start + batch_size]
            pooled, valid = encode_batch(adapter, model, adapter.processor, batch if texts else None, None if texts else batch, device)
            if pooled is None or valid != list(range(len(batch))):
                raise RuntimeError(f"encoder skipped inputs in {start}:{start + len(batch)}")
            values.append(pooled.detach().float().cpu())
            if batch_index == total_batches or batch_index % 100 == 0:
                print(
                    f"CGP_ENCODING_PROGRESS label={label} batch={batch_index}/{total_batches} rows={start + len(batch)}/{len(inputs)}",
                    flush=True,
                )
    return torch.cat(values, dim=0)


def _load_model(config: dict[str, Any], model_root: Path, report_path: Path) -> tuple[Any, Any, Any, Any, Any, Any, dict[str, Any]]:
    from scripts.generate_oea_embeddings import load_model_bundle
    import torch

    report: dict[str, Any] = {"schema_version": 1, "status": "loading", "metrics_path": str(report_path)}
    torch_module, adapter, model, audio_head, text_head, device = load_model_bundle(config, model_root, report)
    checkpoint_path = (model_root / config["checkpoint"]["local_subpath"]).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return torch_module, adapter, model, audio_head, text_head, device, checkpoint


def _evaluate_cached(
    text_hidden: Any,
    audio_hidden: Any,
    labels: Sequence[int],
    text_head: Any,
    audio_head: Any,
    temperature: float,
) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    device = next(text_head.parameters()).device
    with torch.no_grad():
        text = text_head(text_hidden.to(device=device, dtype=next(text_head.parameters()).dtype))
        audio = audio_head(audio_hidden.to(device=device, dtype=next(audio_head.parameters()).dtype))
        sim = F.normalize(text.float(), dim=-1) @ F.normalize(audio.float(), dim=-1).T
    target = torch.tensor(labels, dtype=torch.long, device=sim.device)
    if target.shape != (sim.shape[0],) or int(target.max()) >= sim.shape[1]:
        raise ValueError("validation labels do not match the similarity matrix")
    ranks = (
        (sim.argsort(dim=1, descending=True) == target[:, None])
        .nonzero(as_tuple=False)[:, 1]
        + 1
    )
    return {
        **{
            f"R@{k}": float((ranks <= k).float().mean().item())
            for k in (1, 5, 10)
        },
        "MRR": float((1.0 / ranks.float()).mean().item()),
        "loss": float(
            __import__("torch").nn.functional.cross_entropy(
                sim / temperature, target
            ).item()
        ),
        "ranks": [int(value) for value in ranks.detach().cpu().tolist()],
    }


def _write_eval_config(config: dict[str, Any], checkpoint_path: Path, model_root: Path, output: Path) -> None:
    import hashlib
    updated = json.loads(json.dumps(config))
    relative = checkpoint_path.resolve().relative_to(model_root.resolve()).as_posix()
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    updated["checkpoint"] = {"local_subpath": relative, "size_bytes": checkpoint_path.stat().st_size, "sha256": digest, "repo_id": "CGP-OEA", "revision": digest[:40], "source": "DERIVED_HEAD_ONLY"}
    output.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    if args.max_train_examples < 0 or args.epochs <= 0:
        raise ValueError("max-train-examples must be non-negative and epochs positive")
    if min(args.batch_size, args.text_encode_batch_size, args.audio_encode_batch_size) <= 0:
        raise ValueError("all batch sizes must be positive")
    if args.temperature <= 0 or args.distill_temperature <= 0 or args.geometry_lambda < 0 or args.pca_anchor_lambda < 0:
        raise ValueError("temperatures must be positive and loss weights non-negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    model_root = args.model_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_module, adapter, model, audio_head, text_head, device, checkpoint = _load_model(config, model_root, output_dir / "model_load.json")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("CGP invariant violated: backbone or LoRA remains trainable")
    initial_audio_head = copy.deepcopy(audio_head).to(device)
    initial_text_head = copy.deepcopy(text_head).to(device)

    train_entries = _entry_loader(args.dataset, args.train_csv.resolve(), args.audio_dir.resolve())
    if args.max_train_examples and len(train_entries) > args.max_train_examples:
        order = sorted(range(len(train_entries)), key=lambda i: hashlib.sha256(f"{args.seed}:{i}".encode()).hexdigest())
        train_entries = [train_entries[i] for i in order[: args.max_train_examples]]
    val_entries = _entry_loader(args.dataset, args.val_csv.resolve(), (args.val_audio_dir or args.audio_dir).resolve())
    # Encode unique audio once, while retaining one row per caption for text.
    def unique_entries(entries: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[int]]:
        unique: list[dict[str, Any]] = []
        position: dict[str, int] = {}
        mapping: list[int] = []
        for entry in entries:
            key = str(Path(entry["audio_path"]).resolve())
            if key not in position:
                position[key] = len(unique)
                unique.append(entry)
            mapping.append(position[key])
        return unique, mapping

    train_audio_entries, train_audio_index = unique_entries(train_entries)
    val_audio_entries, val_audio_index = unique_entries(val_entries)
    print(f"CGP_DATASET={args.dataset} train_pairs={len(train_entries)} train_audios={len(train_audio_entries)} val_pairs={len(val_entries)}")
    train_base_text = _encode_hidden(adapter, model, train_entries, device=device, texts=True, disable_lora=True, batch_size=args.text_encode_batch_size, label="train_base_text")
    train_lora_text = _encode_hidden(adapter, model, train_entries, device=device, texts=True, disable_lora=False, batch_size=args.text_encode_batch_size, label="train_lora_text")
    train_base_audio = _encode_hidden(adapter, model, train_audio_entries, device=device, texts=False, disable_lora=True, batch_size=args.audio_encode_batch_size, label="train_base_audio")
    train_lora_audio = _encode_hidden(adapter, model, train_audio_entries, device=device, texts=False, disable_lora=False, batch_size=args.audio_encode_batch_size, label="train_lora_audio")
    val_lora_text = _encode_hidden(adapter, model, val_entries, device=device, texts=True, disable_lora=False, batch_size=args.text_encode_batch_size, label="val_lora_text")
    val_lora_audio = _encode_hidden(adapter, model, val_audio_entries, device=device, texts=False, disable_lora=False, batch_size=args.audio_encode_batch_size, label="val_lora_audio")
    pca_mean, pca_components = fit_base_pca_anchor(
        train_base_text,
        train_base_audio,
        int(next(audio_head.parameters()).shape[0]),
    )
    train_base_text_anchor = apply_base_pca_anchor(train_base_text, pca_mean, pca_components)
    train_base_audio_anchor = apply_base_pca_anchor(train_base_audio, pca_mean, pca_components)
    print(
        f"CGP_PCA_ANCHOR dimension={pca_components.shape[1]} "
        f"lambda={args.pca_anchor_lambda}",
        flush=True,
    )

    if args.head_init_checkpoint:
        init = torch.load(args.head_init_checkpoint.resolve(), map_location="cpu", weights_only=True)
        audio_head.load_state_dict(init["audio_head"], strict=True)
        text_head.load_state_dict(init["text_head"], strict=True)
    audio_head.train(); text_head.train()
    optimizer = torch.optim.AdamW(list(audio_head.parameters()) + list(text_head.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    baseline = _evaluate_cached(
        val_lora_text,
        val_lora_audio,
        val_audio_index,
        text_head=initial_text_head,
        audio_head=initial_audio_head,
        temperature=args.temperature,
    )
    best = {"R@10": -1.0}
    best_path = output_dir / "best.pt"
    for epoch in range(args.epochs):
        batches = make_unique_batches(train_entries, args.batch_size, seed=args.seed + epoch)
        random.shuffle(batches)
        running = 0.0
        for batch_indices in batches:
            if len(batch_indices) < 2:
                continue
            idx = torch.tensor(batch_indices, device="cpu")
            audio_idx = torch.tensor([train_audio_index[i] for i in batch_indices], device="cpu")
            student_text = train_lora_text.index_select(0, idx).to(device)
            student_audio = train_lora_audio.index_select(0, audio_idx).to(device)
            teacher_text = train_base_text.index_select(0, idx).to(device)
            teacher_audio = train_base_audio.index_select(0, audio_idx).to(device)
            projected_text = text_head(student_text)
            projected_audio = audio_head(student_audio)
            loss_nce = symmetric_infonce(projected_text, projected_audio, args.temperature)
            loss_geo = relational_geometry_kl(projected_text, projected_audio, teacher_text, teacher_audio, args.distill_temperature)
            target_text = train_base_text_anchor.index_select(0, idx).to(device)
            target_audio = train_base_audio_anchor.index_select(0, audio_idx).to(device)
            loss_anchor = 1.0 - 0.5 * (
                (projected_text * target_text).sum(dim=-1).mean()
                + (projected_audio * target_audio).sum(dim=-1).mean()
            )
            loss = (
                loss_nce
                + args.geometry_lambda * loss_geo
                + args.pca_anchor_lambda * loss_anchor
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward(); optimizer.step()
            running += float(loss.detach().cpu())
        text_head.eval(); audio_head.eval()
        metrics = _evaluate_cached(
            val_lora_text,
            val_lora_audio,
            val_audio_index,
            text_head,
            audio_head,
            args.temperature,
        )
        text_head.train(); audio_head.train()
        printable_metrics = {key: value for key, value in metrics.items() if key != "ranks"}
        print(f"CGP_EPOCH={epoch + 1} loss={running / max(len(batches), 1):.6f} metrics={json.dumps(printable_metrics, sort_keys=True)}")
        if metrics["R@10"] > best["R@10"]:
            best = metrics
            state = {"text_head": {k: v.detach().cpu() for k, v in text_head.state_dict().items()}, "audio_head": {k: v.detach().cpu() for k, v in audio_head.state_dict().items()}, "lora_state_dict": checkpoint["lora_state_dict"], "config": checkpoint.get("config", {}), "cgp_config": {"geometry_lambda": args.geometry_lambda, "pca_anchor_lambda": args.pca_anchor_lambda, "distill_temperature": args.distill_temperature, "train_dataset": args.dataset, "max_train_examples": len(train_entries), "pca_anchor": "shared_base_hidden_covariance_top_components"}, "metrics": metrics, "baseline_metrics": baseline, "global_step": epoch + 1}
            temporary_checkpoint = best_path.with_suffix(".pt.tmp")
            torch.save(state, temporary_checkpoint)
            temporary_checkpoint.replace(best_path)
    eval_config = output_dir / "eval_config.json"
    _write_eval_config(config, best_path, model_root, eval_config)
    summary = {"schema_version": 1, "status": "complete", "dataset": args.dataset, "train_pairs": len(train_entries), "train_audio_count": len(train_audio_entries), "validation_pairs": len(val_entries), "baseline_metrics": baseline, "best_metrics": best, "checkpoint": str(best_path), "eval_config": str(eval_config), "trainable_parameter_count": sum(p.numel() for p in audio_head.parameters()) + sum(p.numel() for p in text_head.parameters())}
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = train(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
