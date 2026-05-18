#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified audio retrieval evaluation using Hydra configuration.
Supports multiple models (laion_clap, wavcaps, cacophony, mga_clap)
and datasets (clotho, audiocaps).
"""
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import hydra
from omegaconf import DictConfig, OmegaConf

from eval_core import (
    load_eval_split,
    EvalConfig,
    evaluate,
    pretty_print,
    rsum,
    ClipItem
)
from rerankers import build_reranker


def _slugify(value: str) -> str:
    """Lightweight slug helper for filenames."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "value"


def load_audiocaps_split(audio_dir: Path, captions_csv: Path) -> List[ClipItem]:
    """
    Load AudioCaps data (supports AudioCaps v1 single-caption and v2 multi-caption formats).
    CSV format: audiocap_id, youtube_id, start_time, caption
    Audio files are named: {youtube_id}_{start_time}.wav (or .flac / with Y prefix).
    """
    if not captions_csv.exists():
        raise FileNotFoundError(f"CSV not found: {captions_csv}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio dir not found: {audio_dir}")

    from collections import defaultdict

    grouped: Dict[Tuple[str, str], Dict[str, List[str]]] = defaultdict(lambda: {"captions": [], "ids": []})
    missing_files = []

    search_dirs = [audio_dir]
    for subdir in ["audiocaps_raw_audio", "test", "eval", "audio"]:
        subdir_path = audio_dir / subdir
        if subdir_path.exists() and subdir_path.is_dir():
            search_dirs.append(subdir_path)

    def resolve_audio_path(youtube_id: str, start_time: str) -> Optional[Path]:
        possible_names = [
            f"{youtube_id}_{start_time}.wav",
            f"{youtube_id}_{start_time}.flac",
            f"Y{youtube_id}_{start_time}.wav",
            f"{youtube_id}_{int(float(start_time))}.wav",
            f"{youtube_id}.wav",
            f"Y{youtube_id}.wav",
        ]

        for search_dir in search_dirs:
            for fname in possible_names:
                test_path = search_dir / fname
                if test_path.exists():
                    return test_path
        return None

    with captions_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Verify required columns
        required = ["audiocap_id", "youtube_id", "start_time", "caption"]
        for r in required:
            if r not in reader.fieldnames:
                raise RuntimeError(f"CSV missing required column '{r}'. Header={reader.fieldnames}")

        for row in reader:
            youtube_id = row["youtube_id"].strip()
            start_time = row["start_time"].strip()
            caption = row["caption"].strip()
            audiocap_id = row["audiocap_id"].strip()

            if not caption:
                continue

            key = (youtube_id, start_time)
            grouped[key]["captions"].append(caption)
            if audiocap_id:
                grouped[key]["ids"].append(audiocap_id)

    items: List[ClipItem] = []

    for (youtube_id, start_time), info in grouped.items():
        audio_path = resolve_audio_path(youtube_id, start_time)

        if audio_path is None:
            missing_files.append(f"{youtube_id}_{start_time}")
            if len(missing_files) <= 5:
                warnings.warn(f"Audio not found for {youtube_id}_{start_time}")
            continue

        clip_prefix = info["ids"][0] if info["ids"] else f"{youtube_id}_{start_time}"
        clip_id = f"{clip_prefix}_{youtube_id}_{start_time}"

        items.append(ClipItem(
            clip_id=clip_id,
            audio_path=audio_path,
            captions=info["captions"]
        ))

    if not items:
        raise RuntimeError("No items loaded—check paths and audio file naming convention.")

    if missing_files:
        print(f"[WARNING] {len(missing_files)} audio files not found out of {len(items) + len(missing_files)} total")

    return items


def load_model(cfg: DictConfig):
    """Load the retrieval model based on configuration."""
    model_name = cfg.model.name

    if model_name == "laion_clap":
        from models.laion_clap_adapter import LaionClapAdapter
        model = LaionClapAdapter(
            ckpt_path=cfg.model.ckpt_path,
            amodel=cfg.model.amodel,
            tmodel=cfg.model.tmodel,
            resample_sr=cfg.model.resample_sr,
            audio_duration_sec=cfg.model.audio_duration_sec,
        )

    elif model_name == "wavcaps":
        from models.wavcaps_adapter import WavCapsAdapter
        model = WavCapsAdapter(
            repo_path=cfg.model.repo_path,
            ckpt_path=cfg.model.ckpt_path,
            seconds=cfg.model.seconds,
            sr=cfg.model.sr,
            device=cfg.eval.device,
        )

    elif model_name == "cacophony":
        from models.cacophony_adapter import CacophonyAdapter
        model = CacophonyAdapter(
            repo_path=cfg.model.repo_path,
            ckpt_path=cfg.model.ckpt_path,
            seconds=cfg.model.seconds
        )

    elif model_name == "mga_clap":
        from models.mga_clap_adapter import MGAClapAdapter
        model = MGAClapAdapter(
            repo_path=cfg.model.repo_path,
            ckpt_path=cfg.model.ckpt_path,
            seconds=cfg.model.seconds,
            device=cfg.eval.device
        )

    elif model_name == "omni_embed":
        from models.omni_embed_adapter import OmniEmbedAdapter
        model = OmniEmbedAdapter(
            repo_id=cfg.model.repo_id,
            device=cfg.model.get("device", cfg.eval.device),
            cache_dir=cfg.model.get("cache_dir"),
            local_path=cfg.model.get("local_path"),
            trust_remote_code=cfg.model.get("trust_remote_code", True),
            torch_dtype=cfg.model.get("torch_dtype", "auto"),
            text_max_length=cfg.model.get("text_max_length", 2048),
            device_map=cfg.model.get("device_map"),
            attn_implementation=cfg.model.get("attn_implementation"),
            passage_prefix=cfg.model.get("passage_prefix", "passage:"),
            query_prefix=cfg.model.get("query_prefix", "query:"),
            audio_max_length=cfg.model.get("audio_max_length"),
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main evaluation function."""

    print("=" * 80)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Load dataset
    audio_dir = Path(cfg.dataset.audio_dir)
    captions_csv = Path(cfg.dataset.captions_csv)

    if cfg.dataset.name == "clotho":
        items = load_eval_split(audio_dir, captions_csv)
    elif cfg.dataset.name == "audiocaps":
        items = load_audiocaps_split(audio_dir, captions_csv)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset.name}")

    total_clips = len(items)
    total_captions = sum(len(it.captions) for it in items)

    print(f"[INFO] Loaded {total_clips} clips from '{audio_dir.name}' with "
          f"{total_captions} captions.")

    sample_seed = getattr(cfg.eval, "sample_seed", 0)
    def _safe_int(value) -> Optional[int]:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    sample_start = _safe_int(getattr(cfg.eval, "sample_start", None))
    sample_end = _safe_int(getattr(cfg.eval, "sample_end", None))

    sampled_clip_indices: Optional[List[int]] = None
    evaluated_clips = total_clips
    evaluated_captions = total_captions

    requested_sample_size: Optional[int]
    if sample_start is not None or sample_end is not None:
        start = max(0, sample_start or 0)
        end = total_clips if sample_end is None else max(start, min(total_clips, sample_end))
        sampled_clip_indices = list(range(start, end))
        requested_sample_size = end - start
        evaluated_clips = len(sampled_clip_indices)
        evaluated_captions = sum(len(items[idx].captions) for idx in sampled_clip_indices)
        print(
            f"[INFO] Using deterministic slice [{start}:{end}) -> {evaluated_clips} clips "
            f"({evaluated_captions} captions)."
        )
    else:
        try:
            requested_sample_size = int(cfg.eval.sample_size) if cfg.eval.sample_size is not None else None
        except (AttributeError, TypeError, ValueError):
            requested_sample_size = None

        if requested_sample_size:
            sample_count = min(requested_sample_size, total_clips)
            if sample_count > 0 and sample_count < total_clips:
                rng = random.Random(int(sample_seed or 0))
                sampled_clip_indices = sorted(rng.sample(range(total_clips), sample_count))
                evaluated_clips = sample_count
                evaluated_captions = sum(len(items[idx].captions) for idx in sampled_clip_indices)
                print(
                    f"[INFO] Sampling {sample_count}/{total_clips} clips "
                    f"({evaluated_captions} captions) for evaluation (seed={sample_seed})."
                )
        else:
            requested_sample_size = None
            sampled_clip_indices = None

    allow_t2t_audio_rerank = bool(getattr(cfg.eval, "allow_t2t_audio_rerank", False))

    tasks_cfg = getattr(cfg.eval, "tasks", None)
    if not tasks_cfg:
        tasks = ["text_to_audio", "audio_to_text", "text_to_text"]
    else:
        tasks = [str(task) for task in tasks_cfg]
    tasks_normalized = [task.lower() for task in tasks]
    if not tasks_normalized:
        tasks_normalized = ["text_to_audio", "audio_to_text", "text_to_text"]
    tasks_normalized = list(dict.fromkeys(tasks_normalized))
    tasks_set = set(tasks_normalized)

    if tasks_normalized == ["text_to_audio"]:
        evaluated_captions = evaluated_clips

    # Load model
    model = load_model(cfg)

    # Optional reranker
    reranker_cfg = cfg.get("reranker", None)
    reranker = build_reranker(reranker_cfg)
    reranker_mode = None
    if reranker:
        if "text_to_audio" not in tasks_set and not allow_t2t_audio_rerank:
            print("[WARN] Reranker configured but 'text_to_audio' task not selected; reranker will be ignored.")
            reranker = None
        if reranker is not None:
            mode_name = "unknown"
            if reranker_cfg is not None:
                if hasattr(reranker_cfg, "mode"):
                    mode_name = reranker_cfg.mode
                elif hasattr(reranker_cfg, "get"):
                    mode_name = reranker_cfg.get("mode", "unknown")
            reranker_mode = mode_name
            print(f"[INFO] Reranker enabled: {reranker.name} (mode={mode_name}, top_k={reranker.top_k})")

    # Prepare evaluation config
    eval_cfg = EvalConfig(
        eval_mode=cfg.eval.mode,
        ks=tuple(cfg.eval.metrics_k)
    )

    # Run evaluation
    outputs = evaluate(
        items,
        model,
        eval_cfg,
        batch_size_audio=cfg.eval.batch_size_audio,
        batch_size_text=cfg.eval.batch_size_text,
        device=cfg.eval.device,
        reranker=reranker,
        query_clip_indices=sampled_clip_indices,
        tasks=tasks_normalized,
        caption_seed=sample_seed if sampled_clip_indices is not None else None,
        allow_t2t_audio_rerank=allow_t2t_audio_rerank,
    )

    # Print results
    dataset_slug = _slugify(cfg.dataset.name)
    model_slug = _slugify(cfg.model.name)

    tasks_slug = "tasks_" + "-".join(tasks_normalized)

    setup = {
        "dataset": cfg.dataset.display_name,
        "audio_dir": str(audio_dir),
        "captions_csv": str(captions_csv),
        "model": cfg.model.display_name,
        "reranker": reranker.name if reranker else "baseline",
        "total_items": total_clips,
        "num_items": evaluated_clips,
        "sample_seed": sample_seed if sampled_clip_indices is not None else None,
        "sample_start": sample_start if sampled_clip_indices is not None and sample_start is not None else None,
        "sample_end": sample_end if sampled_clip_indices is not None and sample_end is not None else None,
        "num_captions": evaluated_captions,
        "sample_size_requested": requested_sample_size,
        "tasks": tasks_normalized,
        "tasks_slug": tasks_slug,
    }
    pretty_print(outputs, eval_cfg, setup)

    # Save results if requested
    if cfg.output.save_results:
        results_dir = Path(cfg.output.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        if reranker:
            name_slug = _slugify(reranker.name)
            parts = [name_slug]
            if reranker_mode:
                mode_slug = _slugify(reranker_mode)
                if mode_slug not in parts:
                    parts.append(mode_slug)
            if getattr(reranker, "top_k", None):
                parts.append(f"top{reranker.top_k}")
            reranker_slug = "_".join(parts)
        else:
            reranker_slug = "baseline"

        experiment_parts: List[str] = [tasks_slug]
        if sample_start is not None or sample_end is not None:
            start_tag = sample_start if sample_start is not None else 0
            end_tag = sample_end if sample_end is not None else total_clips
            experiment_parts.append(f"slice_{start_tag}_{end_tag}")
        elif sampled_clip_indices is not None and requested_sample_size:
            experiment_parts.append(f"seed{sample_seed}_n{requested_sample_size}")
        else:
            experiment_parts.append("full")

        experiment_suffix = "_".join(experiment_parts)
        experiment_id = f"{dataset_slug}_{model_slug}_{reranker_slug}_{experiment_suffix}"

        experiment_dir = results_dir / dataset_slug / model_slug / reranker_slug
        experiment_dir.mkdir(parents=True, exist_ok=True)

        filename_parts: List[str] = []
        if model_slug:
            filename_parts.append(model_slug)
        if reranker_slug:
            filename_parts.append(reranker_slug)
        filename_parts.append(experiment_suffix)
        filename_base = "_".join(filename_parts)

        # Create filename from config (include reranker info to avoid overwrites)
        output_file = experiment_dir / f"{filename_base}_results.json"

        report = {
            "dataset": cfg.dataset.name,
            "model": cfg.model.name,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "setup": setup,
            "eval_mode": eval_cfg.eval_mode,
            "metrics_k": list(eval_cfg.ks),
            "num_samples": evaluated_clips,
            "num_available_clips": total_clips,
            "num_evaluated_captions": evaluated_captions,
            "sample_size_requested": requested_sample_size,
            "sample_seed": sample_seed if sampled_clip_indices is not None else None,
            "sample_start": sample_start if sampled_clip_indices is not None and sample_start is not None else None,
            "sample_end": sample_end if sampled_clip_indices is not None and sample_end is not None else None,
            "num_query_captions": evaluated_captions,
            "tasks": tasks_normalized,
            "tasks_slug": tasks_slug,
        }

        if outputs.t2a_recalls:
            report["text_to_audio"] = {
                **outputs.t2a_recalls,
                "MRR": outputs.t2a_mrr,
                "DCG": outputs.t2a_dcg,
                "Rsum": rsum(outputs.t2a_recalls, keys=[f"R@{k}" for k in eval_cfg.ks]),
            }
        if outputs.a2t_recalls:
            report["audio_to_text"] = {
                **outputs.a2t_recalls,
                "MRR": outputs.a2t_mrr,
                "DCG": outputs.a2t_dcg,
                "Rsum": rsum(outputs.a2t_recalls, keys=[f"R@{k}" for k in eval_cfg.ks]),
            }
        if outputs.t2t_recalls:
            report["text_to_text"] = {
                **outputs.t2t_recalls,
                "MRR": outputs.t2t_mrr,
                "DCG": outputs.t2t_dcg,
                "Rsum": rsum(outputs.t2t_recalls, keys=[f"R@{k}" for k in eval_cfg.ks])
            }

        diagnostics_file = None
        if outputs.rerank_details and reranker:
            diag_fieldnames = [
                "dataset",
                "model",
                "reranker_slug",
                "experiment_id",
                "sample_start",
                "sample_end",
                "sample_seed",
                "tasks",
                "tasks_slug",
                "top_k",
                "query_index",
                "query_text",
                "target_clip_id",
                "target_candidate_idx",
                "baseline_rank",
                "reranked_rank",
                "target_missing",
                "skipped",
                "original_order",
                "reranked_order",
                "initial_scores",
                "candidate_clip_ids",
                "reranker_mode",
                "diagnostic_type",
                "diagnostic_scores",
                "late_interaction_scores",
                "latency_seconds",
                "gpu_cost_seconds",
                "used_fallback",
                "fallback_reason",
                "partial_order",
                "raw_response",
                "prompt",
                "completed_with",
                "pointwise_scores",
            ]
            diag_file = experiment_dir / f"{filename_base}_diagnostics.csv"
            with diag_file.open("w", newline="", encoding="utf-8") as diag_f:
                writer = csv.DictWriter(diag_f, fieldnames=diag_fieldnames)
                writer.writeheader()
                for detail in outputs.rerank_details:
                    diag = detail.get("diagnostics", {}) or {}
                    listwise_info = diag.get("listwise_info", {}) if isinstance(diag, dict) else {}
                    final_scores = diag.get("final_scores", {}) if isinstance(diag, dict) else {}
                    row = {
                        "dataset": cfg.dataset.name,
                        "model": cfg.model.name,
                        "reranker_slug": reranker_slug,
                        "experiment_id": experiment_id,
                        "sample_start": sample_start if sample_start is not None else "",
                        "sample_end": sample_end if sample_end is not None else "",
                        "sample_seed": sample_seed if sampled_clip_indices is not None else "",
                        "tasks": " ".join(tasks_normalized),
                        "tasks_slug": tasks_slug,
                        "top_k": reranker.top_k if reranker else "",
                        "query_index": detail.get("query_index"),
                        "query_text": detail.get("query_text", ""),
                        "target_clip_id": detail.get("target_clip_id", ""),
                        "target_candidate_idx": detail.get("target_candidate_idx"),
                        "baseline_rank": detail.get("baseline_rank"),
                        "reranked_rank": detail.get("reranked_rank"),
                        "target_missing": detail.get("target_missing_in_reranked", False),
                        "skipped": detail.get("skipped", False),
                        "original_order": " ".join(str(x) for x in detail.get("original_order", [])),
                        "reranked_order": " ".join(str(x) for x in detail.get("reranked_order", [])),
                        "initial_scores": json.dumps(detail.get("initial_scores", {}), ensure_ascii=False),
                        "candidate_clip_ids": " | ".join(detail.get("candidate_clip_ids", [])),
                        "reranker_mode": detail.get("reranker_mode", ""),
                        "diagnostic_type": diag.get("type", "") if isinstance(diag, dict) else "",
                        "diagnostic_scores": json.dumps({str(k): v for k, v in diag.get("scores", {}).items()}, ensure_ascii=False) if isinstance(diag, dict) and diag.get("scores") else "",
                        "late_interaction_scores": json.dumps({str(k): v for k, v in final_scores.items()}, ensure_ascii=False) if final_scores else "",
                        "latency_seconds": detail.get("latency_seconds"),
                        "gpu_cost_seconds": detail.get("gpu_cost_seconds"),
                        "used_fallback": listwise_info.get("used_fallback", diag.get("used_fallback", False)),
                        "fallback_reason": listwise_info.get("fallback_reason", diag.get("fallback_reason")),
                        "partial_order": listwise_info.get("partial_order"),
                        "raw_response": listwise_info.get("response"),
                        "prompt": listwise_info.get("prompt"),
                        "completed_with": " ".join(str(x) for x in listwise_info.get("completed_with", [])),
                        "pointwise_scores": json.dumps({str(k): v for k, v in listwise_info.get("pointwise_scores", {}).items()}, ensure_ascii=False) if listwise_info.get("pointwise_scores") else "",
                    }
                    writer.writerow(row)
            diagnostics_file = diag_file.name
            print(f"[INFO] Wrote rerank diagnostics to {diag_file}")

        if outputs.reranker_summary:
            report["reranker_summary"] = outputs.reranker_summary
        if outputs.rerank_details and reranker:
            report["num_reranked_queries"] = len(outputs.rerank_details)
        if diagnostics_file:
            report["diagnostics_file"] = diagnostics_file

        output_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[INFO] Wrote results to {output_file}")


if __name__ == "__main__":
    main()
