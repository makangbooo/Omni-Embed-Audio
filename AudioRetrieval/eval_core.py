#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING
import csv, json, warnings, argparse, random
from time import perf_counter
import numpy as np
from collections import Counter, defaultdict
import sys

if TYPE_CHECKING:  # Avoid importing heavy reranker dependencies at runtime when unused.
    from rerankers.base import BaseReranker, CandidateItem

# ---------------------------
# Utils
# ---------------------------

def l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, eps, None)

def cosine_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B.T  # assumes L2-normalized

def ranks_from_scores_row(scores_row: np.ndarray, gt_index: int, ignore_index: Optional[int] = None) -> int:
    if ignore_index is not None and 0 <= ignore_index < scores_row.shape[0] and ignore_index != gt_index:
        tmp = scores_row.copy()
        tmp[ignore_index] = -np.inf
        gt_score = tmp[gt_index]
        return int(1 + np.sum(tmp > gt_score))
    gt_score = scores_row[gt_index]
    return int(1 + np.sum(scores_row > gt_score))

def compute_recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks <= k) * 100.0)

def compute_mrr(ranks: np.ndarray) -> float:
    return float(np.mean(1.0 / ranks))

def compute_dcg(ranks: np.ndarray) -> float:
    ranks = np.asarray(ranks, dtype=np.float64)
    gains = 1.0 / np.log2(ranks + 1.0)
    return float(np.mean(gains))

def rsum(recalls: Dict[str, float], keys=("R@1", "R@5", "R@10")) -> float:
    return float(sum(recalls.get(k, 0.0) for k in keys))


def _progress_bar(total: int, label: str):
    if total <= 0:
        return None
    # Enable progress bar by default, allow disabling with LALM_PROGRESS=0
    flag = os.environ.get("LALM_PROGRESS", "1").lower()
    if flag in {"0", "false", "no"}:
        return None
    debug = os.environ.get("LALM_DEBUG_PROGRESS", "").lower() in {"1", "true", "yes"}
    try:
        from tqdm import tqdm  # type: ignore
    except ImportError:
        tqdm = None

    if tqdm is None or not sys.stdout.isatty():
        class _LoggingProgress:
            def __init__(self, total: int, label: str, debug: bool) -> None:
                self.total = total
                self.label = label or "progress"
                self.count = 0
                self.debug = debug
                if self.debug:
                    print(f"[DEBUG] Created logging progress '{self.label}' total={self.total}")

            def update(self, n: int = 1) -> None:
                self.count += n
                pct = (self.count / self.total) * 100 if self.total else 100.0
                print(f"{self.label}: {self.count}/{self.total} ({pct:5.1f}%)", flush=True)

            def close(self) -> None:
                if self.debug:
                    print(f"[DEBUG] Closed logging progress '{self.label}' at {self.count}/{self.total}")

            def __bool__(self) -> bool:
                return True

        return _LoggingProgress(total, label, debug)

    # Force-enable display and use a stable ASCII bar to avoid TTY quirks.
    bar = tqdm(
        total=total,
        desc=label,
        leave=True,
        disable=False,
        dynamic_ncols=True,
        ascii=True,
        file=sys.stdout,
        mininterval=0.2,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]",
    )
    try:
        bar.refresh()
    except Exception:
        pass
    if debug:
        print(f"[DEBUG] Created progress bar '{label}' with total={total}")
    return bar

# ---------------------------
# Data loading (explicit paths)
# ---------------------------

@dataclass
class ClipItem:
    clip_id: str
    audio_path: Path
    captions: List[str]
    query_text: str = None  # Optional: specific query text for text-to-text retrieval (e.g., UIQ)

def load_eval_split(audio_dir: Path, captions_csv: Path, text_field: str = None, query_field: str = None) -> List[ClipItem]:
    """Load dataset with captions. Supports both Clotho and AudioCaps formats.

    Args:
        audio_dir: Path to audio files
        captions_csv: Path to CSV with captions
        text_field: Optional specific text field to use (e.g., 'uiq_pragmatic').
                   If specified, only this field will be used as a single caption per item.
                   If None, uses all caption_1-5 fields.
        query_field: Optional query field for text-to-text retrieval (e.g., 'uiq_pragmatic').
                    If specified, loads all caption_1-5 as targets but uses this field as query_text.
                    Mutually exclusive with text_field.
    """
    if not captions_csv.exists():
        raise FileNotFoundError(f"CSV not found: {captions_csv}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio dir not found: {audio_dir}")

    # Validate mutually exclusive parameters
    if text_field and query_field:
        raise ValueError("text_field and query_field are mutually exclusive. Use text_field for text-to-audio, query_field for text-to-caption.")

    items: List[ClipItem] = []
    with captions_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Detect format: Clotho or AudioCaps
        if "file_name" in reader.fieldnames:
            # Clotho format: file_name,caption_1,caption_2,caption_3,caption_4,caption_5
            required = ["file_name"]
            if text_field:
                # If specific text field is requested, check for it
                if text_field not in reader.fieldnames:
                    raise RuntimeError(f"CSV missing requested text field '{text_field}'. Header={reader.fieldnames}")
                required.append(text_field)
            elif query_field:
                # If query field is requested, need both query field and caption fields
                if query_field not in reader.fieldnames:
                    raise RuntimeError(f"CSV missing requested query field '{query_field}'. Header={reader.fieldnames}")
                required.append(query_field)
                required.extend(["caption_1","caption_2","caption_3","caption_4","caption_5"])
            else:
                # Otherwise require standard caption fields
                required.extend(["caption_1","caption_2","caption_3","caption_4","caption_5"])

            for r in required:
                if r not in reader.fieldnames:
                    raise RuntimeError(f"CSV missing required column '{r}'. Header={reader.fieldnames}")

            for row in reader:
                fname = row["file_name"].strip()
                a_path = (audio_dir / fname)
                if not a_path.exists():
                    warnings.warn(f"Audio missing: {a_path}; skipping")
                    continue

                # Get captions and query text based on parameters
                query_txt = None

                if text_field:
                    # Use only the specified field
                    cap_text = row[text_field].strip()
                    if cap_text:
                        caps = [cap_text]
                    else:
                        continue  # Skip items with empty custom field
                elif query_field:
                    # Use query_field for query_text, caption_1-5 for captions
                    query_txt = row[query_field].strip()
                    if not query_txt:
                        continue  # Skip items with empty query field
                    caps = [row[c].strip() for c in ["caption_1","caption_2","caption_3","caption_4","caption_5"] if row[c].strip()]
                else:
                    # Use all caption_1-5 fields
                    caps = [row[c].strip() for c in ["caption_1","caption_2","caption_3","caption_4","caption_5"] if row[c].strip()]

                if not caps: continue
                items.append(ClipItem(clip_id=Path(fname).stem, audio_path=a_path, captions=caps, query_text=query_txt))

        elif "audiocap_id" in reader.fieldnames:
            # AudioCaps format: audiocap_id,youtube_id,start_time,caption
            # Group captions by audiocap_id
            required = ["audiocap_id", "youtube_id", "start_time", "caption"]
            for r in required:
                if r not in reader.fieldnames:
                    raise RuntimeError(f"CSV missing required column '{r}'. Header={reader.fieldnames}")

            # First pass: group captions by audio file
            from collections import defaultdict
            audio_to_captions: Dict[str, List[str]] = defaultdict(list)
            audio_to_id: Dict[str, str] = {}

            for row in reader:
                audiocap_id = row["audiocap_id"].strip()
                youtube_id = row["youtube_id"].strip()
                start_time = row["start_time"].strip()
                caption = row["caption"].strip()

                # AudioCaps filename format: {youtube_id}_{start_time}.wav
                fname = f"{youtube_id}_{start_time}.wav"

                if caption:
                    audio_to_captions[fname].append(caption)
                    audio_to_id[fname] = audiocap_id

            # Second pass: create ClipItems
            for fname, caps in audio_to_captions.items():
                a_path = audio_dir / fname
                if not a_path.exists():
                    warnings.warn(f"Audio missing: {a_path}; skipping")
                    continue
                clip_id = audio_to_id.get(fname, Path(fname).stem)
                items.append(ClipItem(clip_id=clip_id, audio_path=a_path, captions=caps))

        else:
            raise RuntimeError(f"Unknown CSV format. Header={reader.fieldnames}. Expected Clotho (file_name,caption_1,...) or AudioCaps (audiocap_id,youtube_id,start_time,caption)")

    if not items:
        raise RuntimeError("No items loaded—check paths and CSV contents.")
    return items

# ---------------------------
# Model interface
# ---------------------------

class BaseRetrievalModel:
    def encode_audio(self, paths: List[str], batch_size: int = 64, device: str = "cuda") -> np.ndarray:
        raise NotImplementedError
    def encode_text(self, texts: List[str], batch_size: int = 256, device: str = "cuda") -> np.ndarray:
        raise NotImplementedError

# ---------------------------
# Evaluation (improved to handle variable caption counts)
# ---------------------------

@dataclass
class EvalConfig:
    eval_mode: str  # captionwise | caption_avg | caption_pool
    ks: Tuple[int, ...]

@dataclass
class EvalOutputs:
    t2t_recalls: Optional[Dict[str, float]] = None
    t2t_mrr: Optional[float] = None
    t2t_dcg: Optional[float] = None
    t2a_recalls: Optional[Dict[str, float]] = None
    t2a_mrr: Optional[float] = None
    t2a_dcg: Optional[float] = None
    a2t_recalls: Optional[Dict[str, float]] = None
    a2t_mrr: Optional[float] = None
    a2t_dcg: Optional[float] = None
    reranker_summary: Optional[Dict[str, Any]] = None
    rerank_details: Optional[List[Dict[str, Any]]] = None

def _flatten_captions(items: List[ClipItem]) -> Tuple[List[str], List[int]]:
    texts, owners = [], []
    for i, it in enumerate(items):
        for c in it.captions:
            texts.append(c)
            owners.append(i)
    return texts, owners

def _pack_recalls(ranks: np.ndarray, ks: Tuple[int, ...]) -> Dict[str, float]:
    return {f"R@{k}": compute_recall_at_k(ranks, k) for k in ks}

def _eval_text_to_text_captionwise(
    T: np.ndarray,
    owners: List[int],
    ks: Tuple[int, ...],
    query_indices: Optional[List[int]] = None,
):
    """Evaluate text-to-text retrieval when multiple captions exist for same audio."""
    NT = T.shape[0]
    if query_indices is None:
        indices_iter = list(range(NT))
    else:
        indices_iter = [idx for idx in query_indices if 0 <= idx < NT]
    if not indices_iter:
        return {f"R@{k}": 0.0 for k in ks}, 0.0, 0.0

    S = cosine_sim(T, T)
    from collections import defaultdict
    clip_to_text = defaultdict(list)
    for t_idx, c_idx in enumerate(owners):
        clip_to_text[c_idx].append(t_idx)
    
    ranks_list: List[int] = []
    for i in indices_iter:
        row = S[i]
        same_clip = [j for j in clip_to_text[owners[i]] if j != i]
        if same_clip:  # Only if there are sister captions
            sister_ranks = [ranks_from_scores_row(row, j, ignore_index=i) for j in same_clip]
            ranks_list.append(min(sister_ranks))
    
    if not ranks_list:
        # No valid text-to-text pairs (all single captions)
        return {f"R@{k}": 0.0 for k in ks}, 0.0, 0.0
    
    valid_ranks = np.array(ranks_list, dtype=np.int32)
    return _pack_recalls(valid_ranks, ks), compute_mrr(valid_ranks), compute_dcg(valid_ranks)

def _eval_query_to_caption(
    T_queries: np.ndarray,
    T_captions: np.ndarray,
    caption_owners: List[int],
    ks: Tuple[int, ...],
    query_clip_indices: Optional[List[int]] = None,
):
    """Evaluate query-to-caption retrieval (e.g., UIQ query -> original captions).

    Args:
        T_queries: Query embeddings (N_queries, D) - one per clip
        T_captions: Caption embeddings (N_captions, D) - multiple per clip
        caption_owners: List mapping each caption index to its clip index
        ks: Recall@k values to compute
        query_clip_indices: Optional list of clip indices to use as queries
    """
    N_queries = T_queries.shape[0]

    if query_clip_indices is None:
        query_iter = list(range(N_queries))
    else:
        query_iter = [idx for idx in query_clip_indices if 0 <= idx < N_queries]

    if not query_iter:
        return {f"R@{k}": 0.0 for k in ks}, 0.0, 0.0

    # Compute similarity matrix: queries x captions
    S = cosine_sim(T_queries, T_captions)

    # Group captions by clip
    from collections import defaultdict
    clip_to_captions = defaultdict(list)
    for cap_idx, clip_idx in enumerate(caption_owners):
        clip_to_captions[clip_idx].append(cap_idx)

    ranks_list: List[int] = []
    for query_idx in query_iter:
        row = S[query_idx]  # Similarity scores for this query
        target_captions = clip_to_captions[query_idx]  # Captions from same clip

        if target_captions:
            # Find the best rank among all target captions
            caption_ranks = [ranks_from_scores_row(row, cap_idx) for cap_idx in target_captions]
            ranks_list.append(min(caption_ranks))

    if not ranks_list:
        return {f"R@{k}": 0.0 for k in ks}, 0.0, 0.0

    valid_ranks = np.array(ranks_list, dtype=np.int32)
    return _pack_recalls(valid_ranks, ks), compute_mrr(valid_ranks), compute_dcg(valid_ranks)

def _eval_text_to_text_pool(
    T_all: List[np.ndarray],
    ks: Tuple[int, ...],
    pool: str = "avg",
    query_indices: Optional[List[int]] = None,
):
    """Pool-based text-to-text evaluation."""
    N = len(T_all)
    
    # Handle variable caption counts
    caption_counts = [t.shape[0] for t in T_all]
    
    if pool == "avg":
        bank = np.stack([t.mean(0) for t in T_all])
        Q = np.concatenate(T_all, axis=0)
        owners = np.repeat(np.arange(N, dtype=np.int32), caption_counts)
        S = cosine_sim(Q, bank)
        if query_indices is None:
            query_iter = range(S.shape[0])
        else:
            query_iter = [idx for idx in query_indices if 0 <= idx < S.shape[0]]
        ranks_list = [ranks_from_scores_row(S[i], owners[i]) for i in query_iter]
    else:  # max pooling
        # For max pooling with variable sizes, we need different handling
        Q = np.concatenate(T_all, axis=0)
        owners = np.repeat(np.arange(N, dtype=np.int32), caption_counts)
        if query_indices is None:
            query_iter = range(Q.shape[0])
        else:
            query_iter = [idx for idx in query_indices if 0 <= idx < Q.shape[0]]
        ranks_list = []
        
        for qi in query_iter:
            q = Q[qi]
            scores = np.zeros(N)
            for ci, t_clip in enumerate(T_all):
                # Max similarity to any caption in this clip
                sims = (q @ t_clip.T)
                scores[ci] = sims.max()
            ranks_list.append(ranks_from_scores_row(scores, owners[qi]))
    
    if not ranks_list:
        return {f"R@{k}": 0.0 for k in ks}, 0.0, 0.0
    
    ranks = np.array(ranks_list, dtype=np.int32)
    return _pack_recalls(ranks, ks), compute_mrr(ranks), compute_dcg(ranks)

def evaluate(items: List[ClipItem], model: BaseRetrievalModel, cfg: EvalConfig,
             batch_size_audio: int = 64, batch_size_text: int = 256, device: str = "cuda",
             reranker: Optional[BaseReranker] = None,
             query_clip_indices: Optional[List[int]] = None,
             tasks: Optional[List[str]] = None,
             caption_seed: Optional[int] = None,
             t2t_queries_per_clip: Optional[int] = 1,
             allow_t2t_audio_rerank: bool = False) -> EvalOutputs:
    """
    Evaluate retrieval performance. Handles variable caption counts per audio.
    """

    if reranker is not None:
        from rerankers.base import CandidateItem as _CandidateItem  # type: ignore
        CandidateItem = _CandidateItem  # type: ignore
    else:
        CandidateItem = None  # type: ignore
    
    if tasks is None:
        tasks = ["text_to_audio", "audio_to_text", "text_to_text"]
    tasks_normalized = [task.lower() for task in tasks]
    if cfg.eval_mode != "captionwise":
        tasks_normalized = ["text_to_audio", "audio_to_text", "text_to_text"]
    task_set = set(tasks_normalized)
    do_t2a = "text_to_audio" in task_set
    do_a2t = "audio_to_text" in task_set
    do_t2t = "text_to_text" in task_set

    if not do_t2a and not allow_t2t_audio_rerank:
        reranker = None

    # Flatten all captions
    texts_all, owners_all = _flatten_captions(items)
    query_clip_set = set(query_clip_indices) if query_clip_indices else set(range(len(items)))

    caption_rng = random.Random(caption_seed if caption_seed is not None else 0)
    clip_to_positions: Dict[int, List[int]] = defaultdict(list)
    if texts_all:
        for idx, clip_idx in enumerate(owners_all):
            clip_to_positions[clip_idx].append(idx)

    if do_t2a and not do_a2t and texts_all:
        t2a_selected_indices = []
        for clip_idx in sorted(query_clip_set):
            positions = clip_to_positions.get(clip_idx, [])
            if not positions:
                continue
            chosen = caption_rng.choice(positions)
            t2a_selected_indices.append(chosen)
        t2a_selected_indices.sort()
    elif do_t2a and texts_all:
        t2a_selected_indices = sorted(
            idx
            for clip_idx in sorted(query_clip_set)
            for idx in clip_to_positions.get(clip_idx, [])
        )
    else:
        t2a_selected_indices = []

    if do_t2t and texts_all and t2t_queries_per_clip not in (None, 0):
        t2t_selected_indices: List[int] = []
        per_clip = max(1, int(t2t_queries_per_clip))
        for clip_idx in sorted(query_clip_set):
            positions = clip_to_positions.get(clip_idx, [])
            if not positions:
                continue
            if len(positions) <= per_clip:
                chosen_positions = positions
            else:
                if per_clip == 1:
                    chosen_positions = [caption_rng.choice(positions)]
                else:
                    chosen_positions = caption_rng.sample(positions, per_clip)
            t2t_selected_indices.extend(chosen_positions)
        t2t_selected_indices.sort()
    elif do_t2t and texts_all:
        t2t_selected_indices = sorted(
            idx
            for clip_idx in sorted(query_clip_set)
            for idx in clip_to_positions.get(clip_idx, [])
        )
    else:
        t2t_selected_indices = []

    requires_text = do_t2a or do_a2t or do_t2t
    if requires_text and not texts_all:
        do_t2a = False
        do_a2t = False
        do_t2t = False
        requires_text = False

    # Check if we have query_text for text-to-text retrieval
    has_query_text = all(it.query_text is not None for it in items if it.captions)

    if requires_text and texts_all:
        texts = texts_all
        owners = owners_all
        T = model.encode_text(texts, batch_size=batch_size_text, device=device)

        # If items have query_text, encode those separately for text-to-text queries
        if has_query_text and do_t2t:
            query_texts = [it.query_text for it in items if it.query_text is not None]
            query_owners = list(range(len(items)))  # One query per clip
            T_queries = model.encode_text(query_texts, batch_size=batch_size_text, device=device)
        else:
            T_queries = None
            query_texts = None
            query_owners = None
    else:
        T = None
        T_queries = None
        query_texts = None
        query_owners = None
    
    # Group captions by clip
    N = len(items)
    T_all: List[np.ndarray] = []
    global_to_concat_idx: Dict[int, int] = {}
    concat_offset = 0
    if T is not None:
        for ci in range(N):
            clip_texts_idx = [i for i, owner in enumerate(owners) if owner == ci]
            clip_embeddings = T[clip_texts_idx, :]
            T_all.append(clip_embeddings)
            for local_pos, global_idx in enumerate(clip_texts_idx):
                global_to_concat_idx[global_idx] = concat_offset + local_pos
            concat_offset += len(clip_texts_idx)
    
    # Check if we have multiple captions per audio
    caption_counts = [t.shape[0] for t in T_all] if T_all else []
    non_empty_counts = [c for c in caption_counts if c > 0]
    multi_caption_counts = [c for c in non_empty_counts if c > 1]

    t2t_rec = None
    t2t_mrr = None
    t2t_dcg = None
    t2t_query_indices = t2t_selected_indices if do_t2t else []
    t2t_pool_query_indices = [
        global_to_concat_idx[idx] for idx in t2t_query_indices if idx in global_to_concat_idx
    ]
    if do_t2t and T is not None:
        # Check if we're using query_text for query-to-caption retrieval
        if T_queries is not None:
            # Use query-to-caption evaluation
            t2t_rec, t2t_mrr, t2t_dcg = _eval_query_to_caption(
                T_queries,
                T,
                owners,
                cfg.ks,
                query_clip_indices=list(range(len(items))),
            )
        elif cfg.eval_mode == "captionwise":
            if not multi_caption_counts:
                print("[INFO] Not enough captions per clip for text-to-text evaluation. Skipping text-to-text evaluation.")
            else:
                t2t_rec, t2t_mrr, t2t_dcg = _eval_text_to_text_captionwise(
                    T,
                    owners,
                    cfg.ks,
                    query_indices=t2t_query_indices,
                )
        elif cfg.eval_mode == "caption_avg":
            t2t_rec, t2t_mrr, t2t_dcg = _eval_text_to_text_pool(
                T_all,
                cfg.ks,
                pool="avg",
                query_indices=t2t_pool_query_indices,
            )
        else:  # caption_pool
            t2t_rec, t2t_mrr, t2t_dcg = _eval_text_to_text_pool(
                T_all,
                cfg.ks,
                pool="max",
                query_indices=t2t_pool_query_indices,
            )
    
    # Audio embeddings
    if do_t2a or do_a2t:
        audio_paths = [str(it.audio_path) for it in items]
        A = model.encode_audio(audio_paths, batch_size=batch_size_audio, device=device)
    else:
        A = None

    rerank_stats: Optional[Dict[str, Any]] = None
    rerank_details: List[Dict[str, Any]] = []
    if reranker:
        rerank_stats = {
            "name": reranker.name,
            "top_k": reranker.top_k,
            "num_calls": 0,
            "total_latency": 0.0,
            "total_gpu_cost": 0.0,
            "mode_histogram": Counter(),
            "hits_within_top_k": 0,
        }
    
    t2a_ranks: Optional[np.ndarray] = None
    t2a_ranks_baseline: List[int] = []
    t2a_ranks_reranked: List[int] = []
    a2t_ranks: Optional[np.ndarray] = None
    a2t_ranks_baseline: List[int] = []
    a2t_ranks_reranked: List[int] = []

    # Text-to-audio and audio-to-text evaluation
    t2t_ranks_audio_baseline: List[int] = []
    t2t_ranks_audio_reranked: List[int] = []

    reranker_query_mode = getattr(reranker.client, "query_source", "caption") if reranker else "caption"
    reranker_candidate_mode = getattr(reranker.client, "candidate_source", "audio") if reranker else "audio"

    if cfg.eval_mode == "captionwise":

        if do_t2a and T is not None and A is not None and t2a_selected_indices:
            owners_t2a = [owners_all[i] for i in t2a_selected_indices]
            texts_t2a = [texts_all[i] for i in t2a_selected_indices]
            T_t2a = T[t2a_selected_indices]

            debug = os.environ.get("LALM_DEBUG_PROGRESS", "").lower() in {"1", "true", "yes"}
            if debug:
                print(f"[DEBUG] text_to_audio queries={len(owners_t2a)} (captions={len(texts_t2a)})")

            S_t2a = cosine_sim(T_t2a, A)
            t2a_ranks = np.empty(len(owners_t2a), dtype=np.int32)
            progress_total = len(owners_t2a) if reranker else 0
            progress_label = f"{reranker.name} queries" if reranker else ""
            progress = _progress_bar(progress_total, progress_label)
            try:
                for qi, target_idx in enumerate(owners_t2a):
                    row = S_t2a[qi]
                    baseline_rank = ranks_from_scores_row(row, target_idx)
                    t2a_ranks_baseline.append(int(baseline_rank))

                    if reranker is None:
                        t2a_ranks[qi] = baseline_rank
                        t2a_ranks_reranked.append(int(baseline_rank))
                        if progress:
                            progress.update(1)
                        if debug:
                            print(f"[DEBUG] Progress update (no reranker) {qi+1}/{len(owners_t2a)}")
                        continue

                    sorted_indices = np.argsort(-row)
                    if reranker.top_k > 0:
                        max_candidates = min(reranker.top_k, len(sorted_indices))
                    else:
                        max_candidates = len(sorted_indices)
                    candidate_indices = sorted_indices[:max_candidates]

                    candidate_items: List[CandidateItem] = []
                    for cand_idx in candidate_indices:
                        clip = items[cand_idx]
                        caption_text = clip.captions[0] if clip.captions else None
                        candidate_items.append(
                            CandidateItem(
                                candidate_idx=cand_idx,
                                clip_id=clip.clip_id,
                                audio_path=clip.audio_path,
                                initial_score=float(row[cand_idx]),
                                caption=caption_text,
                                metadata={"captions": clip.captions},
                            )
                        )

                    result = reranker.rerank(texts_t2a[qi], candidate_items)

                    detail: Dict[str, Any] = {
                        "query_index": qi,
                        "query_text": texts_t2a[qi],
                        "target_clip_id": items[target_idx].clip_id,
                        "target_candidate_idx": target_idx,
                        "baseline_rank": int(baseline_rank),
                        "original_order": [int(cand.candidate_idx) for cand in candidate_items],
                        "initial_scores": {
                            int(cand.candidate_idx): float(row[cand.candidate_idx]) for cand in candidate_items
                        },
                        "candidate_clip_ids": [cand.clip_id for cand in candidate_items],
                        "latency_seconds": float(result.latency_seconds),
                        "gpu_cost_seconds": float(result.gpu_cost_seconds),
                    }

                    if rerank_stats is not None:
                        rerank_stats["num_calls"] += 1
                        rerank_stats["total_latency"] += result.latency_seconds
                        rerank_stats["total_gpu_cost"] += result.gpu_cost_seconds
                        mode_key = result.extra.get("mode", reranker.name)
                        rerank_stats["mode_histogram"][mode_key] += 1
                        detail["reranker_mode"] = mode_key

                    diagnostics = result.extra.get("diagnostics") if result.extra else None
                    if diagnostics is not None:
                        detail["diagnostics"] = diagnostics
                        if "original_order" in diagnostics and "original_order" not in detail:
                            detail["original_order"] = diagnostics["original_order"]

                    if not result.candidates:
                        detail["skipped"] = True
                        detail["reranked_order"] = []
                        detail["reranked_rank"] = int(baseline_rank)
                        rerank_details.append(detail)
                        t2a_ranks[qi] = baseline_rank
                        t2a_ranks_reranked.append(int(baseline_rank))
                        if progress:
                            progress.update(1)
                        continue

                    reranked_ids = [int(cand.candidate_idx) for cand in result.candidates]
                    detail["reranked_order"] = reranked_ids
                    if target_idx in reranked_ids:
                        rerank_position = reranked_ids.index(target_idx) + 1
                        detail["reranked_rank"] = int(rerank_position)
                        t2a_ranks[qi] = rerank_position
                        t2a_ranks_reranked.append(int(rerank_position))
                        if rerank_stats is not None:
                            rerank_stats["hits_within_top_k"] += 1
                    else:
                        detail["reranked_rank"] = int(baseline_rank)
                        detail["target_missing_in_reranked"] = True
                        t2a_ranks[qi] = baseline_rank
                        t2a_ranks_reranked.append(int(baseline_rank))

                    rerank_details.append(detail)

                    if progress:
                        progress.update(1)
                    if debug:
                        print(f"[DEBUG] Progress update {qi+1}/{len(owners_t2a)}")
            finally:
                if progress:
                    progress.close()
                if debug:
                    print("[DEBUG] Progress bar closed")

        if (
            do_t2t
            and T is not None
            and reranker is not None
            and allow_t2t_audio_rerank
            and t2t_selected_indices
        ):
            S_t2t = cosine_sim(T, T)
            progress = _progress_bar(len(t2t_selected_indices),
                                     f"{reranker.name} (t2t audio rerank)")
            try:
                for q_pos, caption_idx in enumerate(t2t_selected_indices):
                    query_text = texts_all[caption_idx]
                    target_clip_idx = owners_all[caption_idx]
                    row = S_t2t[caption_idx]

                    # STEP 1: Compute baseline rank against ALL clips (not just top_k)
                    # For each clip, get the best caption similarity
                    all_clip_scores = np.full(N, -np.inf, dtype=np.float32)
                    for cand_idx in range(len(row)):
                        if cand_idx == caption_idx:
                            continue
                        clip_idx = owners_all[cand_idx]
                        score_val = float(row[cand_idx])
                        all_clip_scores[clip_idx] = max(all_clip_scores[clip_idx], score_val)

                    # Compute baseline rank against ALL clips
                    baseline_rank = int(1 + np.sum(all_clip_scores > all_clip_scores[target_clip_idx]))

                    # STEP 2: Build candidate_map with top_k clips for reranking
                    sorted_indices = np.argsort(-row)
                    candidate_map: Dict[int, CandidateItem] = {}
                    seen_clip_scores: Dict[int, float] = {}
                    for cand_idx in sorted_indices:
                        if cand_idx == caption_idx:
                            continue
                        clip_idx = owners_all[cand_idx]
                        if clip_idx not in candidate_map:
                            clip = items[clip_idx]
                            score_val = float(row[cand_idx])
                            seen_clip_scores[clip_idx] = score_val
                            candidate_map[clip_idx] = CandidateItem(
                                candidate_idx=clip_idx,
                                clip_id=clip.clip_id,
                                audio_path=clip.audio_path,
                                initial_score=score_val,
                                caption=clip.captions[0] if clip.captions else None,
                                metadata={
                                    "candidate_caption_index": int(cand_idx),
                                    "candidate_caption": texts_all[cand_idx],
                                },
                            )
                        if reranker.top_k > 0 and len(candidate_map) >= reranker.top_k:
                            break

                    # Ensure target is in candidate_map for reranking
                    if target_clip_idx not in candidate_map:
                        clip = items[target_clip_idx]
                        relevant_indices = [idx for idx, owner_idx in enumerate(owners_all) if owner_idx == target_clip_idx]
                        target_scores = [float(row[idx]) for idx in relevant_indices]
                        score_val = max(target_scores) if target_scores else float(row[target_clip_idx])
                        candidate_map[target_clip_idx] = CandidateItem(
                            candidate_idx=target_clip_idx,
                            clip_id=clip.clip_id,
                            audio_path=clip.audio_path,
                            initial_score=score_val,
                            caption=clip.captions[0] if clip.captions else None,
                            metadata={
                                "candidate_caption_index": relevant_indices[0] if relevant_indices else int(target_clip_idx),
                                "candidate_caption": clip.captions[0] if clip.captions else "",
                            },
                        )
                        seen_clip_scores[target_clip_idx] = score_val

                    candidate_items = list(candidate_map.values())
                    baseline_order = sorted(
                        candidate_items,
                        key=lambda item: seen_clip_scores.get(item.candidate_idx, float(item.initial_score)),
                        reverse=True,
                    )
                    baseline_clip_order = [item.candidate_idx for item in baseline_order]

                    detail: Dict[str, Any] = {
                        "query_index": q_pos,
                        "query_caption_index": int(caption_idx),
                        "query_text": query_text,
                        "target_clip_id": items[target_clip_idx].clip_id,
                        "target_candidate_idx": target_clip_idx,
                        "baseline_rank": int(baseline_rank),
                        "original_order": baseline_clip_order,
                        "initial_scores": {
                            int(item.candidate_idx): float(seen_clip_scores.get(item.candidate_idx, item.initial_score))
                            for item in baseline_order
                        },
                        "candidate_clip_ids": [item.clip_id for item in baseline_order],
                        "reranker_mode": reranker.name,
                        "t2t_audio_rerank": True,
                    }

                    result = reranker.rerank(query_text, candidate_items)

                    if rerank_stats is not None:
                        rerank_stats["num_calls"] += 1
                        rerank_stats["total_latency"] += result.latency_seconds
                        rerank_stats["total_gpu_cost"] += result.gpu_cost_seconds
                        mode_key = result.extra.get("mode", reranker.name)
                        rerank_stats["mode_histogram"][mode_key] += 1
                        detail["reranker_mode"] = mode_key

                    diagnostics = result.extra.get("diagnostics") if result.extra else None
                    if diagnostics is not None:
                        detail["diagnostics"] = diagnostics
                        if "original_order" in diagnostics:
                            detail.setdefault("original_order", diagnostics["original_order"])

                    if not result.candidates:
                        detail["skipped"] = True
                        detail["reranked_order"] = []
                        detail["reranked_rank"] = int(baseline_rank)
                        rerank_details.append(detail)
                        t2t_ranks_audio_baseline.append(int(baseline_rank))
                        t2t_ranks_audio_reranked.append(int(baseline_rank))
                        if progress:
                            progress.update(1)
                        continue

                    reranked_ids = [int(cand.candidate_idx) for cand in result.candidates]
                    detail["reranked_order"] = reranked_ids
                    if target_clip_idx in reranked_ids:
                        reranked_rank = reranked_ids.index(target_clip_idx) + 1
                        detail["reranked_rank"] = int(reranked_rank)
                        t2t_ranks_audio_reranked.append(int(reranked_rank))
                        if rerank_stats is not None:
                            limit = reranker.top_k if reranker.top_k > 0 else len(reranked_ids)
                            if reranked_rank <= limit:
                                rerank_stats["hits_within_top_k"] += 1
                    else:
                        detail["reranked_rank"] = int(baseline_rank)
                        detail["target_missing_in_reranked"] = True
                        t2t_ranks_audio_reranked.append(int(baseline_rank))
                    t2t_ranks_audio_baseline.append(int(baseline_rank))
                    rerank_details.append(detail)
                    if progress:
                        progress.update(1)
            finally:
                if progress:
                    progress.close()

        if do_a2t and T is not None and A is not None:
            S_a2t = cosine_sim(A, T)
            clip_to_text = [[] for _ in range(len(items))]
            for t_idx, c_idx in enumerate(owners):
                clip_to_text[c_idx].append(t_idx)

            a2t_rank_list: List[int] = []
            for ai in sorted(query_clip_set):
                caption_indices = clip_to_text[ai] if ai < len(clip_to_text) else []
                if not caption_indices:
                    continue
                row = S_a2t[ai]
                ranks = [ranks_from_scores_row(row, ti) for ti in caption_indices]
                baseline_rank = min(ranks)
                a2t_rank_list.append(baseline_rank)
                a2t_ranks_baseline.append(baseline_rank)

                if reranker is None:
                    continue

                sorted_indices = np.argsort(-row)
                candidate_items: List[CandidateItem] = []
                seen_indices: set[int] = set()
                for idx in sorted_indices:
                    caption_text = texts_all[idx]
                    clip_idx = owners_all[idx]
                    candidate_items.append(
                        CandidateItem(
                            candidate_idx=idx,
                            clip_id=items[clip_idx].clip_id,
                            audio_path=items[clip_idx].audio_path,
                            initial_score=float(row[idx]),
                            caption=caption_text,
                            metadata={"candidate_caption": caption_text},
                        )
                    )
                    seen_indices.add(idx)
                    if reranker.top_k > 0 and len(candidate_items) >= reranker.top_k:
                        break

                for idx in caption_indices:
                    if idx not in seen_indices:
                        caption_text = texts_all[idx]
                        clip_idx = owners_all[idx]
                        candidate_items.append(
                            CandidateItem(
                                candidate_idx=idx,
                                clip_id=items[clip_idx].clip_id,
                                audio_path=items[clip_idx].audio_path,
                                initial_score=float(row[idx]),
                                caption=caption_text,
                                metadata={"candidate_caption": caption_text},
                            )
                        )

                query_payload = (
                    items[ai].clip_id
                    if reranker_query_mode == "audio"
                    else (items[ai].captions[0] if items[ai].captions else "")
                )

                detail: Dict[str, Any] = {
                    "query_audio_index": int(ai),
                    "query_clip_id": items[ai].clip_id,
                    "caption_indices": [int(idx) for idx in caption_indices],
                    "baseline_rank": int(baseline_rank),
                    "candidate_indices": [int(c.candidate_idx) for c in candidate_items],
                    "initial_scores": {
                        int(c.candidate_idx): float(row[c.candidate_idx]) for c in candidate_items
                    },
                    "latency_seconds": 0.0,
                    "gpu_cost_seconds": 0.0,
                }

                start_rerank = perf_counter()
                result = reranker.rerank(query_payload, candidate_items)
                latency = perf_counter() - start_rerank
                detail["latency_seconds"] = float(latency)

                if rerank_stats is not None:
                    rerank_stats["num_calls"] += 1
                    rerank_stats["total_latency"] += latency
                    rerank_stats["total_gpu_cost"] += result.gpu_cost_seconds
                    mode_key = result.extra.get("mode", reranker.name)
                    rerank_stats["mode_histogram"][mode_key] += 1
                    detail["reranker_mode"] = mode_key

                diagnostics = result.extra.get("diagnostics") if result.extra else None
                if diagnostics is not None:
                    detail["diagnostics"] = diagnostics

                reranked_ids = [int(c.candidate_idx) for c in result.candidates]
                detail["reranked_order"] = reranked_ids
                reranked_rank = baseline_rank
                if reranked_ids:
                    candidate_hits = [
                        reranked_ids.index(idx) + 1
                        for idx in caption_indices
                        if idx in reranked_ids
                    ]
                    if candidate_hits:
                        reranked_rank = min(candidate_hits)
                        if rerank_stats is not None:
                            limit = reranker.top_k if reranker.top_k > 0 else len(reranked_ids)
                            if reranked_rank <= limit:
                                rerank_stats["hits_within_top_k"] += 1
                detail["reranked_rank"] = int(reranked_rank)
                a2t_ranks_reranked.append(int(reranked_rank))
                rerank_details.append(detail)

            if a2t_rank_list:
                if reranker is None or not a2t_ranks_reranked:
                    a2t_ranks = np.array(a2t_rank_list, dtype=np.int32)
                else:
                    a2t_ranks = np.array(a2t_ranks_reranked, dtype=np.int32)
    elif cfg.eval_mode == "caption_avg":
        # Average pooling
        bank = np.stack([t.mean(0) for t in T_all])  # [N, D]
        S_t2a = cosine_sim(bank, A)
        t2a_ranks = np.fromiter((ranks_from_scores_row(S_t2a[i], i) for i in range(N)), dtype=np.int32)
        S_a2t = cosine_sim(A, bank)
        a2t_ranks = np.fromiter((ranks_from_scores_row(S_a2t[i], i) for i in range(N)), dtype=np.int32)
        
    else:  # caption_pool (max)
        # Max pooling
        t2a_ranks = []
        a2t_ranks = []
        
        for ai in range(N):
            # Text→Audio: max similarity from any caption of clip i to audio i
            t_clip = T_all[ai]  # [num_captions, D]
            sims_t2a = (t_clip @ A.T)  # [num_captions, N]
            max_sims_t2a = sims_t2a.max(0)  # [N]
            t2a_ranks.append(ranks_from_scores_row(max_sims_t2a, ai))
            
            # Audio→Text: max similarity from audio i to any caption of clip i
            a_vec = A[ai:ai+1]  # [1, D]
            sims_a2t = []
            for tj, t_clip_j in enumerate(T_all):
                max_sim = (a_vec @ t_clip_j.T).max()
                sims_a2t.append(max_sim)
            sims_a2t = np.array(sims_a2t)
            a2t_ranks.append(ranks_from_scores_row(sims_a2t, ai))
        
        t2a_ranks = np.array(t2a_ranks, dtype=np.int32)
        a2t_ranks = np.array(a2t_ranks, dtype=np.int32)
    
    # Compute metrics
    if isinstance(t2a_ranks, np.ndarray) and t2a_ranks.size > 0:
        t2a_rec = _pack_recalls(t2a_ranks, cfg.ks)
        t2a_mrr = compute_mrr(t2a_ranks)
        t2a_dcg = compute_dcg(t2a_ranks)
    else:
        t2a_rec = None
        t2a_mrr = None
        t2a_dcg = None
    
    if isinstance(a2t_ranks, np.ndarray) and a2t_ranks.size > 0:
        a2t_rec = _pack_recalls(a2t_ranks, cfg.ks)
        a2t_mrr = compute_mrr(a2t_ranks)
        a2t_dcg = compute_dcg(a2t_ranks)
    else:
        a2t_rec = None
        a2t_mrr = None
        a2t_dcg = None

    reranker_summary: Optional[Dict[str, Any]] = None
    if rerank_stats and rerank_stats["num_calls"] > 0:
        summary = dict(rerank_stats)
        summary["avg_latency"] = summary["total_latency"] / summary["num_calls"]
        summary["avg_gpu_cost"] = summary["total_gpu_cost"] / summary["num_calls"]
        hits = summary.get("hits_within_top_k", 0)
        summary["hit_rate_within_top_k"] = hits / summary["num_calls"]
        if reranker and hasattr(reranker.client, "get_listwise_stats"):
            try:
                summary["listwise_stats"] = reranker.client.get_listwise_stats()
            except Exception:
                pass
        summary["mode_histogram"] = dict(summary["mode_histogram"])
        if t2a_ranks_baseline and t2a_ranks_reranked:
            baseline_arr = np.array(t2a_ranks_baseline, dtype=np.int32)
            reranked_arr = np.array(t2a_ranks_reranked, dtype=np.int32)
            summary["text_to_audio_rerank"] = {
                "baseline": {
                    **_pack_recalls(baseline_arr, cfg.ks),
                    "MRR": compute_mrr(baseline_arr),
                    "DCG": compute_dcg(baseline_arr),
                },
                "reranked": {
                    **_pack_recalls(reranked_arr, cfg.ks),
                    "MRR": compute_mrr(reranked_arr),
                    "DCG": compute_dcg(reranked_arr),
                },
            }
        if t2t_ranks_audio_baseline and t2t_ranks_audio_reranked:
            baseline_arr = np.array(t2t_ranks_audio_baseline, dtype=np.int32)
            reranked_arr = np.array(t2t_ranks_audio_reranked, dtype=np.int32)
            summary["text_to_text_audio_rerank"] = {
                "baseline": {
                    **_pack_recalls(baseline_arr, cfg.ks),
                    "MRR": compute_mrr(baseline_arr),
                    "DCG": compute_dcg(baseline_arr),
                },
                "reranked": {
                    **_pack_recalls(reranked_arr, cfg.ks),
                    "MRR": compute_mrr(reranked_arr),
                    "DCG": compute_dcg(reranked_arr),
                },
            }
        if a2t_ranks_baseline and a2t_ranks_reranked:
            baseline_arr = np.array(a2t_ranks_baseline, dtype=np.int32)
            reranked_arr = np.array(a2t_ranks_reranked, dtype=np.int32)
            summary["audio_to_text_rerank"] = {
                "baseline": {
                    **_pack_recalls(baseline_arr, cfg.ks),
                    "MRR": compute_mrr(baseline_arr),
                    "DCG": compute_dcg(baseline_arr),
                },
                "reranked": {
                    **_pack_recalls(reranked_arr, cfg.ks),
                    "MRR": compute_mrr(reranked_arr),
                    "DCG": compute_dcg(reranked_arr),
                },
            }
        reranker_summary = summary
    
    return EvalOutputs(
        t2t_recalls=t2t_rec,
        t2t_mrr=t2t_mrr,
        t2t_dcg=t2t_dcg,
        t2a_recalls=t2a_rec,
        t2a_mrr=t2a_mrr,
        t2a_dcg=t2a_dcg,
        a2t_recalls=a2t_rec,
        a2t_mrr=a2t_mrr,
        a2t_dcg=a2t_dcg,
        reranker_summary=reranker_summary,
        rerank_details=rerank_details if rerank_details else None,
    )

def pretty_print(outputs: EvalOutputs, cfg: EvalConfig, setup: Dict[str, str]) -> None:
    def fmt(rec: Dict[str, float]) -> str:
        return " / ".join(f"R@{k}: {rec.get(f'R@{k}', 0.0):.1f}" for k in cfg.ks)

    dataset_name = setup.get("dataset", "Clotho")
    print(f"\n=== {dataset_name} Evaluation ===")
    print(f"Audio dir: {setup.get('audio_dir')}\nCSV: {setup.get('captions_csv')}\nModel: {setup.get('model')}")
    print(f"Eval mode: {cfg.eval_mode}")
    tasks = setup.get("tasks")
    if tasks:
        print(f"Tasks: {', '.join(str(t) for t in tasks)}")
    if "reranker" in setup:
        print(f"Reranker: {setup['reranker']}")

    total_items = setup.get("total_items")
    evaluated_items = setup.get("num_items")
    if evaluated_items is not None:
        try:
            evaluated_val = int(evaluated_items)
        except (TypeError, ValueError):
            evaluated_val = None
        else:
            if total_items is not None:
                try:
                    total_val = int(total_items)
                except (TypeError, ValueError):
                    total_val = None
                if total_val is not None and total_val != evaluated_val:
                    seed = setup.get("sample_seed")
                    seed_str = f", seed={seed}" if seed is not None else ""
                    print(f"Samples: {evaluated_val} (from {total_val}{seed_str})")
                elif total_val is not None:
                    print(f"Samples: {evaluated_val}")
            else:
                print(f"Samples: {evaluated_val}")
    num_captions = setup.get("num_captions")
    if num_captions is not None:
        try:
            captions_val = int(num_captions)
        except (TypeError, ValueError):
            pass
        else:
            print(f"Query captions: {captions_val}")

    if outputs.t2t_recalls:
        print("\nText → Text")
        print(fmt(outputs.t2t_recalls))
        if outputs.t2t_mrr is not None and outputs.t2t_dcg is not None:
            print(f"MRR: {outputs.t2t_mrr:.4f} | DCG: {outputs.t2t_dcg:.4f}")

    if outputs.t2a_recalls:
        print("\nText → Audio")
        print(fmt(outputs.t2a_recalls))
        if outputs.t2a_mrr is not None and outputs.t2a_dcg is not None:
            print(f"MRR: {outputs.t2a_mrr:.4f} | DCG: {outputs.t2a_dcg:.4f}")
    if outputs.reranker_summary:
        rs = outputs.reranker_summary
        avg_lat_ms = rs["avg_latency"] * 1000.0
        hit_pct = rs["hit_rate_within_top_k"] * 100.0
        mode_hist = ", ".join(f"{k}: {v}" for k, v in rs["mode_histogram"].items())
        print(f"Reranker → {rs['name']} (top_k={rs['top_k']})")
        print(f"  calls: {rs['num_calls']} | hit@top_k: {hit_pct:.1f}% | avg latency: {avg_lat_ms:.1f} ms")
        if mode_hist:
            print(f"  modes: {mode_hist}")

    if outputs.a2t_recalls:
        print("\nAudio → Text")
        print(fmt(outputs.a2t_recalls))
        if outputs.a2t_mrr is not None and outputs.a2t_dcg is not None:
            print(f"MRR: {outputs.a2t_mrr:.4f} | DCG: {outputs.a2t_dcg:.4f}\n")
