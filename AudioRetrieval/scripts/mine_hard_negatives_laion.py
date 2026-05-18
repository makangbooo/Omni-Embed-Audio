"""
Hard Negative Mining using pre-computed LAION-CLAP embeddings.

Replaces MGA-CLAP Stage 1 with LAION-CLAP embeddings, then runs
BGE semantic filtering (Stages 3-4) following the paper's 4-stage pipeline.

Stage 1: Top-K acoustic neighbor retrieval (LAION-CLAP, pre-computed)
Stage 2: Count-based acoustic filtering (~3× final target count)
Stage 3-4: BGE semantic dissimilarity filtering (threshold < 0.7)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset key/caption utilities
# ---------------------------------------------------------------------------

def load_audiocaps_split_keys(csv_path: Path) -> set:
    """Return set of embedding keys for an AudioCaps split."""
    keys = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yt = row.get("youtube_id", "").strip()
            st = row.get("start_time", "").strip()
            if yt and st:
                keys.add(f"audiocaps/{yt}_{st}")
    return keys


def load_clotho_split_keys(csv_path: Path) -> set:
    """Return set of embedding keys for a Clotho split."""
    keys = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fn = row.get("file_name", "").strip()
            if fn:
                keys.add(f"clotho/{Path(fn).stem}_seg0")
    return keys


def load_mecat_keys(meta_dir: Path) -> set:
    """Return set of embedding keys for MeCAT."""
    return {f"mecat/{p.stem}" for p in Path(meta_dir).glob("*.json")}


def load_audiocaps_captions(csv_path: Path) -> Dict[str, List[str]]:
    """Map clip_id → captions for AudioCaps."""
    d: Dict[str, List[str]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yt = row.get("youtube_id", "").strip()
            st = row.get("start_time", "").strip()
            cap = row.get("caption", "").strip()
            if yt and st:
                cid = f"{yt}_{st}"
                d.setdefault(cid, [])
                if cap:
                    d[cid].append(cap)
    return d


def load_clotho_captions(csv_path: Path) -> Dict[str, List[str]]:
    """Map clip_id → captions for Clotho."""
    d: Dict[str, List[str]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fn = row.get("file_name", "").strip()
            if not fn:
                continue
            cid = Path(fn).stem
            caps = [row.get(f"caption_{i}", "").strip() for i in range(1, 6)]
            caps = [c for c in caps if c]
            if caps:
                d[cid] = caps
    return d


def load_mecat_captions(meta_dir: Path) -> Dict[str, List[str]]:
    """Map clip_id → captions for MeCAT."""
    d: Dict[str, List[str]] = {}
    for jp in sorted(Path(meta_dir).glob("*.json")):
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Use short descriptions as primary captions
        caps = data.get("short", [])
        caps = [c for c in caps if c and c.lower() != "none"]
        if caps:
            d[jp.stem] = caps
    return d


def emb_key_to_clip_id(key: str) -> str:
    """Convert embedding key to clip_id (strip dataset prefix and _seg0)."""
    cid = key.split("/", 1)[1] if "/" in key else key
    if cid.endswith("_seg0"):
        cid = cid[:-5]
    return cid


# ---------------------------------------------------------------------------
# Stage 1+2: Acoustic neighbor retrieval + count-based filtering
# ---------------------------------------------------------------------------

def compute_acoustic_neighbors(
    embeddings: np.ndarray,
    keys: np.ndarray,
    caption_dict: Dict[str, List[str]],
    topk: int = 20,
    keep_per_clip: int = 3,
) -> List[Dict[str, Any]]:
    """
    Stage 1: Compute Top-K acoustic neighbors using cosine similarity.
    Stage 2: Keep top `keep_per_clip` per clip (~3× final target).
    """
    n = len(embeddings)
    print(f"Computing neighbors for {n} clips (Top-{topk} → keep {keep_per_clip})")

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-8, None)

    emb_tensor = torch.from_numpy(embeddings)
    topk_actual = min(topk + 1, n)  # +1 to exclude self

    results = []
    batch_size = 256
    for start in tqdm(range(0, n, batch_size), desc="Stage 1: Acoustic retrieval"):
        end = min(start + batch_size, n)
        queries = emb_tensor[start:end]
        sims = torch.matmul(queries, emb_tensor.t())

        for i in range(start, end):
            sims[i - start, i] = -1.0  # exclude self
            scores, indices = torch.topk(sims[i - start], k=min(topk_actual, n - 1))

            clip_id = emb_key_to_clip_id(keys[i])
            neighbors = []
            for k_idx in range(min(keep_per_clip, len(indices))):
                j = indices[k_idx].item()
                nb_id = emb_key_to_clip_id(keys[j])
                neighbors.append({
                    "audio_id": nb_id,
                    "score": float(scores[k_idx].item()),
                    "captions": caption_dict.get(nb_id, []),
                })

            results.append({
                "audio_id": clip_id,
                "captions": caption_dict.get(clip_id, []),
                "neighbors": neighbors,
            })

    return results


# ---------------------------------------------------------------------------
# Stage 3+4: Semantic filtering with BGE
# ---------------------------------------------------------------------------

def semantic_filter(
    records: List[Dict[str, Any]],
    model_name: str = "BAAI/bge-large-en-v1.5",
    device: str = "cuda:4",
    threshold: float = 0.7,
    final_per_clip: int = 1,
    batch_size: int = 64,
) -> List[Dict[str, Any]]:
    """
    Stage 3: Compute semantic similarity between caption pairs.
    Stage 4: Keep pairs with sim < threshold, up to `final_per_clip`.
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading BGE model on {device}...")
    model = SentenceTransformer(model_name, device=device)

    # Pre-encode all unique captions
    all_captions = set()
    for rec in records:
        for cap in rec.get("captions", []):
            all_captions.add(cap)
        for nb in rec.get("neighbors", []):
            for cap in nb.get("captions", []):
                all_captions.add(cap)

    all_captions = sorted(all_captions)
    print(f"Encoding {len(all_captions)} unique captions...")
    cap_embeddings = model.encode(
        all_captions,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    cap_to_idx = {cap: i for i, cap in enumerate(all_captions)}

    filtered_records = []
    total_before = 0
    total_after = 0

    for rec in tqdm(records, desc="Stage 3-4: Semantic filtering"):
        anchor_caps = rec.get("captions", [])
        if not anchor_caps:
            continue

        anchor_idxs = [cap_to_idx[c] for c in anchor_caps if c in cap_to_idx]
        if not anchor_idxs:
            continue
        anchor_embs = cap_embeddings[anchor_idxs]

        hard_negatives = []
        for nb in rec.get("neighbors", []):
            total_before += 1
            nb_caps = nb.get("captions", [])
            if not nb_caps:
                continue

            nb_idxs = [cap_to_idx[c] for c in nb_caps if c in cap_to_idx]
            if not nb_idxs:
                continue
            nb_embs = cap_embeddings[nb_idxs]

            # Max pairwise similarity
            sim = float(np.max(anchor_embs @ nb_embs.T))

            if sim < threshold:
                hard_negatives.append({
                    **nb,
                    "semantic_similarity": round(sim, 4),
                })
                total_after += 1

            if len(hard_negatives) >= final_per_clip:
                break

        if hard_negatives:
            filtered_records.append({
                "audio_id": rec["audio_id"],
                "captions": anchor_caps,
                "hard_negatives": hard_negatives,
            })

    print(f"Kept {total_after}/{total_before} neighbor pairs "
          f"({100 * total_after / max(1, total_before):.1f}%)")
    print(f"Clips with hard negatives: {len(filtered_records)}/{len(records)}")

    return filtered_records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Set DATA_ROOT to the directory containing AudioCaps/, Clotho2.0/, _mecat/.
DATA_ROOT = Path(os.environ.get("AR_DATA_ROOT", "data"))

DATASET_CONFIGS = {
    "audiocaps": {
        "split_keys_fn": lambda: load_audiocaps_split_keys(
            DATA_ROOT / "AudioCaps/v2_meta_data/test.csv"
        ),
        "captions_fn": lambda: load_audiocaps_captions(
            DATA_ROOT / "AudioCaps/v2_meta_data/test.csv"
        ),
    },
    "clotho": {
        "split_keys_fn": lambda: load_clotho_split_keys(
            DATA_ROOT / "Clotho2.0/evaluation_meta/clotho_captions_evaluation.csv"
        ),
        "captions_fn": lambda: load_clotho_captions(
            DATA_ROOT / "Clotho2.0/evaluation_meta/clotho_captions_evaluation.csv"
        ),
    },
    "mecat": {
        "split_keys_fn": lambda: load_mecat_keys(
            DATA_ROOT / "_mecat/_mecat_audio_meta"
        ),
        "captions_fn": lambda: load_mecat_captions(
            DATA_ROOT / "_mecat/_mecat_audio_meta"
        ),
    },
}

EMB_PATH = Path(os.environ.get(
    "AR_LAION_EMB_PATH",
    "embeddings_cache/laion_embeddings/audiodb_embeddings.npz",
))


def main():
    parser = argparse.ArgumentParser(description="Hard negative mining with LAION-CLAP")
    parser.add_argument("--datasets", nargs="+", default=["audiocaps", "clotho", "mecat"],
                        choices=["audiocaps", "clotho", "mecat"])
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/hard_negatives"))
    parser.add_argument("--topk", type=int, default=20, help="Stage 1: Top-K neighbors")
    parser.add_argument("--keep-per-clip", type=int, default=3,
                        help="Stage 2: keep ~3× final target per clip")
    parser.add_argument("--semantic-threshold", type=float, default=0.7,
                        help="Stage 3-4: max semantic similarity to keep")
    parser.add_argument("--final-per-clip", type=int, default=1,
                        help="Stage 4: final hard negatives per clip")
    parser.add_argument("--bge-device", default="cuda:4",
                        help="Device for BGE model")
    args = parser.parse_args()

    # Load embeddings once
    print(f"Loading LAION-CLAP embeddings from {EMB_PATH}...")
    data = np.load(str(EMB_PATH), allow_pickle=True)
    all_keys = data["keys"]
    all_embeddings = data["embeddings"]
    print(f"Loaded {len(all_keys)} embeddings ({all_embeddings.shape})")

    for dataset in args.datasets:
        print(f"\n{'=' * 80}")
        print(f"Processing: {dataset.upper()}")
        print(f"{'=' * 80}")

        cfg = DATASET_CONFIGS[dataset]
        split_keys = cfg["split_keys_fn"]()
        caption_dict = cfg["captions_fn"]()

        # Filter embeddings for this dataset/split
        mask = np.array([k in split_keys for k in all_keys])
        ds_keys = all_keys[mask]
        ds_embeddings = all_embeddings[mask]
        print(f"Found {len(ds_keys)} embeddings for {dataset} eval split")

        if len(ds_keys) == 0:
            print(f"[WARN] No embeddings found for {dataset}, skipping")
            continue

        # Stage 1+2: Acoustic neighbors
        stage1_records = compute_acoustic_neighbors(
            ds_embeddings, ds_keys, caption_dict,
            topk=args.topk,
            keep_per_clip=args.keep_per_clip,
        )

        # Save Stage 1+2 output
        out_dir = args.output_dir / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        stage1_path = out_dir / "stage1_acoustic_neighbors.jsonl"
        with open(stage1_path, "w") as f:
            for rec in stage1_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Saved Stage 1+2 to {stage1_path}")

        # Stage 3+4: Semantic filtering
        filtered = semantic_filter(
            stage1_records,
            device=args.bge_device,
            threshold=args.semantic_threshold,
            final_per_clip=args.final_per_clip,
        )

        stage2_path = out_dir / "stage2_filtered_negatives.jsonl"
        with open(stage2_path, "w") as f:
            for rec in filtered:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Saved Stage 3+4 to {stage2_path}")

        # Summary
        print(f"\n[{dataset.upper()}] Final: {len(filtered)} clips with hard negatives")


if __name__ == "__main__":
    main()
