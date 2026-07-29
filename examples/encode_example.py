"""
Omni-Embed-Audio (OEA) — minimal encoding example.

Loads a trained OEA checkpoint from the Hugging Face Hub
(https://huggingface.co/JudeJiwoo), encodes the five bundled Clotho samples
(examples/data/clotho_samples/) and a small set of natural-language queries
covering all five UIQ formulations, then prints the resulting cosine
similarity matrix and the top-1 retrieved audio per query.

Usage
-----
    pip install -r requirements.txt
    python examples/encode_example.py                          # default: OEA-Qwen3B-Cl
    python examples/encode_example.py --model OEA-Qwen7B-Cl    # try a different checkpoint
    python examples/encode_example.py --device cpu             # CPU fallback (slow)

Available checkpoints (https://huggingface.co/JudeJiwoo):
    OEA-Qwen3B-Cl  | OEA-Qwen3B-AC
    OEA-Qwen7B-Cl  | OEA-Qwen7B-AC
    OEA-Nemo3B-Cl  | OEA-Nemo3B-AC
The trailing tag denotes the dataset the LoRA+projection heads were trained on
(Cl = Clotho, AC = AudioCaps).

Audio sources
-------------
The five WAVs under examples/data/clotho_samples/ are taken from the Clotho v2
evaluation split (Drossos et al., 2020), originally sourced from Freesound.
See examples/data/clotho_samples/captions.jsonl for original filenames,
Freesound IDs, uploaders, and per-clip licenses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Base-model mapping for each released checkpoint.
BASE_MODEL = {
    "OEA-Nemo3B-Cl": "nvidia/omni-embed-nemotron-3b",
    "OEA-Nemo3B-AC": "nvidia/omni-embed-nemotron-3b",
    "OEA-Qwen3B-Cl": "Qwen/Qwen2.5-Omni-3B",
    "OEA-Qwen3B-AC": "Qwen/Qwen2.5-Omni-3B",
    "OEA-Qwen7B-Cl": "Qwen/Qwen2.5-Omni-7B",
    "OEA-Qwen7B-AC": "Qwen/Qwen2.5-Omni-7B",
}

DEFAULT_CHECKPOINT_FILE = "step_40.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model",
        default="OEA-Qwen3B-Cl",
        choices=sorted(BASE_MODEL.keys()),
        help="Released OEA checkpoint to use (default: %(default)s).",
    )
    p.add_argument("--checkpoint-file", default=DEFAULT_CHECKPOINT_FILE,
                   help="Filename inside the HF repo (default: %(default)s).")
    p.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Use an already-downloaded checkpoint instead of Hugging Face Hub.",
    )
    p.add_argument(
        "--base-model-path",
        type=Path,
        default=None,
        help="Use an already-downloaded base-model directory instead of Hugging Face Hub.",
    )
    p.add_argument("--device", default="cuda", help="cuda | cpu (default: %(default)s).")
    p.add_argument("--audio-dir", default=str(REPO_ROOT / "examples/data/clotho_samples"))
    p.add_argument("--manifest", default=str(REPO_ROOT / "examples/data/clotho_samples/captions.jsonl"))
    return p.parse_args()


def load_samples(manifest_path: Path, audio_dir: Path):
    """Return (audio_paths, primary_captions) for the bundled Clotho samples."""
    audio_paths: List[Path] = []
    captions: List[str] = []
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            audio_paths.append(audio_dir / entry["file"])
            captions.append(entry["captions"][0])
    return audio_paths, captions


# Five queries — one per UIQ formulation — referring to the same target clip
# (water_stream.wav). The model should rank that clip highly under every
# formulation if it is robust to query variation (see Section 3.2 of the paper).
DEMO_QUERIES = [
    ("question",   "Can you find a clip where water bubbles and splashes as it flows?"),
    ("imperative", "Find an audio clip of loud bubbling water flowing past."),
    ("paraphrase", "Loud burbling and splashing as a stream flows by."),
    ("tagging",    "water, bubbling, splashing, stream"),
    ("negative",   "Flowing water with bubbles, no human voices or machinery."),
]


def build_model(args: argparse.Namespace):
    """Download the checkpoint, attach LoRA + projection heads, return encoders."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required. `pip install huggingface_hub`.") from exc

    import torch

    from AudioRetrieval.models.omni_embed_adapter import OmniEmbedAdapter
    from AudioRetrieval.training.oea.train_omniembed_lora import (
        ProjectionHead,
        attach_lora,
        safe_torch_load,
    )

    base_repo = BASE_MODEL[args.model]
    print(f"[OEA] base model     : {base_repo}")
    print(f"[OEA] checkpoint repo: JudeJiwoo/{args.model}")
    print(f"[OEA] checkpoint file: {args.checkpoint_file}")

    if args.checkpoint_path is not None:
        ckpt_path = args.checkpoint_path.expanduser().resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"[OEA] checkpoint path: {ckpt_path}")
    else:
        ckpt_path = Path(
            hf_hub_download(
                repo_id=f"JudeJiwoo/{args.model}",
                filename=args.checkpoint_file,
            )
        )

    local_path = None
    if args.base_model_path is not None:
        local_path = args.base_model_path.expanduser().resolve()
        if not local_path.is_dir():
            raise NotADirectoryError(f"Base model directory not found: {local_path}")
        print(f"[OEA] base model path: {local_path}")

    adapter = OmniEmbedAdapter(
        repo_id=base_repo,
        local_path=str(local_path) if local_path is not None else None,
        device=args.device,
        passage_prefix="passage:",
        query_prefix="query:",
    )

    # LoRA configuration matches the values used during OEA training
    # (see conf/training/*.yaml and Section 3.1 of the paper).
    lora_cfg = SimpleNamespace(
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_targets=["q_proj", "k_proj", "v_proj", "o_proj", "qkv", "out_proj"],
    )
    peft_model = attach_lora(adapter.get_underlying_model(), lora_cfg)
    adapter.set_underlying_model(peft_model)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ckpt = safe_torch_load(ckpt_path, map_location=device)

    peft_model.load_state_dict(ckpt["lora_state_dict"], strict=False)

    hidden_size = peft_model.config.text_config.hidden_size
    audio_head = ProjectionHead(hidden_size, 512, 0.1).to(device).eval()
    text_head = ProjectionHead(hidden_size, 512, 0.1).to(device).eval()
    audio_head.load_state_dict(ckpt["audio_head"])
    text_head.load_state_dict(ckpt["text_head"])

    def encode_audio(paths: List[Path]) -> np.ndarray:
        raw = adapter.encode_audio([str(p) for p in paths], batch_size=4)
        with torch.inference_mode():
            return audio_head(torch.from_numpy(raw).to(device)).cpu().float().numpy()

    def encode_text(texts: List[str]) -> np.ndarray:
        raw = adapter.encode_text(texts, batch_size=16)
        with torch.inference_mode():
            return text_head(torch.from_numpy(raw).to(device)).cpu().float().numpy()

    return encode_audio, encode_text


def main() -> int:
    args = parse_args()
    audio_dir = Path(args.audio_dir)
    manifest = Path(args.manifest)
    if not manifest.exists():
        raise SystemExit(f"Sample manifest not found: {manifest}")

    audio_paths, captions = load_samples(manifest, audio_dir)
    print("\nBundled Clotho samples:")
    for p, cap in zip(audio_paths, captions):
        print(f"  - {p.name:<26s} | {cap}")

    encode_audio, encode_text = build_model(args)

    print(f"\nEncoding {len(audio_paths)} audio clip(s) and {len(DEMO_QUERIES)} demo query(ies)…")
    audio_emb = encode_audio(audio_paths)             # (N_audio, 512)
    query_emb = encode_text([q for _, q in DEMO_QUERIES])  # (N_query, 512)

    # Cosine similarity (embeddings are already L2-normalized inside ProjectionHead).
    sim = query_emb @ audio_emb.T                     # (N_query, N_audio)

    audio_names = [p.name for p in audio_paths]
    name_w = max(len(n) for n in audio_names) + 2

    print("\nCosine similarities (query → audio):")
    header = " " * 22 + "".join(n.ljust(name_w) for n in audio_names)
    print(header)
    for (qtype, q), row in zip(DEMO_QUERIES, sim):
        cells = "".join(f"{v:+.3f}".ljust(name_w) for v in row)
        print(f"  {qtype:<11s} | {cells}")

    print("\nTop-1 retrieved audio per query:")
    for (qtype, q), row in zip(DEMO_QUERIES, sim):
        top = int(np.argmax(row))
        print(f"  [{qtype:<10s}] '{q}'")
        print(f"      → {audio_names[top]}  (cos={row[top]:+.3f})")

    print("\nEmbedding shapes:", audio_emb.shape, query_emb.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
