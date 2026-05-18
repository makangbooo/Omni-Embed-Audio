# Examples

A minimal end-to-end usage example for **Omni-Embed-Audio (OEA)**:

- [`encode_example.py`](encode_example.py) — command-line script.
- [`encode_example.ipynb`](encode_example.ipynb) — Jupyter notebook with the same flow plus inline audio playback.

Both run the same pipeline: download a checkpoint from <https://huggingface.co/JudeJiwoo>, attach LoRA + projection heads to the base multimodal LLM, encode the five Clotho clips in [`data/clotho_samples/`](data/clotho_samples), and compare them against five UIQ-style queries (one per query type) via cosine similarity.

```bash
pip install -r ../requirements.txt
python encode_example.py                          # default: OEA-Qwen3B-Cl on CUDA
python encode_example.py --model OEA-Qwen7B-Cl
python encode_example.py --device cpu             # CPU fallback (slow)
```

The five `.wav` files under [`data/clotho_samples/`](data/clotho_samples) are bundled for convenience and originate from the Clotho v2 evaluation split — see [`data/clotho_samples/ATTRIBUTION.md`](data/clotho_samples/ATTRIBUTION.md) for per-clip Freesound IDs, uploaders, and licenses (all CC0 or CC-BY 3.0).
