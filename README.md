# Omni-Embed-Audio (OEA) — Leveraging Multimodal LLMs for Robust Audio-Text Retrieval

Official code, UIQ benchmark, and pretrained checkpoints for the **ACL 2026 (Oral)** paper
**Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval**
by HaeJun Yoo, Yongseop Shin, Insung Lee, Myoung-Wan Koo, and Du-Seong Chang (Sogang University).

- 🤗 Pretrained checkpoints: <https://huggingface.co/JudeJiwoo>
- 🌐 Interactive web demo: <https://omni-embed-audio.github.io>
- 📦 UIQ benchmark (this repo): [`data/UIQ/`](data/UIQ)
- 💻 Minimal usage example: [`examples/encode_example.py`](examples/encode_example.py) · [`examples/encode_example.ipynb`](examples/encode_example.ipynb)

---

## TL;DR

OEA is a retrieval-oriented encoder built on a multimodal LLM backbone (Qwen2.5-Omni-3B/7B, Nemotron-3B) with a *single shared transformer* for text and audio, plus LoRA adapters and 512-d projection heads. We additionally release **User-Intent Queries (UIQ)** — five query formulations (question, imperative, tagging, paraphrase, negative) over AudioCaps / Clotho / MECAT — to evaluate retrieval robustness beyond caption-style queries, and a hard-negative discrimination metric (HNSR / TFR).

Key results from the paper:

- Competitive text-to-audio retrieval vs. M2D-CLAP.
- Dominant **text-to-text** retrieval (+5.5%p mean R@5).
- Substantially better **hard-negative discrimination** (+4.3%p HNSR@10, +34.7% relative TFR@10).

---

## Repository layout

```
AudioRetrieval/         # Python package — see AudioRetrieval/__init__.py
  cli/                  # `python -m AudioRetrieval ...` entry point
  conf/                 # Hydra configs (model / dataset / training / reranker / uiq)
  evaluation/           # Retrieval evaluation with R@k, MAP, HNSR, TFR, Δ-Rank
  embeddings/           # Embedding cache utilities
  models/               # Model adapters: OEA, LAION-CLAP, MGA-CLAP, M2D-CLAP, WavCaps, …
  preprocessing/        # Embedding precomputation + hard-negative mining
  rerankers/            # Second-stage rerankers (XACLE, late-interaction, …)
  training/oea/         # OEA LoRA + projection-head training
  uiq_generation/       # UIQ generation (GPT-backed)
  uiq_toolkit/          # Lightweight standalone UIQ generation toolkit
data/UIQ/               # Released UIQ benchmark (5 query types × 3 datasets)
examples/               # Minimal encoding example (.py + .ipynb) + 5 Clotho clips
requirements.txt
```

## Installation

```bash
git clone https://github.com/JudeJiwoo/AudioRetrieval.git
cd AudioRetrieval
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA 12.4: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124`.

## Quick start: encode audio & text with a pretrained OEA model

The bundled example downloads a checkpoint from Hugging Face, attaches LoRA + projection heads, and encodes the five Clotho clips under `examples/data/clotho_samples/` along with five UIQ-style queries.

```bash
python examples/encode_example.py                          # default: OEA-Qwen3B-Cl
python examples/encode_example.py --model OEA-Qwen7B-Cl    # 7B variant
python examples/encode_example.py --device cpu             # CPU fallback (slow)
```

Or open the notebook: [`examples/encode_example.ipynb`](examples/encode_example.ipynb).

### Released checkpoints (<https://huggingface.co/JudeJiwoo>)

| Checkpoint           | Base model                   | Trained on |
|----------------------|------------------------------|------------|
| `OEA-Qwen3B-Cl`      | `Qwen/Qwen2.5-Omni-3B`       | Clotho     |
| `OEA-Qwen3B-AC`      | `Qwen/Qwen2.5-Omni-3B`       | AudioCaps  |
| `OEA-Qwen7B-Cl`      | `Qwen/Qwen2.5-Omni-7B`       | Clotho     |
| `OEA-Qwen7B-AC`      | `Qwen/Qwen2.5-Omni-7B`       | AudioCaps  |
| `OEA-Nemo3B-Cl`      | `nvidia/omni-embed-nemotron-3b` | Clotho  |
| `OEA-Nemo3B-AC`      | `nvidia/omni-embed-nemotron-3b` | AudioCaps |

Each repo contains `step_40.pt`, a dict with `lora_state_dict`, `audio_head`, `text_head` (and training metadata). The base model is downloaded separately on first use.

## UIQ benchmark

The release contains 13,053 queries — five formulations per audio clip across three datasets:

| Dataset             | question | imperative | paraphrase | tagging | negative |
|---------------------|---------:|-----------:|-----------:|--------:|---------:|
| AudioCaps (test)    | 975      | 975        | 975        | 975     | 630      |
| Clotho (evaluation) | 1 045    | 1 045      | 1 045      | 1 045   | 542      |
| MECAT               | 848      | 848        | 848        | 848     | 409      |

Files live under [`data/UIQ/`](data/UIQ) as JSONL with the schema:

```jsonc
{
  "audio_id":         "Santa Motor.wav",
  "dataset":          "clotho",
  "dataset_slug":     "clotho_evaluation",
  "query_type":       "question",
  "generated_query":  "Can you find audio of a machine whining and squealing while stamping?",
  "original_captions": ["...", "..."],
  "metadata":         {"split": "evaluation", "num_captions": 5},
  "source_model":     "gpt-5.1",
  "regen_model":      "gpt-5.1"
}
```

Negative queries additionally reference a paired hard-negative clip; see [`data/UIQ/README.md`](data/UIQ/README.md).

To regenerate UIQs from scratch, see [`AudioRetrieval/uiq_toolkit/`](AudioRetrieval/uiq_toolkit) or `AudioRetrieval/uiq_generation/`.

## Evaluating a model

```bash
# Baseline retrieval (text → audio, audio → text, text → text)
python -m AudioRetrieval evaluate model=omni_embed dataset=clotho

# UIQ evaluation with all five query types
python -m AudioRetrieval evaluate model=omni_embed dataset=clotho \
    eval.tasks='[text_to_audio]' uiq=clotho_all
```

Hydra configs under [`AudioRetrieval/conf/`](AudioRetrieval/conf) cover the supported models (`laion_clap`, `mga_clap`, `cacophony`, `wavcaps`, `omni_embed`, `omni_embed_7b`) and datasets (`clotho`, `audiocaps`).

## Training your own OEA model

See [`AudioRetrieval/training/oea/`](AudioRetrieval/training/oea) — `train_omniembed_lora.py` is the entry point and `run_training.sh` shows a typical multi-stage launch. Training is parameter-efficient: only LoRA adapters (~11–16M params) and projection heads are updated; backbone weights stay frozen.

## Citation

```bibtex
@inproceedings{yoo2026omniembedaudio,
  title     = {Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval},
  author    = {Yoo, HaeJun and Shin, Yongseop and Lee, Insung and Koo, Myoung-Wan and Chang, Du-Seong},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  note      = {Oral presentation},
  year      = {2026}
}
```

## Acknowledgments

This work was partly supported by the Institute of Information & Communications
Technology Planning & Evaluation (IITP) grant funded by the Korea government
(MSIT) (No. RS-2025-25441313, Professional AI Talent Development Program for
Multimodal AI Agents, Contribution: 50%). This research was also supported by
the MSIT, Korea, under the Top-Tier AI Global HRD invitation program
(RS-2025-25461932) supervised by the IITP.

## License

- **Code** (this repository) — [MIT](LICENSE).
- **UIQ queries** under [`data/UIQ/`](data/UIQ) — released under [CC BY 4.0](data/UIQ/LICENSE). The underlying audio (AudioCaps / Clotho / MECAT) is **not** redistributed here; obtain it from the original sources under their respective licenses.
- **Bundled Clotho samples** under [`examples/data/clotho_samples/`](examples/data/clotho_samples) — each clip retains its original Freesound license (CC0 1.0 or CC-BY 3.0). See [`examples/data/clotho_samples/captions.jsonl`](examples/data/clotho_samples/captions.jsonl) for per-clip attribution.

## Acknowledgements

- **Clotho** — Drossos, Lipping, Virtanen (2020). *Clotho: An audio captioning dataset.* ICASSP 2020. <https://zenodo.org/records/4783391>
- **AudioCaps** — Kim et al. (2019). *AudioCaps: Generating captions for audios in the wild.* NAACL 2019.
- **MECAT** — multilingual & multi-event captioning benchmark.
- Base model weights are © their respective authors: Qwen2.5-Omni (Alibaba) and Omni-Embed-Nemotron-3B (NVIDIA). Please respect their licenses when redistributing derivatives.
