# UIQ Toolkit

This folder is a shareable, standalone extraction of the UIQ-generation code from the `AudioRetrieval` repository.
It is meant for collaborators who only need to generate or extend UIQ data without navigating the full OEA training codebase.

## What is included

- `main.py`: standalone CLI
- `generators.py`: GPT, local LLaMA, and template backends
- `prompts.py`: few-shot prompt templates
- `query_types.py`: query-type enum and legacy bucket mapping
- `io_utils.py`: dataset loading, hard-negative loading, and JSONL export helpers
- `requirements.txt`: minimal dependencies for this toolkit

## Quick start

From the repository root:

```bash
pip install -r uiq_toolkit/requirements.txt
python -m uiq_toolkit --help
```

### GPT example

```bash
python -m uiq_toolkit \
  --backend gpt \
  --model gpt-4 \
  --dataset clotho \
  --captions-csv /path/to/clotho_evaluation.csv \
  --output-dir /path/to/uiq_outputs \
  --query-types question imperative paraphrase tagging
```

### Local LLaMA example

```bash
python -m uiq_toolkit \
  --backend llama \
  --model meta-llama/Llama-2-7b-chat-hf \
  --dataset audiocaps \
  --captions-csv /path/to/audiocaps.csv \
  --output-dir /path/to/uiq_outputs
```

### Cheap smoke test without an LLM

```bash
python -m uiq_toolkit \
  --backend template \
  --dataset clotho \
  --captions-csv /path/to/clotho_evaluation.csv \
  --output-dir /path/to/uiq_outputs \
  --output-format both
```

## Input assumptions

### Clotho

- Expects `file_name`
- Expects `caption_1` to `caption_5`
- `--caption-index` selects which caption column to use

### AudioCaps

- Expects `youtube_id`, `start_time`, and `caption`
- Clip IDs are created as `{youtube_id}_{start_time}`

### Negative queries

If `negative` is included in `--query-types`, pass `--hard-neg-jsonl`.
The loader supports the Stage-2 filtered hard-negative JSONL already produced by this repo (`hard_negatives` list with neighbor captions), and a few simpler direct mapping variants.

## Output files

- `uiq_<query_type>.jsonl`
  - Flat JSONL, one generated query per line
  - Keeps empty/error records, which is useful for debugging failed generations
- `uiq_grouped.jsonl`
  - Groups queries by clip ID
  - Adds the legacy `bucket` names expected by the existing evaluator in this repo
- `metadata.json`
  - Run configuration, counts, and output-file manifest

## Where collaborators should edit

- Change prompts: `uiq_toolkit/prompts.py`
- Add or rename query types: `uiq_toolkit/query_types.py`
- Change hard-negative selection logic: `uiq_toolkit/io_utils.py`
- Add a new backend or adjust generation behavior: `uiq_toolkit/generators.py`

## Notes

- `template` backend is only for pipeline debugging. It does not generate research-quality queries.
- The grouped JSONL format is designed to stay compatible with the current `AudioRetrieval.evaluation.runners.uiq` loader.
- This toolkit is intentionally decoupled from the main package so collaborators can copy this folder out on its own.
