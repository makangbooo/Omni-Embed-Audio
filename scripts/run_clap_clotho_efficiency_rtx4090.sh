#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
source "${ROOT_DIR}/scripts/lib/conda.sh"

MODEL="${1:-}"
if [[ "${MODEL}" != "laion_clap" && "${MODEL}" != "mga_clap" && "${MODEL}" != "m2d_clap" ]]; then
  echo "Usage: $0 <laion_clap|mga_clap|m2d_clap>" >&2
  exit 2
fi
MODEL_ROOT="${MODEL_ROOT:-/home/jg525/models/oea}"
DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
MANIFEST="${DATA_ROOT}/clotho_v2.1/manifests/clotho_evaluation_manifest.jsonl"
BENCHMARK_CONFIG="${ROOT_DIR}/configs/eval/clap_clotho_efficiency_rtx4090.json"
RUN_ID="${MODEL}_clotho_rtx4090_efficiency_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${ROOT_DIR}/results/raw/${RUN_ID}"
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "[ERROR] Refusing to overwrite ${OUTPUT_DIR}" >&2
  exit 3
fi
if [[ -n "$(git status --short)" ]]; then
  echo "[ERROR] Formal benchmark requires a clean Git worktree" >&2
  exit 4
fi
if [[ ! -f "${MANIFEST}" || ! -f "${BENCHMARK_CONFIG}" ]]; then
  echo "[ERROR] Missing Clotho manifest or benchmark config" >&2
  exit 5
fi

CONDA_BASE="$(resolve_conda_base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate oea-repro
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
case "${MODEL}" in
  laion_clap)
    export PYTHONPATH="${MODEL_ROOT}/python/laion-clap-1.1.6-overlay-v3:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
    ;;
  mga_clap)
    export PYTHONPATH="${MODEL_ROOT}/python/mga-clap-runtime-v1:${MODEL_ROOT}/python/laion-clap-1.1.6-overlay-v3:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
    ;;
  m2d_clap)
    export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
    ;;
esac
echo "EXPERIMENT_NAME=${MODEL} Clotho RTX4090 efficiency controlled"
echo "GIT_COMMIT=$(git rev-parse HEAD)"
echo "GPU_USED=yes; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "OEA_OFFICIAL_SOURCE_USED=yes"
python scripts/benchmark_clap_query_encoder.py --model "${MODEL}" --benchmark-config "${BENCHMARK_CONFIG}" --model-root "${MODEL_ROOT}" --manifest "${MANIFEST}" --output-dir "${OUTPUT_DIR}"
echo "METRICS_PATH=${OUTPUT_DIR}/metrics.json"
