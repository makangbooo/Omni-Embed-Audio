#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || "$1" != "--execute" ]]; then
  echo "Usage: bash scripts/run_asrur_oea_collapse_diagnostic.sh --execute" >&2
  exit 2
fi
if [[ -z "${TMUX:-}" ]]; then
  echo "[ERROR] The OEA collapse diagnostic must run inside tmux." >&2
  echo "[INFO] Start one with: tmux new -s asrur_oea_collapse_diag" >&2
  exit 3
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
MODEL_ROOT="${OEA_MODEL_ROOT:-/home/jg525/models/oea}"
PHASE2_CACHE_ROOT="${PHASE2_CACHE_ROOT:-/home/jg525/experiment_cache/asr_uncertainty/fiqa_phase2_d9baf226c075}"
RESOLVED_MODEL_CONFIG="${PHASE2_CACHE_ROOT}/resolved_configs/oea_nemo3b_cl.json"
FIQA_ROOT="${DATA_ROOT}/fiqa_mteb"
SOURCE_MANIFEST="${DATA_ROOT}/squtr/manifests/squtr_en_fiqa_nq_audio_query_manifest.jsonl"
QUERY_COUNT="${ASRUR_OEA_DIAGNOSTIC_QUERY_COUNT:-256}"
NEGATIVE_COUNT="${ASRUR_OEA_DIAGNOSTIC_NEGATIVE_COUNT:-4096}"
SAMPLE_SEED="${ASRUR_OEA_DIAGNOSTIC_SAMPLE_SEED:-20260805}"
BOOTSTRAP_ITERATIONS="${ASRUR_OEA_DIAGNOSTIC_BOOTSTRAP_ITERATIONS:-10000}"
BOOTSTRAP_SEED="${ASRUR_OEA_DIAGNOSTIC_BOOTSTRAP_SEED:-20260805}"
RUN_ID="asrur_oea_collapse_diagnostic_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"
START_EPOCH="$(date +%s)"

mkdir -p "${RUN_DIR}"
record_exit() {
  local exit_code=$?
  local finished_epoch
  finished_epoch="$(date +%s)"
  printf '%s\n' "${exit_code}" > "${RUN_DIR}/wrapper_exit_code.txt"
  printf '%s\n' "$((finished_epoch - START_EPOCH))" > "${RUN_DIR}/elapsed_seconds.txt"
}
trap record_exit EXIT
exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)

echo "===== ASRUR OEA COLLAPSE ATTRIBUTION ====="
echo "query_count=${QUERY_COUNT}"
echo "negative_document_count=${NEGATIVE_COUNT}"
echo "sample_seed=${SAMPLE_SEED}"
echo "bootstrap_iterations=${BOOTSTRAP_ITERATIONS}"
echo "estimated_total_time=20-60 minutes on one RTX 4090"
echo "run_dir=${RUN_DIR}"
echo "formal_cache_mutation=disabled"
echo "checkpoint_selection=disabled"
echo "training=disabled"
echo "network=disabled"
date -Is

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "[ERROR] Formal diagnostic requires a clean Git worktree." >&2
  git status --short --untracked-files=all >&2
  exit 4
fi
for path in \
  "${RESOLVED_MODEL_CONFIG}" \
  "${FIQA_ROOT}/corpus.jsonl" \
  "${FIQA_ROOT}/qrels/test.jsonl" \
  "${SOURCE_MANIFEST}" \
  "${PHASE2_CACHE_ROOT}/oea/clean/cache_manifest.json" \
  "${PHASE2_CACHE_ROOT}/vanilla/clean/cache_manifest.json"
do
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Missing prerequisite: ${path}" >&2
    exit 4
  fi
done
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git status --short --untracked-files=all > "${RUN_DIR}/git_status.txt"
hostname > "${RUN_DIR}/hostname.txt"

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate oea-repro

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python scripts/validate_single_bf16_gpu.py \
  --output "${RUN_DIR}/gpu_preflight.json"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
if [[ "${GPU_NAME}" != *"RTX 4090"* ]]; then
  echo "[ERROR] Approved diagnostic hardware is one RTX 4090; observed: ${GPU_NAME}" >&2
  exit 5
fi
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version \
  --format=csv > "${RUN_DIR}/gpu_info.txt"

python scripts/diagnose_asrur_oea_collapse.py \
  --resolved-model-config "${RESOLVED_MODEL_CONFIG}" \
  --model-root "${MODEL_ROOT}" \
  --fiqa-root "${FIQA_ROOT}" \
  --audio-manifest "${SOURCE_MANIFEST}" \
  --phase2-cache-root "${PHASE2_CACHE_ROOT}" \
  --query-count "${QUERY_COUNT}" \
  --negative-document-count "${NEGATIVE_COUNT}" \
  --sample-seed "${SAMPLE_SEED}" \
  --bootstrap-iterations "${BOOTSTRAP_ITERATIONS}" \
  --bootstrap-seed "${BOOTSTRAP_SEED}" \
  --output "${RUN_DIR}/collapse_diagnostic.json"

nvidia-smi > "${RUN_DIR}/gpu_final.txt"
echo "[INFO] OEA collapse attribution completed"
echo "[INFO] Run directory: ${RUN_DIR}"
