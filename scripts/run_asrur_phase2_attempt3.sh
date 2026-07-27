#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || "$1" != "--execute" ]]; then
  echo "Usage: bash scripts/run_asrur_phase2_attempt3.sh --execute" >&2
  exit 2
fi
if [[ -z "${TMUX:-}" ]]; then
  echo "[ERROR] Phase-2 attempt 3 must be run inside tmux." >&2
  echo "[INFO] Start one with: tmux new -s asrur_phase2_attempt3" >&2
  exit 3
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GIT_COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="${GIT_COMMIT:0:12}"

export DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
export MODELS_ROOT="${MODELS_ROOT:-/home/jg525/models}"
export OEA_MODEL_ROOT="${OEA_MODEL_ROOT:-${MODELS_ROOT}/oea}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export PHASE2_REUSE_CACHE_ROOT="${PHASE2_REUSE_CACHE_ROOT:-/home/jg525/experiment_cache/asr_uncertainty/fiqa_phase2_ee637721c1f7}"
export PHASE2_CACHE_ROOT="${PHASE2_CACHE_ROOT:-/home/jg525/experiment_cache/asr_uncertainty/fiqa_phase2_${COMMIT_SHORT}}"
export PHASE2_RESULT_ROOT="${PHASE2_RESULT_ROOT:-${ROOT_DIR}/results/raw/asrur_phase2_fiqa_${COMMIT_SHORT}}"
export PHASE2_REQUIRED_REUSE_COUNT=11

echo "===== ASRUR PHASE-2 ATTEMPT 3 ====="
echo "git_commit=${GIT_COMMIT}"
echo "tmux_session=${TMUX}"
echo "gpu_scope=1x RTX 4090 24GB (previously user-approved G1)"
echo "reuse_cache_root=${PHASE2_REUSE_CACHE_ROOT}"
echo "required_reuse_count=${PHASE2_REQUIRED_REUSE_COUNT}"
echo "cache_root=${PHASE2_CACHE_ROOT}"
echo "result_root=${PHASE2_RESULT_ROOT}"
echo "resume_policy=same commit and same cache root"
date -Is

exec bash scripts/run_asrur_phase2_frozen_retrieval.sh --execute
