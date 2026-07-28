#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || "$1" != "--execute" ]]; then
  echo "Usage: bash scripts/run_asrur_phase2_attempt5.sh --execute" >&2
  exit 2
fi
if [[ -z "${TMUX:-}" ]]; then
  echo "[ERROR] Phase-2 attempt 5 must run inside tmux." >&2
  echo "[INFO] Start one with: tmux new -s asrur_phase2_attempt5" >&2
  exit 3
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GIT_COMMIT="$(git rev-parse HEAD)"
COMMIT_SHORT="${GIT_COMMIT:0:12}"
ATTEMPT4_CACHE="/home/jg525/experiment_cache/asr_uncertainty/fiqa_phase2_d9baf226c075"

export DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
export MODELS_ROOT="${MODELS_ROOT:-/home/jg525/models}"
export OEA_MODEL_ROOT="${OEA_MODEL_ROOT:-${MODELS_ROOT}/oea}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export PHASE2_REUSE_CACHE_ROOT="${PHASE2_REUSE_CACHE_ROOT:-${ATTEMPT4_CACHE}}"
export PHASE2_REUSE_WHISPER=0
export PHASE2_REQUIRED_REUSE_COUNT=11
export PHASE2_CACHE_ROOT="${PHASE2_CACHE_ROOT:-/home/jg525/experiment_cache/asr_uncertainty/fiqa_phase2_${COMMIT_SHORT}}"
export PHASE2_RESULT_ROOT="${PHASE2_RESULT_ROOT:-${ROOT_DIR}/results/raw/asrur_phase2_fiqa_${COMMIT_SHORT}}"
export ASRUR_WHISPER_MAX_RECORD_ATTEMPTS=3
unset PHASE2_WHISPER_RESUME_SOURCE_ROOT
unset PHASE2_WHISPER_RESUME_SOURCE_GIT_COMMIT

echo "===== ASRUR PHASE-2 ATTEMPT 5 ====="
echo "git_commit=${GIT_COMMIT}"
echo "tmux_session=${TMUX}"
echo "gpu_scope=1x RTX 4090 24GB"
echo "complete_cache_reuse_root=${PHASE2_REUSE_CACHE_ROOT}"
echo "required_complete_reuse_count=${PHASE2_REQUIRED_REUSE_COUNT}"
echo "reuse_components=OEA,vanilla_Omni,BGE_only"
echo "reuse_whisper=${PHASE2_REUSE_WHISPER}"
echo "invalid_whisper_source=${ATTEMPT4_CACHE}/whisper"
echo "invalid_whisper_policy=preserve_as_failure_evidence_never_reuse"
echo "new_cache_root=${PHASE2_CACHE_ROOT}"
echo "new_result_root=${PHASE2_RESULT_ROOT}"
echo "whisper_record_attempt_limit=${ASRUR_WHISPER_MAX_RECORD_ATTEMPTS}"
echo "content_gate=min_unique_8,max_mode_fraction_0.95,max_WER_0.95"
echo "prohibited=old Whisper reuse, partial shard import, cache overwrite"
date -Is

exec bash scripts/run_asrur_phase2_frozen_retrieval.sh --execute
