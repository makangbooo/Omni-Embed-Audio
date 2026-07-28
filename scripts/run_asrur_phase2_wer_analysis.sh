#!/usr/bin/env bash
set -euo pipefail

REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-/home/jg525/Omni-Embed-Audio}"
DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
CACHE_ROOT="${CACHE_ROOT:-/home/jg525/experiment_cache/asr_uncertainty/fiqa_phase2_dc995b5b14ba}"
PHASE2_RESULT_ROOT="${PHASE2_RESULT_ROOT:-${REMOTE_PROJECT_DIR}/results/raw/asrur_phase2_fiqa_dc995b5b14ba}"
PYTHON_BIN="${PYTHON_BIN:-/home/jg525/miniconda3/envs/oea-repro/bin/python}"

cd "${REMOTE_PROJECT_DIR}"

RUN_ID="${RUN_ID:-asrur_phase2_wer_analysis_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${REMOTE_PROJECT_DIR}/logs/${RUN_ID}"
OUTPUT_DIR="${REMOTE_PROJECT_DIR}/results/raw/${RUN_ID}"
mkdir -p "${RUN_DIR}"

START_EPOCH="$(date +%s)"
echo "[INFO] CPU-only WER stratification"
echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] Output directory: ${OUTPUT_DIR}"
echo "[INFO] Cache root: ${CACHE_ROOT}"
echo "[INFO] Phase-2 result root: ${PHASE2_RESULT_ROOT}"
echo "[INFO] GPU disabled"
echo "[INFO] Training disabled"
date -Is

set +e
CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" \
  scripts/analyze_asrur_phase2_wer.py \
  --cache-root "${CACHE_ROOT}" \
  --result-root "${PHASE2_RESULT_ROOT}" \
  --queries "${DATA_ROOT}/fiqa_mteb/queries.jsonl" \
  --output-dir "${OUTPUT_DIR}" \
  > >(tee "${RUN_DIR}/stdout.log") \
  2> >(tee "${RUN_DIR}/stderr.log" >&2)
RUN_EXIT=$?
set -e

END_EPOCH="$(date +%s)"
ELAPSED_SECONDS="$((END_EPOCH - START_EPOCH))"
printf '%s\n' "${RUN_EXIT}" > "${RUN_DIR}/wrapper_exit_code.txt"
printf '%s\n' "${ELAPSED_SECONDS}" > "${RUN_DIR}/elapsed_seconds.txt"
printf '%s\n' "$(git rev-parse HEAD)" > "${RUN_DIR}/execution_git_commit.txt"

echo "[INFO] wrapper_exit_code=${RUN_EXIT}"
echo "[INFO] elapsed_seconds=${ELAPSED_SECONDS}"
echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] Output directory: ${OUTPUT_DIR}"
date -Is
exit "${RUN_EXIT}"
