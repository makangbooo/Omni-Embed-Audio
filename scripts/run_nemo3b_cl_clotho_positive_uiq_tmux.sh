#!/usr/bin/env bash
set -uo pipefail

if [[ -z "${TMUX:-}" ]]; then
  echo "[ERROR] This long task must run inside tmux." >&2
  echo "[INFO] Create/attach a named session, then run this script in its visible pane." >&2
  exit 2
fi

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <completed-nemo3b-cl-caption-embedding-directory>" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CAPTION_EMBEDDING_DIR="$(cd "$1" && pwd)"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-oea_nemo3b_clotho_positive_uiq_embeddings_seed42_${RUN_STAMP}}"
SUITE_ID="${SUITE_ID:-oea_nemo3b_clotho_positive_uiq_suite_seed42_${RUN_STAMP}}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/results/raw}"
UIQ_OUTPUT_DIR="${RESULT_ROOT}/${RUN_ID}"
SUITE_DIR="${RESULT_ROOT}/${SUITE_ID}"
LOG_DIR="${TMUX_LOG_DIR:-${LOG_ROOT:-${ROOT_DIR}/logs}/tmux_${RUN_ID}_${RUN_STAMP}}"
LOG_FILE="${LOG_DIR}/combined.log"

if [[ -e "${LOG_DIR}" ]]; then
  echo "[ERROR] Log directory exists; refusing to overwrite: ${LOG_DIR}" >&2
  exit 3
fi
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2> >(tee -a "${LOG_FILE}" >&2)

START_EPOCH="$(date +%s)"
START_TIME="$(date -Is)"
GIT_COMMIT="$(git rev-parse HEAD)"
GIT_STATUS="$(git status --short)"

format_duration() {
  local total="$1"
  printf '%02d:%02d:%02d' "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
}

echo "EXPERIMENT_NAME=OEA-Nemo3B-Cl Clotho positive UIQ Tables 12-15"
echo "GIT_COMMIT=${GIT_COMMIT}"
echo "GIT_STATUS_SHORT=${GIT_STATUS:-<clean>}"
echo "MODEL=JudeJiwoo/OEA-Nemo3B-Cl@9588912298afca0b11f5895b864ae28083f35022/step_450_best_inference_only.pt"
echo "DATASET=Clotho v2.1 evaluation; 1,045 candidates; 4,180 released positive UIQ queries"
echo "RESOURCES=1 CUDA GPU for embeddings, then CPU-only retrieval finalization"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv,noheader || true
echo "TOTAL_WORKLOAD=4,180 GPU text encodes + 4 CPU retrieval protocols"
echo "ESTIMATED_TOTAL_TIME=8-25 minutes"
echo "ETA_BASIS=Qwen3B 4,180-query lock-bound runs completed in 312-334 seconds on RTX 4090; range adds Nemo/model-load and CPU-suite margin"
echo "CACHE_DIRECTORY=${UIQ_OUTPUT_DIR}/chunks"
echo "RESULT_DIRECTORIES=${UIQ_OUTPUT_DIR};${SUITE_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "START_TIME=${START_TIME}"
echo "RESUME=Reuse the same RUN_ID; verified chunks are preserved and incomplete chunks are recomputed"

CURRENT_STAGE="uiq_embedding"
RUN_ID="${RUN_ID}" RESULT_ROOT="${RESULT_ROOT}" \
  bash scripts/run_nemo3b_cl_clotho_positive_uiq_embeddings.sh
RUN_RC=$?

if [[ "${RUN_RC}" -eq 0 ]]; then
  CURRENT_STAGE="positive_uiq_cpu_suite"
  SUITE_ID="${SUITE_ID}" RESULT_ROOT="${RESULT_ROOT}" \
    bash scripts/run_nemo3b_cl_clotho_positive_uiq_suite.sh \
      "${CAPTION_EMBEDDING_DIR}" "${UIQ_OUTPUT_DIR}"
  RUN_RC=$?
fi

END_EPOCH="$(date +%s)"
END_TIME="$(date -Is)"
ELAPSED="$((END_EPOCH - START_EPOCH))"
if [[ "${RUN_RC}" -eq 0 ]]; then
  COMPLETION_STATUS="complete"
  FAILED_STAGE="none"
  ERROR_SUMMARY="none"
  METRICS_PATH="${SUITE_DIR}/suite_metrics.json"
else
  COMPLETION_STATUS="failed"
  FAILED_STAGE="${CURRENT_STAGE}"
  ERROR_SUMMARY="See the last error lines above and ${LOG_FILE}; all completed chunks and attempt evidence were preserved."
  METRICS_PATH="${UIQ_OUTPUT_DIR}/generation_metrics.json"
fi

echo "FINAL_RUN_RC=${RUN_RC}"
echo "START_TIME=${START_TIME}"
echo "END_TIME=${END_TIME}"
echo "TOTAL_ELAPSED=$(format_duration "${ELAPSED}")"
echo "COMPLETION_STATUS=${COMPLETION_STATUS}"
echo "RESULT_DIRECTORIES=${UIQ_OUTPUT_DIR};${SUITE_DIR}"
echo "LOG_DIRECTORY=${LOG_DIR}"
echo "METRICS_PATH=${METRICS_PATH}"
echo "FAILED_STAGE=${FAILED_STAGE}"
echo "ERROR_SUMMARY=${ERROR_SUMMARY}"
echo "[INFO] The experiment is finished. This tmux pane now remains in an interactive shell."

exec bash -i
