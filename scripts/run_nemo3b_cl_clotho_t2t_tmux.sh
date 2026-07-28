#!/usr/bin/env bash
set -uo pipefail

if [[ -z "${TMUX:-}" ]]; then
  echo "[ERROR] This long task must run inside tmux." >&2
  echo "[INFO] Create/attach a named session, then run this script in its visible pane." >&2
  exit 2
fi

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <completed-nemo3b-cl-embedding-directory>" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

EMBEDDING_DIR="$(cd "$1" && pwd)"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
SUITE_ID="${SUITE_ID:-oea_nemo3b_clotho_t2t_suite_seed42_${RUN_STAMP}}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/results/raw}"
SUITE_DIR="${RESULT_ROOT}/${SUITE_ID}"
LOG_DIR="${TMUX_LOG_DIR:-${LOG_ROOT:-${ROOT_DIR}/logs}/tmux_${SUITE_ID}_${RUN_STAMP}}"
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
  printf '%02d:%02d:%02d' \
    "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
}

echo "EXPERIMENT_NAME=OEA-Nemo3B-Cl Clotho Table 3 T2T"
echo "GIT_COMMIT=${GIT_COMMIT}"
echo "GIT_STATUS_SHORT=${GIT_STATUS:-<clean>}"
echo "MODEL=JudeJiwoo/OEA-Nemo3B-Cl@9588912298afca0b11f5895b864ae28083f35022/step_450_best_inference_only.pt"
echo "DATASET=Clotho v2.1 evaluation; 1,045 candidates; 5,225 caption queries"
echo "RESOURCES=CPU-only; GPU disabled; cpu_count=$(nproc)"
free -h || true
echo "TOTAL_WORKLOAD=2 predeclared T2T protocols; 1,045 seed-0 queries + 5,225 all-caption sensitivity queries"
echo "ESTIMATED_TOTAL_TIME=5-15 minutes"
echo "ETA_BASIS=Existing 1,045x512 audio and 5,225x512 caption embeddings; runtime range includes hash validation, two full rankings, and artifact finalization"
echo "CACHE_DIRECTORY=${EMBEDDING_DIR} (read-only input)"
echo "RESULT_DIRECTORY=${SUITE_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "START_TIME=${START_TIME}"
echo "RESUME=Reuse the same SUITE_ID; complete protocol outputs are hash-checked and reused"

CURRENT_STAGE="t2t_cpu_suite"
SUITE_ID="${SUITE_ID}" RESULT_ROOT="${RESULT_ROOT}" \
  bash scripts/run_nemo3b_cl_clotho_t2t_suite.sh "${EMBEDDING_DIR}"
RUN_RC=$?

END_EPOCH="$(date +%s)"
END_TIME="$(date -Is)"
ELAPSED="$((END_EPOCH - START_EPOCH))"
if [[ "${RUN_RC}" -eq 0 ]]; then
  COMPLETION_STATUS="complete"
  FAILED_STAGE="none"
  ERROR_SUMMARY="none"
else
  COMPLETION_STATUS="failed"
  FAILED_STAGE="${CURRENT_STAGE}"
  ERROR_SUMMARY="See the last error lines above and ${LOG_FILE}; completed protocol artifacts and attempt evidence were preserved."
fi

echo "FINAL_RUN_RC=${RUN_RC}"
echo "START_TIME=${START_TIME}"
echo "END_TIME=${END_TIME}"
echo "TOTAL_ELAPSED=$(format_duration "${ELAPSED}")"
echo "COMPLETION_STATUS=${COMPLETION_STATUS}"
echo "RESULT_DIRECTORY=${SUITE_DIR}"
echo "LOG_DIRECTORY=${LOG_DIR}"
echo "METRICS_PATH=${SUITE_DIR}/suite_metrics.json"
echo "FAILED_STAGE=${FAILED_STAGE}"
echo "ERROR_SUMMARY=${ERROR_SUMMARY}"
echo "[INFO] The experiment is finished. This tmux pane now remains in an interactive shell."

exec bash -i
