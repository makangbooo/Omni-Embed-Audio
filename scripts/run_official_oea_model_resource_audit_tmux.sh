#!/usr/bin/env bash
set -uo pipefail

if [[ -z "${TMUX:-}" ]]; then
  echo "[ERROR] This long audit must run inside a visible tmux pane." >&2
  echo "[INFO] Create or attach a named session, then run this script in that pane." >&2
  exit 2
fi

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <variant_id>" >&2
  exit 2
fi

VARIANT_ID="$1"
case "${VARIANT_ID}" in
  oea_nemo3b)
    PAPER_MODEL="OEA-Nemo3B"
    EXPECTED_BYTES=18889955098
    EXPECTED_ASSETS=2
    ;;
  oea_nemo3b_cl)
    PAPER_MODEL="OEA-Nemo3B (+Cl)"
    EXPECTED_BYTES=18949027203
    EXPECTED_ASSETS=2
    ;;
  oea_qwen3b)
    PAPER_MODEL="OEA-Qwen3B"
    EXPECTED_BYTES=21515017517
    EXPECTED_ASSETS=2
    ;;
  oea_qwen3b_cl)
    PAPER_MODEL="OEA-Qwen3B (+Cl)"
    EXPECTED_BYTES=21515014895
    EXPECTED_ASSETS=2
    ;;
  oea_qwen7b)
    PAPER_MODEL="OEA-Qwen7B"
    EXPECTED_BYTES=40319908316
    EXPECTED_ASSETS=2
    ;;
  oea_qwen7b_cl)
    PAPER_MODEL="OEA-Qwen7B (+Cl)"
    EXPECTED_BYTES=40319908438
    EXPECTED_ASSETS=2
    ;;
  *)
    echo "[ERROR] Unsupported official OEA variant: ${VARIANT_ID}" >&2
    exit 2
    ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
AUDIT_RUN_ID="official_model_resource_audit_${VARIANT_ID}_${RUN_STAMP}"
AUDIT_DIR="${ROOT_DIR}/logs/${AUDIT_RUN_ID}"
LOG_DIR="${TMUX_LOG_DIR:-${ROOT_DIR}/logs/tmux_${AUDIT_RUN_ID}}"
LOG_FILE="${LOG_DIR}/combined.log"
METRICS_PATH="${AUDIT_DIR}/model_resource_audit.json"

if [[ -e "${LOG_DIR}" || -e "${AUDIT_DIR}" ]]; then
  echo "[ERROR] Refusing to reuse an audit or tmux log directory." >&2
  echo "[ERROR] AUDIT_DIR=${AUDIT_DIR}" >&2
  echo "[ERROR] LOG_DIR=${LOG_DIR}" >&2
  exit 3
fi
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2> >(tee -a "${LOG_FILE}" >&2)

START_EPOCH="$(date +%s)"
START_TIME="$(date -Is)"
GIT_COMMIT="$(git rev-parse HEAD)"
GIT_STATUS="$(git status --short)"

format_duration() {
  local total="${1:-0}"
  if (( total < 0 )); then total=0; fi
  printf '%02d:%02d:%02d' "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
}

if [[ -n "${GIT_STATUS}" ]]; then
  echo "[ERROR] Formal resource audit requires a clean Git worktree." >&2
  printf '%s\n' "${GIT_STATUS}" >&2
  exit 3
fi

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate oea-repro
export CUDA_VISIBLE_DEVICES=""

echo "EXPERIMENT_NAME=${PAPER_MODEL} exact model resource hash/provenance audit"
echo "PAPER_EXPERIMENT_STAGE=MODEL-03/04 resource gate before Tables 2, 3, and 12-15 evaluation"
echo "GIT_COMMIT=${GIT_COMMIT}"
echo "GIT_STATUS_SHORT=<clean>"
echo "MODEL=${PAPER_MODEL}; pinned base snapshot + official checkpoint snapshot"
echo "DATASET=none"
echo "RESOURCES=CPU only; GPU disabled; Hugging Face revision metadata network access"
echo "TOTAL_WORKLOAD=${EXPECTED_ASSETS} assets; approximately ${EXPECTED_BYTES} local bytes hashed"
echo "ESTIMATED_TOTAL_TIME=10-60 minutes"
echo "ETA_BASIS=pinned snapshot sizes and shared-storage sequential-read variability documented in docs/model_resource_audit.md"
echo "CACHE_DIRECTORY=none; no model or data download"
echo "RESULT_DIRECTORY=${AUDIT_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "METRICS_PATH=${METRICS_PATH}"
echo "START_TIME=${START_TIME}"
echo "RESUME=not required; interruption leaves model files unchanged and a new unique audit run can be started"

RUN_ID="${AUDIT_RUN_ID}" \
  bash scripts/run_official_oea_model_resource_audit.sh "${VARIANT_ID}" &
AUDIT_PID=$!
LAST_EPOCH="${START_EPOCH}"
LAST_RCHAR=0

while kill -0 "${AUDIT_PID}" 2>/dev/null; do
  sleep 15
  NOW_EPOCH="$(date +%s)"
  ELAPSED="$((NOW_EPOCH - START_EPOCH))"
  INTERVAL="$((NOW_EPOCH - LAST_EPOCH))"
  PYTHON_PID="$(pgrep -P "${AUDIT_PID}" -f 'audit_official_oea_variant_resources.py' | head -n 1 || true)"
  RCHAR="${LAST_RCHAR}"
  if [[ -n "${PYTHON_PID}" && -r "/proc/${PYTHON_PID}/io" ]]; then
    RCHAR="$(awk '$1 == "rchar:" {print $2}' "/proc/${PYTHON_PID}/io")"
  fi
  if [[ ! "${RCHAR}" =~ ^[0-9]+$ ]]; then RCHAR=0; fi
  ACCOUNTED_BYTES="${RCHAR}"
  if (( ACCOUNTED_BYTES > EXPECTED_BYTES )); then ACCOUNTED_BYTES="${EXPECTED_BYTES}"; fi

  PROCESSED_ASSETS=0
  AUDIT_STATUS="running"
  if [[ -f "${METRICS_PATH}" ]]; then
    read -r PROCESSED_ASSETS AUDIT_STATUS < <(
      python - "${METRICS_PATH}" <<'PY'
import json
import sys
from pathlib import Path

try:
    report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except Exception:
    print("0 running")
else:
    print(len(report.get("assets", [])), report.get("status", "running"))
PY
    )
  fi
  if [[ "${AUDIT_STATUS}" != "running" && "${PROCESSED_ASSETS}" -ge "${EXPECTED_ASSETS}" ]]; then
    # The Python child can exit between polling samples. Once its atomic report
    # confirms that every asset was processed, report completed workload rather
    # than a stale /proc byte counter from the preceding sample.
    ACCOUNTED_BYTES="${EXPECTED_BYTES}"
  fi

  DELTA_BYTES="$((RCHAR - LAST_RCHAR))"
  if (( DELTA_BYTES < 0 )); then DELTA_BYTES=0; fi
  CURRENT_BPS=0
  if (( INTERVAL > 0 )); then CURRENT_BPS="$((DELTA_BYTES / INTERVAL))"; fi
  OVERALL_BPS=0
  if (( ELAPSED > 0 )); then OVERALL_BPS="$((ACCOUNTED_BYTES / ELAPSED))"; fi
  REMAINING_SECONDS=0
  if (( OVERALL_BPS > 0 && ACCOUNTED_BYTES < EXPECTED_BYTES )); then
    REMAINING_SECONDS="$(((EXPECTED_BYTES - ACCOUNTED_BYTES) / OVERALL_BPS))"
  fi
  PERCENT="$(awk -v done="${ACCOUNTED_BYTES}" -v total="${EXPECTED_BYTES}" 'BEGIN {printf "%.1f", done * 100 / total}')"
  CURRENT_MIB_S="$(awk -v value="${CURRENT_BPS}" 'BEGIN {printf "%.2f", value / 1048576}')"
  FINISH_TIME="unknown"
  if (( REMAINING_SECONDS > 0 )); then
    FINISH_TIME="$(date -d "@$((${NOW_EPOCH} + REMAINING_SECONDS))" -Is)"
  fi

  echo "PROGRESS stage=metadata_and_content_audit assets=${PROCESSED_ASSETS}/${EXPECTED_ASSETS} bytes=${ACCOUNTED_BYTES}/${EXPECTED_BYTES} percent=${PERCENT}% stage_elapsed=$(format_duration "${ELAPSED}") overall_elapsed=$(format_duration "${ELAPSED}") current_throughput_mib_s=${CURRENT_MIB_S} stage_eta=$(format_duration "${REMAINING_SECONDS}") overall_eta=$(format_duration "${REMAINING_SECONDS}") estimated_finish=${FINISH_TIME} audit_status=${AUDIT_STATUS}"
  LAST_EPOCH="${NOW_EPOCH}"
  LAST_RCHAR="${RCHAR}"
done

wait "${AUDIT_PID}"
RUN_RC=$?
END_EPOCH="$(date +%s)"
END_TIME="$(date -Is)"
ELAPSED="$((END_EPOCH - START_EPOCH))"
AUDIT_STATUS="missing"
if [[ -f "${METRICS_PATH}" ]]; then
  AUDIT_STATUS="$(python - "${METRICS_PATH}" <<'PY'
import json
import sys
from pathlib import Path

print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("status", "missing"))
PY
)"
fi

if [[ "${RUN_RC}" -eq 0 && "${AUDIT_STATUS}" == "complete" ]]; then
  COMPLETION_STATUS="complete"
  FAILED_STAGE="none"
  ERROR_SUMMARY="none"
elif [[ "${RUN_RC}" -eq 2 || "${AUDIT_STATUS}" == "incomplete" ]]; then
  COMPLETION_STATUS="incomplete"
  FAILED_STAGE="metadata_and_content_audit"
  ERROR_SUMMARY="A pinned resource is incomplete; no model file was modified. Review the metrics and combined log."
else
  COMPLETION_STATUS="failed"
  FAILED_STAGE="metadata_and_content_audit"
  ERROR_SUMMARY="The audit failed validation or metadata access; no model file was modified. Review the metrics and combined log."
fi

echo "FINAL_RUN_RC=${RUN_RC}"
echo "START_TIME=${START_TIME}"
echo "END_TIME=${END_TIME}"
echo "TOTAL_ELAPSED=$(format_duration "${ELAPSED}")"
echo "COMPLETION_STATUS=${COMPLETION_STATUS}"
echo "AUDIT_STATUS=${AUDIT_STATUS}"
echo "RESULT_DIRECTORY=${AUDIT_DIR}"
echo "LOG_DIRECTORY=${LOG_DIR}"
echo "METRICS_PATH=${METRICS_PATH}"
echo "FAILED_STAGE=${FAILED_STAGE}"
echo "ERROR_SUMMARY=${ERROR_SUMMARY}"
echo "[INFO] The audit is finished. This tmux pane now remains in an interactive shell."

exec bash -i
