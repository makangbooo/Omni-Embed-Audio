#!/usr/bin/env bash
set -uo pipefail

if [[ -z "${TMUX:-}" ]]; then
  echo "[ERROR] This long checkpoint task must run inside a visible tmux pane." >&2
  exit 2
fi
if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 <variant_id> <resource-audit-json> <expected-audit-sha256>" >&2
  exit 2
fi

VARIANT_ID="$1"
RESOURCE_AUDIT="$2"
EXPECTED_AUDIT_SHA256="$3"
case "${VARIANT_ID}" in
  oea_nemo3b)
    PAPER_MODEL="OEA-Nemo3B"
    CHECKPOINT_SUBPATH="OEA-Nemo3B-AC/step_400_best.pt"
    DERIVED_SUBPATH="OEA-Nemo3B-AC/step_400_best_inference_only.pt"
    SOURCE_BYTES=9466826153
    ;;
  oea_nemo3b_cl)
    PAPER_MODEL="OEA-Nemo3B (+Cl)"
    CHECKPOINT_SUBPATH="OEA-Nemo3B-Cl/step_450_best.pt"
    DERIVED_SUBPATH="OEA-Nemo3B-Cl/step_450_best_inference_only.pt"
    SOURCE_BYTES=9466826217
    ;;
  oea_qwen3b)
    PAPER_MODEL="OEA-Qwen3B"
    CHECKPOINT_SUBPATH="OEA-Qwen3B-AC/step_350.pt"
    DERIVED_SUBPATH="OEA-Qwen3B-AC/step_350_inference_only.pt"
    SOURCE_BYTES=9466835918
    ;;
  oea_qwen3b_cl)
    PAPER_MODEL="OEA-Qwen3B (+Cl)"
    CHECKPOINT_SUBPATH="OEA-Qwen3B-Cl/step_40.pt"
    DERIVED_SUBPATH="OEA-Qwen3B-Cl/step_40_inference_only.pt"
    SOURCE_BYTES=9466833858
    ;;
  oea_qwen7b)
    PAPER_MODEL="OEA-Qwen7B"
    CHECKPOINT_SUBPATH="OEA-Qwen7B-AC/step_300.pt"
    DERIVED_SUBPATH="OEA-Qwen7B-AC/step_300_inference_only.pt"
    SOURCE_BYTES=17940602533
    ;;
  oea_qwen7b_cl)
    PAPER_MODEL="OEA-Qwen7B (+Cl)"
    CHECKPOINT_SUBPATH="OEA-Qwen7B-Cl/step_330.pt"
    DERIVED_SUBPATH="OEA-Qwen7B-Cl/step_330_inference_only.pt"
    SOURCE_BYTES=17940602661
    ;;
  *)
    echo "[ERROR] Unsupported official OEA variant: ${VARIANT_ID}" >&2
    exit 2
    ;;
esac
if [[ ! "${EXPECTED_AUDIT_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "[ERROR] Expected resource-audit SHA256 must be lowercase hexadecimal." >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

MODEL_ROOT="${MODEL_ROOT:-/home/jg525/models/oea}"
RESOURCE_AUDIT="$(readlink -f "${RESOURCE_AUDIT}")"
SOURCE_CHECKPOINT="${MODEL_ROOT}/${CHECKPOINT_SUBPATH}"
DERIVED_CHECKPOINT="${MODEL_ROOT}/${DERIVED_SUBPATH}"
TOTAL_BYTES="$((SOURCE_BYTES * 2))"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="official_checkpoint_lock_${VARIANT_ID}_${RUN_STAMP}"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"
PREPARATION_DIR="${RUN_DIR}/checkpoint_preparation"
PREPARATION_MANIFEST="${PREPARATION_DIR}/preparation_manifest.json"
PORTABLE_LOCK="${RUN_DIR}/${VARIANT_ID}.portable_model_lock.json"
LOG_FILE="${RUN_DIR}/combined.log"

if [[ ! -f "${RESOURCE_AUDIT}" ]]; then
  echo "[ERROR] Resource audit does not exist: ${RESOURCE_AUDIT}" >&2
  exit 2
fi
ACTUAL_AUDIT_SHA256="$(sha256sum "${RESOURCE_AUDIT}" | awk '{print $1}')"
if [[ "${ACTUAL_AUDIT_SHA256}" != "${EXPECTED_AUDIT_SHA256}" ]]; then
  echo "[ERROR] Resource-audit SHA256 mismatch." >&2
  echo "[ERROR] actual=${ACTUAL_AUDIT_SHA256}" >&2
  exit 2
fi
if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
  echo "[ERROR] Source checkpoint is missing: ${SOURCE_CHECKPOINT}" >&2
  exit 2
fi
if [[ "$(stat -c %s "${SOURCE_CHECKPOINT}")" -ne "${SOURCE_BYTES}" ]]; then
  echo "[ERROR] Source checkpoint size differs from the fixed registry." >&2
  exit 2
fi
if [[ -e "${RUN_DIR}" ]]; then
  echo "[ERROR] Refusing to reuse run directory: ${RUN_DIR}" >&2
  exit 3
fi
mkdir -p "${RUN_DIR}"
exec > >(tee -a "${LOG_FILE}") 2> >(tee -a "${LOG_FILE}" >&2)

START_EPOCH="$(date +%s)"
START_TIME="$(date -Is)"
GIT_COMMIT="$(git rev-parse HEAD)"
GIT_STATUS="$(git status --short --untracked-files=all)"

format_duration() {
  local total="${1:-0}"
  if (( total < 0 )); then total=0; fi
  printf '%02d:%02d:%02d' "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
}

write_value() {
  printf '%s\n' "$2" > "${RUN_DIR}/$1"
}

write_value git_commit.txt "${GIT_COMMIT}"
write_value git_status.txt "${GIT_STATUS}"
{
  printf 'MODEL_ROOT=%q ' "${MODEL_ROOT}"
  printf '%q ' "$0" "$@"
  printf '\n'
} > "${RUN_DIR}/command.sh"

if [[ -n "${GIT_STATUS}" ]]; then
  echo "[ERROR] Formal checkpoint preparation requires a clean Git worktree." >&2
  printf '%s\n' "${GIT_STATUS}" >&2
  exit 3
fi

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate oea-repro
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODE="extract_new_derived"
PREPARATION_MODE=()
if [[ -f "${DERIVED_CHECKPOINT}" ]]; then
  MODE="verify_existing_derived"
  PREPARATION_MODE=(--verify-existing-derived)
fi

echo "EXPERIMENT_NAME=${PAPER_MODEL} checkpoint structure, inference-only artifact, and portable lock"
echo "PAPER_EXPERIMENT_STAGE=official checkpoint gate before Tables 2, 3, and 12-15"
echo "GIT_COMMIT=${GIT_COMMIT}"
echo "GIT_STATUS_SHORT=<clean>"
echo "MODEL=${PAPER_MODEL}; source=${SOURCE_CHECKPOINT}"
echo "DATASET=none"
echo "RESOURCES=CPU only; GPU disabled; network disabled"
echo "CPU_MODEL=$(lscpu | awk -F: '$1 == "Model name" {sub(/^[[:space:]]+/, "", $2); print $2; exit}')"
echo "CPU_COUNT=$(nproc)"
echo "MEMORY=$(free -h | awk '$1 == "Mem:" {print $2}')"
echo "TOTAL_WORKLOAD=2 source-checkpoint validation passes; approximately ${TOTAL_BYTES} bytes"
echo "ESTIMATED_TOTAL_TIME=10-45 minutes"
echo "ETA_BASIS=two sequential reads of the fixed ${SOURCE_BYTES}-byte checkpoint; shared-storage throughput dominates"
echo "CACHE_DIRECTORY=none"
echo "RESULT_DIRECTORY=${RUN_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "METRICS_PATH=${PREPARATION_MANIFEST}"
echo "RESOURCE_AUDIT=${RESOURCE_AUDIT}"
echo "RESOURCE_AUDIT_SHA256=${ACTUAL_AUDIT_SHA256}"
echo "DERIVED_CHECKPOINT=${DERIVED_CHECKPOINT}"
echo "PORTABLE_MODEL_LOCK=${PORTABLE_LOCK}"
echo "MODE=${MODE}"
echo "START_TIME=${START_TIME}"
echo "RESUME=rerun with a new RUN_ID; a completed derived artifact is strictly verified and never overwritten"

python scripts/prepare_official_oea_checkpoint.py \
  --variant "${VARIANT_ID}" \
  --model-root "${MODEL_ROOT}" \
  --output-dir "${PREPARATION_DIR}" \
  "${PREPARATION_MODE[@]}" &
PREPARATION_PID=$!
LAST_EPOCH="${START_EPOCH}"
LAST_ACCOUNTED=0
CURRENT_STAGE="checkpoint_inspection"
STAGE_START="${START_EPOCH}"

while kill -0 "${PREPARATION_PID}" 2>/dev/null; do
  sleep 15
  NOW_EPOCH="$(date +%s)"
  ELAPSED="$((NOW_EPOCH - START_EPOCH))"
  NEXT_STAGE="checkpoint_inspection"
  PREPARATION_STATUS="running"
  INSPECTION_DONE=0
  EXTRACTION_DONE=0
  if [[ -f "${PREPARATION_MANIFEST}" ]]; then
    read -r PREPARATION_STATUS INSPECTION_DONE EXTRACTION_DONE < <(
      python - "${PREPARATION_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

try:
    report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except Exception:
    print("running 0 0")
else:
    print(
        report.get("status", "running"),
        int(report.get("inspection_exit_code") == 0),
        int(report.get("extraction_exit_code") == 0),
    )
PY
    )
  fi
  if (( INSPECTION_DONE == 1 )); then NEXT_STAGE="checkpoint_extraction"; fi
  if [[ "${PREPARATION_STATUS}" == "complete" ]]; then NEXT_STAGE="checkpoint_preparation_complete"; fi
  if [[ "${NEXT_STAGE}" != "${CURRENT_STAGE}" ]]; then
    CURRENT_STAGE="${NEXT_STAGE}"
    STAGE_START="${NOW_EPOCH}"
  fi

  CHILD_PID="$(pgrep -P "${PREPARATION_PID}" | head -n 1 || true)"
  CHILD_RCHAR=0
  if [[ -n "${CHILD_PID}" && -r "/proc/${CHILD_PID}/io" ]]; then
    CHILD_RCHAR="$(awk '$1 == "rchar:" {print $2}' "/proc/${CHILD_PID}/io")"
  fi
  if [[ ! "${CHILD_RCHAR}" =~ ^[0-9]+$ ]]; then CHILD_RCHAR=0; fi
  if (( CHILD_RCHAR > SOURCE_BYTES )); then CHILD_RCHAR="${SOURCE_BYTES}"; fi

  if [[ "${CURRENT_STAGE}" == "checkpoint_inspection" ]]; then
    ACCOUNTED="${CHILD_RCHAR}"
    STAGE_BYTES="${CHILD_RCHAR}"
  elif [[ "${CURRENT_STAGE}" == "checkpoint_extraction" ]]; then
    ACCOUNTED="$((SOURCE_BYTES + CHILD_RCHAR))"
    STAGE_BYTES="${CHILD_RCHAR}"
  else
    ACCOUNTED="${TOTAL_BYTES}"
    STAGE_BYTES="${SOURCE_BYTES}"
  fi
  if (( ACCOUNTED < LAST_ACCOUNTED )); then ACCOUNTED="${LAST_ACCOUNTED}"; fi
  INTERVAL="$((NOW_EPOCH - LAST_EPOCH))"
  DELTA="$((ACCOUNTED - LAST_ACCOUNTED))"
  if (( DELTA < 0 )); then DELTA=0; fi
  CURRENT_BPS=0
  if (( INTERVAL > 0 )); then CURRENT_BPS="$((DELTA / INTERVAL))"; fi
  OVERALL_BPS=0
  if (( ELAPSED > 0 )); then OVERALL_BPS="$((ACCOUNTED / ELAPSED))"; fi
  STAGE_REMAINING=0
  if (( CURRENT_BPS > 0 && STAGE_BYTES < SOURCE_BYTES )); then
    STAGE_REMAINING="$(((SOURCE_BYTES - STAGE_BYTES) / CURRENT_BPS))"
  fi
  OVERALL_REMAINING=0
  if (( OVERALL_BPS > 0 && ACCOUNTED < TOTAL_BYTES )); then
    OVERALL_REMAINING="$(((TOTAL_BYTES - ACCOUNTED) / OVERALL_BPS))"
  fi
  PERCENT="$(awk -v done="${ACCOUNTED}" -v total="${TOTAL_BYTES}" 'BEGIN {printf "%.1f", done * 100 / total}')"
  CURRENT_MIB_S="$(awk -v value="${CURRENT_BPS}" 'BEGIN {printf "%.2f", value / 1048576}')"
  FINISH_TIME="unknown"
  if (( OVERALL_REMAINING > 0 )); then
    FINISH_TIME="$(date -d "@$((${NOW_EPOCH} + OVERALL_REMAINING))" -Is)"
  fi
  echo "PROGRESS stage=${CURRENT_STAGE} bytes=${ACCOUNTED}/${TOTAL_BYTES} percent=${PERCENT}% stage_elapsed=$(format_duration "$((NOW_EPOCH - STAGE_START))") overall_elapsed=$(format_duration "${ELAPSED}") current_throughput_mib_s=${CURRENT_MIB_S} stage_eta=$(format_duration "${STAGE_REMAINING}") overall_eta=$(format_duration "${OVERALL_REMAINING}") estimated_finish=${FINISH_TIME} preparation_status=${PREPARATION_STATUS}"
  LAST_EPOCH="${NOW_EPOCH}"
  LAST_ACCOUNTED="${ACCOUNTED}"
done

wait "${PREPARATION_PID}"
PREPARATION_RC=$?
write_value preparation_exit_code.txt "${PREPARATION_RC}"

LOCK_RC=99
if [[ "${PREPARATION_RC}" -eq 0 ]]; then
  echo "PROGRESS stage=model_lock bytes=${TOTAL_BYTES}/${TOTAL_BYTES} percent=100.0% stage_elapsed=00:00:00 overall_elapsed=$(format_duration "$(($(date +%s) - START_EPOCH))") current_throughput_mib_s=0.00 stage_eta=00:00:00 overall_eta=00:00:00 estimated_finish=$(date -Is) preparation_status=complete"
  python scripts/build_official_oea_model_lock.py \
    --variant "${VARIANT_ID}" \
    --model-resource-audit "${RESOURCE_AUDIT}" \
    --checkpoint-preparation "${PREPARATION_MANIFEST}" \
    --output "${PORTABLE_LOCK}"
  LOCK_RC=$?
fi
write_value model_lock_exit_code.txt "${LOCK_RC}"

FINAL_RC="${PREPARATION_RC}"
if [[ "${PREPARATION_RC}" -eq 0 ]]; then FINAL_RC="${LOCK_RC}"; fi
write_value final_exit_code.txt "${FINAL_RC}"
if [[ -f "${DERIVED_CHECKPOINT}" ]]; then
  sha256sum "${DERIVED_CHECKPOINT}" > "${RUN_DIR}/derived_checkpoint_sha256.txt"
fi
if [[ -f "${PORTABLE_LOCK}" ]]; then
  sha256sum "${PORTABLE_LOCK}" > "${RUN_DIR}/portable_model_lock_sha256.txt"
fi

END_EPOCH="$(date +%s)"
END_TIME="$(date -Is)"
ELAPSED="$((END_EPOCH - START_EPOCH))"
if [[ "${FINAL_RC}" -eq 0 ]]; then
  COMPLETION_STATUS="complete"
  FAILED_STAGE="none"
  ERROR_SUMMARY="none"
elif [[ "${PREPARATION_RC}" -ne 0 ]]; then
  COMPLETION_STATUS="failed"
  FAILED_STAGE="checkpoint_preparation"
  ERROR_SUMMARY="Structure inspection or derived-checkpoint extraction failed; source checkpoint was not modified."
else
  COMPLETION_STATUS="failed"
  FAILED_STAGE="model_lock"
  ERROR_SUMMARY="Portable model-lock validation failed; checkpoint preparation evidence was retained."
fi

echo "FINAL_RUN_RC=${FINAL_RC}"
echo "START_TIME=${START_TIME}"
echo "END_TIME=${END_TIME}"
echo "TOTAL_ELAPSED=$(format_duration "${ELAPSED}")"
echo "COMPLETION_STATUS=${COMPLETION_STATUS}"
echo "RESULT_DIRECTORY=${RUN_DIR}"
echo "LOG_DIRECTORY=${RUN_DIR}"
echo "METRICS_PATH=${PREPARATION_MANIFEST}"
echo "DERIVED_CHECKPOINT=${DERIVED_CHECKPOINT}"
echo "PORTABLE_MODEL_LOCK=${PORTABLE_LOCK}"
echo "FAILED_STAGE=${FAILED_STAGE}"
echo "ERROR_SUMMARY=${ERROR_SUMMARY}"
cat "${RUN_DIR}/derived_checkpoint_sha256.txt" 2>/dev/null || true
cat "${RUN_DIR}/portable_model_lock_sha256.txt" 2>/dev/null || true
echo "[INFO] The checkpoint task is finished. This tmux pane now remains in an interactive shell."

exec bash -i
