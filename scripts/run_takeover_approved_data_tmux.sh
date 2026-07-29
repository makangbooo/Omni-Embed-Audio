#!/usr/bin/env bash
set -uo pipefail

if [[ -z "${TMUX:-}" ]]; then
  echo "[ERROR] This data task must run inside a visible tmux pane." >&2
  exit 2
fi
if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <data04|data06|data08>" >&2
  exit 2
fi

PIPELINE_ID="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -n "$(git status --short)" ]]; then
  echo "[ERROR] Refusing a formal data run from a dirty worktree." >&2
  git status --short >&2
  exit 3
fi

case "${PIPELINE_ID}" in
  data04)
    EXPERIMENT_NAME="DATA-04 existing-asset verification plus DATA-05 MECAT validation"
    DATASET="MECAT-Caption 00A/test public 848-row release"
    TOTAL_STAGES=2
    ESTIMATED_RANGE="2-20 minutes"
    ETA_BASIS="173 MB existing-asset SHA256 plus extraction and decode of 848 FLAC files"
    TOTAL_ESTIMATED_SECONDS=600
    STAGE_NAMES=("data04_asset_verify_or_resume" "data05_extract_decode_validate")
    STAGE_ESTIMATES=(120 480)
    CACHE_DIRECTORY="/home/jg525/datasets/oea/mecat_caption_be4a24c3/.cache"
    RESULT_DIRECTORY="/home/jg525/datasets/oea/mecat_caption_be4a24c3"
    METRICS_PATH="/home/jg525/datasets/oea/mecat_caption_be4a24c3/manifests/mecat_00a_test_manifest.jsonl"
    ;;
  data06)
    EXPERIMENT_NAME="DATA-06 AudioCaps metadata download plus DATA-07 validation"
    DATASET="AudioCaps v2 metadata at d004db3ea1b01cf4fd0347dd8d27db90cadc8809"
    TOTAL_STAGES=2
    ESTIMATED_RANGE="1-10 minutes"
    ETA_BASIS="6.9 MB pinned metadata plus schema, count, and released-UIQ validation"
    TOTAL_ESTIMATED_SECONDS=300
    STAGE_NAMES=("data06_download_verify" "data07_metadata_uiq_validate")
    STAGE_ESTIMATES=(120 180)
    CACHE_DIRECTORY="/home/jg525/datasets/oea/audiocaps_v2_d004db3/.cache"
    RESULT_DIRECTORY="/home/jg525/datasets/oea/audiocaps_v2_d004db3"
    METRICS_PATH="/home/jg525/datasets/oea/audiocaps_v2_d004db3/manifests"
    ;;
  data08)
    EXPERIMENT_NAME="DATA-08 WavCaps metadata download plus DATA-09 leakage audit"
    DATASET="WavCaps metadata at 0930ec11ded28fa0eaa910fde2f6fc3538acbeac"
    TOTAL_STAGES=2
    ESTIMATED_RANGE="5-30 minutes"
    ETA_BASIS="176.9 MB pinned metadata plus a 403,050-row duration and leakage pass"
    TOTAL_ESTIMATED_SECONDS=900
    STAGE_NAMES=("data08_download_verify" "data09_duration_leakage_audit")
    STAGE_ESTIMATES=(480 420)
    CACHE_DIRECTORY="/home/jg525/datasets/oea/wavcaps_0930ec11/.cache"
    RESULT_DIRECTORY="/home/jg525/datasets/oea/wavcaps_0930ec11"
    METRICS_PATH="/home/jg525/datasets/oea/wavcaps_0930ec11/manifests/wavcaps_all_metadata_audit_manifest.jsonl"
    ;;
  *)
    echo "[ERROR] Unsupported pipeline: ${PIPELINE_ID}" >&2
    exit 2
    ;;
esac

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="oea_takeover_${PIPELINE_ID}_${RUN_STAMP}"
LOG_DIRECTORY="${ROOT_DIR}/logs/${RUN_ID}"
LOG_FILE="${LOG_DIRECTORY}/combined.log"
if [[ -e "${LOG_DIRECTORY}" ]]; then
  echo "[ERROR] Log directory exists; refusing to overwrite: ${LOG_DIRECTORY}" >&2
  exit 4
fi
mkdir -p "${LOG_DIRECTORY}"
exec > >(tee -a "${LOG_FILE}") 2> >(tee -a "${LOG_FILE}" >&2)

START_EPOCH="$(date +%s)"
START_TIME="$(date -Is)"
GIT_COMMIT="$(git rev-parse HEAD)"
CURRENT_STAGE="preflight"

format_duration() {
  local total="$1"
  printf '%02d:%02d:%02d' "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
}

print_progress() {
  local stage_index="$1"
  local stage_name="$2"
  local stage_start="$3"
  local stage_estimate="$4"
  local now stage_elapsed overall_elapsed stage_remaining overall_remaining
  local completed percent throughput expected_epoch
  now="$(date +%s)"
  stage_elapsed="$((now - stage_start))"
  overall_elapsed="$((now - START_EPOCH))"
  stage_remaining="$((stage_estimate > stage_elapsed ? stage_estimate - stage_elapsed : 0))"
  overall_remaining="$((TOTAL_ESTIMATED_SECONDS > overall_elapsed ? TOTAL_ESTIMATED_SECONDS - overall_elapsed : 0))"
  completed="$((stage_index - 1))"
  percent="$((completed * 100 / TOTAL_STAGES))"
  throughput="$(awk -v n="${completed}" -v s="${overall_elapsed}" 'BEGIN { if (s > 0) printf "%.4f", n / s; else print "0.0000" }')"
  expected_epoch="$((now + overall_remaining))"
  echo "[PROGRESS] stage=${stage_name} current=${completed}/${TOTAL_STAGES} percent=${percent}% stage_elapsed=$(format_duration "${stage_elapsed}") overall_elapsed=$(format_duration "${overall_elapsed}") throughput=${throughput}_completed_stages_per_second stage_remaining=$(format_duration "${stage_remaining}") overall_remaining=$(format_duration "${overall_remaining}") expected_completion=$(date -d "@${expected_epoch}" -Is)"
}

run_selected_stage() {
  local stage_index="$1"
  case "${PIPELINE_ID}:${stage_index}" in
    data04:1) bash scripts/download_data04_mecat_00a_test.sh ;;
    data04:2) bash scripts/run_data05_mecat_validation.sh ;;
    data06:1) bash scripts/download_data06_audiocaps_v2_metadata.sh ;;
    data06:2) bash scripts/run_data07_audiocaps_v2_metadata_validation.sh ;;
    data08:1) bash scripts/download_data08_wavcaps_metadata.sh ;;
    data08:2) bash scripts/run_data09_wavcaps_metadata_audit.sh ;;
    *) return 2 ;;
  esac
}

echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "GIT_COMMIT=${GIT_COMMIT}"
echo "MODEL=none"
echo "DATASET=${DATASET}"
echo "RESOURCES=CPU-only; nproc=$(nproc); CUDA_VISIBLE_DEVICES is disabled by each child stage"
free -h || true
echo "TOTAL_WORKLOAD=${TOTAL_STAGES} stages"
echo "ESTIMATED_TOTAL_TIME=${ESTIMATED_RANGE}"
echo "ETA_BASIS=${ETA_BASIS}"
echo "CACHE_DIRECTORY=${CACHE_DIRECTORY}"
echo "RESULT_DIRECTORY=${RESULT_DIRECTORY}"
echo "LOG_FILE=${LOG_FILE}"
echo "START_TIME=${START_TIME}"

RUN_RC=0
FAILED_STAGE="none"
for ((index = 1; index <= TOTAL_STAGES; index++)); do
  CURRENT_STAGE="${STAGE_NAMES[index - 1]}"
  STAGE_ESTIMATE="${STAGE_ESTIMATES[index - 1]}"
  STAGE_START="$(date +%s)"
  echo "[INFO] Starting stage ${index}/${TOTAL_STAGES}: ${CURRENT_STAGE}"
  run_selected_stage "${index}" &
  CHILD_PID=$!
  while kill -0 "${CHILD_PID}" 2>/dev/null; do
    print_progress "${index}" "${CURRENT_STAGE}" "${STAGE_START}" "${STAGE_ESTIMATE}"
    sleep 10
  done
  wait "${CHILD_PID}"
  RUN_RC=$?
  if [[ "${RUN_RC}" -ne 0 ]]; then
    FAILED_STAGE="${CURRENT_STAGE}"
    break
  fi
  COMPLETED_NOW="$((index))"
  ELAPSED_NOW="$(($(date +%s) - START_EPOCH))"
  echo "[PROGRESS] stage=${CURRENT_STAGE} current=${COMPLETED_NOW}/${TOTAL_STAGES} percent=$((COMPLETED_NOW * 100 / TOTAL_STAGES))% stage_elapsed=$(format_duration "$(($(date +%s) - STAGE_START))") overall_elapsed=$(format_duration "${ELAPSED_NOW}") throughput=$(awk -v n="${COMPLETED_NOW}" -v s="${ELAPSED_NOW}" 'BEGIN { if (s > 0) printf "%.4f", n / s; else print "0.0000" }')_completed_stages_per_second stage_remaining=00:00:00 overall_remaining=$(format_duration "$((TOTAL_ESTIMATED_SECONDS > ELAPSED_NOW ? TOTAL_ESTIMATED_SECONDS - ELAPSED_NOW : 0))") expected_completion=$(date -d "@$(($(date +%s) + (TOTAL_ESTIMATED_SECONDS > ELAPSED_NOW ? TOTAL_ESTIMATED_SECONDS - ELAPSED_NOW : 0)))" -Is)"
done

END_EPOCH="$(date +%s)"
END_TIME="$(date -Is)"
TOTAL_ELAPSED="$((END_EPOCH - START_EPOCH))"
if [[ "${RUN_RC}" -eq 0 ]]; then
  COMPLETION_STATUS="complete"
  ERROR_SUMMARY="none"
else
  COMPLETION_STATUS="failed"
  ERROR_SUMMARY="See the last error lines and ${LOG_FILE}; child-stage evidence and partial downloads were preserved."
fi

echo "FINAL_RUN_RC=${RUN_RC}"
echo "START_TIME=${START_TIME}"
echo "END_TIME=${END_TIME}"
echo "TOTAL_ELAPSED=$(format_duration "${TOTAL_ELAPSED}")"
echo "COMPLETION_STATUS=${COMPLETION_STATUS}"
echo "RESULT_DIRECTORY=${RESULT_DIRECTORY}"
echo "LOG_DIRECTORY=${LOG_DIRECTORY}"
echo "METRICS_PATH=${METRICS_PATH}"
echo "FAILED_STAGE=${FAILED_STAGE}"
echo "ERROR_SUMMARY=${ERROR_SUMMARY}"
echo "[INFO] The data task is finished. This tmux pane now remains in an interactive shell."

exec bash -i
