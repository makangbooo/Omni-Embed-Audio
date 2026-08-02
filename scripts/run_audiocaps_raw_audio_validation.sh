#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
AUDIO_ROOT="${DATA_ROOT}/audiocaps_raw_audio"
DATASET_ROOT="${DATA_ROOT}/audiocaps_v2_d004db3"
SOURCE_MANIFEST="${DATASET_ROOT}/manifests/audiocaps_v2_test_manifest.jsonl"
OUTPUT_MANIFEST="${DATASET_ROOT}/manifests/audiocaps_v2_test_audio_manifest.jsonl"
EXTRACTION_REPORT="${DATA_ROOT}/audiocaps_raw_audio.extraction_report.json"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="audiocaps_raw_audio_validation_${STAMP}"
LOG_DIR="${ROOT_DIR}/logs/${RUN_ID}"
LOG_FILE="${LOG_DIR}/combined.log"
REPORT="${LOG_DIR}/audiocaps_audio_validation.json"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2> >(tee -a "${LOG_FILE}" >&2)

echo "EXPERIMENT_NAME=AudioCaps author raw-audio test-split binding and decode gate"
echo "GIT_COMMIT=$(git rev-parse HEAD)"
echo "DATASET=AudioCaps v2 test; expected 975 audio and 4,875 captions"
echo "GPU_USED=no"
echo "OEA_OFFICIAL_SOURCE_USED=no; this is an independent data-integrity gate before OEA model evaluation"
echo "OEA_SOURCE_FILES=none"
echo "ESTIMATED_TOTAL_TIME=5-20 minutes"
echo "DOWNLOADS_REQUIRED=no"
echo "OVERWRITE_DELETE_RISK=none"
echo "FAILURE_RESUME_BEHAVIOR=failed timestamped report is retained; rerun creates a new log directory; existing differing output manifest is never overwritten"
echo "AUDIO_ROOT=${AUDIO_ROOT}"
echo "OUTPUT_MANIFEST=${OUTPUT_MANIFEST}"
echo "REPORT=${REPORT}"
echo "LOG_FILE=${LOG_FILE}"

GIT_STATUS="$(git status --short | sed -E '/^\?\? scripts\/\.__dpc[[:xdigit:]]+$/d')"
if [[ -n "${GIT_STATUS}" ]]; then
  printf '%s\n' "${GIT_STATUS}" >&2
  echo "FINAL_RUN_RC=3"
  echo "COMPLETION_STATUS=failed_dirty_worktree"
  exit 3
fi

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate oea-repro
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

python scripts/validate_audiocaps_raw_audio.py \
  --audio-root "${AUDIO_ROOT}" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --extraction-report "${EXTRACTION_REPORT}" \
  --output-manifest "${OUTPUT_MANIFEST}" \
  --output-report "${REPORT}"
RC=$?
echo "FINAL_RUN_RC=${RC}"
if [[ "${RC}" -eq 0 ]]; then
  echo "COMPLETION_STATUS=complete"
else
  echo "COMPLETION_STATUS=failed"
fi
echo "RESULT_DIRECTORY=${LOG_DIR}"
echo "METRICS_PATH=${REPORT}"
exit "${RC}"
