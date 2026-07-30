#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${TMUX:-}" ]]; then
  SESSION_NAME="laion_clap_resources_$(date +%Y%m%d_%H%M%S)"
  tmux new-session -d -s "${SESSION_NAME}" \
    "cd \"${ROOT_DIR}\" && exec bash scripts/run_laion_clap_resource_pipeline_tmux.sh"
  echo "TMUX_SESSION=${SESSION_NAME}"
  echo "GPU_USED=no"
  echo "OEA_OFFICIAL_SOURCE_USED=yes"
  echo "ESTIMATED_TOTAL_TIME=5-20 minutes"
  echo "ATTACH_COMMAND=tmux attach -t ${SESSION_NAME}"
  echo "CAPTURE_COMMAND=tmux capture-pane -pt ${SESSION_NAME} -S -120"
  exit 0
fi

# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

MODEL_ROOT="${MODEL_ROOT:-/home/jg525/models/oea}"
MANIFEST="${ROOT_DIR}/configs/resources/model05_laion_clap.json"
REQUIREMENTS="${ROOT_DIR}/configs/resources/laion_clap_1_1_6_overlay.requirements.txt"
OVERLAY_ROOT="${MODEL_ROOT}/python/laion-clap-1.1.6-overlay-v3"
OVERLAY_MARKER="${OVERLAY_ROOT}/.oea_requirements_sha256"
RUN_ID="laion_clap_resource_pipeline_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"
LOG_FILE="${RUN_DIR}/combined.log"
DOWNLOAD_REPORT="${RUN_DIR}/download_manifest.json"
LOCK_OUTPUT="${RUN_DIR}/laion_clap.portable_model_lock.json"

if [[ -e "${RUN_DIR}" ]]; then
  echo "[ERROR] Refusing to reuse run directory: ${RUN_DIR}" >&2
  exit 2
fi
mkdir -p "${RUN_DIR}"
exec > >(tee -a "${LOG_FILE}") 2> >(tee -a "${LOG_FILE}" >&2)

START_TIME="$(date -Is)"
START_EPOCH="$(date +%s)"
GIT_COMMIT="$(git rev-parse HEAD)"
GIT_STATUS="$(git status --short)"

finish() {
  local rc="$1"
  local status="$2"
  local stage="$3"
  local elapsed
  elapsed="$(($(date +%s) - START_EPOCH))"
  printf '%s\n' "${rc}" > "${RUN_DIR}/exit_code.txt"
  echo "FINAL_RUN_RC=${rc}"
  echo "START_TIME=${START_TIME}"
  echo "END_TIME=$(date -Is)"
  echo "TOTAL_ELAPSED_SECONDS=${elapsed}"
  echo "COMPLETION_STATUS=${status}"
  echo "FAILED_STAGE=${stage}"
  echo "ERROR_SUMMARY=$([[ ${rc} -eq 0 ]] && echo none || echo "See combined.log")"
  echo "RESULT_DIRECTORY=${RUN_DIR}"
  echo "METRICS_PATH=${LOCK_OUTPUT}"
  echo "TMUX_RETAINED_SHELL=yes"
  exec bash -i
}

if [[ -n "${GIT_STATUS}" ]]; then
  printf '%s\n' "${GIT_STATUS}" >&2
  finish 3 failed preflight
fi

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate oea-repro
export CUDA_VISIBLE_DEVICES=""

echo "EXPERIMENT_NAME=LAION-CLAP immutable resource and portable-lock preparation"
echo "GIT_COMMIT=${GIT_COMMIT}"
echo "MODEL=LAION-CLAP public-code default; laion-clap 1.1.6; non-fusion 630k-audioset-best.pt"
echo "DATASET=none"
echo "GPU_USED=no"
echo "OEA_OFFICIAL_SOURCE_USED=yes"
echo "ESTIMATED_TOTAL_TIME=5-20 minutes"
echo "TOTAL_WORKLOAD=1.864 GB checkpoint + 3 tokenizer snapshots + isolated Python overlay"
echo "MODEL_ROOT=${MODEL_ROOT}"
echo "HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}"
echo "RESULT_DIRECTORY=${RUN_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "START_TIME=${START_TIME}"

EXPECTED_REQUIREMENTS_SHA="$(sha256sum "${REQUIREMENTS}" | awk '{print $1}')"
if [[ -d "${OVERLAY_ROOT}" ]]; then
  if [[ ! -f "${OVERLAY_MARKER}" ]] || \
     [[ "$(cat "${OVERLAY_MARKER}")" != "${EXPECTED_REQUIREMENTS_SHA}" ]]; then
    echo "[ERROR] Existing package overlay lacks the expected immutable marker." >&2
    finish 4 failed package_overlay
  fi
  echo "STAGE_END=package_overlay RC=0 reused=true"
else
  OVERLAY_ATTEMPT="${MODEL_ROOT}/python/.laion-clap-1.1.6-overlay.${RUN_ID}"
  mkdir -p "$(dirname "${OVERLAY_ATTEMPT}")"
  echo "STAGE_START=package_overlay"
  python -m pip install \
    --no-deps --require-hashes --target "${OVERLAY_ATTEMPT}" \
    --requirement "${REQUIREMENTS}"
  INSTALL_RC=$?
  if [[ "${INSTALL_RC}" -ne 0 ]]; then
    echo "[ERROR] Partial overlay attempt preserved: ${OVERLAY_ATTEMPT}" >&2
    finish "${INSTALL_RC}" failed package_overlay
  fi
  printf '%s\n' "${EXPECTED_REQUIREMENTS_SHA}" > \
    "${OVERLAY_ATTEMPT}/.oea_requirements_sha256"
  mv "${OVERLAY_ATTEMPT}" "${OVERLAY_ROOT}"
  echo "STAGE_END=package_overlay RC=0 reused=false"
fi

echo "STAGE_START=resource_download"
python scripts/download_model_assets.py \
  --manifest "${MANIFEST}" \
  --model-root "${MODEL_ROOT}" \
  --output "${DOWNLOAD_REPORT}" \
  --max-download-attempts 20 \
  --retry-backoff-seconds 30 \
  --max-retry-delay-seconds 600 \
  --max-workers 1
DOWNLOAD_RC=$?
echo "STAGE_END=resource_download RC=${DOWNLOAD_RC}"
if [[ "${DOWNLOAD_RC}" -ne 0 ]]; then
  finish "${DOWNLOAD_RC}" failed resource_download
fi

export PYTHONPATH="${OVERLAY_ROOT}:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "STAGE_START=portable_model_lock"
python scripts/build_laion_clap_portable_lock.py \
  --model-root "${MODEL_ROOT}" \
  --overlay-root "${OVERLAY_ROOT}" \
  --download-report "${DOWNLOAD_REPORT}" \
  --manifest "${MANIFEST}" \
  --requirements "${REQUIREMENTS}" \
  --output "${LOCK_OUTPUT}"
LOCK_RC=$?
echo "STAGE_END=portable_model_lock RC=${LOCK_RC}"
if [[ "${LOCK_RC}" -ne 0 ]]; then
  finish "${LOCK_RC}" failed portable_model_lock
fi

sha256sum "${DOWNLOAD_REPORT}" "${LOCK_OUTPUT}" "${LOG_FILE}" \
  > "${RUN_DIR}/artifact_sha256.txt"
finish 0 complete none
