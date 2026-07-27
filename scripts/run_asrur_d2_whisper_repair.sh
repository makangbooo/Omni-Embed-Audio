#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODELS_ROOT="${MODELS_ROOT:-/home/jg525/models}"
MODEL_DIR="${MODELS_ROOT}/whisper-large-v3"
REVISION="06f233fe06e710322aca913c1bc4249a0d71fce1"
EXPECTED_SIZE="3087130976"
EXPECTED_SHA256="a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
SOURCE_URL="${HF_ENDPOINT%/}/openai/whisper-large-v3/resolve/${REVISION}/model.safetensors?download=true"
FINAL_FILE="${MODEL_DIR}/model.safetensors"
PART_FILE="${FINAL_FILE}.part"
RUN_ID="${RUN_ID:-asrur_d2_whisper_repair_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"
mkdir -p "${RUN_DIR}"

record_wrapper_exit() {
  local exit_code=$?
  printf '%s\n' "${exit_code}" > "${RUN_DIR}/wrapper_exit_code.txt"
}
record_signal() {
  local signal_name=$1
  local exit_code=$2
  printf '%s\n' "${signal_name}" > "${RUN_DIR}/termination_signal.txt"
  exit "${exit_code}"
}
trap record_wrapper_exit EXIT
trap 'record_signal SIGHUP 129' HUP
trap 'record_signal SIGINT 130' INT
trap 'record_signal SIGTERM 143' TERM

{
  printf 'MODELS_ROOT=%q ' "${MODELS_ROOT}"
  printf 'HF_ENDPOINT=%q ' "${HF_ENDPOINT}"
  printf '%q\n' "$0" "$@"
} > "${RUN_DIR}/command.sh"
printf '%s\n' "${SOURCE_URL}" > "${RUN_DIR}/source_url.txt"
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git status --short --untracked-files=all > "${RUN_DIR}/git_status.txt"
hostname > "${RUN_DIR}/hostname.txt"
df -hT "${MODELS_ROOT}" > "${RUN_DIR}/disk_before.txt" 2>&1 || true

exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)
export CUDA_VISIBLE_DEVICES=""

echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] GPU disabled"
echo "[INFO] Only D2 model.safetensors is in scope"
echo "[INFO] D3 and D4 are not read by the downloader"
echo "[INFO] Partial file is retained for resume after interruption"

if [[ -s "${RUN_DIR}/git_status.txt" ]]; then
  echo "[ERROR] D2 repair requires a clean project Git worktree." >&2
  cat "${RUN_DIR}/git_status.txt" >&2
  exit 2
fi
if [[ ! -d "${MODEL_DIR}/.git" ]]; then
  echo "[ERROR] Missing D2 Git checkout: ${MODEL_DIR}" >&2
  exit 3
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "[ERROR] curl is required." >&2
  exit 3
fi
if ! command -v sha256sum >/dev/null 2>&1; then
  echo "[ERROR] sha256sum is required." >&2
  exit 3
fi

ACTUAL_REVISION="$(git -C "${MODEL_DIR}" rev-parse HEAD)"
printf '%s\n' "${ACTUAL_REVISION}" > "${RUN_DIR}/model_git_commit.txt"
if [[ "${ACTUAL_REVISION}" != "${REVISION}" ]]; then
  echo "[ERROR] D2 revision mismatch: expected ${REVISION}, got ${ACTUAL_REVISION}" >&2
  exit 4
fi

verify_file() {
  local path=$1
  local size
  local digest
  size="$(stat -c '%s' "${path}")"
  if [[ "${size}" != "${EXPECTED_SIZE}" ]]; then
    echo "[ERROR] D2 weight size mismatch: expected ${EXPECTED_SIZE}, got ${size}" >&2
    return 1
  fi
  digest="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${digest}" != "${EXPECTED_SHA256}" ]]; then
    echo "[ERROR] D2 weight SHA256 mismatch: expected ${EXPECTED_SHA256}, got ${digest}" >&2
    return 1
  fi
}

if [[ -e "${FINAL_FILE}" ]]; then
  if [[ ! -f "${FINAL_FILE}" || -L "${FINAL_FILE}" ]]; then
    echo "[ERROR] Existing D2 final path is not a regular non-symlink file." >&2
    exit 5
  fi
  verify_file "${FINAL_FILE}"
  echo "[INFO] Existing D2 weight already matches; download skipped"
else
  if [[ -e "${PART_FILE}" && ( ! -f "${PART_FILE}" || -L "${PART_FILE}" ) ]]; then
    echo "[ERROR] Existing partial path is not a regular non-symlink file." >&2
    exit 5
  fi
  echo "[INFO] Downloading/resuming D2 from the pinned immutable revision"
  echo "[INFO] Source: ${SOURCE_URL}"
  curl \
    --fail \
    --location \
    --http1.1 \
    --continue-at - \
    --output "${PART_FILE}" \
    --retry 50 \
    --retry-all-errors \
    --retry-delay 15 \
    --connect-timeout 60 \
    --max-time 0 \
    "${SOURCE_URL}"

  verify_file "${PART_FILE}"
  if [[ -e "${FINAL_FILE}" ]]; then
    echo "[ERROR] Final file appeared during download; refusing to overwrite it." >&2
    exit 6
  fi
  mv -- "${PART_FILE}" "${FINAL_FILE}"
  echo "[INFO] Verified D2 weight moved atomically into place"
fi

stat -c '%s %n' "${FINAL_FILE}" > "${RUN_DIR}/weight_size.txt"
sha256sum "${FINAL_FILE}" > "${RUN_DIR}/weight_sha256.txt"
df -hT "${MODELS_ROOT}" > "${RUN_DIR}/disk_after.txt" 2>&1 || true

echo "[INFO] Starting the existing D2-D4 strict offline audit"
set +e
MODELS_ROOT="${MODELS_ROOT}" bash scripts/run_asrur_model_audit.sh
AUDIT_EXIT_CODE=$?
set -e
printf '%s\n' "${AUDIT_EXIT_CODE}" > "${RUN_DIR}/strict_audit_exit_code.txt"
if [[ "${AUDIT_EXIT_CODE}" -ne 0 ]]; then
  echo "[ERROR] D2 download passed, but the full D2-D4 audit failed." >&2
  exit "${AUDIT_EXIT_CODE}"
fi

echo "[INFO] D2 repair and D2-D4 strict offline audit completed"
echo "[INFO] Repair artifacts: ${RUN_DIR}"
