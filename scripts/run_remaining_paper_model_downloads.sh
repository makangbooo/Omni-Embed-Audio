#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

MODEL_ROOT="${MODEL_ROOT:-/home/jg525/models/oea}"
MANIFEST="${ROOT_DIR}/configs/resources/remaining_paper_models.json"
TOOLS_REQUIREMENTS="${ROOT_DIR}/configs/resources/paper_model_download_tools.requirements.txt"
TOOLS_OVERLAY="${MODEL_ROOT}/python/paper-model-download-tools-2"
RUN_ID="paper_model_downloads_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"
LOG_FILE="${RUN_DIR}/combined.log"

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
  echo "ERROR_SUMMARY=$([[ ${rc} -eq 0 ]] && echo 'Robust checkpoint remains unpublished' || echo 'See per-job logs')"
  echo "RESULT_DIRECTORY=${RUN_DIR}"
  echo "METRICS_PATH=${RUN_DIR}/download_summary.txt"
  exit "${rc}"
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
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "${MODEL_ROOT}"
AVAILABLE_KB="$(df -Pk "${MODEL_ROOT}" | awk 'NR==2 {print $4}')"
REQUIRED_KB=$((8 * 1024 * 1024))
if [[ -z "${AVAILABLE_KB}" ]] || (( AVAILABLE_KB < REQUIRED_KB )); then
  echo "[ERROR] At least 8 GiB free space is required below ${MODEL_ROOT}." >&2
  finish 4 failed disk_preflight
fi

echo "EXPERIMENT_NAME=Remaining paper model resource downloads"
echo "GIT_COMMIT=${GIT_COMMIT}"
echo "MODEL=Robust-CLAP source,MGA-CLAP,M2D-CLAP,BGE-large-en-v1.5"
echo "DATASET=none"
echo "GPU_USED=no"
echo "OEA_OFFICIAL_SOURCE_USED=yes"
echo "PARALLEL_DOWNLOADS=6"
echo "ESTIMATED_TOTAL_TIME=10-60 minutes"
echo "MODEL_ROOT=${MODEL_ROOT}"
echo "HF_ENDPOINT=${HF_ENDPOINT}"
echo "MANIFEST=${MANIFEST}"
echo "RESULT_DIRECTORY=${RUN_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "START_TIME=${START_TIME}"
echo "ROBUST_CHECKPOINT_STATUS=blocked_not_published"

install_download_tools() {
  local expected_sha marker attempt
  marker="${TOOLS_OVERLAY}/.oea_requirements_sha256"
  expected_sha="$(sha256sum "${TOOLS_REQUIREMENTS}" | awk '{print $1}')"
  if [[ -f "${marker}" ]] && [[ "$(cat "${marker}")" == "${expected_sha}" ]]; then
    echo "DOWNLOAD_TOOLS reused=true"
    return 0
  fi
  if [[ -e "${TOOLS_OVERLAY}" ]]; then
    echo "[ERROR] Existing download-tools overlay has an unexpected identity: ${TOOLS_OVERLAY}" >&2
    return 2
  fi
  attempt="${MODEL_ROOT}/python/.paper-model-download-tools.${RUN_ID}"
  mkdir -p "$(dirname "${attempt}")"
  python -m pip install --no-deps --require-hashes --target "${attempt}" \
    --requirement "${TOOLS_REQUIREMENTS}" || return $?
  printf '%s\n' "${expected_sha}" > "${attempt}/.oea_requirements_sha256"
  mv "${attempt}" "${TOOLS_OVERLAY}"
  echo "DOWNLOAD_TOOLS reused=false"
}

download_source() {
  local name="$1" repository="$2" revision="$3" destination="$4"
  local marker archive attempt
  marker="${destination}/.source_revision"
  if [[ -f "${marker}" ]] && [[ "$(cat "${marker}")" == "${revision}" ]]; then
    echo "SOURCE model=${name} status=reused revision=${revision} path=${destination}"
    return 0
  fi
  if [[ -e "${destination}" ]]; then
    echo "[ERROR] Existing source has no matching revision marker: ${destination}" >&2
    return 2
  fi
  archive="${MODEL_ROOT}/downloads/${name,,}-${revision}.tar.gz"
  attempt="${destination}.attempt-${RUN_ID}"
  mkdir -p "$(dirname "${archive}")" "${attempt}"
  curl --fail --location --retry 10 --retry-all-errors --continue-at - \
    --output "${archive}" \
    "https://codeload.github.com/${repository}/tar.gz/${revision}" || return $?
  tar -xzf "${archive}" --strip-components=1 --directory "${attempt}" || return $?
  printf '%s\n' "${revision}" > "${attempt}/.source_revision"
  mv "${attempt}" "${destination}"
  echo "SOURCE model=${name} status=complete revision=${revision} path=${destination}"
}

download_mga_checkpoint() {
  local destination attempt
  destination="${MODEL_ROOT}/mga-clap/pretrained_models/models/model.pt"
  if [[ -s "${destination}" ]]; then
    echo "CHECKPOINT model=MGA-CLAP status=reused path=${destination}"
    sha256sum "${destination}"
    return 0
  fi
  attempt="${destination}.attempt-${RUN_ID}"
  mkdir -p "$(dirname "${destination}")"
  PYTHONPATH="${TOOLS_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}" \
    python -m gdown 1RWTuVMEPy-L0uK6WYIX2wwxHjD1YSQFz \
    --output "${attempt}" || return $?
  [[ -s "${attempt}" ]] || return 3
  mv "${attempt}" "${destination}"
  echo "CHECKPOINT model=MGA-CLAP status=complete path=${destination}"
  sha256sum "${destination}"
}

download_m2d_checkpoint() {
  local url archive destination checkpoint actual_sha attempt
  url="https://github.com/nttcslab/m2d/releases/download/v0.5.0/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025.zip"
  archive="${MODEL_ROOT}/downloads/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025.zip"
  destination="${MODEL_ROOT}/m2d-clap/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025"
  checkpoint="${destination}/checkpoint-30.pth"
  if [[ -s "${checkpoint}" ]]; then
    echo "CHECKPOINT model=M2D-CLAP status=reused path=${checkpoint}"
    sha256sum "${checkpoint}"
    return 0
  fi
  if [[ -e "${destination}" ]]; then
    echo "[ERROR] Partial M2D destination exists: ${destination}" >&2
    return 2
  fi
  mkdir -p "$(dirname "${archive}")" "$(dirname "${destination}")"
  curl --fail --location --retry 20 --retry-all-errors --continue-at - \
    --output "${archive}" "${url}" || return $?
  actual_sha="$(sha256sum "${archive}" | awk '{print $1}')"
  if [[ "${actual_sha}" != "fd193ae591720df7f1e27ed728ce127e0309b8bd427f0f4b3e5cd17d7ee5e1e1" ]]; then
    echo "[ERROR] M2D archive SHA256 mismatch: ${actual_sha}" >&2
    return 4
  fi
  attempt="${MODEL_ROOT}/m2d-clap/.m2d-extract-${RUN_ID}"
  mkdir -p "${attempt}"
  python - "${archive}" "${attempt}" <<'PY'
import sys
import zipfile
from pathlib import Path

archive, destination = map(Path, sys.argv[1:])
with zipfile.ZipFile(archive) as source:
    source.extractall(destination)
PY
  local extracted
  extracted="${attempt}/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025"
  [[ -s "${extracted}/checkpoint-30.pth" ]] || return 5
  mv "${extracted}" "${destination}"
  echo "CHECKPOINT model=M2D-CLAP status=complete path=${checkpoint}"
  sha256sum "${checkpoint}"
}

download_bge() {
  local destination marker revision
  destination="${MODEL_ROOT}/bge-large-en-v1.5"
  marker="${destination}/.source_revision"
  revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09"
  if [[ -f "${marker}" ]] && [[ "$(cat "${marker}")" == "${revision}" ]]; then
    echo "SNAPSHOT model=BGE-large-en-v1.5 status=reused revision=${revision} path=${destination}"
    return 0
  fi
  mkdir -p "${destination}"
  python - "${destination}" "${revision}" <<'PY'
import sys
from huggingface_hub import snapshot_download

destination, revision = sys.argv[1:]
snapshot_download(
    repo_id="BAAI/bge-large-en-v1.5",
    revision=revision,
    local_dir=destination,
    allow_patterns=[
        "1_Pooling/config.json",
        "config.json",
        "config_sentence_transformers.json",
        "model.safetensors",
        "modules.json",
        "sentence_bert_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    ],
    max_workers=4,
)
PY
  printf '%s\n' "${revision}" > "${marker}"
  echo "SNAPSHOT model=BGE-large-en-v1.5 status=complete revision=${revision} path=${destination}"
}

echo "STAGE_START=download_tools"
install_download_tools
TOOLS_RC=$?
echo "STAGE_END=download_tools RC=${TOOLS_RC}"
if [[ "${TOOLS_RC}" -ne 0 ]]; then
  finish "${TOOLS_RC}" failed download_tools
fi

declare -A PIDS=()
start_job() {
  local name="$1"
  shift
  (
    set +e
    "$@" 2>&1 | sed -u "s/^/[${name}] /" | tee -a "${RUN_DIR}/${name}.log"
    rc="${PIPESTATUS[0]}"
    printf '%s\n' "${rc}" > "${RUN_DIR}/${name}.exit_code.txt"
    exit "${rc}"
  ) &
  PIDS["${name}"]=$!
}

echo "STAGE_START=parallel_resource_downloads"
start_job robust_source download_source \
  Robust-CLAP ramaneswaran/linguistic_robust_clap \
  d08d0e3c545fa22df0930fc0d090741aaa9e2cc1 \
  "${MODEL_ROOT}/robust-clap/source"
start_job mga_source download_source \
  MGA-CLAP Ming-er/MGA-CLAP \
  48ca5a5cd22cd34427e118bd8cf332090ec54770 \
  "${MODEL_ROOT}/mga-clap/source"
start_job mga_checkpoint download_mga_checkpoint
start_job m2d_source download_source \
  M2D-CLAP nttcslab/m2d \
  3d0c4de9447c404a8d3f9f37e04f53bc902e09b3 \
  "${MODEL_ROOT}/m2d-clap/source"
start_job m2d_checkpoint download_m2d_checkpoint
start_job bge_snapshot download_bge

OVERALL_RC=0
for name in robust_source mga_source mga_checkpoint m2d_source m2d_checkpoint bge_snapshot; do
  if wait "${PIDS[${name}]}"; then
    echo "JOB_END=${name} RC=0"
  else
    rc=$?
    echo "JOB_END=${name} RC=${rc}"
    OVERALL_RC=1
  fi
done
echo "STAGE_END=parallel_resource_downloads RC=${OVERALL_RC}"

{
  echo "ROBUST_CHECKPOINT_STATUS=blocked_not_published"
  for name in robust_source mga_source mga_checkpoint m2d_source m2d_checkpoint bge_snapshot; do
    echo "${name}=$(cat "${RUN_DIR}/${name}.exit_code.txt")"
  done
  find "${MODEL_ROOT}/robust-clap" "${MODEL_ROOT}/mga-clap" \
    "${MODEL_ROOT}/m2d-clap" "${MODEL_ROOT}/bge-large-en-v1.5" \
    -maxdepth 4 -type f -printf '%s %p\n' 2>/dev/null || true
} | tee "${RUN_DIR}/download_summary.txt"

sha256sum "${MANIFEST}" "${LOG_FILE}" > "${RUN_DIR}/artifact_sha256.txt"
if [[ "${OVERALL_RC}" -ne 0 ]]; then
  finish 1 failed parallel_resource_downloads
fi
finish 0 complete_with_known_blocker none
