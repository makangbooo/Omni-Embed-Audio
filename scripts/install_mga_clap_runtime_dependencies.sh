#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

MODEL_ROOT="${MODEL_ROOT:-/home/jg525/models/oea}"
REQUIREMENTS="${ROOT_DIR}/configs/resources/mga_clap_runtime_overlay.requirements.txt"
OVERLAY_ROOT="${MODEL_ROOT}/python/mga-clap-runtime-v1"
OVERLAY_MARKER="${OVERLAY_ROOT}/.oea_requirements_sha256"
RUN_ID="mga_clap_runtime_dependencies_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/logs/${RUN_ID}"
LOG_FILE="${LOG_DIR}/combined.log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2> >(tee -a "${LOG_FILE}" >&2)

echo "EXPERIMENT_NAME=MGA-CLAP runtime dependency installation"
echo "MODEL=MGA-CLAP official source"
echo "GPU_USED=no"
echo "OEA_OFFICIAL_SOURCE_USED=yes"
echo "DEPENDENCY=ruamel.yaml==0.18.10"
echo "OVERLAY_ROOT=${OVERLAY_ROOT}"
echo "LOG_FILE=${LOG_FILE}"

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate oea-repro

EXPECTED_MARKER="$(sha256sum "${REQUIREMENTS}" | awk '{print $1}')"
if [[ -d "${OVERLAY_ROOT}" ]]; then
  if [[ ! -f "${OVERLAY_MARKER}" ]] || \
     [[ "$(<"${OVERLAY_MARKER}")" != "${EXPECTED_MARKER}" ]]; then
    echo "[ERROR] Existing MGA runtime overlay has the wrong marker." >&2
    echo "FINAL_RUN_RC=2"
    echo "COMPLETION_STATUS=failed_overlay_mismatch"
    exit 2
  fi
  echo "INSTALL_STATUS=reused"
else
  ATTEMPT="${MODEL_ROOT}/python/.mga-clap-runtime-v1.${RUN_ID}"
  mkdir -p "$(dirname "${ATTEMPT}")"
  python -m pip install \
    --no-deps --require-hashes --target "${ATTEMPT}" \
    --requirement "${REQUIREMENTS}"
  RC=$?
  if [[ "${RC}" -ne 0 ]]; then
    echo "[ERROR] Partial install preserved: ${ATTEMPT}" >&2
    echo "FINAL_RUN_RC=${RC}"
    echo "COMPLETION_STATUS=failed_install"
    exit "${RC}"
  fi
  printf '%s\n' "${EXPECTED_MARKER}" > "${ATTEMPT}/.oea_requirements_sha256"
  mv "${ATTEMPT}" "${OVERLAY_ROOT}"
  echo "INSTALL_STATUS=complete"
fi

PYTHONPATH="${OVERLAY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  python -c 'import ruamel.yaml; print("RUAMEL_YAML_IMPORT=complete")'
RC=$?
if [[ "${RC}" -ne 0 ]]; then
  echo "FINAL_RUN_RC=${RC}"
  echo "COMPLETION_STATUS=failed_import"
  exit "${RC}"
fi

echo "FINAL_RUN_RC=0"
echo "COMPLETION_STATUS=complete"
echo "RESULT_DIRECTORY=${LOG_DIR}"
