#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

ENV_NAME="oea-repro"
RUN_ID="clap_resource_probe_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"
OUTPUT="${RUN_DIR}/clap_resource_probe.json"

if [[ -e "${RUN_DIR}" ]]; then
  echo "[ERROR] Refusing to reuse probe directory: ${RUN_DIR}" >&2
  exit 2
fi
if [[ -n "$(git status --short)" ]]; then
  echo "[ERROR] CLAP resource probe requires a clean Git worktree." >&2
  git status --short >&2
  exit 2
fi
mkdir "${RUN_DIR}"

record_wrapper_exit() {
  local exit_code=$?
  printf '%s\n' "${exit_code}" > "${RUN_DIR}/exit_code.txt"
}
trap record_wrapper_exit EXIT

git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git status --short > "${RUN_DIR}/git_status.txt"
printf '%q\n' "$0" > "${RUN_DIR}/command.sh"
exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES=""

echo "EXPERIMENT_NAME=CLAP local resource readiness probe"
echo "GIT_COMMIT=$(cat "${RUN_DIR}/git_commit.txt")"
echo "MODEL=LAION-CLAP,Robust-CLAP,MGA-CLAP,M2D-CLAP"
echo "DATASET=none"
echo "GPU_USED=no"
echo "OEA_OFFICIAL_SOURCE_USED=no"
echo "ESTIMATED_TOTAL_TIME=1-5 minutes"
echo "OPERATION=local stat, nested Git metadata, package metadata, and checkpoint SHA256 only"
echo "RESULT_DIRECTORY=${RUN_DIR}"
echo "LOG_FILE=${RUN_DIR}/stdout.log"

set +e
python scripts/probe_clap_resources.py --output "${OUTPUT}"
PROBE_RC=$?
set -e

echo "FINAL_RUN_RC=${PROBE_RC}"
echo "COMPLETION_STATUS=$([[ ${PROBE_RC} -eq 0 ]] && echo complete || echo failed)"
echo "METRICS_PATH=${OUTPUT}"
echo "RESULT_DIRECTORY=${RUN_DIR}"
exit "${PROBE_RC}"
