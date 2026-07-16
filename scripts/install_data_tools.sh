#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ENV_NAME="oea-repro"
CONDA_CHANNEL="${CONDA_CHANNEL_URL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge}"
SEVEN_ZIP_SPEC="7zip=26.02"
RUN_ID="environment_data_tools_$(date +%Y%m%d_%H%M%S)"
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
  printf 'CONDA_CHANNEL_URL=%q ' "${CONDA_CHANNEL}"
  printf '%q ' "$0" "$@"
  printf '\n'
} > "${RUN_DIR}/command.sh"
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git status --short > "${RUN_DIR}/git_status_before.txt"
hostname > "${RUN_DIR}/hostname.txt"
pgrep -af '[c]onda.*(install|create|update)|[p]ython.*-m pip|[p]ip install' \
  > "${RUN_DIR}/package_manager_processes_before.txt" || true
pgrep -af '[d]ownload_model_assets.py|[s]moke_oea|[v]alidate_clotho_evaluation.py|[t]orchrun|[a]ccelerate launch' \
  > "${RUN_DIR}/oea_processes_before.txt" || true

if [[ -s "${RUN_DIR}/package_manager_processes_before.txt" ]]; then
  echo "[ERROR] Another package installation appears to be active in the shared environment." >&2
  cat "${RUN_DIR}/package_manager_processes_before.txt" >&2
  exit 4
fi
if [[ -s "${RUN_DIR}/oea_processes_before.txt" ]]; then
  echo "[ERROR] An OEA model/data job is using the shared environment." >&2
  cat "${RUN_DIR}/oea_processes_before.txt" >&2
  exit 5
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)

echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] Conda environment: ${ENV_NAME}"
echo "[INFO] Conda channel: ${CONDA_CHANNEL}"
echo "[INFO] Package: ${SEVEN_ZIP_SPEC}"

conda list -n "${ENV_NAME}" > "${RUN_DIR}/conda_list_before.txt"

set +e
conda install \
  --name "${ENV_NAME}" \
  --override-channels \
  --channel "${CONDA_CHANNEL}" \
  --freeze-installed \
  "${SEVEN_ZIP_SPEC}" \
  --yes
INSTALL_EXIT_CODE=$?
set -e
printf '%s\n' "${INSTALL_EXIT_CODE}" > "${RUN_DIR}/install_exit_code.txt"
if [[ "${INSTALL_EXIT_CODE}" -ne 0 ]]; then
  echo "[ERROR] Data-tool installation failed; see the preserved Conda output." >&2
  exit "${INSTALL_EXIT_CODE}"
fi

conda activate "${ENV_NAME}"
export CUDA_VISIBLE_DEVICES=""

if ! command -v 7zz >/dev/null 2>&1; then
  echo "[ERROR] 7zip was installed but the expected 7zz executable is unavailable." >&2
  exit 6
fi

7zz i > "${RUN_DIR}/7zz_info.txt" 2>&1
conda list -n "${ENV_NAME}" > "${RUN_DIR}/conda_list_after.txt"
conda list -n "${ENV_NAME}" --explicit > "${RUN_DIR}/conda_explicit_after.txt"
conda env export -n "${ENV_NAME}" --no-builds > "${RUN_DIR}/environment_resolved_after.yml"
git status --short > "${RUN_DIR}/git_status_after.txt"

echo "[INFO] Data-tool installation completed"
echo "[INFO] 7zz: $(command -v 7zz)"
sed -n '1,3p' "${RUN_DIR}/7zz_info.txt"
echo "[INFO] Audit artifacts: ${RUN_DIR}"
