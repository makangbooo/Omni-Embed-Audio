#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ENV_NAME="oea-repro"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu126"
RUN_ID="environment_setup_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"

mkdir -p "${RUN_DIR}"

{
  printf '%q ' "$0" "$@"
  printf '\n'
} > "${RUN_DIR}/command.sh"
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git status --short > "${RUN_DIR}/git_status.txt"
nvidia-smi > "${RUN_DIR}/gpu_info.txt" 2>&1 || true

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda is not available." | tee "${RUN_DIR}/stderr.log" >&2
  exit 2
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "[ERROR] Conda environment '${ENV_NAME}' already exists." | tee "${RUN_DIR}/stderr.log" >&2
  echo "[ERROR] Refusing to modify or overwrite it; inspect it first." | tee -a "${RUN_DIR}/stderr.log" >&2
  exit 3
fi

exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)

echo "[INFO] Run directory: ${RUN_DIR}"
echo "[INFO] Creating ${ENV_NAME} from environment.yml"
conda env create --file environment.yml
conda activate "${ENV_NAME}"

python --version
python -m pip --version

echo "[INFO] Installing the official PyTorch 2.7.1 CUDA 12.6 wheels"
python -m pip install \
  torch==2.7.1 \
  torchvision==0.22.1 \
  torchaudio==2.7.1 \
  --index-url "${PYTORCH_INDEX}"

echo "[INFO] Installing the audited direct dependency candidates"
python -m pip install --requirement requirements-lock.txt

echo "[INFO] Checking dependency consistency"
python -m pip check

echo "[INFO] Running the no-model-download environment check"
python scripts/check_environment.py \
  --output "${RUN_DIR}/environment_check.json"

python -m pip freeze --all > "${RUN_DIR}/requirements-freeze.txt"
conda env export --no-builds > "${RUN_DIR}/environment-resolved.yml"
conda list --explicit > "${RUN_DIR}/conda-explicit.txt"

echo "[INFO] Environment setup completed"
echo "[INFO] Audit artifacts: ${RUN_DIR}"
