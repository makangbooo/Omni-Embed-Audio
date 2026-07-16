#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ENV_NAME="oea-repro"
PYTORCH_WHEELHOUSE="${PYTORCH_WHEELHOUSE_URL:-https://mirrors.aliyun.com/pytorch-wheels/cu126}"
PYPI_INDEX="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
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

if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fx "${ENV_NAME}" >/dev/null; then
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
echo "[INFO] PyTorch wheelhouse: ${PYTORCH_WHEELHOUSE}"
echo "[INFO] Dependency index: ${PYPI_INDEX}"
python -m pip install \
  'torch==2.7.1+cu126' \
  'torchvision==0.22.1+cu126' \
  'torchaudio==2.7.1+cu126' \
  --find-links "${PYTORCH_WHEELHOUSE}" \
  --index-url "${PYPI_INDEX}"

echo "[INFO] Installing the audited direct dependency candidates"
echo "[INFO] PyPI index: ${PYPI_INDEX}"
python -m pip install \
  --index-url "${PYPI_INDEX}" \
  --requirement requirements-lock.txt

echo "[INFO] Checking dependency consistency"
python -m pip check

echo "[INFO] Running the CPU-safe no-model-download environment check"
python scripts/check_environment.py \
  --mode cpu \
  --output "${RUN_DIR}/environment_check.json"

python -m pip freeze --all > "${RUN_DIR}/requirements-freeze.txt"
conda env export --no-builds > "${RUN_DIR}/environment-resolved.yml"
conda list --explicit > "${RUN_DIR}/conda-explicit.txt"

echo "[INFO] Environment setup completed"
echo "[INFO] Audit artifacts: ${RUN_DIR}"
