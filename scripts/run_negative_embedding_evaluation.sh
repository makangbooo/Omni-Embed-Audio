#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 <config.json> <output-directory>" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$1"
OUTPUT_DIR="$2"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "[ERROR] Output directory is not empty: ${OUTPUT_DIR}" >&2
  exit 3
fi
mkdir -p "${OUTPUT_DIR}"
{
  printf '%q ' "$0" "$@"
  printf '\n'
} > "${OUTPUT_DIR}/command.sh"
git rev-parse HEAD > "${OUTPUT_DIR}/git_commit.txt"
git status --short > "${OUTPUT_DIR}/git_status.txt"
{
  printf 'CUDA_VISIBLE_DEVICES=disabled\n'
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version \
      --format=csv,noheader || true
  else
    printf 'nvidia-smi=unavailable\n'
  fi
} > "${OUTPUT_DIR}/gpu_info.txt" 2>&1

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate oea-repro
export CUDA_VISIBLE_DEVICES=""

{
  printf 'timestamp=%s\n' "$(date -Is)"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'conda_prefix=%s\n' "${CONDA_PREFIX:-}"
  python --version
  python -c 'import numpy; print(f"numpy={numpy.__version__}")'
} > "${OUTPUT_DIR}/environment.txt" 2>&1

set +e
python scripts/evaluate_negative_embedding_artifacts.py \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  > >(tee "${OUTPUT_DIR}/stdout.log") \
  2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)
EXIT_CODE=$?
set -e
printf '%s\n' "${EXIT_CODE}" > "${OUTPUT_DIR}/exit_code.txt"
exit "${EXIT_CODE}"
