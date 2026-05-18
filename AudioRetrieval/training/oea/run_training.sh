#!/bin/bash
# Wrapper to run OEA training with clean Python environment

unset PYTHONPATH
unset BNB_CUDA_VERSION
export PYTHONNOUSERSITE=""

export PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    export PYTHON_BIN="$(command -v python3)"
  else
    echo "[ERROR] Could not find python interpreter." >&2
    exit 1
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${PYTHON_BIN}" "${SCRIPT_DIR}/train_omniembed_lora.py" "$@"

