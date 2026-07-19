#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <completed-qwen3b-ac-embedding-directory>" >&2
  exit 2
fi

export RETRIEVAL_SUITE_CONFIG="${ROOT_DIR}/configs/eval/qwen3b_clotho_retrieval_suite.json"
export RETRIEVAL_SUITE_PREFIX="oea_qwen3b_ac_clotho_retrieval_suite_seed42"

exec bash "${ROOT_DIR}/scripts/run_qwen3b_clotho_retrieval_suite.sh" "$1"
