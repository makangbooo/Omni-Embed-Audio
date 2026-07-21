#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RETRIEVAL_SUITE_CONFIG="${ROOT_DIR}/configs/eval/qwen3b_cl_clotho_a2t_suite.json"
export RETRIEVAL_SUITE_PREFIX="oea_qwen3b_clotho_a2t_suite_seed42"
exec bash "${ROOT_DIR}/scripts/run_qwen3b_clotho_retrieval_suite.sh" "$@"
