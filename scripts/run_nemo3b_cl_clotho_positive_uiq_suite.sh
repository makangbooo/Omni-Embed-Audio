#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export POSITIVE_UIQ_SUITE_CONFIG="${ROOT_DIR}/configs/eval/nemo3b_cl_clotho_positive_uiq_suite.json"
export POSITIVE_UIQ_SUITE_PREFIX="oea_nemo3b_clotho_positive_uiq_suite_seed42"

exec bash "${ROOT_DIR}/scripts/run_qwen3b_ac_clotho_positive_uiq_suite.sh" "$@"
