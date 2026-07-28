#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export RETRIEVAL_SUITE_CONFIG="${ROOT_DIR}/configs/eval/nemo3b_cl_clotho_t2t_suite.json"
export RETRIEVAL_SUITE_PREFIX="oea_nemo3b_clotho_t2t_suite_seed42"

# The underlying runner is model-agnostic. The tracked config and model lock
# bind this invocation to the Nemo3B-Cl checkpoint and the two missing T2T
# protocols. Completed T2A protocols are intentionally not repeated.
exec bash "${ROOT_DIR}/scripts/run_qwen3b_clotho_retrieval_suite.sh" "$@"
