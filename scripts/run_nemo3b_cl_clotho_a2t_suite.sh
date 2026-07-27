#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <completed-nemo3b-cl-embedding-directory>" >&2
  exit 2
fi

export RETRIEVAL_SUITE_CONFIG="${ROOT_DIR}/configs/eval/nemo3b_cl_clotho_a2t_suite.json"
export RETRIEVAL_SUITE_PREFIX="oea_nemo3b_clotho_a2t_suite_seed42"

# The underlying runner is model-agnostic despite its historical filename.
# The tracked suite config and canonical model lock provide the Nemo binding.
exec bash "${ROOT_DIR}/scripts/run_qwen3b_clotho_retrieval_suite.sh" "$1"
