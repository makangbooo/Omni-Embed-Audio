#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export UIQ_EMBEDDING_CONFIG="${ROOT_DIR}/configs/eval/nemo3b_cl_clotho_positive_uiq_embeddings.json"
export BASE_PROTOCOL_CONFIG="${ROOT_DIR}/configs/eval/nemo3b_cl_clotho_embeddings.json"
export MODEL_LOCK="${ROOT_DIR}/results/model_locks/oea_nemo3b_cl.json"
export UIQ_RUN_PREFIX="oea_nemo3b_clotho_positive_uiq_embeddings_seed42"

exec bash "${ROOT_DIR}/scripts/run_qwen3b_ac_clotho_positive_uiq_embeddings.sh" "$@"
