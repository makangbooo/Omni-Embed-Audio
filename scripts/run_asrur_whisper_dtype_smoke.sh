#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || "$1" != "--execute" ]]; then
  echo "Usage: $0 --execute" >&2
  echo "Requires the separately approved one-RTX-4090 Phase-2 repair smoke." >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"

DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
MODELS_ROOT="${MODELS_ROOT:-/home/jg525/models}"
SOURCE_MANIFEST="${DATA_ROOT}/squtr/manifests/squtr_en_fiqa_nq_audio_query_manifest.jsonl"
MAIN_CONFIG="${ROOT_DIR}/configs/asr_uncertainty_reranking/main_experiment.json"
MODEL_RESOURCE_MANIFEST="${ROOT_DIR}/configs/asr_uncertainty_reranking/resources/models.json"
RUN_ID="asrur_whisper_dtype_smoke_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/logs/${RUN_ID}"
SMOKE_INPUT="${RUN_DIR}/fiqa_clean_one_record.jsonl"
SMOKE_CACHE="${RUN_DIR}/whisper_cache"
START_EPOCH="$(date +%s)"

if [[ -e "${RUN_DIR}" ]]; then
  echo "[ERROR] Refusing to reuse run directory: ${RUN_DIR}" >&2
  exit 2
fi
mkdir -p "${RUN_DIR}"

record_exit() {
  local exit_code=$?
  local finished_epoch
  finished_epoch="$(date +%s)"
  printf '%s\n' "${exit_code}" > "${RUN_DIR}/wrapper_exit_code.txt"
  printf '%s\n' "$((finished_epoch - START_EPOCH))" > "${RUN_DIR}/elapsed_seconds.txt"
}
trap record_exit EXIT

exec > >(tee "${RUN_DIR}/stdout.log") 2> >(tee "${RUN_DIR}/stderr.log" >&2)

mark_stage() {
  local stage="$1"
  printf '%s\n' "${stage}" > "${RUN_DIR}/current_stage.txt"
  printf '%s\t%s\n' "$(date -Is)" "${stage}" >> "${RUN_DIR}/stage_timeline.tsv"
  echo "[STAGE] ${stage}"
}

mark_stage preflight
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "[ERROR] Formal smoke requires a clean Git worktree." >&2
  git status --short --untracked-files=all >&2
  exit 3
fi
for path in "${SOURCE_MANIFEST}" "${MAIN_CONFIG}" "${MODEL_RESOURCE_MANIFEST}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Missing prerequisite: ${path}" >&2
    exit 4
  fi
done

git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git status --short --untracked-files=all > "${RUN_DIR}/git_status.txt"
hostname > "${RUN_DIR}/hostname.txt"
date -Is > "${RUN_DIR}/started_at.txt"

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate oea-repro

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python scripts/validate_single_bf16_gpu.py \
  --output "${RUN_DIR}/gpu_preflight.json"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
if [[ "${GPU_NAME}" != *"RTX 4090"* ]]; then
  echo "[ERROR] Approved repair-smoke hardware is one RTX 4090; observed: ${GPU_NAME}" >&2
  exit 5
fi
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version \
  --format=csv > "${RUN_DIR}/gpu_info.txt"

mark_stage strict_model_asset_audit
python scripts/verify_asrur_model_assets.py \
  --manifest "${MODEL_RESOURCE_MANIFEST}" \
  --model-root "${MODELS_ROOT}" \
  --output "${RUN_DIR}/d2_d4_model_audit.json"

mark_stage select_one_fiqa_clean_record
python - "${SOURCE_MANIFEST}" "${SMOKE_INPUT}" <<'PY'
import json
import sys
from pathlib import Path

from AudioRetrieval.asr_uncertainty_reranking.data import squtr_subset_name

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
selected = None
with source.open("r", encoding="utf-8") as stream:
    for line_number, line in enumerate(stream, start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            squtr_subset_name(str(row.get("subset", ""))) == "fiqa"
            and row.get("condition") == "clean"
        ):
            selected = row
            break
if selected is None:
    raise RuntimeError("no SQuTR-FiQA clean record is available for smoke")
audio_path = Path(str(selected["audio_path"]))
if not audio_path.is_file():
    raise FileNotFoundError(audio_path)
destination.write_text(
    json.dumps(selected, ensure_ascii=False, allow_nan=False, sort_keys=True)
    + "\n",
    encoding="utf-8",
)
print(f"[INFO] Selected smoke record {selected['record_id']}")
print(f"[INFO] Audio path {audio_path}")
PY

mark_stage four_beam_generation_and_teacher_forced_scoring
python scripts/generate_asrur_frozen_caches.py whisper \
  --config "${MAIN_CONFIG}" \
  --output-dir "${SMOKE_CACHE}" \
  --dataset SQuTR-FiQA \
  --split test_clean_dtype_smoke \
  --device cuda:0 \
  --dtype bfloat16 \
  --input "${SMOKE_INPUT}" \
  --subset fiqa \
  --conditions clean

mark_stage nbest_artifact_validation
python - \
  "${SMOKE_CACHE}/nbest.jsonl" \
  "${SMOKE_CACHE}/run_identity.json" \
  "${RUN_DIR}/smoke_audit.json" <<'PY'
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

source = Path(sys.argv[1])
identity_path = Path(sys.argv[2])
destination = Path(sys.argv[3])
rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
identity = json.loads(identity_path.read_text(encoding="utf-8"))
if len(rows) != 1:
    raise RuntimeError(f"expected one Whisper row, got {len(rows)}")
decode = identity.get("decode", {})
expected_score_method = (
    "teacher_forced_conditional_logprob_float32_cross_entropy"
)
if decode.get("token_score_method") != expected_score_method:
    raise RuntimeError("Whisper cache identity has the wrong token-score method")
if decode.get("beam_transition_scores_used") is not False:
    raise RuntimeError("Whisper cache identity did not disable beam transition scores")
hypotheses = rows[0].get("hypotheses")
if not isinstance(hypotheses, list) or len(hypotheses) != 4:
    raise RuntimeError("Whisper smoke did not produce exactly four hypotheses")
if [value.get("rank") for value in hypotheses] != [1, 2, 3, 4]:
    raise RuntimeError("Whisper hypothesis ranks are not 1..4")
for index, value in enumerate(hypotheses):
    score = value.get("average_token_logprob")
    if not isinstance(score, (int, float)) or not math.isfinite(score):
        raise RuntimeError(f"hypothesis {index} has invalid proxy score")
    sequence_score = value.get("sequence_score")
    if (
        not isinstance(sequence_score, (int, float))
        or not math.isfinite(sequence_score)
    ):
        raise RuntimeError(f"hypothesis {index} has invalid beam sequence score")
    if not isinstance(value.get("valid_token_count"), int) or value["valid_token_count"] <= 0:
        raise RuntimeError(f"hypothesis {index} has no valid generated token")
payload = {
    "schema_version": 1,
    "status": "complete",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "record_id": rows[0]["record_id"],
    "hypothesis_count": 4,
    "proxy_scores_finite": True,
    "beam_sequence_scores_finite": True,
    "token_score_method": expected_score_method,
    "beam_transition_scores_used": False,
    "nbest_path": str(source.resolve()),
    "nbest_size_bytes": source.stat().st_size,
    "nbest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
}
destination.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

mark_stage complete
nvidia-smi > "${RUN_DIR}/gpu_final.txt"
echo "[INFO] Whisper dtype repair smoke completed"
echo "[INFO] Run directory: ${RUN_DIR}"
