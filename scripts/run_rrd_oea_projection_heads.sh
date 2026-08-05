#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 || "$1" != "--execute" ]]; then
  echo "Usage: bash scripts/run_rrd_oea_projection_heads.sh --execute <oea_nemo3b_cl|oea_qwen3b_cl>" >&2
  exit 2
fi
if [[ -z "${TMUX:-}" ]]; then
  echo "[ERROR] Run inside a normal tmux session: tmux new -s rrd_oea" >&2
  exit 3
fi

VARIANT="$2"
case "${VARIANT}" in
  oea_nemo3b_cl) CONFIG="configs/eval/nemo3b_cl_clotho_embeddings.json" ;;
  oea_qwen3b_cl) CONFIG="configs/eval/qwen3b_cl_clotho_embeddings.json" ;;
  *) echo "[ERROR] unknown variant: ${VARIANT}" >&2; exit 2 ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck source=scripts/lib/conda.sh
source "${ROOT_DIR}/scripts/lib/conda.sh"
DATA_ROOT="${DATA_ROOT:-/home/jg525/datasets/oea}"
MODEL_ROOT="${OEA_MODEL_ROOT:-/home/jg525/models/oea}"
TRAIN_CSV="${RRD_TRAIN_CSV:-${DATA_ROOT}/audiocaps_v2_d004db3/metadata/train.csv}"
VAL_CSV="${RRD_VAL_CSV:-${DATA_ROOT}/audiocaps_v2_d004db3/metadata/val.csv}"
AUDIO_DIR="${RRD_AUDIO_DIR:-${DATA_ROOT}/audiocaps_raw_audio}"
REPLAY_ROOT="${RRD_REPLAY_ROOT:-${DATA_ROOT}/fiqa_mteb}"
OUT_BASE="${RRD_OUTPUT_ROOT:-${MODEL_ROOT}/rrd_oea}"
RUN_ID="${VARIANT}_audiocaps_fiqa_train_replay_$(date +%Y%m%d_%H%M%S)"
OUT="${OUT_BASE}/${RUN_ID}"
mkdir -p "${OUT}"
exec > >(tee "${OUT}/stdout.log") 2> >(tee "${OUT}/stderr.log" >&2)

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate oea-repro
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "===== RETRIEVAL-REPLAY DISTILLATION OEA ====="
echo "variant=${VARIANT} audio_dataset=audiocaps replay_dataset=fiqa replay_split=train"
echo "test_replay=disabled nq_replay=disabled uiq=disabled intent_gate=disabled"
echo "backbone_training=disabled lora_training=disabled"
for path in \
  "${ROOT_DIR}/${CONFIG}" "${TRAIN_CSV}" "${VAL_CSV}" "${AUDIO_DIR}" \
  "${REPLAY_ROOT}/corpus.jsonl" "${REPLAY_ROOT}/queries.jsonl" \
  "${REPLAY_ROOT}/qrels/train.jsonl"; do
  [[ -e "${path}" ]] || { echo "[ERROR] missing prerequisite: ${path}" >&2; exit 4; }
done

python - "${REPLAY_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
splits = {}
for split in ("train", "dev", "test"):
    path = root / "qrels" / f"{split}.jsonl"
    if not path.is_file():
        if split == "train":
            raise FileNotFoundError(path)
        continue
    query_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            query_id = row.get("query-id", row.get("query_id"))
            if query_id is None:
                raise ValueError(f"missing query ID: {path}:{line_number}")
            query_ids.add(str(query_id))
    splits[split] = query_ids
for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
    if left in splits and right in splits:
        overlap = splits[left] & splits[right]
        if overlap:
            raise ValueError(f"FiQA split query leakage {left}/{right}: {sorted(overlap)[:20]}")
print("RRD_SPLIT_AUDIT=" + json.dumps({key: len(value) for key, value in splits.items()}, sort_keys=True))
PY

python scripts/train_cgp_oea_projection_heads.py \
  --config "${ROOT_DIR}/${CONFIG}" \
  --model-root "${MODEL_ROOT}" \
  --dataset audiocaps \
  --train-csv "${TRAIN_CSV}" \
  --audio-dir "${AUDIO_DIR}" \
  --val-csv "${VAL_CSV}" \
  --val-audio-dir "${AUDIO_DIR}" \
  --output-dir "${OUT}" \
  --max-train-examples "${RRD_MAX_TRAIN_EXAMPLES:-8192}" \
  --batch-size "${RRD_BATCH_SIZE:-32}" \
  --epochs "${RRD_EPOCHS:-5}" \
  --geometry-lambda "${RRD_GEOMETRY_LAMBDA:-0.5}" \
  --pca-anchor-lambda "${RRD_PCA_ANCHOR_LAMBDA:-0.25}" \
  --distill-temperature "${RRD_DISTILL_TEMPERATURE:-0.07}" \
  --replay-root "${REPLAY_ROOT}" \
  --replay-split train \
  --max-replay-examples "${RRD_MAX_REPLAY_EXAMPLES:-4096}" \
  --replay-batch-size "${RRD_REPLAY_BATCH_SIZE:-32}" \
  --replay-lambda "${RRD_REPLAY_LAMBDA:-1.0}" \
  --replay-nce-lambda "${RRD_REPLAY_NCE_LAMBDA:-0.25}"

echo "[INFO] RRD checkpoint: ${OUT}/best.pt"
echo "[INFO] Evaluation config: ${OUT}/eval_config.json"
echo "[INFO] Evaluate the locked FiQA/NQ holdout gate before Qwen or UIQ work."
