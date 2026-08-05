#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 || "$1" != "--execute" ]]; then
  echo "Usage: bash scripts/run_cgp_oea_projection_heads.sh --execute <oea_nemo3b_cl|oea_qwen3b_cl>" >&2
  exit 2
fi
if [[ -z "${TMUX:-}" ]]; then
  echo "[ERROR] Run inside a normal tmux session: tmux new -s cgp_oea_mvp" >&2
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
TRAIN_CSV="${CGP_TRAIN_CSV:-${DATA_ROOT}/audiocaps_v2_d004db3/metadata/train.csv}"
VAL_CSV="${CGP_VAL_CSV:-${DATA_ROOT}/audiocaps_v2_d004db3/metadata/val.csv}"
AUDIO_DIR="${CGP_AUDIO_DIR:-${DATA_ROOT}/audiocaps_raw_audio}"
OUT_BASE="${CGP_OUTPUT_ROOT:-${MODEL_ROOT}/cgp_oea_mvp}"
RUN_ID="${VARIANT}_audiocaps_$(date +%Y%m%d_%H%M%S)"
OUT="${OUT_BASE}/${RUN_ID}"
mkdir -p "${OUT}"
exec > >(tee "${OUT}/stdout.log") 2> >(tee "${OUT}/stderr.log" >&2)

CONDA_BASE="$(resolve_conda_base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate oea-repro
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "===== CGP-OEA PROJECTION-HEAD-ONLY MVP ====="
echo "variant=${VARIANT} dataset=audiocaps"
echo "train_csv=${TRAIN_CSV} val_csv=${VAL_CSV} audio_dir=${AUDIO_DIR}"
echo "negative_uiq=disabled intent_gate=disabled backbone_training=disabled"
for path in "${ROOT_DIR}/${CONFIG}" "${TRAIN_CSV}" "${VAL_CSV}" "${AUDIO_DIR}"; do
  [[ -e "${path}" ]] || { echo "[ERROR] missing prerequisite: ${path}" >&2; exit 4; }
done

python scripts/train_cgp_oea_projection_heads.py \
  --config "${ROOT_DIR}/${CONFIG}" \
  --model-root "${MODEL_ROOT}" \
  --dataset audiocaps \
  --train-csv "${TRAIN_CSV}" \
  --audio-dir "${AUDIO_DIR}" \
  --val-csv "${VAL_CSV}" \
  --val-audio-dir "${AUDIO_DIR}" \
  --output-dir "${OUT}" \
  --max-train-examples "${CGP_MAX_TRAIN_EXAMPLES:-8192}" \
  --batch-size "${CGP_BATCH_SIZE:-32}" \
  --epochs "${CGP_EPOCHS:-5}" \
  --geometry-lambda "${CGP_GEOMETRY_LAMBDA:-0.5}" \
  --distill-temperature "${CGP_DISTILL_TEMPERATURE:-0.07}"

echo "[INFO] CGP checkpoint: ${OUT}/best.pt"
echo "[INFO] Evaluation config: ${OUT}/eval_config.json"
echo "[INFO] Do not start negative UIQ or intent-gate training until FiQA/NQ gate is checked."
