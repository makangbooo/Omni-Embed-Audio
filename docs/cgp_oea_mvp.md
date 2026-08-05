# CGP-OEA MVP

CGP-OEA (Capability-preserving Geometry Projection) is a gated, head-only
experiment.  It loads the released OEA checkpoint, freezes the base model and
all LoRA tensors, and updates only `audio_head` and `text_head`.

For each positive audio-caption pair, the script records the frozen base
hidden states as a teacher and the frozen LoRA hidden states as the student
input.  The trainable heads use

`symmetric InfoNCE + lambda * cross-modal geometry KL + mu * base-PCA anchor`

where the KL term matches the teacher's row and column similarity distributions
inside each batch.  The PCA anchor is a frozen 512-dimensional coordinate system
fit from the training split's base hidden states; it prevents the student heads
from learning an AudioCaps-only rotated geometry. Training data is AudioCaps or
Clotho only; FiQA/NQ and all UIQ files are evaluation-only.

The intended order is:

1. Run `scripts/run_asrur_oea_collapse_diagnostic.sh --execute` in tmux for
   Nemo3B/Qwen3B x FiQA/NQ four-space attribution.
2. Run `scripts/run_cgp_oea_projection_heads.sh --execute <variant>` for the
   head-only MVP.  Override `CGP_TRAIN_CSV`, `CGP_VAL_CSV`, and
   `CGP_AUDIO_DIR` if the remote AudioCaps layout differs.
3. Re-run the attribution matrix with `NEMO_CONFIG` or `QWEN_CONFIG` pointing
   to the generated `eval_config.json`.
4. Use `scripts/summarize_cgp_oea_gate.py` to check recovery.  A minimum of
   half of the Base-hidden to Full-OEA MRR and Recall@10 gap must be recovered,
   while validation recall may drop by at most two percentage points.

Negative UIQ loss is deliberately absent from this MVP.  It is allowed only
after the gate passes for both model variants and both datasets.  The intent
gate is the final ablation, after the negative-loss result is frozen.
