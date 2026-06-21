#!/usr/bin/env bash
# OPTIONAL: convert a HuggingFace MoE checkpoint to Megatron-Core format for realistic routing skew (example: Mixtral-8x7B).
set -euo pipefail

MEGATRON_DIR="${MEGATRON_DIR:?set MEGATRON_DIR to the Megatron-LM repo root}"
HF_MODEL="${HF_MODEL:-mistralai/Mixtral-8x7B-v0.1}"   # any HF MoE repo or local dir
SAVE_DIR="${SAVE_DIR:?set SAVE_DIR for the mcore checkpoint output}"
TP="${TP:-1}"
EP="${EP:-8}"

python "${MEGATRON_DIR}/tools/checkpoint/convert.py" \
  --model-type GPT \
  --loader llama_mistral \
  --saver mcore \
  --load-dir "${HF_MODEL}" \
  --save-dir "${SAVE_DIR}" \
  --target-tensor-parallel-size "${TP}" \
  --target-expert-parallel-size "${EP}" \
  --checkpoint-type hf

echo "[convert] wrote mcore checkpoint to ${SAVE_DIR}"
echo "Then run training with: --load ${SAVE_DIR} (drop --mock-data, use the real tokenizer/data)"
