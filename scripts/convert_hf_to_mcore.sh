#!/usr/bin/env bash
# OPTIONAL: convert a HuggingFace Mixtral checkpoint to Megatron-Core format.
# Qwen3 uses Megatron Bridge instead; this pinned Megatron converter has no Qwen3 loader.
set -euo pipefail

MEGATRON_DIR="${MEGATRON_DIR:?set MEGATRON_DIR to the Megatron-LM repo root}"
HF_MODEL="${HF_MODEL:-mistralai/Mixtral-8x7B-v0.1}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:?set TOKENIZER_MODEL to the local Mixtral tokenizer.model path}"
SAVE_DIR="${SAVE_DIR:?set SAVE_DIR for the mcore checkpoint output}"
TP="${TP:-1}"
EP="${EP:-8}"

python "${MEGATRON_DIR}/tools/checkpoint/convert.py" \
  --model-type GPT \
  --loader mixtral_hf \
  --saver core \
  --load-dir "${HF_MODEL}" \
  --save-dir "${SAVE_DIR}" \
  --tokenizer-model "${TOKENIZER_MODEL}" \
  --target-tensor-parallel-size "${TP}" \
  --target-expert-parallel-size "${EP}"

echo "[convert] wrote mcore checkpoint to ${SAVE_DIR}"
echo "Then run training with: --load ${SAVE_DIR} (drop --mock-data, use the real tokenizer/data)"
