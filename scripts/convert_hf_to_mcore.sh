#!/usr/bin/env bash
# OPTIONAL: convert a HuggingFace MoE checkpoint to Megatron-Core format.
#
# Phase B with run_phaseB.sh needs NO real checkpoint (it uses --mock-data, which
# is enough to validate the EPLB pipeline + determinism). Use this only when you
# want REALISTIC routing skew from a pretrained MoE (e.g. Mixtral / Qwen-MoE /
# DeepSeek), then point training at the converted checkpoint with --load.
#
# Two supported paths:
#   (A) Megatron Bridge (recommended; HF<->Megatron recipes):
#         pip install megatron-bridge
#         # see https://github.com/NVIDIA/Megatron-LM (Megatron Bridge) for the
#         # per-model recipe; it writes an mcore-format dist checkpoint.
#   (B) Megatron-LM's tools/checkpoint/convert.py (loader/saver plugins).
#
# Example for Mixtral-8x7B via path (B). Adapt loader/args for other models.
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
