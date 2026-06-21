#!/usr/bin/env bash
# Real-model launcher: train an open MoE (Mixtral-8x7B / Qwen3-30B-A3B) on real data, multi-node, EPLB_MODE off|observe|apply.
set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1

# --- required paths / artifacts ----------------------------------------------
MEGATRON_DIR="${MEGATRON_DIR:?set MEGATRON_DIR to the Megatron-LM repo root}"
EPLB_DIR="${EPLB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to the mcore checkpoint dir (from convert_hf_to_mcore.sh / Megatron Bridge)}"
DATA_PATH="${DATA_PATH:?set DATA_PATH to the preprocessed data prefix (from prepare_data.sh, no .bin/.idx suffix)}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:?set TOKENIZER_MODEL to the HF repo/dir matching the checkpoint}"
SAVE_DIR="${SAVE_DIR:-}"                    # optional: where to write new checkpoints

# --- which open model (architecture recipe) ----------------------------------
MODEL="${MODEL:-qwen3_30b_a3b}"             # qwen3_30b_a3b | mixtral8x7b

# --- cluster topology (4x GB200 = 4 nodes x 4 GPUs by default) ----------------
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NNODES="${NNODES:-4}"
NODE_RANK="${NODE_RANK:-0}"                 # set per node: 0,1,2,3
MASTER_ADDR="${MASTER_ADDR:-localhost}"     # set to node-0 address on every node
MASTER_PORT="${MASTER_PORT:-6000}"
WORLD_SIZE=$(( GPUS_PER_NODE * NNODES ))

# --- parallelism (override via env; EP must divide world/(TP*PP)) -------------
TP="${TP:-2}"
PP="${PP:-1}"
EP="${EP:-8}"

# --- EPLB mode ----------------------------------------------------------------
EPLB_MODE="${EPLB_MODE:-observe}"           # off (pure Megatron) | observe (Phase B) | apply (Phase C)
export EPLB_MODE GPUS_PER_NODE
export PYTHONPATH="${MEGATRON_DIR}:${EPLB_DIR}:${PYTHONPATH:-}"

# --- per-model architecture args (must match the checkpoint config) -----------
if [[ "${MODEL}" == "mixtral8x7b" ]]; then
  MODEL_ARGS=(
    --use-mcore-models --disable-bias-linear --untie-embeddings-and-output-weights
    --seq-length 4096 --max-position-embeddings 32768
    --num-layers 32 --hidden-size 4096 --ffn-hidden-size 14336
    --num-attention-heads 32 --group-query-attention --num-query-groups 8
    --normalization RMSNorm --position-embedding-type rope --rotary-base 1000000
    --swiglu --no-masked-softmax-fusion --no-position-embedding
    --attention-dropout 0.0 --hidden-dropout 0.0
  )
  MOE_ARGS=(
    --num-experts 8 --moe-router-topk 2
    --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 1e-2
    --moe-token-dispatcher-type alltoall
  )
elif [[ "${MODEL}" == "qwen3_30b_a3b" ]]; then
  MODEL_ARGS=(
    --use-mcore-models --disable-bias-linear --untie-embeddings-and-output-weights
    --seq-length 8192 --max-position-embeddings 8192
    --num-layers 48 --hidden-size 2048 --ffn-hidden-size 6144
    --num-attention-heads 32 --kv-channels 128
    --group-query-attention --num-query-groups 4 --qk-layernorm
    --normalization RMSNorm --norm-epsilon 1e-6
    --position-embedding-type rope --rotary-base 1000000 --rotary-percent 1.0
    --swiglu --no-masked-softmax-fusion --attention-softmax-in-fp32
    --attention-dropout 0.0 --hidden-dropout 0.0
    --make-vocab-size-divisible-by 128
  )
  MOE_ARGS=(
    --num-experts 128 --moe-router-topk 8 --moe-ffn-hidden-size 768
    --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 1e-3
    --moe-token-dispatcher-type alltoall --moe-layer-freq 1
  )
else
  echo "unknown MODEL=${MODEL} (expected qwen3_30b_a3b | mixtral8x7b)" >&2
  exit 1
fi

# Phase C reference dispatcher needs SequentialMLP; off/observe use the fast native path (TE + grouped GEMM).
if [[ "${EPLB_MODE}" == "apply" ]]; then
  MODEL_ARGS+=(--transformer-impl local)
  echo "[run_real_moe] EPLB_MODE=apply -> forcing SequentialMLP (local impl, no grouped GEMM): reference path, slower"
else
  MODEL_ARGS+=(--transformer-impl transformer_engine --moe-grouped-gemm)
fi

PARALLEL_ARGS=(
  --tensor-model-parallel-size "${TP}"
  --pipeline-model-parallel-size "${PP}"
  --expert-model-parallel-size "${EP}"
  --use-distributed-optimizer
  --sequence-parallel
  --distributed-backend nccl
)

DATA_ARGS=(
  --tokenizer-type HuggingFaceTokenizer
  --tokenizer-model "${TOKENIZER_MODEL}"
  --data-path "${DATA_PATH}"
  --split 99,1,0
)

TRAIN_ARGS=(
  --micro-batch-size "${MICRO_BATCH_SIZE:-1}"
  --global-batch-size "${GLOBAL_BATCH_SIZE:-256}"
  --train-iters "${TRAIN_ITERS:-50}"
  --lr 1e-5 --min-lr 1e-6 --lr-decay-style cosine --lr-warmup-iters 5
  --weight-decay 0.1 --clip-grad 1.0
  --bf16
  --log-interval 1 --eval-interval 1000000 --eval-iters 0
)

LOAD_ARGS=(--load "${CHECKPOINT}" --no-load-optim --no-load-rng --dist-ckpt-strictness log_unexpected)
[[ -n "${SAVE_DIR}" ]] && LOAD_ARGS+=(--save "${SAVE_DIR}" --save-interval "${SAVE_INTERVAL:-1000}")

DISTRIBUTED_ARGS=(
  --nproc_per_node "${GPUS_PER_NODE}"
  --nnodes "${NNODES}"
  --node_rank "${NODE_RANK}"
  --master_addr "${MASTER_ADDR}"
  --master_port "${MASTER_PORT}"
)

echo "[run_real_moe] model=${MODEL} mode=${EPLB_MODE} world=${WORLD_SIZE} TP=${TP} PP=${PP} EP=${EP}"
torchrun "${DISTRIBUTED_ARGS[@]}" \
  "${EPLB_DIR}/scripts/pretrain_eplb_moe.py" \
  "${MODEL_ARGS[@]}" "${MOE_ARGS[@]}" "${PARALLEL_ARGS[@]}" \
  "${DATA_ARGS[@]}" "${TRAIN_ARGS[@]}" "${LOAD_ARGS[@]}"
