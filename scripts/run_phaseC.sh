#!/usr/bin/env bash
# Phase C launcher: end-to-end MoE training on Megatron with Scale-EPLB ACTIVE.
#
# Unlike Phase B (observe-only), this binds every MoELayer to the replication
# dispatcher (EPLB_MODE=apply): tokens are split across replicas per plan.q, replica
# weights are materialised from main(e), and gradients aggregate back to main(e).
# Training (loss/backward/optimizer/checkpoint) is plain Megatron.
#
# IMPORTANT: do NOT pass --moe-grouped-gemm. The v1 binding supports SequentialMLP
# (clean per-expert weights); GroupedMLP's fused weights are not yet sliced here.
#
# Prereqs on the cluster:
#   pip install -e /path/to/EP_balance
#   git clone https://github.com/NVIDIA/Megatron-LM && pip install -e Megatron-LM
#
# Usage (single node, 4 GB200):
#   MEGATRON_DIR=/path/to/Megatron-LM EPLB_DIR=/path/to/EP_balance bash scripts/run_phaseC.sh
# Full cluster (4 nodes x 4 GPUs): set NNODES=4 and per-node NODE_RANK/MASTER_ADDR/MASTER_PORT.
set -euo pipefail

# --- paths -------------------------------------------------------------------
MEGATRON_DIR="${MEGATRON_DIR:?set MEGATRON_DIR to the Megatron-LM repo root}"
EPLB_DIR="${EPLB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# --- cluster topology --------------------------------------------------------
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"        # GB200 per node
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6000}"
WORLD_SIZE=$(( GPUS_PER_NODE * NNODES ))

# --- MoE / parallelism (override via env) ------------------------------------
EP_SIZE="${EP_SIZE:-$WORLD_SIZE}"          # expert parallel = all ranks by default
NUM_EXPERTS="${NUM_EXPERTS:-16}"           # must be divisible by EP_SIZE
TOPK="${TOPK:-2}"
TRAIN_ITERS="${TRAIN_ITERS:-20}"

export GPUS_PER_NODE EPLB_MODE="apply"
export PYTHONPATH="${MEGATRON_DIR}:${EPLB_DIR}:${PYTHONPATH:-}"

MODEL_ARGS=(
  --num-layers 4
  --hidden-size 512
  --num-attention-heads 8
  --seq-length 1024
  --max-position-embeddings 1024
  --position-embedding-type rope
  --transformer-impl local
)

# NOTE: SequentialMLP experts (no --moe-grouped-gemm). Our dispatcher replaces the
# token dispatcher at runtime, but Megatron still constructs one; alltoall is fine.
MOE_ARGS=(
  --num-experts "${NUM_EXPERTS}"
  --moe-router-topk "${TOPK}"
  --moe-ffn-hidden-size 512
  --moe-token-dispatcher-type alltoall
  --expert-model-parallel-size "${EP_SIZE}"
)

PARALLEL_ARGS=(
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --distributed-backend nccl
)

DATA_ARGS=(
  --mock-data
  --tokenizer-type NullTokenizer
  --vocab-size 32000
)

TRAIN_ARGS=(
  --micro-batch-size 1
  --global-batch-size "${WORLD_SIZE}"
  --train-iters "${TRAIN_ITERS}"
  --eval-iters 0
  --lr 1e-4
  --min-lr 1e-5
  --lr-decay-style constant
  --bf16
  --log-interval 1
)

DISTRIBUTED_ARGS=(
  --nproc_per_node "${GPUS_PER_NODE}"
  --nnodes "${NNODES}"
  --node_rank "${NODE_RANK}"
  --master_addr "${MASTER_ADDR}"
  --master_port "${MASTER_PORT}"
)

echo "[run_phaseC] APPLY  world=${WORLD_SIZE} EP=${EP_SIZE} experts=${NUM_EXPERTS} topk=${TOPK}"
torchrun "${DISTRIBUTED_ARGS[@]}" \
  "${EPLB_DIR}/scripts/pretrain_eplb_moe.py" \
  "${MODEL_ARGS[@]}" "${MOE_ARGS[@]}" "${PARALLEL_ARGS[@]}" \
  "${DATA_ARGS[@]}" "${TRAIN_ARGS[@]}"
