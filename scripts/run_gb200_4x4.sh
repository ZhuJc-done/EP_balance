#!/usr/bin/env bash
# Multi-node launcher for 4 x GB200 nodes (4 GPUs each = 16 ranks) running Scale-EPLB on Megatron-LM.
# Auto-detects Slurm for rank/master discovery; defaults to a self-contained mock-data MoE smoke test
# in observe mode (no checkpoint/data needed). Set EPLB_MODE=apply for the active dispatcher, or REAL=1
# to forward to scripts/run_real_moe.sh with the same multi-node + GB200 env.
set -euo pipefail

# --- paths -------------------------------------------------------------------
MEGATRON_DIR="${MEGATRON_DIR:?set MEGATRON_DIR to the Megatron-LM repo root}"
EPLB_DIR="${EPLB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="${MEGATRON_DIR}:${EPLB_DIR}:${PYTHONPATH:-}"

# --- cluster topology: auto-detect Slurm, else manual per-node env -----------
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"                       # GB200: 4 GPUs per node
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  NNODES="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-4}}"
  NODE_RANK="${SLURM_NODEID:-0}"
  MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
  MASTER_PORT="${MASTER_PORT:-29500}"
else
  NNODES="${NNODES:-4}"
  NODE_RANK="${NODE_RANK:?on each node set NODE_RANK=0..NNODES-1 (or launch under Slurm)}"
  MASTER_ADDR="${MASTER_ADDR:?set MASTER_ADDR to node-0 hostname/IP on every node}"
  MASTER_PORT="${MASTER_PORT:-29500}"
fi
WORLD_SIZE=$(( GPUS_PER_NODE * NNODES ))

# --- GB200 / Blackwell runtime + NCCL/RDMA env (override per fabric) ----------
export CUDA_DEVICE_MAX_CONNECTIONS=1                      # required by Megatron for correct comm overlap
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5}"                 # adjust to your CX7 HCA names (e.g. mlx5_0,mlx5_1)
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"   # control-plane NIC for rendezvous
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"        # RoCE GID index; ignore for pure InfiniBand
# Uncomment/adjust as your fabric requires:
# export NCCL_NET_GDR_LEVEL=PIX
# export NCCL_IB_DISABLE=0
# export NCCL_DEBUG=INFO; export NCCL_DEBUG_SUBSYS=INIT,NET

# --- forward to the real-model launcher if requested -------------------------
if [[ "${REAL:-0}" == "1" ]]; then
  export GPUS_PER_NODE NNODES NODE_RANK MASTER_ADDR MASTER_PORT
  echo "[run_gb200_4x4] REAL=1 -> scripts/run_real_moe.sh (model=${MODEL:-qwen3_30b_a3b} mode=${EPLB_MODE:-observe})"
  exec bash "${EPLB_DIR}/scripts/run_real_moe.sh"
fi

# --- smoke-test MoE / parallelism (override via env) -------------------------
EPLB_MODE="${EPLB_MODE:-observe}"                         # observe (Phase B, safest) | apply (Phase C) | off
EP_SIZE="${EP_SIZE:-$WORLD_SIZE}"                         # expert parallel = all 16 ranks by default
NUM_EXPERTS="${NUM_EXPERTS:-32}"                          # must be divisible by EP_SIZE
TOPK="${TOPK:-4}"
TRAIN_ITERS="${TRAIN_ITERS:-20}"
export GPUS_PER_NODE EPLB_MODE

MODEL_ARGS=(
  --num-layers 4
  --hidden-size 1024
  --num-attention-heads 8
  --seq-length 2048
  --max-position-embeddings 2048
  --position-embedding-type rope
  --swiglu
  --disable-bias-linear
  --transformer-impl local                               # SequentialMLP: works without TE and for apply mode
)

MOE_ARGS=(
  --num-experts "${NUM_EXPERTS}"
  --moe-router-topk "${TOPK}"
  --moe-ffn-hidden-size 1024
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
  --eval-interval 1000000
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

echo "[run_gb200_4x4] mode=${EPLB_MODE} nodes=${NNODES} gpn=${GPUS_PER_NODE} world=${WORLD_SIZE} node_rank=${NODE_RANK} master=${MASTER_ADDR}:${MASTER_PORT} EP=${EP_SIZE} experts=${NUM_EXPERTS} topk=${TOPK}"
torchrun "${DISTRIBUTED_ARGS[@]}" \
  "${EPLB_DIR}/scripts/pretrain_eplb_moe.py" \
  "${MODEL_ARGS[@]}" "${MOE_ARGS[@]}" "${PARALLEL_ARGS[@]}" \
  "${DATA_ARGS[@]}" "${TRAIN_ARGS[@]}"
