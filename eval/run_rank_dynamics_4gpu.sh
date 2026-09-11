#!/usr/bin/env bash
# Single-node 4-GPU capture with offline contiguous mapping to virtual EP32.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-qwen3_30b_a3b}"
if [[ "${MODEL}" != "qwen3_30b_a3b" ]]; then
  echo "run_rank_dynamics_4gpu.sh requires MODEL=qwen3_30b_a3b for rank=expert//4" >&2
  exit 1
fi

export NNODES="${NNODES:-1}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-13401}"
export EXPECTED_WORLD_SIZE=4
export CAPTURE_EP=4
export CAPTURE_GLOBAL_BATCH_SIZE=4

# Qwen3 has 128 logical experts. TARGET_RANKS=32 gives the requested
# contiguous fixed placement: virtual_rank(expert) = expert // 4.
export TARGET_RANKS="${TARGET_RANKS:-32}"

# One EP4 occurrence contains 4 x SEQ_LEN tokens. Summing eight consecutive
# occurrences matches the token count of one EP32 occurrence.
export OCCURRENCE_GROUP="${OCCURRENCE_GROUP:-8}"
export EVAL_ITERS="${EVAL_ITERS:-128}"

SEED="${SEED:-1234}"
export SEED
export RUN_TAG="${RUN_TAG:-scratch_4gpu_virtual_ep32_seed${SEED}_${ARNOLD_TRIAL_ID:-manual}}"

exec bash "${SCRIPT_DIR}/run_rank_dynamics_32gpu.sh" "$@"
