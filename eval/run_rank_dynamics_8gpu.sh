#!/usr/bin/env bash
# Single-node 8-GPU forward-only capture with offline mapping to virtual EP32.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-qwen3_30b_a3b}"
if [[ "${MODEL}" != "qwen3_30b_a3b" ]]; then
  echo "run_rank_dynamics_8gpu.sh requires MODEL=qwen3_30b_a3b for rank=expert//4" >&2
  exit 1
fi

export NNODES=1
export GPUS_PER_NODE=8
export NODE_RANK=0
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-13401}"
export EXPECTED_WORLD_SIZE=8
export CAPTURE_EP=8
export CAPTURE_GLOBAL_BATCH_SIZE=8

# Qwen3 has 128 experts. This is the virtual fixed EP32 placement:
# virtual_rank(expert) = expert // 4.
export TARGET_RANKS=32

# EP8 sees one quarter of an EP32 token volume per occurrence. Group four
# consecutive captures; 64 raw occurrences produce 16 virtual EP32 points.
export OCCURRENCE_GROUP=4
export EVAL_ITERS="${EVAL_ITERS:-64}"

SEED="${SEED:-1234}"
export SEED
export RUN_TAG="${RUN_TAG:-scratch_8gpu_virtual_ep32_seed${SEED}_${ARNOLD_TRIAL_ID:-manual}}"

exec bash "${SCRIPT_DIR}/run_rank_dynamics_32gpu.sh" "$@"
