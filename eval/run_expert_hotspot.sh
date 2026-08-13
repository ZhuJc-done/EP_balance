#!/usr/bin/env bash
# Frozen-checkpoint Megatron evaluation that dumps raw MoE routing for hotspot analysis.
set -euo pipefail

EPLB_DIR="${EPLB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MEGATRON_DIR="${MEGATRON_DIR:?set MEGATRON_DIR to the Megatron-LM repo root}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to a trained MCore checkpoint}"
DATA_PATH="${DATA_PATH:?set DATA_PATH to a Megatron indexed-data prefix (without .bin/.idx)}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-Qwen/Qwen3-30B-A3B}"
MODEL="${MODEL:-qwen3_30b_a3b}"
WORKLOAD="${WORKLOAD:-workload}"

NNODES="${NNODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-6000}"
TP="${TP:-1}"
PP="${PP:-1}"
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
if (( TP <= 0 || PP <= 0 || WORLD_SIZE % (TP * PP) != 0 )); then
  echo "world size ${WORLD_SIZE} must be divisible by positive TP*PP=$((TP * PP))" >&2
  exit 1
fi
EP="${EP:-$((WORLD_SIZE / (TP * PP)))}"

MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
DATA_PARALLEL_SIZE=$((WORLD_SIZE / (TP * PP)))
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((MICRO_BATCH_SIZE * DATA_PARALLEL_SIZE))}"
SEQ_LEN="${SEQ_LEN:-4096}"
EVAL_ITERS="${EVAL_ITERS:-50}"

TRACE_OUT="${TRACE_OUT:-${EPLB_DIR}/logs/hotspot_${WORKLOAD}.pt}"
LOG_FILE="${LOG_FILE:-${EPLB_DIR}/logs/hotspot_${WORKLOAD}_node${NODE_RANK}.log}"
TRACE_FLUSH_EVERY="${TRACE_FLUSH_EVERY:-$([[ "${MODEL}" == "mixtral8x7b" ]] && echo 32 || echo 48)}"

if [[ ! -f "${MEGATRON_DIR}/pretrain_gpt.py" ]]; then
  echo "Megatron pretrain entrypoint not found: ${MEGATRON_DIR}/pretrain_gpt.py" >&2
  exit 1
fi
if [[ ! -d "${CHECKPOINT}" ]]; then
  echo "MCore checkpoint directory not found: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${DATA_PATH}.bin" || ! -f "${DATA_PATH}.idx" ]]; then
  echo "Megatron indexed data not found: expected ${DATA_PATH}.bin and ${DATA_PATH}.idx" >&2
  exit 1
fi
if (( EP <= 0 || DATA_PARALLEL_SIZE % EP != 0 )); then
  echo "EP=${EP} must divide world/(TP*PP)=${DATA_PARALLEL_SIZE}" >&2
  exit 1
fi
if (( PP != 1 )); then
  echo "Hotspot capture currently requires PP=1 so rank 0 observes every MoE layer" >&2
  exit 1
fi
if (( GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * DATA_PARALLEL_SIZE) != 0 )); then
  echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must divide by micro_batch*DP=$((MICRO_BATCH_SIZE * DATA_PARALLEL_SIZE))" >&2
  exit 1
fi
if [[ -n "${ROUTER_SKEW:-}" && "${ALLOW_SYNTHETIC_SKEW:-0}" != "1" ]]; then
  echo "ROUTER_SKEW is synthetic and disabled for real hotspot capture; unset it or set ALLOW_SYNTHETIC_SKEW=1" >&2
  exit 1
fi
if [[ "${NODE_RANK}" == "0" && -e "${TRACE_OUT}" && "${OVERWRITE:-0}" != "1" ]]; then
  echo "Trace already exists: ${TRACE_OUT} (set OVERWRITE=1 to replace it)" >&2
  exit 1
fi

mkdir -p "$(dirname "${TRACE_OUT}")" "$(dirname "${LOG_FILE}")"

# Observe computes/logs a candidate plan but leaves Megatron's dispatcher unchanged.
# Evaluation mode freezes the checkpoint and bypasses optimizer construction/updates.
export EPLB_DIR MEGATRON_DIR CHECKPOINT DATA_PATH TOKENIZER_MODEL MODEL
export NNODES GPUS_PER_NODE NODE_RANK MASTER_ADDR MASTER_PORT TP PP EP
export MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE SEQ_LEN
export EPLB_MODE=observe
export MOCK=0
export ROUTER_BALANCING=none
export EPLB_TRACE_OUT="${TRACE_OUT}"
export EPLB_TRACE_MAX=0
export EPLB_TRACE_EVERY="${TRACE_FLUSH_EVERY}"
export TRAIN_ITERS=1
export LR_WARMUP_ITERS=0
export NUM_WORKERS="${NUM_WORKERS:-0}"
export LOG_FILE

if [[ "${NODE_RANK}" == "0" ]]; then
  RUN_METADATA="${TRACE_OUT%.pt}.run.env"
  {
    echo "WORKLOAD=${WORKLOAD}"
    echo "MODEL=${MODEL}"
    echo "CHECKPOINT=${CHECKPOINT}"
    echo "DATA_PATH=${DATA_PATH}"
    echo "TOKENIZER_MODEL=${TOKENIZER_MODEL}"
    echo "WORLD_SIZE=${WORLD_SIZE}"
    echo "TP=${TP}"
    echo "PP=${PP}"
    echo "EP=${EP}"
    echo "MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}"
    echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
    echo "SEQ_LEN=${SEQ_LEN}"
    echo "EVAL_ITERS=${EVAL_ITERS}"
    echo "TRACE_OUT=${TRACE_OUT}"
  } > "${RUN_METADATA}"
fi

echo "[run_expert_hotspot] frozen checkpoint, unmodified Megatron dispatch"
echo "[run_expert_hotspot] workload=${WORKLOAD} trace=${TRACE_OUT}"
echo "[run_expert_hotspot] world=${WORLD_SIZE} TP=${TP} PP=${PP} EP=${EP} seq=${SEQ_LEN}"

exec bash "${EPLB_DIR}/scripts/run_real_moe.sh" \
  --skip-train \
  --eval-iters "${EVAL_ITERS}" \
  --split 0,100,0 \
  --ckpt-format torch_dist \
  "$@"
