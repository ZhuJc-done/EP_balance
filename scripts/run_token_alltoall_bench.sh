#!/usr/bin/env bash
# Pure NCCL All-to-All-v baseline for MoE token dispatch/combine.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EPLB_DIR="${EPLB_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# Use the same NCCL runtime as the training launchers.
source "${SCRIPT_DIR}/env_nccl_2307.sh"

MODEL="${MODEL:-qwen3_30b_a3b}"
case "${MODEL}" in
  qwen3_30b_a3b)
    MODEL_HIDDEN_SIZE=2048
    MODEL_TOPK=8
    MODEL_SEQ_LEN=8192
    ;;
  deepseek_v2_160e)
    MODEL_HIDDEN_SIZE=5120
    MODEL_TOPK=6
    MODEL_SEQ_LEN=4096
    ;;
  glm45_air)
    MODEL_HIDDEN_SIZE=4096
    MODEL_TOPK=8
    MODEL_SEQ_LEN=4096
    ;;
  *)
    echo "unknown MODEL=${MODEL} (expected qwen3_30b_a3b | deepseek_v2_160e | glm45_air)" >&2
    exit 1
    ;;
esac

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NNODES="${NNODES:-4}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6010}"
WORLD_SIZE=$(( GPUS_PER_NODE * NNODES ))

if [[ "${MASTER_ADDR}" == *:* ]]; then
  export NCCL_SOCKET_FAMILY=AF_INET6
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-=eth0}"
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
fi

EP_SIZE="${EP_SIZE:-${EP:-${WORLD_SIZE}}}"
SEQ_LEN="${SEQ_LEN:-${MODEL_SEQ_LEN}}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
TOKENS_PER_RANK="${TOKENS_PER_RANK:-$(( SEQ_LEN * MICRO_BATCH_SIZE ))}"
TOPK="${TOPK:-${MODEL_TOPK}}"
HIDDEN_SIZE="${HIDDEN_SIZE:-${MODEL_HIDDEN_SIZE}}"
METADATA_ELEMENTS="${METADATA_ELEMENTS:-0}"
ROW_ELEMENTS="${ROW_ELEMENTS:-$(( HIDDEN_SIZE + METADATA_ELEMENTS ))}"
DTYPE="${DTYPE:-bfloat16}"
RATIOS="${RATIOS:-1,2,4,8}"
WARMUP="${WARMUP:-10}"
REPEATS="${REPEATS:-50}"
LINK_GBPS="${LINK_GBPS:-400}"
MAX_BUFFER_MIB="${MAX_BUFFER_MIB:-8192}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${EPLB_EXP_DIR:-${EPLB_DIR}/logs}/network_a2a}"
RUN_NAME="${RUN_NAME:-${MODEL}_ep${EP_SIZE}_tok${TOKENS_PER_RANK}_topk${TOPK}_h${ROW_ELEMENTS}}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${OUTPUT_ROOT}/${RUN_NAME}}"

echo "[token-a2a] model=${MODEL} world=${WORLD_SIZE} EP=${EP_SIZE} nodes=${NNODES}x${GPUS_PER_NODE}"
echo "[token-a2a] rows/rank=$(( TOKENS_PER_RANK * TOPK )) row_elements=${ROW_ELEMENTS} dtype=${DTYPE} ratios=${RATIOS}"
echo "[token-a2a] output=${OUTPUT_PREFIX}.{json,csv}"

DISTRIBUTED_ARGS=(
  --nproc_per_node "${GPUS_PER_NODE}"
  --nnodes "${NNODES}"
  --node_rank "${NODE_RANK}"
  --master_addr "${MASTER_ADDR}"
  --master_port "${MASTER_PORT}"
)

torchrun "${DISTRIBUTED_ARGS[@]}" \
  "${EPLB_DIR}/eval/benchmark_token_alltoall.py" \
  --ep-size "${EP_SIZE}" \
  --tokens-per-rank "${TOKENS_PER_RANK}" \
  --topk "${TOPK}" \
  --row-elements "${ROW_ELEMENTS}" \
  --dtype "${DTYPE}" \
  --ratios "${RATIOS}" \
  --warmup "${WARMUP}" \
  --repeats "${REPEATS}" \
  --link-gbps "${LINK_GBPS}" \
  --max-buffer-mib "${MAX_BUFFER_MIB}" \
  --output-prefix "${OUTPUT_PREFIX}" \
  "$@"
