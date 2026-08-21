#!/usr/bin/env bash
# Multi-node launcher for 4 x GB200 nodes (4 GPUs each = 16 ranks) running Scale-EPLB on Megatron-LM.
# Auto-detects Slurm for rank/master discovery; defaults to a self-contained mock-data MoE smoke test
# in observe mode (no checkpoint/data needed). Set EPLB_MODE=apply for the active dispatcher, or REAL=1
# to forward to scripts/run_real_moe.sh with the same multi-node + GB200 env.
set -euo pipefail

# Pin runtime and JIT linkage to the validated NCCL build before Python starts.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_nccl_2307.sh"

# --- paths -------------------------------------------------------------------
MEGATRON_DIR="${MEGATRON_DIR:?set MEGATRON_DIR to the Megatron-LM repo root}"
EPLB_DIR="${EPLB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Drop forks that ship a regular `megatron` package and would hijack the community PEP420
# namespace (e.g. ByteDance mariana); space-separated, override via EPLB_STRIP_PYTHONPATH.
STRIP_PYTHONPATH="${EPLB_STRIP_PYTHONPATH:-/opt/tiger/mariana}"
_clean_pp="${PYTHONPATH:-}"
for _p in ${STRIP_PYTHONPATH}; do
  _clean_pp="$(printf '%s' "${_clean_pp}" | tr ':' '\n' | grep -vxF "${_p}" | paste -sd: -)"
done
export PYTHONPATH="${MEGATRON_DIR}:${EPLB_DIR}:${_clean_pp}"

# --- cluster topology: auto-detect Slurm, else manual per-node env -----------
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"                       # GB200: 4 GPUs per node
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  NNODES="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-4}}"
  NODE_RANK="${SLURM_NODEID:-0}"
  MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
  MASTER_PORT="${MASTER_PORT:-29500}"
else
  NNODES="${NNODES:-4}"
  NODE_RANK="${NODE_RANK:-0}"
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  MASTER_PORT="${MASTER_PORT:-29500}"
  if [[ "${NNODES}" -gt 1 ]]; then          # multi-node needs explicit rendezvous coordinates
    : "${MASTER_ADDR:?multi-node: set MASTER_ADDR to node-0 IP on every node}"
    [[ "${MASTER_ADDR}" != "127.0.0.1" ]] || { echo "multi-node: MASTER_ADDR must be node-0 IP, not 127.0.0.1" >&2; exit 1; }
  fi
fi
WORLD_SIZE=$(( GPUS_PER_NODE * NNODES ))

# --- runtime env -------------------------------------------------------------
export CUDA_DEVICE_MAX_CONNECTIONS=1                      # required by Megatron for correct comm overlap
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"
if [[ "${NNODES}" -gt 1 ]]; then
  # Multi-node: NCCL auto-detects NICs + IB by default; we don't hardcode any fabric names.
  # Only pin the control-plane socket NIC, because under docker auto-detect may pick a
  # non-routable veth/bridge/lo. Default to the interface backing the default route
  # (guaranteed to reach other nodes); override with NCCL_SOCKET_IFNAME / NCCL_IB_HCA / NCCL_IB_GID_INDEX.
  if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
    _def_if="$(ip -o route get 1.1.1.1 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p' | head -n1)"
    [[ -n "${_def_if}" ]] && export NCCL_SOCKET_IFNAME="${_def_if}"
  fi
  [[ -n "${NCCL_IB_HCA:-}" ]] && export NCCL_IB_HCA               # else NCCL auto-detects all IB devices
  [[ -n "${NCCL_IB_GID_INDEX:-}" ]] && export NCCL_IB_GID_INDEX   # set only for RoCE if the default GID is wrong
  echo "[run_gb200_4x4] NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-<auto>} NCCL_IB_HCA=${NCCL_IB_HCA:-<auto>}"
elif [[ -z "${SLURM_JOB_ID:-}" ]]; then
  # Clear site-baked NCCL defaults (IPv6 socket family, CGA cluster size) that break plain single-node torchrun.
  unset NCCL_SOCKET_FAMILY NCCL_CGA_CLUSTER_SIZE NCCL_SOCKET_IFNAME 2>/dev/null || true
fi

# --- forward to the real-model launcher if requested -------------------------
if [[ "${REAL:-0}" == "1" ]]; then
  export GPUS_PER_NODE NNODES NODE_RANK MASTER_ADDR MASTER_PORT
  echo "[run_gb200_4x4] REAL=1 -> scripts/run_real_moe.sh (model=${MODEL:-qwen3_30b_a3b} mode=${EPLB_MODE:-observe})"
  exec bash "${EPLB_DIR}/scripts/run_real_moe.sh" "$@"    # forward any extra Megatron flags (e.g. --recompute-*)
fi

# --- smoke-test MoE / parallelism (override via env) -------------------------
EPLB_MODE="${EPLB_MODE:-observe}"                         # observe (Phase B, safest) | apply (Phase C) | off
EP_SIZE="${EP_SIZE:-$WORLD_SIZE}"                         # expert parallel = all 16 ranks by default
NUM_EXPERTS="${NUM_EXPERTS:-32}"                          # must be divisible by EP_SIZE
TOPK="${TOPK:-4}"
TRAIN_ITERS="${TRAIN_ITERS:-20}"
export GPUS_PER_NODE EPLB_MODE

# --- debug / profiling toggles (optional) ------------------------------------
# EPLB_PROFILE=1 -> aggregate native/* or apply/* CUDA-event timing.
# EPLB_DEBUG_TIMING=1 -> print one forward breakdown per MoE invocation in off/observe/apply.
export EPLB_PROFILE="${EPLB_PROFILE:-0}"
export EPLB_PROFILE_EVERY="${EPLB_PROFILE_EVERY:-20}"
export EPLB_DEBUG_TIMING="${EPLB_DEBUG_TIMING:-0}"

# EPLB_REMATERIALIZE=1 (apply mode) -> don't hold replica expert weights across fwd->bwd; free them
# after forward and re-broadcast at backward start (weight recompute). Trades extra broadcast for memory.
export EPLB_REMATERIALIZE="${EPLB_REMATERIALIZE:-0}"

if [[ "${EPLB_MODE}" == "apply" && "${EPLB_ADAPTER:-alltoall}" == "deepep" ]]; then
  [[ "${EPLB_WEIGHT_COMM:-}" == "gin" && "${EPLB_GIN_FENCE:-}" == "signal" ]] || {
    echo "ElasticBuffer smoke requires EPLB_WEIGHT_COMM=gin EPLB_GIN_FENCE=signal" >&2; exit 1;
  }
  [[ "${EPLB_PROFILE}" == "0" && "${PROFILE_TRACE:-0}" == "0" ]] || {
    echo "zero-sync smoke requires EPLB_PROFILE=0 PROFILE_TRACE=0" >&2; exit 1;
  }
  python - <<'PY'
import deep_ep
import nccl_gin
from eplb.integration.eplb_manager import _nccl_runtime_version
assert hasattr(deep_ep, "ElasticBuffer"), "DeepEP ElasticBuffer is unavailable"
nccl_gin._ensure_loaded()
v = _nccl_runtime_version()
assert v == (2, 30, 7), f"NCCL 2.30.7 runtime required, got {v}"
print(f"[run_gb200_4x4] transport={deep_ep.ElasticBuffer.__module__}.ElasticBuffer "
      f"GIN=ready NCCL={v}")
PY
fi

# PROFILE_TRACE=1 -> Megatron's native PyTorch profiler emits a perfetto/chrome trace under
# PROFILE_DIR; the eplb/* record_function labels show up inline so you can read the call stack.
PROFILE_ARGS=()
if [[ "${PROFILE_TRACE:-0}" == "1" ]]; then
  PROFILE_ARGS=(
    --profile
    --use-pytorch-profiler
    --profile-step-start "${PROFILE_STEP_START:-5}"
    --profile-step-end "${PROFILE_STEP_END:-7}"
    --profile-ranks 0
    --tensorboard-dir "${PROFILE_DIR:-./eplb_torch_profiler}"
  )
fi

MODEL_ARGS=(
  --num-layers 4
  --hidden-size 1024
  --num-attention-heads 8
  --seq-length 2048
  --max-position-embeddings 2048
  --position-embedding-type rope
  --swiglu
  --disable-bias-linear
  --transformer-impl local                               # SequentialMLP: required by the apply-mode binding
  # pure PyTorch kernels: no fused-kernel extension required
  --no-rope-fusion
  --no-masked-softmax-fusion
  --no-bias-swiglu-fusion
  --no-gradient-accumulation-fusion
  --no-persist-layer-norm
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

if [[ "${NNODES}" -eq 1 && -z "${SLURM_JOB_ID:-}" ]]; then
  # single-node: let torchrun pick a free ephemeral port (avoids EADDRINUSE on shared hosts)
  DISTRIBUTED_ARGS=( --standalone --nnodes 1 --nproc_per_node "${GPUS_PER_NODE}" )
  RDZV_DESC="standalone(auto-port)"
else
  DISTRIBUTED_ARGS=(
    --nproc_per_node "${GPUS_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
  )
  RDZV_DESC="${MASTER_ADDR}:${MASTER_PORT}"
fi

# Tee stdout+stderr to a timestamped per-node log (pipefail keeps torchrun's exit code); override path via LOG_FILE, disable with LOG=0.
LOG="${LOG:-1}"
LOG_FILE="${LOG_FILE:-${EPLB_DIR}/logs/run_${EPLB_MODE}_node${NODE_RANK}.log}"
echo "[run_gb200_4x4] mode=${EPLB_MODE} nodes=${NNODES} gpn=${GPUS_PER_NODE} world=${WORLD_SIZE} node_rank=${NODE_RANK} rdzv=${RDZV_DESC} EP=${EP_SIZE} experts=${NUM_EXPERTS} topk=${TOPK}"
[[ "${LOG}" != "0" ]] && { mkdir -p "$(dirname "${LOG_FILE}")"; echo "[run_gb200_4x4] logging to ${LOG_FILE}"; }
# Extra Megatron flags go last so argparse lets them override the smoke-test recipe above; captured at
# top level because `$@` inside a function would resolve to the function's own arguments.
EXTRA_ARGS=("$@")
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "[run_gb200_4x4] extra Megatron args: ${EXTRA_ARGS[*]}"
run_torchrun() {
  torchrun "${DISTRIBUTED_ARGS[@]}" \
    "${EPLB_DIR}/scripts/pretrain_eplb_moe.py" \
    "${MODEL_ARGS[@]}" "${MOE_ARGS[@]}" "${PARALLEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" "${TRAIN_ARGS[@]}" "${PROFILE_ARGS[@]}" "${EXTRA_ARGS[@]}"
}
if [[ "${LOG}" != "0" ]]; then
  run_torchrun 2>&1 | tee "${LOG_FILE}"
else
  run_torchrun
fi
