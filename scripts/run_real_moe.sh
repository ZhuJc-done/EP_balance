#!/usr/bin/env bash
# Real-model launcher: train a supported open-MoE recipe on real data, multi-node, EPLB_MODE off|observe|apply.
set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1

# Pin runtime and JIT linkage to the validated NCCL build before Python starts.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env_nccl_2307.sh"

# --- required paths / artifacts ----------------------------------------------
MEGATRON_DIR="${MEGATRON_DIR:?set MEGATRON_DIR to the Megatron-LM repo root}"
EPLB_DIR="${EPLB_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
# Drop regular-package Megatron forks that would override the community
# Megatron PEP420 namespace even when MEGATRON_DIR appears first.
_clean_pp=""
IFS=':' read -r -a _pp_parts <<< "${PYTHONPATH:-}"
for _entry in "${_pp_parts[@]}"; do
  [[ -z "${_entry}" ]] && continue
  _keep=1
  for _strip in ${EPLB_STRIP_PYTHONPATH:-/opt/tiger/mariana}; do
    [[ "${_entry}" == "${_strip}" ]] && _keep=0
  done
  if [[ "${_keep}" == "1" ]]; then
    _clean_pp="${_clean_pp:+${_clean_pp}:}${_entry}"
  fi
done
export PYTHONPATH="${MEGATRON_DIR}:${EPLB_DIR}${_clean_pp:+:${_clean_pp}}"

# MOCK=1: real model architecture, but mock-data + random init (no checkpoint/data/tokenizer needed) --
# a pure-Megatron "does it run / throughput / memory" baseline.
# MOCK=0 uses real tokenized data. Set FROM_SCRATCH=1 to random-initialize the model without a
# checkpoint; otherwise CHECKPOINT remains required for the pretrained-weight path.
MOCK="${MOCK:-0}"
FROM_SCRATCH="${FROM_SCRATCH:-0}"
if [[ "${MOCK}" != "0" && "${MOCK}" != "1" ]]; then
  echo "invalid MOCK=${MOCK} (expected 0 or 1)" >&2
  exit 1
fi
if [[ "${FROM_SCRATCH}" != "0" && "${FROM_SCRATCH}" != "1" ]]; then
  echo "invalid FROM_SCRATCH=${FROM_SCRATCH} (expected 0 or 1)" >&2
  exit 1
fi
if [[ "${MOCK}" == "1" ]]; then
  CHECKPOINT="" ; DATA_PATH="" ; TOKENIZER_MODEL=""
else
  DATA_PATH="${DATA_PATH:?set DATA_PATH to the preprocessed data prefix (from prepare_data.sh, no .bin/.idx suffix)}"
  TOKENIZER_MODEL="${TOKENIZER_MODEL:?set TOKENIZER_MODEL to the HF repo/dir used to preprocess the data}"
  if [[ "${FROM_SCRATCH}" == "1" ]]; then
    CHECKPOINT=""
    echo "[run_real_moe] FROM_SCRATCH=1 -> real data + random model initialization (no checkpoint)"
  else
    CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to the mcore checkpoint dir, or set FROM_SCRATCH=1}"
  fi
fi
SAVE_DIR="${SAVE_DIR:-}"                    # optional: where to write new checkpoints

# --- which open model (architecture recipe) ----------------------------------
MODEL="${MODEL:-qwen3_30b_a3b}"
# qwen3_30b_a3b | deepseek_v2_160e | glm45_air

# --- cluster topology (4x GB200 = 4 nodes x 4 GPUs by default) ----------------
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NNODES="${NNODES:-4}"
NODE_RANK="${NODE_RANK:-0}"                 # set per node: 0,1,2,3
MASTER_ADDR="${MASTER_ADDR:-localhost}"     # set to node-0 address on every node
MASTER_PORT="${MASTER_PORT:-6000}"
WORLD_SIZE=$(( GPUS_PER_NODE * NNODES ))

# Some cluster images inject NCCL_SOCKET_FAMILY=AF_INET. Override it when the
# rendezvous address is an IPv6 literal; otherwise NCCL cannot select eth0 on
# IPv6-only nodes and fails with "Bootstrap : no socket interface found".
if [[ "${MASTER_ADDR}" == *:* ]]; then
  export NCCL_SOCKET_FAMILY=AF_INET6
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-=eth0}"
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
  echo "[run_real_moe] IPv6 bootstrap: NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY}" \
       "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}"
fi

# --- parallelism (override via env; EP must divide world/(TP*PP)) -------------
TP="${TP:-2}"
PP="${PP:-1}"
EP="${EP:-8}"

# --- EPLB mode ----------------------------------------------------------------
EPLB_MODE="${EPLB_MODE:-observe}"           # off (pure Megatron) | observe (Phase B) | apply (Phase C)
EPLB_PLAN_SOLVER="${EPLB_PLAN_SOLVER:-scale}" # scale | fastermoe | deepseek | flexmoe | lplb
case "${EPLB_PLAN_SOLVER}" in
  scale|fastermoe|deepseek|flexmoe|lplb) ;;
  *)
    echo "unknown EPLB_PLAN_SOLVER=${EPLB_PLAN_SOLVER}; expected scale|fastermoe|deepseek|flexmoe|lplb" >&2
    exit 1
    ;;
esac
export EPLB_MODE EPLB_PLAN_SOLVER GPUS_PER_NODE

# --- optional routing-trace capture (observe mode) ---------------------------
# EPLB_TRACE_OUT=<path> makes rank 0 dump the real gathered Ω[R,E] per
# (layer, micro-batch); EPLB_TRACE_MAX caps sample count (0=all), EPLB_TRACE_EVERY
# is the flush cadence. Replay it through every load balancer with:
#   python -m baseline.benchmark --trace <path> --strategies scale,eplb,fastermoe,flexmoe,lplb
if [[ -n "${EPLB_TRACE_OUT:-}" ]]; then
  export EPLB_TRACE_OUT EPLB_TRACE_MAX EPLB_TRACE_EVERY
  echo "[run_real_moe] EPLB_TRACE_OUT=${EPLB_TRACE_OUT} (routing trace -> baseline replay)"
fi

# --- optional sweep / instrumentation knobs ----------------------------------
# EPLB_N_SLOT=<int>         per-rank physical slot budget N_slot (default 2x mains-per-rank). Each slot
#                           above the mains floor carries one expert replica -> the balance-vs-memory knob.
# EPLB_PLAN_SOLVER=<name>    placement/quota policy in apply mode: scale (default), fastermoe,
#                           deepseek (main-fixed), flexmoe, or lplb. All policies reuse the same
#                           weight materialization, token transport, expert kernels, and grad reduction.
# EPLB_FASTERMOE_BW_NET=<B/s>, EPLB_FASTERMOE_BW_MM=<FLOP/s> tune its cost model.
# EPLB_DEEPSEEK_NUM_GROUPS=<n> sets the requested hierarchy group count (default 8; invalid values
#                           fall back to a topology-compatible count).
# EPLB_FLEXMOE_THRESHOLD=<f> sets its expansion trigger (default 1.2).
# EPLB_LPLB_ROOT=<path>, EPLB_LPLB_TOPOLOGY=auto|cube|ring configure optional compiled LPLB.
# EPLB_PROFILE=1            periodic per-region latency + peak-memory summary. CUDA events are queued and
#                           resolved in one batch, so this adds no per-region host sync.
# EPLB_PROFILE_ALL_RANKS=1  every rank prints its own summary (required for max-vs-mean straggler analysis).
# EPLB_PROFILE_EVERY=<n>    summary cadence; EPLB_PROFILE_RESET_AT=<n> resets peak memory after warmup.
# EPLB_DEBUG_TIMING=1       print compact per-MoE forward/backward timing (apply backward ends at reverse-dispatch).
#                           off: native router/dispatch/GEMM/combine; observe: native stages + solver;
#                           apply: Scale-EPLB stages + expert transfer, remote payload MiB and effective
#                           payload GB/s; backward token comm, Dgrad/activation/Wgrad and GIN
#                           re-pull/grad-reduce are reported separately.
#                           Invocation-boundary sync makes this diagnostic only: never quote the
#                           instrumented end-to-end step time. Use ALL_RANKS for cluster bandwidth.
#
# apply-mode backends (ignored in off/observe, which use Megatron's own dispatcher):
# EPLB_ADAPTER=deepep       token dispatch/combine through ElasticBuffer (default `alltoall`).
# EPLB_DEEPEP_HYBRID=1     use ElasticBuffer's NVLink-intra/GIN-inter hybrid path (default 1).
# EPLB_DEEPEP_MAX_TOKENS_PER_RANK=<int> optional static routing-unit bound per chunk.
# EPLB_WEIGHT_COMM=gin      replica expert weights AND their grad reduce-to-main over the in-tree
#                           nccl_gin backend, both directions device-initiated (default:
#                           host-driven dist.broadcast). Independent of the token transport.
#                           Replica weights are never kept across forward->backward: backward
#                           re-pulls them off the schedule cached at plan time, so EPLB_OVERLAP and
#                           EPLB_REMATERIALIZE do not apply -- this path is always the re-pull one.
# EPLB_GIN_FENCE=signal     device-stream barrier (LSA inside the node, GIN rail across) instead of
#                           the default host dist.barrier. Set this for real runs: a host barrier is
#                           not stream-ordered, so the backward re-pull cannot overlap Wgrad.
# EPLB_GIN_LSA=0            force every replica transfer onto GIN's network path even for peers whose
#                           memory is mapped here. Default is to move those over NVLink with
#                           an SM90+ TMA-staged copy; with the EP group inside a node that is the
#                           whole weight channel, so this is for A/B measurement only.
# EPLB_GIN_LSA_TMA=0        keep LSA/NVLink routing but restore vector load/store for A/B or fallback.
# Expert compute is SM90-only ragged grouped GEMM: there is no padded per-slot cap.
# EPLB_CHUNKS=2             split this rank's routing units in two and pipeline dispatch/compute/
#                           combine across a compute and a comm stream, so dispatch(c2) hides behind
#                           compute(c1) and combine(c1) behind compute(c2); only the first dispatch
#                           and last combine stay exposed. Token-side only: the replica weights are
#                           acquired once per layer and shared by the chunks, in both directions.
#                           Composes with gin and with EPLB_OVERLAP=1.
# EPLB_MANUAL_BWD=0         hand back the ordering of the chunked pipeline to autograd (default 1 =
#                           one node for dispatch/GEMM/combine, both directions scheduled by hand).
#                           Numerically identical; under autograd the forward weight pull cannot hide
#                           behind dispatch(c1) and the backward reduce-to-main is forced to run last
#                           in the layer with nothing left to overlap. Keep it at 1 except when
#                           A/B-ing the schedule.
# EPLB_REMATERIALIZE=1      dist.broadcast transport only: free replica weights after forward and
#                           re-broadcast them in backward.
for _knob in EPLB_N_SLOT EPLB_PLAN_SOLVER EPLB_FASTERMOE_BW_NET EPLB_FASTERMOE_BW_MM \
             EPLB_DEEPSEEK_NUM_GROUPS EPLB_FLEXMOE_THRESHOLD EPLB_LPLB_ROOT EPLB_LPLB_TOPOLOGY \
             EPLB_PROFILE EPLB_PROFILE_ALL_RANKS EPLB_PROFILE_EVERY EPLB_PROFILE_RESET_AT \
             EPLB_DEBUG_TIMING \
             EPLB_ADAPTER EPLB_DEEPEP_HYBRID EPLB_DEEPEP_MAX_TOKENS_PER_RANK \
             EPLB_MAX_RECV_ROWS \
             EPLB_WEIGHT_COMM EPLB_GIN_FENCE EPLB_GIN_LSA \
             EPLB_CHUNKS EPLB_MANUAL_BWD EPLB_OVERLAP EPLB_REMATERIALIZE; do
  if [[ -n "${!_knob:-}" ]]; then export "${_knob}"; fi
done
_recv_rows="${EPLB_MAX_RECV_ROWS:-0}"
case "${_recv_rows,,}" in
  auto|[0-9]*) ;;
  *)
    echo "unknown EPLB_MAX_RECV_ROWS=${EPLB_MAX_RECV_ROWS}; expected a non-negative int or 'auto'" >&2
    exit 1
    ;;
esac
if [[ "${_recv_rows,,}" == "auto" ]]; then
  echo "[run_real_moe] EPLB_MAX_RECV_ROWS=auto -> receive budget read from the solved plan;" \
       "costs one D2H sync per layer, so this is not a zero-sync throughput measurement"
fi
unset _recv_rows
if [[ -n "${EPLB_N_SLOT:-}" ]]; then echo "[run_real_moe] EPLB_N_SLOT=${EPLB_N_SLOT} (slot budget override)"; fi
if [[ "${EPLB_DEBUG_TIMING:-0}" != "0" ]]; then
  echo "[run_real_moe] EPLB_DEBUG_TIMING=1 -> per-MoE forward/backward breakdown; throughput is perturbed"
fi
if [[ "${EPLB_MODE}" == "apply" ]]; then
  echo "[run_real_moe] apply backends: plan=${EPLB_PLAN_SOLVER} adapter=${EPLB_ADAPTER:-alltoall} weight_comm=${EPLB_WEIGHT_COMM:-broadcast}" \
       "chunks=${EPLB_CHUNKS:-1} manual_bwd=${EPLB_MANUAL_BWD:-1}" \
       "overlap=${EPLB_OVERLAP:-0} remat=${EPLB_REMATERIALIZE:-0}"
  if [[ "${EPLB_WEIGHT_COMM:-}" == "gin" && "${EPLB_GIN_LSA:-1}" == "0" ]]; then
    echo "[run_real_moe] WARNING: EPLB_GIN_LSA=0 sends intra-node replica traffic to the NIC" \
         "instead of over NVLink. Only do this to measure the difference."
  fi
  if [[ "${EPLB_ADAPTER:-alltoall}" == "deepep" ]]; then
    for _legacy in EPLB_DEEPEP_STATIC EPLB_DEEPEP_ALLOW_MNNVL EPLB_DEEPEP_NVL_BYTES \
                   EPLB_DEEPEP_RDMA_BYTES EPLB_DEEPEP_MAX_RECV; do
      if [[ -v "${_legacy}" ]]; then
        echo "${_legacy} is a removed legacy Buffer setting; ElasticBuffer configures transport itself" >&2
        exit 1
      fi
    done
    [[ "${EPLB_WEIGHT_COMM:-}" == "gin" ]] || {
      echo "zero-sync Elastic mode requires EPLB_WEIGHT_COMM=gin" >&2; exit 1;
    }
    [[ "${EPLB_GIN_FENCE:-}" == "signal" ]] || {
      echo "zero-sync Elastic mode requires EPLB_GIN_FENCE=signal" >&2; exit 1;
    }
    [[ "${EPLB_PROFILE:-0}" == "0" && "${PROFILE_TRACE:-0}" == "0" ]] || {
      echo "zero-sync Elastic mode requires EPLB_PROFILE=0 PROFILE_TRACE=0" >&2; exit 1;
    }
    if [[ "${EPLB_DEBUG_TIMING:-0}" != "0" ]]; then
      echo "[run_real_moe] WARNING: EPLB_DEBUG_TIMING synchronizes each MoE invocation;" \
           "phase times are diagnostic, and this run is not a zero-sync throughput measurement."
    fi
    python - <<'PY'
import deep_ep
import nccl_gin
from eplb.integration.eplb_manager import _nccl_runtime_version
assert hasattr(deep_ep, "ElasticBuffer"), "DeepEP ElasticBuffer is unavailable"
nccl_gin._ensure_loaded()
v = _nccl_runtime_version()
assert v == (2, 30, 7), f"NCCL 2.30.7 runtime required, got {v}"
print(f"[run_real_moe] transport={deep_ep.ElasticBuffer.__module__}.ElasticBuffer "
      f"hybrid={__import__('os').environ.get('EPLB_DEEPEP_HYBRID', '1')} NCCL={v}")
PY
  fi
fi

# PROFILE_TRACE=1 -> Megatron's native PyTorch profiler writes a chrome/perfetto trace per profiled
# rank. The eplb/* record_function labels include native/{route,dispatch,expert_gemm,combine}
# in off/observe and apply/{route,dispatch,expert_compute,expert_gemm,combine,weight_move}
# in apply; observe additionally includes solve and all_gather_omega. Start past step ~5: the first
# iterations pay CUDA-solver JIT and cuBLAS autotuning and are not representative.
#
# PROFILE_DIR is one output dir per run, holding tb/ and torch_profile/. Megatron derives the trace
# path as `<tensorboard-dir>/../torch_profile`, so tb/ is nested one level down to keep the trace
# inside PROFILE_DIR -- otherwise every run resolves to the same logs/torch_profile and each
# overwrites the previous rank-<N>.json.gz. Set PROFILE_DIR per run when comparing modes or sweeps.
PROFILE_ARGS=()
if [[ "${PROFILE_TRACE:-0}" == "1" ]]; then
  PROFILE_RUN_DIR="${PROFILE_DIR:-${EPLB_DIR}/logs/prof_${EPLB_MODE}}"
  PROFILE_ARGS=(
    --profile
    --use-pytorch-profiler
    --profile-step-start "${PROFILE_STEP_START:-8}"
    --profile-step-end "${PROFILE_STEP_END:-10}"
    --profile-ranks ${PROFILE_RANKS:-0}                   # unquoted on purpose: PROFILE_RANKS="0 8" must word-split into nargs
    --tensorboard-dir "${PROFILE_RUN_DIR}/tb"
  )
  # CPU/Python call stacks; off by default because with_stack inflates the trace and slows the
  # profiled steps enough to distort the step time read off the same run.
  [[ "${PROFILE_STACK:-0}" == "1" ]] && PROFILE_ARGS+=(--pytorch-profiler-collect-callstack)
  [[ "${PROFILE_SHAPES:-0}" == "1" ]] && PROFILE_ARGS+=(--pytorch-profiler-collect-shapes)
  echo "[run_real_moe] PROFILE_TRACE=1 -> steps ${PROFILE_STEP_START:-8}..${PROFILE_STEP_END:-10}, ranks ${PROFILE_RANKS:-0}"
  echo "[run_real_moe] trace -> ${PROFILE_RUN_DIR}/torch_profile/rank-<N>.json.gz"
fi

# --- per-model architecture args (must match the checkpoint when loading) -----
source "${SCRIPT_DIR}/model_recipes.sh"
configure_model_recipe "${MODEL}"
echo "[run_real_moe] depth=${MODEL_NUM_LAYERS}/${MODEL_FULL_NUM_LAYERS} layers" \
     "(dense=${MODEL_DENSE_PREFIX_LAYERS}, MoE=${MODEL_MOE_LAYERS}); override with NUM_LAYERS"

# Router load balancing. ROUTER_BALANCING=none turns the aux loss off so the routing skew survives to
# the dispatcher: aux_loss actively flattens the very imbalance an expert load balancer exists to
# exploit, so leaving it on understates every balancer in the comparison. Appended after the per-model
# recipe because argparse keeps the last value for a repeated flag.
ROUTER_BALANCING="${ROUTER_BALANCING:-${MODEL_DEFAULT_ROUTER_BALANCING}}"
MOE_ARGS+=(--moe-router-load-balancing-type "${ROUTER_BALANCING}")
if [[ "${ROUTER_BALANCING}" == "none" ]]; then
  MOE_ARGS+=(--moe-aux-loss-coeff 0.0)
  echo "[run_real_moe] ROUTER_BALANCING=none -> aux loss off, routing skew left intact"
fi
# --moe-expert-capacity-factor is deliberately never set: unset means no token is dropped and every
# routed token reaches its expert, which the load-balancing comparison depends on. Do not add it.

# ROUTER_SKEW=<std> injects a shared per-expert bias ~ N(0, |std|) into the router logits, which is how
# a controlled routing skew is produced without a trained checkpoint: |std| sets the magnitude, so
# sweeping it sweeps the imbalance a balancer has to absorb. std<0 draws the bias once per layer and
# reuses it (stationary skew -- the measurable case); std>0 redraws every forward pass. Megatron also
# randomises the logits themselves here, so the loss is meaningless under this flag: use it for step
# time / imbalance, never for the loss-curve comparison.
if [[ -n "${ROUTER_SKEW:-}" ]]; then
  MOE_ARGS+=(--moe-router-force-biased "${ROUTER_SKEW}")
  [[ "${ROUTER_SKEW}" == -* ]] && _skew_kind="fixed per layer" || _skew_kind="redrawn per step"
  echo "[run_real_moe] ROUTER_SKEW=${ROUTER_SKEW} -> synthetic expert bias (${_skew_kind}); loss is not meaningful"
fi

# SequentialMLP for every mode: the apply-mode binding needs clean per-expert weights, and one shared
# kernel path keeps off/observe/apply step times comparable.
MODEL_ARGS+=(
  --transformer-impl local
  --no-rope-fusion --no-masked-softmax-fusion --no-bias-swiglu-fusion
  --no-gradient-accumulation-fusion --no-persist-layer-norm
)

# DEEPEP=1 (off/observe only): route Megatron's own MoE dispatch through DeepEP (flex backend, uses deep_ep.Buffer).
# apply mode replaces Megatron's dispatcher outright, so this flag does nothing there -- its DeepEP
# transport is selected with EPLB_ADAPTER=deepep instead.
DEEPEP_ARGS=()
if [[ "${DEEPEP:-0}" == "1" ]]; then
  if [[ "${EPLB_MODE}" == "apply" ]]; then
    echo "[run_real_moe] DEEPEP=1 ignored in apply mode (EPLB owns the dispatcher; use EPLB_ADAPTER=deepep)"
  else
    DEEPEP_ARGS=(--moe-token-dispatcher-type flex --moe-enable-deepep --moe-router-dtype fp32)   # overrides the alltoall set in MOE_ARGS (argparse: last wins); DeepEP requires fp32 probs
    echo "[run_real_moe] DEEPEP=1 -> Megatron native DeepEP dispatch (flex)"
  fi
fi

USE_DISTRIBUTED_OPTIMIZER="${USE_DISTRIBUTED_OPTIMIZER:-1}"
if [[ "${USE_DISTRIBUTED_OPTIMIZER}" != "0" && "${USE_DISTRIBUTED_OPTIMIZER}" != "1" ]]; then
  echo "invalid USE_DISTRIBUTED_OPTIMIZER=${USE_DISTRIBUTED_OPTIMIZER} (expected 0 or 1)" >&2
  exit 1
fi
PARALLEL_ARGS=(
  --tensor-model-parallel-size "${TP}"
  --pipeline-model-parallel-size "${PP}"
  --expert-model-parallel-size "${EP}"
  --distributed-backend nccl
)
[[ "${USE_DISTRIBUTED_OPTIMIZER}" == "1" ]] && PARALLEL_ARGS+=(--use-distributed-optimizer)
[[ "${TP}" -gt 1 ]] && PARALLEL_ARGS+=(--sequence-parallel)   # sequence parallel requires TP>1

if [[ "${MOCK}" == "1" ]]; then
  DATA_ARGS=(--mock-data --tokenizer-type NullTokenizer --vocab-size "${VOCAB_SIZE:-32000}")
else
  DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "${TOKENIZER_MODEL}"
    --data-path "${DATA_PATH}"
    --split 99,1,0
  )
fi

TRAIN_ITERS="${TRAIN_ITERS:-50}"
# warmup must be < train-iters (Megatron asserts); default to ~10% capped below the total.
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-$(( TRAIN_ITERS > 10 ? 5 : (TRAIN_ITERS > 1 ? 1 : 0) ))}"
TRAIN_ARGS=(
  --micro-batch-size "${MICRO_BATCH_SIZE:-1}"
  --global-batch-size "${GLOBAL_BATCH_SIZE:-256}"
  --train-iters "${TRAIN_ITERS}"
  --lr 1e-5 --min-lr 1e-6 --lr-decay-style cosine --lr-warmup-iters "${LR_WARMUP_ITERS}"
  --weight-decay 0.1 --clip-grad 1.0
  --bf16
  --log-interval 1 --eval-interval 1000000 --eval-iters 0
  --num-workers "${NUM_WORKERS:-0}"
)

if [[ "${MOCK}" == "1" || "${FROM_SCRATCH}" == "1" ]]; then
  LOAD_ARGS=()                              # random init, no checkpoint
else
  LOAD_ARGS=(--load "${CHECKPOINT}" --no-load-optim --no-load-rng --dist-ckpt-strictness log_unexpected)
fi
[[ -n "${SAVE_DIR}" ]] && LOAD_ARGS+=(--save "${SAVE_DIR}" --save-interval "${SAVE_INTERVAL:-1000}")

DISTRIBUTED_ARGS=(
  --nproc_per_node "${GPUS_PER_NODE}"
  --nnodes "${NNODES}"
  --node_rank "${NODE_RANK}"
  --master_addr "${MASTER_ADDR}"
  --master_port "${MASTER_PORT}"
)

# Tee stdout+stderr to a timestamped per-node log (pipefail keeps torchrun's exit code); override path via LOG_FILE, disable with LOG=0.
LOG="${LOG:-1}"
LOG_ROOT="${EPLB_LOG_DIR:-${EPLB_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/real_${MODEL}_${EPLB_MODE}_node${NODE_RANK}.log}"
echo "[run_real_moe] model=${MODEL} mode=${EPLB_MODE} world=${WORLD_SIZE} TP=${TP} PP=${PP} EP=${EP}"
[[ "${LOG}" != "0" ]] && { mkdir -p "$(dirname "${LOG_FILE}")"; echo "[run_real_moe] logging to ${LOG_FILE}"; }
# Extra Megatron flags passed to this script go last, so argparse lets them override the recipe above
# (e.g. `--recompute-granularity full --recompute-method uniform`). Captured at top level because `$@`
# inside a function would resolve to the function's own arguments.
EXTRA_ARGS=("$@")
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "[run_real_moe] extra Megatron args: ${EXTRA_ARGS[*]}"
run_torchrun() {
  torchrun "${DISTRIBUTED_ARGS[@]}" \
    "${EPLB_DIR}/scripts/pretrain_eplb_moe.py" \
    "${MODEL_ARGS[@]}" "${MOE_ARGS[@]}" "${DEEPEP_ARGS[@]}" "${PARALLEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" "${TRAIN_ARGS[@]}" "${LOAD_ARGS[@]}" "${PROFILE_ARGS[@]}" "${EXTRA_ARGS[@]}"
}
if [[ "${LOG}" != "0" ]]; then
  run_torchrun 2>&1 | tee "${LOG_FILE}"
else
  run_torchrun
fi