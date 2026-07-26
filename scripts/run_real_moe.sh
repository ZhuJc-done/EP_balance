#!/usr/bin/env bash
# Real-model launcher: train an open MoE (Mixtral-8x7B / Qwen3-30B-A3B) on real data, multi-node, EPLB_MODE off|observe|apply.
set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1

# DeepEP's extension links libnccl.so from the nvidia-nccl wheel (install_deepep.sh); put it on the
# runtime linker path so `import deep_ep` resolves. No-op if the wheel isn't installed.
_nccl_lib="$(python -c 'import nvidia.nccl as n,os;print(os.path.join(n.__path__[0],"lib"))' 2>/dev/null || true)"
if [ -n "${_nccl_lib}" ] && [ -d "${_nccl_lib}" ]; then
  export LD_LIBRARY_PATH="${_nccl_lib}:${LD_LIBRARY_PATH:-}"
fi

# --- required paths / artifacts ----------------------------------------------
MEGATRON_DIR="${MEGATRON_DIR:?set MEGATRON_DIR to the Megatron-LM repo root}"
EPLB_DIR="${EPLB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# MOCK=1: real model architecture, but mock-data + random init (no checkpoint/data/tokenizer needed) --
# a pure-Megatron "does it run / throughput / memory" baseline. MOCK=0 (default) needs real artifacts.
MOCK="${MOCK:-0}"
if [[ "${MOCK}" == "1" ]]; then
  CHECKPOINT="" ; DATA_PATH="" ; TOKENIZER_MODEL=""
else
  CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to the mcore checkpoint dir (from convert_hf_to_mcore.sh / Megatron Bridge)}"
  DATA_PATH="${DATA_PATH:?set DATA_PATH to the preprocessed data prefix (from prepare_data.sh, no .bin/.idx suffix)}"
  TOKENIZER_MODEL="${TOKENIZER_MODEL:?set TOKENIZER_MODEL to the HF repo/dir matching the checkpoint}"
fi
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

# --- optional routing-trace capture (observe mode) ---------------------------
# EPLB_TRACE_OUT=<path> makes rank 0 dump the real gathered Lambda[R,E] per
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
# EPLB_PROFILE=1            periodic per-region latency + peak-memory summary. CUDA events are queued and
#                           resolved in one batch, so this adds no per-region host sync.
# EPLB_PROFILE_ALL_RANKS=1  every rank prints its own summary (required for max-vs-mean straggler analysis).
# EPLB_PROFILE_EVERY=<n>    summary cadence; EPLB_PROFILE_RESET_AT=<n> resets peak memory after warmup.
for _knob in EPLB_N_SLOT EPLB_PROFILE EPLB_PROFILE_ALL_RANKS EPLB_PROFILE_EVERY EPLB_PROFILE_RESET_AT; do
  if [[ -n "${!_knob:-}" ]]; then export "${_knob}"; fi
done
if [[ -n "${EPLB_N_SLOT:-}" ]]; then echo "[run_real_moe] EPLB_N_SLOT=${EPLB_N_SLOT} (slot budget override)"; fi

# PROFILE_TRACE=1 -> Megatron's native PyTorch profiler writes a chrome/perfetto trace to
# ${PROFILE_DIR}/../torch_profile/rank-<N>.json.gz. The eplb/* record_function labels (solve,
# all_gather_lambda, apply/dispatch, apply/expert_compute, apply/combine, apply/weight_move) appear
# inline, so the trace resolves where EP time actually goes. Start past step ~5: the first iterations
# pay CUDA-solver JIT and cuBLAS autotuning and are not representative.
PROFILE_ARGS=()
if [[ "${PROFILE_TRACE:-0}" == "1" ]]; then
  PROFILE_ARGS=(
    --profile
    --use-pytorch-profiler
    --profile-step-start "${PROFILE_STEP_START:-8}"
    --profile-step-end "${PROFILE_STEP_END:-10}"
    --profile-ranks ${PROFILE_RANKS:-0}                   # unquoted on purpose: PROFILE_RANKS="0 8" must word-split into nargs
    --tensorboard-dir "${PROFILE_DIR:-${EPLB_DIR}/logs/tb_${EPLB_MODE}}"
  )
  # CPU/Python call stacks; off by default because with_stack inflates the trace and slows the
  # profiled steps enough to distort the step time read off the same run.
  [[ "${PROFILE_STACK:-0}" == "1" ]] && PROFILE_ARGS+=(--pytorch-profiler-collect-callstack)
  [[ "${PROFILE_SHAPES:-0}" == "1" ]] && PROFILE_ARGS+=(--pytorch-profiler-collect-shapes)
  echo "[run_real_moe] PROFILE_TRACE=1 -> steps ${PROFILE_STEP_START:-8}..${PROFILE_STEP_END:-10}, ranks ${PROFILE_RANKS:-0}"
fi

# --- per-model architecture args (must match the checkpoint config) -----------
if [[ "${MODEL}" == "mixtral8x7b" ]]; then
  MODEL_ARGS=(
    --use-mcore-models --disable-bias-linear --untie-embeddings-and-output-weights
    --seq-length "${SEQ_LEN:-4096}" --max-position-embeddings 32768
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
    --seq-length "${SEQ_LEN:-8192}" --max-position-embeddings 8192
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

# Router load balancing. ROUTER_BALANCING=none turns the aux loss off so the routing skew survives to
# the dispatcher: aux_loss actively flattens the very imbalance an expert load balancer exists to
# exploit, so leaving it on understates every balancer in the comparison. Appended after the per-model
# recipe because argparse keeps the last value for a repeated flag.
ROUTER_BALANCING="${ROUTER_BALANCING:-aux_loss}"
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
# In apply mode EPLB replaces the dispatcher, so this is ignored there (DeepEPAdapter is not wired yet).
DEEPEP_ARGS=()
if [[ "${DEEPEP:-0}" == "1" ]]; then
  if [[ "${EPLB_MODE}" == "apply" ]]; then
    echo "[run_real_moe] DEEPEP=1 ignored in apply mode (EPLB owns the dispatcher; DeepEPAdapter not wired yet)"
  else
    DEEPEP_ARGS=(--moe-token-dispatcher-type flex --moe-enable-deepep --moe-router-dtype fp32)   # overrides the alltoall set in MOE_ARGS (argparse: last wins); DeepEP requires fp32 probs
    echo "[run_real_moe] DEEPEP=1 -> Megatron native DeepEP dispatch (flex)"
  fi
fi

PARALLEL_ARGS=(
  --tensor-model-parallel-size "${TP}"
  --pipeline-model-parallel-size "${PP}"
  --expert-model-parallel-size "${EP}"
  --use-distributed-optimizer
  --distributed-backend nccl
)
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

if [[ "${MOCK}" == "1" ]]; then
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
LOG_FILE="${LOG_FILE:-${EPLB_DIR}/logs/real_${MODEL}_${EPLB_MODE}_node${NODE_RANK}.log}"
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
