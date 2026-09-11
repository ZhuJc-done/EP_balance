#!/usr/bin/env bash
# Configurable capture engine for DAPO-Math and StarCoder rank dynamics.
# Defaults retain the original 4-node x 8-GPU setup; use run_rank_dynamics_4gpu.sh
# for a single-node capture followed by virtual EP32 analysis.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EPLB_DIR="${EPLB_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
source "${EPLB_DIR}/scripts/env_hdfs.sh"

MEGATRON_DIR="${MEGATRON_DIR:-${HOME}/Megatron-LM}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-${EPLB_TOKENIZER_DIR}/qwen3_30b_a3b}"
MODEL="${MODEL:-qwen3_30b_a3b}"
case "${MODEL}" in
  qwen3_30b_a3b|glm45_air) NUM_EXPERTS_FOR_MAPPING=128 ;;
  deepseek_v2_160e) NUM_EXPERTS_FOR_MAPPING=160 ;;
  *)
    echo "unknown MODEL=${MODEL}" >&2
    exit 1
    ;;
esac
SEED="${SEED:-1234}"
NUM_LAYERS="${NUM_LAYERS:-48}"
SEQ_LEN="${SEQ_LEN:-4096}"
EVAL_ITERS="${EVAL_ITERS:-16}"
WORKERS="${WORKERS:-16}"
FORCE_DATA="${FORCE_DATA:-0}"
OVERWRITE="${OVERWRITE:-0}"
RANK_DYNAMICS_DATASET="${RANK_DYNAMICS_DATASET:-both}"
case "${RANK_DYNAMICS_DATASET}" in
  dapo_math|starcoder|both) ;;
  *)
    echo "unknown RANK_DYNAMICS_DATASET=${RANK_DYNAMICS_DATASET}; expected dapo_math, starcoder, or both" >&2
    exit 1
    ;;
esac

GPUS_PER_NODE="${GPUS_PER_NODE:-${ARNOLD_WORKER_GPU:-8}}"
NNODES="${NNODES:-${SLURM_NNODES:-${ARNOLD_WORKER_NUM:-4}}}"
NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-${ARNOLD_ID:-}}}"
if [[ -z "${NODE_RANK}" ]]; then
  if [[ "${NNODES}" == "1" ]]; then
    NODE_RANK=0
  else
    echo "cannot determine NODE_RANK; set it to 0..$((NNODES - 1)) on each node" >&2
    exit 1
  fi
fi

MASTER_ADDR="${MASTER_ADDR:-}"
if [[ -z "${MASTER_ADDR}" && -n "${SLURM_JOB_NODELIST:-}" ]]; then
  MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | awk 'NR == 1 {print; exit}')"
fi
MASTER_ADDR="${MASTER_ADDR:-${ARNOLD_WORKER_0_HOST:-}}"
MASTER_PORT="${MASTER_PORT:-13401}"

WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
EXPECTED_WORLD_SIZE="${EXPECTED_WORLD_SIZE:-32}"
CAPTURE_EP="${CAPTURE_EP:-${WORLD_SIZE}}"
TARGET_RANKS="${TARGET_RANKS:-32}"
OCCURRENCE_GROUP="${OCCURRENCE_GROUP:-1}"
CAPTURE_GLOBAL_BATCH_SIZE="${CAPTURE_GLOBAL_BATCH_SIZE:-${WORLD_SIZE}}"
if (( WORLD_SIZE != EXPECTED_WORLD_SIZE )); then
  echo "expected ${EXPECTED_WORLD_SIZE} GPUs, got NNODES=${NNODES} x GPUS_PER_NODE=${GPUS_PER_NODE}" >&2
  exit 1
fi
if (( CAPTURE_EP <= 0 || WORLD_SIZE % CAPTURE_EP != 0 )); then
  echo "CAPTURE_EP=${CAPTURE_EP} must divide WORLD_SIZE=${WORLD_SIZE}" >&2
  exit 1
fi
if (( TARGET_RANKS <= 0 || NUM_EXPERTS_FOR_MAPPING % TARGET_RANKS != 0 )); then
  echo "TARGET_RANKS=${TARGET_RANKS} must divide the model's ${NUM_EXPERTS_FOR_MAPPING} experts" >&2
  exit 1
fi
if (( OCCURRENCE_GROUP <= 0 || EVAL_ITERS % OCCURRENCE_GROUP != 0 )); then
  echo "EVAL_ITERS=${EVAL_ITERS} must be divisible by positive OCCURRENCE_GROUP=${OCCURRENCE_GROUP}" >&2
  exit 1
fi
if (( NODE_RANK < 0 || NODE_RANK >= NNODES )); then
  echo "NODE_RANK=${NODE_RANK} is outside 0..$((NNODES - 1))" >&2
  exit 1
fi
if [[ -z "${MASTER_ADDR}" ]]; then
  echo "cannot determine MASTER_ADDR; set it to the node-0 IPv4/IPv6 address" >&2
  exit 1
fi
if [[ ! -f "${MEGATRON_DIR}/tools/preprocess_data.py" ]]; then
  echo "Megatron preprocess tool not found under ${MEGATRON_DIR}" >&2
  exit 1
fi
if [[ ! -f "${TOKENIZER_MODEL}/tokenizer_config.json" ]]; then
  echo "Qwen tokenizer not found under ${TOKENIZER_MODEL}" >&2
  exit 1
fi

first_existing_file() {
  local candidate
  for candidate in "$@"; do
    if [[ -s "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

DAPO_RAW="${DAPO_RAW:-}"
if [[ -z "${DAPO_RAW}" ]]; then
  DAPO_RAW="$(first_existing_file \
    "${EPLB_RAW_DATA_DIR}/mixed_1b_components/dapo_math.jsonl" \
    "${EPLB_RAW_DATA_DIR}/dapo_math.jsonl")" || {
      echo "DAPO-Math raw JSONL is missing; run scripts/prepare_mixed_1b.sh first" >&2
      exit 1
    }
fi
STARCODER_RAW="${STARCODER_RAW:-}"
if [[ -z "${STARCODER_RAW}" ]]; then
  STARCODER_RAW="$(first_existing_file \
    "${EPLB_RAW_DATA_DIR}/mixed_1b_components/starcoder.jsonl" \
    "${EPLB_RAW_DATA_DIR}/starcoder.jsonl")" || {
      echo "StarCoder raw JSONL is missing; prepare or download StarCoderData first" >&2
      exit 1
    }
fi

if [[ -z "${TOKEN_BUDGET:-}" ]]; then
  DAPO_MANIFEST="${DAPO_RAW%.jsonl}.manifest.json"
  if [[ ! -s "${DAPO_MANIFEST}" ]]; then
    echo "cannot infer an equal token budget: missing ${DAPO_MANIFEST}" >&2
    exit 1
  fi
  TOKEN_BUDGET="$(
    python - "${DAPO_MANIFEST}" <<'PY'
import json
import sys

tokens = int(json.load(open(sys.argv[1], encoding="utf-8"))["accepted_tokens"])
if tokens <= 0:
    raise ValueError("accepted_tokens must be positive")
print(tokens)
PY
  )"
fi

DATASET_ROOT="${EPLB_DATA_ROOT}/rank_dynamics_${TOKEN_BUDGET}"
SUBSET_DIR="${DATASET_ROOT}/raw"
INDEX_DIR="${DATASET_ROOT}/indexed"
DAPO_JSONL="${SUBSET_DIR}/dapo_math.jsonl"
STARCODER_JSONL="${SUBSET_DIR}/starcoder.jsonl"
DAPO_PREFIX="${INDEX_DIR}/dapo_math"
STARCODER_PREFIX="${INDEX_DIR}/starcoder"
DAPO_DATA_PATH="${DAPO_PREFIX}_text_document"
STARCODER_DATA_PATH="${STARCODER_PREFIX}_text_document"

RUN_TAG="${RUN_TAG:-scratch_seed${SEED}_${ARNOLD_TRIAL_ID:-manual}}"
RUN_DIR="${EPLB_EXP_DIR}/rank_dynamics/${RUN_TAG}"
DAPO_TRACE="${RUN_DIR}/dapo_math.pt"
STARCODER_TRACE="${RUN_DIR}/starcoder.pt"
FIGURE="${RUN_DIR}/rank_dynamics.pdf"
mkdir -p "${RUN_DIR}"

index_is_current() {
  local prefix="$1"
  local source_jsonl="$2"
  [[ -s "${prefix}_text_document.bin" && -s "${prefix}_text_document.idx" \
     && "${prefix}_text_document.bin" -nt "${source_jsonl}" \
     && "${prefix}_text_document.idx" -nt "${source_jsonl}" ]]
}

preprocess_jsonl() {
  local source_jsonl="$1"
  local prefix="$2"
  if [[ "${FORCE_DATA}" != "1" ]] && index_is_current "${prefix}" "${source_jsonl}"; then
    echo "[rank-dynamics] reuse indexed data: ${prefix}_text_document"
    return
  fi
  rm -f "${prefix}_text_document.bin" "${prefix}_text_document.idx"
  PYTHONPATH="${MEGATRON_DIR}:${PYTHONPATH:-}" \
    python "${MEGATRON_DIR}/tools/preprocess_data.py" \
      --input "${source_jsonl}" \
      --json-keys text \
      --tokenizer-type HuggingFaceTokenizer \
      --tokenizer-model "${TOKENIZER_MODEL}" \
      --output-prefix "${prefix}" \
      --append-eod \
      --workers "${WORKERS}"
}

prepare_data() {
  mkdir -p "${SUBSET_DIR}" "${INDEX_DIR}"
  local force_args=()
  [[ "${FORCE_DATA}" == "1" ]] && force_args+=(--force)
  python "${SCRIPT_DIR}/prepare_rank_dynamics_data.py" \
    --input "${DAPO_RAW}" \
    --output "${DAPO_JSONL}" \
    --tokenizer-model "${TOKENIZER_MODEL}" \
    --token-budget "${TOKEN_BUDGET}" \
    "${force_args[@]}"
  python "${SCRIPT_DIR}/prepare_rank_dynamics_data.py" \
    --input "${STARCODER_RAW}" \
    --output "${STARCODER_JSONL}" \
    --tokenizer-model "${TOKENIZER_MODEL}" \
    --token-budget "${TOKEN_BUDGET}" \
    "${force_args[@]}"
  preprocess_jsonl "${DAPO_JSONL}" "${DAPO_PREFIX}"
  preprocess_jsonl "${STARCODER_JSONL}" "${STARCODER_PREFIX}"
}

PREPARE_KEY="${RUN_TAG//[^[:alnum:]_.-]/_}"
READY_FILE="${INDEX_DIR}/.ready_${PREPARE_KEY}"
FAILED_FILE="${INDEX_DIR}/.failed_${PREPARE_KEY}"
if (( NODE_RANK == 0 )); then
  rm -f "${READY_FILE}" "${FAILED_FILE}"
  if prepare_data; then
    {
      echo "TOKEN_BUDGET=${TOKEN_BUDGET}"
      echo "DAPO_DATA_PATH=${DAPO_DATA_PATH}"
      echo "STARCODER_DATA_PATH=${STARCODER_DATA_PATH}"
    } > "${READY_FILE}"
  else
    status=$?
    echo "rank-0 data preparation failed with status ${status}" > "${FAILED_FILE}"
    exit "${status}"
  fi
else
  timeout_seconds="${DATA_WAIT_TIMEOUT:-3600}"
  deadline=$((SECONDS + timeout_seconds))
  echo "[rank-dynamics] node ${NODE_RANK} waiting for rank-0 data preparation"
  while [[ ! -s "${READY_FILE}" ]]; do
    if [[ -s "${FAILED_FILE}" ]]; then
      cat "${FAILED_FILE}" >&2
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      echo "timed out after ${timeout_seconds}s waiting for ${READY_FILE}" >&2
      exit 1
    fi
    sleep 5
  done
fi

for data_path in "${DAPO_DATA_PATH}" "${STARCODER_DATA_PATH}"; do
  if [[ ! -s "${data_path}.bin" || ! -s "${data_path}.idx" ]]; then
    echo "indexed data is incomplete: ${data_path}.{bin,idx}" >&2
    exit 1
  fi
done

if (( NODE_RANK == 0 )); then
  {
    echo "INIT_MODE=random"
    echo "SEED=${SEED}"
    echo "MODEL=${MODEL}"
    echo "RANK_DYNAMICS_DATASET=${RANK_DYNAMICS_DATASET}"
    echo "NUM_LAYERS=${NUM_LAYERS}"
    echo "WORLD_SIZE=${WORLD_SIZE}"
    echo "CAPTURE_EP=${CAPTURE_EP}"
    echo "TARGET_RANKS=${TARGET_RANKS}"
    echo "OCCURRENCE_GROUP=${OCCURRENCE_GROUP}"
    echo "GLOBAL_BATCH_SIZE=${CAPTURE_GLOBAL_BATCH_SIZE}"
    echo "SEQ_LEN=${SEQ_LEN}"
    echo "EVAL_ITERS=${EVAL_ITERS}"
    echo "TOKEN_BUDGET=${TOKEN_BUDGET}"
    echo "DAPO_RAW=${DAPO_RAW}"
    echo "STARCODER_RAW=${STARCODER_RAW}"
  } > "${RUN_DIR}/run.env"
fi

run_capture() {
  local workload="$1"
  local data_path="$2"
  local trace_out="$3"
  local port="$4"
  echo "[rank-dynamics] starting ${workload} on node ${NODE_RANK}"
  WORKLOAD="${workload}" \
  MEGATRON_DIR="${MEGATRON_DIR}" \
  EPLB_DIR="${EPLB_DIR}" \
  DATA_PATH="${data_path}" \
  TOKENIZER_MODEL="${TOKENIZER_MODEL}" \
  MODEL="${MODEL}" \
  NUM_LAYERS="${NUM_LAYERS}" \
  FROM_SCRATCH=1 \
  NNODES="${NNODES}" \
  GPUS_PER_NODE="${GPUS_PER_NODE}" \
  NODE_RANK="${NODE_RANK}" \
  MASTER_ADDR="${MASTER_ADDR}" \
  MASTER_PORT="${port}" \
  TP=1 \
  PP=1 \
  EP="${CAPTURE_EP}" \
  MICRO_BATCH_SIZE=1 \
  GLOBAL_BATCH_SIZE="${CAPTURE_GLOBAL_BATCH_SIZE}" \
  SEQ_LEN="${SEQ_LEN}" \
  EVAL_ITERS="${EVAL_ITERS}" \
  TRACE_OUT="${trace_out}" \
  TRACE_FLUSH_EVERY="${NUM_LAYERS}" \
  OVERWRITE="${OVERWRITE}" \
  ROUTER_SKEW= \
  USE_DISTRIBUTED_OPTIMIZER=0 \
  EPLB_PROFILE=0 \
  PROFILE_TRACE=0 \
  EPLB_DEBUG_TIMING=0 \
  NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
  LOG_FILE="${RUN_DIR}/${workload}_node${NODE_RANK}.log" \
    bash "${SCRIPT_DIR}/run_expert_hotspot.sh" \
      --seed "${SEED}" \
      --moe-router-dtype fp32 \
      --disable-gloo-process-groups
}

echo "[rank-dynamics] random initialization, no checkpoint, no optimizer updates"
echo "[rank-dynamics] world=${WORLD_SIZE} (${NNODES}x${GPUS_PER_NODE}) capture_EP=${CAPTURE_EP} master=[${MASTER_ADDR}]:${MASTER_PORT}"
echo "[rank-dynamics] selected dataset=${RANK_DYNAMICS_DATASET}"
echo "[rank-dynamics] virtual ranks=${TARGET_RANKS}, occurrence group=${OCCURRENCE_GROUP}"
echo "[rank-dynamics] equal corpus budget=${TOKEN_BUDGET}, raw/grouped occurrences=${EVAL_ITERS}/$((EVAL_ITERS / OCCURRENCE_GROUP))"
case "${RANK_DYNAMICS_DATASET}" in
  dapo_math)
    run_capture dapo_math "${DAPO_DATA_PATH}" "${DAPO_TRACE}" "${MASTER_PORT}"
    ;;
  starcoder)
    run_capture starcoder "${STARCODER_DATA_PATH}" "${STARCODER_TRACE}" "$((MASTER_PORT + 1))"
    ;;
  both)
    run_capture dapo_math "${DAPO_DATA_PATH}" "${DAPO_TRACE}" "${MASTER_PORT}"
    run_capture starcoder "${STARCODER_DATA_PATH}" "${STARCODER_TRACE}" "$((MASTER_PORT + 1))"
    ;;
esac

if (( NODE_RANK == 0 )); then
  if [[ -s "${DAPO_TRACE}" && -s "${STARCODER_TRACE}" ]]; then
    plot_args=(
      --trace "DAPO-Math=${DAPO_TRACE}"
      --trace "StarCoderData=${STARCODER_TRACE}"
      --max-occurrences "${EVAL_ITERS}"
      --target-ranks "${TARGET_RANKS}"
      --occurrence-group "${OCCURRENCE_GROUP}"
      --output "${FIGURE}"
    )
    if [[ -n "${REPRESENTATIVE_LAYER:-}" ]]; then
      plot_args+=(--layer "${REPRESENTATIVE_LAYER}")
    fi
    PYTHONPATH="${EPLB_DIR}:${PYTHONPATH:-}" \
      python "${SCRIPT_DIR}/plot_rank_dynamics.py" "${plot_args[@]}"
    echo "[rank-dynamics] both traces complete: ${RUN_DIR}"
  else
    echo "[rank-dynamics] ${RANK_DYNAMICS_DATASET} capture complete: ${RUN_DIR}"
    echo "[rank-dynamics] joint plot deferred until dapo_math.pt and starcoder.pt both exist"
  fi
fi
