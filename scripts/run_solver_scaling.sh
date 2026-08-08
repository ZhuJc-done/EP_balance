#!/usr/bin/env bash
# Benchmark fast_solver.cu as logical rank and expert dimensions grow.
set -euo pipefail

EPLB_DIR="${EPLB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-${EPLB_DIR}/logs/solver_scaling}"

# Space-separated sweep values.
RANKS="${RANKS:-8 16 32 64 128}"
EXPERTS="${EXPERTS:-64 128 256 384 512 640 768 1024}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
RANK_SWEEP_EXPERTS="${RANK_SWEEP_EXPERTS:-640}"
EXPERT_SWEEP_RANKS="${EXPERT_SWEEP_RANKS:-32}"
TOKENS_PER_RANK="${TOKENS_PER_RANK:-4096}"
TOP_K="${TOP_K:-8}"
EXTRA_SLOTS="${EXTRA_SLOTS:-2}"
WARMUP="${WARMUP:-20}"
ITERATIONS="${ITERATIONS:-200}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

read -r -a RANK_VALUES <<< "${RANKS}"
read -r -a EXPERT_VALUES <<< "${EXPERTS}"

mkdir -p "${OUT_DIR}"
cd "${EPLB_DIR}"

CURRENT_TMP=""
cleanup() {
  if [[ -n "${CURRENT_TMP}" ]]; then
    rm -f "${CURRENT_TMP}"
  fi
}
trap cleanup EXIT

run_case() {
  local sweep="$1"
  local ranks="$2"
  local experts="$3"

  if (( ranks % GPUS_PER_NODE != 0 )); then
    echo "logical ranks (${ranks}) must be divisible by GPUS_PER_NODE (${GPUS_PER_NODE})" >&2
    exit 1
  fi
  if (( experts % ranks != 0 )); then
    echo "experts (${experts}) must be divisible by logical ranks (${ranks})" >&2
    exit 1
  fi

  local nodes=$((ranks / GPUS_PER_NODE))
  local output="${OUT_DIR}/${sweep}_r${ranks}_e${experts}.json"
  CURRENT_TMP="${output}.tmp"
  echo "[solver-scaling] ${sweep}: R=${ranks} E=${experts} -> ${output}"

  "${PYTHON_BIN}" tests/test_gpu_solver.py \
    --nodes "${nodes}" \
    --gpus-per-node "${GPUS_PER_NODE}" \
    --experts "${experts}" \
    --tokens-per-rank "${TOKENS_PER_RANK}" \
    --top-k "${TOP_K}" \
    --extra-slots "${EXTRA_SLOTS}" \
    --warmup "${WARMUP}" \
    --iterations "${ITERATIONS}" \
    --cuda-device "${CUDA_DEVICE}" \
    --json > "${CURRENT_TMP}"

  "${PYTHON_BIN}" -m json.tool "${CURRENT_TMP}" >/dev/null
  mv "${CURRENT_TMP}" "${output}"
  CURRENT_TMP=""
}

for ranks in "${RANK_VALUES[@]}"; do
  run_case "rank_scale" "${ranks}" "${RANK_SWEEP_EXPERTS}"
done

for experts in "${EXPERT_VALUES[@]}"; do
  run_case "expert_scale" "${EXPERT_SWEEP_RANKS}" "${experts}"
done

echo "[solver-scaling] completed $(( ${#RANK_VALUES[@]} + ${#EXPERT_VALUES[@]} )) runs"
echo "[solver-scaling] JSON results saved in ${OUT_DIR}"
