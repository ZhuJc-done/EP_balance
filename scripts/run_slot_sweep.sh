#!/usr/bin/env bash
# Sweep the per-rank replica-slot budget and save one benchmark JSON per seed/slot.
set -euo pipefail

EPLB_DIR="${EPLB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-${EPLB_DIR}/logs/slot_sweep}"

# Space-separated lists; override with e.g. SLOTS="1 2" SEEDS="0 1 2".
SLOTS="${SLOTS:-1 2 3 4}"
SEEDS="${SEEDS:-0}"

STRATEGIES="${STRATEGIES:-scale,eplb,fastermoe,flexmoe,lplb}"
NODES="${NODES:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
EXPERTS="${EXPERTS:-640}"
TOKENS_PER_RANK="${TOKENS_PER_RANK:-4096}"
TOP_K="${TOP_K:-8}"
SKEW="${SKEW:-1.5}"
HISTORY_SKEW="${HISTORY_SKEW:-1.3}"
HISTORY_SEED_OFFSET="${HISTORY_SEED_OFFSET:-1000}"
HOTSPOT_RANKS="${HOTSPOT_RANKS:-0.25}"

# Warm up every solver before timing so lazy CUDA compilation/loading is excluded.
# The JSON ``solve_ms`` field is the hot solver-only average for every strategy.
WARMUP="${WARMUP:-5}"
ITERATIONS="${ITERATIONS:-20}"
MAPPING_ITERATIONS="${MAPPING_ITERATIONS:-3}"

# Ring has one topology edge per rank and therefore supports every N_slot=1..4.
# Keeping it fixed also prevents topology changes from confounding the slot scan.
LPLB_TOPOLOGY="${LPLB_TOPOLOGY:-ring}"
LPLB_ROOT="${LPLB_ROOT:-/home/tiger/LPLB}"

read -r -a SLOT_VALUES <<< "${SLOTS}"
read -r -a SEED_VALUES <<< "${SEEDS}"

mkdir -p "${OUT_DIR}"
cd "${EPLB_DIR}"

CURRENT_TMP=""
cleanup() {
  if [[ -n "${CURRENT_TMP}" ]]; then
    rm -f "${CURRENT_TMP}"
  fi
}
trap cleanup EXIT

for seed in "${SEED_VALUES[@]}"; do
  history_seed=$((seed + HISTORY_SEED_OFFSET))
  for slot in "${SLOT_VALUES[@]}"; do
    output="${OUT_DIR}/baseline_skew${SKEW}_slot${slot}_seed${seed}.json"
    CURRENT_TMP="${output}.tmp"
    echo "[slot-sweep] seed=${seed} slot=${slot} -> ${output}"

    "${PYTHON_BIN}" -m baseline.benchmark \
      --strategies "${STRATEGIES}" \
      --nodes "${NODES}" \
      --gpus-per-node "${GPUS_PER_NODE}" \
      --experts "${EXPERTS}" \
      --n-slot "${slot}" \
      --tokens-per-rank "${TOKENS_PER_RANK}" \
      --top-k "${TOP_K}" \
      --skew "${SKEW}" \
      --history-skew "${HISTORY_SKEW}" \
      --hotspot-ranks "${HOTSPOT_RANKS}" \
      --seed "${seed}" \
      --history-seed "${history_seed}" \
      --warmup "${WARMUP}" \
      --iterations "${ITERATIONS}" \
      --mapping-iterations "${MAPPING_ITERATIONS}" \
      --include-no-balance \
      --solver-only-timing \
      --lplb-root "${LPLB_ROOT}" \
      --lplb-topology "${LPLB_TOPOLOGY}" \
      --require-lplb \
      --json > "${CURRENT_TMP}"

    "${PYTHON_BIN}" -m json.tool "${CURRENT_TMP}" >/dev/null
    mv "${CURRENT_TMP}" "${output}"
    CURRENT_TMP=""
  done
done

echo "[slot-sweep] completed $(( ${#SEED_VALUES[@]} * ${#SLOT_VALUES[@]} )) runs in ${OUT_DIR}"
