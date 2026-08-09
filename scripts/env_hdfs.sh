#!/usr/bin/env bash
# Source this on every node to keep reusable artifacts on the Merlin HDFS mount.

export EPLB_DATA_ROOT="${EPLB_DATA_ROOT:-/mnt/hdfs/__MERLIN_USER_DIR__/eplb_data}"
export EPLB_RAW_DATA_DIR="${EPLB_RAW_DATA_DIR:-${EPLB_DATA_ROOT}/raw}"
export EPLB_INDEXED_DATA_DIR="${EPLB_INDEXED_DATA_DIR:-${EPLB_DATA_ROOT}/indexed}"
export EPLB_TOKENIZER_DIR="${EPLB_TOKENIZER_DIR:-${EPLB_DATA_ROOT}/tokenizers}"
export EPLB_CHECKPOINT_DIR="${EPLB_CHECKPOINT_DIR:-${EPLB_DATA_ROOT}/checkpoints}"
export EPLB_LOG_DIR="${EPLB_LOG_DIR:-${EPLB_DATA_ROOT}/logs}"

# Persist Hugging Face downloads across disposable machines.
export HF_HOME="${HF_HOME:-${EPLB_DATA_ROOT}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if ! mkdir -p \
  "${EPLB_RAW_DATA_DIR}" \
  "${EPLB_INDEXED_DATA_DIR}" \
  "${EPLB_TOKENIZER_DIR}" \
  "${EPLB_CHECKPOINT_DIR}" \
  "${EPLB_LOG_DIR}" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}"; then
  echo "cannot create EPLB data directories under ${EPLB_DATA_ROOT}; check that the HDFS mount is available and writable" >&2
  return 1 2>/dev/null || exit 1
fi
