#!/usr/bin/env bash
# Re-index an existing mixed JSONL with model-specific tokenizers.
#
# This intentionally does not call prepare_mixed_1b.sh: that script chooses
# documents to meet a tokenizer-specific budget and can rewrite raw artifacts.
# Here the raw documents are read-only and each model receives its own indexed
# DATA_PATH under indexed/<source>_<model>_text_document.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env_hdfs.sh"

MEGATRON_DIR="${MEGATRON_DIR:-${HOME}/Megatron-LM}"
SOURCE_NAME="${SOURCE_NAME:-mixed_1b}"
SOURCE_JSONL="${SOURCE_JSONL:-${EPLB_RAW_DATA_DIR}/${SOURCE_NAME}.jsonl}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-${EPLB_RAW_DATA_DIR}/${SOURCE_NAME}.manifest.json}"
MODELS="${MODELS:-deepseek_v2_160e glm45_air}"
WORKERS="${WORKERS:-16}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

[[ "${FORCE}" == "0" || "${FORCE}" == "1" ]] || {
  echo "invalid FORCE=${FORCE} (expected 0 or 1)" >&2
  exit 1
}
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || {
  echo "invalid DRY_RUN=${DRY_RUN} (expected 0 or 1)" >&2
  exit 1
}
[[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "invalid WORKERS=${WORKERS} (expected a positive integer)" >&2
  exit 1
}
[[ -s "${SOURCE_JSONL}" ]] || {
  echo "raw input not found or empty: ${SOURCE_JSONL}" >&2
  exit 1
}
[[ -f "${MEGATRON_DIR}/tools/preprocess_data.py" ]] || {
  echo "Megatron preprocess tool not found under ${MEGATRON_DIR}" >&2
  exit 1
}

mkdir -p "${EPLB_INDEXED_DATA_DIR}"

tokenizer_for_model() {
  case "${1}" in
    deepseek_v2_160e | glm45_air)
      printf '%s/%s\n' "${EPLB_TOKENIZER_DIR}" "${1}"
      ;;
    *)
      echo "unsupported model=${1} (expected deepseek_v2_160e or glm45_air)" >&2
      return 1
      ;;
  esac
}

validate_index() {
  local data_path="${1:?data path is required}"
  PYTHONPATH="${MEGATRON_DIR}:${PYTHONPATH:-}" python - "${data_path}" <<'PY'
import sys

from megatron.core.datasets.indexed_dataset import IndexedDataset

dataset = IndexedDataset(sys.argv[1], mmap=True)
tokens = int(dataset.sequence_lengths.sum())
if not len(dataset) or not tokens:
    raise RuntimeError("indexed dataset is empty")
print(f"[reindex] indexed documents={len(dataset)} tokens={tokens} DATA_PATH={sys.argv[1]}")
PY
}

write_metadata() {
  local model="${1:?model is required}"
  local tokenizer="${2:?tokenizer is required}"
  local data_path="${3:?data path is required}"
  local metadata_path="${4:?metadata path is required}"
  MODEL_NAME="${model}" TOKENIZER_PATH="${tokenizer}" DATA_PATH_VALUE="${data_path}" \
    SOURCE_JSONL_VALUE="${SOURCE_JSONL}" SOURCE_MANIFEST_VALUE="${SOURCE_MANIFEST}" \
    METADATA_PATH="${metadata_path}" python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

source_manifest = Path(os.environ["SOURCE_MANIFEST_VALUE"])
metadata = {
    "model": os.environ["MODEL_NAME"],
    "tokenizer": os.environ["TOKENIZER_PATH"],
    "raw_jsonl": os.environ["SOURCE_JSONL_VALUE"],
    "source_manifest": str(source_manifest) if source_manifest.is_file() else None,
    "data_path": os.environ["DATA_PATH_VALUE"],
    "created_at": datetime.now(timezone.utc).isoformat(),
}
Path(os.environ["METADATA_PATH"]).write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
}

reindex_model() {
  local model="${1:?model is required}"
  local tokenizer
  tokenizer="$(tokenizer_for_model "${model}")"
  local output_prefix="${EPLB_INDEXED_DATA_DIR}/${SOURCE_NAME}_${model}"
  local data_path="${output_prefix}_text_document"
  local final_bin="${data_path}.bin"
  local final_idx="${data_path}.idx"
  local metadata_path="${output_prefix}.reindex.json"

  [[ -f "${tokenizer}/tokenizer_config.json" ]] || {
    echo "tokenizer not found under ${tokenizer}" >&2
    return 1
  }

  if [[ "${FORCE}" == "0" && -s "${final_bin}" && -s "${final_idx}" && -s "${metadata_path}" ]]; then
    echo "[reindex] reuse model=${model} DATA_PATH=${data_path}"
    return 0
  fi
  if [[ "${FORCE}" == "0" && ( -e "${final_bin}" || -e "${final_idx}" || -e "${metadata_path}" ) ]]; then
    echo "[reindex] incomplete or untracked output for ${model}: ${output_prefix}" >&2
    echo "          inspect it, then rerun with FORCE=1 to replace only this model's output." >&2
    return 1
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[reindex] would index ${SOURCE_JSONL} with ${tokenizer}"
    echo "[reindex] would write DATA_PATH=${data_path}"
    return 0
  fi

  local work_prefix="${output_prefix}.tmp.$$.${RANDOM}"
  local work_data_path="${work_prefix}_text_document"
  echo "[reindex] model=${model} tokenizer=${tokenizer}"
  echo "[reindex] input=${SOURCE_JSONL}"
  echo "[reindex] output=${data_path}"
  python "${MEGATRON_DIR}/tools/preprocess_data.py" \
    --input "${SOURCE_JSONL}" \
    --json-keys text \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "${tokenizer}" \
    --output-prefix "${work_prefix}" \
    --append-eod \
    --workers "${WORKERS}"

  [[ -s "${work_data_path}.bin" && -s "${work_data_path}.idx" ]] || {
    echo "[reindex] preprocessing did not produce a complete temporary index for ${model}" >&2
    return 1
  }
  validate_index "${work_data_path}"

  mv -f "${work_data_path}.bin" "${final_bin}"
  mv -f "${work_data_path}.idx" "${final_idx}"
  write_metadata "${model}" "${tokenizer}" "${data_path}" "${metadata_path}"
  echo "[reindex] complete model=${model} DATA_PATH=${data_path}"
}

for model in ${MODELS}; do
  reindex_model "${model}"
done
