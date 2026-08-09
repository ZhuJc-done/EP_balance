#!/usr/bin/env bash
# Download, mix, and index a reproducible 1B-token multi-domain corpus on HDFS.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env_hdfs.sh"

MEGATRON_DIR="${MEGATRON_DIR:-${HOME}/Megatron-LM}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-${EPLB_TOKENIZER_DIR}/qwen3_30b_a3b}"
MIX_NAME="${MIX_NAME:-mixed_1b}"
TOTAL_TOKENS="${TOTAL_TOKENS:-1000000000}"
WORKERS="${WORKERS:-32}"
FORCE="${FORCE:-0}"

COMPONENT_DIR="${EPLB_RAW_DATA_DIR}/${MIX_NAME}_components"
MIX_JSONL="${EPLB_RAW_DATA_DIR}/${MIX_NAME}.jsonl"
MIX_MANIFEST="${EPLB_RAW_DATA_DIR}/${MIX_NAME}.manifest.json"
OUTPUT_PREFIX="${EPLB_INDEXED_DATA_DIR}/${MIX_NAME}"
mkdir -p "${COMPONENT_DIR}" "${EPLB_INDEXED_DATA_DIR}"

[[ -f "${MEGATRON_DIR}/tools/preprocess_data.py" ]] || {
  echo "Megatron preprocess tool not found under ${MEGATRON_DIR}" >&2
  exit 1
}
[[ -f "${TOKENIZER_MODEL}/tokenizer_config.json" ]] || {
  echo "Tokenizer not found under ${TOKENIZER_MODEL}" >&2
  exit 1
}

export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
if command -v hf >/dev/null && ! hf auth whoami >/dev/null 2>&1; then
  echo "[mixed] WARNING: Hugging Face is anonymous; run 'hf auth login' to avoid anonymous limits" >&2
fi

component_is_complete() {
  local manifest="$1"
  local budget="$2"
  local allow_short="$3"
  [[ "${FORCE}" != "1" && -s "${manifest}" ]] || return 1
  python - "${manifest}" "${budget}" "${allow_short}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
accepted = int(manifest.get("accepted_tokens", 0))
complete = (
    int(manifest.get("requested_token_budget", -1)) == int(sys.argv[2])
    and accepted > 0
    and (sys.argv[3] == "1" or accepted >= int(sys.argv[2]))
)
raise SystemExit(0 if complete else 1)
PY
}

prepare_component() {
  local workload="$1"
  local budget="$2"
  local max_rows="$3"
  local allow_short="$4"
  local output="${COMPONENT_DIR}/${workload}.jsonl"
  local manifest="${COMPONENT_DIR}/${workload}.manifest.json"
  shift 4

  if component_is_complete "${manifest}" "${budget}" "${allow_short}" && [[ -s "${output}" ]]; then
    echo "[mixed] reuse ${workload}: ${manifest}"
    return
  fi

  echo "[mixed] download ${workload}: target=${budget} tokens"
  if ! python "${SCRIPT_DIR}/prepare_open_workload.py" \
      --workload "${workload}" \
      --tokenizer-model "${TOKENIZER_MODEL}" \
      --token-budget "${budget}" \
      --max-samples 0 \
      --max-source-rows "${max_rows}" \
      --max-document-tokens 4096 \
      --truncate-long-documents \
      --shuffle-buffer 1 \
      --output-jsonl "${output}" \
      "$@"; then
    if component_is_complete "${manifest}" "${budget}" "${allow_short}" && [[ -s "${output}" ]]; then
      echo "[mixed] ${workload} data is complete despite downloader shutdown error"
      return 0
    fi
    return 1
  fi
}

prepare_dolma() {
  local budget="$1"
  local base_jsonl="${COMPONENT_DIR}/dolma.jsonl"
  local base_manifest="${COMPONENT_DIR}/dolma.manifest.json"
  local extra_jsonl="${COMPONENT_DIR}/dolma_extra.jsonl"
  local extra_manifest="${COMPONENT_DIR}/dolma_extra.manifest.json"

  if component_is_complete "${base_manifest}" "${budget}" 0 && [[ -s "${base_jsonl}" ]]; then
    echo "[mixed] reuse dolma: ${base_manifest}"
    return
  fi

  prepare_component dolma "${budget}" 0 1 --source-file-limit 1
  local base_tokens
  base_tokens="$(python - "${base_manifest}" <<'PY'
import json
import sys

print(int(json.load(open(sys.argv[1], encoding="utf-8"))["accepted_tokens"]))
PY
)"
  local remaining=$(( budget - base_tokens ))
  (( remaining > 0 )) || return

  if ! component_is_complete "${extra_manifest}" "${remaining}" 0 || [[ ! -s "${extra_jsonl}" ]]; then
    python "${SCRIPT_DIR}/prepare_open_workload.py" \
      --workload dolma \
      --dataset-id json \
      --dataset-config "" \
      --data-file "https://olmo-data.org/dolma-v1_6-8B-sample/v1_5r2_sample-0001.json.gz" \
      --tokenizer-model "${TOKENIZER_MODEL}" \
      --token-budget "${remaining}" \
      --max-samples 0 \
      --max-source-rows 0 \
      --max-document-tokens 4096 \
      --truncate-long-documents \
      --shuffle-buffer 1 \
      --output-jsonl "${extra_jsonl}"
  fi

  python - "${base_jsonl}" "${base_manifest}" "${extra_jsonl}" "${extra_manifest}" "${budget}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

base_path, base_manifest_path, extra_path, extra_manifest_path = map(Path, sys.argv[1:5])
budget = int(sys.argv[5])
base = json.loads(base_manifest_path.read_text(encoding="utf-8"))
extra = json.loads(extra_manifest_path.read_text(encoding="utf-8"))
temporary = base_path.with_suffix(".merged.tmp")
with temporary.open("wb") as destination:
    for source in (base_path, extra_path):
        with source.open("rb") as stream:
            shutil.copyfileobj(stream, destination)
temporary.replace(base_path)
base["accepted_samples"] = int(base["accepted_samples"]) + int(extra["accepted_samples"])
base["accepted_tokens"] = int(base["accepted_tokens"]) + int(extra["accepted_tokens"])
base["requested_token_budget"] = budget
if int(base["accepted_tokens"]) != budget:
    raise RuntimeError(f"Dolma merge produced {base['accepted_tokens']} tokens, expected {budget}")
base_manifest_path.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
PY
}

DAPO_TARGET=$(( TOTAL_TOKENS * 3 / 100 ))
prepare_component dapo_math "${DAPO_TARGET}" 100000 1
DAPO_ACTUAL="$(python - "${COMPONENT_DIR}/dapo_math.manifest.json" <<'PY'
import json
import sys

print(int(json.load(open(sys.argv[1], encoding="utf-8"))["accepted_tokens"]))
PY
)"

FINEWEB2_TARGET=$(( TOTAL_TOKENS * 20 / 100 ))
DOLMA_TARGET=$(( TOTAL_TOKENS * 12 / 100 ))
STARCODER_TARGET=$(( TOTAL_TOKENS * 10 / 100 ))
PES2O_TARGET=$(( TOTAL_TOKENS * 10 / 100 ))
FINEWEB_TARGET=$(( TOTAL_TOKENS - FINEWEB2_TARGET - DOLMA_TARGET - STARCODER_TARGET - PES2O_TARGET - DAPO_ACTUAL ))

declare -a NAMES=(fineweb fineweb2_zh dolma starcoder pes2o)
declare -a TARGETS=("${FINEWEB_TARGET}" "${FINEWEB2_TARGET}" "${DOLMA_TARGET}" "${STARCODER_TARGET}" "${PES2O_TARGET}")
declare -a PIDS=()
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  if [[ "${name}" == "dolma" ]]; then
    prepare_dolma "${TARGETS[$i]}" >"${COMPONENT_DIR}/${name}.download.log" 2>&1 &
  else
    prepare_component "${name}" "${TARGETS[$i]}" 0 0 >"${COMPONENT_DIR}/${name}.download.log" 2>&1 &
  fi
  PIDS+=("$!")
  echo "[mixed] started ${name} pid=${PIDS[-1]} log=${COMPONENT_DIR}/${name}.download.log"
done

failed=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "[mixed] completed ${NAMES[$i]}"
  else
    echo "[mixed] FAILED ${NAMES[$i]}: see ${COMPONENT_DIR}/${NAMES[$i]}.download.log" >&2
    failed=1
  fi
done
(( failed == 0 )) || exit 1

export MIX_COMPONENT_DIR="${COMPONENT_DIR}"
export MIX_JSONL MIX_MANIFEST TOKENIZER_MODEL TOTAL_TOKENS MIX_NAME
if [[ "${FORCE}" == "1" || ! -s "${MIX_JSONL}" || ! -s "${MIX_MANIFEST}" ]]; then
  python - <<'PY'
import json
import os
from pathlib import Path

names = ["fineweb", "fineweb2_zh", "dolma", "starcoder", "pes2o", "dapo_math"]
component_dir = Path(os.environ["MIX_COMPONENT_DIR"])
output_path = Path(os.environ["MIX_JSONL"])
manifest_path = Path(os.environ["MIX_MANIFEST"])
summaries = []
for name in names:
    manifest = json.loads((component_dir / f"{name}.manifest.json").read_text(encoding="utf-8"))
    summaries.append(
        {
            "name": name,
            "jsonl": str(component_dir / f"{name}.jsonl"),
            "samples": int(manifest["accepted_samples"]),
            "tokens": int(manifest["accepted_tokens"]),
        }
    )

accepted_tokens = sum(item["tokens"] for item in summaries)
requested_tokens = int(os.environ["TOTAL_TOKENS"])
if accepted_tokens != requested_tokens:
    raise RuntimeError(f"component token total is {accepted_tokens}, expected {requested_tokens}")

counts = [item["samples"] for item in summaries]
total_documents = sum(counts)
streams = [open(item["jsonl"], encoding="utf-8") for item in summaries]
emitted = [0] * len(summaries)
output_path.parent.mkdir(parents=True, exist_ok=True)
try:
    with output_path.open("w", encoding="utf-8") as output:
        for position in range(total_documents):
            active = [i for i, count in enumerate(counts) if emitted[i] < count]
            source = max(
                active,
                key=lambda i: ((position + 1) * counts[i] / total_documents) - emitted[i],
            )
            line = streams[source].readline()
            if not line:
                raise RuntimeError(f"unexpected EOF in {summaries[source]['jsonl']}")
            output.write(line)
            emitted[source] += 1
            if (position + 1) % 100_000 == 0:
                print(f"[mixed] merged {position + 1}/{total_documents} documents", flush=True)
finally:
    for stream in streams:
        stream.close()

manifest = {
    "name": os.environ["MIX_NAME"],
    "requested_token_budget": requested_tokens,
    "accepted_samples": total_documents,
    "accepted_tokens": accepted_tokens,
    "tokenizer": os.environ["TOKENIZER_MODEL"],
    "mixing": "deterministic smooth document interleave",
    "components": summaries,
}
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"[mixed] wrote {total_documents} documents / {accepted_tokens} tokens to {output_path}")
PY
else
  echo "[mixed] reuse final JSONL: ${MIX_JSONL}"
fi

if [[ "${FORCE}" == "1" || ! -s "${OUTPUT_PREFIX}_text_document.bin" || ! -s "${OUTPUT_PREFIX}_text_document.idx" ]]; then
  echo "[mixed] preprocessing ${MIX_JSONL}"
  python "${MEGATRON_DIR}/tools/preprocess_data.py" \
    --input "${MIX_JSONL}" \
    --json-keys text \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "${TOKENIZER_MODEL}" \
    --output-prefix "${OUTPUT_PREFIX}" \
    --append-eod \
    --workers "${WORKERS}"
else
  echo "[mixed] reuse indexed dataset: ${OUTPUT_PREFIX}_text_document"
fi

PYTHONPATH="${MEGATRON_DIR}:${PYTHONPATH:-}" python - "${OUTPUT_PREFIX}_text_document" <<'PY'
import sys

from megatron.core.datasets.indexed_dataset import IndexedDataset

dataset = IndexedDataset(sys.argv[1], mmap=True)
tokens = int(dataset.sequence_lengths.sum())
print(f"[mixed] indexed documents={len(dataset)} tokens={tokens} billions={tokens / 1e9:.6f}")
print(f"[mixed] DATA_PATH={sys.argv[1]}")
PY
