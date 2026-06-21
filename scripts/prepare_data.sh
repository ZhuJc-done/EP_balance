#!/usr/bin/env bash
# Tokenize a HF text dataset into Megatron .bin/.idx using the model's tokenizer (output prefix -> DATA_PATH for training).
set -euo pipefail

MEGATRON_DIR="${MEGATRON_DIR:?set MEGATRON_DIR to the Megatron-LM repo root}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-Qwen/Qwen3-30B-A3B}"   # HF repo/dir matching the checkpoint
DATASET="${DATASET:-Salesforce/wikitext}"                  # any HF dataset with a text column
DATASET_CONFIG="${DATASET_CONFIG:-wikitext-103-raw-v1}"
SPLIT="${SPLIT:-train}"
MAX_DOCS="${MAX_DOCS:-200000}"                             # cap docs for a quick demo (0 = all)
OUT_DIR="${OUT_DIR:-$HOME/eplb_data}"
OUT_PREFIX="${OUT_PREFIX:-${OUT_DIR}/corpus}"
WORKERS="${WORKERS:-16}"

mkdir -p "${OUT_DIR}"
JSONL="${OUT_DIR}/corpus.jsonl"

# 1) dump the dataset to one-json-per-line with a "text" field
python - "$DATASET" "$DATASET_CONFIG" "$SPLIT" "$MAX_DOCS" "$JSONL" <<'PY'
import json, sys
from datasets import load_dataset
name, config, split, max_docs, out = sys.argv[1:6]
max_docs = int(max_docs)
ds = load_dataset(name, config, split=split, streaming=True)
n = 0
with open(out, "w") as f:
    for ex in ds:
        text = (ex.get("text") or "").strip()
        if not text:
            continue
        f.write(json.dumps({"text": text}) + "\n")
        n += 1
        if max_docs and n >= max_docs:
            break
print(f"wrote {n} docs -> {out}")
PY

# 2) tokenize into Megatron indexed format (.bin/.idx)
python "${MEGATRON_DIR}/tools/preprocess_data.py" \
  --input "${JSONL}" \
  --json-keys text \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model "${TOKENIZER_MODEL}" \
  --output-prefix "${OUT_PREFIX}" \
  --append-eod \
  --workers "${WORKERS}"

echo "[prepare_data] done. Use in training:  DATA_PATH=${OUT_PREFIX}_text_document"
