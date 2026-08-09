#!/usr/bin/env bash
# Install Scale-EPLB, pinned Megatron-LM, and pinned DeepEP on a fresh CUDA/PyTorch image.
# Persistent datasets, tokenizers, checkpoints, logs, and HF caches live on the HDFS mount.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EPLB_DIR="${EPLB_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
source "${SCRIPT_DIR}/env_hdfs.sh"

LOCAL_SRC_ROOT="${LOCAL_SRC_ROOT:-${HOME}}"
export MEGATRON_DIR="${MEGATRON_DIR:-${LOCAL_SRC_ROOT}/Megatron-LM}"
export DEEPEP_DIR="${DEEPEP_DIR:-${LOCAL_SRC_ROOT}/DeepEP}"
export NCCL_PKG="${NCCL_PKG:-nvidia-nccl-cu13>=2.30.4}"

python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is not available in this PyTorch image"
assert torch.version.cuda and torch.version.cuda.split(".", 1)[0] == "13", (
    f"this bootstrap installs nvidia-nccl-cu13, but PyTorch uses CUDA {torch.version.cuda}"
)
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY
command -v nvcc >/dev/null || {
  echo "nvcc is required to build DeepEP but is not on PATH" >&2
  exit 1
}

python -m pip install setuptools wheel ninja packaging
python -m pip install -e "${EPLB_DIR}[dev]"

MEGATRON_DIR="${MEGATRON_DIR}" \
  bash "${SCRIPT_DIR}/install_megatron.sh"
DEEPEP_DIR="${DEEPEP_DIR}" NCCL_PKG="${NCCL_PKG}" \
  bash "${SCRIPT_DIR}/install_deepep.sh"

_nccl_lib="$(python -c 'import nvidia.nccl as n,os;print(os.path.join(n.__path__[0],"lib"))')"
export LD_LIBRARY_PATH="${_nccl_lib}:${LD_LIBRARY_PATH:-}"

TOKENIZER_SOURCE="${TOKENIZER_SOURCE:-Qwen/Qwen3-30B-A3B}"
TOKENIZER_SAVE_DIR="${TOKENIZER_SAVE_DIR:-${EPLB_TOKENIZER_DIR}/qwen3_30b_a3b}"
if [[ "${PREPARE_TOKENIZER:-1}" == "1" && ! -f "${TOKENIZER_SAVE_DIR}/tokenizer_config.json" ]]; then
  TOKENIZER_SOURCE="${TOKENIZER_SOURCE}" TOKENIZER_SAVE_DIR="${TOKENIZER_SAVE_DIR}" python - <<'PY'
import os
from transformers import AutoTokenizer

source = os.environ["TOKENIZER_SOURCE"]
target = os.environ["TOKENIZER_SAVE_DIR"]
tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
tokenizer.save_pretrained(target)
print(f"tokenizer: {source} -> {target}")
PY
elif [[ -f "${TOKENIZER_SAVE_DIR}/tokenizer_config.json" ]]; then
  echo "tokenizer already present: ${TOKENIZER_SAVE_DIR}"
fi

python - <<'PY'
import deep_ep
import megatron.core
import nccl_gin
import torch

assert hasattr(deep_ep, "ElasticBuffer"), "DeepEP ElasticBuffer is unavailable"
assert hasattr(deep_ep, "topk_idx_t"), "DeepEP Elastic routing dtype is unavailable"
nccl = torch.cuda.nccl.version()
if isinstance(nccl, tuple):
    assert nccl >= (2, 30, 4), f"NCCL 2.30.4+ required, got {nccl}"
print("Scale-EPLB bootstrap complete")
print("Megatron-Core:", getattr(megatron.core, "__version__", "unknown"))
print("DeepEP Elastic:", getattr(deep_ep, "__version__", "unknown"))
print("NCCL:", nccl)
print("GIN module:", nccl_gin.__file__)
print("CUDA capability:", torch.cuda.get_device_capability(0))
PY

printf '%s\n' \
  "EPLB_DIR=${EPLB_DIR}" \
  "MEGATRON_DIR=${MEGATRON_DIR}" \
  "DEEPEP_DIR=${DEEPEP_DIR}" \
  "EPLB_DATA_ROOT=${EPLB_DATA_ROOT}" \
  "TOKENIZER_MODEL=${TOKENIZER_SAVE_DIR}"
