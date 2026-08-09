#!/usr/bin/env bash
# Install DeepEP V2 ElasticBuffer for Scale-EPLB's zero-sync token transport.
# Override via env: DEEPEP_DIR, DEEPEP_REPO, DEEPEP_COMMIT, TORCH_CUDA_ARCH_LIST.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPEP_DIR="${DEEPEP_DIR:-${HOME}/DeepEP}"
DEEPEP_REPO="${DEEPEP_REPO:-https://github.com/deepseek-ai/DeepEP.git}"
# Validated with the apply-mode DeepEP adapter on GB200. Pin a new SHA only after
# rerunning tests/test_gin_weights.py and a real-model training step.
DEEPEP_COMMIT="${DEEPEP_COMMIT:-af9a0403188392824fc3057452822235873e0612}"
NCCL_PKG="nvidia-nccl-cu13==2.30.7"

# Default arch from the live GPU (Blackwell -> 10.0a, Hopper -> 9.0a); DeepEP's default 9.0 breaks on Blackwell.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  TORCH_CUDA_ARCH_LIST="$(python - <<'PY'
import torch
if torch.cuda.is_available():
    maj, minr = torch.cuda.get_device_capability(0)
    print(f"{maj}.{minr}a")
else:
    print("9.0")
PY
)"
fi
export TORCH_CUDA_ARCH_LIST
echo "[install_deepep] dir=$DEEPEP_DIR commit=$DEEPEP_COMMIT arch=$TORCH_CUDA_ARCH_LIST nccl=$NCCL_PKG"

# Install and select the exact NCCL runtime validated with ElasticBuffer and GIN.
python -m pip install "$NCCL_PKG" --no-deps
source "${SCRIPT_DIR}/env_nccl_2307.sh"

if [ ! -d "$DEEPEP_DIR/.git" ]; then
  git clone "$DEEPEP_REPO" "$DEEPEP_DIR"
fi
git -C "$DEEPEP_DIR" fetch origin "$DEEPEP_COMMIT" || git -C "$DEEPEP_DIR" fetch origin
git -C "$DEEPEP_DIR" checkout "$DEEPEP_COMMIT"

NCCL_LIB_DIR="${NCCL_HOME}/lib"
if [ -n "$NCCL_LIB_DIR" ] && [ -d "$NCCL_LIB_DIR" ]; then
  export LIBRARY_PATH="${NCCL_LIB_DIR}:${LIBRARY_PATH:-}"
  echo "[install_deepep] LIBRARY_PATH += ${NCCL_LIB_DIR} (link-time NCCL search path)"
fi

# Install into the user site-packages because the system Python prefix may be read-only.
( cd "$DEEPEP_DIR" && python setup.py install --user )

python - <<'PY'
import deep_ep
import ctypes

assert hasattr(deep_ep, "ElasticBuffer"), "pinned DeepEP does not export ElasticBuffer"
assert hasattr(deep_ep, "topk_idx_t"), "pinned DeepEP lacks the Elastic routing dtype"
paths = {
    line.split()[-1] for line in open("/proc/self/maps", encoding="utf-8")
    if "libnccl.so" in line and line.split()[-1].startswith("/")
}
assert len(paths) == 1, f"expected one loaded NCCL runtime, got {paths}"
lib = ctypes.CDLL(next(iter(paths)))
encoded = ctypes.c_int()
assert lib.ncclGetVersion(ctypes.byref(encoded)) == 0
value = encoded.value
version = value // 10000, (value % 10000) // 100, value % 100
assert version == (2, 30, 7), f"NCCL 2.30.7 required, got {version}"
print("[install_deepep] ElasticBuffer OK:", deep_ep.__file__)
print("[install_deepep] NCCL:", version)
PY
echo "[install_deepep] done -> EPLB_ADAPTER=deepep uses ElasticBuffer only"
