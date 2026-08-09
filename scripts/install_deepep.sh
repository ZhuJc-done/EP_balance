#!/usr/bin/env bash
# Install DeepEP V2 ElasticBuffer for Scale-EPLB's zero-sync token transport.
# Override via env: DEEPEP_DIR, DEEPEP_REPO, DEEPEP_COMMIT, NCCL_PKG, TORCH_CUDA_ARCH_LIST.
set -euo pipefail

DEEPEP_DIR="${DEEPEP_DIR:-${HOME}/DeepEP}"
DEEPEP_REPO="${DEEPEP_REPO:-https://github.com/deepseek-ai/DeepEP.git}"
# Validated with the apply-mode DeepEP adapter on GB200. Pin a new SHA only after
# rerunning tests/test_gin_weights.py and a real-model training step.
DEEPEP_COMMIT="${DEEPEP_COMMIT:-af9a0403188392824fc3057452822235873e0612}"
NCCL_PKG="${NCCL_PKG:-nvidia-nccl-cu13>=2.30.4}"

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

# NCCL into the Python env so DeepEP auto-locates it (Device API + GIN; 2.30.4+).
python -m pip install "$NCCL_PKG" --no-deps

if [ ! -d "$DEEPEP_DIR/.git" ]; then
  git clone "$DEEPEP_REPO" "$DEEPEP_DIR"
fi
git -C "$DEEPEP_DIR" fetch origin "$DEEPEP_COMMIT" || git -C "$DEEPEP_DIR" fetch origin
git -C "$DEEPEP_DIR" checkout "$DEEPEP_COMMIT"

NCCL_LIB_DIR="$(python - <<'PY'
import os
try:
    import deep_ep.find_pkgs as fp  # DeepEP ships the same resolver it uses in setup.py
    print(f"{fp.find_nccl_root()}/lib")
except Exception:
    import nvidia.nccl, os
    # nvidia is a PEP420 namespace pkg -> __file__ is None; use __path__ for the install dir.
    print(os.path.join(nvidia.nccl.__path__[0], "lib"))
PY
)"
if [ -n "$NCCL_LIB_DIR" ] && [ -d "$NCCL_LIB_DIR" ]; then
  export LIBRARY_PATH="${NCCL_LIB_DIR}:${LIBRARY_PATH:-}"
  echo "[install_deepep] LIBRARY_PATH += ${NCCL_LIB_DIR} (link-time NCCL search path)"
fi

# Install into the user site-packages because the system Python prefix may be read-only.
( cd "$DEEPEP_DIR" && python setup.py install --user )

python - <<'PY'
import deep_ep
import torch

assert hasattr(deep_ep, "ElasticBuffer"), "pinned DeepEP does not export ElasticBuffer"
assert hasattr(deep_ep, "topk_idx_t"), "pinned DeepEP lacks the Elastic routing dtype"
version = torch.cuda.nccl.version()
version = tuple(version) if isinstance(version, tuple) else version
if isinstance(version, tuple):
    assert version >= (2, 30, 4), f"NCCL 2.30.4+ required, got {version}"
print("[install_deepep] ElasticBuffer OK:", deep_ep.__file__)
print("[install_deepep] NCCL:", version)
PY
echo "[install_deepep] done -> EPLB_ADAPTER=deepep uses ElasticBuffer only"
