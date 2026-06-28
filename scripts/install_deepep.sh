#!/usr/bin/env bash
# Install DeepEP (OPTIONAL sync-free EP transport) for Scale-EPLB (external dependency, NOT vendored).
# Not required to run/test: Phase C defaults to AllToAllAdapter (torch all_to_all_single).
# Install this only to wire DeepEPAdapter on a DeepEP-capable cluster.
#
# DeepEP V2 uses the lightweight NCCL Gin backend (no NVSHMEM needed).
# Override via env: DEEPEP_DIR, DEEPEP_REPO, DEEPEP_COMMIT, NCCL_PKG, TORCH_CUDA_ARCH_LIST.
set -euo pipefail

DEEPEP_DIR="${DEEPEP_DIR:-/opt/tiger/DeepEP}"
DEEPEP_REPO="${DEEPEP_REPO:-https://github.com/deepseek-ai/DeepEP.git}"
DEEPEP_COMMIT="${DEEPEP_COMMIT:-main}"        # pin a SHA once validated on-cluster
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
pip install "$NCCL_PKG" --no-deps

if [ ! -d "$DEEPEP_DIR/.git" ]; then
  git clone "$DEEPEP_REPO" "$DEEPEP_DIR"
fi
if [ "$DEEPEP_COMMIT" != "main" ]; then
  git -C "$DEEPEP_DIR" fetch origin "$DEEPEP_COMMIT"
  git -C "$DEEPEP_DIR" checkout "$DEEPEP_COMMIT"
fi

NCCL_LIB_DIR="$(python - <<'PY'
import os
try:
    import deep_ep.find_pkgs as fp  # DeepEP ships the same resolver it uses in setup.py
    print(f"{fp.find_nccl_root()}/lib")
except Exception:
    import nvidia.nccl, os
    print(os.path.join(os.path.dirname(nvidia.nccl.__file__), "lib"))
PY
)"
if [ -n "$NCCL_LIB_DIR" ] && [ -d "$NCCL_LIB_DIR" ]; then
  export LIBRARY_PATH="${NCCL_LIB_DIR}:${LIBRARY_PATH:-}"
  echo "[install_deepep] LIBRARY_PATH += ${NCCL_LIB_DIR} (link-time NCCL search path)"
fi

( cd "$DEEPEP_DIR" && python setup.py install )

python -c "import deep_ep; print('[install_deepep] import OK:', deep_ep.__file__)"
echo "[install_deepep] done -> swap AllToAllAdapter() for DeepEPAdapter() in bind_eplb_to_moe_layer"
