#!/usr/bin/env bash
# Install NVIDIA Transformer Engine (OPTIONAL fast path: TE + grouped GEMM) for Scale-EPLB.
# Not required to run/test: the launchers default to `--transformer-impl local` + `--no-*-fusion`.
# Install this only for the TE fast path (HAS_TE=1 in run_real_moe.sh).
#
# TE's PyTorch extension #include "nccl.h", which neither CUDA nor the torch-bundled NCCL runtime
# ships. We add nvidia-nccl-cuXX (headers + libnccl.so), point the compiler at it, then build TE.
# Override via env: NCCL_PKG, NVTE_CUDA_ARCHS, TE_SPEC, PIN_PROTOBUF.
set -euo pipefail

NCCL_PKG="${NCCL_PKG:-nvidia-nccl-cu13}"
TE_SPEC="${TE_SPEC:-transformer-engine[pytorch]}"
# TE[pytorch] pulls onnx, which drags protobuf>=4.25; that breaks byted pb2 code + megatron's
# `import wandb` (databus). Pin back to a byted/Megatron-compatible protobuf. Set PIN_PROTOBUF="" to skip.
PIN_PROTOBUF="${PIN_PROTOBUF:-3.20.3}"

# TE build target arch: derive from the live GPU (Blackwell GB200 -> 100, Hopper -> 90).
if [ -z "${NVTE_CUDA_ARCHS:-}" ]; then
  NVTE_CUDA_ARCHS="$(python - <<'PY'
import torch
if torch.cuda.is_available():
    maj, minr = torch.cuda.get_device_capability(0)
    print(f"{maj}{minr}")
else:
    print("90")
PY
)"
fi
export NVTE_CUDA_ARCHS NVTE_FRAMEWORK=pytorch
echo "[install_te] nccl=$NCCL_PKG arch=$NVTE_CUDA_ARCHS spec=$TE_SPEC pin_protobuf=${PIN_PROTOBUF:-<skip>}"

# 1) NCCL headers + lib into the Python env (torch bundles only the runtime, no nccl.h).
pip install "$NCCL_PKG"
NCCL_ROOT="$(python -c 'import nvidia.nccl as n; print(n.__path__[0])')"
# the linker (-lnccl) wants an unversioned libnccl.so; the wheel only ships libnccl.so.2.
if [ ! -e "$NCCL_ROOT/lib/libnccl.so" ] && ls "$NCCL_ROOT/lib/"libnccl.so.* >/dev/null 2>&1; then
  ln -sf "$(cd "$NCCL_ROOT/lib" && ls libnccl.so.* | head -n1)" "$NCCL_ROOT/lib/libnccl.so"
fi
echo "[install_te] NCCL_ROOT=$NCCL_ROOT"

# 2) Point the compiler/linker at the NCCL headers + lib (must be in the SAME shell as the build).
export CPATH="$NCCL_ROOT/include:/usr/local/cuda/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$NCCL_ROOT/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$NCCL_ROOT/lib:/usr/local/cuda/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$NCCL_ROOT/lib:${LD_LIBRARY_PATH:-}"
test -f "$NCCL_ROOT/include/nccl.h" || { echo "[install_te] nccl.h missing under $NCCL_ROOT/include" >&2; exit 1; }

# 3) Build TE against the current torch (no build isolation = reuse installed torch, not a fresh one).
pip install --no-build-isolation --no-cache-dir "$TE_SPEC"

# 4) Restore a protobuf that byted SDKs + Megatron (`import wandb` -> databus pb2) can load.
if [ -n "$PIN_PROTOBUF" ]; then
  pip install "protobuf==$PIN_PROTOBUF"
fi

# 5) Self-check: TE + its PyTorch layers import (libnccl.so must be findable at runtime).
LD_LIBRARY_PATH="$NCCL_ROOT/lib:${LD_LIBRARY_PATH:-}" python - <<'PY'
import transformer_engine as te
import transformer_engine.pytorch as tep
print("[install_te] transformer_engine:", te.__version__)
assert hasattr(tep, "Linear") and hasattr(tep, "LayerNormMLP"), "TE pytorch layers missing"
print("[install_te] import OK")
PY

echo "[install_te] done."
echo "[install_te] RUNTIME: export LD_LIBRARY_PATH=\"$NCCL_ROOT/lib:\$LD_LIBRARY_PATH\" before launching (TE needs libnccl.so)."
