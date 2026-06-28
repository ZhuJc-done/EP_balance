#!/usr/bin/env bash
# Install community NVIDIA Megatron-LM for Scale-EPLB (external dependency, NOT vendored).
# Clones a pinned commit, pip-installs it editable, adds the GPT/MoE Python deps, and
# self-checks that `import megatron` resolves to THIS tree (not an internal fork).
#
# Override via env: MEGATRON_DIR, MEGATRON_REPO, MEGATRON_COMMIT.
set -euo pipefail

MEGATRON_DIR="${MEGATRON_DIR:-/opt/tiger/Megatron-LM}"
MEGATRON_REPO="${MEGATRON_REPO:-https://github.com/NVIDIA/Megatron-LM.git}"
# Validated with Scale-EPLB on GB200 (megatron-core 0.19.0+0ff7226f6). Pin a new SHA after re-validating.
MEGATRON_COMMIT="${MEGATRON_COMMIT:-0ff7226f6}"

echo "[install_megatron] dir=$MEGATRON_DIR repo=$MEGATRON_REPO commit=$MEGATRON_COMMIT"

if [ ! -d "$MEGATRON_DIR/.git" ]; then
  git clone "$MEGATRON_REPO" "$MEGATRON_DIR"
fi
git -C "$MEGATRON_DIR" fetch origin "$MEGATRON_COMMIT" || git -C "$MEGATRON_DIR" fetch origin
git -C "$MEGATRON_DIR" checkout "$MEGATRON_COMMIT"

# Editable install + the deps the GPT/MoE path needs. Transformer-Engine / Apex are OPTIONAL:
# the launchers use `--transformer-impl local` + `--no-*-fusion`, so we don't require them here.
pip install -e "$MEGATRON_DIR"
pip install transformers einops sentencepiece tiktoken regex

# Self-check: a concrete submodule must resolve INSIDE $MEGATRON_DIR (megatron is a namespace pkg,
# so `megatron.__file__` is None and unreliable; check megatron.training instead).
MEGATRON_DIR="$MEGATRON_DIR" PYTHONPATH="$MEGATRON_DIR${PYTHONPATH:+:$PYTHONPATH}" python - <<'PY'
import os, importlib
mt = importlib.import_module("megatron.training")
mc = importlib.import_module("megatron.core")
want = os.environ["MEGATRON_DIR"]
print("megatron.training:", mt.__file__)
print("megatron.core    :", getattr(mc, "__version__", "?"))
assert mt.__file__ and mt.__file__.startswith(want), \
    f"megatron resolved OUTSIDE {want} (an internal fork shadows it; strip it from PYTHONPATH)"
print("[install_megatron] import OK")
PY
echo "[install_megatron] done -> set MEGATRON_DIR=$MEGATRON_DIR for the launchers"
