"""Launcher model presets should remain architecture-faithful and selectable by name."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


RECIPE_FILE = Path(__file__).resolve().parents[1] / "scripts" / "model_recipes.sh"


def _recipe(model: str, *, num_layers: int | None = None):
    env = os.environ.copy()
    if num_layers is None:
        env.pop("NUM_LAYERS", None)
    else:
        env["NUM_LAYERS"] = str(num_layers)
    script = r'''
set -euo pipefail
source "$1"
configure_model_recipe "$2"
printf 'DEFAULT=%s\n' "${MODEL_DEFAULT_ROUTER_BALANCING}"
printf 'MODEL=%s\n' "${MODEL_ARGS[@]}"
printf 'MOE=%s\n' "${MOE_ARGS[@]}"
'''
    proc = subprocess.run(
        ["bash", "-c", script, "recipe-test", str(RECIPE_FILE), model],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    model_args = []
    moe_args = []
    default = None
    for line in proc.stdout.splitlines():
        key, value = line.split("=", 1)
        if key == "MODEL":
            model_args.append(value)
        elif key == "MOE":
            moe_args.append(value)
        elif key == "DEFAULT":
            default = value
    return model_args, moe_args, default


def _value_after(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_deepseek_v2_defaults_to_full_recipe():
    model, moe, default = _recipe("deepseek_v2_160e")

    assert _value_after(model, "--num-layers") == "60"
    assert _value_after(model, "--hidden-size") == "5120"
    assert "--multi-latent-attention" in model
    assert _value_after(model, "--q-lora-rank") == "1536"
    assert _value_after(moe, "--num-experts") == "160"
    assert _value_after(moe, "--moe-router-topk") == "6"
    assert _value_after(moe, "--moe-shared-expert-intermediate-size") == "3072"
    assert _value_after(moe, "--moe-layer-freq") == "([0]*1+[1]*59)"
    assert _value_after(moe, "--moe-router-num-groups") == "8"
    assert default == "seq_aux_loss"


def test_glm45_air_defaults_to_full_recipe():
    model, moe, default = _recipe("glm45_air")

    assert _value_after(model, "--num-layers") == "46"
    assert _value_after(model, "--hidden-size") == "4096"
    assert _value_after(model, "--num-attention-heads") == "96"
    assert _value_after(model, "--num-query-groups") == "8"
    assert _value_after(model, "--rotary-percent") == "0.5"
    assert _value_after(moe, "--num-experts") == "128"
    assert _value_after(moe, "--moe-router-topk") == "8"
    assert _value_after(moe, "--moe-shared-expert-intermediate-size") == "1408"
    assert _value_after(moe, "--moe-layer-freq") == "([0]*1+[1]*45)"
    assert "--moe-router-enable-expert-bias" in moe
    assert "--moe-shared-expert-overlap" not in moe
    assert default == "seq_aux_loss"


def test_num_layers_override_keeps_one_dense_prefix():
    model, moe, _ = _recipe("deepseek_v2_160e", num_layers=6)

    assert _value_after(model, "--num-layers") == "6"
    assert _value_after(moe, "--moe-layer-freq") == "([0]*1+[1]*5)"


def test_existing_qwen_recipe_remains_selectable():
    model, moe, default = _recipe("qwen3_30b_a3b")

    assert _value_after(model, "--num-layers") == "48"
    assert _value_after(moe, "--num-experts") == "128"
    assert default == "aux_loss"


def test_num_layers_override_applies_to_qwen():
    model, moe, _ = _recipe("qwen3_30b_a3b", num_layers=5)

    assert _value_after(model, "--num-layers") == "5"
    assert _value_after(moe, "--moe-layer-freq") == "1"
