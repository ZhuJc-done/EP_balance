"""Zero-fork Megatron-LM ``main`` entrypoint for Scale-EPLB; EPLB_MODE=observe (Phase B) or apply (Phase C)."""

from __future__ import annotations

import os
import time

import torch

import pretrain_gpt as pg  # NVIDIA Megatron-LM example script (repo root)
from megatron.core import parallel_state as mpu
from megatron.core.enums import ModelType
from megatron.training import get_args, pretrain

from eplb import EPLBConfig, Topology
from eplb.integration import EPLBRebalancer, bind_eplb_to_moe_layer, find_moe_layers
from eplb.integration.megatron import build_spec_for_megatron, setup_eplb_observer

_KEEPALIVE = []  # keep hooks / rebalancers alive for the process lifetime


def _expert_param_bytes(args) -> int:
    """Estimate ``|W_e|`` bytes for one expert (gated MLP: w1 + w3 + w2)."""
    dtype_bytes = 2 if (getattr(args, "bf16", False) or getattr(args, "fp16", False)) else 4
    ffn = getattr(args, "moe_ffn_hidden_size", None) or args.ffn_hidden_size
    num_params = 3 * args.hidden_size * ffn
    return int(num_params * dtype_bytes)


def _eplb_params(args):
    """Common (num_experts, ep, weight_bytes, s_tok, n_slot, gpus_per_node) for the solver."""
    dtype_bytes = 2 if (getattr(args, "bf16", False) or getattr(args, "fp16", False)) else 4
    ep = args.expert_model_parallel_size
    return dict(
        num_experts=args.num_experts,
        ep=ep,
        weight_bytes_each=_expert_param_bytes(args),
        s_tok=args.hidden_size * dtype_bytes,
        n_slot=max(2, 2 * (args.num_experts // ep)),
        gpus_per_node=int(os.environ.get("GPUS_PER_NODE", "8")),
    )


def model_provider(
    pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None
):
    """Megatron's GPT model_provider plus the EPLB observer (Phase B) or dispatcher binding (Phase C)."""
    model = pg.model_provider(
        pg.gpt_builder,
        pre_process,
        post_process,
        vp_stage=vp_stage,
        config=config,
        pg_collection=pg_collection,
    )
    args = get_args()
    mode = os.environ.get("EPLB_MODE", "observe")
    if mode == "off" or not getattr(args, "num_experts", None):
        return model

    p = _eplb_params(args)
    if mode == "observe":
        rank0 = (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
        hook, handles = setup_eplb_observer(
            model, num_experts=p["num_experts"], weight_bytes_each=p["weight_bytes_each"],
            s_tok=p["s_tok"], n_slot=p["n_slot"], gpus_per_node=p["gpus_per_node"],
            logger=(print if rank0 else None),
        )
        _KEEPALIVE.append((hook, handles))
    elif mode == "apply":
        ep_group = mpu.get_expert_model_parallel_group()
        ep_size = mpu.get_expert_model_parallel_world_size()
        device = next(model.parameters()).device
        gpn = p["gpus_per_node"] if ep_size % p["gpus_per_node"] == 0 else ep_size
        topo = Topology.from_nvlink_rdma(ep_size // gpn, gpn, 1, 8, device=device)
        for layer_id, moe in enumerate(find_moe_layers(model)):
            spec = build_spec_for_megatron(
                p["num_experts"], ep_size, p["weight_bytes_each"], p["s_tok"], p["n_slot"], device
            )
            reb = EPLBRebalancer(topo, spec, EPLBConfig())
            bind_eplb_to_moe_layer(moe, reb, ep_group, layer_id)
            _KEEPALIVE.append(reb)
    else:
        raise ValueError(f"unknown EPLB_MODE={mode!r} (expected observe|apply|off)")
    return model


if __name__ == "__main__":
    pg.set_startup_timestamps(program_start=time.time(), main_entry=time.time())

    # Temporary for transition to core datasets (matches pretrain_gpt.py).
    setattr(pg.train_valid_test_datasets_provider, "is_distributed", True)

    # Optionally enable inprocess restart (no-op unless configured).
    pretrain_fn, store = pg.inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    args = pg.parse_and_validate_args(
        extra_args_provider=pg.add_modelopt_args if pg.has_nvidia_modelopt else None,
        args_defaults={"tokenizer_type": "NullTokenizer"},
    )
    model_cfg = pg.gpt_config_from_args(args)
    full_config = pg.pretrain_cfg_container_from_args(args, model_cfg)

    pretrain_fn(
        full_config,
        pg.train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        pg.forward_step,
        store=store,
        get_embedding_ranks=pg.get_embedding_ranks,
    )
