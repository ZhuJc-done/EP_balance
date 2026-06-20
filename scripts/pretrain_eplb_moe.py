"""Zero-fork Megatron entrypoint that attaches the Scale-EPLB Phase B observer (see scripts/README.md)."""

from __future__ import annotations

import os

import torch

import pretrain_gpt as pg  # NVIDIA Megatron-LM example script (repo root)
from megatron.core.enums import ModelType
from megatron.training import get_args, pretrain

from eplb.integration.megatron import setup_eplb_observer

_OBSERVERS = []  # keep hook handles alive for the process lifetime


def _expert_param_bytes(args) -> int:
    """Estimate ``|W_e|`` bytes for one expert (gated MLP: w1 + w3 + w2)."""
    dtype_bytes = 2 if (getattr(args, "bf16", False) or getattr(args, "fp16", False)) else 4
    ffn = getattr(args, "moe_ffn_hidden_size", None) or args.ffn_hidden_size
    num_params = 3 * args.hidden_size * ffn
    return int(num_params * dtype_bytes)


def model_provider(pre_process=True, post_process=True):
    """Megatron's GPT model_provider, plus the EPLB observer when MoE + EPLB_OBSERVE."""
    model = pg.model_provider(pre_process, post_process)
    args = get_args()
    enabled = os.environ.get("EPLB_OBSERVE", "1") == "1"
    if enabled and getattr(args, "num_experts", None):
        dtype_bytes = 2 if (getattr(args, "bf16", False) or getattr(args, "fp16", False)) else 4
        ep = args.expert_model_parallel_size
        rank0 = (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
        hook, handles = setup_eplb_observer(
            model,
            num_experts=args.num_experts,
            weight_bytes_each=_expert_param_bytes(args),
            s_tok=args.hidden_size * dtype_bytes,
            n_slot=max(2, 2 * (args.num_experts // ep)),
            gpus_per_node=int(os.environ.get("GPUS_PER_NODE", "8")),
            logger=(print if rank0 else None),
        )
        _OBSERVERS.append((hook, handles))
    return model


if __name__ == "__main__":
    pretrain(
        pg.train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        pg.forward_step,
        args_defaults={"tokenizer_type": "NullTokenizer"},
    )
