"""Local-attention compatibility for Megatron's multi-latent attention.

Megatron 0ff7226 builds local MLA with ``DotProductAttention`` but passes the
distinct query/key and value head widths accepted only by its Transformer
Engine wrapper. The local attention math already consumes those dimensions
from the runtime tensor shapes; this adapter adds the missing constructor
contract and fixes the flattened value projection width.
"""

from __future__ import annotations

import inspect
import math

from megatron.core.models import backends
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.utils import divide


class LocalMLADotProductAttention(DotProductAttention):
    """Megatron local dot-product attention with distinct MLA QK/V widths."""

    def __init__(
        self,
        config,
        layer_number: int,
        attn_mask_type,
        attention_type: str,
        attention_dropout=None,
        softmax_scale=None,
        k_channels: int | None = None,
        v_channels: int | None = None,
        cp_comm_type: str | None = None,
        pg_collection=None,
    ) -> None:
        super().__init__(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            attention_dropout=attention_dropout,
            softmax_scale=softmax_scale,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )

        self.k_channels = int(k_channels or config.kv_channels)
        self.v_channels = int(v_channels or config.kv_channels)
        self.hidden_size_per_attention_head = self.k_channels
        self.hidden_size_per_partition = divide(
            self.v_channels * config.num_attention_heads,
            self.pg_collection.tp.size(),
        )

        if softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(self.k_channels)
            if config.apply_query_key_layer_scaling:
                self.softmax_scale /= self.layer_number


def install_local_mla_attention_compat() -> bool:
    """Install the adapter into Megatron's local backend when it is needed.

    Returns ``True`` only when this call changed the backend. A future Megatron
    revision that natively accepts both channel arguments is left untouched.
    """

    current = backends.DotProductAttention
    parameters = inspect.signature(current.__init__).parameters
    if "k_channels" in parameters and "v_channels" in parameters:
        return False
    backends.DotProductAttention = LocalMLADotProductAttention
    return True


__all__ = ["LocalMLADotProductAttention", "install_local_mla_attention_compat"]
