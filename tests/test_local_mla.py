import math
from types import SimpleNamespace

from megatron.core.models import backends
from megatron.core.transformer.dot_product_attention import DotProductAttention

from eplb.integration.local_mla import (
    LocalMLADotProductAttention,
    install_local_mla_attention_compat,
)


class _TPGroup:
    def __init__(self, world_size):
        self.world_size = world_size

    def size(self):
        return self.world_size


def test_local_mla_uses_distinct_query_key_and_value_widths(monkeypatch):
    def fake_base_init(
        self,
        *,
        config,
        layer_number,
        attn_mask_type,
        attention_type,
        attention_dropout,
        softmax_scale,
        cp_comm_type,
        pg_collection,
    ):
        self.config = config
        self.layer_number = max(1, layer_number)
        self.pg_collection = pg_collection
        self.softmax_scale = softmax_scale

    monkeypatch.setattr(DotProductAttention, "__init__", fake_base_init)
    config = SimpleNamespace(
        kv_channels=128,
        num_attention_heads=128,
        apply_query_key_layer_scaling=False,
    )
    pg_collection = SimpleNamespace(tp=_TPGroup(4))

    attention = LocalMLADotProductAttention(
        config=config,
        layer_number=1,
        attn_mask_type=object(),
        attention_type="self",
        softmax_scale=0.25,
        k_channels=192,
        v_channels=128,
        pg_collection=pg_collection,
    )

    assert attention.hidden_size_per_attention_head == 192
    assert attention.hidden_size_per_partition == 4096
    assert attention.softmax_scale == 0.25


def test_local_mla_default_scale_uses_query_key_width(monkeypatch):
    def fake_base_init(self, **kwargs):
        self.config = kwargs["config"]
        self.layer_number = max(1, kwargs["layer_number"])
        self.pg_collection = kwargs["pg_collection"]
        self.softmax_scale = None

    monkeypatch.setattr(DotProductAttention, "__init__", fake_base_init)
    config = SimpleNamespace(
        kv_channels=128,
        num_attention_heads=128,
        apply_query_key_layer_scaling=True,
    )

    attention = LocalMLADotProductAttention(
        config=config,
        layer_number=2,
        attn_mask_type=object(),
        attention_type="self",
        k_channels=192,
        v_channels=128,
        pg_collection=SimpleNamespace(tp=_TPGroup(1)),
    )

    assert math.isclose(attention.softmax_scale, 1.0 / math.sqrt(192) / 2)


def test_installer_patches_only_incompatible_megatron_versions(monkeypatch):
    monkeypatch.setattr(backends, "DotProductAttention", DotProductAttention)
    assert install_local_mla_attention_compat()
    assert backends.DotProductAttention is LocalMLADotProductAttention
    assert not install_local_mla_attention_compat()

    class FutureDotProductAttention:
        def __init__(self, k_channels=None, v_channels=None):
            pass

    monkeypatch.setattr(backends, "DotProductAttention", FutureDotProductAttention)
    assert not install_local_mla_attention_compat()
    assert backends.DotProductAttention is FutureDotProductAttention
