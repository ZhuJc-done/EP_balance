import torch

from eplb.integration import profiling
from eplb.integration import megatron_timing
from eplb.integration.megatron_timing import NativeMoETimingBinding


class _AttentionBoundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, events):
        ctx.events = events
        return hidden_states.view_as(hidden_states)

    @staticmethod
    def backward(ctx, grad_hidden_states):
        ctx.events.append("attention_backward")
        return grad_hidden_states, None


class _DelayedWgradBoundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, events):
        ctx.events = events
        return hidden_states.view_as(hidden_states)

    @staticmethod
    def backward(ctx, grad_hidden_states):
        ctx.events.append("delayed_wgrad")
        return grad_hidden_states, None


class _ReverseDispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, probs, events):
        ctx.events = events
        return hidden_states.view_as(hidden_states), probs.view_as(probs)

    @staticmethod
    def backward(ctx, grad_hidden_states, grad_probs):
        ctx.events.append("reverse_dispatch")
        return grad_hidden_states, grad_probs, None


class _Router(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * 0.125


class _Experts(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * 2.0, None


class _TokenDispatcher:
    def __init__(self, events):
        self.events = events

    def token_dispatch(self, hidden_states, probs):
        return _ReverseDispatch.apply(hidden_states, probs, self.events)


class _FakeMoELayer(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.events = events
        self.router = _Router()
        self.experts = _Experts()
        self.token_dispatcher = _TokenDispatcher(events)

    def dispatch(self, hidden_states, probs):
        # Mirrors Megatron's optional delayed-Wgrad node, which is placed
        # outside token_dispatch and must not extend moe_bwd_total.
        hidden_states = _DelayedWgradBoundary.apply(hidden_states, self.events)
        return self.token_dispatcher.token_dispatch(hidden_states, probs)

    def combine(self, output):
        return output * 3.0

    def forward(self, hidden_states):
        probs = self.router(hidden_states)
        dispatched, dispatched_probs = self.dispatch(hidden_states, probs)
        output, bias = self.experts(dispatched + dispatched_probs)
        return self.combine(output), bias


def test_native_totals_end_at_reverse_dispatch_before_delayed_wgrad_and_attention(
    monkeypatch,
):
    events = []

    monkeypatch.setattr(profiling, "debug_enabled", lambda: True)
    monkeypatch.setattr(profiling, "begin_debug_window", lambda: None)
    monkeypatch.setattr(
        profiling,
        "start_debug_interval",
        lambda **_kwargs: events.append("timer_start") or object(),
    )

    def finish(name, _start, **_kwargs):
        events.append(name)

    def emit(context="", *, mode="apply"):
        events.append(f"emit:{mode}:{context}")

    monkeypatch.setattr(profiling, "finish_debug_interval", finish)
    monkeypatch.setattr(profiling, "emit_backward_debug", emit)
    monkeypatch.setattr(
        profiling,
        "maybe_summary",
        lambda _logger, *, context="", missing=None: events.append(f"summary:{context}"),
    )

    layer = _FakeMoELayer(events)
    binding = NativeMoETimingBinding(
        layer, layer_id=2, mode="off", logger=None
    )
    try:
        source = torch.randn(4, 8, requires_grad=True)
        hidden_states = _AttentionBoundary.apply(source, events)
        output, bias = layer(hidden_states)

        assert bias is None
        assert "native/moe_fwd_total" in events
        assert "summary:mode=off layer=2 mb=0" in events

        events.clear()
        output.sum().backward()

        combine_bwd = events.index("native/combine_bwd")
        expert_bwd = events.index("native/expert_bwd")
        reverse = events.index("reverse_dispatch")
        dispatch_bwd = events.index("native/dispatch_bwd")
        endpoint = events.index("native/moe_bwd_total")
        delayed_wgrad = events.index("delayed_wgrad")
        attention = events.index("attention_backward")
        assert combine_bwd < expert_bwd < reverse < dispatch_bwd
        assert dispatch_bwd < endpoint < delayed_wgrad < attention
        assert "emit:off:layer=2 mb=0" in events
    finally:
        binding.remove()


def _reset_wire_patch(monkeypatch):
    monkeypatch.setattr(megatron_timing, "_active_wire_phase", None)
    monkeypatch.setattr(megatron_timing, "_original_all_to_all_single", None)
    monkeypatch.setattr(megatron_timing, "_wire_patch_users", 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group=None: 4)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda _group=None: 1)


def test_native_wire_timer_wraps_only_sync_collective(monkeypatch):
    _reset_wire_patch(monkeypatch)
    calls = []
    samples = []

    def collective(*args, **kwargs):
        calls.append((args, kwargs))
        return None

    monkeypatch.setattr(torch.distributed, "all_to_all_single", collective)
    monkeypatch.setattr(
        profiling,
        "start_debug_interval",
        lambda **kwargs: ("start", kwargs["device"]),
    )
    monkeypatch.setattr(
        profiling,
        "finish_debug_interval",
        lambda name, start, **kwargs: samples.append((name, start, kwargs)),
    )

    megatron_timing._install_wire_timing()
    try:
        inp = torch.zeros((4, 8), dtype=torch.float32)
        out = torch.empty_like(inp)
        with megatron_timing._wire_scope("native/dispatch_wire"):
            torch.distributed.all_to_all_single(
                out,
                inp,
                output_split_sizes=[1, 1, 1, 1],
                input_split_sizes=[1, 1, 1, 1],
                group=object(),
            )
    finally:
        megatron_timing._uninstall_wire_timing()

    assert len(calls) == 1
    assert samples == [
        (
            "native/dispatch_wire",
            ("start", inp.device),
            {"payload_bytes": 3 * 8 * 4},
        )
    ]


def test_native_async_wire_timer_ends_after_work_wait(monkeypatch):
    _reset_wire_patch(monkeypatch)
    events = []

    class Work:
        def wait(self):
            events.append("wait")
            return True

    monkeypatch.setattr(
        torch.distributed,
        "all_to_all_single",
        lambda *args, **kwargs: Work(),
    )
    monkeypatch.setattr(
        profiling,
        "start_debug_interval",
        lambda **_kwargs: events.append("start") or object(),
    )
    monkeypatch.setattr(
        profiling,
        "finish_debug_interval",
        lambda name, _start, **_kwargs: events.append(f"finish:{name}"),
    )

    megatron_timing._install_wire_timing()
    try:
        inp = torch.zeros((4, 8))
        with megatron_timing._wire_scope("native/combine_wire"):
            work = torch.distributed.all_to_all_single(
                torch.empty_like(inp),
                inp,
                output_split_sizes=[1, 1, 1, 1],
                input_split_sizes=[1, 1, 1, 1],
                group=object(),
                async_op=True,
            )
            assert events == ["start"]
            work.wait()
    finally:
        megatron_timing._uninstall_wire_timing()

    assert events == ["start", "wait", "finish:native/combine_wire"]
