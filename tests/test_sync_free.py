"""The sync-free forward must be compute-invariant: outputs + main(e) grads match the reference."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from eplb import EPLBConfig, ProblemSpec, Topology, solve
from eplb.integration.eplb_manager import AllToAllAdapter, sync_free_moe_forward
from eplb.loads import Loads

W = 4
E = 4
H = 8
F = 16
T = 32


def _mlp(x, w):  # ground-truth convention (no transpose): relu(x @ W1) @ W2
    return torch.relu(x @ w[0]) @ w[1]


def _batched_mlp(x, w):  # x[S,N,H], w0[S,H,F], w1[S,F,H]
    return torch.bmm(torch.relu(torch.bmm(x, w[0])), w[1])


def _grouped_mlp(x, w, offs):  # ragged twin of _batched_mlp; x[T,H] packed slot-major
    h = torch.relu(torch._grouped_mm(x, w[0], offs=offs))
    return torch._grouped_mm(h, w[1], offs=offs)


def _global_data():
    g = torch.Generator().manual_seed(1234)
    probs = torch.tensor([0.7, 0.1, 0.1, 0.1], dtype=torch.float64)
    unit_expert = torch.multinomial(probs, W * T, replacement=True, generator=g).reshape(W, T)
    unit_prob = 0.5 + torch.rand(W, T, generator=g)
    tokens = torch.randn(W, T, H, generator=g)
    base_w1 = torch.randn(E, H, F, generator=g) * 0.1
    base_w2 = torch.randn(E, F, H, generator=g) * 0.1
    return unit_expert, unit_prob, tokens, base_w1, base_w2


def _ground_truth(unit_expert, unit_prob, tokens, base_w1, base_w2):
    gt_w1 = [base_w1[e].clone().requires_grad_(True) for e in range(E)]
    gt_w2 = [base_w2[e].clone().requires_grad_(True) for e in range(E)]
    results = []
    for r in range(W):
        res = torch.zeros(T, H)
        for t in range(T):
            e = int(unit_expert[r, t])
            y = _mlp(tokens[r, t:t + 1], (gt_w1[e], gt_w2[e]))
            res = res.index_add(0, torch.tensor([t]), unit_prob[r, t] * y)
        results.append(res)
    loss = sum(r.sum() for r in results)
    loss.backward()
    return torch.stack(results), gt_w1, gt_w2


def _trace_backward_order(order):
    """Patch the three backward collectives to append their name, and return an undo callable."""
    import eplb.integration.eplb_manager as _em
    import eplb.integration.overlap as _ov

    adapter, transport = _em.AllToAllAdapter, _ov.BroadcastReplicaTransport
    saved = (adapter.dispatch_chunk_bwd, adapter.combine_chunk_bwd, transport.reduce_grads)

    def tag(name, fn):
        def wrapper(self, *a, **k):
            order.append(name)
            return fn(self, *a, **k)
        return wrapper

    adapter.dispatch_chunk_bwd = tag("dispatch_bwd", saved[0])
    adapter.combine_chunk_bwd = tag("combine_bwd", saved[1])
    transport.reduce_grads = tag("reduce", saved[2])

    def undo():
        adapter.dispatch_chunk_bwd, adapter.combine_chunk_bwd, transport.reduce_grads = saved

    return undo


def _worker(rank, port, rematerialize=False, overlap=False, chunks=1, manual=True, order=None,
            transpose_w=False):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    # EPLB_CHUNKS drives the two-chunk overlap path inside sync_free_moe_forward (env read at call time)
    if chunks >= 2:
        os.environ["EPLB_CHUNKS"] = str(chunks)
    else:
        os.environ.pop("EPLB_CHUNKS", None)
    os.environ["EPLB_MANUAL_BWD"] = "1" if manual else "0"
    dist.init_process_group(backend="gloo", rank=rank, world_size=W)

    unit_expert, unit_prob, tokens, base_w1, base_w2 = _global_data()
    topo = Topology.from_nvlink_rdma(1, W, 1, 8)
    spec = ProblemSpec.uniform_main_placement(E, W, weight_bytes_each=1000, s_tok=1, n_slot=4)
    omega = torch.stack(
        [torch.bincount(unit_expert[r], minlength=E) for r in range(W)]
    ).to(torch.int64)
    plan = solve(Loads(omega), topo, spec, EPLBConfig())
    assert int(plan.num_replicas().sum().item()) > E, "test needs at least one replica"

    # transpose_w mirrors Megatron's [out, in] parameter layout, used as x @ W.t(). Same maths, but the
    # backward's Wgrad then has to land in the transposed layout, which is a separate code path.
    tw = (lambda t: t.t().contiguous()) if transpose_w else (lambda t: t.clone())
    w1 = tw(base_w1[rank]).requires_grad_(True)
    w2 = tw(base_w2[rank]).requires_grad_(True)
    weights_local = {rank: (w1, w2)}
    weight_shapes = [(F, H), (H, F)] if transpose_w else [(H, F), (F, H)]
    cap = W * T

    # The weight pull must be kicked off by the block's backward pre-hook, not by the first chunk's
    # backward: that difference is the whole overlap window (reverse combine + scatter backward vs a
    # single Wgrad), and it is invisible in the grads below, so it is asserted directly.
    leases = []
    if overlap:
        import eplb.integration.manual_block as _mb
        import eplb.integration.overlap as _ov
        _orig_lease = _ov._ReplicaLease

        class _TracedLease(_orig_lease):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                leases.append(self)

        # both backends construct the lease; manual_block bound the name at import
        _ov._ReplicaLease = _mb._ReplicaLease = _TracedLease

    undo_order = _trace_backward_order(order) if order is not None else None

    result = sync_free_moe_forward(
        tokens=tokens[rank],
        unit_token_idx=torch.arange(T, dtype=torch.int64),
        unit_expert=unit_expert[rank].to(torch.int64),
        unit_prob=unit_prob[rank].to(torch.float32),
        plan=plan, spec=spec, weights_local=weights_local,
        weight_shapes=weight_shapes, batched_mlp_fn=_batched_mlp, cap=cap,
        adapter=AllToAllAdapter(),
        rematerialize=rematerialize,
        overlap=overlap, gated=False, act=torch.relu, transpose_w=transpose_w,
    )
    result.sum().backward()
    if overlap:
        _ov._ReplicaLease = _mb._ReplicaLease = _orig_lease
        assert len(leases) == 1, f"expected one lease per layer, got {len(leases)}"
        assert leases[0].prefetched, "weight pull was not started by the block's backward pre-hook"
    if undo_order is not None:
        undo_order()
        # Both reverse combines go out up front; each dispatch^-1 follows its chunk's Dgrad, ahead of
        # that chunk's Wgrad; the reduce goes last, once every Wgrad has accumulated. Moving the
        # reduce earlier would put it ahead of a dispatch^-1 that the upstream backward is waiting on.
        assert list(order) == [
            "combine_bwd", "combine_bwd", "dispatch_bwd", "dispatch_bwd", "reduce"
        ], f"backward collectives issued in the wrong order: {list(order)}"

    gathered = [torch.empty(T, H) for _ in range(W)]
    dist.all_gather(gathered, result.detach().contiguous())
    gt_results, gt_w1, gt_w2 = _ground_truth(unit_expert, unit_prob, tokens, base_w1, base_w2)

    if rank == 0:
        got = torch.stack(gathered)
        assert torch.allclose(got, gt_results, atol=1e-4, rtol=1e-3), \
            f"sync_free outputs differ: max={float((got - gt_results).abs().max())}"

    g1, g2 = (w1.grad.t(), w2.grad.t()) if transpose_w else (w1.grad, w2.grad)
    assert torch.allclose(g1, gt_w1[rank].grad, atol=1e-4, rtol=1e-3), \
        f"W1 grad mismatch rank {rank}: max={float((g1 - gt_w1[rank].grad).abs().max())}"
    assert torch.allclose(g2, gt_w2[rank].grad, atol=1e-4, rtol=1e-3), \
        f"W2 grad mismatch rank {rank}: max={float((g2 - gt_w2[rank].grad).abs().max())}"

    dist.destroy_process_group()


def test_sync_free_is_compute_invariant():
    mp.spawn(_worker, args=(6021, False), nprocs=W, join=True)


def test_sync_free_rematerialize_matches_reference():
    mp.spawn(_worker, args=(6022, True, False), nprocs=W, join=True)


def test_sync_free_overlap_matches_reference():
    # Level B: async re-materialisation + hand-written GEMM backward must match the reference grads
    mp.spawn(_worker, args=(6023, False, True), nprocs=W, join=True)


def test_sync_free_two_chunk_matches_reference():
    # EPLB_CHUNKS=2: token-chunked dispatch/compute/combine pipeline (weights re-materialised once,
    # shared by both chunks). On CPU/gloo the comm stream is a no-op, so this validates the chunk
    # split/merge + shared-weight grad math (compute-invariance) rather than the GPU stream overlap.
    mp.spawn(_worker, args=(6024, False, False, 2), nprocs=W, join=True)


def test_sync_free_two_chunk_overlap_matches_reference():
    """EPLB_CHUNKS=2 on the hand-scheduled backward: both chunks share one weight lease.

    This is the configuration the cluster runs use (with GIN swapped in for the broadcast transport,
    which the lease is agnostic to), and it is the combination with the most ways to be silently
    wrong: each chunk's backward must see the same re-acquired stacks, both chunks' Wgrads must
    accumulate before the single reduce-to-main, and the re-acquisition must happen once per layer --
    a per-chunk acquire would issue mismatched collectives. All three show up here as wrong grads."""
    mp.spawn(_worker, args=(6025, False, True, 2, True), nprocs=W, join=True)


def test_sync_free_two_chunk_overlap_transposed_weights():
    """Megatron's [out, in] weight layout through the hand-scheduled backward.

    The Wgrads are produced directly in parameter layout there (swapped bmm operands) so the transport
    can reinterpret the bytes without staging a contiguous copy of the whole slot stack. Getting the
    swap wrong transposes the gradient, which every `transpose_w=False` test above is blind to."""
    mp.spawn(_worker, args=(6028, False, True, 2, True, None, True), nprocs=W, join=True)


def test_sync_free_two_chunk_backward_issue_order():
    """The hand-scheduled backward must issue its collectives in the order the overlap depends on.

    Grads cannot see this: any order produces the same numbers. What differs is what each transfer
    has to hide behind, which is the entire reason the pipeline is hand-scheduled rather than left to
    autograd (which is forced to put the reduce last). The worker asserts the order it observed."""
    mp.spawn(_worker, args=(6027, False, True, 2, True, []), nprocs=W, join=True)


def test_sync_free_two_chunk_overlap_autograd_matches_reference():
    """Same configuration with EPLB_MANUAL_BWD=0, i.e. autograd ordering the backward instead.

    The two backends must be numerically indistinguishable -- the hand-written schedule only moves
    *when* each collective is issued -- so running both against the same reference is what keeps the
    hand-written one honest as it is tuned."""
    mp.spawn(_worker, args=(6026, False, True, 2, False), nprocs=W, join=True)


def test_overlap_backward_matches_autograd_gated_transpose():
    """No-replica, single-process: the hand-written gated/transpose GEMM backward matches autograd."""
    import torch.nn.functional as Fnn

    from eplb.integration.grouped_mlp import grouped_expert_mlp, make_batched_gated_mlp
    from eplb.integration.overlap import overlapped_grouped_expert_mlp

    torch.manual_seed(0)
    S, cap, Hd, Ff, Tt = 3, 8, 6, 4, 17
    recv_slot = torch.randint(0, S, (Tt,), dtype=torch.int64)
    group_sizes = torch.bincount(recv_slot, minlength=S).to(torch.int64)
    recv_tokens = torch.randn(Tt, Hd)

    # Megatron layout: W1 [2F, H] (gated), W2 [H, F]; used as x @ W.t()
    w1 = (torch.randn(S, 2 * Ff, Hd) * 0.1)
    w2 = (torch.randn(S, Hd, Ff) * 0.1)

    w1_ref = w1.clone().requires_grad_(True)
    w2_ref = w2.clone().requires_grad_(True)
    out_ref = grouped_expert_mlp(
        recv_tokens, recv_slot, group_sizes, (w1_ref, w2_ref),
        make_batched_gated_mlp(gated=True, act=Fnn.silu), cap,
    )
    out_ref.sum().backward()

    weights_local = {e: (w1[e].clone().requires_grad_(True), w2[e].clone().requires_grad_(True)) for e in range(S)}
    out_ov = overlapped_grouped_expert_mlp(
        recv_tokens, recv_slot, group_sizes, weights_local,
        slot_to_e=torch.arange(S, dtype=torch.int64),
        main_rank=torch.zeros(S, dtype=torch.int64),
        replicated=[], weight_shapes=[(2 * Ff, Hd), (Hd, Ff)], cap=cap,
        gated=True, act=Fnn.silu, transpose_w=True, my_rank=0, n_slot=S, group=None,
    )
    out_ov.sum().backward()

    assert torch.allclose(out_ov, out_ref, atol=1e-6), \
        f"output mismatch: {float((out_ov - out_ref).abs().max())}"
    for e in range(S):
        assert torch.allclose(weights_local[e][0].grad, w1_ref.grad[e], atol=1e-5), \
            f"W1 grad mismatch slot {e}: {float((weights_local[e][0].grad - w1_ref.grad[e]).abs().max())}"
        assert torch.allclose(weights_local[e][1].grad, w2_ref.grad[e], atol=1e-5), \
            f"W2 grad mismatch slot {e}: {float((weights_local[e][1].grad - w2_ref.grad[e]).abs().max())}"


def test_routing_to_units():
    from eplb.integration.megatron_moe import _routing_to_units

    rmap = torch.tensor([[True, False, True], [False, True, False]])
    probs = torch.tensor([[0.6, 0.0, 0.4], [0.0, 1.0, 0.0]])
    tok, exp, p = _routing_to_units(probs, rmap, num_tokens=2, num_experts=3)
    assert tok.tolist() == [0, 0, 1]
    assert exp.tolist() == [0, 2, 1]
    assert torch.allclose(p, torch.tensor([0.6, 0.4, 1.0]))


def test_eplb_megatron_forward_adds_shared_expert_output(monkeypatch):
    """Apply mode must preserve the native routed + shared expert sum."""
    from types import SimpleNamespace

    import eplb.integration.megatron_moe as binding

    call_order = []
    hidden = torch.randn(2, 1, 3, requires_grad=True)

    def router(x):
        call_order.append("route")
        probs = x.new_tensor([[1.0, 0.0], [0.0, 1.0]])
        return probs, probs.bool()

    def shared_experts_compute(x):
        call_order.append("shared")
        return 3.0 * x

    def fake_sync_free_moe_forward(**kwargs):
        assert kwargs["plan"] == "test-plan"
        assert kwargs["unit_expert"].tolist() == [0, 1]
        return 2.0 * kwargs["tokens"]

    class Rebalancer:
        spec = SimpleNamespace(num_experts=2)

        @staticmethod
        def rebalance(local_row, layer_id, mb, group):
            assert local_row.tolist() == [1, 1]
            return SimpleNamespace(plan="test-plan")

    def local_expert():
        return SimpleNamespace(
            linear_fc1=SimpleNamespace(weight=torch.empty(4, 3)),
            linear_fc2=SimpleNamespace(weight=torch.empty(3, 2)),
        )

    layer = SimpleNamespace(
        config=SimpleNamespace(moe_router_topk=1),
        router=router,
        experts=SimpleNamespace(local_experts=[local_expert(), local_expert()]),
        use_shared_expert=True,
        shared_expert_overlap=False,
        shared_experts_compute=shared_experts_compute,
        _eplb={
            "reb": Rebalancer(),
            "group": None,
            "layer_id": 0,
            "mb": 0,
            "gated": True,
            "act": torch.nn.functional.silu,
            "batched_mlp_fn": None,
            "adapter": object(),
            "rematerialize": False,
            "overlap": False,
        },
    )
    monkeypatch.setattr(binding, "sync_free_moe_forward", fake_sync_free_moe_forward)
    monkeypatch.setattr(binding.profiling, "enabled", lambda: False)

    output, bias = binding.eplb_moe_forward(layer, hidden)
    assert bias is None
    assert call_order == ["shared", "route"]
    assert torch.allclose(output, 5.0 * hidden)

    output.sum().backward()
    assert torch.allclose(hidden.grad, torch.full_like(hidden, 5.0))


def test_eplb_binding_rejects_shared_expert_overlap():
    """Megatron's dispatcher owns the shared-expert overlap state machine."""
    from types import SimpleNamespace

    from eplb.integration.megatron_moe import bind_eplb_to_moe_layer

    layer = SimpleNamespace(
        config=SimpleNamespace(moe_router_topk=2),
        use_shared_expert=True,
        shared_expert_overlap=True,
        shared_experts_compute=lambda hidden: hidden,
    )
    with pytest.raises(NotImplementedError, match="moe_shared_expert_overlap=False"):
        bind_eplb_to_moe_layer(layer, rebalancer=object(), ep_group=None)


@pytest.mark.parametrize("hidden", [16, 512, 1024, 2048, 4096, 5120, 7168, 8192])
@pytest.mark.parametrize("dtype,eligible", [(torch.bfloat16, True), (torch.float16, False)])
def test_elastic_payload_requires_aligned_bf16(hidden, dtype, eligible):
    """ElasticBuffer accepts only BF16 rows aligned to one int4."""
    from eplb.integration.eplb_manager import DEEPEP_ROW_ALIGN, DeepEPAdapter, _payload_pad_cols

    elem = torch.empty((), dtype=dtype).element_size()
    pad_cols = _payload_pad_cols(hidden, elem)

    row_bytes = (hidden + pad_cols) * elem
    assert row_bytes % DEEPEP_ROW_ALIGN == 0, f"H={hidden} {dtype} pads to a {row_bytes}B row"
    assert DeepEPAdapter._deepep_eligible(
        torch.empty((4, hidden + pad_cols), dtype=dtype)
    ) is eligible


def test_elastic_buffer_is_shared_and_uses_static_shape(monkeypatch):
    """Adapters with one layout share one ElasticBuffer allocation."""
    import sys
    import types

    import eplb.integration.eplb_manager as manager

    calls = []
    fake_deep_ep = types.ModuleType("deep_ep")

    def fake_buffer(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    fake_deep_ep.ElasticBuffer = fake_buffer
    fake_deep_ep.topk_idx_t = torch.int64
    monkeypatch.setitem(sys.modules, "deep_ep", fake_deep_ep)
    monkeypatch.setenv("EPLB_DEEPEP_HYBRID", "1")
    monkeypatch.delenv("EPLB_DEEPEP_NUM_SMS", raising=False)
    manager._ELASTIC_BUFFERS.clear()
    group = object()
    payload = torch.zeros((4, 16), dtype=torch.bfloat16)
    a0 = manager.DeepEPAdapter(max_tokens_per_rank=8)
    a1 = manager.DeepEPAdapter(max_tokens_per_rank=8)
    assert a0._num_sms == a1._num_sms == 16
    assert a0._get_buffer(group, payload) is a1._get_buffer(group, payload)
    assert len(calls) == 1
    assert calls[0][0] == (group,)
    assert calls[0][1] == {
        "num_max_tokens_per_rank": 8,
        "hidden": 16,
        "num_topk": 1,
        "deterministic": False,
        "allow_hybrid_mode": True,
    }
    manager._ELASTIC_BUFFERS.clear()


def test_elastic_adapter_configuration_is_strict(monkeypatch):
    """Elastic mode rejects legacy or host-synchronizing configurations."""
    import sys
    import types

    from eplb.integration.grouped_mlp import ragged_available
    from eplb.integration.megatron_moe import _make_adapter

    fake_deep_ep = types.ModuleType("deep_ep")
    fake_deep_ep.ElasticBuffer = object
    fake_deep_ep.topk_idx_t = torch.int64
    monkeypatch.setitem(sys.modules, "deep_ep", fake_deep_ep)
    monkeypatch.setenv("EPLB_ADAPTER", "deepep")
    for key in (
        "EPLB_CAP", "EPLB_WEIGHT_COMM", "EPLB_GIN_FENCE", "EPLB_PROFILE",
        "EPLB_DEBUG_TIMING", "PROFILE_TRACE", "EPLB_DEEPEP_ALLOW_MNNVL",
    ):
        monkeypatch.delenv(key, raising=False)

    # A cap is only mandatory for the padded expert path; the ragged one sizes itself on device.
    monkeypatch.setenv("EPLB_GROUPED_GEMM", "0")
    with pytest.raises(ValueError, match="EPLB_CAP"):
        _make_adapter()
    monkeypatch.delenv("EPLB_GROUPED_GEMM")
    if ragged_available():
        with pytest.raises(ValueError, match="WEIGHT_COMM"):
            _make_adapter()   # no EPLB_CAP set, and none needed

    monkeypatch.setenv("EPLB_CAP", "16")
    with pytest.raises(ValueError, match="WEIGHT_COMM"):
        _make_adapter()
    monkeypatch.setenv("EPLB_WEIGHT_COMM", "gin")
    with pytest.raises(ValueError, match="GIN_FENCE"):
        _make_adapter()
    monkeypatch.setenv("EPLB_GIN_FENCE", "signal")
    assert _make_adapter().uses_padded_layout()
    monkeypatch.setenv("EPLB_DEBUG_TIMING", "1")
    with pytest.warns(RuntimeWarning, match="synchronizes once per MoE invocation"):
        assert _make_adapter().uses_padded_layout()
    monkeypatch.setenv("EPLB_PROFILE", "1")
    with pytest.raises(ValueError, match="EPLB_PROFILE=0"):
        _make_adapter()
    monkeypatch.delenv("EPLB_PROFILE")
    monkeypatch.delenv("EPLB_DEBUG_TIMING")
    monkeypatch.setenv("EPLB_DEEPEP_ALLOW_MNNVL", "1")
    with pytest.raises(ValueError, match="legacy"):
        _make_adapter()


def test_elastic_dispatch_uses_synthetic_experts_padding_and_no_sync(monkeypatch):
    """Dispatch keeps counts on device and autograd uses combine/dispatch transposes."""
    import sys
    import types

    import eplb.integration.eplb_manager as manager

    class FakeElastic:
        def __init__(self, group, **kwargs):
            self.default_max = kwargs["num_max_tokens_per_rank"]
            self.calls = []

        def dispatch(self, x, topk_idx=None, handle=None, **kwargs):
            self.calls.append(("dispatch", dict(kwargs), topk_idx, handle))
            if handle is not None:
                topk_idx = handle.topk_idx
            n, h = topk_idx.shape[0], x.shape[1]
            m = self.default_max
            recv = x.new_zeros((m, h))
            recv[:n] = x[:n]
            recv_idx = torch.full((m, 1), -1, dtype=torch.int64)
            recv_idx[:n] = topk_idx
            if handle is None:
                counts = torch.bincount(topk_idx[:, 0], minlength=2).to(torch.int64)
                handle = types.SimpleNamespace(
                    topk_idx=topk_idx.clone(),
                    num_experts=2,
                    psum_num_recv_tokens_per_scaleup_rank=torch.tensor([n], device=x.device),
                    psum_num_recv_tokens_per_expert=torch.cumsum(counts, 0),
                )
            return recv, recv_idx, None, handle, None

        def combine(self, x, handle, **kwargs):
            self.calls.append(("combine", dict(kwargs), None, handle))
            return x[:handle.topk_idx.shape[0]], None, None

    fake_deep_ep = types.ModuleType("deep_ep")
    fake_deep_ep.ElasticBuffer = FakeElastic
    fake_deep_ep.topk_idx_t = torch.int64
    monkeypatch.setitem(sys.modules, "deep_ep", fake_deep_ep)
    monkeypatch.setenv("EPLB_DEEPEP_NUM_SMS", "8")
    manager._ELASTIC_BUFFERS.clear()

    adapter = manager.DeepEPAdapter(max_tokens_per_rank=4)
    payload = torch.randn((3, 256), dtype=torch.bfloat16, requires_grad=True)
    routes = torch.tensor([1, 0, 1], dtype=torch.int64)
    recv = adapter.dispatch_chunk(
        payload, torch.tensor([3]), torch.tensor([3]), object(), tag=7,
        route_idx=routes, n_slot=2, cap=2,
    )
    slot, valid, sizes = adapter.recv_layout(7)
    assert recv.shape == (4, 256)
    assert slot.tolist() == [1, 0, 1, 0]
    assert valid.tolist() == [True, True, True, False]
    assert sizes.tolist() == [1, 2]

    combined = adapter.combine_chunk(
        recv[:, :8] * 2, torch.tensor([3]), torch.tensor([3]), object(), tag=7
    )
    combined.sum().backward()
    assert torch.equal(payload.grad[:, :8], torch.full_like(payload.grad[:, :8], 2))
    assert torch.count_nonzero(payload.grad[:, 8:]) == 0

    calls = adapter._buffer.calls
    first = calls[0]
    assert torch.equal(first[2].flatten(), routes)
    assert first[1]["num_experts"] == 2
    assert first[1]["num_max_tokens_per_rank"] == 4
    assert first[1]["expert_alignment"] == 1
    assert first[1]["do_cpu_sync"] is False
    assert first[1]["do_expand"] is False
    assert all(call[1]["num_sms"] == 8 for call in calls)
    cached = [c for c in calls if c[0] == "dispatch" and c[3] is not None]
    assert cached and cached[0][1]["do_cpu_sync"] is False
    assert cached[0][1]["do_expand"] is False
    with pytest.raises(RuntimeError, match="EPLB_CAP"):
        adapter.dispatch_chunk(
            payload.detach(), torch.tensor([3]), torch.tensor([3]), adapter._group, tag=8,
            route_idx=routes, n_slot=2, cap=1,
        )
    manager._ELASTIC_BUFFERS.clear()


@pytest.mark.parametrize(
    "chunks,overlap,manual",
    [(1, False, False), (2, False, False), (2, True, False), (2, True, True)],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_elastic_padded_layout_matches_reference(monkeypatch, chunks, overlap, manual, device):
    """Plain, overlapped and manual chunk paths ignore ElasticBuffer padding.

    On CUDA this is also the end-to-end cover for the ragged grouped GEMM: all three compute
    paths take it there, while CPU keeps exercising the padded fallback.
    """
    import dataclasses
    import sys
    import types

    import eplb.integration.eplb_manager as manager
    from eplb.integration.grouped_mlp import ragged_available
    from eplb.plan import Plan

    if device == "cuda" and not ragged_available():
        pytest.skip("ragged path needs torch._grouped_mm (CUDA SM90+, 16-bit float)")

    class FakeElastic:
        def __init__(self, group, **kwargs):
            self.max_tokens = kwargs["num_max_tokens_per_rank"]

        def dispatch(self, x, topk_idx=None, handle=None, **kwargs):
            if handle is not None:
                topk_idx = handle.topk_idx
            n, h = topk_idx.shape[0], x.shape[1]
            recv = x.new_zeros((self.max_tokens, h))
            recv[:n] = x[:n]
            recv_idx = torch.full((self.max_tokens, 1), -1, dtype=torch.int64, device=x.device)
            recv_idx[:n] = topk_idx
            if handle is None:
                counts = torch.bincount(topk_idx[:, 0], minlength=2).to(torch.int64)
                handle = types.SimpleNamespace(
                    topk_idx=topk_idx.clone(),
                    num_experts=2,
                    psum_num_recv_tokens_per_scaleup_rank=torch.tensor([n], device=x.device),
                    psum_num_recv_tokens_per_expert=torch.cumsum(counts, 0),
                )
            return recv, recv_idx, None, handle, None

        def combine(self, x, handle, **kwargs):
            return x[:handle.topk_idx.shape[0]], None, None

    fake_deep_ep = types.ModuleType("deep_ep")
    fake_deep_ep.ElasticBuffer = FakeElastic
    fake_deep_ep.topk_idx_t = torch.int64
    monkeypatch.setitem(sys.modules, "deep_ep", fake_deep_ep)
    monkeypatch.setenv("EPLB_CHUNKS", str(chunks))
    monkeypatch.setenv("EPLB_MANUAL_BWD", "1" if manual else "0")
    manager._ELASTIC_BUFFERS.clear()

    torch.manual_seed(7)
    n, e, h, f = 8, 2, 16, 32   # widths are 16-byte aligned so CUDA takes the ragged path
    unit_expert = torch.tensor([0, 1, 0, 1, 1, 0, 0, 1], dtype=torch.int64)
    spec = ProblemSpec.uniform_main_placement(e, 1, weight_bytes_each=1, s_tok=1, n_slot=2)
    topo = Topology.from_nvlink_rdma(1, 1, 1, 1)
    plan = solve(
        Loads(torch.bincount(unit_expert, minlength=e).reshape(1, e)),
        topo, spec, EPLBConfig(),
    )
    plan = Plan(x=plan.x.to(device), q=plan.q.to(device), theta=plan.theta)
    spec = dataclasses.replace(
        spec, main_rank=spec.main_rank.to(device), weight_bytes=spec.weight_bytes.to(device)
    )
    unit_expert = unit_expert.to(device)

    tokens = torch.randn(n, h, dtype=torch.bfloat16, device=device, requires_grad=True)
    w0 = [torch.randn(h, f, dtype=torch.bfloat16, device=device).requires_grad_(True) for _ in range(e)]
    w1 = [torch.randn(f, h, dtype=torch.bfloat16, device=device).requires_grad_(True) for _ in range(e)]
    weights = {i: (w0[i], w1[i]) for i in range(e)}

    # Count the ragged GEMMs so a silent fall back to the padded path cannot pass as coverage.
    ragged_calls = []
    if device == "cuda":
        real_grouped_mm = torch._grouped_mm

        def counting_grouped_mm(*args, **kwargs):
            ragged_calls.append(1)
            return real_grouped_mm(*args, **kwargs)

        monkeypatch.setattr(torch, "_grouped_mm", counting_grouped_mm)

    ref_tokens = tokens.detach().clone().requires_grad_(True)
    rw0 = [w.detach().clone().requires_grad_(True) for w in w0]
    rw1 = [w.detach().clone().requires_grad_(True) for w in w1]
    ref = torch.zeros_like(ref_tokens)
    for expert in range(e):
        idx = torch.nonzero(unit_expert == expert, as_tuple=False).flatten()
        y = torch.relu(ref_tokens[idx] @ rw0[expert]) @ rw1[expert]
        ref = ref.index_copy(0, idx, y)

    got = sync_free_moe_forward(
        tokens=tokens,
        unit_token_idx=torch.arange(n, device=device),
        unit_expert=unit_expert,
        unit_prob=torch.ones(n, dtype=torch.bfloat16, device=device),
        plan=plan, spec=spec, weights_local=weights,
        weight_shapes=[torch.Size((h, f)), torch.Size((f, h))],
        batched_mlp_fn=_batched_mlp, cap=n, grouped_mlp_fn=_grouped_mlp,
        adapter=manager.DeepEPAdapter(max_tokens_per_rank=n),
        overlap=overlap, gated=False, act=torch.relu,
    )
    assert torch.allclose(got, ref, atol=2e-2, rtol=2e-2)
    if device == "cuda":
        assert ragged_calls, "expected the ragged grouped GEMM to run on CUDA"
    got.float().sum().backward()
    ref.float().sum().backward()
    assert torch.allclose(tokens.grad, ref_tokens.grad, atol=2e-2, rtol=2e-2)
    for expert in range(e):
        assert torch.allclose(w0[expert].grad, rw0[expert].grad, atol=2e-2, rtol=2e-2)
        assert torch.allclose(w1[expert].grad, rw1[expert].grad, atol=2e-2, rtol=2e-2)
    manager._ELASTIC_BUFFERS.clear()
