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


@pytest.mark.parametrize("hidden", [16, 512, 1024, 2048, 4096, 5120, 7168, 8192])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dispatch_payload_alignment_matches_deepep_screen(hidden, dtype):
    """The padded dispatch row must land on the alignment the adapter screens for.

    These are two independent encodings of one DeepEP kernel constraint -- the payload builders pad
    the row, ``_deepep_eligible`` decides whether DeepEP is used at all -- and they drifted once:
    the padding targeted 16B while the kernel wants an even number of 16B chunks. Nothing failed.
    Every dispatch and combine quietly took the ``all_to_all_single`` fallback, which reads its
    split sizes on the host, so the integration was inert and each layer paid back the D2H that
    adopting DeepEP was meant to remove. Only a check like this one surfaces that.

    Rows too wide for the kernel's per-warp TMA buffer are a separate matter: declining those is
    correct, and the assertion below deliberately does not demand eligibility for them.
    """
    from eplb.integration.eplb_manager import DEEPEP_ROW_ALIGN, DeepEPAdapter, _payload_pad_cols

    elem = torch.empty((), dtype=dtype).element_size()
    pad_cols = _payload_pad_cols(hidden, elem)
    assert pad_cols >= 1, "the physical-expert id needs a column of its own"

    row_bytes = (hidden + pad_cols) * elem
    assert row_bytes % DEEPEP_ROW_ALIGN == 0, f"H={hidden} {dtype} pads to a {row_bytes}B row"
    if row_bytes // 2 + 8 <= 8192:  # within the TMA buffer, so alignment is the only thing left
        assert DeepEPAdapter._deepep_eligible(torch.empty((4, hidden + pad_cols), dtype=dtype))
