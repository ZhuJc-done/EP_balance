"""The sync-free forward must be compute-invariant: outputs + main(e) grads match the reference."""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from eplb import EPLBConfig, ProblemSpec, Topology, solve
from eplb.integration.sync_free import AllToAllAdapter, sync_free_moe_forward
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


def _worker(rank, port, rematerialize=False, overlap=False):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=W)

    unit_expert, unit_prob, tokens, base_w1, base_w2 = _global_data()
    topo = Topology.from_nvlink_rdma(1, W, 1, 8)
    spec = ProblemSpec.uniform_main_placement(E, W, weight_bytes_each=1000, s_tok=1, n_slot=4)
    lam = torch.stack([torch.bincount(unit_expert[r], minlength=E) for r in range(W)]).to(torch.int64)
    plan = solve(Loads(lam), topo, spec, EPLBConfig())
    assert int(plan.num_replicas().sum().item()) > E, "test needs at least one replica"

    w1 = base_w1[rank].clone().requires_grad_(True)
    w2 = base_w2[rank].clone().requires_grad_(True)
    weights_local = {rank: (w1, w2)}
    cap = W * T

    result = sync_free_moe_forward(
        tokens=tokens[rank],
        unit_token_idx=torch.arange(T, dtype=torch.int64),
        unit_expert=unit_expert[rank].to(torch.int64),
        unit_prob=unit_prob[rank].to(torch.float32),
        plan=plan, spec=spec, weights_local=weights_local,
        weight_shapes=[(H, F), (F, H)], batched_mlp_fn=_batched_mlp, cap=cap,
        adapter=AllToAllAdapter(),
        rematerialize=rematerialize,
        overlap=overlap, gated=False, act=torch.relu, transpose_w=False,
    )
    result.sum().backward()

    gathered = [torch.empty(T, H) for _ in range(W)]
    dist.all_gather(gathered, result.detach().contiguous())
    gt_results, gt_w1, gt_w2 = _ground_truth(unit_expert, unit_prob, tokens, base_w1, base_w2)

    if rank == 0:
        got = torch.stack(gathered)
        assert torch.allclose(got, gt_results, atol=1e-4, rtol=1e-3), \
            f"sync_free outputs differ: max={float((got - gt_results).abs().max())}"

    assert torch.allclose(w1.grad, gt_w1[rank].grad, atol=1e-4, rtol=1e-3), \
        f"W1 grad mismatch rank {rank}: max={float((w1.grad - gt_w1[rank].grad).abs().max())}"
    assert torch.allclose(w2.grad, gt_w2[rank].grad, atol=1e-4, rtol=1e-3), \
        f"W2 grad mismatch rank {rank}: max={float((w2.grad - gt_w2[rank].grad).abs().max())}"

    dist.destroy_process_group()


def test_sync_free_is_compute_invariant():
    mp.spawn(_worker, args=(6021, False), nprocs=W, join=True)


def test_sync_free_rematerialize_matches_reference():
    mp.spawn(_worker, args=(6022, True, False), nprocs=W, join=True)


def test_sync_free_overlap_matches_reference():
    # Level B: async re-materialisation + hand-written GEMM backward must match the reference grads
    mp.spawn(_worker, args=(6023, False, True), nprocs=W, join=True)


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
