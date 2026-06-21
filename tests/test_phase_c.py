"""Phase C correctness: the replication dispatcher is compute-invariant (outputs + W_e grad match a single instance)."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from eplb import EPLBConfig, ProblemSpec, Topology, solve
from eplb.integration.dispatcher import replicated_moe_forward
from eplb.loads import Loads

W = 4          # EP world size (one NVLink domain)
E = 4          # experts (main(e) = e)
H = 8          # hidden size
F = 16         # expert ffn size
T = 32         # tokens per rank (top_k = 1 -> one unit per token)


def _mlp(x, w):
    return torch.relu(x @ w[0]) @ w[1]


def _global_data():
    """Deterministic global routing + weights, identical on every rank."""
    g = torch.Generator().manual_seed(1234)
    probs = torch.tensor([0.7, 0.1, 0.1, 0.1], dtype=torch.float64)  # expert 0 is hot
    unit_expert = torch.multinomial(probs, W * T, replacement=True, generator=g).reshape(W, T)
    unit_prob = 0.5 + torch.rand(W, T, generator=g)
    tokens = torch.randn(W, T, H, generator=g)
    base_w1 = torch.randn(E, H, F, generator=g) * 0.1
    base_w2 = torch.randn(E, F, H, generator=g) * 0.1
    return unit_expert, unit_prob, tokens, base_w1, base_w2


def _ground_truth(unit_expert, unit_prob, tokens, base_w1, base_w2):
    """Single-instance reference: every expert computed once, full-batch loss."""
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


def _worker(rank, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=W)

    unit_expert, unit_prob, tokens, base_w1, base_w2 = _global_data()

    topo = Topology.from_nvlink_rdma(num_nodes=1, gpus_per_node=W, intra_cost=1, inter_cost=8)
    spec = ProblemSpec.uniform_main_placement(
        num_experts=E, num_ranks=W, weight_bytes_each=1000, s_tok=1, n_slot=4,
    )
    lam = torch.stack([torch.bincount(unit_expert[r], minlength=E) for r in range(W)]).to(torch.int64)
    plan = solve(Loads(lam), topo, spec, EPLBConfig())
    assert int(plan.num_replicas().sum().item()) > E, "test needs at least one replica"

    # this rank owns expert `rank` (uniform main placement)
    w1 = base_w1[rank].clone().requires_grad_(True)
    w2 = base_w2[rank].clone().requires_grad_(True)
    weights_local = {rank: (w1, w2)}

    result = replicated_moe_forward(
        tokens=tokens[rank],
        unit_token_idx=torch.arange(T, dtype=torch.int64),
        unit_expert=unit_expert[rank].to(torch.int64),
        unit_prob=unit_prob[rank].to(torch.float32),
        plan=plan, spec=spec, weights_local=weights_local,
        weight_shapes=[(H, F), (F, H)], mlp_fn=_mlp,
    )
    result.sum().backward()

    # gather all ranks' outputs and compare to the single-instance reference
    gathered = [torch.empty(T, H) for _ in range(W)]
    dist.all_gather(gathered, result.detach().contiguous())
    gt_results, gt_w1, gt_w2 = _ground_truth(unit_expert, unit_prob, tokens, base_w1, base_w2)

    if rank == 0:
        got = torch.stack(gathered)
        assert torch.allclose(got, gt_results, atol=1e-4, rtol=1e-3), \
            f"outputs differ: max={float((got - gt_results).abs().max())}"

    # the aggregated gradient on main(e)=rank must match the full-batch reference
    assert torch.allclose(w1.grad, gt_w1[rank].grad, atol=1e-4, rtol=1e-3), \
        f"W1 grad mismatch on rank {rank}: max={float((w1.grad - gt_w1[rank].grad).abs().max())}"
    assert torch.allclose(w2.grad, gt_w2[rank].grad, atol=1e-4, rtol=1e-3), \
        f"W2 grad mismatch on rank {rank}: max={float((w2.grad - gt_w2[rank].grad).abs().max())}"

    dist.destroy_process_group()


def test_replication_is_compute_invariant():
    mp.spawn(_worker, args=(6005,), nprocs=W, join=True)


def test_routing_to_units():
    from eplb.integration.megatron_moe import _routing_to_units

    rmap = torch.tensor([[True, False, True], [False, True, False]])
    probs = torch.tensor([[0.6, 0.0, 0.4], [0.0, 1.0, 0.0]])
    tok, exp, p = _routing_to_units(probs, rmap, num_tokens=2, num_experts=3)
    assert tok.tolist() == [0, 0, 1]
    assert exp.tolist() == [0, 2, 1]
    assert torch.allclose(p, torch.tensor([0.6, 0.4, 1.0]))


def test_build_expert_mlp_fn_gated():
    import torch.nn.functional as Fnn

    from eplb.integration.megatron_moe import build_expert_mlp_fn

    class _Cfg:
        gated_linear_unit = True
        activation_func = staticmethod(Fnn.silu)

    fn = build_expert_mlp_fn(_Cfg())
    x = torch.randn(5, H)
    w1 = torch.randn(2 * F, H)   # Megatron Linear: [out, in]
    w2 = torch.randn(H, F)
    got = fn(x, (w1, w2))

    h = x @ w1.t()
    gate, up = torch.chunk(h, 2, dim=-1)
    ref = (Fnn.silu(gate) * up) @ w2.t()
    assert torch.allclose(got, ref, atol=1e-6)
