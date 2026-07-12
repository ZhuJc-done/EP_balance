"""Cluster validation for the NCCL GIN weight-replication backend (EPLB_WEIGHT_COMM=gin).

Mirrors :mod:`tests.test_sync_free`: the GIN forward must be compute-invariant -- outputs and
main(e) gradients match the same single-device ground truth as the ``dist.broadcast`` path.

This exercises real device-initiated GIN put/get, so it needs:
  * >= 2 CUDA GPUs, an NCCL build with ``ginType != NONE`` (NCCL >= 2.30, GDAKI/GIN available), and
  * the built ``nccl_gin`` extension importable.
It is SKIPPED automatically otherwise (e.g. the CPU/gloo CI box), and is therefore NOT a
substitute for running it once on the target cluster.

Run explicitly on the cluster:
    EPLB_WEIGHT_COMM=gin RUN_GIN_TESTS=1 pytest -s tests/test_gin_weights.py
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from eplb import EPLBConfig, ProblemSpec, Topology, solve
from eplb.integration.eplb_manager import AllToAllAdapter, sync_free_moe_forward
from eplb.loads import Loads

W = 2   # ranks / GPUs (keep small: 1 replica is enough to exercise the get/put + reduce)
E = 4
H = 8
F = 16
T = 32


def _batched_mlp(x, w):  # x[S,N,H], w0[S,H,F], w1[S,F,H]
    return torch.bmm(torch.relu(torch.bmm(x, w[0])), w[1])


def _mlp(x, w):
    return torch.relu(x @ w[0]) @ w[1]


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


def _worker(rank, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["EPLB_WEIGHT_COMM"] = "gin"
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=W)

    unit_expert, unit_prob, tokens, base_w1, base_w2 = _global_data()
    topo = Topology.from_nvlink_rdma(1, W, 1, 8)
    spec = ProblemSpec.uniform_main_placement(E, W, weight_bytes_each=1000, s_tok=1, n_slot=4)
    lam = torch.stack([torch.bincount(unit_expert[r], minlength=E) for r in range(W)]).to(torch.int64)
    plan = solve(Loads(lam), topo, spec, EPLBConfig())
    assert int(plan.num_replicas().sum().item()) > E, "test needs at least one replica"

    w1 = base_w1[rank].clone().to(dev).requires_grad_(True)
    w2 = base_w2[rank].clone().to(dev).requires_grad_(True)
    weights_local = {rank: (w1, w2)}

    result = sync_free_moe_forward(
        tokens=tokens[rank].to(dev),
        unit_token_idx=torch.arange(T, dtype=torch.int64, device=dev),
        unit_expert=unit_expert[rank].to(torch.int64).to(dev),
        unit_prob=unit_prob[rank].to(torch.float32).to(dev),
        plan=plan, spec=spec, weights_local=weights_local,
        weight_shapes=[(H, F), (F, H)], batched_mlp_fn=_batched_mlp, cap=W * T,
        adapter=AllToAllAdapter(),
    )
    result.sum().backward()

    gathered = [torch.empty(T, H, device=dev) for _ in range(W)]
    dist.all_gather(gathered, result.detach().contiguous())
    gt_results, gt_w1, gt_w2 = _ground_truth(unit_expert, unit_prob, tokens, base_w1, base_w2)

    if rank == 0:
        got = torch.stack(gathered).cpu()
        assert torch.allclose(got, gt_results, atol=1e-4, rtol=1e-3), \
            f"GIN outputs differ: max={float((got - gt_results).abs().max())}"
    assert torch.allclose(w1.grad.cpu(), gt_w1[rank].grad, atol=1e-4, rtol=1e-3), \
        f"W1 grad mismatch rank {rank}: max={float((w1.grad.cpu() - gt_w1[rank].grad).abs().max())}"
    assert torch.allclose(w2.grad.cpu(), gt_w2[rank].grad, atol=1e-4, rtol=1e-3), \
        f"W2 grad mismatch rank {rank}: max={float((w2.grad.cpu() - gt_w2[rank].grad).abs().max())}"

    dist.destroy_process_group()


def _gin_available() -> bool:
    if os.environ.get("RUN_GIN_TESTS", "0").strip().lower() in ("0", "", "false", "no"):
        return False
    if not (torch.cuda.is_available() and torch.cuda.device_count() >= W):
        return False
    try:
        import nccl_gin  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _gin_available(),
                    reason="needs RUN_GIN_TESTS=1, >=2 GPUs with GIN-capable NCCL, and nccl_gin built")
def test_gin_weights_compute_invariant():
    mp.spawn(_worker, args=(6031,), nprocs=W, join=True)


def test_replica_schedule_device_math():
    """CPU check of the D2H-free replication schedule (no nccl_gin / CUDA needed).

    2 ranks, experts 0,1 main on rank0 and 2,3 on rank1 -> local_slot_of_e = [0,1,0,1].
    From rank 0's view, a slot layout [expert2(remote), expert0(local), empty] must yield a get
    schedule that fetches only the remote slot and gathers the local one on device.
    """
    from eplb.integration.gin_weights import _replica_schedule

    local_slot_of_e = torch.tensor([0, 1, 0, 1], dtype=torch.int64)  # main_rank = [0,0,1,1]
    slot_to_e = torch.tensor([2, 0, -1], dtype=torch.int64)          # rank 0's local slots
    main_of_slot = torch.tensor([1, 0, -1], dtype=torch.int64)

    peers, ls, is_local = _replica_schedule(slot_to_e, main_of_slot, local_slot_of_e, my_rank=0)

    assert peers.dtype == torch.int32
    assert peers.tolist() == [1, -1, -1]          # only slot 0 (expert2 on rank1) is a remote get
    assert ls.tolist() == [0, 0, 0]               # local_slot(e2)=0, local_slot(e0)=0
    assert is_local.tolist() == [False, True, False]  # slot 1 (expert0) filled by on-device gather
