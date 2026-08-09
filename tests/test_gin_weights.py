"""Cluster validation for the device-initiated apply-mode backends, which carry different traffic:
``nccl_gin`` (``EPLB_WEIGHT_COMM=gin``) pulls replica expert weights from ``main(e)`` and reduces
their gradients back to it, while DeepEP (``EPLB_ADAPTER=deepep``) moves only the routed tokens.

Mirrors :mod:`tests.test_sync_free`: each backend must be compute-invariant -- outputs and main(e)
gradients match the same single-device ground truth as the ``dist.broadcast`` + ``all_to_all``
path. :mod:`tests.test_sync_free` runs on gloo/CPU and pins ``AllToAllAdapter``, so neither backend
has any coverage there; this module is the only place they are checked.

These exercise real device-initiated GIN put/get and DeepEP kernels, so they need:
  * >= 2 CUDA GPUs, an NCCL build with ``ginType != NONE`` (NCCL >= 2.30, GDAKI/GIN available),
  * the built ``nccl_gin`` extension importable, and
  * the ``deep_ep`` package for the DeepEP case.
They are SKIPPED automatically otherwise (e.g. the CPU/gloo CI box), and are therefore NOT a
substitute for running them once on the target cluster.

Run explicitly on the cluster:
    EPLB_WEIGHT_COMM=gin RUN_GIN_TESTS=1 pytest -s tests/test_gin_weights.py
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from eplb import EPLBConfig, Plan, ProblemSpec, Topology, solve
from eplb.integration.eplb_manager import AllToAllAdapter, DeepEPAdapter, sync_free_moe_forward
from eplb.loads import Loads

W = 2   # ranks / GPUs (keep small: 1 replica is enough to exercise the get/put + reduce)
E = 4
H = 16
F = 32
T = 32


def _batched_mlp(x, w):  # x[S,N,H], w0[S,H,F], w1[S,F,H]
    return torch.bmm(torch.relu(torch.bmm(x, w[0])), w[1])


def _mlp(x, w):
    return torch.relu(x @ w[0]) @ w[1]


def _global_data(dtype=torch.float32):
    g = torch.Generator().manual_seed(1234)
    probs = torch.tensor([0.7, 0.1, 0.1, 0.1], dtype=torch.float64)
    unit_expert = torch.multinomial(probs, W * T, replacement=True, generator=g).reshape(W, T)
    unit_prob = 0.5 + torch.rand(W, T, generator=g)
    # Round to the transport dtype here so the fp32 reference below consumes exactly these values:
    # the residual gap is then accumulation order plus the output rounding, not input quantisation.
    tokens = torch.randn(W, T, H, generator=g).to(dtype)
    base_w1 = (torch.randn(E, H, F, generator=g) * 0.1).to(dtype)
    base_w2 = (torch.randn(E, F, H, generator=g) * 0.1).to(dtype)
    return unit_expert, unit_prob, tokens, base_w1, base_w2


def _ground_truth(unit_expert, unit_prob, tokens, base_w1, base_w2):
    tokens, base_w1, base_w2 = tokens.float(), base_w1.float(), base_w2.float()
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


def _worker(rank, port, transport="alltoall", dtype=torch.float32, fence="barrier", chunks=1,
            manual=True, lsa=True):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["EPLB_WEIGHT_COMM"] = "gin"
    os.environ["EPLB_GIN_FENCE"] = fence
    os.environ["EPLB_GIN_LSA"] = "1" if lsa else "0"
    if chunks >= 2:
        os.environ["EPLB_CHUNKS"] = str(chunks)
    else:
        os.environ.pop("EPLB_CHUNKS", None)
    os.environ["EPLB_MANUAL_BWD"] = "1" if manual else "0"
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=W)

    unit_expert, unit_prob, tokens, base_w1, base_w2 = _global_data(dtype)
    topo = Topology.from_nvlink_rdma(1, W, 1, 8)
    spec = ProblemSpec.uniform_main_placement(E, W, weight_bytes_each=1000, s_tok=1, n_slot=4)
    omega = torch.stack(
        [torch.bincount(unit_expert[r], minlength=E) for r in range(W)]
    ).to(torch.int64)
    plan = solve(Loads(omega), topo, spec, EPLBConfig())
    assert int(plan.num_replicas().sum().item()) > E, "test needs at least one replica"
    # Solve on the host so every rank gets a bit-identical plan without relying on the GPU solver
    # being reproducible across devices, then move it where the forward needs it: the placement and
    # quota tables are indexed by device tensors, and in a real run the rebalancer has already
    # produced them on device.
    spec = ProblemSpec(E, spec.main_rank.to(dev), spec.weight_bytes.to(dev), spec.s_tok, spec.n_slot)
    plan = Plan(plan.x.to(dev), plan.q.to(dev), plan.theta)

    # weights_local is keyed by *expert*, and main(e) = e % W, so with E > W each rank is main of
    # several experts and has to supply params for all of them. A missing entry is not an error:
    # its slot stays zero-filled and only the output comparison notices.
    my_experts = [e for e in range(E) if e % W == rank]
    weights_local = {
        e: (base_w1[e].clone().to(dev).requires_grad_(True),
            base_w2[e].clone().to(dev).requires_grad_(True))
        for e in my_experts
    }

    local_tokens = tokens[rank].to(dev)
    if transport == "deepep":
        assert DeepEPAdapter._deepep_eligible(local_tokens), (
            f"ElasticBuffer requires BF16 aligned rows, got dtype={dtype}, "
            f"row={H * local_tokens.element_size()}B"
        )
        adapter = DeepEPAdapter()
    else:
        adapter = AllToAllAdapter()

    result = sync_free_moe_forward(
        tokens=local_tokens,
        unit_token_idx=torch.arange(T, dtype=torch.int64, device=dev),
        unit_expert=unit_expert[rank].to(torch.int64).to(dev),
        unit_prob=unit_prob[rank].to(torch.float32).to(dev),
        plan=plan, spec=spec, weights_local=weights_local,
        weight_shapes=[(H, F), (F, H)], batched_mlp_fn=_batched_mlp, cap=W * T,
        adapter=adapter,
        # `overlap` is left False on purpose: selecting the GIN weight backend is enough to route
        # through GinReplicaTransport, which holds no replica clone and re-pulls in backward.
        gated=False, act=torch.relu, transpose_w=False,
    )
    result.sum().backward()

    gathered = [torch.empty(T, H, device=dev, dtype=result.dtype) for _ in range(W)]
    dist.all_gather(gathered, result.detach().contiguous())
    gt_results, gt_w1, gt_w2 = _ground_truth(unit_expert, unit_prob, tokens, base_w1, base_w2)

    # bf16 keeps ~3 decimal digits, so the bar is looser there -- still far tighter than the
    # order-1 errors a routing, placement or grad-reduction bug produces.
    atol, rtol = (3e-2, 3e-2) if dtype is torch.bfloat16 else (1e-4, 1e-3)
    if rank == 0:
        got = torch.stack(gathered).float().cpu()
        assert torch.allclose(got, gt_results, atol=atol, rtol=rtol), \
            f"GIN outputs differ: max={float((got - gt_results).abs().max())}"
    for e, (w1, w2) in weights_local.items():
        assert torch.allclose(w1.grad.float().cpu(), gt_w1[e].grad, atol=atol, rtol=rtol), \
            f"W1 grad mismatch e{e} rank {rank}: max={float((w1.grad.float().cpu() - gt_w1[e].grad).abs().max())}"
        assert torch.allclose(w2.grad.float().cpu(), gt_w2[e].grad, atol=atol, rtol=rtol), \
            f"W2 grad mismatch e{e} rank {rank}: max={float((w2.grad.float().cpu() - gt_w2[e].grad).abs().max())}"

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
    """The GIN path end to end: forward pulls replica slots, backward re-pulls them and reduces.

    Outputs and main(e) gradients must match the single-device reference. Cluster-only (real GIN
    get/put plus the side-stream re-pull)."""
    mp.spawn(_worker, args=(6031,), nprocs=W, join=True)


@pytest.mark.skipif(not _gin_available(),
                    reason="needs RUN_GIN_TESTS=1, >=2 GPUs with GIN-capable NCCL, and nccl_gin built")
def test_gin_weights_signal_fence_compute_invariant():
    """Same bar with ``EPLB_GIN_FENCE=signal``, which is the configuration real runs need.

    The default ``dist.barrier`` fence is host-launched and not stream-ordered, so the backward
    re-pull cannot actually overlap Wgrad under it. The signal fence is what makes the overlap real
    (and the path capture-safe), and it carries its own per-index epoch counters, so it needs
    validating separately: a miscounted epoch shows up as a stale read, i.e. wrong numbers here."""
    mp.spawn(_worker, args=(6032, "alltoall", torch.float32, "signal"), nprocs=W, join=True)


@pytest.mark.skipif(not _gin_available(),
                    reason="needs RUN_GIN_TESTS=1, >=2 GPUs with GIN-capable NCCL, and nccl_gin built")
def test_gin_weights_network_only_matches_lsa():
    """``EPLB_GIN_LSA=0``: force every peer through GIN even when its window is mapped here.

    The batched kernels pick their transport per descriptor -- load/store for peers in the LSA team,
    GIN for the rest -- so on a single node the default path exercises almost none of the GIN code
    and this variant exercises almost none of the LSA code. Both must produce the same numbers as
    the single-device reference, which is what pins the two halves of the branch against each other:
    a wrong peer index in ``ncclGetPeerPointer`` (it takes a world rank and converts internally,
    unlike ``ncclGetLsaPointer``) reads another GPU's slot and shows up here as a mismatch, while
    passing under whichever half the box happens to take by default."""
    mp.spawn(_worker, args=(6036, "alltoall", torch.float32, "barrier", 1, True, False), nprocs=W,
             join=True)


@pytest.mark.skipif(not _gin_available(),
                    reason="needs RUN_GIN_TESTS=1, >=2 GPUs with GIN-capable NCCL, and nccl_gin built")
def test_gin_weights_two_chunk_signal_fence_compute_invariant():
    """Two-chunk hand-scheduled backward on the device fence, without needing DeepEP.

    The cluster configuration is this plus DeepEP tokens, and the DeepEP cases below are skipped
    wherever its NCCL build and the GIN-capable one disagree -- which is most places, since DeepEP
    pins the library it was built against. This keeps the weight channel's half of that
    configuration covered: one lease shared by both chunks, re-acquired once in backward, with the
    reduce ordered by ``world_fence`` on the weight stream rather than by a host barrier."""
    mp.spawn(_worker, args=(6037, "alltoall", torch.float32, "signal", 2, True), nprocs=W, join=True)


def _sync_worker(rank, port):
    """Drive the weight channel alone under CUDA's sync debug mode."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["EPLB_GIN_FENCE"] = "signal"
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=W)

    from eplb.integration.gin_weights import GinReplicaTransport, GinWeightReplicator

    dtype = torch.float32
    unit_expert, _, _, base_w1, base_w2 = _global_data(dtype)
    topo = Topology.from_nvlink_rdma(1, W, 1, 8)
    spec = ProblemSpec.uniform_main_placement(E, W, weight_bytes_each=1000, s_tok=1, n_slot=4)
    omega = torch.stack(
        [torch.bincount(unit_expert[r], minlength=E) for r in range(W)]
    ).to(torch.int64)
    plan_x = solve(Loads(omega), topo, spec, EPLBConfig()).x.to(dev)

    shapes = [torch.Size((H, F)), torch.Size((F, H))]
    replicator = GinWeightReplicator(
        group=None, num_experts=E, n_slot=spec.n_slot, main_rank=spec.main_rank.to(dev),
        weight_shapes=shapes, dtype=dtype, device=dev,
    )
    my_experts = [e for e in range(E) if e % W == rank]
    main_of = {e: (base_w1[e].to(dev), base_w2[e].to(dev)) for e in my_experts}
    meta = {"main_experts": my_experts, "transpose_w": False, "transport": None}
    w_eff = [torch.zeros((spec.n_slot, *s), dtype=dtype, device=dev) for s in shapes]
    g_slot = [torch.ones((spec.n_slot, *s), dtype=dtype, device=dev) for s in shapes]

    def one_step():
        transport = GinReplicaTransport(replicator, plan_x)
        transport.materialize_replicas(meta, main_of, w_eff[0], w_eff[1], dtype, dev, None)
        transport.reduce_grads(meta, g_slot[0], g_slot[1], dtype, dev)

    one_step()                      # warm the allocator, the comm and the barrier slots
    torch.cuda.synchronize()
    dist.barrier()

    torch.cuda.set_sync_debug_mode("error")
    try:
        one_step()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    dist.destroy_process_group()


@pytest.mark.skipif(not _gin_available(),
                    reason="needs RUN_GIN_TESTS=1, >=2 GPUs with GIN-capable NCCL, and nccl_gin built")
def test_gin_weight_channel_issues_no_host_sync():
    """Schedule derivation, both transfers and all four fences, with any host sync raising.

    Sync-freedom is the property the whole weight channel is built for and the one thing that never
    fails loudly: a stray ``.item()`` or a host barrier costs a pipeline bubble per layer and still
    produces correct numbers, so only this catches it. Covers the parts a regression would land in --
    ``slot_tables`` / ``_replica_schedule`` (device tables, no ``nonzero``), the batched get and put
    (device-resident descriptors, both the LSA and network branches), and ``world_fence`` (a kernel,
    where ``dist.barrier`` would be a host block). The token channel is excluded on purpose: it only
    reaches this bar under DeepEP, and ``all_to_all_single`` takes its splits on the host by design."""
    mp.spawn(_sync_worker, args=(6038,), nprocs=W, join=True)


def _deepep_available() -> bool:
    if not _gin_available():
        return False
    try:
        import deep_ep
    except Exception:
        return False
    return hasattr(deep_ep, "ElasticBuffer")


@pytest.mark.skipif(not _deepep_available(),
                    reason="needs the GIN prerequisites plus the 'deep_ep' package")
def test_deepep_tokens_with_gin_weights_compute_invariant():
    """ElasticBuffer tokens plus GIN weights/grads match the reference.

    This is the configuration the cluster runs use, and it is the only test that covers it. The
    backends never talk to each other -- DeepEP moves tokens, nccl_gin moves expert weights and
    reduces their grads to main(e) -- but they agree on slot ordering: tokens arrive grouped by
    physical slot and the weight stack is indexed by that same slot. A disagreement there is
    silently wrong numbers, not a crash. bf16 because that is what DeepEP's kernels move."""
    mp.spawn(_worker, args=(6033, "deepep", torch.bfloat16), nprocs=W, join=True)


@pytest.mark.skipif(not _deepep_available(),
                    reason="needs the GIN prerequisites plus the 'deep_ep' package")
def test_deepep_gin_two_chunk_compute_invariant():
    """ElasticBuffer + GIN with two manual chunks matches the reference.

    Chunking multiplies the token-side work but must not multiply the weight-side: both chunks read
    one acquired stack, and one shared lease re-acquires once in backward. If that sharing broke, GIN
    would issue a second round of gets whose fences no longer line up across ranks -- so this is also
    the test that pins the chunk count out of the weight channel's collective schedule."""
    mp.spawn(_worker, args=(6034, "deepep", torch.bfloat16, "signal", 2, True), nprocs=W, join=True)


@pytest.mark.skipif(not _deepep_available(),
                    reason="needs the GIN prerequisites plus the 'deep_ep' package")
def test_deepep_gin_two_chunk_autograd_matches_reference():
    """ElasticBuffer + GIN with two autograd chunks matches the reference.

    The hand-scheduled backward drives DeepEP's transpose legs itself (``buffer.combine`` for
    ``dispatch^-1``, ``buffer.dispatch`` for ``combine^-1``) instead of going through the autograd
    Functions, reusing the dispatch handle it captured in forward. Nothing off-cluster exercises that,
    and a handle mixed up between chunks would land grads on the wrong ranks -- which is only visible
    as a numeric mismatch against the same run with autograd doing the ordering."""
    mp.spawn(_worker, args=(6035, "deepep", torch.bfloat16, "signal", 2, False), nprocs=W, join=True)


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
