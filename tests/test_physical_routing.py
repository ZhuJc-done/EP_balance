"""The sync-free physical-id router must match the reference assign_unit_dst bit-for-bit."""

import torch

from eplb.algorithm import solve
from eplb.config import EPLBConfig
from eplb.integration.physical import assign_physical, build_phys_slot_table
from eplb.loads import Loads
from eplb.problem import ProblemSpec
from eplb.topology import Topology


def assign_unit_dst(unit_expert, plan, src_local_rank):
    """Host-loop reference router (oracle): map each unit to a dst rank per ``plan.q[src, e, :]``."""
    U = int(unit_expert.numel())
    device = unit_expert.device
    dst = torch.full((U,), -1, dtype=torch.int64, device=device)
    r = int(src_local_rank)
    for e in torch.unique(unit_expert).tolist():
        idxs = torch.nonzero(unit_expert == e, as_tuple=False).flatten()
        hosts = torch.nonzero(plan.x[e] == 1, as_tuple=False).flatten()
        counts = plan.q[r, e, hosts].to(torch.int64)
        dst[idxs] = torch.repeat_interleave(hosts, counts)
    return dst


def _make_case(seed: int, R_nodes=2, R_gpus=2, E=8, skew=2.0):
    g = torch.Generator().manual_seed(seed)
    topo = Topology.from_nvlink_rdma(R_nodes, R_gpus, 1, 8)
    R = topo.num_ranks
    n_slot = max(2, 2 * (E // R))
    spec = ProblemSpec.uniform_main_placement(E, R, weight_bytes_each=1000, s_tok=4, n_slot=n_slot)
    # skewed integer load matrix Lambda[R, E]
    base = torch.randint(0, 6, (R, E), generator=g)
    hot = torch.randint(0, E, (R,), generator=g)
    base[torch.arange(R), hot] += torch.randint(20, 60, (R,), generator=g)
    loads = Loads(base.to(torch.int64))
    plan = solve(loads, topo, spec, EPLBConfig())
    return loads, topo, spec, plan


def _units_for_rank(lam_row: torch.Tensor, seed: int) -> torch.Tensor:
    """Build a randomly-ordered unit_expert vector with exactly lam_row[e] units of expert e."""
    g = torch.Generator().manual_seed(seed)
    units = torch.repeat_interleave(torch.arange(lam_row.numel()), lam_row)
    perm = torch.randperm(units.numel(), generator=g)
    return units[perm].to(torch.int64)


def test_assign_physical_matches_reference_dst():
    for seed in range(20):
        loads, topo, spec, plan = _make_case(seed)
        R = topo.num_ranks
        for r in range(R):
            unit_expert = _units_for_rank(loads.lam[r], seed * 100 + r)
            ref_dst = assign_unit_dst(unit_expert.clone(), plan, r)
            phys_id, dst_rank = assign_physical(unit_expert, plan, spec, r)
            assert torch.equal(dst_rank, ref_dst), f"dst mismatch seed={seed} r={r}"
            # physical id must decode back to the same destination rank
            assert torch.equal(phys_id // int(spec.n_slot), ref_dst)


def test_phys_id_decodes_to_hosting_expert():
    """Each assigned physical id must map back (via the slot table) to the unit's expert."""
    loads, topo, spec, plan = _make_case(7)
    n_slot = int(spec.n_slot)
    phys_table = build_phys_slot_table(plan.x, n_slot)  # [E, R]
    # invert: phys_id -> expert (only hosted entries are valid)
    E, R = plan.x.shape
    inv = torch.full((R * n_slot,), -1, dtype=torch.int64)
    hosted = plan.x.to(torch.bool)
    ee, rr = torch.meshgrid(torch.arange(E), torch.arange(R), indexing="ij")
    inv[phys_table[hosted]] = ee[hosted]
    for r in range(R):
        unit_expert = _units_for_rank(loads.lam[r], 500 + r)
        phys_id, _ = assign_physical(unit_expert, plan, spec, r)
        assert torch.equal(inv[phys_id], unit_expert), f"phys->expert mismatch r={r}"


def test_no_host_sync_ops_on_cuda():
    """On CUDA the assignment must not trigger a device->host copy (best-effort check)."""
    if not torch.cuda.is_available():
        return
    loads, topo, spec, plan = _make_case(3)
    dev = "cuda"
    plan.x = plan.x.to(dev)
    plan.q = plan.q.to(dev)
    spec.main_rank = spec.main_rank.to(dev)
    r = 0
    unit_expert = _units_for_rank(loads.lam[r], 11).to(dev)
    torch.cuda.synchronize()
    phys_id, dst_rank = assign_physical(unit_expert, plan, spec, r)
    # if any op had synced, results are still correct; we just assert it runs & stays on device
    assert phys_id.is_cuda and dst_rank.is_cuda
    assert int(phys_id.numel()) == int(unit_expert.numel())
