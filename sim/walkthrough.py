"""A tiny, fully-traced example of how the Scale-EPLB solver runs (run: python -m sim.walkthrough)."""
import torch

from eplb import EPLBConfig, ProblemSpec, Topology, check_constraints, compute_metrics, solve
from eplb.loads import Loads
from eplb.algorithm import _assign_quota

torch.set_printoptions(linewidth=120)

# ---- tiny cluster: 2 nodes x 2 gpus = 4 ranks, 2 NVLink domains ----
topo = Topology.from_nvlink_rdma(num_nodes=2, gpus_per_node=2, intra_cost=1, inter_cost=8)
# main(e): e0->r0, e1->r1, e2->r2, e3->r3 ; weights tiny so C6 can trigger
spec = ProblemSpec(
    num_experts=4,
    main_rank=torch.tensor([0, 1, 2, 3]),
    weight_bytes=torch.tensor([10, 10, 10, 10]),
    s_tok=2,
    n_slot=2,
)
cfg = EPLBConfig()

# routing data Lambda[r, e]: expert 0 is extremely hot, from BOTH domains
lam = torch.tensor([
    [40, 2, 1, 1],   # rank 0 (domain 0)
    [38, 2, 1, 1],   # rank 1 (domain 0)
    [30, 1, 2, 1],   # rank 2 (domain 1)
    [28, 1, 1, 2],   # rank 3 (domain 1)
])
loads = Loads(lam)

print("dom(r)      =", topo.domain_of_rank.tolist())
print("cost matrix =\n", topo.cost)
print("main(e)     =", spec.main_rank.tolist())
print("Lambda[r,e] =\n", lam)
print("lambda_e    =", loads.expert_load().tolist())
print("T[d,e]      =\n", loads.domain_demand(topo.domain_of_rank, topo.num_domains))

# baseline: no replication, every token to main(e)
base = torch.zeros(4, dtype=torch.int64)
base.index_add_(0, spec.main_rank, loads.expert_load())
print("baseline rank load =", base.tolist(), " tau =", int(base.max()))

# solved plan and per-rank load
plan = solve(loads, topo, spec, cfg)
print("\n=== FINAL PLAN ===")
print("x[e,r] placement =\n", plan.x)
print("num_replicas/e   =", plan.num_replicas().tolist())
print("slots used/rank  =", plan.slots_used().tolist(), " (N_slot=2)")
print("rank load L[r']  =", plan.rank_load().tolist(), " tau =", plan.tau)

m = compute_metrics(plan, loads, topo, spec, cfg)
print("imbalance        =", round(m.imbalance, 3))
print("Phi_token        =", m.phi_token, " Phi_weight =", m.phi_weight)
print("constraints OK   =", check_constraints(plan, loads, topo, spec, cfg).ok)

# show the routing of rank 0's hot-expert tokens
print("\nq[r=0, e=0, :] (where rank0's 40 hot tokens go) =", plan.q[0, 0].tolist())
print("q[r=2, e=0, :] (where rank2's 30 hot tokens go) =", plan.q[2, 0].tolist())
