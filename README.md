# Scale-EPLB

A deterministic, CPU-sync-free **Expert-Parallelism Load Balancer** for MoE
training on heterogeneous clusters (NVLink domains + RDMA). It rebalances expert
compute by **replicating** hot experts (never rearranging them), planning replica
placement and token routing so per-rank load is even and cross-domain traffic is
minimized.

## What this repo is

1. A **reference solver** — a correct, bit-identical PyTorch implementation of the
   Stage 0/1/2 algorithm you can run and validate entirely on CPU (no GPU needed).
2. An **integration harness for Megatron-LM** — glue (`eplb/integration/megatron.py`)
   and ready-to-run cluster scripts (`scripts/`) to capture real MoE routing,
   solve, and verify determinism on real GPUs, staged so the risky parts come last.

The hot path is structured so it can later be swapped for a single-SM CUDA kernel
and wired into Megatron-LM + DeepEP.

> Design follows the *Scale-EPLB 问题定义* doc and targets the *scale-EPLB 实验计划*
> (4 nodes × 8 GB200, K=1 per-mb × per-layer replanning, stateless replicas).

## Why

EP load imbalance is severe: a few hot experts overload some GPUs while others
idle, stretching the EP step. Scale-EPLB fixes this **at deployment time without
touching model quality** (no balance-loss / capacity caps), via three ideas:

1. **Replication, not rearrangement.** `main(e)` is immutable; we only add
   replicas. Logical→physical mapping stays fixed, so gradients aggregate cleanly
   back to one optimizer owner.
2. **Topology-aware.** Cheap intra-domain NVLink vs expensive inter-domain RDMA.
   A cross-domain replica is created only when its one-time weight move (counted
   ×2 for the gradient return in training) beats repeatedly shipping that
   domain's tokens.
3. **No CPU sync.** One all-gather of the integer load matrix `Λ`, then every
   rank solves locally and **bit-identically** on-device. No broadcast, no CPU
   decision.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # runtime: torch, numpy
pip install -e ".[dev]"     # + pytest
# pip install -e ".[oracle]"  # optional OR-Tools MILP oracle (future; gap measurement)
```

## Quick start (CPU, no GPU)

```bash
# single-process simulation: build a 4x8 topology, skewed load, solve, verify
python -m sim.run_sim --nodes 4 --gpus 8 --experts 64 --skew 1.5

# multi-process determinism check (gloo): every rank computes a bit-identical plan
python -m sim.run_dist --world-size 8 --experts 64 --skew 1.5

# Megatron-shaped API demo (simulated routing, no Megatron needed)
python -m examples.megatron_eplb_hook

# tests
pytest -q
```

Example `run_sim` output (imbalance 8.9× → 1.08×):

```
Baseline (no replication):  tau=  218957  imbalance= 8.909
Scale-EPLB plan          :  tau=   26441  imbalance= 1.076
Makespan reduction       :   8.281x
Constraints C1-C7: OK
```

## Library usage

```python
import torch
from eplb import EPLBConfig, Topology, ProblemSpec, Loads, solve, compute_metrics

topo = Topology.from_nvlink_rdma(num_nodes=4, gpus_per_node=8,
                                 intra_cost=1, inter_cost=8)
spec = ProblemSpec.uniform_main_placement(
    num_experts=64, num_ranks=32,
    weight_bytes_each=44_000_000, s_tok=7168 * 2, n_slot=4,
)
loads = Loads(lam=torch.randint(0, 500, (32, 64)))   # [R, E] token counts

plan = solve(loads, topo, spec, EPLBConfig())
print(plan.x)          # [E, R] placement (1 = instance present)
print(plan.q)          # [R, E, R] routing quota q[src, e, dst]
print(compute_metrics(plan, loads, topo, spec))
```

### In a training loop (per micro-batch, K=1)

```python
from eplb import EPLBRebalancer
from eplb.distributed import local_counts_from_routing

reb = EPLBRebalancer(topo, spec, EPLBConfig())   # once per EP layer

# forward:
local_row = local_counts_from_routing(routed_expert_ids, num_experts=64)
result = reb.rebalance(local_row, layer_id=L, micro_batch_id=mb)  # all-gather + solve
#   -> use result.plan.q to drive your DeepEP dispatch

# backward:
reb.backward(layer_id=L, micro_batch_id=mb)      # re-derives plan, aggregates grads
```

## Run on a real cluster (Megatron-LM)

Integration is **staged** so the high-risk parts (changing token dispatch) come
last. See `scripts/README.md` for the push-and-run guide.

- **Phase A — determinism / latency, no Megatron.** Verify bit-identical plans
  (E3) and solver latency (E2) over NCCL at scale:
  ```bash
  torchrun --nproc_per_node=8 -m sim.run_dist --backend nccl --experts 64
  ```
- **Phase B — observe real routing, zero Megatron source edits.** Attach PyTorch
  forward hooks to every MoE router to capture `Λ`, solve, and log/verify each
  forward — dispatch unchanged. One call: `eplb.integration.megatron.setup_eplb_observer`.
  Launch a tiny MoE with mock data via `scripts/run_phaseB.sh` (no data prep).
- **Phase C — apply.** A self-contained, gloo-verified reference dispatcher
  (`eplb/integration/dispatcher.py:replicated_moe_forward`) makes the plan take
  effect: it splits each expert's tokens across replicas per `plan.q`, materialises
  replica weights from `main(e)` (differentiable, so backward aggregates every
  replica's gradient to the one optimizer owner), and is **compute-invariant** vs a
  single instance (`tests/test_phase_c.py`). Remaining: bind it into Megatron's
  `MoELayer.forward`, and (later) a DeepEP-fused fast path for peak performance.

```python
# Phase B, inside Megatron's model_provider (see scripts/pretrain_eplb_moe.py):
from eplb.integration.megatron import setup_eplb_observer, assert_plan_replicated

hook, handles = setup_eplb_observer(
    model, num_experts=args.num_experts, weight_bytes_each=expert_param_bytes,
    s_tok=args.hidden_size * 2, n_slot=2 * (args.num_experts // ep_size),
    gpus_per_node=8, logger=print,
)
# ... run a few iters; rank 0 prints "[EPLB] layer=.. tau=.. imbalance=.." ...
assert assert_plan_replicated(hook.last_plan, hook.ep_group)   # E3 on real routing
```

## How the solver maps to the formulation

| Doc symbol | Code |
|---|---|
| `Λ = [λ_{r,e}]` | `Loads.lam` `[R, E]` |
| `λ_e`, `T_{d→e}` | `Loads.expert_load()`, `Loads.domain_demand()` |
| `dom(r)`, `c_{r,r'}` | `Topology.domain_of_rank`, `Topology.cost` |
| `main(e)`, `‖W_e‖`, `s_tok`, `N_slot` | `ProblemSpec.main_rank / weight_bytes / s_tok / n_slot` |
| `x_{e,r}`, `q_{r,e,r'}`, `τ` | `Plan.x`, `Plan.q`, `Plan.tau` |
| objective `α τ + β Φ_token + γ Φ_weight` | `Metrics.objective` |

Algorithm (`eplb/algorithm.py`), mirroring the doc's "算法设计":

- **Stage 0** precompute `λ_e`, `T[d,e]`, break-even threshold
  `T*[e] = ceil(2‖W_e‖ / (η·s_tok))`.
- **Stage 1** cross-domain replication gate (**C6**): admit a replica of `e` in
  domain `d` iff `‖W_e‖ < 2·T[d,e]·s_tok`, greedily by marginal benefit, under
  the slot budget.
- **Stage 2** intra-domain balancing: relieve the busiest rank by replicating its
  top expert **inside its own domain**, then water-filling quota assignment with
  strict domain-local serving (cross-domain only when no in-domain instance).

Constraints **C1–C7** are checked by `eplb.check_constraints`:
C1 conservation, C2 reachability, C3 makespan, C4 slot budget, C5 quota
granularity, C6 cross-domain gate, C7 main fixed.

### Determinism contract

Every decision uses **integer** arithmetic with a fully specified,
rank-independent tie-break order (value → expert id → rank id). Inputs are
integer token counts. Therefore all ranks that all-gather the same `Λ` produce a
**bit-identical** `Plan` (verified by `sim/run_dist.py` and
`tests/test_determinism.py`). The backward pass can recompute the plan from the
cached `Λ` instead of storing it (`EPLBRebalancer(cache_plans=False)`).

## Project layout

```
eplb/
  config.py        EPLBConfig (α/β/γ, η, u_min, slot/gate knobs)
  topology.py      Topology: NVLink domains + RDMA cost matrix
  problem.py       ProblemSpec: main placement, weights, slot budget
  loads.py         Loads: Λ matrix + aggregates
  plan.py          Plan: x placement, q quota, tau
  algorithm.py     solve(): Stage 0/1/2 deterministic solver
  metrics.py       compute_metrics(), check_constraints() (C1-C7)
  distributed.py   all_gather_lambda() (gloo/nccl), routing->counts helper
  integration/
    rebalancer.py  EPLBRebalancer: collect -> solve -> apply, ring buffer
    hooks.py       WeightMaterializer / Dispatcher interfaces (placeholders)
    megatron.py    Megatron-Core glue: capture Λ, build spec, observe hooks (Phase B)
    comm.py        autograd all-to-all + broadcast-from-main (grad reduces to main)
    dispatcher.py  replicated_moe_forward: replication-aware dispatch/combine (Phase C)
    megatron_moe.py  bind MoELayer.forward to the replication dispatcher (Phase C apply)
examples/
  megatron_eplb_hook.py   Megatron-shaped API demo + insertion-point docs
scripts/
  pretrain_eplb_moe.py    zero-fork Megatron entrypoint (attaches the observer)
  run_phaseB.sh           torchrun launcher: tiny MoE + mock data
  convert_hf_to_mcore.sh  optional HF -> mcore checkpoint conversion
sim/
  workload.py      synthetic skewed Λ generators
  run_sim.py       single-process end-to-end demo
  run_dist.py      multi-process bit-identical verification (E3)
  walkthrough.py   tiny fully-traced example of one solve
tests/             pytest: constraints, determinism, metrics
```

## Status & roadmap

**Implemented:** deterministic reference solver, C1–C7 checks, metrics, real
`torch.distributed` all-gather (gloo/NCCL), Megatron rebalancer with ring buffer +
deterministic backward, **Phase B** Megatron observer (forward-hook Λ capture, no
source edits) + cluster scripts, **Phase C** reference replication dispatcher
(gloo-verified compute-invariant dispatch/combine + per-mb weight materialization +
gradient aggregation to `main(e)`), CPU simulator, tests.

**Placeholders (interfaces defined, backend not wired):**

- **Weight (re)materialization** — `eplb/integration/hooks.py:WeightMaterializer`.
  Production: Triton/NVSHMEM prefetch of stateless replica weights, backward
  re-materialize + gradient aggregation to `main(e)`. Currently `NullWeightMaterializer`.
- **DeepEP dispatch/combine** — `hooks.py:Dispatcher`. Production: translate
  `plan.q` into DeepEP's dynamic per-call layout.

**Next steps (see the experiment plan):**

1. Bind `replicated_moe_forward` into Megatron's `MoELayer.forward` (flatten top-k
   routing into units, pass the expert weights), then a DeepEP-fused fast path.
2. Calibrate the cost model `c_{r,r'}`, `s_tok`, `η` from measured NVLink/RDMA
   bandwidth.
3. Single-SM CUDA solver kernel to hit the ~100µs solving-overhead target (E2),
   validated bit-for-bit against this Python oracle.
4. Optional OR-Tools MILP oracle to measure the greedy's optimality gap.
