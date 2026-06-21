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

The reference solver, simulator and tests are **pure CPU** and need only
`torch>=2.1` and `numpy>=1.24` (Python `>=3.9`) — no GPU required.

### Environment used / validated

The CPU solver and tests run anywhere. The Megatron-LM integration (Phase B/C
below) was developed and validated on:

| Component | Version |
|---|---|
| Hardware | 4× NVIDIA GB200 (single node, `aarch64` Grace-Blackwell) |
| OS / Python | Linux `aarch64` / Python 3.12.4 |
| CUDA | 13.1 |
| PyTorch | 2.9.1 (cu13 build) |
| TransformerEngine | 2.16.0 (built from source for `sm_100`) |
| Megatron-Core / Megatron-LM | 0.19.0 (`main`) |
| NCCL / cuDNN | `nvidia-nccl-cu13` 2.30.7 / `nvidia-cudnn-cu13` 9.22 |

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

Example `run_sim` output (imbalance 8.9× → 2.2×):

```
Baseline (no replication):  tau=  218957  imbalance= 8.909
Scale-EPLB plan          :  tau=   53988  imbalance= 2.197
Makespan reduction       :   4.056x
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
  # on the 4x GB200 box: --nproc_per_node=4 --master_addr=localhost --master_port=6010 \
  #                      -m sim.run_dist --backend nccl --world-size 4 --nodes 1 --gpus 4 --experts 64
  ```
  Expected tail: every rank prints the same `plan_hash`/`tau` and
  `E3 determinism check: PASS`.
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

## Tests

```bash
pip install -e ".[dev]"
pytest -q          # constraints (C1-C7), determinism, metrics, Phase C dispatcher
```

The suite is CPU-only; the determinism and Phase C tests spawn one `gloo` process
per rank to exercise the real all-gather / all-to-all path.

Notes for running the full suite:

- **Runtime.** The reference solver re-runs the full `O(R·E)` quota assignment on
  every replica-insertion attempt (clarity over speed; the fast path is a future
  CUDA kernel). A full `pytest` therefore takes a few minutes — the
  `test_constraints` / `test_determinism` parametrisations dominate, not a hang.
- **Loopback rendezvous.** The gloo-spawn tests rendezvous on `localhost` and a
  low port. On sandboxed pods that block wildcard (`0.0.0.0` / `[::]`) binds and
  the ephemeral port range — bind only loopback + low ports — keep `MASTER_ADDR`
  at `localhost` (not `127.0.0.1`, which forces a wildcard bind) and pick a low
  `--port` (e.g. `6000`). This is why `sim/run_dist.py`'s default `--port` may
  need overriding there.

## Validated on 4× GB200 (this environment)

End-to-end runs that passed on the box in [the env table](#environment-used--validated).
All three Megatron phases use `--transformer-impl local` (no hard TE dependency),
`--swiglu` experts and `--mock-data`, on `GPUS_PER_NODE=4` with a low `MASTER_PORT`:

```bash
# one-time: make `eplb` importable + point at Megatron
pip install -e .
export MEGATRON_DIR=~/Megatron-LM EPLB_DIR=~/EP_balance GPUS_PER_NODE=4

# Phase A — NCCL determinism (E3): all 4 ranks emit one bit-identical plan_hash
torchrun --nproc_per_node=4 --master_addr=localhost --master_port=6010 \
  -m sim.run_dist --backend nccl --world-size 4 --nodes 1 --gpus 4 --experts 64

# Phase B — observe real Megatron routing (dispatch unchanged)
MASTER_PORT=6008 bash scripts/run_phaseB.sh
#   -> [EPLB] layer=0..3 mb=0 tau=~2100 imbalance≈1.0 replicas=~14   (uniform mock data)

# Phase C — apply the replication dispatcher (loss decreases; EPLB active each mb)
MASTER_PORT=6009 bash scripts/run_phaseC.sh
#   -> iteration 1..20 | lm loss: 10.4 -> 9.45
```

If Megatron's MoE build pulls in a source-built TransformerEngine, also export the
NCCL/cuDNN wheel libs before launching, e.g.:

```bash
NV=$(python -c "import nvidia,os;print(os.path.dirname(nvidia.__file__))")
export LD_LIBRARY_PATH=$NV/nccl/lib:$NV/cudnn/lib:${LD_LIBRARY_PATH:-}
```
