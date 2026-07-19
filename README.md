# Scale-EPLB

A deterministic, CPU-sync-free **Expert-Parallelism Load Balancer** for MoE
training on heterogeneous clusters (NVLink domains + RDMA). It rebalances expert
compute by **replicating** hot experts (never rearranging them), planning replica
placement and token routing so per-rank load is even and cross-domain traffic is
minimized.

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

### Environment used / validated

The CPU solver and tests run anywhere. The Megatron-LM integration was developed
and validated on:

| Component | Version |
|---|---|
| Hardware | NVIDIA GB200 (`aarch64` Grace-Blackwell), driver 580.126.20 |
| OS / Python | Linux `aarch64` (kernel 6.14) / Python 3.12.4 |
| CUDA | 13.2 (PyTorch cu13 build) |
| PyTorch | 2.11.0 (cu13 build) |
| Megatron-Core / Megatron-LM | 0.19.0 (`main`, commit `0ff7226f6`) |
| NCCL / cuDNN | 2.28.9 / 9.22 |

Megatron-LM, DeepEP and Transformer Engine are **external dependencies** (not vendored):
install them with the helper scripts below, which pin versions and self-check the import.

### Cluster install (Megatron integration)

```bash
# 1) clone this repo
git clone https://github.com/ZhuJc-done/EP_balance.git
cd /home/tiger/EP_balance

# 2) external deps (each pins a commit / self-checks the import)
bash scripts/install_megatron.sh     # required: community Megatron-LM -> $MEGATRON_DIR
bash scripts/install_deepep.sh       # optional: DeepEP transport (NCCL Gin backend)
bash scripts/install_te.sh           # optional: Transformer Engine (TE + grouped-GEMM fast path)

# 3) make `eplb` importable
pip install -e /home/tiger/EP_balance
```

`install_deepep.sh` / `install_te.sh` are optional: the launchers run without them
(`AllToAllAdapter` for dispatch, `--transformer-impl local` for experts). Install
**DeepEP** for high-performance all-to-all transport, and **Transformer Engine** for
fused kernels + grouped-GEMM.
Run recipes (single-node, multi-node 2×4 / 4×4, observe/apply, baselines) are in
[`scripts/README.md`](scripts/README.md).

## Quick start (CPU solver / simulation)

```bash
pip install -e ".[dev]"

# single-process simulation: build a 4x8 topology, skewed load, solve, verify
python -m sim.run_sim --nodes 4 --gpus 8 --experts 64 --skew 1.5

# multi-process determinism check (gloo): every rank computes a bit-identical plan
python -m sim.run_dist --world-size 8 --experts 64 --skew 1.5

# tests
pytest -q
```

## GPU solver backends

```bash
# Select the CUDA path in an application whose load/topology/spec tensors are on CUDA.
export EPLB_SOLVER_BACKEND=fast

# CUDA fast path (default benchmark mode)
python tests/test_gpu_solver.py --nodes 4 --gpus-per-node 8 --experts 640

# Exact Triton comparison
python tests/test_gpu_solver.py --solver triton --nodes 4 --gpus-per-node 8 --experts 640
```

The CUDA extension is compiled and cached on first use; JIT build time is
reported separately and is not part of steady-state solver latency.

Example `run_sim` output (imbalance 8.9× → 2.2×):

```
Baseline (no replication):  tau=  218957  imbalance= 8.909
Scale-EPLB plan          :  tau=   53988  imbalance= 2.197
Makespan reduction       :   4.056x
Constraints C1-C7: OK
```
