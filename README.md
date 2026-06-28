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
| Megatron-Core / Megatron-LM | 0.19.0 (`main`) |
| NCCL / cuDNN | 2.28.9 / 9.22 |


## Quick start

```bash
pip install -e ".[dev]"

# single-process simulation: build a 4x8 topology, skewed load, solve, verify
python -m sim.run_sim --nodes 4 --gpus 8 --experts 64 --skew 1.5

# multi-process determinism check (gloo): every rank computes a bit-identical plan
python -m sim.run_dist --world-size 8 --experts 64 --skew 1.5

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
