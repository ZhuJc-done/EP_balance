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
3. **No CPU sync.** One all-gather of the integer load matrix `Ω`, then every
   rank solves locally and **bit-identically** on-device. No broadcast, no CPU
   decision.

### Environment used / validated

The CPU solver and tests run anywhere. The Megatron-LM integration was developed
and validated on:

| Component | Version |
|---|---|
| OS / Python | Linux `aarch64` (kernel 6.14) / Python 3.12.4 |
| CUDA | 13.2 (PyTorch cu13 build) |
| PyTorch | 2.11.0 (cu13 build) |
| Megatron-Core / Megatron-LM | 0.19.0 (commit `0ff7226f6d8eba14c385a5d2ea658f92e4dcf40f`) |
| DeepEP | 2.0.0 (commit `af9a0403188392824fc3057452822235873e0612`) |
| NCCL / cuDNN | 2.28.9 / 9.22 |

Megatron-LM and DeepEP are **external dependencies** (not vendored): install them with the
helper scripts below, which pin versions and self-check the import.

### Cluster install (Megatron integration)

```bash
# 1) clone this repo
git clone https://github.com/ZhuJc-done/EP_balance.git
cd /home/tiger/EP_balance

# 2) external deps (each pins a commit / self-checks the import)
bash scripts/install_megatron.sh     # required: community Megatron-LM -> $MEGATRON_DIR
bash scripts/install_deepep.sh       # optional: DeepEP transport (NCCL Gin backend)

# 3) make `eplb` importable
pip install -e /home/tiger/EP_balance
```

## GPU solver

The planner is organized as two algorithms:

1. **Algorithm 1 — Inter-node Placement**
   - **Inter-node Placement:** admit cross-domain replicas from per-domain demand.
   - **Update Routing:** construct `Q[src, expert, dst]` from the resulting placement,
     preferring same-domain instances and deterministically splitting each quota.
2. **Algorithm 2 — Intra-node Replication**
   - repair rank imbalance independently inside each topology domain by adding
     domain-local replicas and incrementally updating the affected entries of `Q`.