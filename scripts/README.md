# Scale-EPLB cluster scripts

Run MoE on real Megatron-LM with Scale-EPLB, in two stages selected by `EPLB_MODE`:

- **Phase B (`observe`)** — attach forward-hook observers to capture real routing,
  solve, and log/verify each forward; **dispatch unchanged**, no Megatron source
  edits. Validates solver latency (E2) and bit-identical plans (E3) on real routing.
- **Phase C (`apply`)** — bind every MoELayer to the replication dispatcher so the
  plan **takes effect**: tokens split across replicas per `plan.q`, replica weights
  materialised from `main(e)`, gradients aggregated back. Training stays plain Megatron.

## What's here

| File | Purpose |
|---|---|
| `pretrain_eplb_moe.py` | Zero-fork entrypoint; `EPLB_MODE=observe\|apply\|off` in `model_provider`. |
| `run_phaseB.sh` | `torchrun` launcher, observe mode, tiny MoE + `--mock-data`. |
| `run_phaseC.sh` | `torchrun` launcher, apply mode, end-to-end training. |
| `convert_hf_to_mcore.sh` | Optional: convert a HF MoE checkpoint to mcore for realistic skew. |

## One-time setup (on the cluster)

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
pip install -e Megatron-LM            # + transformer-engine/apex per its install guide
pip install -e /path/to/EP_balance    # makes `eplb` importable
```

## Phase B — observe (recommended first)

```bash
MEGATRON_DIR=/path/to/Megatron-LM EPLB_DIR=/path/to/EP_balance \
  bash scripts/run_phaseB.sh
```

Expected per-forward log on rank 0:

```
[EPLB] layer=0 mb=0 tau=12458 imbalance=1.014 replicas=75 phi_token=86075
```

`imbalance` is the would-be makespan ratio under the EPLB plan vs. the current
placement; with `--mock-data` (near-uniform routing) it stays close to 1.0 — use a
real checkpoint (`convert_hf_to_mcore.sh`) to see meaningful skew.

## Phase C — apply (end-to-end training)

```bash
MEGATRON_DIR=/path/to/Megatron-LM EPLB_DIR=/path/to/EP_balance \
  bash scripts/run_phaseC.sh
```

Trains normally (loss should decrease) with EPLB rebalancing each micro-batch.
Full cluster (4 nodes x 4 GPUs): set `NNODES=4` and per-node `NODE_RANK`,
`MASTER_ADDR`, `MASTER_PORT`; `EP_SIZE` defaults to the world size.

## Notes

- **Do not pass `--moe-grouped-gemm`** in apply mode: the v1 binding supports
  `SequentialMLP` (clean per-expert weights); `GroupedMLP` fused weights are not yet
  sliced. Math is identical, just without the fused grouped GEMM kernel.
- Tune via env: `GPUS_PER_NODE`, `EP_SIZE`, `NUM_EXPERTS` (divisible by `EP_SIZE`),
  `TOPK`, `TRAIN_ITERS`. `EPLB_MODE=off` runs plain Megatron.
- `--transformer-impl local` avoids a hard Transformer-Engine dependency; switch to
  `transformer_engine` if your Megatron build requires it for MoE.
- Expert bias is ignored in the apply path (v1); peak-performance DeepEP fusion is
  a later step.
```
