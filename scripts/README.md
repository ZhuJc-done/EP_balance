# Scale-EPLB cluster scripts

Run MoE on real Megatron-LM with Scale-EPLB, in two stages selected by `EPLB_MODE`:

- **Phase B (`observe`)** — attach forward-hook observers to capture real routing,
  solve, and log/verify each forward; **dispatch unchanged**, no Megatron source
  edits. Validates solver latency (E2) and bit-identical plans (E3) on real routing.
- **Phase C (`apply`)** — bind every MoELayer to the sync-free dispatcher so the
  plan **takes effect**: tokens routed to physical instances per `plan.q`, replica weights
  materialised from `main(e)`, gradients aggregated back. Training stays plain Megatron.

## What's here

| File | Purpose |
|---|---|
| `pretrain_eplb_moe.py` | Zero-fork entrypoint; `EPLB_MODE=observe\|apply\|off` in `model_provider`. |
| `run_phaseB.sh` | `torchrun` launcher, observe mode, tiny MoE + `--mock-data`. |
| `run_phaseC.sh` | `torchrun` launcher, apply mode, end-to-end training. |
| `run_gb200_4x4.sh` | Multi-node 4 nodes x 4 GB200 launcher: Slurm auto-discovery + GB200 NCCL/RDMA env; mock-data smoke test by default, `REAL=1` forwards to `run_real_moe.sh`. |
| `sbatch_gb200_4x4.sbatch` | Slurm wrapper (`sbatch`) that `srun`s `run_gb200_4x4.sh` (1 task/node). |
| `convert_hf_to_mcore.sh` | Optional: convert a HF MoE checkpoint to mcore for realistic skew. |
| `install_megatron.sh` | Clone+install a pinned community Megatron-LM and self-check `import megatron`. |
| `install_deepep.sh` | Optional: clone+build DeepEP (NCCL Gin backend) for the sync-free transport. |

## One-time setup (on the cluster)

Megatron-LM and DeepEP are **external dependencies** (not vendored / not submodules):
they must be compiled+installed into the environment and are often already provided at a
fixed cluster path. Install them with the helper scripts (each pins a commit and self-checks
the import), then install `eplb`:

```bash
# Megatron-LM (required) — pinned commit, editable install, import self-check
MEGATRON_DIR=/opt/tiger/Megatron-LM bash scripts/install_megatron.sh

# DeepEP (optional) — only for the fully sync-free DeepEPAdapter on a DeepEP-capable cluster
DEEPEP_DIR=/opt/tiger/DeepEP bash scripts/install_deepep.sh

pip install -e /path/to/EP_balance   # makes `eplb` importable
```

Or do it by hand (TE/Apex optional — the launchers use `--transformer-impl local`):

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git && pip install -e Megatron-LM
pip install transformers einops sentencepiece tiktoken regex
pip install -e /path/to/EP_balance
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

## Multi-node 4 x GB200 (4 nodes x 4 GPUs = 16 ranks)

`run_gb200_4x4.sh` adds Slurm rank/master auto-discovery and a GB200/Blackwell
NCCL+RDMA env block on top of the entrypoint. Start with the safe observe-mode
smoke test (no checkpoint/data), then move to apply / a real model.

Under Slurm (recommended online):

```bash
MEGATRON_DIR=/path/to/Megatron-LM EPLB_DIR=/path/to/EP_balance \
  sbatch scripts/sbatch_gb200_4x4.sbatch          # observe-mode smoke test on 16 GPUs
```

Manual (run on every node, set NODE_RANK=0..3 and MASTER_ADDR=node-0):

```bash
MEGATRON_DIR=/path/to/Megatron-LM EPLB_DIR=/path/to/EP_balance \
NNODES=4 NODE_RANK=$RANK MASTER_ADDR=$HEAD MASTER_PORT=29500 \
  bash scripts/run_gb200_4x4.sh
```

Then escalate: `EPLB_MODE=apply` (active dispatcher), or `REAL=1 MODEL=qwen3_30b_a3b`
plus `CHECKPOINT`/`DATA_PATH`/`TOKENIZER_MODEL` to forward to `run_real_moe.sh`.
Adjust `NCCL_IB_HCA` / `NCCL_SOCKET_IFNAME` / `NCCL_IB_GID_INDEX` to your fabric.

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
