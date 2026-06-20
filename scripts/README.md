# Scale-EPLB cluster scripts (Phase B)

Run a tiny MoE on real Megatron-LM with the Scale-EPLB observer attached, to
validate the solver on **real routing** — solver latency (E2) and cross-rank
bit-identical plans (E3) — **without editing Megatron-Core source**.

## What's here

| File | Purpose |
|---|---|
| `pretrain_eplb_moe.py` | Zero-fork entrypoint: reuses Megatron's `pretrain_gpt.py`, wraps `model_provider` to attach forward-hook observers (`setup_eplb_observer`). |
| `run_phaseB.sh` | `torchrun` launcher: tiny MoE GPT + `--mock-data` (no data prep). |
| `convert_hf_to_mcore.sh` | Optional: convert a HF MoE checkpoint to mcore for realistic skew. |

## One-time setup (on the cluster)

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
pip install -e Megatron-LM            # + transformer-engine/apex per its install guide
pip install -e /path/to/EP_balance    # makes `eplb` importable
```

## Run Phase B (single node, 8 GPUs)

```bash
MEGATRON_DIR=/path/to/Megatron-LM EPLB_DIR=/path/to/EP_balance \
  bash scripts/run_phaseB.sh
```

Multi-node: set `NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` on each node.
Tune via env: `GPUS_PER_NODE`, `EP_SIZE`, `NUM_EXPERTS` (divisible by `EP_SIZE`),
`TOPK`, `TRAIN_ITERS`. Set `EPLB_OBSERVE=0` for a plain Megatron run.

Expected per-forward log on rank 0:

```
[EPLB] layer=0 mb=0 tau=12458 imbalance=1.014 replicas=75 phi_token=86075
```

`imbalance` is the would-be makespan ratio under the EPLB plan vs. the current
placement; with `--mock-data` (near-uniform routing) it stays close to 1.0 — use
a real checkpoint (`convert_hf_to_mcore.sh`) to see meaningful skew.

## Notes

- Phase B does **not** change token dispatch; it only observes. Making the plan
  take effect (changing dispatch + replica weights) is **Phase C** and requires
  editing `megatron/core/transformer/moe/token_dispatcher.py`.
- `--transformer-impl local` avoids a hard Transformer-Engine dependency; switch
  to `transformer_engine` if your Megatron build requires it for MoE.
