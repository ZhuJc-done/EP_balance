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
| `run_real_moe.sh` | Real-model launcher (Qwen3-30B-A3B / Mixtral); `REAL=1` from `run_gb200_4x4.sh` forwards here. `MOCK=1` = mock-data + random init, `DEEPEP=1` = native DeepEP dispatch. |
| `install_megatron.sh` | Clone+install pinned community Megatron-LM, self-check `import megatron`. |
| `install_deepep.sh` | Optional: clone+build DeepEP (NCCL Gin backend) for the sync-free transport. |
| `convert_hf_to_mcore.sh` | Optional: convert a HF MoE checkpoint to mcore for realistic skew. |

> **Install** (clone + `install_megatron.sh` / `install_deepep.sh` + `pip install -e`)
> lives in the [top-level README](../README.md#cluster-install-megatron-integration). This file is the
> **run book**: launchers, run recipes, toggles, and troubleshooting.

## Phase B — observe (recommended first)

```bash
MEGATRON_DIR=/path/to/Megatron-LM EPLB_DIR=/path/to/EP_balance \
  bash scripts/run_phaseB.sh
```

Expected per-forward log on rank 0:

```
[EPLB] layer=0 mb=0 tau=12458 imbalance=1.014 replicas=75 phi_token=86075
```

`imbalance` is `tau / mean_load`: the makespan the solved plan achieves, over the ideal
perfectly-even load. It is the **residual** imbalance *after* rebalancing, so `1.0` means the
plan is perfect and low values are the solver working, **not** evidence that the input was
uniform. It says nothing on its own about how skewed the routing was — for that, compare
against the input skew (see [`ROUTER_SKEW`](#dialing-the-skew-router_skew)) or replay the
captured trace through the baselines below.

### Capture a routing trace for the baseline comparison

Set `EPLB_TRACE_OUT` in observe mode to dump the real gathered `Lambda[R, E]`
per (layer, micro-batch); rank 0 writes a self-describing file (topology, `main(e)`
placement, weight bytes, `s_tok`, `n_slot`). Replay it through **every** load
balancer (Scale-EPLB, DeepSeek EPLB, FasterMoE, FlexMoE, LPLB) offline:

```bash
# capture during a real (or MOCK=1) observe run
EPLB_MODE=observe EPLB_TRACE_OUT=logs/trace.pt \
MEGATRON_DIR=/path/to/Megatron-LM EPLB_DIR=/path/to/EP_balance \
  bash scripts/run_real_moe.sh

# score all baselines on the captured routing (no need to re-pass topology args)
python -m baseline.benchmark --trace logs/trace.pt \
  --strategies scale,eplb,fastermoe,flexmoe,lplb
```

Optional: `EPLB_TRACE_MAX` caps the number of captured samples (0 = all),
`EPLB_TRACE_EVERY` sets the disk-flush cadence. See `baseline/README.md` for how
each strategy's reported quality is defined.

## Measuring: latency breakdown, straggler, and the N_slot sweep

`EPLB_PROFILE=1` emits a periodic per-region summary. CUDA events are **queued** and
resolved in one batch, so timing injects no per-region host sync — an instrumented run
and an undisturbed end-to-end step time can come from the same job.

| Region | Covers |
|---|---|
| `solve` | the placement + quota solver (CUDA kernel) |
| `all_gather_lambda` | the single `Lambda[R,E]` all-gather |
| `apply/route` | router, plus flattening top-k selections into routing units |
| `apply/dispatch` | Stage 2 token all-to-all |
| `apply/expert_compute` | Stage 4 replica materialisation + batched expert MLP |
| `apply/combine` | Stage 5 output all-to-all |
| `apply/weight_move` | replica weight broadcast / GIN pull (nested inside expert compute) |

A synchronous EP step is paced by its slowest rank, so straggler analysis needs
**every** rank to report, not just rank 0:

```bash
EPLB_MODE=apply EPLB_PROFILE=1 EPLB_PROFILE_ALL_RANKS=1 EPLB_PROFILE_RESET_AT=1 \
  bash scripts/run_real_moe.sh
# every line is tagged [EPLB-profile r<RANK>] -> compare max vs mean across ranks
grep 'apply/expert_compute' logs/real_*.log
```

`N_slot` budget sweep (balance gain vs replica memory and weight-move bandwidth):

```bash
for NS in 16 20 24 32 48; do
  EPLB_MODE=apply EPLB_N_SLOT=$NS EPLB_PROFILE=1 EPLB_PROFILE_RESET_AT=1 \
    LOG_FILE=logs/nslot_${NS}.log bash scripts/run_real_moe.sh
done
grep 'peak memory' logs/nslot_*.log
```

`EPLB_N_SLOT` must be at least the mains each rank hosts (`num_experts / EP`), otherwise
the launcher fails fast with an explicit error. Under Level A (`EPLB_REMATERIALIZE=1`)
the `apply/weight_move` count includes the backward recompute — that doubling is exactly
what re-materialisation trades for memory.

### Reading the trace in Perfetto

`PROFILE_TRACE=1` turns on Megatron's PyTorch profiler; the region labels above are emitted as
`record_function` ranges **unconditionally** (no `EPLB_PROFILE` needed), so they show up inline
against the real CPU and CUDA rows:

```bash
PROFILE_TRACE=1 PROFILE_STEP_START=8 PROFILE_STEP_END=10 \
EPLB_MODE=apply ... bash scripts/run_real_moe.sh
# -> <PROFILE_DIR>/../torch_profile/rank-<N>.json.gz   (default PROFILE_DIR=logs/tb_<mode>)
```

Open the `.json.gz` directly at [ui.perfetto.dev](https://ui.perfetto.dev) (gzipped Chrome
traces are handled natively) and search for `eplb/`. Knobs: `PROFILE_RANKS="0 8"` (space
separated, default rank 0), `PROFILE_STACK=1` for Python/CPU call stacks, `PROFILE_SHAPES=1`
for tensor shapes.

Keep the window short (2–3 steps) and start it past step ~5: the first iterations pay CUDA-solver
JIT and cuBLAS autotuning, and `PROFILE_STACK=1` inflates the trace enough to distort the step
time read off the same run — take step time from an unprofiled run.

`off` / `observe` / `apply` all run `--transformer-impl local` (SequentialMLP, fusions off),
so a step-time delta between them comes from the EP part:

```bash
EPLB_MODE=off   ... bash scripts/run_real_moe.sh   # baseline
EPLB_MODE=apply ... bash scripts/run_real_moe.sh   # Scale-EPLB
```

### Keeping the routing skew (turn the aux loss off)

Megatron's default `aux_loss` router regulariser actively flattens expert load — the exact
imbalance a load balancer exists to exploit. Measuring balancers with it on understates all of
them, so set `ROUTER_BALANCING=none` (also forces `--moe-aux-loss-coeff 0.0`):

```bash
ROUTER_BALANCING=none EPLB_MODE=observe ... bash scripts/run_real_moe.sh
```

Watch `load_balancing_loss` in the training log: under `aux_loss` it decays as the router is
pushed toward uniform. Note that turning it off does **not** create skew on its own — a
randomly initialised router over `--mock-data` still routes near-uniformly (only multinomial
noise, `max/mean` ≈ 1.0 at 128 experts), which is why a mock run reports `imbalance` ≈ 1.01
no matter which balancer is used.

### Dialing the skew: `ROUTER_SKEW`

`ROUTER_SKEW=<std>` sets `--moe-router-force-biased`, which adds a per-expert bias
`b_e ~ N(0, |std|)` to the router logits. The bias is drawn from the *data-parallel* RNG seed,
which Megatron leaves rank-independent (`data_parallel_seed = seed`), so **every rank gets the
same `b_e`** — expert `e` is hot everywhere, which is the globally-coherent hot-expert case a
balancer has to fix. A negative `std` draws the bias once per layer and reuses it, giving a
*stationary* skew (use this; positive redraws every step). Measured at `E=128, top-k=8, EP=32`:

| `ROUTER_SKEW` | max/mean per expert | max/mean per rank |
|---|---|---|
| unset / `0` | 1.03 | 1.02 |
| `-0.25` | 2.9 | 1.6 |
| `-0.5` | 5.0 | 2.5 |
| `-0.75` | 8.2 | 3.4 |
| `-1.5` | 13.6 | 4.6 |
| `-2.0` | 15.1 | 7.2 |

Per-expert skew saturates at `E/top-k = 16` (one expert can absorb at most every token). The
per-rank column is the load the solver actually sees, and it is noisy in `std`: it depends on
*which* hot experts happen to share a rank under the contiguous `e -> e//(E/EP)` split, so fix
the seed and average a few draws before quoting a number.

```bash
for S in 0 -0.25 -0.5 -0.75 -1.5; do
  for M in off apply; do
    ROUTER_BALANCING=none ROUTER_SKEW=$S EPLB_MODE=$M EPLB_PROFILE=1 \
      LOG_FILE=logs/skew${S}_${M}.log bash scripts/run_real_moe.sh
  done
done
```

Two caveats. Megatron randomises the logits themselves on this path, so the **loss is
meaningless** under `ROUTER_SKEW` — never use it for the loss-curve comparison, only for step
time and imbalance. And the skew is synthetic: pair the sweep with one real-checkpoint run
(`convert_hf_to_mcore.sh`) to show the operating point a trained router actually lands on.

Token dropping is off by design: the launcher never passes `--moe-expert-capacity-factor`, and
unset means every routed token reaches its expert (`transformer_config.py`: "None means no token
dropping"). Adding it would silently drop the tail of the hot experts — exactly the load the
balancer is supposed to move — so don't.

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
`NCCL_SOCKET_IFNAME` is auto-detected (default route); set it (and `NCCL_IB_HCA`) only to override.

## Multi-node real model, 2 nodes x 4 GPUs = 8 ranks

Run the **same command on both nodes**, changing only `NODE_RANK` (0 / 1); `MASTER_ADDR`
is node-0's IP on both, and `MASTER_PORT` must be free on node-0 (pick a fresh high port if
you hit `EADDRINUSE`). Below is a pure-Megatron baseline (real Qwen3-30B-A3B architecture,
mock data + random init — no checkpoint/tokenizer needed):

```bash
# node0 (NODE_RANK=0); node1 is identical with NODE_RANK=1
pkill -9 -f 'torchrun|torch\.distributed|pretrain_eplb_moe' 2>/dev/null || true
MEGATRON_DIR=/home/tiger/Megatron-LM \
REAL=1 MOCK=1 MODEL=qwen3_30b_a3b EPLB_MODE=off \
TP=1 PP=1 EP=8 GLOBAL_BATCH_SIZE=8 SEQ_LEN=1024 TRAIN_ITERS=5 \
NNODES=2 NODE_RANK=0 MASTER_ADDR=<NODE0_IP> MASTER_PORT=34567 GPUS_PER_NODE=4 \
  bash scripts/run_gb200_4x4.sh
```

Toggles: `DEEPEP=1` (native DeepEP dispatch, off/observe only), `EPLB_MODE=observe|apply`
(attach EPLB), drop `MOCK=1` + add `CHECKPOINT`/`DATA_PATH`/`TOKENIZER_MODEL` for real
training. Experts always run the unfused `local` path, so absolute step times are higher
than a production run — raise `GLOBAL_BATCH_SIZE`/`SEQ_LEN` with that in mind.

## Notes

- **Do not pass `--moe-grouped-gemm`**: the v1 binding supports `SequentialMLP` (clean
  per-expert weights); `GroupedMLP` fused weights are not yet sliced. Math is identical,
  just without the fused grouped GEMM kernel.
- Tune via env: `GPUS_PER_NODE`, `EP_SIZE`, `NUM_EXPERTS` (divisible by `EP_SIZE`),
  `TOPK`, `TRAIN_ITERS`. `EPLB_MODE=off` runs plain Megatron.
- Expert bias is ignored in the apply path (v1); peak-performance DeepEP fusion is
  a later step.
```
