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
| `run_real_moe.sh` | Real-model launcher (Qwen3-30B-A3B / Mixtral); `REAL=1` from `run_gb200_4x4.sh` forwards here. `MOCK=1` = mock-data + random init; `MOCK=0 FROM_SCRATCH=1` = real data + random init; `DEEPEP=1` = native DeepEP dispatch. |
| `run_slot_sweep.sh` | Sweep `N_slot=1..4` with workload seed 0 and save one non-detail baseline JSON per configuration; uses a fixed LPLB Ring topology. |
| `run_solver_scaling.sh` | Sweep the Scale-EPLB CUDA solver over logical rank and expert counts and save hot-kernel JSON results. |
| `prepare_open_workload.py` | Download task/corpus workloads, extract model inputs, and optionally build Megatron `.bin/.idx`. |
| `eval/plot_solver_scaling.py` | Read an existing solver-scaling JSON directory and independently generate PNG/PDF plots. |
| `install_megatron.sh` | Clone+install pinned community Megatron-LM, self-check `import megatron`. |
| `install_deepep.sh` | Optional: clone+build DeepEP (NCCL Gin backend) for the sync-free transport. |
| `convert_hf_to_mcore.sh` | Optional: convert HF Mixtral to mcore. Qwen3 needs a preconverted Megatron Bridge checkpoint. |

> **Install** (clone + `install_megatron.sh` / `install_deepep.sh` + `pip install -e`)
> lives in the [top-level README](../README.md#cluster-install-megatron-integration). This file is the
> **run book**: launchers, run recipes, toggles, and troubleshooting.

## Fresh machine: install once, keep artifacts on HDFS

Use local disk for source trees and compiled extensions; use the shared HDFS
mount for datasets, Hugging Face caches, tokenizers, checkpoints, and logs.
Building DeepEP directly under the HDFS mount is slower and can break compiler
file-locking semantics.

Prerequisites in the image: CUDA-enabled PyTorch, `nvcc`, `git`, and access to
`/mnt/hdfs/__MERLIN_USER_DIR__`.

```bash
cd "${HOME}"
git clone https://github.com/ZhuJc-done/EP_balance.git
cd EP_balance

# Installs into the image's active Python environment.
bash scripts/bootstrap_new_machine.sh
```

The bootstrap installs Scale-EPLB plus pinned Megatron-LM and DeepEP, saves the
Qwen tokenizer, installs `nvidia-nccl-cu13>=2.30.4` with `--no-deps`, and
creates this persistent layout:

```text
/mnt/hdfs/__MERLIN_USER_DIR__/eplb_data/
├── cache/huggingface/   # HF model/dataset downloads
├── raw/                 # JSONL + manifests
├── indexed/             # Megatron .bin/.idx
├── tokenizers/
├── checkpoints/
└── logs/
```

On every new shell and every training node:

```bash
cd "${HOME}/EP_balance"
source scripts/env_hdfs.sh
export MEGATRON_DIR="${HOME}/Megatron-LM"
export DEEPEP_DIR="${HOME}/DeepEP"
```

The HDFS directories are shared, but source trees, Python packages, and compiled
DeepEP extensions are local to each machine. Run the bootstrap on every fresh
machine, or bake its result into a derived image. Prepare each dataset only once.

### Prepare real data

Generic Hugging Face datasets with a `text` column:

```bash
source scripts/env_hdfs.sh
MEGATRON_DIR="${HOME}/Megatron-LM" \
TOKENIZER_MODEL="${EPLB_TOKENIZER_DIR}/qwen3_30b_a3b" \
DATA_NAME=wikitext103 \
DATASET=Salesforce/wikitext DATASET_CONFIG=wikitext-103-raw-v1 \
MAX_DOCS=0 bash scripts/prepare_data.sh

# Training prefix:
# DATA_PATH=${EPLB_INDEXED_DATA_DIR}/wikitext103_text_document
```

For DAPO-Math and the other workload adapters:

```bash
source scripts/env_hdfs.sh
python scripts/prepare_open_workload.py \
  --workload dapo_math \
  --tokenizer-model "${EPLB_TOKENIZER_DIR}/qwen3_30b_a3b" \
  --preprocess \
  --megatron-dir "${HOME}/Megatron-LM"

# DATA_PATH=${EPLB_INDEXED_DATA_DIR}/dapo_math_text_document
```

Qwen3 training needs an already converted Megatron Bridge checkpoint. Store it
at `${EPLB_CHECKPOINT_DIR}/qwen3_30b_a3b_mcore`. The pinned community
Megatron converter does not contain a Qwen3 loader. For Mixtral, use:

```bash
source scripts/env_hdfs.sh
MEGATRON_DIR="${HOME}/Megatron-LM" \
HF_MODEL=mistralai/Mixtral-8x7B-v0.1 \
TOKENIZER_MODEL=/path/to/local/tokenizer.model \
SAVE_DIR="${EPLB_CHECKPOINT_DIR}/mixtral8x7b_mcore" \
TP=1 EP=8 bash scripts/convert_hf_to_mcore.sh
```

## Phase B — observe (recommended first)

```bash
MEGATRON_DIR=/path/to/Megatron-LM EPLB_DIR=/path/to/EP_balance \
  bash scripts/run_phaseB.sh
```

Expected per-forward log on rank 0:

```
[EPLB] layer=0 mb=0 theta=12458 imbalance=1.014 replicas=75 phi_token=86075
```

`imbalance` is `theta / mean_load`: the maximum rank load over the ideal
perfectly-even load. It is the **residual** imbalance *after* rebalancing, so `1.0` means the
plan is perfect and low values are the solver working, **not** evidence that the input was
uniform. It says nothing on its own about how skewed the routing was — for that, compare
against the input skew (see [`ROUTER_SKEW`](#dialing-the-skew-router_skew)) or replay the
captured trace through the baselines below.

### Capture a routing trace for the baseline comparison

Set `EPLB_TRACE_OUT` in observe mode to dump the real gathered `Ω[R, E]`
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

Solver-only scalability does not require a distributed launch. One physical GPU
simulates the full logical problem and times only the hot `fast_solver.cu` kernels;
JIT compilation is recorded separately and excluded from the plotted latency:

```bash
# Step 1: run the benchmark and write JSON only
bash scripts/run_solver_scaling.sh
# -> logs/solver_scaling/{rank_scale,expert_scale}_r*_e*.json

# Step 2: plot the existing JSON results
python eval/plot_solver_scaling.py --input-dir logs/solver_scaling
# -> logs/solver_scaling/solver_scaling.{png,pdf}
```

The plot uses `kernel_only.min_us`, the fastest measured iteration for each
configuration, and labels every point directly. Mean, p50, p95, and max remain
available in the JSON for variability analysis but are not drawn.

The default rank sweep holds `E=640` while varying `R=8..64`; the expert sweep
holds `R=32` while varying `E=64..1024`. Both use four replica slots per rank,
20 warmups, and 200 measured iterations. Override any dimension with environment
variables, for example:

```bash
RANKS="8 16 32 64 128 256" RANK_SWEEP_EXPERTS=2048 \
EXPERTS="128 256 512 1024 2048" EXPERT_SWEEP_RANKS=32 \
EXTRA_SLOTS=4 ITERATIONS=500 bash scripts/run_solver_scaling.sh
```

To enable the deterministic Stage 2 patience probe at every scale for an A/B
comparison, keep its JSON separate from the fixed-iteration default:

```bash
STAGE2_PATIENCE_ALL_SCALES=1 \
OUT_DIR=logs/solver_scaling_patience_all \
bash scripts/run_solver_scaling.sh
python eval/plot_solver_scaling.py --input-dir logs/solver_scaling_patience_all
```

Use an idle GPU for timing; unrelated kernels contaminate CUDA-event latency.
Select the benchmark device with `CUDA_DEVICE=<id>`.

`EPLB_PROFILE=1` emits a periodic per-region summary. CUDA events are **queued** and
resolved in one batch, so timing injects no per-region host sync — an instrumented run
and an undisturbed end-to-end step time can come from the same job.

| Region | Covers |
|---|---|
| `solve` | the placement + quota solver (CUDA kernel) |
| `all_gather_omega` | the single `Ω[R,E]` all-gather |
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

### Apply-mode memory: what actually allocates

Two per-layer buffers, both live across forward→backward, and the small one is not the problem:

| Buffer | Size | Qwen3-30B-A3B, EP=8, seq 4096 |
|---|---|---|
| `w_stacked` (per-slot expert weights) | `n_slot x \|W_e\|` | 297 MB |
| `x_pad` + its activations (dense grouped-GEMM batch) | `n_slot x cap x H` | **~12.7 GB** |

There is **no persistent or symmetric buffer**: `w_stacked` is a fresh
`[n_slot+1, *weight_shape]` allocated on every layer's forward, and *all* slots are copied into
it — the rank's own mains as well as the replicas — because the grouped GEMM needs one contiguous
stack. So even `EPLB_N_SLOT = num_experts/EP` (no replication headroom at all) still costs one
full duplicate of the layer's expert weights.

The dominant term is `cap`, the per-slot capacity that `grouped_expert_mlp` pads its dense
`[n_slot, cap, H]` batch to. `cap` is derived from `group_sizes.max()` — the exact per-slot token
count under the solved plan — which keeps the padded batch close to the real token count. Because
it is derived from the plan, **a better-balanced plan is also a cheaper one**. `EPLB_CAP=<int>`
overrides it; setting it *below* the true per-slot max silently drops tokens, so only raise it.

If apply mode still OOMs, in order of effect:

```bash
EPLB_REMATERIALIZE=1   # dist.broadcast path only: checkpoints the replication + expert GEMM so
                       # backward re-broadcasts instead of holding the stack. The GIN path never
                       # holds it, so this knob does not apply there.
EPLB_N_SLOT=16         # = num_experts/EP: no replica headroom, halves both buffers. Also removes
                       # the balancing freedom, so use it to isolate memory, not to benchmark.
SEQ_LEN=2048           # cap scales with tokens per rank, so this halves the padded batch too.
```

The GIN path keeps no replica weights across forward→backward: backward re-acquires them with a
second `get_batched` off the schedule cached at plan time, so the routing is never re-derived and
the resident cost is one layer's slots rather than `n_layers x n_slot x |W_e|` (~14 GB for a
48-layer Qwen3-30B-A3B at EP=8). Carrying them instead is not an option the window supports: it is
a single buffer that every layer recycles and the backward's gradient staging overwrites, so its
contents would have to be copied into ordinary memory and pinned there by autograd.

`--recompute-granularity selective --recompute-modules core_attn` only touches attention and does
nothing for the MoE path — it will not fix an apply-mode OOM.

### Reading the trace in Perfetto

`PROFILE_TRACE=1` turns on Megatron's PyTorch profiler; the region labels above are emitted as
`record_function` ranges **unconditionally** (no `EPLB_PROFILE` needed), so they show up inline
against the real CPU and CUDA rows:

```bash
PROFILE_TRACE=1 PROFILE_STEP_START=8 PROFILE_STEP_END=10 \
EPLB_MODE=apply ... bash scripts/run_real_moe.sh
# -> <PROFILE_DIR>/torch_profile/rank-<N>.json.gz   (default PROFILE_DIR=logs/prof_<mode>)
```

Open the `.json.gz` directly at [ui.perfetto.dev](https://ui.perfetto.dev) (gzipped Chrome
traces are handled natively) and search for `eplb/`. Knobs: `PROFILE_RANKS="0 8"` (space
separated, default rank 0 only — on a multi-node run pick one rank per node), `PROFILE_STACK=1`
for Python/CPU call stacks, `PROFILE_SHAPES=1` for tensor shapes. The launcher echoes the
resolved trace path. **Set `PROFILE_DIR` per run** when comparing modes or sweeping: the file is
named only by rank, so two runs sharing a `PROFILE_DIR` overwrite each other.

Only the `apply` trace shows the EP path Scale-EPLB replaces (`apply/dispatch`,
`apply/expert_compute`, `apply/combine`, `apply/weight_move`). An `observe` trace has just
`eplb/solve` and `eplb/all_gather_omega` laid over Megatron's own dispatcher, which is the
right picture for costing the solver but not for the end-to-end breakdown.

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
(use `convert_hf_to_mcore.sh` for Mixtral, or a preconverted Megatron Bridge checkpoint for
Qwen3) to show the operating point a trained router actually lands on.

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
`MASTER_ADDR`, `MASTER_PORT`; set `EP` so it divides `world_size / (TP * PP)`.

Real Qwen3-30B-A3B checkpoint + real Megatron indexed data + DeepEP token
dispatch (run on every node, changing only `NODE_RANK`):

```bash
source scripts/env_hdfs.sh
MEGATRON_DIR="${HOME}/Megatron-LM" \
EPLB_DIR="${HOME}/EP_balance" \
CHECKPOINT="${EPLB_CHECKPOINT_DIR}/qwen3_30b_a3b_mcore" \
DATA_PATH="${EPLB_INDEXED_DATA_DIR}/dapo_math_text_document" \
TOKENIZER_MODEL="${EPLB_TOKENIZER_DIR}/qwen3_30b_a3b" \
MOCK=0 FROM_SCRATCH=0 MODEL=qwen3_30b_a3b \
EPLB_MODE=apply EPLB_ADAPTER=deepep EPLB_DEEPEP_ALLOW_MNNVL=1 \
GPUS_PER_NODE=4 NNODES=4 NODE_RANK=<0..3> \
MASTER_ADDR=<node0-ip> MASTER_PORT=29500 \
TP=2 PP=1 EP=8 MICRO_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=256 \
SEQ_LEN=4096 TRAIN_ITERS=1000 \
  bash scripts/run_real_moe.sh
```

The Qwen recipe fixes the model vocabulary at the checkpoint's `151936` rows
and defaults to the Megatron Bridge `pre_mlp_layernorm.*` checkpoint layout.
Set `EPLB_SEPARATE_MLP_NORM_CKPT=0` only if a checkpoint already uses fused
`mlp.linear_fc1.layer_norm_*` keys.

The default optimizer is distributed Adam. For a memory-constrained functional
smoke test only, `USE_DISTRIBUTED_OPTIMIZER=0` plus
`--optimizer sgd --sgd-momentum 0.0` avoids Adam's FP32 optimizer states; do not
use that substitution as the convergence recipe.

### Apply-mode backends: DeepEP tokens + GIN weights

Apply mode moves two independent things across ranks, and they use different backends:

* **Tokens** — dispatch and combine, i.e. the routing traffic. `EPLB_ADAPTER=deepep` puts this on
  DeepEP. (`DEEPEP=1` is a different knob: it configures *Megatron's* flex dispatcher, which apply
  mode replaces outright, so it does nothing here.)
* **Expert weights and their gradients** — forward pulls each replica slot from `main(e)`
  (`nccl_gin.get_batched`), backward pushes replica grads back to `main(e)`'s scratch column
  (`nccl_gin.put_batched`) where the owner sums them. Both directions are the in-tree
  `nccl_gin/` backend over NCCL symmetric memory, selected by `EPLB_WEIGHT_COMM=gin`. DeepEP is
  not involved in either.

| Knob | Default | Device-initiated setting |
|---|---|---|
| `EPLB_ADAPTER` | `alltoall` | `deepep` — token dispatch/combine |
| `EPLB_WEIGHT_COMM` | host-driven `dist.broadcast` | `gin` — replica weight pull + grad reduce-to-main |
| `EPLB_GIN_FENCE` | `barrier` (host, not stream-ordered) | `signal` — device-stream, capture-safe |
| `EPLB_GIN_LSA` | `1` — intra-node peers over NVLink | `0` forces everything onto the network (A/B only) |
| `EPLB_CAP` | derived from the plan (one scalar D2H) | pin it to remove that read |
| `EPLB_DEEPEP_STATIC` | `0` | Reserved; the launcher rejects `1` until static padded-row handling is implemented |

```bash
EPLB_MODE=apply EPLB_ADAPTER=deepep EPLB_WEIGHT_COMM=gin \
EPLB_GIN_FENCE=signal EPLB_CAP=<int> \
  ... bash scripts/run_real_moe.sh
```

Selecting `gin` is enough — it always takes the re-pull path, so `EPLB_OVERLAP` is only meaningful
for the `dist.broadcast` transport.

### Which wire each replica transfer takes

One `ncclCommWindowRegister` maps the symmetric buffers for both transports at once: `ncclGinRegister`
for the network side, `cuMemMap` + `cuMemSetAccess` for peers whose memory is load/store reachable
(the LSA team, i.e. NVLink or PCIe P2P inside the node). The batched kernels pick per descriptor —
LSA-team peers are read and written with vector load/store, everyone else through GIN's RDMA — so
with the expert-parallel group inside one node the weight channel never reaches the NIC. Rank 0
prints the split at startup:

```
[eplb-gin] world=8 lsa_team=8 lsa_path=on -> up to 7 of 7 peers over NVLink, the rest over GIN
```

`lsa_team=1` means no peer is mapped and everything is going over the network; check P2P
availability before reading any weight-channel timing.

**Set `EPLB_GIN_FENCE=signal` for real runs.** The backward re-pull is started by a pre-hook on the
MoE block's output, so it is in flight across the scatter backward, the reverse combine all-to-all
and the Wgrad of GEMM-2 before Dgrad needs it. The default `dist.barrier` fence throws that away:
it is host-blocking, so hoisting the pull earlier only moves the CPU stall earlier. The signal fence
is enqueued on the weight stream and leaves the window intact. It is a barrier session rather than a
mesh of `ncclSignal`s, for the same reason the transfers split: a signal needs a GIN connection and
there is none to a peer inside the node, so the mesh form fails outright on a single-node EP group.

### NCCL build

GIN needs NCCL >= 2.29 with `ginType != NONE`, which is not what PyTorch bundles (2.27 at the time of
writing). The extension compiles against `$NCCL_HOME` but at runtime resolves `libnccl.so.2` to
whichever copy was loaded first, so without a preload it silently calls into PyTorch's older library
and `ncclCommQueryProperties` fails with `invalid argument`:

```bash
export LD_PRELOAD=$NCCL_HOME/lib/libnccl.so.2
```

DeepEP additionally compares the loaded `libnccl.so` byte-for-byte against the one under its NCCL
root, and by default that root is the `nvidia-nccl` wheel -- whose build usually reports
`ginType=NONE`. Point it at the GIN-capable copy instead and both backends share one library:

```bash
export EP_NCCL_ROOT_DIR=$NCCL_HOME          # what DeepEP compares against
export LD_PRELOAD=$NCCL_HOME/lib/libnccl.so.2
```

Without the first line `EPLB_ADAPTER=deepep EPLB_WEIGHT_COMM=gin` aborts at import with
`Invalid NCCL versions`; without the second, GIN init reports `ginType=NONE`.

`EPLB_CHUNKS=2` composes with all of the above. It splits this rank's routing units in two and
pipelines them across a compute and a comm stream, so `dispatch(c2)` hides behind `compute(c1)` and
`combine(c1)` behind `compute(c2)`, leaving only the first dispatch and last combine exposed. It is
a purely token-side split: the replica weights are acquired once per layer and shared by every
chunk, in both directions, so the chunk count does not appear in the weight traffic or in the
weight channel's collective schedule.

The chunked pipeline is one autograd node whose backward is written out by hand
(`eplb/integration/manual_block.py`); `EPLB_MANUAL_BWD=0` hands the ordering back to autograd. The two
are numerically identical and both are covered by the same reference tests — what the hand-written
schedule buys is two placements autograd cannot express:

| | autograd | hand-scheduled |
|---|---|---|
| forward weight pull | exposed before the first dispatch | on the weight stream, hidden behind `dispatch(c1)` |
| Dgrad vs Wgrad in a chunk | fused in one node, Wgrad first (it needs no weight) | Dgrad first, `dispatch⁻¹(k)` issued between the two, Wgrad under that transfer |
| backward reduce-to-main | last in the layer because autograd puts it there — it is the node nearest the parameters, so every chunk's Wgrad must reach it first | last in the layer because it is the one transfer nothing in the layer waits on |

Only Dgrad is on anyone's critical path: it produces `grad_x`, which the token channel carries back to
the token owners. The Wgrads feed the parameter reduction alone, so issuing `dispatch⁻¹(k)` between the
two halves gets the transfer out a Wgrad earlier and gives the Wgrads something to hide under. What
this gives up is the cushion for a late weight pull — Wgrad-first needs no weight and could absorb one
— which is why the pull is prefetched from the block output rather than from the expert backward.

The reduce overlap is real only when the weight channel has its own transport: `dist.reduce` shares
the token all-to-alls' NCCL communicator, and NCCL serialises same-communicator work whatever stream
it was enqueued on, so under the broadcast transport the reordering is host-side only. Under `gin` the
reduction is device-initiated on a separate channel and genuinely runs concurrently.

Ordering is invisible in gradients — every schedule produces the same numbers — so it is pinned
directly by `test_sync_free_two_chunk_backward_issue_order`, which asserts the sequence
`combine⁻¹(c1), combine⁻¹(c2), dispatch⁻¹(c1), dispatch⁻¹(c2), reduce`.

Pin `EPLB_CAP` when `EPLB_WEIGHT_COMM=gin`. The cap is a *token/compute-side* quantity — it sizes
`grouped_expert_mlp`'s padded `[n_slot, cap, H]` batch and DeepEP's static recv buffer — and has
nothing to do with the weight channel. It only becomes visible here because the broadcast path
already pays a control-plane D2H (`dist.broadcast` needs a host-side `src`) that the cap rides
along in for free, whereas GIN's schedule is fully device-resident, so deriving the cap becomes
the one standalone host read left in that block. The launcher prints a note when it is unset.
Size it from a run's per-slot max and only ever raise it: pinning it *below* the true max
silently drops tokens.

**Validate the combination before spending cluster hours on it.** The CPU suite pins
`AllToAllAdapter` over gloo, so neither backend is exercised there. On the target cluster:

```bash
EPLB_WEIGHT_COMM=gin RUN_GIN_TESTS=1 pytest -s tests/test_gin_weights.py
```

This checks outputs and `main(e)` gradients against a single-device reference for GIN alone, for
the GIN rematerialise/overlap path, and for DeepEP tokens combined with GIN weights. The two
backends never talk to each other, but they agree on slot *ordering* — tokens arrive grouped by
physical slot and the weight stack is indexed by that same slot — so a disagreement between them
produces silently wrong numbers rather than a crash, which is why the combination is worth its own
case. The DeepEP case runs in bf16 on purpose: DeepEP's kernels only move 16B-aligned bf16/fp16
rows and fall back to `all_to_all_single` for anything else, so an fp32 run would pass without
entering DeepEP at all.

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
plus `DATA_PATH`/`TOKENIZER_MODEL` to forward to `run_real_moe.sh`. Choose either
`FROM_SCRATCH=1` for random initialization or `CHECKPOINT=<mcore-dir>` for pretrained weights.
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
(attach EPLB). For real data without a checkpoint, drop `MOCK=1` and add
`FROM_SCRATCH=1 DATA_PATH=<prefix> TOKENIZER_MODEL=<repo-or-dir>`. To load pretrained
weights instead, omit `FROM_SCRATCH=1` and add `CHECKPOINT=<mcore-dir>`. Experts always
run the unfused `local` path, so absolute step times are higher
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
