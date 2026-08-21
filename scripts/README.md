# Scale-EPLB cluster scripts

Run MoE on real Megatron-LM with three behaviors selected by `EPLB_MODE`:

- **Baseline (`off`)** — execute Megatron's native MoE path unchanged; optional timing wrappers
  observe its router, token dispatch, experts, and combine methods.
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
| `run_real_moe.sh` | Real-model launcher; `REAL=1` from `run_gb200_4x4.sh` forwards here. `MOCK=1` = mock-data + random init; `MOCK=0 FROM_SCRATCH=1` = real data + random init; `DEEPEP=1` = native DeepEP dispatch. |
| `model_recipes.sh` | Architecture presets for Qwen3, 160E DeepSeek-V2, and 128E GLM-4.5-Air, with launch-time depth override. |
| `run_slot_sweep.sh` | Sweep `N_slot=1..4`; save raw JSON, flat CSV, seed summary CSV, and PNG/PDF under the shared experiment directory. |
| `run_solver_scaling.sh` | Sweep the Scale-EPLB CUDA solver over logical rank and expert counts; save raw JSON, flat CSV, and PNG/PDF under the shared experiment directory. |
| `export_sweep_csv.py` | Flatten existing slot-sweep or solver-scaling JSON reports into CSV without rerunning a benchmark. |
| `prepare_open_workload.py` | Download task/corpus workloads, extract model inputs, and optionally build Megatron `.bin/.idx`. |
| `eval/plot_solver_scaling.py` | Read an existing solver-scaling JSON directory and independently generate PNG/PDF plots. |
| `install_megatron.sh` | Clone+install pinned community Megatron-LM, self-check `import megatron`. |
| `install_deepep.sh` | Optional: clone+build DeepEP (NCCL Gin backend) for the sync-free transport. |

> **Install** (clone + `install_megatron.sh` / `install_deepep.sh` + `pip install -e`)
> lives in the [top-level README](../README.md#cluster-install-megatron-integration). This file is the
> **run book**: launchers, run recipes, toggles, and troubleshooting.

## Model selection

Select the architecture with one environment variable:

```bash
MODEL=qwen3_30b_a3b
MODEL=deepseek_v2_160e
MODEL=glm45_air
```

Every recipe defaults to the official layer count: Qwen3 has 48 MoE layers,
DeepSeek-V2 has 60 layers (1 dense + 59 MoE), and GLM-4.5-Air has 46 layers
(1 dense + 45 MoE). Set the total Transformer depth at launch with
`NUM_LAYERS=<N>`; the mixed models preserve their dense prefix automatically:

```bash
# 1 dense + 5 MoE layers
MODEL=deepseek_v2_160e NUM_LAYERS=6 bash scripts/run_real_moe.sh
MODEL=glm45_air NUM_LAYERS=6 bash scripts/run_real_moe.sh

# Qwen has no dense prefix, so this is 5 MoE layers
MODEL=qwen3_30b_a3b NUM_LAYERS=5 bash scripts/run_real_moe.sh
```

Reduced random-init models can run with `MOCK=1` or `FROM_SCRATCH=1`; loading
weights requires a Megatron-Core checkpoint converted and truncated to the
selected `NUM_LAYERS`.

Shared experts remain outside EPLB placement and are added to the routed-expert
output in `apply` mode. Shared-expert communication overlap is deliberately
disabled in all three modes so their step times use the same execution schedule.

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
Qwen tokenizer, installs `nvidia-nccl-cu13==2.30.7` with `--no-deps`, and
creates this persistent layout:

```text
/mnt/hdfs/__MERLIN_USER_DIR__/eplb_data/
├── cache/huggingface/   # HF model/dataset downloads
├── raw/                 # JSONL + manifests
├── indexed/             # Megatron .bin/.idx
├── tokenizers/
├── checkpoints/
├── exp/                 # benchmark JSON, CSV, PNG, and PDF artifacts
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
at `${EPLB_CHECKPOINT_DIR}/qwen3_30b_a3b_mcore`. The pinned community Megatron
converter does not contain a Qwen3 loader; use Megatron Bridge's
`examples/conversion/convert_checkpoints.py import` instead. `FROM_SCRATCH=1`
skips the checkpoint entirely and random-initializes the model.

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
# Runs the benchmark, then writes JSON + CSV + PNG/PDF.
bash scripts/run_solver_scaling.sh
# -> ${EPLB_EXP_DIR}/solver_scaling/
#    ├── {rank_scale,expert_scale}_r*_e*.json
#    ├── solver_scaling.csv
#    └── solver_scaling.{png,pdf}
```

The plot uses `kernel_only.min_us`, the fastest measured iteration for each
configuration, and labels every point directly. Mean, p50, p95, and max remain
available in `solver_scaling.csv` for variability analysis but are not drawn.

The N_slot sweep follows the same layout:

```bash
bash scripts/run_slot_sweep.sh
# -> ${EPLB_EXP_DIR}/slot_sweep/
#    ├── baseline_skew*_slot*_seed*.json
#    ├── slot_sweep.csv              # one row per strategy / seed / N_slot
#    ├── slot_sweep_summary.csv      # seed-aggregated key metrics
#    └── slot_imbalance.{png,pdf}
```

Set `PLOT=0` to produce JSON and CSV without invoking Matplotlib. The standalone
plot scripts also use `${EPLB_EXP_DIR}` by default after `source scripts/env_hdfs.sh`.

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
OUT_DIR="${EPLB_EXP_DIR}/solver_scaling_patience_all" \
bash scripts/run_solver_scaling.sh
```

Use an idle GPU for timing; unrelated kernels contaminate CUDA-event latency.
Select the benchmark device with `CUDA_DEVICE=<id>`.

For a compact forward/backward breakdown of each MoE invocation, enable debug timing in any mode:

```bash
# Pure Megatron baseline: native router / dispatcher / experts remain unchanged.
EPLB_MODE=off EPLB_DEBUG_TIMING=1 \
  bash scripts/run_real_moe.sh
# [EPLB-debug r0] mode=off layer=0 mb=0 moe_fwd_total=...ms \
#   solver=n/a omega_gather=n/a \
#   router=0.391ms expert_transfer=n/a dispatch=2.107ms \
#   expert_gemm=4.936ms combine=1.988ms
# [EPLB-debug r0] mode=off direction=backward layer=0 mb=0 \
#   moe_bwd_total=...ms combine_bwd=...ms expert_bwd=...ms dispatch_bwd=...ms

# Native Megatron execution plus Ω collection and the EPLB solver; plan is not applied.
EPLB_MODE=observe EPLB_DEBUG_TIMING=1 \
  bash scripts/run_real_moe.sh

# Scale-EPLB dispatcher.
EPLB_MODE=apply EPLB_DEBUG_TIMING=1 \
  bash scripts/run_real_moe.sh
# [EPLB-debug r0] mode=apply layer=0 mb=0 ... \
#   moe_fwd_total=...ms \
#   expert_transfer=1.824ms/512.00MiB/294.36GB/s \
#   expert_transfer_wire=...ms(x2)/512.00MiB/...GB/s \
#   dispatch=2.107ms/64.00MiB/31.86GB/s ... \
#   combine=1.988ms/64.00MiB/33.75GB/s
# [EPLB-debug r0] mode=apply direction=backward layer=0 mb=0 \
#   moe_bwd_total=...ms \
#   expert_repull=1.791ms/512.00MiB/299.71GB/s \
#   expert_repull_wire=...ms(x2)/512.00MiB/...GB/s \
#   combine_bwd=...ms(x2)/64.00MiB/...GB/s \
#   expert_dgrad=...ms(x4) activation_bwd=...ms(x2) \
#   dispatch_bwd=...ms(x2)/64.00MiB/...GB/s expert_wgrad=...ms(x2) \
#   expert_grad_reduce=2.031ms/512.00MiB/264.30GB/s \
#   expert_grad_put_wire=...ms(x2)/512.00MiB/...GB/s
```

`EPLB_DEBUG_TIMING=1` synchronizes at each invocation boundary so that one line contains only
that layer/micro-batch. It is for diagnosis; do not quote the instrumented run's end-to-end step
time. `off` and `observe` wrap the native Megatron leaf methods without replacing
`MoELayer.forward`; `expert_transfer=n/a` is expected because native Megatron does not replicate
expert weights. `off` also reports `solver=n/a` and `omega_gather=n/a`. Launch through
`run_real_moe.sh` (or `pretrain_eplb_moe.py`): invoking Megatron's unmodified `pretrain_gpt.py`
directly does not install these wrappers.

Set `EPLB_PROFILE_ALL_RANKS=1` to print one line per rank. Under `EPLB_CHUNKS>=2`, an `(xN)`
suffix means the displayed value is the sum of the `N` chunk events. Dispatch, expert GEMM,
combine, and expert transfer can overlap in the apply pipeline, so their values are not additive.
`moe_fwd_total` spans the complete MoE forward invocation in both native (`off`/`observe`) and
Scale-EPLB (`apply`) modes. `moe_bwd_total` starts when the final MoE output receives its gradient.
On the default manual apply path it ends at the event immediately following the last
reverse-dispatch on the token-communication stream. On the native path, its graph endpoint is
inserted directly before `token_dispatch`, so the backward event fires immediately after that
dispatch collective's inverse communication. This marker sits inside Megatron's optional
delayed-Wgrad boundary: delayed Wgrad and attention backward are therefore excluded. Apply's
following Wgrad/gradient-reduce tail, including time for which gradient-reduce remains live under
attention backward, is excluded too. Stock native autograd usually executes expert Wgrad before
reverse-dispatch, so that work remains inside native `moe_bwd_total`; the two modes have the same
endpoint but not necessarily identical internal composition. The backward line is emitted
immediately with its layer/micro-batch identity. Other reference autograd paths still defer their
phase samples to the next forward boundary (or process exit).

The manual apply path reports its two expert Dgrad GEMMs per chunk as `expert_dgrad`, activation
backward separately, and both Wgrad GEMMs per chunk as `expert_wgrad`. Native Megatron's local
tensor-parallel linear computes Dgrad and Wgrad inside one autograd function, so non-invasive
instrumentation reports the complete native experts subgraph as `expert_bwd`; it cannot split that
node into Dgrad/Wgrad without modifying Megatron's linear backward.

Transfer fields also report logical remote payload and effective payload bandwidth:

```text
GB/s = payload_bytes / (elapsed_ms * 1e6)
```

For GIN/TMA, payload is the sum of `W1 + W2` bytes for this rank's remote replica slots; local
main-owned slots are excluded. `expert_transfer` is the forward pull, `expert_repull` is the
backward pull, and `expert_grad_reduce` is the gradient put/reduce-to-main. Their elapsed times
cover the complete runtime operation (fences, staging and local reduction where applicable), so
the result is end-to-end effective bandwidth rather than a raw copy-engine peak.

The corresponding `*_wire` fields time only the two `get_batched` or `put_batched` CUDA kernels
for W1 and W2 (hence `(x2)`). They exclude both world fences, local HBM staging, effective-stack
assembly, scratch clearing, and the owner-side gradient sum. For GIN put, this is sender-side
kernel/flush time; the following fence still defines when every owner may safely consume the data.

For token transport, payload counts only routing units whose destination is another rank.
Dispatch includes the transported row's metadata/alignment padding; DeepEP combine includes its
aligned row width. DeepEP protocol headers and the separately transported top-k index are not
counted. Therefore the printed number is **effective tensor-payload bandwidth**, not physical
NVLink/InfiniBand wire bandwidth. Obtain the latter from NVLink/NIC performance counters.

In a hybrid run, TMA/LSA and GIN descriptors execute inside the same batched kernel, so one elapsed
time cannot identify separate TMA and GIN bandwidths. Benchmark them independently:

```bash
# All remote peers in one LSA team: TMA/NVLink weight path.
EPLB_MODE=apply EPLB_WEIGHT_COMM=gin EPLB_GIN_LSA=1 EPLB_DEBUG_TIMING=1 \
  bash scripts/run_real_moe.sh

# Force every remote descriptor through network GIN.
EPLB_MODE=apply EPLB_WEIGHT_COMM=gin EPLB_GIN_LSA=0 EPLB_DEBUG_TIMING=1 \
  bash scripts/run_real_moe.sh
```

For a distributed aggregate, collect every rank and compute
`sum(payload_bytes across ranks) / max(elapsed time across ranks)`. Do not sum per-rank GB/s.

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
| `apply/expert_gemm` | only the two batched expert GEMMs and activation |
| `apply/combine` | Stage 5 output all-to-all |
| `apply/weight_move` | replica weight broadcast / GIN pull; may overlap token communication |
| `apply/weight_repull` | backward replica-weight re-pull on the weight stream |
| `apply/grad_move` | replica gradient put/reduce-to-main |
| `native/route` | Megatron router forward (`off` / `observe`) |
| `native/dispatch` | Megatron `token_dispatch` communication |
| `native/expert_gemm` | Megatron native experts module |
| `native/combine` | Megatron `token_combine` communication |
| `native/shared_expert` | native shared-expert module when it executes serially |

A synchronous EP step is paced by its slowest rank, so straggler analysis needs
**every** rank to report, not just rank 0:

```bash
EPLB_MODE=apply EPLB_PROFILE=1 EPLB_PROFILE_ALL_RANKS=1 EPLB_PROFILE_RESET_AT=1 \
  bash scripts/run_real_moe.sh
# every line is tagged [EPLB-profile r<RANK>] -> compare max vs mean across ranks
grep 'apply/expert_compute' logs/real_*.log
```

For the per-layer debug breakdown, collect every rank and let the extractor take max-over-ranks
for each matching `(layer, mb, phase)` before summing layers:

```bash
EPLB_DEBUG_TIMING=1 EPLB_PROFILE_ALL_RANKS=1 \
  bash scripts/run_real_moe.sh
python eval/extract_eplb_debug.py \
  --log logs/real_*_node*.log \
  --out-dir results/debug_breakdown \
  --warmup 100 --expected-ranks 32
```

`critical_rank.csv` contains the critical rank and latency for every layer invocation. The summary
uses `sum_layer(max_rank(phase_ms))`; it does not take the maximum of per-rank medians.

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

Elastic/GIN runs normally require `EPLB_PROFILE=0 PROFILE_TRACE=0` to preserve the zero-sync
contract. `EPLB_DEBUG_TIMING=1` is the explicit diagnostic exception: it is accepted and prints
the same breakdown, but the launcher warns that the run is no longer a zero-sync throughput
measurement.

### Apply-mode memory: what actually allocates

The SM90 production path has two relevant buffer families:

| Buffer | Size | Qwen3-30B-A3B, EP=8, seq 4096 |
|---|---|---|
| `w_stacked` (per-slot expert weights) | `n_slot x \|W_e\|` | 297 MB |
| packed token rows + grouped-GEMM activations | proportional to actual/receive-bound rows | bounded by `EPLB_MAX_RECV_ROWS` |

There is **no persistent or symmetric buffer**: `w_stacked` is a fresh
`[n_slot+1, *weight_shape]` allocated on every layer's forward, and *all* slots are copied into
it — the rank's own mains as well as the replicas — because the grouped GEMM needs one contiguous
stack. So even `EPLB_N_SLOT = num_experts/EP` (no replication headroom at all) still costs one
full duplicate of the layer's expert weights.

There is no dense `[n_slot, cap, H]` compute batch in production. SM90
`torch._grouped_mm` consumes slot-major packed rows plus device offsets and computes exactly the
received token rows, including forward, Dgrad, and Wgrad. The old padded per-slot-capacity setting
is not part of the training interface.

If apply mode still OOMs, in order of effect:

```bash
EPLB_REMATERIALIZE=1   # dist.broadcast path only: checkpoints the replication + expert GEMM so
                       # backward re-broadcasts instead of holding the stack. The GIN path never
                       # holds it, so this knob does not apply there.
EPLB_N_SLOT=16         # = num_experts/EP: no replica headroom, shrinks the weight stack. Also removes
                       # the balancing freedom, so use it to isolate memory, not to benchmark.
EPLB_MAX_RECV_ROWS=... # bound packed rows after dispatch; exceeding it aborts asynchronously.
SEQ_LEN=2048           # reduces the real token rows and their grouped-GEMM activations.
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

The `off` and `observe` traces expose Megatron's unchanged native path as `native/route`,
`native/dispatch`, `native/expert_gemm`, and `native/combine`; `observe` additionally shows
`solve` and `all_gather_omega`. The `apply` trace exposes the replacement path as
`apply/dispatch`, `apply/expert_compute`, `apply/expert_gemm`, `apply/combine`, and
`apply/weight_move`.

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
(a preconverted Megatron Bridge checkpoint for Qwen3) to show the operating point a trained
router actually lands on.

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

Real Qwen3-30B-A3B checkpoint + real Megatron indexed data + zero-sync
ElasticBuffer/GIN (run on every node, changing only `NODE_RANK`):

```bash
source scripts/env_hdfs.sh
MEGATRON_DIR="${HOME}/Megatron-LM" \
EPLB_DIR="${HOME}/EP_balance" \
CHECKPOINT="${EPLB_CHECKPOINT_DIR}/qwen3_30b_a3b_mcore" \
DATA_PATH="${EPLB_INDEXED_DATA_DIR}/dapo_math_text_document" \
TOKENIZER_MODEL="${EPLB_TOKENIZER_DIR}/qwen3_30b_a3b" \
MOCK=0 FROM_SCRATCH=0 MODEL=qwen3_30b_a3b \
EPLB_MODE=apply EPLB_ADAPTER=deepep \
EPLB_WEIGHT_COMM=gin EPLB_GIN_FENCE=signal \
EPLB_DEEPEP_HYBRID=1 EPLB_PROFILE=0 PROFILE_TRACE=0 \
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

* **Tokens** — `EPLB_ADAPTER=deepep` uses only DeepEP V2 `ElasticBuffer`; there is no legacy
  `Buffer` or `all_to_all_single` fallback. (`DEEPEP=1` configures Megatron's replaced dispatcher.)
* **Expert weights and their gradients** — forward pulls each replica slot from `main(e)`
  (`nccl_gin.get_batched`), backward pushes replica grads back to `main(e)`'s scratch column
  (`nccl_gin.put_batched`) where the owner sums them. Both directions are the in-tree
  `nccl_gin/` backend over NCCL symmetric memory, selected by `EPLB_WEIGHT_COMM=gin`. DeepEP is
  not involved in either.

| Knob | Default | Device-initiated setting |
|---|---|---|
| `EPLB_ADAPTER` | `alltoall` | `deepep` — ElasticBuffer token dispatch/combine |
| `EPLB_WEIGHT_COMM` | host-driven `dist.broadcast` | `gin` — replica weight pull + grad reduce-to-main |
| `EPLB_GIN_FENCE` | `barrier` (host, not stream-ordered) | `signal` — device-stream, capture-safe |
| `EPLB_GIN_LSA` | `1` — intra-node peers over NVLink | `0` forces everything onto the network (A/B only) |
| `EPLB_GIN_LSA_TMA` | `1` — SM90+ TMA copy | `0` keeps LSA routing but restores vector load/store (A/B/fallback) |
| Expert compute | SM90+ BF16 ragged `torch._grouped_mm` | no padded fallback or per-slot cap |
| `EPLB_DEEPEP_HYBRID` | `1` | NVLink scale-up plus GIN scale-out |
| `EPLB_DEEPEP_MAX_TOKENS_PER_RANK` | first chunk's static size | optional larger per-chunk input bound |

```bash
EPLB_MODE=apply EPLB_ADAPTER=deepep EPLB_WEIGHT_COMM=gin \
EPLB_GIN_FENCE=signal \
EPLB_PROFILE=0 PROFILE_TRACE=0 \
  ... bash scripts/run_real_moe.sh
```

This is the only supported zero-sync recipe. `EPLB_DEEPEP_STATIC`, `ALLOW_MNNVL`, and the legacy
NVL/RDMA byte knobs are removed. ElasticBuffer initialization and GIN symmetric-buffer setup each
synchronize once before warmup; iterations use GPU prefix counts and never read receive sizes on host.

### Which wire each replica transfer takes

One `ncclCommWindowRegister` maps the symmetric buffers for both transports at once: `ncclGinRegister`
for the network side, `cuMemMap` + `cuMemSetAccess` for peers whose memory is load/store reachable
(the LSA team, i.e. NVLink or PCIe P2P inside the node). The batched kernels pick per descriptor —
LSA-team peers are read and written with an SM90+ TMA-staged copy, everyone else through GIN's RDMA — so
with the expert-parallel group inside one node the weight channel never reaches the NIC. Rank 0
prints the split at startup:

```
[eplb-gin] world=8 lsa_team=8 lsa_path=tma -> up to 7 of 7 peers over NVLink, the rest over GIN
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

ElasticBuffer plus GIN requires NCCL >= 2.30.4 with `ginType != NONE`; this repository pins 2.30.7.
Every launcher sources `scripts/env_nccl_2307.sh` before Python starts. It sets `NCCL_HOME` and
`EP_NCCL_ROOT_DIR`, removes other NCCL entries from `LD_PRELOAD`, and pins build/runtime linkage
to the same wheel. Source it explicitly before manual Python or pytest commands:

```bash
source scripts/env_nccl_2307.sh
```

`torch.cuda.nccl.version()` still reports PyTorch's compile-time 2.28.9; it cannot change without
rebuilding PyTorch. Startup instead checks `ncclGetVersion` on the loaded shared object and must
print `(2, 30, 7)`. The adapter uses `EP_REUSE_NCCL_COMM=0` so Elastic owns a compatible
communicator; that one-time setup may sync.

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
| backward reduce-to-main | last in the layer because autograd puts it there — it is the node nearest the parameters, so every chunk's Wgrad must reach it first | issued after the Wgrads on the weight stream; the token branch continues through router/attention, and an end-of-backward callback joins before expert `AccumulateGrad`/DDP |

Only Dgrad is on anyone's critical path: it produces `grad_x`, which the token channel carries back to
the token owners. The Wgrads feed the parameter reduction alone, so issuing `dispatch⁻¹(k)` between the
two halves gets the transfer out a Wgrad earlier and gives the Wgrads something to hide under. What
this gives up is the cushion for a late weight pull — Wgrad-first needs no weight and could absorb one
— which is why the pull is prefetched from the block output rather than from the expert backward.

The reduce overlap is real only when the weight channel has its own transport: `dist.reduce` shares
the token all-to-alls' NCCL communicator, and NCCL serialises same-communicator work whatever stream
it was enqueued on, so under the broadcast transport the reordering is host-side only. Under `gin` the
reduction is device-initiated on a separate channel and genuinely runs concurrently. Expert weights
are deliberately not direct differentiable inputs of the manual node: otherwise autograd schedules
their leaf accumulators before following the token gradient into router/attention and the required
stream wait would still block the critical path. The queued leaf-only callback preserves normal
parameter hooks and Megatron `main_grad`/DDP behavior after the asynchronous reduction is complete.

Ordering is invisible in gradients — every schedule produces the same numbers — so it is pinned
directly by `test_sync_free_two_chunk_backward_issue_order`, which asserts the sequence
`combine⁻¹(c1), combine⁻¹(c2), dispatch⁻¹(c1), dispatch⁻¹(c2), reduce`.

Elastic compute is SM90+ BF16 ragged grouped GEMM only. Per-slot counts remain on device as grouped
offsets, so there is no padded compute batch or per-slot capacity setting. With
`do_cpu_sync=False`, non-expanded Elastic receive storage is still the worst case
`[EP * max_units_per_chunk, padded_hidden]`; use fixed-size chunks and lower
`EPLB_DEEPEP_MAX_TOKENS_PER_RANK`, or set a validated total `EPLB_MAX_RECV_ROWS`, when memory is
tight.

**Validate the combination before spending cluster hours on it.** The CPU suite pins
`AllToAllAdapter` over gloo, so neither backend is exercised there. On the target cluster:

```bash
source scripts/env_nccl_2307.sh
EPLB_WEIGHT_COMM=gin EPLB_GIN_FENCE=signal RUN_GIN_TESTS=1 \
  pytest -s tests/test_gin_weights.py
```

This checks outputs plus token, router and `main(e)` weight gradients for single/two chunks,
autograd/manual backward, and runs the warmed full Elastic+GIN forward/backward under
`torch.cuda.set_sync_debug_mode("error")`. The two
backends never talk to each other, but they agree on slot *ordering* — tokens arrive grouped by
physical slot and the weight stack is indexed by that same slot — so a disagreement between them
produces silently wrong numbers rather than a crash. Elastic tests run in BF16 and reject unsupported
types rather than falling back to a host-synchronized transport.

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

ElasticBuffer zero-sync smoke (same command on all four nodes; change only `NODE_RANK`):

```bash
MEGATRON_DIR=/home/tiger/Megatron-LM EPLB_DIR=/home/tiger/EP_balance \
NNODES=4 NODE_RANK=<0..3> MASTER_ADDR=<node0-ip> MASTER_PORT=29500 \
EPLB_MODE=apply EPLB_ADAPTER=deepep \
EPLB_WEIGHT_COMM=gin EPLB_GIN_FENCE=signal \
EPLB_N_SLOT=4 \
EPLB_CHUNKS=2 EPLB_DEEPEP_MAX_TOKENS_PER_RANK=4096 \
EPLB_PROFILE=0 PROFILE_TRACE=0 \
EP_SIZE=16 NUM_EXPERTS=32 TOPK=4 TRAIN_ITERS=3 \
  bash scripts/run_gb200_4x4.sh
```

Here `SEQ_LEN=2048`, `TOPK=4`, so each rank has 8192 routing units and each of two chunks has
4096. Ragged grouped GEMM computes only the received rows. The log must print
`transport=deep_ep.buffers.elastic.ElasticBuffer GIN=ready`; keep synchronous in-training
profilers disabled for this zero-sync check.

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
