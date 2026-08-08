# Load-balancer baselines

This directory provides a common benchmark harness for:

- `scale-eplb`: this repository's GPU-native placement and exact quota solver.
- `deepseek-eplb`: the official DeepSeek EPLB placement heuristic, vendored
  from the MIT-licensed reference implementation to avoid the `eplb` package
  name collision.
- `fastermoe`: FasterMoE's dynamic-shadowing load balancer (PPoPP'22,
  Algorithm 1), adapted from the reference `_global_policy` prototype.
- `flexmoe`: FlexMoE's dynamic device-placement load balancer (SIGMOD'23,
  Algorithms 1-2), a vExpert-based Expand/Shrink greedy.
- `lplb`: an adapter around the official compiled LPLB `Planner`.

## Run

```bash
cd /home/tiger/EP_balance

# Scale-EPLB + DeepSeek EPLB + FasterMoE + FlexMoE (works with this repository's deps)
python -m baseline.benchmark --strategies scale,eplb,fastermoe,flexmoe

# Include official LPLB after building its CUDA extension
python -m baseline.benchmark --strategies scale,eplb,fastermoe,flexmoe,lplb

# Machine-readable output
python -m baseline.benchmark --strategies scale,eplb,fastermoe,flexmoe --json
```

LPLB must be built separately because it requires CUDA >= 12.6.3,
cuSolverDx/cuBLASDx, and a compiled extension:

```bash
cd /home/tiger/LPLB
./download-mathdx.sh
pip install --no-build-isolation -e .
```

The default comparison uses 32 EP ranks, 640 logical experts and
`N_slot=4` additional replica slots per rank. Each rank therefore has 20 main
experts plus 4 replica slots (24 physical slots total), which is compatible
with LPLB's 8-rank/two-edge cube topology.

LPLB is not restricted to even replica budgets. The harness exposes
`--lplb-topology auto|cube|ring`: `auto` selects the official one-edge Ring for
odd `N_slot` and the two-edge Cube for even `N_slot`. For an `N_slot=1..4`
experiment, use `--lplb-topology ring` for every point so that only the slot
budget changes; the one-edge Ring supports every positive integer budget.

## Unified `N_slot` semantics

Throughout this baseline harness, `--n-slot` means the number of **additional
expert replica slots available per rank**. It never includes the
`num_experts / num_ranks` main experts. Consequently:

- main slots per rank = `num_experts / num_ranks`;
- physical slots per rank = main slots + `N_slot`;
- global replica budget = `num_ranks * N_slot`.

Scale-EPLB and FlexMoE may leave some replica slots unused. DeepSeek EPLB and
LPLB consume their configured physical layout. FasterMoE's original global
shadow policy has no slot limit, so this harness applies the same per-rank
replica cap before accepting another globally shadowed expert. Results report
both actual `replicas` (additional copies only) and `physical_instances`
(mains plus copies).

## Detailed placement output

Add `--details` to a synthetic benchmark to emit the full placement-derived
plan:

```bash
python -m baseline.benchmark --details --json \
  > logs/baseline_skew1.5_details.json
```

For each strategy, `placement_details` contains:

- `replica_transfer_events`: logical expert, source main rank, destination rank,
  and number of copied instances;
- `relocation_events`: experts whose original common main is not retained by a
  reordering baseline;
- `changed_experts`: all hosts of every replicated or relocated expert;
- `rank_summary`: main experts, hosted instances, replica usage, and rank load
  before/after planning.

These events are derived from each planner's placement matrix. They describe
the planned logical movement but are not a measured NCCL/GIN transfer trace.
Detailed output is currently limited to a single synthetic placement; trace
replay generates one placement per sample and therefore rejects `--details`.

## Replay a real routing trace

Instead of the synthetic Zipf workload, every baseline can be scored on the
*real* per-(layer, micro-batch) routing captured during Megatron training. Run
observe mode (Phase B) with `EPLB_TRACE_OUT` set to dump the gathered
`Ω[R, E]` matrices, then replay that file through the harness:

```bash
# 1) capture the real trace during training (rank 0 writes the file)
cd /home/tiger/EP_balance
EPLB_MODE=observe EPLB_TRACE_OUT=logs/trace.pt \
MEGATRON_DIR=/home/tiger/Megatron-LM \
  bash scripts/run_real_moe.sh          # add MODEL/CHECKPOINT/... as usual

# 2) score every baseline on the captured routing
python -m baseline.benchmark --trace logs/trace.pt \
  --strategies scale,eplb,fastermoe,flexmoe,lplb
```

The trace is self-describing: it records the topology, `main(e)` placement,
per-expert weight bytes, `s_tok` and the physical slot count used during the
run. Replay subtracts the evenly placed main experts to recover the common
replica-only `N_slot`, so you do **not** re-pass `--nodes`, `--experts`,
`--n-slot`, etc. Extra knobs:

- `--trace-max-samples N` — replay only the first `N` samples.
- `EPLB_TRACE_MAX` / `EPLB_TRACE_EVERY` (capture side) — cap the number of
  captured samples and set the disk-flush cadence.

In `--trace` mode each strategy replans per sample from **that** sample's load
(so DeepSeek/LPLB use the current batch as their own placement history), and the
CLI prints per-strategy `solve_ms(mean)`, `theta(mean)`, and `imbalance(mean/p90)`
aggregated over all replayed samples. The same `quality=` caveats below apply.

## Interpreting timings

The CLI deliberately does not print a single shared `latency` column because
the repositories implement different portions of the control path:

- Scale-EPLB reports `solve_ms`/`total_ms` for placement plus exact quota
  generation.
- DeepSeek EPLB reports `placement_ms`, including aggregation and GPU-to-CPU
  transfer. Its repository does not implement runtime token routing, so
  `total_ms` is unavailable.
- LPLB reports static `placement_ms` separately from per-batch `solve_ms`
  (`solve_probs`). Token counting, distributed reduction and physical-index
  mapping are not included because this harness starts from a common aggregated
  load matrix.

Do not compare `placement_ms` from one row to `solve_ms` from another as if they
represented the same operation.

DeepSeek EPLB and LPLB placement use an independent synthetic historical load;
quality is evaluated on the current load. Scale-EPLB is intentionally dynamic
and solves from the current per-source load matrix.

## Interpreting quality

The algorithms do not expose identical outputs:

- Scale-EPLB emits exact `Q[src, expert, dst]`; its reported load is realized
  directly from that quota tensor.
- DeepSeek EPLB emits placement only; its reported load uses the algorithm's
  ideal equal-split-across-replicas assumption.
- FasterMoE emits a per-batch set of shadowed (globally replicated) hot experts;
  its reported load is the token load after each shadowed expert is spread back
  onto its source ranks. Its shadow decision is dynamic (recomputed every batch).
  The original policy is unconstrained, but this harness stops shadow admission
  when another global copy would exceed the common per-rank `--n-slot` budget.
- FlexMoE replicates hot experts across `--n-slot` vExpert slots per rank until the
  balance ratio (`max_g load_g / mean_g`) drops under `--flexmoe-threshold`, then
  packs the vExperts onto ranks; its reported load is the even-split per-vExpert
  load (Eq. 6). It never changes token routing and deliberately tolerates residual
  imbalance up to the threshold, so it may use fewer than all available slots.
- LPLB emits LP split ratios for a fixed redundant topology; its reported load
  is reconstructed from those ratios.

Every synthetic JSON row includes `quality_rank_load_by_rank`; element `r` is
the token-expert assignment load used to evaluate logical rank `r`.
`quality_theta`, `quality_mean_load`, and `quality_imbalance` are respectively
the maximum, mean, and max/mean ratio of this vector.

Quality values must be read together with the printed `quality=` label. Do not
treat the three `quality_theta` values as identical end-to-end dispatch
measurements. Scale-EPLB and FlexMoE may use fewer than all available replica
slots, while DeepSeek EPLB and the configured LPLB topology consume the fixed
replica budget.

DeepSeek EPLB runs on CPU by design. Scale-EPLB and LPLB are timed with CUDA
events after warm-up; mixed CPU/CUDA paths use synchronized wall time.
