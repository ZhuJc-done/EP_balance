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

The default comparison uses 32 EP ranks, 64 logical experts and four physical
slots per rank. This gives two original and two redundant experts per rank,
which is compatible with LPLB's 8-rank/two-edge cube topology.

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
per-expert weight bytes, `s_tok` and `n_slot` used during the run, so the replay
rebuilds an identical `Topology`/`ProblemSpec` automatically — you do **not**
re-pass `--nodes`, `--experts`, `--n-slot`, etc. Extra knobs:

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
  onto its source ranks. Its shadow decision is dynamic (recomputed every batch)
  and, like the original, is unconstrained by `--n-slot`, since shadowing is a
  transient per-iteration weight broadcast rather than a permanent slot placement.
- FlexMoE replicates hot experts across `--n-slot` vExpert slots per rank until the
  balance ratio (`max_g load_g / mean_g`) drops under `--flexmoe-threshold`, then
  packs the vExperts onto ranks; its reported load is the even-split per-vExpert
  load (Eq. 6). It never changes token routing and deliberately tolerates residual
  imbalance up to the threshold, so it may use fewer than all available slots.
- LPLB emits LP split ratios for a fixed redundant topology; its reported load
  is reconstructed from those ratios.

Quality values must be read together with the printed `quality=` label. Do not
treat the three `quality_theta` values as identical end-to-end dispatch
measurements. Scale-EPLB may also use fewer than all available slots, while
DeepSeek EPLB and the configured LPLB topology consume the fixed physical
replica budget.

DeepSeek EPLB runs on CPU by design. Scale-EPLB and LPLB are timed with CUDA
events after warm-up; mixed CPU/CUDA paths use synchronized wall time.
