# Real-MoE expert-hotspot experiment

This workflow measures prompt-conditioned routing from a frozen, trained MoE
checkpoint. It does not use synthetic router bias and does not apply EPLB to
Megatron's dispatcher.

No dataset download, checkpoint conversion, or GPU run is performed by these
scripts until the corresponding command is invoked.

## Directory layout

- `../scripts/prepare_open_workload.py`: download, filter, and index each corpus.
- `run_expert_hotspot.sh`: run frozen-checkpoint routing capture.
- `plot_hotspots_with_max_mean.py`: two snapshot hotspots above the layer line plot.
- `plot_expert_max_mean_by_layer.py`: primary max/mean-versus-layer line plot.
- `plot_expert_max_mean.py`: layer-by-micro-batch max/mean heatmap.
- `plot_expert_hotspots.py`: expert-ID hotspot heatmaps and placement metrics.
- `plot_gin_replica_transport.py`: schematic of the GIN replica transport (needs no trace).
- `PAPER_FIGURE_TEXT.md`: paper-ready caption and body text in English and Chinese.
- `calibrate_router_skew.py`, `extract_routing_imbalance.py`: the synthetic-skew
  workflow described at the end of this file, which is *not* part of the frozen
  checkpoint capture above.

The capture wrapper delegates model execution to the shared
`scripts/run_real_moe.sh`; that launcher remains under `scripts/` because it is
also used by non-evaluation EPLB workflows.

## 1. Prepare one indexed corpus per domain

Install the optional data dependency:

```bash
pip install datasets transformers matplotlib
```

The primary panel keeps the already prepared DAPO-Math corpus and replaces the
other UltraEP-style workloads with FineWeb, Dolma, FineWeb2 Chinese,
StarCoderData, and peS2o. The existing DAPO manifest contains 1,524,504 Qwen
tokens, so use that as the cap for every replacement corpus:

```bash
cd /home/tiger/EP_balance

TOKEN_BUDGET=1524504
for WORKLOAD in fineweb dolma fineweb2_zh pes2o; do
  python scripts/prepare_open_workload.py \
    --workload "${WORKLOAD}" \
    --tokenizer-model Qwen/Qwen3-30B-A3B \
    --output-dir /home/tiger/eplb_data \
    --max-samples 0 \
    --token-budget "${TOKEN_BUDGET}" \
    --max-document-tokens 4096 \
    --truncate-long-documents \
    --shuffle-buffer 2000 \
    --preprocess \
    --megatron-dir /home/tiger/Megatron-LM
done
```

StarCoderData is gated. Accept its Hugging Face terms and authenticate once,
then build a code workload whose token budget is split equally across Python,
C++, Java, and Rust:

```bash
hf auth login

python scripts/prepare_open_workload.py \
  --workload starcoder \
  --tokenizer-model Qwen/Qwen3-30B-A3B \
  --output-dir /home/tiger/eplb_data \
  --max-samples 0 \
  --token-budget 1524504 \
  --max-document-tokens 4096 \
  --truncate-long-documents \
  --shuffle-buffer 2000 \
  --preprocess \
  --megatron-dir /home/tiger/Megatron-LM
```

The independent primary workload names are `dapo_math`, `fineweb`, `dolma`,
`fineweb2_zh`, `starcoder`, and `pes2o`. The older `openscience`, `codeforces`,
`swe_bench`, `gpqa`, `longbench`, and `wikitext` adapters remain available for
external comparison. Dolma and peS2o are streamed from their official JSON
files because current `datasets` releases no longer execute their legacy
loading scripts. By default Dolma reads one official sample shard; increase
`--source-file-limit` only for a larger token budget.

Documents are deduplicated by default, and `--max-source-rows 100000` prevents
an unexpectedly expanded upstream split from being scanned without bound. The
current DAPO Hub split reports 1,791,700 rows despite the dataset name; keep the
existing fixed corpus rather than treating all rows as unique problems.

Each run writes:

- `<workload>.jsonl`: prompt-only `{"text": ...}` documents.
- `<workload>.manifest.json`: source, seed, accepted count, token budget, and
  length statistics.
- `<workload>_text_document.{bin,idx}`: Megatron indexed data.

Megatron's GPT dataset builder packs these documents into fixed `SEQ_LEN`
training/evaluation sequences. Running one domain per job prevents cross-domain
mixing, but this is a training-style forward workload rather than a Poisson
request-level serving replay.

## 2. Convert the trained Qwen checkpoint

Use the official Megatron Bridge converter; the pinned community Megatron
converter has no Qwen3 loader.

```bash
python examples/conversion/convert_checkpoints.py import \
  --hf-model Qwen/Qwen3-30B-A3B \
  --megatron-path /home/tiger/checkpoints/qwen3_30b_a3b_mcore
```

## 3. Capture frozen-model routing

The wrapper enforces `EPLB_MODE=observe`, evaluation-only execution, no synthetic
router skew, and PP=1. Observe mode computes a diagnostic plan but leaves the
actual Megatron dispatch unchanged.

Example on one 4×GB200 node:

```bash
cd /home/tiger/EP_balance

WORKLOAD=dapo_math \
MEGATRON_DIR=/home/tiger/Megatron-LM \
CHECKPOINT=/home/tiger/checkpoints/qwen3_30b_a3b_mcore \
TOKENIZER_MODEL=Qwen/Qwen3-30B-A3B \
DATA_PATH=/home/tiger/eplb_data/dapo_math_text_document \
NNODES=1 GPUS_PER_NODE=4 TP=1 PP=1 EP=4 \
SEQ_LEN=4096 EVAL_ITERS=50 \
TRACE_OUT=/home/tiger/EP_balance/logs/hotspot_dapo_math.pt \
bash eval/run_expert_hotspot.sh
```

Repeat with the other `DATA_PATH`, `WORKLOAD`, and `TRACE_OUT` values.

Trace v3 stores one `Ω[source_ep_rank, logical_expert]` matrix per layer
occurrence. Router counts are summed over TP/SP and CP before the EP all-gather.
For PP>1, each pipeline stage would need its own writer, so the review version
intentionally rejects PP>1.

## 4. Plot snapshot hotspots with the layer line plot

```bash
python eval/plot_hotspots_with_max_mean.py \
  --trace DAPO-Math=eval/data/hotspot_dapo_math.pt \
  --trace StarCoderData=eval/data/hotspot_starcoder.pt \
  --occurrence 0 \
  --output figs/dapo_vs_starcoder_hotspots_with_max_mean.pdf
```

The two snapshot heatmaps are placed side by side on the top row. The
max/mean-versus-layer comparison spans the row below them. The output suffix
selects PDF, PNG, or SVG; PDF uses embedded TrueType fonts and a 300 DPI
rasterization for the heatmap cells.

## 5. Plot expert max/mean versus layer

The solid line is the mean of the per-micro-batch expert max/mean ratios at
each layer. The shaded band is the P10--P90 range across micro-batches, so the
plot exposes both persistent layer hotspots and input-dependent variation.

```bash
python eval/plot_expert_max_mean_by_layer.py \
  --trace DAPO-Math=eval/data/hotspot_dapo_math.pt \
  --trace StarCoderData=eval/data/hotspot_starcoder.pt \
  --output figs/dapo_vs_starcoder_max_mean_by_layer.png \
  --title "Qwen3-30B-A3B routing imbalance by layer"
```

The CSV contains the mean, median, percentile band, min/max across
micro-batches, and a separate whole-trace aggregate ratio for every layer.
Pass `--show-aggregate` to add the aggregate ratio as a dashed line.

## 6. Plot comparable heatmaps and raw imbalance

```bash
python eval/plot_expert_hotspots.py \
  --trace Web=logs/hotspot_fineweb.pt \
  --trace MultiDomain=logs/hotspot_dolma.pt \
  --trace Chinese=logs/hotspot_fineweb2_zh.pt \
  --trace Code=logs/hotspot_starcoder.pt \
  --trace Math=logs/hotspot_dapo_math.pt \
  --trace Science=logs/hotspot_pes2o.pt \
  --view both \
  --occurrence 0 \
  --normalization share \
  --output figs/real_expert_hotspots.png \
  --title "Qwen3-30B-A3B expert routing"
```

Outputs:

- PNG: rows are workloads; columns are one microbatch occurrence and the
  whole-trace aggregate.
- CSV: per-layer expert max/mean, original-placement rank max/mean, hottest
  expert, hottest rank, and assignment totals.

`omega.sum(0)` is the logical-expert load. Original no-balancing receiving-rank
load is computed by accumulating those expert counts through `meta.main_rank`;
`omega.sum(1)` is source traffic and is not the receiving-rank imbalance.

The aggregate column is a descriptive aggregate of captured requests. It is not
equivalent to training with Megatron's `global_aux_loss`. A true
microbatch-loss versus global-batch-loss comparison requires two checkpoints
trained from the same initialization with `aux_loss` and `global_aux_loss`,
followed by this same frozen evaluation.

## Related: synthetic router skew, without a trained checkpoint

Separate from the frozen-checkpoint capture above. `ROUTER_SKEW=<std>` makes
`run_real_moe.sh` pass Megatron's `--moe-router-force-biased`, which *replaces* the
router logits with noise plus a globally shared per-expert bias. That manufactures a
controllable imbalance in a few hundred steps instead of waiting for experts to
specialise, at the cost of a meaningless loss: use it for step time and imbalance only.

Pick the magnitude before spending GPU time. `calibrate_router_skew.py` reproduces that
sampler offline and reports the imbalance each `std` yields, including the spread across
bias draws (a run gets exactly one draw, so the same `ROUTER_SKEW` at a different `--seed`
lands elsewhere):

```bash
python eval/calibrate_router_skew.py --target-rank-imbalance 1.5
```

Rank-level skew is much weaker than expert-level skew because the bias is drawn per
expert while each rank owns `E / R` of them. `--bias-mode block` bounds what a co-located
hotspot pattern would give; Megatron cannot express it today.

Then measure what actually happened. The `[EPLB] ... imbalance=` line an observe run
prints is the residual *after* replica placement, so the raw skew needs `EPLB_TRACE_OUT`:

```bash
EPLB_MODE=observe ROUTER_SKEW=-0.5 EPLB_TRACE_OUT=traces/skew05.pt ... \
  bash scripts/run_real_moe.sh ...

python eval/extract_routing_imbalance.py \
  --trace traces/skew05.pt \
  --log logs/skew05_observe_node0.log \
  --out-dir exp/skew05
```

That reports raw expert and rank max/mean per layer, joins the residual on `(layer, mb)`,
and derives the share of the excess the solver absorbed. Write the trace to local disk:
`flush_trace` rewrites the whole file every `EPLB_TRACE_EVERY` samples, which the HDFS
FUSE mount handles poorly.
