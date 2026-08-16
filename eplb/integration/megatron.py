"""Capture Megatron-Core MoE routing into ``Ω`` and solve an EPLB plan."""

from __future__ import annotations

import torch

from . import profiling
from ..config import EPLBConfig
from ..loads import Loads
from ..metrics import compute_metrics
from ..problem import ProblemSpec
from ..topology import Topology
from .rebalancer import EPLBRebalancer


def omega_row_from_routing_map(routing_map: torch.Tensor) -> torch.Tensor:
    """Megatron ``routing_map`` ``[num_tokens, num_experts]`` (bool/int) -> int64 ``[E]`` counts."""
    return routing_map.to(torch.int64).sum(dim=0)


def omega_row_from_topk_indices(topk_indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Top-k expert ids ``[num_tokens, k]`` (or flat) -> int64 ``[E]`` counts."""
    return torch.bincount(
        topk_indices.to(torch.int64).flatten(), minlength=num_experts
    ).to(torch.int64)


def reduce_router_counts_tp_cp(local_counts: torch.Tensor, router) -> torch.Tensor:
    """Sum token-sharded router counts over Megatron's TP×CP group.

    EP ranks own different source-token shards and must remain separate rows in
    ``Ω[R,E]``. TP/SP and CP, however, split one source rank's tokens, so
    those dimensions must be reduced before the EP all-gather.
    """
    group = getattr(router, "tp_cp_group", None)
    dist = torch.distributed
    if (
        group is None
        or not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size(group) <= 1
    ):
        return local_counts
    counts = local_counts.contiguous().clone()
    dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)
    return counts


def build_spec_for_megatron(
    num_experts: int,
    ep_size: int,
    weight_bytes_each: int,
    s_tok: int,
    n_slot: int,
    device: torch.device | str = "cpu",
) -> ProblemSpec:
    """Build a :class:`ProblemSpec` for Megatron's contiguous split (expert ``e`` -> rank ``e//(E/ep)``).

    Args:
        num_experts: Total routed experts ``E`` (must be divisible by ``ep_size``).
        ep_size: Expert-parallel world size (= number of ranks the solver sees).
        weight_bytes_each: Bytes of one expert's parameters ``|W_e|``.
        s_tok: Bytes of one token's activation (hidden_dim * dtype_size).
        n_slot: Per-rank instance slot budget ``N_slot``.
        device: Tensor device.

    Returns:
        A validated :class:`~eplb.problem.ProblemSpec`.
    """
    if num_experts % ep_size != 0:
        raise ValueError("num_experts must be divisible by ep_size")
    num_local = num_experts // ep_size
    main_rank = torch.arange(num_experts, device=device, dtype=torch.int64) // num_local
    weight_bytes = torch.full((num_experts,), int(weight_bytes_each), dtype=torch.int64, device=device)
    spec = ProblemSpec(num_experts, main_rank, weight_bytes, s_tok, n_slot)
    spec.validate(ep_size)
    return spec


class MegatronEPLBHook:
    """Per-layer hook: capture ``Ω`` and solve/log a plan."""

    def __init__(
        self,
        rebalancer: EPLBRebalancer,
        mode: str = "observe",
        ep_group=None,
        logger=None,
        *,
        trace_out: str | None = None,
        trace_max: int = 0,
        trace_every: int = 25,
        counts_reduced_over_tp_cp: bool = False,
        owns_timing_window: bool = True,
    ) -> None:
        if mode not in ("observe", "apply"):
            raise ValueError("mode must be 'observe' or 'apply'")
        self.reb = rebalancer
        self.mode = mode
        self.ep_group = ep_group
        self.logger = logger
        self.last_plan = None

        # Optional: persist the real gathered Ω[R,E] per (layer, mb) so the
        # baseline harness can replay this exact routing through every strategy.
        self.trace_out = trace_out
        self.trace_max = int(trace_max)
        self.trace_every = max(1, int(trace_every))
        self.counts_reduced_over_tp_cp = bool(counts_reduced_over_tp_cp)
        self.owns_timing_window = bool(owns_timing_window)
        self._trace_samples: list[dict] = []
        self._trace_dirty = 0
        self._trace_ordinal = 0
        if self.trace_out is not None:
            self._trace_meta = self._build_trace_meta()
            import atexit

            atexit.register(self.flush_trace)

    def step(self, local_counts: torch.Tensor, layer_id: int, micro_batch_id: int):
        """Run one rebalance for ``(layer, mb)`` from this rank's expert counts.

        Args:
            local_counts: int64 ``[E]`` this EP rank's per-expert token counts.
            layer_id: MoE layer id (e.g. ``self.layer_number``).
            micro_batch_id: Micro-batch id (the backward "virtual layer" key).

        Returns:
            The solved :class:`~eplb.plan.Plan`.
        """
        if self.owns_timing_window:
            profiling.begin_debug_window()
        res = self.reb.rebalance(local_counts, layer_id, micro_batch_id, group=self.ep_group)
        self.last_plan = res.plan
        if self.logger is not None:
            self._log(res.plan, layer_id, micro_batch_id)
        self._maybe_dump(layer_id, micro_batch_id)
        return res.plan

    def backward(self, layer_id: int, micro_batch_id: int):
        """Re-derive the forward plan and aggregate replica gradients (delegates to rebalancer)."""
        return self.reb.backward(layer_id, micro_batch_id)

    def _log(self, plan, layer_id: int, micro_batch_id: int) -> None:
        omega = self.reb._omega_ring[(int(layer_id), int(micro_batch_id))]
        m = compute_metrics(
            plan, Loads(omega), self.reb.topo, self.reb.spec, self.reb.cfg
        )
        extra = f" solve_ms={profiling.last_ms('solve'):.3f}" if profiling.enabled() else ""
        self.logger(
            f"[EPLB] layer={layer_id} mb={micro_batch_id} "
            f"theta={m.theta} imbalance={m.imbalance:.3f} "
            f"replicas={m.total_replicas} phi_token={m.phi_token}{extra}"
        )
        if self.owns_timing_window:
            profiling.maybe_summary(
                self.logger, context=f"mode=observe layer={layer_id} mb={micro_batch_id}"
            )

    # -- trace capture (Phase B -> baseline replay) -----------------------------
    def _build_trace_meta(self) -> dict:
        """Self-describing header so the replay rebuilds an identical Topology/ProblemSpec."""
        spec, topo = self.reb.spec, self.reb.topo
        return {
            "format_version": 3,
            "num_ranks": topo.num_ranks,
            "num_experts": int(spec.num_experts),
            "omega_semantics": "source_ep_rank_by_logical_expert_token_assignments",
            "counts_reduced_over_tp_cp": self.counts_reduced_over_tp_cp,
            "s_tok": int(spec.s_tok),
            "n_slot": int(spec.n_slot),
            "num_domains": topo.num_domains,
            "main_rank": spec.main_rank.detach().cpu().clone(),
            "weight_bytes": spec.weight_bytes.detach().cpu().clone(),
            "domain_of_rank": topo.domain_of_rank.detach().cpu().clone(),
            "cost": topo.cost.detach().cpu().clone(),
        }

    def _maybe_dump(self, layer_id: int, micro_batch_id: int) -> None:
        """Append the gathered ``Ω[R,E]`` and periodically flush."""
        if self.trace_out is None:
            return
        if self.trace_max and len(self._trace_samples) >= self.trace_max:
            return
        omega = self.reb._omega_ring.get((int(layer_id), int(micro_batch_id)))
        if omega is None:
            return
        self._trace_samples.append(
            {
                "layer": int(layer_id),
                "mb": int(micro_batch_id),
                "ordinal": self._trace_ordinal,
                "omega": omega.detach().to("cpu", torch.int64).clone(),
            }
        )
        self._trace_ordinal += 1
        self._trace_dirty += 1
        if self._trace_dirty >= self.trace_every:
            self.flush_trace()

    def flush_trace(self) -> None:
        """Atomically write the accumulated trace to ``trace_out`` (safe to call repeatedly)."""
        if self.trace_out is None or not self._trace_samples:
            return
        import os
        import tempfile

        out_dir = os.path.dirname(self.trace_out) or "."
        os.makedirs(out_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
        os.close(fd)
        torch.save({"meta": self._trace_meta, "samples": self._trace_samples}, tmp)
        os.replace(tmp, self.trace_out)  # atomic on POSIX
        self._trace_dirty = 0


def _extract_routing_map(output, num_experts: int) -> torch.Tensor:
    """Pull the ``[num_tokens, num_experts]`` routing_map (int/bool preferred) from a router output."""
    candidates = output if isinstance(output, (tuple, list)) else (output,)
    intlike = [
        t for t in candidates
        if isinstance(t, torch.Tensor) and (t.dtype == torch.bool or not t.is_floating_point())
    ]
    # routing_map: int/bool tensor whose last dim spans all experts
    for t in intlike:
        if t.dim() >= 2 and t.shape[-1] == num_experts:
            return t.reshape(-1, num_experts)
    # top-k indices: int tensor whose last dim is k (< E) or flat
    for t in intlike:
        return torch.nn.functional.one_hot(
            t.to(torch.int64).flatten(), num_classes=num_experts
        )
    raise TypeError("could not locate a routing_map / index tensor in router output")


def attach_router_observers(
    model,
    hook: "MegatronEPLBHook",
    num_experts: int,
    micro_batch_id_fn=None,
    router_class_name: str = "TopKRouter",
    reduce_tp_cp: bool = True,
    prefer_initial_forward: bool = False,
):
    """Register forward hooks on every MoE router to capture ``Ω``.

    Args:
        model: The Megatron model (``nn.Module``) after construction.
        hook: A :class:`MegatronEPLBHook` in ``observe`` mode.
        num_experts: Total routed experts ``E``.
        micro_batch_id_fn: Optional ``() -> int`` returning the current micro-batch id.
        router_class_name: Class name of the router module to match (default ``TopKRouter``).
        reduce_tp_cp: Sum router counts over Megatron's TP×CP token-sharding group
            before retaining EP ranks as separate ``Ω`` rows.
        prefer_initial_forward: With reentrant activation checkpointing, observe the initial
            no-grad forward and skip its later recompute. This keeps solver timing inside an
            enclosing native-MoE forward timing window.

    Returns:
        List of hook handles; call ``.remove()`` on each to detach.
    """
    handles = []
    state = {"layer": 0, "calls_by_layer": {}, "pending_recompute_by_layer": {}}

    def make_cb(layer_id: int):
        def cb(_module, _inputs, output):
            # Reentrant activation checkpointing runs the original forward under
            # no_grad and recomputes it with gradients enabled. Standalone observation keeps the
            # recompute, as before. Native whole-layer timing instead keeps the initial forward so
            # its solver sample lands in the same timing window as dispatch/GEMM/combine.
            if getattr(_module, "training", False):
                pending = state["pending_recompute_by_layer"].get(layer_id, 0)
                if not torch.is_grad_enabled():
                    if not prefer_initial_forward:
                        return output
                    state["pending_recompute_by_layer"][layer_id] = pending + 1
                elif prefer_initial_forward and pending:
                    state["pending_recompute_by_layer"][layer_id] = pending - 1
                    return output
            rmap = _extract_routing_map(output, num_experts)
            counts = omega_row_from_routing_map(rmap)
            if reduce_tp_cp:
                counts = reduce_router_counts_tp_cp(counts, _module)
            if micro_batch_id_fn is not None:
                mb = micro_batch_id_fn()
            else:
                # Megatron does not expose a stable microbatch id to module hooks.
                # Per-layer occurrence is deterministic for PP=1 and keeps the nth
                # forward aligned across layers for offline hotspot aggregation.
                mb = state["calls_by_layer"].get(layer_id, 0)
                state["calls_by_layer"][layer_id] = mb + 1
            hook.step(counts, layer_id=layer_id, micro_batch_id=mb)
            return output

        return cb

    for _name, module in model.named_modules():
        if type(module).__name__ == router_class_name:
            handles.append(module.register_forward_hook(make_cb(state["layer"])))
            state["layer"] += 1
    return handles


def setup_eplb_observer(
    model,
    *,
    num_experts: int,
    weight_bytes_each: int,
    s_tok: int,
    n_slot: int,
    gpus_per_node: int | None = None,
    intra_cost: int = 1,
    inter_cost: int = 8,
    cfg: EPLBConfig | None = None,
    logger=print,
    micro_batch_id_fn=None,
    router_class_name: str = "TopKRouter",
    trace_out: str | None = None,
    trace_max: int = 0,
    trace_every: int = 25,
    reduce_tp_cp: bool = True,
    native_timing: bool = False,
):
    """One-call Phase B setup (call once after model build): read Megatron's EP state, build the rebalancer, attach observers.

    Args:
        model: The Megatron model (``nn.Module``) after construction.
        num_experts: Total routed experts ``E``.
        weight_bytes_each: Bytes of one expert's parameters ``|W_e|``.
        s_tok: Bytes of one token's activation (hidden_dim * dtype_size).
        n_slot: Per-rank instance slot budget ``N_slot``.
        gpus_per_node: GPUs per NVLink domain; defaults to the EP world size.
        intra_cost: Per-token NVLink cost (relative).
        inter_cost: Per-token RDMA cost (relative).
        cfg: Solver config (defaults to :class:`EPLBConfig`).
        logger: Callable for per-layer metric lines (e.g. ``print``); ``None`` to silence.
        micro_batch_id_fn: Optional ``() -> int`` for the current micro-batch id.
        router_class_name: Router module class name to match.
        trace_out: If set, persist the gathered ``Ω[R,E]`` per (layer, mb) to this
            path so ``baseline.benchmark --trace`` can replay the real routing through
            every baseline. Pass a non-``None`` value only on the rank that should write.
        trace_max: Cap on the number of captured samples (0 = unbounded).
        trace_every: Flush the trace to disk every this many captured samples.
        reduce_tp_cp: Sum TP/SP and CP token shards before the EP all-gather.
        native_timing: Let an enclosing native-MoE binding own the timing window and report after
            dispatch, expert compute, and combine have also completed.

    Returns:
        ``(hook, handles)`` -- the :class:`MegatronEPLBHook` and its forward-hook handles.
    """
    from megatron.core import parallel_state as mpu  # lazy: only needed on the cluster

    ep_group = mpu.get_expert_model_parallel_group()
    ep_size = mpu.get_expert_model_parallel_world_size()
    device = next(model.parameters()).device

    # one NVLink domain per node; approximate node layout from EP size / gpus_per_node
    gpn = gpus_per_node or ep_size
    if gpn <= 0 or ep_size % gpn != 0:
        gpn = ep_size
    num_nodes = ep_size // gpn
    topo = Topology.from_nvlink_rdma(num_nodes, gpn, intra_cost, inter_cost, device)
    spec = build_spec_for_megatron(num_experts, ep_size, weight_bytes_each, s_tok, n_slot, device)
    hook = MegatronEPLBHook(
        EPLBRebalancer(topo, spec, cfg or EPLBConfig()),
        mode="observe", ep_group=ep_group, logger=logger,
        trace_out=trace_out, trace_max=trace_max, trace_every=trace_every,
        counts_reduced_over_tp_cp=reduce_tp_cp,
        owns_timing_window=not native_timing,
    )
    handles = attach_router_observers(
        model,
        hook,
        num_experts,
        micro_batch_id_fn,
        router_class_name,
        reduce_tp_cp=reduce_tp_cp,
        prefer_initial_forward=native_timing,
    )
    return hook, handles


def assert_plan_replicated(plan, group=None) -> bool:
    """E3 check: confirm every rank in ``group`` holds a bit-identical plan (no-op if not distributed).

    Args:
        plan: A solved :class:`~eplb.plan.Plan`.
        group: Optional process group (defaults to the world group).

    Returns:
        True if all ranks agree (or distributed is not initialized); False otherwise.
    """
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return True
    flat = torch.cat([
        torch.tensor([int(plan.theta)], dtype=torch.int64, device=plan.x.device),
        plan.x.reshape(-1).to(torch.int64),
        plan.q.reshape(-1).to(torch.int64),
    ])
    gathered = [torch.empty_like(flat) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, flat, group=group)
    return all(torch.equal(g, flat) for g in gathered)
