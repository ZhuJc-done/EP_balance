"""Capture Megatron-Core MoE routing into ``Lambda``, solve, and (Phase B) log/verify a plan."""

from __future__ import annotations

import torch

from ..config import EPLBConfig
from ..loads import Loads
from ..metrics import compute_metrics
from ..problem import ProblemSpec
from ..topology import Topology
from .rebalancer import EPLBRebalancer


def lambda_row_from_routing_map(routing_map: torch.Tensor) -> torch.Tensor:
    """Megatron ``routing_map`` ``[num_tokens, num_experts]`` (bool/int) -> int64 ``[E]`` counts."""
    return routing_map.to(torch.int64).sum(dim=0)


def lambda_row_from_topk_indices(topk_indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Top-k expert ids ``[num_tokens, k]`` (or flat) -> int64 ``[E]`` counts."""
    return torch.bincount(
        topk_indices.to(torch.int64).flatten(), minlength=num_experts
    ).to(torch.int64)


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
    """Per-layer hook: capture ``Lambda``, solve, log; ``mode='apply'`` also drives a backend."""

    def __init__(
        self,
        rebalancer: EPLBRebalancer,
        mode: str = "observe",
        ep_group=None,
        logger=None,
    ) -> None:
        if mode not in ("observe", "apply"):
            raise ValueError("mode must be 'observe' or 'apply'")
        self.reb = rebalancer
        self.mode = mode
        self.ep_group = ep_group
        self.logger = logger
        self.last_plan = None

    def step(self, local_counts: torch.Tensor, layer_id: int, micro_batch_id: int):
        """Run one rebalance for ``(layer, mb)`` from this rank's expert counts.

        Args:
            local_counts: int64 ``[E]`` this EP rank's per-expert token counts.
            layer_id: MoE layer id (e.g. ``self.layer_number``).
            micro_batch_id: Micro-batch id (the backward "virtual layer" key).

        Returns:
            The solved :class:`~eplb.plan.Plan`.
        """
        res = self.reb.rebalance(local_counts, layer_id, micro_batch_id, group=self.ep_group)
        self.last_plan = res.plan
        if self.logger is not None:
            self._log(res.plan, layer_id, micro_batch_id)
        return res.plan

    def backward(self, layer_id: int, micro_batch_id: int):
        """Re-derive the forward plan and aggregate replica gradients (delegates to rebalancer)."""
        return self.reb.backward(layer_id, micro_batch_id)

    def _log(self, plan, layer_id: int, micro_batch_id: int) -> None:
        lam = self.reb._lambda_ring[(int(layer_id), int(micro_batch_id))]
        m = compute_metrics(plan, Loads(lam), self.reb.topo, self.reb.spec, self.reb.cfg)
        self.logger(
            f"[EPLB] layer={layer_id} mb={micro_batch_id} "
            f"tau={m.tau} imbalance={m.imbalance:.3f} "
            f"replicas={m.total_replicas} phi_token={m.phi_token}"
        )


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
):
    """Register forward hooks on every MoE router to capture ``Lambda`` (observe mode, no source edits).

    Args:
        model: The Megatron model (``nn.Module``) after construction.
        hook: A :class:`MegatronEPLBHook` in ``observe`` mode.
        num_experts: Total routed experts ``E``.
        micro_batch_id_fn: Optional ``() -> int`` returning the current micro-batch id.
        router_class_name: Class name of the router module to match (default ``TopKRouter``).

    Returns:
        List of hook handles; call ``.remove()`` on each to detach.
    """
    handles = []
    state = {"layer": 0}

    def make_cb(layer_id: int):
        def cb(_module, _inputs, output):
            rmap = _extract_routing_map(output, num_experts)
            counts = lambda_row_from_routing_map(rmap)
            mb = micro_batch_id_fn() if micro_batch_id_fn is not None else 0
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
    )
    handles = attach_router_observers(model, hook, num_experts, micro_batch_id_fn, router_class_name)
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
        torch.tensor([int(plan.tau)], dtype=torch.int64, device=plan.x.device),
        plan.x.reshape(-1).to(torch.int64),
        plan.q.reshape(-1).to(torch.int64),
    ])
    gathered = [torch.empty_like(flat) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, flat, group=group)
    return all(torch.equal(g, flat) for g in gathered)
