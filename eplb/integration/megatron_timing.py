"""Non-invasive timing bindings for Megatron's native MoE execution path."""

from __future__ import annotations

import functools
import types
from typing import Callable, List, Optional

import torch

from . import profiling


def _device_hint(args, kwargs):
    """Return the first tensor device found in a method call."""
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor):
            return value.device
        if isinstance(value, (tuple, list)):
            for item in value:
                if isinstance(item, torch.Tensor):
                    return item.device
    return None


def _finish_native_backward(state: dict) -> None:
    """Close the interval at the native token dispatcher's inverse communication."""
    _finish_native_phase(state, "native/dispatch_bwd")
    start = state.get("start")
    if start is None:
        return
    profiling.finish_debug_interval("native/moe_bwd_total", start)
    state["start"] = None
    profiling.emit_backward_debug(
        state.get("context", ""), mode=state.get("mode", "off")
    )


def _start_native_phase(state: dict, phase: str, device) -> None:
    starts = state["phase_starts"]
    if phase not in starts:
        starts[phase] = profiling.start_debug_interval(device=device)


def _finish_native_phase(state: dict, phase: str) -> None:
    start = state["phase_starts"].pop(phase, None)
    if start is not None:
        profiling.finish_debug_interval(phase, start)


class _NativePhaseBackwardStart(torch.autograd.Function):
    """Start one native backward subphase at its forward output boundary."""

    @staticmethod
    def forward(ctx, output: torch.Tensor, state: dict, phase: str):
        ctx.state = state
        ctx.phase = phase
        return output.view_as(output)

    @staticmethod
    def backward(ctx, grad_output):
        _start_native_phase(ctx.state, ctx.phase, grad_output.device)
        return grad_output, None, None


class _NativePhaseBackwardStartPair(torch.autograd.Function):
    """Start a subphase after both token-dispatch output gradients arrive."""

    @staticmethod
    def forward(ctx, first: torch.Tensor, second: torch.Tensor, state: dict, phase: str):
        ctx.state = state
        ctx.phase = phase
        return first.view_as(first), second.view_as(second)

    @staticmethod
    def backward(ctx, grad_first, grad_second):
        device = grad_first.device if grad_first is not None else grad_second.device
        _start_native_phase(ctx.state, ctx.phase, device)
        return grad_first, grad_second, None, None


class _NativePhaseBackwardEnd(torch.autograd.Function):
    """Finish one native backward subphase at its forward input boundary."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor, state: dict, phase: str):
        ctx.state = state
        ctx.phase = phase
        return input_.view_as(input_)

    @staticmethod
    def backward(ctx, grad_input):
        _finish_native_phase(ctx.state, ctx.phase)
        return grad_input, None, None


class _NativeMoEBackwardStart(torch.autograd.Function):
    """Start timing when the final native MoE output receives its gradient."""

    @staticmethod
    def forward(ctx, output: torch.Tensor, state: dict) -> torch.Tensor:
        ctx.state = state
        return output.view_as(output)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        ctx.state["start"] = profiling.start_debug_interval(device=grad_output.device)
        return grad_output, None


class _NativeMoEBackwardEnd(torch.autograd.Function):
    """End timing after token and probability reverse-dispatch have completed."""

    @staticmethod
    def forward(
        ctx, hidden_states: torch.Tensor, probs: torch.Tensor, state: dict
    ):
        ctx.state = state
        return hidden_states.view_as(hidden_states), probs.view_as(probs)

    @staticmethod
    def backward(ctx, grad_hidden_states, grad_probs):
        _finish_native_backward(ctx.state)
        return grad_hidden_states, grad_probs, None


class _NativeMoEBackwardEndHidden(torch.autograd.Function):
    """Hidden-only endpoint for token dispatchers that do not consume probabilities."""

    @staticmethod
    def forward(ctx, hidden_states: torch.Tensor, state: dict):
        ctx.state = state
        return hidden_states.view_as(hidden_states)

    @staticmethod
    def backward(ctx, grad_hidden_states):
        _finish_native_backward(ctx.state)
        return grad_hidden_states, None


def _mark_primary_output(output, state: dict):
    """Place the backward-start marker on a MoELayer's primary tensor output."""
    if isinstance(output, torch.Tensor):
        return _NativeMoEBackwardStart.apply(output, state)
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return (_NativeMoEBackwardStart.apply(output[0], state), *output[1:])
    if isinstance(output, list) and output and isinstance(output[0], torch.Tensor):
        return [_NativeMoEBackwardStart.apply(output[0], state), *output[1:]]
    return output


def _mark_phase_output(output, state: dict, phase: str):
    if isinstance(output, torch.Tensor):
        return _NativePhaseBackwardStart.apply(output, state, phase)
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return (
            _NativePhaseBackwardStart.apply(output[0], state, phase),
            *output[1:],
        )
    return output


def _mark_dispatch_outputs(output, state: dict):
    if (
        isinstance(output, tuple)
        and len(output) >= 2
        and isinstance(output[0], torch.Tensor)
        and isinstance(output[1], torch.Tensor)
    ):
        first, second = _NativePhaseBackwardStartPair.apply(
            output[0], output[1], state, "native/dispatch_bwd"
        )
        return (first, second, *output[2:])
    return _mark_phase_output(output, state, "native/dispatch_bwd")


class NativeMoETimingBinding:
    """Time one native Megatron ``MoELayer`` without replacing its execution logic."""

    def __init__(
        self,
        moe_layer,
        *,
        layer_id: int,
        mode: str,
        logger: Optional[Callable[[str], None]],
    ) -> None:
        if mode not in ("off", "observe"):
            raise ValueError("native MoE timing mode must be 'off' or 'observe'")
        if getattr(moe_layer, "_eplb_native_timing", None) is not None:
            raise RuntimeError("native MoE timing is already bound to this layer")

        self.moe_layer = moe_layer
        self.layer_id = int(layer_id)
        self.mode = mode
        self.logger = logger
        self.micro_batch_id = 0
        self._restore = []
        self._active_backward_state = None

        try:
            # Wrap the leaf operations rather than reimplementing MoELayer.forward. In observe mode,
            # the router's forward hook runs after this router region closes, so solver work is not
            # accidentally charged to the router.
            self._wrap_required(moe_layer.router, "forward", "native/route")
            self._wrap_required(moe_layer, "dispatch", "native/dispatch")
            # Megatron's local tensor-parallel linear computes Dgrad and Wgrad
            # inside one autograd Function. Keep the native source untouched and
            # time that complete experts subgraph as one `expert_bwd` region.
            self._wrap_backward_required(
                moe_layer.experts,
                "forward",
                "native/expert_gemm",
                "native/expert_bwd",
            )
            self._wrap_backward_required(
                moe_layer,
                "combine",
                "native/combine",
                "native/combine_bwd",
            )
            self._wrap_token_dispatch_boundary()

            shared_experts = getattr(moe_layer, "shared_experts", None)
            if shared_experts is not None and callable(getattr(shared_experts, "forward", None)):
                self._wrap_required(shared_experts, "forward", "native/shared_expert")

            self._wrap_outer_forward()
            moe_layer._eplb_native_timing = self
        except Exception:
            self.remove()
            raise

    def _replace(self, owner, method_name: str, replacement) -> None:
        namespace = vars(owner)
        self._restore.append(
            (owner, method_name, method_name in namespace, namespace.get(method_name))
        )
        setattr(owner, method_name, replacement)

    def _wrap_required(self, owner, method_name: str, region: str) -> None:
        original = getattr(owner, method_name, None)
        if not callable(original):
            owner_name = type(owner).__name__ if owner is not None else "None"
            raise RuntimeError(
                f"native MoE timing requires callable {owner_name}.{method_name}; "
                "the installed Megatron MoE API is incompatible"
            )

        @functools.wraps(original)
        def timed(_owner, *args, **kwargs):
            with profiling.record(
                region,
                time_it=True,
                device=_device_hint(args, kwargs),
            ):
                return original(*args, **kwargs)

        self._replace(owner, method_name, types.MethodType(timed, owner))

    def _wrap_backward_required(
        self, owner, method_name: str, forward_region: str, backward_region: str
    ) -> None:
        """Time a method's forward and its complete autograd backward subgraph."""
        original = getattr(owner, method_name, None)
        if not callable(original):
            owner_name = type(owner).__name__ if owner is not None else "None"
            raise RuntimeError(
                f"native MoE timing requires callable {owner_name}.{method_name}; "
                "the installed Megatron MoE API is incompatible"
            )

        @functools.wraps(original)
        def timed(_owner, *args, **kwargs):
            state = self._active_backward_state
            positional = list(args)
            if state is not None:
                if positional and isinstance(positional[0], torch.Tensor):
                    positional[0] = _NativePhaseBackwardEnd.apply(
                        positional[0], state, backward_region
                    )
                elif isinstance(kwargs.get("hidden_states"), torch.Tensor):
                    kwargs["hidden_states"] = _NativePhaseBackwardEnd.apply(
                        kwargs["hidden_states"], state, backward_region
                    )
            with profiling.record(
                forward_region,
                time_it=True,
                device=_device_hint(tuple(positional), kwargs),
            ):
                output = original(*tuple(positional), **kwargs)
            if state is not None:
                output = _mark_phase_output(output, state, backward_region)
            return output

        self._replace(owner, method_name, types.MethodType(timed, owner))

    def _wrap_token_dispatch_boundary(self) -> None:
        """Mark the graph immediately before native dispatch communication.

        The marker is inside ``MoELayer.dispatch``'s optional delayed-Wgrad
        wrapper. Its backward therefore fires directly after the dispatch
        collective's inverse operation, before delayed Wgrad and attention
        backward.
        """
        dispatcher = getattr(self.moe_layer, "token_dispatcher", None)
        original = getattr(dispatcher, "token_dispatch", None)
        if not callable(original):
            name = type(dispatcher).__name__ if dispatcher is not None else "None"
            raise RuntimeError(
                f"native MoE timing requires callable {name}.token_dispatch"
            )

        @functools.wraps(original)
        def marked(_dispatcher, *args, **kwargs):
            state = self._active_backward_state
            if state is not None and not state["dispatch_marked"]:
                positional = list(args)
                hidden_states = positional[0] if positional else kwargs.get("hidden_states")
                probs = positional[1] if len(positional) > 1 else kwargs.get("probs")
                if isinstance(hidden_states, torch.Tensor):
                    if isinstance(probs, torch.Tensor):
                        hidden_states, probs = _NativeMoEBackwardEnd.apply(
                            hidden_states, probs, state
                        )
                    else:
                        hidden_states = _NativeMoEBackwardEndHidden.apply(
                            hidden_states, state
                        )
                    if positional:
                        positional[0] = hidden_states
                        if len(positional) > 1:
                            positional[1] = probs
                        elif "probs" in kwargs:
                            kwargs["probs"] = probs
                    else:
                        kwargs["hidden_states"] = hidden_states
                        if "probs" in kwargs:
                            kwargs["probs"] = probs
                    args = tuple(positional)
                    state["dispatch_marked"] = True
            output = original(*args, **kwargs)
            if state is not None and state["dispatch_marked"]:
                output = _mark_dispatch_outputs(output, state)
            return output

        self._replace(
            dispatcher,
            "token_dispatch",
            types.MethodType(marked, dispatcher),
        )

    def _wrap_outer_forward(self) -> None:
        original = self.moe_layer.forward

        @functools.wraps(original)
        def timed_forward(_layer, *args, **kwargs):
            mb = self.micro_batch_id
            self.micro_batch_id += 1
            profiling.begin_debug_window()
            fwd_start = profiling.start_debug_interval(
                device=_device_hint(args, kwargs)
            )
            state = None
            if profiling.debug_enabled():
                state = {
                    "context": f"layer={self.layer_id} mb={mb}",
                    "mode": self.mode,
                    "start": None,
                    "dispatch_marked": False,
                    "phase_starts": {},
                }
            previous_state = self._active_backward_state
            self._active_backward_state = state
            try:
                output = original(*args, **kwargs)
            finally:
                self._active_backward_state = previous_state
            profiling.finish_debug_interval("native/moe_fwd_total", fwd_start)
            if state is not None and state["dispatch_marked"]:
                output = _mark_primary_output(output, state)

            missing = {"expert_transfer": "n/a"}
            if self.mode == "off":
                missing.update({"solver": "n/a", "omega_gather": "n/a"})
            profiling.maybe_summary(
                self.logger,
                context=f"mode={self.mode} layer={self.layer_id} mb={mb}",
                missing=missing,
            )
            return output

        self._replace(
            self.moe_layer,
            "forward",
            types.MethodType(timed_forward, self.moe_layer),
        )

    def remove(self) -> None:
        """Restore all methods changed by this binding."""
        for owner, method_name, had_instance_value, old_value in reversed(self._restore):
            if had_instance_value:
                setattr(owner, method_name, old_value)
            elif method_name in vars(owner):
                delattr(owner, method_name)
        self._restore.clear()
        if getattr(self.moe_layer, "_eplb_native_timing", None) is self:
            delattr(self.moe_layer, "_eplb_native_timing")


def bind_native_moe_timing(
    model,
    *,
    mode: str,
    logger: Optional[Callable[[str], None]] = print,
) -> List[NativeMoETimingBinding]:
    """Bind native-stage timing to every Megatron ``MoELayer`` in ``model``."""
    layers = [module for module in model.modules() if type(module).__name__ == "MoELayer"]
    return [
        NativeMoETimingBinding(layer, layer_id=layer_id, mode=mode, logger=logger)
        for layer_id, layer in enumerate(layers)
    ]
