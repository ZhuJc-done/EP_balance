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

        try:
            # Wrap the leaf operations rather than reimplementing MoELayer.forward. In observe mode,
            # the router's forward hook runs after this router region closes, so solver work is not
            # accidentally charged to the router.
            self._wrap_required(moe_layer.router, "forward", "native/route")
            self._wrap_required(moe_layer, "dispatch", "native/dispatch")
            self._wrap_required(moe_layer.experts, "forward", "native/expert_gemm")
            self._wrap_required(moe_layer, "combine", "native/combine")

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

    def _wrap_outer_forward(self) -> None:
        original = self.moe_layer.forward

        @functools.wraps(original)
        def timed_forward(_layer, *args, **kwargs):
            mb = self.micro_batch_id
            self.micro_batch_id += 1
            profiling.begin_debug_window()
            output = original(*args, **kwargs)

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
