"""Runtime policy for eager/compiled GDKVM execution.

This module owns implementation/runtime choices, not scientific semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


COMPILE_MODES = {
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
}


@dataclass(frozen=True)
class RuntimePolicy:
    profile_id: str = "eager-reference"
    compile_enabled: bool = False
    compile_mode: str = "default"
    compile_fullgraph: bool = False
    compile_dynamic: bool | None = None
    channels_last: bool = False
    cudnn_benchmark: bool = False
    optimize_ddp: bool | None = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "compile_enabled": self.compile_enabled,
            "compile_mode": self.compile_mode,
            "compile_fullgraph": self.compile_fullgraph,
            "compile_dynamic": self.compile_dynamic,
            "channels_last": self.channels_last,
            "cudnn_benchmark": self.cudnn_benchmark,
            "optimize_ddp": self.optimize_ddp,
        }


def _get(mapping: Any, key: str, default: Any) -> Any:
    if mapping is None:
        return default
    getter = getattr(mapping, "get", None)
    if getter is not None:
        return getter(key, default)
    if isinstance(mapping, Mapping):
        return mapping.get(key, default)
    return default


def resolve_runtime_policy(cfg: Any) -> RuntimePolicy:
    runtime = _get(cfg, "runtime", None)
    compile_cfg = _get(runtime, "compile", None)

    mode = str(_get(compile_cfg, "mode", "default"))
    if mode not in COMPILE_MODES:
        raise ValueError(
            f"unsupported torch.compile mode {mode!r}; expected one of {sorted(COMPILE_MODES)}"
        )

    dynamic = _get(compile_cfg, "dynamic", None)
    if dynamic not in (None, True, False):
        raise ValueError("runtime.compile.dynamic must be true, false, or null")

    optimize_ddp = _get(compile_cfg, "optimize_ddp", False)
    if optimize_ddp not in (None, True, False):
        raise ValueError("runtime.compile.optimize_ddp must be true, false, or null")

    return RuntimePolicy(
        profile_id=str(_get(runtime, "profile_id", "eager-reference")),
        compile_enabled=bool(_get(compile_cfg, "enabled", False)),
        compile_mode=mode,
        compile_fullgraph=bool(_get(compile_cfg, "fullgraph", False)),
        compile_dynamic=dynamic,
        channels_last=bool(_get(runtime, "channels_last", False)),
        cudnn_benchmark=bool(_get(runtime, "cudnn_benchmark", False)),
        optimize_ddp=optimize_ddp,
    )


def prepare_model_runtime(model: Any, torch_module: Any, policy: RuntimePolicy) -> Any:
    """Apply declared runtime policy before DDP wrapping.

    Eager execution is the reference path. Compilation is opt-in.
    """
    if policy.channels_last:
        model = model.to(memory_format=torch_module.channels_last)

    if hasattr(torch_module.backends, "cudnn"):
        torch_module.backends.cudnn.benchmark = policy.cudnn_benchmark

    if not policy.compile_enabled:
        return model

    compile_fn = getattr(torch_module, "compile", None)
    if compile_fn is None:
        raise RuntimeError(
            f"runtime profile {policy.profile_id!r} requires torch.compile, "
            "but this PyTorch build does not provide it"
        )

    if policy.optimize_ddp is not None:
        dynamo = getattr(torch_module, "_dynamo", None)
        config = getattr(dynamo, "config", None)
        if config is not None and hasattr(config, "optimize_ddp"):
            config.optimize_ddp = policy.optimize_ddp

    return compile_fn(
        model,
        mode=policy.compile_mode,
        fullgraph=policy.compile_fullgraph,
        dynamic=policy.compile_dynamic,
    )
