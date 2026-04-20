from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from typing import Any

import torch
from torch.optim import AdamW

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None


_AMP_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def configure_runtime_environment(*, cudnn_benchmark: bool = True) -> None:
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    if torch.cuda.is_available() and torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)


def resolve_num_workers(requested: int) -> int:
    workers = int(requested)
    if workers >= 0:
        return workers
    cpu_count = os.cpu_count() or 0
    if cpu_count <= 2:
        return 0
    return max(min(cpu_count - 2, 8), 1)


def resolve_dataloader_kwargs(
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool | None = None,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> dict[str, Any]:
    resolved_workers = resolve_num_workers(num_workers)
    should_pin = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": True,
        "num_workers": int(resolved_workers),
        "pin_memory": bool(should_pin),
    }
    if resolved_workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = max(int(prefetch_factor), 1)
    return kwargs


def resolve_amp_dtype(*, device: torch.device, requested: str) -> tuple[bool, torch.dtype]:
    dtype = _AMP_DTYPES.get(str(requested).strip().lower())
    if dtype is None:
        raise ValueError(f"Unsupported AMP dtype: {requested}")
    if device.type != "cuda" or dtype == torch.float32:
        return False, torch.float32
    if dtype == torch.bfloat16:
        major, _minor = torch.cuda.get_device_capability(device=device)
        if major < 8:
            return True, torch.float16
    return True, dtype


def autocast_context(*, device: torch.device, enabled: bool, amp_dtype: torch.dtype) -> Any:
    if not bool(enabled) or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def build_grad_scaler(*, enabled: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=bool(enabled and torch.cuda.is_available()))
    return torch.cuda.amp.GradScaler(enabled=bool(enabled and torch.cuda.is_available()))


def _resolve_bnb_optimizer(name: str) -> type | None:
    if bnb is None:
        return None
    normalized = str(name).strip().lower()
    candidates = {
        "adam8bit": "Adam8bit",
        "adamw8bit": "AdamW8bit",
        "paged_adamw8bit": "PagedAdamW8bit",
        "pagedadamw8bit": "PagedAdamW8bit",
    }
    cls_name = candidates.get(normalized)
    if cls_name is None:
        return None
    return getattr(bnb.optim, cls_name, None)


def build_optimizer(
    params: Sequence[torch.nn.Parameter],
    *,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    params_list = [param for param in params if param.requires_grad]
    if not params_list:
        raise RuntimeError("No trainable parameters found")

    requested = str(optimizer_name).strip().lower()
    if requested == "auto":
        requested = "paged_adamw8bit" if torch.cuda.is_available() and bnb is not None else "adamw"

    if requested != "adamw":
        bnb_cls = _resolve_bnb_optimizer(requested)
        if bnb_cls is not None:
            optimizer = bnb_cls(params_list, lr=float(lr), weight_decay=float(weight_decay))
            return optimizer, {"optimizer_name": requested, "uses_bitsandbytes": True}

    optimizer = AdamW(params_list, lr=float(lr), weight_decay=float(weight_decay))
    fallback_name = "adamw" if requested == "adamw" else f"{requested}->adamw"
    return optimizer, {"optimizer_name": fallback_name, "uses_bitsandbytes": False}


def maybe_enable_gradient_checkpointing(model: torch.nn.Module, *, enabled: bool) -> bool:
    if not bool(enabled):
        return False
    candidates = [model]
    inner = getattr(model, "model", None)
    if isinstance(inner, torch.nn.Module):
        candidates.append(inner)
    for candidate in candidates:
        fn = getattr(candidate, "gradient_checkpointing_enable", None)
        if callable(fn):
            fn()
            return True
    return False


def maybe_compile_module(
    module: torch.nn.Module,
    *,
    enabled: bool,
    mode: str = "default",
    fullgraph: bool = False,
) -> torch.nn.Module:
    if not bool(enabled):
        return module
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return module
    compile_mode = None if str(mode) == "default" else str(mode)
    return compile_fn(module, mode=compile_mode, fullgraph=bool(fullgraph))


def backward_step(
    loss: torch.Tensor,
    *,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    params: Iterable[torch.nn.Parameter],
    max_grad_norm: float = 0.0,
    after_unscale: Any | None = None,
) -> None:
    if scaler is not None and bool(getattr(scaler, "is_enabled", lambda: False)()):
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if callable(after_unscale):
            after_unscale()
        if float(max_grad_norm) > 0.0:
            torch.nn.utils.clip_grad_norm_(list(params), max_norm=float(max_grad_norm))
        scaler.step(optimizer)
        scaler.update()
        return

    loss.backward()
    if callable(after_unscale):
        after_unscale()
    if float(max_grad_norm) > 0.0:
        torch.nn.utils.clip_grad_norm_(list(params), max_norm=float(max_grad_norm))
    optimizer.step()
