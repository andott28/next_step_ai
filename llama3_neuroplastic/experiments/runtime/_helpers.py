"""Standalone helper functions and constants used by both the loader and the runtime."""
from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download


def _assert_bnb_ready(tensor: torch.Tensor, name: str, *, expected_device: torch.device) -> None:
    """Guard against CUDA illegal memory access in bitsandbytes kernels.

    bitsandbytes CUDA ops (ops.cu) require:
      1. The tensor lives on the expected CUDA device.
      2. The storage is contiguous (stride[0] == 1 for a 1-D tensor).
    Violating either causes a silent illegal memory access or a wrong-value
    dequantisation that only surfaces as a numeric NaN or a CUDA error on a
    completely unrelated later operation.
    """
    if tensor.device != expected_device:
        raise RuntimeError(
            f"bitsandbytes tensor '{name}' is on {tensor.device} but expected {expected_device}. "
            "Ensure the weight and absmax tensors are moved to the target device before calling "
            "_bnb_dequant_impl / dequantize_blockwise."
        )
    if not tensor.is_contiguous():
        raise RuntimeError(
            f"bitsandbytes tensor '{name}' is not contiguous (shape={tuple(tensor.shape)}, "
            f"strides={tensor.stride()}). Call .contiguous() before passing to bitsandbytes kernels."
        )



_SAFETENSORS_DTYPE_TO_TORCH: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

_DEFAULT_SPARSE_BASIS_TOP_K = 8
_SPARSE_MLP_EXECUTION_CANONICAL = "exact_blockwise_sparse"
_SPARSE_MLP_EXECUTION_ALIASES = {
    "exact_intermediate_sparse": _SPARSE_MLP_EXECUTION_CANONICAL,
}
_SPARSE_MLP_EXECUTION_CHOICES = {
    "auto",
    _SPARSE_MLP_EXECUTION_CANONICAL,
    *_SPARSE_MLP_EXECUTION_ALIASES.keys(),
}


def _normalize_sparse_mlp_execution(mode: str | None) -> str:
    normalized = str(mode or "auto").strip().lower() or "auto"
    normalized = _SPARSE_MLP_EXECUTION_ALIASES.get(normalized, normalized)
    if normalized not in _SPARSE_MLP_EXECUTION_CHOICES:
        raise RuntimeError(
            "Sparse MLP execution must be one of: auto, exact_blockwise_sparse, exact_intermediate_sparse"
        )
    return normalized


def _make_meta_parameter(shape: Sequence[int], *, dtype: torch.dtype) -> nn.Parameter:
    return nn.Parameter(torch.empty(tuple(int(dim) for dim in shape), device="meta", dtype=dtype), requires_grad=False)


def _configure_linear_shape_only(
    linear: nn.Linear,
    *,
    in_features: int,
    out_features: int,
    bias: bool,
    dtype: torch.dtype,
) -> None:
    linear.in_features = int(in_features)
    linear.out_features = int(out_features)
    linear.weight = _make_meta_parameter((int(out_features), int(in_features)), dtype=dtype)
    if bias:
        linear.bias = _make_meta_parameter((int(out_features),), dtype=dtype)
    else:
        linear.register_parameter("bias", None)


def _configure_llama_mlp_shape_only(mlp: nn.Module, *, hidden_size: int, intermediate_size: int, bias: bool, dtype: torch.dtype) -> None:
    if not hasattr(mlp, "gate_proj") or not hasattr(mlp, "up_proj") or not hasattr(mlp, "down_proj"):
        raise RuntimeError("Expected a standard Llama MLP with gate_proj/up_proj/down_proj")
    if hasattr(mlp, "hidden_size"):
        mlp.hidden_size = int(hidden_size)
    if hasattr(mlp, "intermediate_size"):
        mlp.intermediate_size = int(intermediate_size)
    _configure_linear_shape_only(
        mlp.gate_proj,
        in_features=int(hidden_size),
        out_features=int(intermediate_size),
        bias=bool(bias),
        dtype=dtype,
    )
    _configure_linear_shape_only(
        mlp.up_proj,
        in_features=int(hidden_size),
        out_features=int(intermediate_size),
        bias=bool(bias),
        dtype=dtype,
    )
    _configure_linear_shape_only(
        mlp.down_proj,
        in_features=int(intermediate_size),
        out_features=int(hidden_size),
        bias=bool(bias),
        dtype=dtype,
    )


def _torch_dtype_itemsize(dtype: torch.dtype) -> int:
    return int(torch.empty((), dtype=dtype).element_size())


def _unpermute_q_factor_rows(factor_u: torch.Tensor, head_perm: torch.Tensor, *, head_dim: int) -> torch.Tensor:
    if factor_u.ndim != 2:
        raise RuntimeError(f"Expected a rank-2 q factor, got shape {tuple(factor_u.shape)}")
    num_heads = int(head_perm.numel())
    if int(factor_u.shape[0]) != int(num_heads * int(head_dim)):
        raise RuntimeError(
            f"Q factor rows {int(factor_u.shape[0])} do not match num_heads*head_dim "
            f"({num_heads} * {int(head_dim)})."
        )
    aligned = factor_u.view(num_heads, int(head_dim), int(factor_u.shape[1]))
    out = torch.empty_like(aligned)
    out.index_copy_(0, head_perm.to(device=out.device, dtype=torch.long), aligned)
    return out.view_as(factor_u)


def _unpermute_o_factor_cols(factor_v: torch.Tensor, head_perm: torch.Tensor, *, head_dim: int) -> torch.Tensor:
    if factor_v.ndim != 2:
        raise RuntimeError(f"Expected a rank-2 o factor, got shape {tuple(factor_v.shape)}")
    num_heads = int(head_perm.numel())
    if int(factor_v.shape[1]) != int(num_heads * int(head_dim)):
        raise RuntimeError(
            f"O factor cols {int(factor_v.shape[1])} do not match num_heads*head_dim "
            f"({num_heads} * {int(head_dim)})."
        )
    aligned = factor_v.view(int(factor_v.shape[0]), num_heads, int(head_dim))
    out = torch.empty_like(aligned)
    out.index_copy_(1, head_perm.to(device=out.device, dtype=torch.long), aligned)
    return out.view_as(factor_v)


def _unpermute_headwise_tensor(tensor: torch.Tensor, head_perm: torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 1:
        raise RuntimeError(f"Expected tensor with head axis, got shape {tuple(tensor.shape)}")
    num_heads = int(head_perm.numel())
    if int(tensor.shape[0]) != num_heads:
        raise RuntimeError(
            f"Headwise tensor first dimension {int(tensor.shape[0])} does not match head_perm size {num_heads}."
        )
    out = torch.empty_like(tensor)
    out.index_copy_(0, head_perm.to(device=out.device, dtype=torch.long), tensor)
    return out


def _tensor_byte_view_cpu(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cpu":
        raise RuntimeError("Expected a CPU tensor for byte-view file I/O")
    if not tensor.is_contiguous():
        raise RuntimeError("Expected a contiguous CPU tensor for byte-view file I/O")
    return tensor.view(torch.uint8).reshape(-1)


def _readinto_cpu_tensor(file_obj: Any, tensor: torch.Tensor) -> None:
    byte_view = _tensor_byte_view_cpu(tensor)
    target_np = byte_view.numpy()
    target_mv = memoryview(target_np).cast("B")
    offset = 0
    total = int(byte_view.numel())
    chunk_bytes = 8 * 1024 * 1024
    while offset < total:
        want = min(chunk_bytes, total - offset)
        chunk = file_obj.read(int(want))
        got = len(chunk)
        if got <= 0:
            raise EOFError(f"Unexpected EOF while reading {total} bytes from safetensors shard")
        target_mv[offset : offset + got] = chunk
        offset += got


def _resolve_ram_cache_limit_bytes() -> int | None:
    raw = os.getenv("STREAMING_RAM_CACHE_MAX_GB", "").strip()
    if raw:
        try:
            limit_gb = float(raw)
        except ValueError:
            limit_gb = 0.0
        if limit_gb > 0:
            return int(limit_gb * (1024 ** 3))
        return None




    try:
        import psutil
        raw_frac = os.getenv("STREAMING_RAM_CACHE_AUTO_FRACTION", "").strip()
        try:
            auto_frac = float(raw_frac) if raw_frac else 0.50
        except ValueError:
            auto_frac = 0.50
        auto_frac = max(0.20, min(auto_frac, 0.90))
        available_bytes = psutil.virtual_memory().available
        min_headroom = int(6 * (1024 ** 3))
        target_headroom = int(18 * (1024 ** 3))
        proportional_headroom = int(available_bytes * 0.60)
        headroom = min(target_headroom, max(min_headroom, proportional_headroom))
        limit = min(int(available_bytes * auto_frac), max(0, int(available_bytes) - int(headroom)))
        limit = max(int(4 * (1024 ** 3)), int(limit))
        limit_gb = limit / (1024 ** 3)
        frac_pct = auto_frac * 100.0
        print(f"[ram_cache] auto-limit: {limit_gb:.1f} GB "
              f"({frac_pct:.0f} % of {available_bytes / (1024**3):.1f} GB available, "
              f"{headroom / (1024**3):.1f} GB transient headroom; "
              f"override with STREAMING_RAM_CACHE_MAX_GB)", flush=True)
        return limit
    except Exception:
        _default_gb = 20
        print(f"[ram_cache] psutil unavailable; defaulting to {_default_gb} GB limit. "
              f"Set STREAMING_RAM_CACHE_MAX_GB to override.", flush=True)
        return int(_default_gb * (1024 ** 3))


def _resolve_pin_ram_cache_default() -> bool:
    raw = os.getenv("STREAMING_RAM_CACHE_PIN", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    if os.name == "nt":
        return False
    return bool(torch.cuda.is_available())


def _resolve_background_prefetch_default() -> bool:
    raw = os.getenv("STREAMING_BACKGROUND_PREFETCH", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return True


def _resolve_windows_batch_preload_default() -> bool:
    raw = os.getenv("STREAMING_WINDOWS_BATCH_PRELOAD", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return os.name == "nt"


def _resolve_show_progress_default() -> bool:
    raw = os.getenv("STREAMING_SHOW_PROGRESS", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return False


def _resolve_gpu_lm_head_default() -> bool:
    raw = os.getenv("STREAMING_GPU_LM_HEAD", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(torch.cuda.is_available())


def _cuda_capability(device: torch.device) -> tuple[int, int]:
    if device.type != "cuda":
        return (0, 0)
    index = device.index if device.index is not None else torch.cuda.current_device()
    return tuple(int(x) for x in torch.cuda.get_device_capability(index))


def _is_windows_pre_ampere_cuda(device: torch.device) -> bool:
    if os.name != "nt" or device.type != "cuda":
        return False
    major, _minor = _cuda_capability(device)
    return major != 0 and major < 8


def _is_cuda_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "out of memory" in text
        or "cudaerrormemoryallocation" in text
        or "cublas_status_alloc_failed" in text
        or "hip error out of memory" in text
    )


def _add_offset_inplace_gpu(absmax: torch.Tensor, offset: Any, *, chunk_elems: int = 1 << 20) -> None:
    """Add nested-quant offset into a GPU absmax buffer with low peak VRAM.

    For vector offsets on CPU, adding the full tensor at once can trigger an
    implicit large GPU temporary. Apply in chunks instead.
    """
    if not torch.is_tensor(offset):
        absmax.add_(float(offset))
        return

    if int(offset.numel()) <= 1:
        absmax.add_(float(offset.item()))
        return

    if offset.device.type == "cuda":
        absmax.add_(offset.to(device=absmax.device, dtype=torch.float32, non_blocking=False))
        return

    off_cpu = offset.reshape(-1).to(dtype=torch.float32, device="cpu")
    n = int(off_cpu.numel())
    if n != int(absmax.numel()):
        absmax.add_(off_cpu.to(device=absmax.device, dtype=torch.float32, non_blocking=False))
        return

    step = max(1024, int(chunk_elems))
    for start in range(0, n, step):
        end = min(start + step, n)
        piece = off_cpu[start:end].to(device=absmax.device, dtype=torch.float32, non_blocking=False)
        absmax[start:end].add_(piece)


def _resolve_snapshot_dir(model_name_or_path: str, *, local_files_only: bool) -> Path:
    candidate = Path(model_name_or_path)
    if candidate.exists():
        return candidate.resolve()
    snapshot_dir = snapshot_download(
        repo_id=str(model_name_or_path),
        local_files_only=bool(local_files_only),
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*", "special_tokens_map.json"],
    )
    return Path(snapshot_dir).resolve()
