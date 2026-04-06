from __future__ import annotations

import copy
import concurrent.futures
import gc
import json
import os
import threading
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch.nn.functional as F
import bitsandbytes.functional as bnb_functional
from bitsandbytes.backends.cuda.ops import _dequantize_4bit_impl as _bnb_dequant_impl
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig

try:
    from transformers.cache_utils import DynamicCache
except Exception:  # pragma: no cover
    DynamicCache = None

from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

try:
    from ..token_posting_archive import TokenPostingArchive
except ImportError:
    try:
        from llama3_neuroplastic.token_posting_archive import TokenPostingArchive
    except ImportError:
        TokenPostingArchive = None  # type: ignore

try:
    from ..gqa_taylor_ssd import GQATaylorSSDSelfAttention, TaylorSSDLayerCache
except ImportError:  # pragma: no cover
    from gqa_taylor_ssd import GQATaylorSSDSelfAttention, TaylorSSDLayerCache

try:
    from ..basis_fitting import fit_layer_basis
except ImportError:  # pragma: no cover
    from basis_fitting import fit_layer_basis

try:
    from ..triton_sparse_mlp import (
        triton_sparse_input_linear_4bit,
        triton_sparse_mlp_available,
        triton_sparse_output_linear_4bit,
    )
except Exception:  # pragma: no cover
    try:
        from triton_sparse_mlp import (
            triton_sparse_input_linear_4bit,
            triton_sparse_mlp_available,
            triton_sparse_output_linear_4bit,
        )
    except Exception:  # pragma: no cover
        triton_sparse_input_linear_4bit = None
        triton_sparse_output_linear_4bit = None

        def triton_sparse_mlp_available() -> bool:
            return False

import struct as _struct

# Maps safetensors dtype string → torch.dtype for _load_safetensors_direct.
_SAFETENSORS_DTYPE_TO_TORCH: Dict[str, "torch.dtype"] = {
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
        getattr(mlp, "gate_proj"),
        in_features=int(hidden_size),
        out_features=int(intermediate_size),
        bias=bool(bias),
        dtype=dtype,
    )
    _configure_linear_shape_only(
        getattr(mlp, "up_proj"),
        in_features=int(hidden_size),
        out_features=int(intermediate_size),
        bias=bool(bias),
        dtype=dtype,
    )
    _configure_linear_shape_only(
        getattr(mlp, "down_proj"),
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
    offset = 0
    total = int(byte_view.numel())
    chunk_bytes = 8 * 1024 * 1024
    while offset < total:
        want = min(chunk_bytes, total - offset)
        chunk_np = np.fromfile(file_obj, dtype=np.uint8, count=int(want))
        got = int(chunk_np.size)
        if got <= 0:
            raise EOFError(f"Unexpected EOF while reading {total} bytes from safetensors shard")
        target_np[offset : offset + got] = chunk_np
        offset += got


def _load_safetensors_direct(
    shard_path: Path, keys: Sequence[str]
) -> Dict[str, torch.Tensor]:
    """Load specific tensors from a safetensors file via direct I/O (no mmap).

    On Windows, repeated safe_open (mmap) calls on a 5 GB shard produce an
    access-violation in torch/storage.py after a few opens — even when each
    context manager is properly closed — because Windows does not reliably
    release the VA region between opens of a large file.  Reading each tensor
    directly with a seek + read sidesteps the issue entirely while producing
    a regular (non-mmap) CPU tensor indistinguishable from the mmap path.
    """
    keys_set = set(keys)
    out: Dict[str, torch.Tensor] = {}
    with open(str(shard_path), "rb") as f:
        header_size = _struct.unpack("<Q", f.read(8))[0]
        header: Dict[str, Any] = json.loads(f.read(header_size).decode("utf-8"))
        data_base = 8 + header_size  # byte offset where tensor data begins

        for key, meta in header.items():
            if key == "__metadata__" or key not in keys_set:
                continue
            if not isinstance(meta, dict):
                continue
            dtype_str = str(meta.get("dtype", ""))
            torch_dtype = _SAFETENSORS_DTYPE_TO_TORCH.get(dtype_str)
            if torch_dtype is None:
                continue
            shape: List[int] = list(meta.get("shape", []))
            start, end = meta["data_offsets"]
            nbytes = end - start
            if nbytes <= 0:
                continue
            itemsize = _torch_dtype_itemsize(torch_dtype)
            if nbytes % itemsize != 0:
                continue
            f.seek(data_base + start)
            tensor_shape = tuple(shape) if shape else (int(nbytes) // itemsize,)
            tensor = torch.empty(tensor_shape, dtype=torch_dtype)
            _readinto_cpu_tensor(f, tensor)
            out[key] = tensor
    return out


def _resolve_ram_cache_limit_bytes() -> Optional[int]:
    raw = os.getenv("STREAMING_RAM_CACHE_MAX_GB", "").strip()
    if raw:
        try:
            limit_gb = float(raw)
        except ValueError:
            limit_gb = 0.0
        if limit_gb > 0:
            return int(limit_gb * (1024 ** 3))
        return None  # explicitly disabled via "0"

    # Auto-detect: reserve >=90 % of currently available system RAM for the cache.
    # STREAMING_RAM_CACHE_AUTO_FRACTION may increase this (up to 98 %) but can
    # never lower below 90 %. This keeps the default stable for one-command runs.
    # This keeps the OS and other processes comfortable while maximising the number
    # of hot layers (each 405B decoder layer is ~1.6 GB of NF4 bytes).
    try:
        import psutil
        raw_frac = os.getenv("STREAMING_RAM_CACHE_AUTO_FRACTION", "").strip()
        try:
            auto_frac = float(raw_frac) if raw_frac else 0.90
        except ValueError:
            auto_frac = 0.90
        auto_frac = max(0.90, min(auto_frac, 0.98))
        available_bytes = psutil.virtual_memory().available
        limit = int(available_bytes * auto_frac)
        limit_gb = limit / (1024 ** 3)
        frac_pct = auto_frac * 100.0
        print(f"[ram_cache] auto-limit: {limit_gb:.1f} GB "
              f"({frac_pct:.0f} % of {available_bytes / (1024**3):.1f} GB available; "
              f"override with STREAMING_RAM_CACHE_MAX_GB)", flush=True)
        return limit
    except Exception:
        # psutil unavailable or failed — fall back to a conservative 20 GB default
        # so we never silently consume all RAM on an unknown machine.
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
    if os.name == "nt":
        return False
    return True


def _resolve_windows_batch_preload_default() -> bool:
    raw = os.getenv("STREAMING_WINDOWS_BATCH_PRELOAD", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return False


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


def _cuda_capability(device: torch.device) -> Tuple[int, int]:
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
        # Conservative fallback for unexpected shapes.
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


class ShardedSafetensorLoader:
    def __init__(
        self,
        snapshot_dir: Path,
        *,
        cache_shard_handles: Optional[bool] = None,
        pin_ram_cache: Optional[bool] = None,
    ) -> None:
        self.snapshot_dir = Path(snapshot_dir)
        index_path = self.snapshot_dir / "model.safetensors.index.json"
        self.weight_map: Dict[str, str] = {}
        if index_path.exists():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            raw_map = payload.get("weight_map", {})
            self.weight_map = {str(k): str(v) for k, v in raw_map.items()}
        else:
            safetensor_files = sorted(self.snapshot_dir.glob("*.safetensors"))
            if not safetensor_files:
                raise RuntimeError(f"No safetensors shards found in {self.snapshot_dir}")
            with safe_open(str(safetensor_files[0]), framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    self.weight_map[str(name)] = safetensor_files[0].name

        self._available_names = set(self.weight_map.keys())
        self._quant_aux_by_base: Dict[str, List[str]] = defaultdict(list)
        for name in self.weight_map:
            if ".weight." not in name:
                continue
            base = name.split(".weight.", 1)[0] + ".weight"
            self._quant_aux_by_base[base].append(name)
        # Windows + many concurrent safe_open mappings for 405B shards reliably
        # hard-crash once the stream crosses into later shards. Re-open on demand
        # there; keep the old cached-handle fast path elsewhere.
        if cache_shard_handles is None:
            cache_shard_handles = os.name != "nt"
        self._cache_shard_handles = bool(cache_shard_handles)
        self._shard_handles: Dict[str, Any] = {}
        # RAM weight cache: maps full parameter name → (weight_bytes, quant_aux_dict)
        # Both tensors are pinned (page-locked) for fast DMA to GPU.
        # After the first forward pass all weights live here; subsequent tokens
        # do RAM→GPU transfers instead of SSD→GPU (typically 5–10× faster).
        self._ram_cache: Dict[str, Tuple[torch.Tensor, Dict[str, torch.Tensor]]] = {}
        self._ram_cache_lru: "OrderedDict[str, None]" = OrderedDict()
        self._ram_cache_entry_bytes: Dict[str, int] = {}
        self._ram_cache_current_bytes: int = 0
        self._ram_cache_limit_bytes: Optional[int] = _resolve_ram_cache_limit_bytes()
        self._ram_cache_enabled: bool = True
        self._pin_ram_cache: bool = _resolve_pin_ram_cache_default() if pin_ram_cache is None else bool(pin_ram_cache)
        self._ram_cache_lock = threading.Lock()
        self._tensor_load_lock = threading.Lock()
        self._quant_meta_cache: Dict[str, Dict[str, Any]] = {}
        self._quant_meta_lock = threading.Lock()
        self._h2d_copy_scratch: Dict[str, torch.Tensor] = {}
        self._h2d_copy_events: Dict[str, Any] = {}
        self._direct_header_cache: Dict[str, Tuple[int, Dict[str, Any]]] = {}
        self._direct_header_lock = threading.Lock()

    @staticmethod
    def _entry_nbytes(weight: torch.Tensor, quant_aux: Dict[str, torch.Tensor]) -> int:
        total = int(weight.numel() * weight.element_size())
        for tensor in quant_aux.values():
            total += int(tensor.numel() * tensor.element_size())
        return total

    def _maybe_pin_cpu_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self._pin_ram_cache:
            return tensor
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        if tensor.is_pinned():
            return tensor
        try:
            return tensor.pin_memory()
        except Exception:
            return tensor

    def prepare_h2d_source(
        self,
        tensor: torch.Tensor,
        *,
        dtype: Optional[torch.dtype] = None,
        pin_override: Optional[bool] = None,
    ) -> torch.Tensor:
        if tensor.device.type != "cpu":
            tensor = tensor.to(device="cpu")
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        should_pin = self._pin_ram_cache if pin_override is None else bool(pin_override)
        if not should_pin:
            return tensor
        if tensor.is_pinned():
            return tensor
        try:
            return tensor.pin_memory()
        except Exception:
            return tensor

    def _stage_h2d_source_via_scratch(
        self,
        tensor: torch.Tensor,
        *,
        dtype: Optional[torch.dtype] = None,
        scratch_key: str,
    ) -> torch.Tensor:
        if tensor.device.type != "cpu":
            tensor = tensor.to(device="cpu")
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        if tensor.is_pinned():
            return tensor

        previous = self._h2d_copy_events.get(scratch_key)
        if previous is not None:
            try:
                previous.synchronize()
            except Exception:
                pass

        numel = int(tensor.numel())
        scratch = self._h2d_copy_scratch.get(scratch_key)
        if scratch is None or scratch.dtype != tensor.dtype or int(scratch.numel()) < numel:
            scratch = torch.empty(numel, dtype=tensor.dtype, pin_memory=True)
            self._h2d_copy_scratch[scratch_key] = scratch

        view = scratch[:numel].view(tuple(tensor.shape))
        view.copy_(tensor, non_blocking=False)
        return view

    def _record_h2d_scratch_use(self, scratch_key: Optional[str], *, device: torch.device) -> None:
        if not scratch_key or device.type != "cuda":
            return
        event = self._h2d_copy_events.get(scratch_key)
        if event is None:
            event = torch.cuda.Event()
            self._h2d_copy_events[scratch_key] = event
        event.record(torch.cuda.current_stream(device))

    def _evict_ram_cache_locked(self) -> None:
        if self._ram_cache_limit_bytes is None:
            return
        while self._ram_cache_current_bytes > self._ram_cache_limit_bytes and self._ram_cache_lru:
            victim, _ = self._ram_cache_lru.popitem(last=False)
            self._ram_cache.pop(victim, None)
            self._ram_cache_current_bytes -= self._ram_cache_entry_bytes.pop(victim, 0)

    def _get_direct_shard_header(self, shard_name: str) -> Tuple[int, Dict[str, Any]]:
        cached = self._direct_header_cache.get(shard_name)
        if cached is not None:
            return cached
        shard_path = self.snapshot_dir / str(shard_name)
        with open(str(shard_path), "rb") as f:
            header_size = _struct.unpack("<Q", f.read(8))[0]
            header: Dict[str, Any] = json.loads(f.read(header_size).decode("utf-8"))
        payload = (8 + int(header_size), header)
        with self._direct_header_lock:
            existing = self._direct_header_cache.get(shard_name)
            if existing is not None:
                return existing
            self._direct_header_cache[shard_name] = payload
        return payload

    def _get_tensor_direct_meta(self, full_name: str) -> Tuple[Path, int, Dict[str, Any]]:
        shard_name = self.weight_map.get(full_name)
        if shard_name is None:
            raise KeyError(f"Tensor '{full_name}' not found in safetensors index")
        data_base, header = self._get_direct_shard_header(shard_name)
        meta = header.get(full_name)
        if not isinstance(meta, dict):
            raise KeyError(f"Tensor '{full_name}' metadata missing from safetensors header")
        return self.snapshot_dir / shard_name, int(data_base), meta

    def load_rows(self, name: str, row_indices: Sequence[int]) -> torch.Tensor:
        full_name = str(name)
        rows_cpu = torch.as_tensor(list(row_indices), dtype=torch.long, device=torch.device("cpu")).reshape(-1)
        if os.name != "nt":
            tensor = self.load_parameter(full_name)
            if tensor.ndim != 2:
                raise RuntimeError(f"Row loading requires a 2D dense tensor, got shape {tuple(tensor.shape)}")
            return tensor.index_select(0, rows_cpu)

        shard_path, data_base, meta = self._get_tensor_direct_meta(full_name)
        dtype_str = str(meta.get("dtype", ""))
        torch_dtype = _SAFETENSORS_DTYPE_TO_TORCH.get(dtype_str)
        if torch_dtype is None:
            raise RuntimeError(f"Unsupported safetensors dtype '{dtype_str}' for row loading")
        shape = list(meta.get("shape", []))
        if len(shape) != 2:
            raise RuntimeError(f"Row loading requires a 2D dense tensor, got shape {tuple(shape)}")
        num_rows = int(shape[0])
        num_cols = int(shape[1])
        if int(rows_cpu.numel()) == 0:
            return torch.empty((0, num_cols), dtype=torch_dtype)
        if bool(((rows_cpu < 0) | (rows_cpu >= num_rows)).any()):
            raise IndexError(f"Row indices out of range for '{full_name}' with {num_rows} rows")

        unique_rows, inverse = rows_cpu.unique(sorted=True, return_inverse=True)
        itemsize = _torch_dtype_itemsize(torch_dtype)
        row_bytes = int(num_cols) * int(itemsize)
        start_offset, _end_offset = meta["data_offsets"]
        out_unique = torch.empty((int(unique_rows.numel()), num_cols), dtype=torch_dtype)

        with open(str(shard_path), "rb") as f:
            cursor = 0
            total = int(unique_rows.numel())
            while cursor < total:
                run_start = cursor
                run_first = int(unique_rows[cursor].item())
                cursor += 1
                while cursor < total and int(unique_rows[cursor].item()) == int(unique_rows[cursor - 1].item()) + 1:
                    cursor += 1
                dest = out_unique[run_start:cursor]
                f.seek(int(data_base) + int(start_offset) + run_first * row_bytes)
                _readinto_cpu_tensor(f, dest)

        return out_unique.index_select(0, inverse)

    def _load_exact_tensors(self, names: Sequence[str]) -> Dict[str, torch.Tensor]:
        requested = [str(name) for name in names if str(name) in self._available_names]
        by_shard: Dict[str, List[str]] = defaultdict(list)
        for name in requested:
            by_shard[self.weight_map[name]].append(name)

        out: Dict[str, torch.Tensor] = {}
        with self._tensor_load_lock:
            for shard_name, shard_keys in by_shard.items():
                shard_path = self.snapshot_dir / shard_name
                if self._cache_shard_handles:
                    if shard_name not in self._shard_handles:
                        self._shard_handles[shard_name] = safe_open(str(shard_path), framework="pt", device="cpu")
                    handle = self._shard_handles[shard_name]
                    for key in shard_keys:
                        out[key] = handle.get_tensor(key)
                    continue

                if os.name == "nt":
                    # Bypass mmap (safe_open) entirely on Windows: repeated
                    # MapViewOfFile calls on a 5 GB shard cause access-violations
                    # in torch/storage.__getitem__ even after each context-manager
                    # exits.  Direct file I/O produces identical tensors without
                    # touching the VA-space-limited Windows mmap machinery.
                    direct = _load_safetensors_direct(shard_path, shard_keys)
                    out.update(direct)
                    continue
                with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                    for key in shard_keys:
                        out[key] = handle.get_tensor(key)
        return out

    def load_parameter(self, name: str) -> torch.Tensor:
        full_name = str(name)
        weight, quant_aux = self._load_raw_for_param(full_name)
        if not quant_aux:
            return weight

        quant_state = bnb_functional.QuantState.from_dict(quant_aux, device=torch.device("cpu"))
        return bnb_functional.dequantize_4bit(weight, quant_state=quant_state).cpu()

    def _get_cached_quant_meta(
        self,
        full_name: str,
        quant_aux: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        cached = self._quant_meta_cache.get(full_name)
        if cached is not None:
            return cached

        quant_state = bnb_functional.QuantState.from_dict(quant_aux, device=torch.device("cpu"))
        absmax = quant_state.absmax
        if quant_state.nested:
            absmax = bnb_functional.dequantize_blockwise(quant_state.absmax, quant_state.state2)
            absmax = absmax + quant_state.offset

        meta = {
            "absmax_cpu": absmax.to(dtype=torch.float32).contiguous(),
            "blocksize": int(quant_state.blocksize),
            "quant_type": str(quant_state.quant_type),
            "dtype": quant_state.dtype,
        }
        with self._quant_meta_lock:
            existing = self._quant_meta_cache.get(full_name)
            if existing is not None:
                return existing
            self._quant_meta_cache[full_name] = meta
        return meta

    def _load_raw_for_param(self, full_name: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return (weight_bytes, quant_aux) for a parameter, using RAM cache when warm.

        On cache miss the tensors are loaded from disk and stored in the RAM cache
        as pinned (page-locked) memory so subsequent tokens DMA directly to the GPU
        staging buffer without going through the SSD.
        """
        if self._ram_cache_enabled:
            with self._ram_cache_lock:
                cached = self._ram_cache.get(full_name)
                if cached is not None and self._ram_cache_limit_bytes is not None:
                    self._ram_cache_lru.move_to_end(full_name, last=True)
            if cached is not None:
                weight, quant_aux = cached
                return weight, dict(quant_aux)

        aux_names = self._quant_aux_by_base.get(full_name, [])
        tensors = self._load_exact_tensors([full_name, *aux_names])
        if full_name not in tensors:
            raise KeyError(f"Tensor '{full_name}' not found in safetensors index")

        weight = tensors[full_name]
        quant_aux = {k: v for k, v in tensors.items() if k != full_name}

        if self._ram_cache_enabled:
            if os.name == "nt" and self._cache_shard_handles:
                weight = weight.clone().contiguous()
                quant_aux = {k: v.clone().contiguous() for k, v in quant_aux.items()}
            weight = self._maybe_pin_cpu_tensor(weight.contiguous())
            quant_aux = {k: self._maybe_pin_cpu_tensor(v.contiguous()) for k, v in quant_aux.items()}
            # Store in pageable or pinned CPU memory depending on configuration.
            # Pinned RAM enables non_blocking DMA on the H2D stream for faster
            # transfers once the cache is warm.
            with self._ram_cache_lock:
                cached = self._ram_cache.get(full_name)
                if cached is not None:
                    if self._ram_cache_limit_bytes is not None:
                        self._ram_cache_lru.move_to_end(full_name, last=True)
                    weight_cached, quant_aux_cached = cached
                    return weight_cached, dict(quant_aux_cached)
                self._ram_cache[full_name] = (weight, quant_aux)
                if self._ram_cache_limit_bytes is not None:
                    self._ram_cache_lru[full_name] = None
                    self._ram_cache_lru.move_to_end(full_name, last=True)
                    self._ram_cache_entry_bytes[full_name] = self._entry_nbytes(weight, quant_aux)
                    self._ram_cache_current_bytes += self._ram_cache_entry_bytes[full_name]
                    self._evict_ram_cache_locked()
            return weight, dict(quant_aux)

        return weight, dict(quant_aux)

    def load_parameter_into(
        self,
        name: str,
        out: torch.Tensor,
        dtype: torch.dtype,
        staging: Optional[torch.Tensor] = None,
        absmax_staging: Optional[torch.Tensor] = None,
        nested_absmax_staging: Optional[torch.Tensor] = None,
        state2_absmax_staging: Optional[torch.Tensor] = None,
        code_staging: Optional[torch.Tensor] = None,
        byte_counter: Optional[Any] = None,
    ) -> None:
        """Dequantize NF4 weight directly into a pre-allocated GPU skeleton buffer.

        All GPU tensors are pre-allocated: `staging` (uint8, NF4 bytes),
        `absmax_staging` (fp32, dequantized absmax output),
        `nested_absmax_staging` (uint8, doubly-quantized absmax input),
        `state2_absmax_staging` (fp32, secondary quant scales),
        `code_staging` (fp32, dequant codebook).

        QuantState is created on CPU (fast dict parsing, zero GPU allocs).
        Small tensors are memcpy'd into the pre-allocated GPU buffers and all
        dequant kernels run on GPU — same speed as GPU QuantState, but zero
        cudaMalloc calls that would fragment the pool and crash at layer ~48.

        On the first forward pass weights are loaded from disk and stored in the
        RAM cache.  All subsequent tokens hit the RAM cache, eliminating SSD I/O.
        """
        full_name = str(name)
        weight, quant_aux = self._load_raw_for_param(full_name)

        def _copy_cpu_into(
            dest: torch.Tensor,
            src: torch.Tensor,
            *,
            dtype: Optional[torch.dtype] = None,
            scratch_key: str = "generic",
        ) -> torch.Tensor:
            prepared = self._stage_h2d_source_via_scratch(src, dtype=dtype, scratch_key=scratch_key)
            dest.copy_(prepared, non_blocking=bool(prepared.is_pinned()))
            self._record_h2d_scratch_use(scratch_key, device=dest.device)
            return prepared

        if not quant_aux:
            if byte_counter is not None:
                _prepared = self._stage_h2d_source_via_scratch(weight, dtype=dtype, scratch_key="dense_out")
                byte_counter(int(_prepared.numel() * _prepared.element_size()))
                out.copy_(_prepared, non_blocking=bool(_prepared.is_pinned()))
                self._record_h2d_scratch_use("dense_out", device=out.device)
            else:
                _copy_cpu_into(out, weight, dtype=dtype, scratch_key="dense_out")
            return

        # DMA NF4 bytes into the pre-allocated staging buffer — no cudaMalloc.
        n = weight.numel()
        _prepared_weight = self._stage_h2d_source_via_scratch(
            weight.reshape(-1),
            dtype=staging.dtype if staging is not None else None,
            scratch_key="nf4_weight",
        )
        traffic_bytes = int(_prepared_weight.numel() * _prepared_weight.element_size())
        if staging is not None:
            weight_gpu = staging[:n]
            weight_gpu.copy_(_prepared_weight, non_blocking=bool(_prepared_weight.is_pinned()))
            self._record_h2d_scratch_use("nf4_weight", device=weight_gpu.device)
        else:
            weight_gpu = _prepared_weight.to(device=out.device, non_blocking=bool(_prepared_weight.is_pinned()))

        if absmax_staging is not None:
            quant_meta = self._get_cached_quant_meta(full_name, quant_aux)
            absmax_cpu = quant_meta["absmax_cpu"]
            traffic_bytes += int(absmax_cpu.numel() * absmax_cpu.element_size())
            n_abs = absmax_cpu.numel()
            absmax = absmax_staging[:n_abs]
            _copy_cpu_into(absmax, absmax_cpu, dtype=absmax.dtype, scratch_key="nf4_absmax")
            if absmax.dtype != torch.float32:
                absmax = absmax.float()

            if n * 2 != out.numel():
                raise RuntimeError(
                    f"Shape mismatch for '{full_name}': {n} NF4 bytes Ã¢â€ â€™ {n * 2} elements "
                    f"but out has {out.numel()} elements (shape {out.shape})"
                )

            _bnb_dequant_impl(
                weight_gpu,
                absmax,
                int(quant_meta["blocksize"]),
                str(quant_meta["quant_type"]),
                quant_meta["dtype"],
                out=out,
            )
            if byte_counter is not None:
                byte_counter(int(traffic_bytes))
            del absmax
            return

        if absmax_staging is not None:
            # Fast path: keep QuantState parsing on CPU and use the pre-allocated
            # GPU staging buffers for the nested absmax decode as well.
            quant_state_cpu = bnb_functional.QuantState.from_dict(quant_aux, device=torch.device("cpu"))
            if quant_state_cpu.nested:
                traffic_bytes += int(quant_state_cpu.absmax.numel() * quant_state_cpu.absmax.element_size())
                traffic_bytes += int(
                    quant_state_cpu.state2.absmax.numel() * quant_state_cpu.state2.absmax.element_size()
                )
                traffic_bytes += int(quant_state_cpu.state2.code.numel() * quant_state_cpu.state2.code.element_size())
                if (
                    nested_absmax_staging is not None
                    and state2_absmax_staging is not None
                    and code_staging is not None
                ):
                    n_nested = quant_state_cpu.absmax.numel()
                    nested_gpu = nested_absmax_staging[:n_nested]
                    _copy_cpu_into(nested_gpu, quant_state_cpu.absmax, dtype=nested_gpu.dtype)

                    n_s2 = quant_state_cpu.state2.absmax.numel()
                    s2_gpu = state2_absmax_staging[:n_s2]
                    _copy_cpu_into(s2_gpu, quant_state_cpu.state2.absmax, dtype=s2_gpu.dtype)

                    _copy_cpu_into(
                        code_staging[: quant_state_cpu.state2.code.numel()],
                        quant_state_cpu.state2.code,
                        dtype=code_staging.dtype,
                    )
                    code_gpu = code_staging[: quant_state_cpu.state2.code.numel()]

                    absmax = absmax_staging[:n_nested]
                    bnb_functional.dequantize_blockwise(
                        nested_gpu,
                        absmax=s2_gpu,
                        code=code_gpu,
                        out=absmax,
                        blocksize=quant_state_cpu.state2.blocksize,
                    )
                    _add_offset_inplace_gpu(absmax.reshape(-1), quant_state_cpu.offset)
                else:
                    quant_state_cpu = None
            else:
                traffic_bytes += int(quant_state_cpu.absmax.numel() * quant_state_cpu.absmax.element_size())
                n_abs = quant_state_cpu.absmax.numel()
                absmax = absmax_staging[:n_abs]
                _copy_cpu_into(absmax, quant_state_cpu.absmax, dtype=absmax.dtype)
                if absmax.dtype != torch.float32:
                    absmax = absmax.float()

            if quant_state_cpu is not None:
                if n * 2 != out.numel():
                    raise RuntimeError(
                        f"Shape mismatch for '{full_name}': {n} NF4 bytes â†’ {n * 2} elements "
                        f"but out has {out.numel()} elements (shape {out.shape})"
                    )

                _bnb_dequant_impl(
                    weight_gpu,
                    absmax,
                    quant_state_cpu.blocksize,
                    quant_state_cpu.quant_type,
                    quant_state_cpu.dtype,
                    out=out,
                )
                if byte_counter is not None:
                    byte_counter(int(traffic_bytes))
                del absmax, quant_state_cpu
                return

        # Create QuantState on GPU so absmax tensors are already on the right device.
        quant_state = bnb_functional.QuantState.from_dict(quant_aux, device=out.device)
        traffic_bytes += int(quant_state.absmax.numel() * quant_state.absmax.element_size())
        if quant_state.nested:
            traffic_bytes += int(quant_state.state2.absmax.numel() * quant_state.state2.absmax.element_size())
            traffic_bytes += int(quant_state.state2.code.numel() * quant_state.state2.code.element_size())

        if n * 2 != out.numel():
            raise RuntimeError(
                f"Shape mismatch for '{full_name}': {n} NF4 bytes → {n * 2} elements "
                f"but out has {out.numel()} elements (shape {out.shape})"
            )

        # Use _bnb_dequant_impl directly rather than the high-level dequantize_4bit.
        # The high-level function routes through torch.ops custom-op dispatch which
        # converts the 'shape' argument from torch.Size → list; bitsandbytes then
        # checks `out.shape == shape` where out.shape is torch.Size and shape is a
        # list — Python considers them unequal and raises RuntimeError even when the
        # dimensions are identical.  _bnb_dequant_impl avoids that dispatch path.
        if quant_state.nested:
            absmax = bnb_functional.dequantize_blockwise(quant_state.absmax, quant_state.state2)
            absmax = absmax.float() + quant_state.offset
        else:
            absmax = quant_state.absmax.float()
        _bnb_dequant_impl(weight_gpu, absmax, quant_state.blocksize, quant_state.quant_type, quant_state.dtype, out=out)
        if byte_counter is not None:
            byte_counter(int(traffic_bytes))
        del absmax, quant_state

    def load_module_state(
        self,
        *,
        prefix: str,
        expected_keys: Iterable[str],
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        state: Dict[str, torch.Tensor] = {}
        for key in expected_keys:
            full_name = f"{prefix}{key}"
            tensor = self.load_parameter(full_name)
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                tensor = tensor.to(dtype=dtype)
            state[str(key)] = tensor
        return state


class StreamingLlamaRuntime:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
        taylor_layers: Optional[List[int]] = None,
        taylor_feature_map: str = "hybrid_performer",
        taylor_local_window: int = 64,
        taylor_feature_dim: int = 64,
        taylor_state_decay: float = 1.0,
        local_files_only: bool = True,
        ram_cache: bool = True,
        ram_cache_pinned: Optional[bool] = None,
        materialize_lm_head: bool = True,
        sparse_basis_path: Optional[str] = None,
        sparse_top_k: Optional[int] = None,
        vram_hot_cache_gb: Optional[float] = None,
        hot_block_threshold: float = 0.80,
        attn_head_importance_path: Optional[str] = None,
        attn_share_path: Optional[str] = None,
        attn_active_heads: Optional[int] = None,
        attn_head_activity_threshold: float = 0.10,
        attn_min_active_heads: int = 16,
        attn_max_active_heads: Optional[int] = None,
        enable_triton_fused_sparse_mlp: bool = True,
        enable_cuda_h2d_overlap: bool = True,
        kv_basis_path: Optional[str] = None,
        kv_sparse_top_k: Optional[int] = None,
        # ── Token-posting sparse attention ────────────────────────────────────
        attn_token_posting_path: Optional[str] = None,
        attn_retrieval_ring_size: int = 256,
        attn_retrieval_num_sinks: int = 16,
        attn_retrieval_candidates: int = 64,
        attn_retrieval_r_query: int = 6,
        attn_retrieval_token_topk: int = 8,
        attn_retrieval_archive_capacity: int = 16384,
    ) -> None:
        self.snapshot_dir = _resolve_snapshot_dir(model_name_or_path, local_files_only=bool(local_files_only))
        self.config = AutoConfig.from_pretrained(str(self.snapshot_dir), local_files_only=bool(local_files_only))
        if str(getattr(self.config, "model_type", "")) != "llama":
            raise RuntimeError(f"Streaming runtime only supports llama models, got {self.config.model_type!r}")

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        # AutoConfig.from_pretrained does not set _attn_implementation (that happens inside
        # PreTrainedModel.from_pretrained).  Leaving it as None causes a KeyError in
        # ALL_ATTENTION_FUNCTIONS when layer.self_attn is called directly (e.g. in the
        # no-cache data-collection path).  'sdpa' is available on any PyTorch 2.x + CUDA
        # setup and is faster than 'eager' for the single-token forward we do here.
        if getattr(self.config, "_attn_implementation", None) is None:
            self.config._attn_implementation = "sdpa"
        self._debug_steps = os.getenv("STREAMING_DEBUG_STEPS", "").strip().lower() in {"1", "true", "yes", "on"}
        self._debug_sync_cuda = os.getenv("STREAMING_DEBUG_SYNC_CUDA", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._sparse_basis_bias_mode = os.getenv("STREAMING_SPARSE_BIAS_MODE", "selected").strip().lower() or "selected"
        if self._sparse_basis_bias_mode not in {"selected", "none"}:
            raise RuntimeError(
                "STREAMING_SPARSE_BIAS_MODE must be one of: selected, none"
            )
        self._sparse_basis_execution = (
            os.getenv("STREAMING_SPARSE_BASIS_EXECUTION", "full_output_latent").strip().lower()
            or "full_output_latent"
        )
        if self._sparse_basis_execution not in {"full_output_latent", "routed_blocks"}:
            raise RuntimeError(
                "STREAMING_SPARSE_BASIS_EXECUTION must be one of: full_output_latent, routed_blocks"
            )
        self._sparse_attn_prefill_mode = os.getenv("STREAMING_SPARSE_ATTN_PREFILL_MODE", "dense").strip().lower() or "dense"
        if self._sparse_attn_prefill_mode not in {"dense", "sparse"}:
            raise RuntimeError(
                "STREAMING_SPARSE_ATTN_PREFILL_MODE must be one of: dense, sparse"
            )
        self._sparse_kv_prefill_mode = os.getenv("STREAMING_SPARSE_KV_PREFILL_MODE", "dense").strip().lower() or "dense"
        if self._sparse_kv_prefill_mode not in {"dense", "sparse"}:
            raise RuntimeError(
                "STREAMING_SPARSE_KV_PREFILL_MODE must be one of: dense, sparse"
            )
        self._attn_share_prefill_mode = os.getenv("STREAMING_ATTN_SHARE_PREFILL_MODE", "dense").strip().lower() or "dense"
        if self._attn_share_prefill_mode not in {"dense", "shared"}:
            raise RuntimeError(
                "STREAMING_ATTN_SHARE_PREFILL_MODE must be one of: dense, shared"
            )
        _guard_chunk_blocks_raw = os.getenv("STREAMING_GUARD_MLP_CHUNK_BLOCKS", "").strip()
        self._guard_mlp_chunk_blocks = int(_guard_chunk_blocks_raw) if _guard_chunk_blocks_raw else 64
        self._guard_mlp_chunk_blocks = max(1, int(self._guard_mlp_chunk_blocks))
        self._show_progress = _resolve_show_progress_default()
        self._prefer_gpu_lm_head = _resolve_gpu_lm_head_default()
        if _is_windows_pre_ampere_cuda(self.device):
            self._prefer_gpu_lm_head = False
        self.loader = ShardedSafetensorLoader(self.snapshot_dir, pin_ram_cache=ram_cache_pinned)
        self.loader._ram_cache_enabled = bool(ram_cache)
        self._enable_background_prefetch = bool(ram_cache) and _resolve_background_prefetch_default()
        self._enable_windows_batch_preload = bool(
            os.name == "nt"
            and ram_cache
            and _resolve_windows_batch_preload_default()
        )
        self._enable_cuda_h2d_overlap = bool(enable_cuda_h2d_overlap and torch.cuda.is_available())
        self._h2d_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=self.device) if self._enable_cuda_h2d_overlap else None
        )
        self.num_layers = int(getattr(self.config, "num_hidden_layers", 0))
        if self.num_layers <= 0:
            raise RuntimeError("Invalid llama config: num_hidden_layers must be > 0")

        self._allow_taylor_with_sparse_attn = os.getenv(
            "STREAMING_ALLOW_TAYLOR_WITH_SPARSE_ATTN",
            "",
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._taylor_requested_layers = (
            set(range(self.num_layers))
            if taylor_layers is None
            else {int(idx) for idx in taylor_layers if 0 <= int(idx) < self.num_layers}
        )
        self._taylor_auto_disabled_for_sparse_attn = bool(
            attn_head_importance_path
            and str(attn_head_importance_path).strip()
            and int(len(self._taylor_requested_layers)) > 0
            and not self._allow_taylor_with_sparse_attn
        )
        self.taylor_layer_set = (
            set()
            if self._taylor_auto_disabled_for_sparse_attn
            else set(self._taylor_requested_layers)
        )
        self.taylor_feature_map = str(taylor_feature_map)
        self.taylor_local_window = int(taylor_local_window)
        self.taylor_feature_dim = int(taylor_feature_dim)
        self.taylor_state_decay = float(taylor_state_decay)
        self._triton_fused_sparse_mlp = bool(
            enable_triton_fused_sparse_mlp
            and triton_sparse_mlp_available()
            and torch.cuda.is_available()
            and self.device.type == "cuda"
        )
        if self._triton_fused_sparse_mlp and _is_windows_pre_ampere_cuda(self.device):
            self._triton_fused_sparse_mlp = False
            print(
                "[sparse] Triton fused 4-bit sparse MLP disabled on Windows pre-Ampere; "
                "using GPU dequant + GEMM path.",
                flush=True,
            )
        if vram_hot_cache_gb is None:
            raw_hot_gb = os.getenv("STREAMING_VRAM_HOT_CACHE_GB", "").strip()
            if raw_hot_gb:
                try:
                    vram_hot_cache_gb = float(raw_hot_gb)
                except ValueError:
                    vram_hot_cache_gb = None
            elif self.device.type == "cuda":
                vram_hot_cache_gb = 6.4
        self._vram_hot_cache_limit_bytes: Optional[int] = (
            int(float(vram_hot_cache_gb) * (1024 ** 3))
            if vram_hot_cache_gb is not None and float(vram_hot_cache_gb) > 0.0
            else None
        )
        self._vram_hot_cache_enabled: bool = self._vram_hot_cache_limit_bytes is not None
        self._vram_hot_cache_used_bytes: int = 0
        self._vram_nf4_cache: Dict[str, Dict[str, Any]] = {}
        self._vram_hot_cache_pressure_warned: bool = False
        self._vram_hot_cache_oom_warned: bool = False
        self._vram_hot_cache_disable_reason: Optional[str] = None
        self._vram_hot_cache_margin_bytes: int = int(1.0 * (1024 ** 3))
        if self.device.type == "cuda":
            try:
                _, _vram_total = torch.cuda.mem_get_info(self.device)
                # Keep at least 20% VRAM free for transient tensors/kernels.
                self._vram_hot_cache_margin_bytes = max(
                    self._vram_hot_cache_margin_bytes,
                    int(float(_vram_total) * 0.05),
                )
            except Exception:
                pass
        self._hot_block_threshold = float(max(0.0, min(1.0, hot_block_threshold)))
        self._mlp_hot_blocks_by_layer: Dict[int, torch.Tensor] = {}
        self._traffic_current_phase: str = "idle"
        self._traffic_bytes_by_phase: Dict[str, int] = defaultdict(int)
        self._traffic_layer_visits_by_phase: Dict[str, int] = defaultdict(int)
        self._traffic_bytes_by_phase_layer: Dict[Tuple[str, int], int] = defaultdict(int)
        self._traffic_layer_visits_by_phase_layer: Dict[Tuple[str, int], int] = defaultdict(int)
        self._traffic_bytes_by_phase_tag: Dict[Tuple[str, str], int] = defaultdict(int)
        self._last_traffic_report: Optional[Dict[str, Any]] = None
        self._session_token_ids_cpu: Optional[torch.LongTensor] = None
        self._session_last_logits_cpu: Optional[torch.Tensor] = None
        self._h2d_stage_slots: Dict[str, int] = defaultdict(int)
        _target_layer_mb_raw = os.getenv("STREAMING_TARGET_LAYER_MB", "").strip()
        try:
            self._target_layer_traffic_mb = float(_target_layer_mb_raw) if _target_layer_mb_raw else 30.0
        except ValueError:
            self._target_layer_traffic_mb = 30.0

        self._embed_weight_name = "model.embed_tokens.weight"
        self._embed_weight_cpu: Optional[torch.Tensor] = None
        self._embed_row_cache: Dict[int, torch.Tensor] = {}
        self._embed_row_cache_lock = threading.Lock()
        self._materialize_lm_head = bool(materialize_lm_head)

        self.norm = LlamaRMSNorm(int(self.config.hidden_size), eps=float(self.config.rms_norm_eps)).to(
            device=self.device,
            dtype=self.dtype,
        )
        self.norm.weight.data.copy_(
            self.loader.load_parameter("model.norm.weight").to(device=self.device, dtype=self.dtype)
        )
        self.norm.requires_grad_(False)

        # Keep lm_head as a direct CPU tensor alongside embed_tokens.
        self._lm_head_weight_name = (
            "lm_head.weight"
            if "lm_head.weight" in self.loader.weight_map
            else self._embed_weight_name
        )
        self._lm_head_weight_cpu: Optional[torch.Tensor] = None
        self._lm_head_weight_gpu: Optional[torch.Tensor] = None
        self._lm_head_gpu_attempted = False

        self.rotary_emb = LlamaRotaryEmbedding(self.config, device=self.device)
        self._taylor_caches: List[Optional[TaylorSSDLayerCache]] = [None for _ in range(self.num_layers)]
        self._dense_cache = DynamicCache(config=self.config) if DynamicCache is not None else None

        # ── Token-posting sparse attention archive ────────────────────────────
        self._token_archive: Optional["TokenPostingArchive"] = None
        self._retrieval_layers: Set[int] = set()
        self._retrieval_candidates: int = int(attn_retrieval_candidates)
        if attn_token_posting_path is not None and TokenPostingArchive is not None:
            _tp_path = Path(attn_token_posting_path)
            if _tp_path.exists():
                _tp_ckpt = torch.load(str(_tp_path), map_location="cpu", weights_only=False)
                _tp_cfg = _tp_ckpt.get("config", {})
                _tp_layers = list(_tp_cfg.get("retrieval_layers", []))
                _tp_rank = int(_tp_cfg.get("basis_rank", 32))
                _tp_G = int(_tp_cfg.get("num_kv_groups", getattr(self.config, "num_key_value_heads", 8)))
                _tp_D = int(_tp_cfg.get("head_dim", getattr(self.config, "head_dim", 128)))
                self._retrieval_layers = set(_tp_layers)
                self._token_archive = TokenPostingArchive(
                    retrieval_layers=_tp_layers,
                    num_kv_groups=_tp_G,
                    head_dim=_tp_D,
                    basis_rank=_tp_rank,
                    ring_size=int(attn_retrieval_ring_size),
                    num_sinks=int(attn_retrieval_num_sinks),
                    archive_capacity=int(attn_retrieval_archive_capacity),
                    token_topk=int(attn_retrieval_token_topk),
                    r_query=int(attn_retrieval_r_query),
                    candidates=int(attn_retrieval_candidates),
                    device=self.device,
                    dtype=self.dtype,
                )
                # Load per-(layer, group) PCA basis from checkpoint.
                for _ls_key, _ls_val in _tp_ckpt.get("layer_states", {}).items():
                    _ls_idx = int(_ls_key)
                    _grp_bases = _ls_val.get("group_bases", [])
                    _idf_weights = _ls_val.get("idf_weights", [])
                    _key_means = _ls_val.get("key_means", [])
                    import numpy as _np
                    for _g_idx in range(len(_grp_bases)):
                        self._token_archive.load_basis(
                            _ls_idx,
                            _g_idx,
                            _np.asarray(_grp_bases[_g_idx], dtype=_np.float32),
                            _np.asarray(_idf_weights[_g_idx], dtype=_np.float32),
                            _np.asarray(_key_means[_g_idx], dtype=_np.float32)
                            if _g_idx < len(_key_means) else None,
                        )
                print(
                    f"[token_posting] loaded basis for {len(_tp_layers)} retrieval layers "
                    f"| rank={_tp_rank} G={_tp_G} D={_tp_D}",
                    flush=True,
                )
            else:
                print(f"[token_posting] path not found: {_tp_path}", flush=True)
        skeleton_config = copy.deepcopy(self.config)
        skeleton_config.intermediate_size = 1
        self._layer_skeleton = LlamaDecoderLayer(skeleton_config, layer_idx=0).to(device=self.device, dtype=self.dtype)
        _configure_llama_mlp_shape_only(
            self._layer_skeleton.mlp,
            hidden_size=int(self.config.hidden_size),
            intermediate_size=int(self.config.intermediate_size),
            bias=bool(getattr(self.config, "mlp_bias", False)),
            dtype=self.dtype,
        )
        for p in self._layer_skeleton.parameters():
            p.requires_grad = False
        self._layer_skeleton.eval()
        # Keep only shape metadata for the decoder MLP inside the skeleton.
        # Sparse and dense streaming MLP paths load the real weights on demand;
        # preallocating a dense 405B MLP layer here wastes >5 GB of host RAM.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._shared_taylor_attn = None
        if self.taylor_layer_set:
            self._shared_taylor_attn = GQATaylorSSDSelfAttention.from_llama_attention(
                source_attn=self._layer_skeleton.self_attn,
                layer_idx=0,
                feature_map=self.taylor_feature_map,
                local_window=self.taylor_local_window,
                feature_dim=self.taylor_feature_dim,
                state_decay=self.taylor_state_decay,
            ).to(device=self.device)
            self._shared_taylor_attn.eval()
            for p in self._shared_taylor_attn.parameters():
                p.requires_grad = False
        elif self._taylor_auto_disabled_for_sparse_attn:
            print(
                "[taylor] disabled for streamed sparse-attention runtime; "
                "set STREAMING_ALLOW_TAYLOR_WITH_SPARSE_ATTN=1 to override.",
                flush=True,
            )

        # Background thread pool (1 worker) that warms the RAM cache for the
        # next layer while the GPU processes the current one.
        self._prefetch_executor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="layer_prefetch")
            if self._enable_background_prefetch
            else None
        )
        self._prefetch_lock = threading.Lock()
        self._prefetch_pending_layers: Set[int] = set()
        # Cached skeleton state refs avoid rebuilding state_dict() on every streamed layer load.
        _layer_state = self._layer_skeleton.state_dict(keep_vars=True)
        self._layer_state_items: List[Tuple[str, torch.Tensor, bool]] = [
            (str(k), v, bool(v.is_floating_point()))
            for k, v in _layer_state.items()
        ]
        self._layer_param_keys: List[str] = [
            k for k, _v, _is_fp in self._layer_state_items if bool(_is_fp)
        ]
        if self._prefetch_executor is not None and self._layer_param_keys:
            self._schedule_prefetch_layer(0)

        # Pre-allocate one fixed staging buffer for the NF4 uint8 bytes.
        # Sized to the largest weight in any decoder layer (gate_proj / up_proj / down_proj).
        # By allocating this once at startup—before any layer is streamed—we guarantee
        # a single contiguous CUDA allocation and avoid the pool fragmentation that
        # accumulates over ~48 layers and causes STATUS_ACCESS_VIOLATION when a later
        # weight.to(device=...) forces a new cudaMalloc near the 8 GB VRAM ceiling.
        _h = int(self.config.hidden_size)
        # Attention-only staging: load_parameter_into is only used for attention weights.
        # MLP weights are loaded via load_parameter (CPU dequant) in _dense_mlp_forward_streaming_fast.
        # Largest attention weight is q_proj/o_proj [hidden, hidden].
        _max_nf4_bytes = _h * _h // 2  # NF4 = 2 values per byte
        # absmax has one fp32 value per block of 64 weight elements.
        # num_blocks = max_weight_numel / 64 = (_max_nf4_bytes * 2) / 64 = _max_nf4_bytes / 32
        _max_absmax_numel = _max_nf4_bytes // 32
        if torch.cuda.is_available():
            self._nf4_staging: Optional[torch.Tensor] = torch.empty(
                _max_nf4_bytes, dtype=torch.uint8, device=self.device
            )
            # Pre-allocated fp32 output buffer for dequantize_blockwise(absmax).
            # Without this, each large weight allocates ~54 MB from the pool for the
            # dequantized absmax. Repeated alloc/free fragments the 80 MB free pool
            # until no contiguous block remains and cudaMalloc crashes.
            self._absmax_staging: Optional[torch.Tensor] = torch.empty(
                _max_absmax_numel, dtype=torch.float32, device=self.device
            )
            # --- Zero-alloc QuantState staging ---
            # QuantState.from_dict(device=GPU) allocates ~15 MB of small GPU tensors
            # per weight (nested absmax uint8 + state2 float32 + code).  Over 7 weights
            # × 48 layers these fragment the CUDA pool → STATUS_ACCESS_VIOLATION.
            # Pre-allocating them here lets us create QuantState on CPU (fast dict
            # parsing, no GPU allocs) then memcpy the small tensors into these fixed
            # buffers before calling the GPU dequant kernels.
            self._nested_absmax_staging: Optional[torch.Tensor] = torch.empty(
                _max_absmax_numel, dtype=torch.uint8, device=self.device
            )  # ~13 MB for 405B — holds the doubly-quantized absmax bytes
            _max_s2_absmax = max(_max_absmax_numel // 64, 1024)
            self._state2_absmax_staging: Optional[torch.Tensor] = torch.empty(
                _max_s2_absmax, dtype=torch.float32, device=self.device
            )  # ~830 KB — holds secondary quantization scales
            self._code_staging: Optional[torch.Tensor] = torch.empty(
                256, dtype=torch.float32, device=self.device
            )  # 1 KB — dequantization codebook (same for all NF4 weights)
        else:
            self._nf4_staging = None
            self._absmax_staging = None
            self._nested_absmax_staging = None
            self._state2_absmax_staging = None
            self._code_staging = None
        # Try to pre-allocate MLP staging for dense calibration/collection paths.
        # Sparse-basis inference no longer depends on this buffer: uncovered
        # layers use the exact chunked 4-bit guard path instead.
        _ffn = int(getattr(self.config, "intermediate_size", _h * 4))
        self._mlp_proj_staging: Optional[torch.Tensor] = None
        self._mlp_proj_staging_numel: int = int(_ffn * _h)
        self._dense_mlp_staging_warned: bool = False
        # Skip the ~1.6 GB MLP staging buffer when sparse routing is active.
        # Sparse inference uses the learned-basis path on covered layers and the
        # exact chunked dense guard path on uncovered layers.
        _skip_mlp_staging = bool(sparse_basis_path and str(sparse_basis_path).strip())
        if torch.cuda.is_available() and not _skip_mlp_staging:
            try:
                self._mlp_proj_staging = torch.empty(
                    self._mlp_proj_staging_numel, dtype=self.dtype, device=self.device
                )
            except Exception as _e:
                if "out of memory" in str(_e).lower() or "alloc" in str(_e).lower():
                    import warnings
                    warnings.warn(
                        f"[StreamingLlamaRuntime] Insufficient VRAM for MLP staging "
                        f"({self._mlp_proj_staging_numel * 2 // 1024 // 1024} MB); "
                        "dense calibration paths may degrade to zero passthrough."
                    )
                    self._dense_mlp_staging_warned = True
                else:
                    raise

        # ── Sparse MLP routing weights ────────────────────────────────────────
        # Loaded from the learned-basis checkpoint produced by
        # init_learned_basis_from_dense_mlp.py. Kept in CPU RAM and moved to GPU
        # lazily when a layer is processed.
        #
        # For each sparse layer, the checkpoint stores:
        #   encoder_weight [basis_rank, hidden_size] — projects hidden → latent
        #   encoder_bias   [basis_rank]
        #   decoder_blocks [num_output_blocks, basis_rank, block_size] — reconstructs output blocks
        #
        # These blocks live in output space over hidden_size, not in
        # intermediate FFN space. Covered layers must stay on the learned-basis
        # reconstruction path; uncovered layers run the exact chunked 4-bit
        # dense guard path over the original FFN weights.
        self._sparse_routing: Dict[int, Dict[str, torch.Tensor]] = {}
        self._sparse_top_k: int = 0
        self._sparse_runtime_top_k: int = 0
        self._sparse_block_size: int = 32
        self._sparse_num_blocks: int = 0
        self._sparse_top_k_by_layer: Dict[int, int] = {}
        self._sparse_basis_top_k_by_layer: Dict[int, int] = {}
        self._sparse_semantic_block_score_normalized: bool = False
        self._sparse_param_cache: Dict[str, Dict[str, Any]] = {}
        self._sparse_explicit_layer_selection: Set[int] = set()
        self._sparse_checkpoint_basis_rank: int = 64
        self._upper_decode_guard_layers: Set[int] = set()
        self._session_sparse_route_layers: Set[int] = set()

        if sparse_basis_path and str(sparse_basis_path).strip():
            _payload = torch.load(str(sparse_basis_path), map_location="cpu", weights_only=False)
            _cfg = _payload.get("config", {})
            _raw_selection = _payload.get("layer_selection", [])
            if isinstance(_raw_selection, (list, tuple)):
                self._sparse_explicit_layer_selection = {
                    int(_idx) for _idx in _raw_selection if 0 <= int(_idx) < self.num_layers
                }
            self._sparse_block_size = int(_cfg.get("block_size", 32))
            self._sparse_num_blocks = int(
                _cfg.get("num_blocks", int(self.config.hidden_size) // self._sparse_block_size)
            )
            self._sparse_checkpoint_basis_rank = int(_cfg.get("basis_rank", self._sparse_checkpoint_basis_rank))
            # top_k: honour explicit override, else use 2 % of checkpoint blocks as default
            _num_blocks = max(1, int(self._sparse_num_blocks))
            _default_top_k = max(1, int(round(_num_blocks * 0.02)))
            self._sparse_top_k = int(sparse_top_k) if sparse_top_k is not None else _default_top_k
            self._sparse_runtime_top_k = int(self._sparse_top_k)
            _layer_states = _payload.get("layer_states", {})
            _stats = _payload.get("stats", {})
            _block_top_k_by_layer = _cfg.get("top_k_by_layer", {}) or {}
            _basis_top_k_by_layer = _cfg.get("basis_top_k_by_layer", {}) or {}
            _default_basis_top_k = int(_cfg.get("basis_top_k", _DEFAULT_SPARSE_BASIS_TOP_K))
            self._sparse_semantic_block_score_normalized = bool(
                _cfg.get("semantic_block_score_normalized", False)
            )
            _block_importance_by_layer = {}
            if isinstance(_stats, dict):
                _bib = _stats.get("block_importance_by_layer")
                if isinstance(_bib, dict):
                    _block_importance_by_layer = _bib
            _basis_device = self.device if self.device.type == "cuda" else torch.device("cpu")
            for _lidx_s, _state in _layer_states.items():
                _lidx = int(_lidx_s)
                if not (0 <= _lidx < self.num_layers):
                    continue
                _layer_num_blocks = int(_state["decoder_blocks"].shape[0])
                _expected_output_blocks = int(self.config.hidden_size) // int(self._sparse_block_size)
                if _layer_num_blocks != _expected_output_blocks:
                    raise RuntimeError(
                        f"Sparse basis layer {_lidx} stores {_layer_num_blocks} output blocks, "
                        f"expected {_expected_output_blocks}. The learned-basis artifact is defined "
                        "over hidden-size output blocks and cannot be mixed with FFN intermediate blocks."
                    )
                _basis_rank = int(_state["encoder_weight"].shape[0])
                _layer_top_k = int(
                    _block_top_k_by_layer.get(
                        str(_lidx),
                        _block_top_k_by_layer.get(_lidx, self._sparse_top_k),
                    )
                )
                _layer_top_k = int(max(1, min(_layer_top_k, _layer_num_blocks)))
                _basis_top_k = int(
                    _basis_top_k_by_layer.get(
                        str(_lidx),
                        _basis_top_k_by_layer.get(_lidx, _default_basis_top_k),
                    )
                )
                _basis_top_k = int(max(1, min(_basis_top_k, _basis_rank)))
                _dec = _state["decoder_blocks"].to(
                    device=_basis_device,
                    dtype=self.dtype,
                    non_blocking=False,
                ).contiguous()
                _dec_bias = _state.get("decoder_bias")
                if torch.is_tensor(_dec_bias):
                    _dec_bias = _dec_bias.to(
                        device=_basis_device,
                        dtype=self.dtype,
                        non_blocking=False,
                    ).contiguous()
                _scale = _state.get("scale")
                _dec_norm_t = _dec.norm(dim=-1).transpose(0, 1).contiguous()
                if self._sparse_semantic_block_score_normalized:
                    _dec_norm_t = F.normalize(_dec_norm_t, p=2.0, dim=-1, eps=1e-6)
                _dec_full_weight = None
                _dec_full_bias = None
                if self._sparse_basis_execution == "full_output_latent":
                    _dec_full_weight = _dec.permute(0, 2, 1).reshape(
                        _layer_num_blocks * int(self._sparse_block_size),
                        _basis_rank,
                    ).contiguous()
                    if _dec_bias is not None and self._sparse_basis_bias_mode != "none":
                        _dec_full_bias = _dec_bias.reshape(-1).contiguous()
                # decoder_blocks are output-space basis blocks over hidden_size, not
                # intermediate FFN blocks; keep them on the learned-basis path only.
                self._sparse_routing[_lidx] = {
                    "enc_w": _state["encoder_weight"].to(
                        device=_basis_device,
                        dtype=self.dtype,
                        non_blocking=False,
                    ).contiguous(),
                    "enc_b": _state["encoder_bias"].to(
                        device=_basis_device,
                        dtype=self.dtype,
                        non_blocking=False,
                    ).contiguous(),
                    "dec": _dec,
                    "dec_bias": _dec_bias,
                    "dec_full_weight": _dec_full_weight,
                    "dec_full_bias": _dec_full_bias,
                    "dec_norm_t": _dec_norm_t,
                    "scale": float(_scale.item()) if torch.is_tensor(_scale) and int(_scale.numel()) == 1 else 1.0,
                    "top_k": _layer_top_k,
                    "basis_top_k": _basis_top_k,
                }
                self._sparse_top_k_by_layer[_lidx] = int(_layer_top_k)
                self._sparse_basis_top_k_by_layer[_lidx] = int(_basis_top_k)
                hot_blocks = self._derive_hot_blocks_for_layer(
                    layer_state=_state,
                    layer_stats=_stats.get(str(_lidx), {}) if isinstance(_stats, dict) else {},
                    block_importance_by_layer=_block_importance_by_layer.get(str(_lidx), None),
                    layer_num_blocks=int(_state["decoder_blocks"].shape[0]),
                )
                if hot_blocks is not None and int(hot_blocks.numel()) > 0:
                    self._mlp_hot_blocks_by_layer[_lidx] = hot_blocks
            _pct = int(round(self._sparse_top_k * 100 / max(_num_blocks, 1)))
            _hot_layers = len(self._mlp_hot_blocks_by_layer)
            _hot_blocks_total = sum(int(v.numel()) for v in self._mlp_hot_blocks_by_layer.values())
            print(f"[sparse] loaded routing for {len(self._sparse_routing)}/{self.num_layers} layers "
                  f"| top_k={self._sparse_top_k}/{_num_blocks} blocks ({_pct}%) "
                  f"| block_size={self._sparse_block_size} "
                  f"| exec={self._sparse_basis_execution}", flush=True)
            if self._sparse_explicit_layer_selection:
                _upper_guard_start = max(self._sparse_explicit_layer_selection) + 1
                self._upper_decode_guard_layers = {
                    int(_idx)
                    for _idx in range(_upper_guard_start, int(self.num_layers))
                    if int(_idx) not in self._sparse_explicit_layer_selection
                }
                print(
                    f"[sparse] explicit layer selection: {len(self._sparse_explicit_layer_selection)}/{self.num_layers} "
                    f"MLP layers; non-selected layers use exact streamed dense guard execution",
                    flush=True,
                )
            if _hot_layers > 0:
                print(
                    f"[sparse] hot-block map ready for {_hot_layers} layers "
                    f"({int(_hot_blocks_total)} total blocks, threshold={self._hot_block_threshold:.2f})",
                    flush=True,
                )
            if self._vram_hot_cache_limit_bytes is not None:
                print(
                    f"[sparse] VRAM hot-cache budget: {self._vram_hot_cache_limit_bytes / (1024 ** 3):.2f} GB",
                    flush=True,
                )

        # ── Sparse Attention Head routing ─────────────────────────────────────
        # Loaded from the importance checkpoint produced by
        # init_learned_attn_head_importance.py.  At inference, only the top-K
        # heads' NF4 bytes for q_proj (row gather) and o_proj (column gather)
        # are transferred from CPU RAM to GPU, saving up to 75–87% of attention
        # PCIe bandwidth per token.
        #
        # _attn_active_head_indices[layer_idx] = Tensor[K] ranked by static
        # attention importance. _get_attn_active_heads sorts the live subset for
        # gather/scatter so reducing K later keeps the most important heads
        # instead of the lowest numeric head ids.
        # _attn_head_importance[layer_idx]     = Tensor[num_heads] mean norms (for
        #                                        dynamic re-ranking via Taylor state_z)
        self._attn_active_head_indices: Dict[int, torch.Tensor] = {}
        self._attn_head_importance: Dict[int, torch.Tensor] = {}
        self._attn_active_heads: int = 0
        self._attn_dynamic_threshold = float(max(0.0, min(1.0, attn_head_activity_threshold)))
        self._attn_min_active_heads = max(1, int(attn_min_active_heads))
        self._attn_max_active_heads = int(attn_max_active_heads) if attn_max_active_heads is not None else 0
        self._attn_runtime_head_counts: Dict[int, int] = {}
        self._attn_zero_only_layers: set[int] = set()
        self._attn_sparse_disabled_reason: Optional[str] = None
        # Metadata cache: code + dequantised absmax per weight (no packed bytes ?
        # raw bytes are fetched O(1) from loader._ram_cache on every call).
        self._attn_sparse_param_meta: Dict[str, Dict[str, Any]] = {}
        self._attn_hot_head_cache: Dict[int, Dict[str, Any]] = {}
        self._attn_loaded_q_rows: Optional[torch.Tensor] = None
        self._attn_loaded_o_cols: Optional[torch.Tensor] = None
        self._attn_qo_state: str = "unknown"
        self._cpu_scratch: Dict[str, torch.Tensor] = {}
        # GPU FP16 staging buffer for dequantised partial q_proj rows: [K*head_dim, hidden].
        self._attn_q_head_staging: Optional[torch.Tensor] = None
        self._attn_share_groups: Dict[str, Dict[str, Any]] = {}
        self._attn_share_layer_state: Dict[int, Dict[str, Any]] = {}
        self._attn_share_exact_layers: set[int] = set()

        # ── Sparse K/V routing ─────────────────────────────────────────────────
        self._kv_routing: Dict[int, Dict[str, Any]] = {}
        self._kv_sparse_top_k: int = 0
        self._kv_sparse_block_size: int = 32
        self._kv_num_col_blocks: int = 0
        self._kv_sparse_param_cache: Dict[str, Dict[str, Any]] = {}
        self._kv_hot_blocks_by_layer: Dict[int, torch.Tensor] = {}
        self._attn_kv_hot_block_cache: Dict[str, Any] = {}
        self._kv_block_usage_ema: Dict[int, torch.Tensor] = {}
        self._kv_block_usage_votes: Dict[int, torch.Tensor] = {}
        self._kv_block_banked_until: Dict[int, torch.Tensor] = {}
        self._kv_bank_step: int = 0
        self._kv_loaded_cols: Optional[torch.Tensor] = None

        if attn_head_importance_path and str(attn_head_importance_path).strip():
            _attn_payload = torch.load(
                str(attn_head_importance_path), map_location="cpu", weights_only=False
            )
            _attn_cfg = _attn_payload.get("config", {})
            _H   = int(_attn_cfg.get("num_heads",    getattr(self.config, "num_attention_heads", 128)))
            _D   = int(_attn_cfg.get("head_dim",     getattr(self.config, "head_dim", 128)))
            _Hid = int(_attn_cfg.get("hidden_size",  getattr(self.config, "hidden_size", 16384)))
            requested_heads = int(attn_active_heads) if attn_active_heads is not None else max(1, min(self._attn_min_active_heads, _H))
            _num_kv = int(_attn_cfg.get("num_kv_heads", getattr(self.config, "num_key_value_heads", max(1, _H // 16))))
            _per_head_weight_bytes = (_Hid * _D // 2) + (_Hid * _D // 64 * 4)
            _per_head_qo_mb = float(2 * _per_head_weight_bytes) / float(1024 ** 2)
            _kv_weight_bytes = (_num_kv * _D * _Hid // 2) + (_num_kv * _D * _Hid // 64 * 4)
            _kv_total_mb = float(2 * _kv_weight_bytes) / float(1024 ** 2)
            _budget_heads = max(
                1,
                int((float(self._target_layer_traffic_mb) - float(_kv_total_mb)) // max(_per_head_qo_mb, 1e-6)),
            )
            explicit_pool = self._attn_max_active_heads > 0
            if not explicit_pool:
                if attn_active_heads is not None:
                    # Without an explicit --attn-max-active-heads override, treat the
                    # requested value as the decode candidate pool size and keep the
                    # live per-token head count budget-clamped. This preserves a much
                    # richer Taylor/dynamic choice set than collapsing the whole pool
                    # down to the traffic budget itself.
                    self._attn_max_active_heads = int(requested_heads)
                else:
                    self._attn_max_active_heads = int(_budget_heads)
            self._attn_max_active_heads = max(1, min(self._attn_max_active_heads, _H))
            self._attn_min_active_heads = max(1, min(self._attn_min_active_heads, self._attn_max_active_heads))
            if attn_active_heads is not None:
                K = max(
                    1,
                    min(
                        self._attn_max_active_heads,
                        max(
                            min(self._attn_min_active_heads, self._attn_max_active_heads),
                            min(requested_heads, _budget_heads),
                        ),
                    ),
                )
            else:
                K = max(1, min(requested_heads, self._attn_max_active_heads))
            self._attn_active_heads = K

            for _lidx_s, _state in _attn_payload.get("layer_states", {}).items():
                _lidx = int(_lidx_s)
                if not (0 <= _lidx < self.num_layers):
                    continue
                imp = _state["importance"].float()          # [num_heads]
                self._attn_head_importance[_lidx] = imp
                top_k = torch.topk(imp, k=min(self._attn_max_active_heads, _H), largest=True).indices.contiguous()
                self._attn_active_head_indices[_lidx] = top_k

            if torch.cuda.is_available():
                # Pre-allocate FP16 buffer for dequantised partial q_proj rows.
                self._attn_q_head_staging = torch.empty(
                    self._attn_max_active_heads * _D * _Hid, dtype=self.dtype, device=self.device
                )

            _pct_a = int(round(K * 100 // max(_H, 1)))
            print(
                f"[sparse_attn] loaded importance for {len(self._attn_active_head_indices)}/{self.num_layers} layers "
                f"| active_heads={K}/{_H} ({_pct_a}%)"
                f" | min={self._attn_min_active_heads} max={self._attn_max_active_heads}"
                f" threshold={self._attn_dynamic_threshold:.2f}"
                f" | target_layer_mb={self._target_layer_traffic_mb:.1f}"
                f" | prefill_qo={self._sparse_attn_prefill_mode}"
                f" prefill_kv={self._sparse_kv_prefill_mode}",
                flush=True,
            )

        # ── Sparse K/V basis loading ───────────────────────────────────────────
        if attn_share_path and str(attn_share_path).strip():
            _share_payload = torch.load(str(attn_share_path), map_location="cpu", weights_only=False)
            _share_cfg = _share_payload.get("config", {})
            _hidden_size_cfg = int(_share_cfg.get("hidden_size", getattr(self.config, "hidden_size", 0)))
            _num_heads_cfg = int(_share_cfg.get("num_heads", getattr(self.config, "num_attention_heads", 0)))
            _num_kv_heads_cfg = int(_share_cfg.get("num_kv_heads", getattr(self.config, "num_key_value_heads", 0) or getattr(self.config, "num_attention_heads", 0)))
            _head_dim_cfg = int(_share_cfg.get("head_dim", getattr(self.config, "head_dim", 0)))
            _hidden_size_model = int(getattr(self.config, "hidden_size", 0))
            _num_heads_model = int(getattr(self.config, "num_attention_heads", 0))
            _num_kv_heads_model = int(getattr(self.config, "num_key_value_heads", 0) or _num_heads_model)
            _head_dim_model = int(getattr(self.config, "head_dim", 0))
            if (
                _hidden_size_cfg != _hidden_size_model
                or _num_heads_cfg != _num_heads_model
                or _num_kv_heads_cfg != _num_kv_heads_model
                or _head_dim_cfg != _head_dim_model
            ):
                raise RuntimeError(
                    "Attention-sharing checkpoint dimensions do not match the loaded model: "
                    f"artifact hidden/heads/kv_heads/head_dim={_hidden_size_cfg}/{_num_heads_cfg}/{_num_kv_heads_cfg}/{_head_dim_cfg}, "
                    f"model={_hidden_size_model}/{_num_heads_model}/{_num_kv_heads_model}/{_head_dim_model}."
                )
            _share_dtype = self.dtype
            for _gid_raw, _group_state in (_share_payload.get("group_states", {}) or {}).items():
                _gid = str(_gid_raw)
                _entry: Dict[str, Any] = {
                    "layers": tuple(int(v) for v in _group_state.get("layers", [])),
                    "sharing_format": str(_group_state.get("sharing_format", "matrix_v1")),
                }
                for _name in (
                    "q_base_u",
                    "q_base_v",
                    "o_base_u",
                    "o_base_v",
                    "k_base_u",
                    "k_base_v",
                    "v_base_u",
                    "v_base_v",
                    "q_base_u_heads",
                    "q_base_v_heads",
                    "o_base_u_heads",
                    "o_base_v_heads",
                    "k_base_u_heads",
                    "k_base_v_heads",
                    "v_base_u_heads",
                    "v_base_v_heads",
                ):
                    _tensor = _group_state.get(_name)
                    if torch.is_tensor(_tensor):
                        _entry[_name] = self.loader.prepare_h2d_source(
                            _tensor.to(dtype=_share_dtype).contiguous(),
                            dtype=_share_dtype,
                        )
                self._attn_share_groups[_gid] = _entry
            for _exact_layer in (_share_payload.get("exact_layers", []) or []):
                _exact_layer_idx = int(_exact_layer)
                if 0 <= _exact_layer_idx < self.num_layers:
                    self._attn_share_exact_layers.add(_exact_layer_idx)
            for _lidx_raw, _layer_state in (_share_payload.get("layer_states", {}) or {}).items():
                _lidx = int(_lidx_raw)
                if not (0 <= _lidx < self.num_layers):
                    continue
                _gid = str(_layer_state.get("group_id", ""))
                if _gid == "" or _gid not in self._attn_share_groups:
                    self._attn_share_exact_layers.add(_lidx)
                    continue
                _head_perm = torch.as_tensor(
                    _layer_state.get("head_perm", list(range(_num_heads_model))),
                    dtype=torch.long,
                    device=torch.device("cpu"),
                ).contiguous()
                if int(_head_perm.numel()) != _num_heads_model:
                    raise RuntimeError(
                        f"Attention-sharing layer {_lidx} head_perm has {int(_head_perm.numel())} entries; "
                        f"expected {_num_heads_model}."
                    )
                _kv_head_perm = torch.as_tensor(
                    _layer_state.get("kv_head_perm", list(range(_num_kv_heads_model))),
                    dtype=torch.long,
                    device=torch.device("cpu"),
                ).contiguous()
                if int(_kv_head_perm.numel()) != _num_kv_heads_model:
                    raise RuntimeError(
                        f"Attention-sharing layer {_lidx} kv_head_perm has {int(_kv_head_perm.numel())} entries; "
                        f"expected {_num_kv_heads_model}."
                    )
                _q_resid_u = _layer_state.get("q_resid_u")
                _q_resid_v = _layer_state.get("q_resid_v")
                _o_resid_u = _layer_state.get("o_resid_u")
                _o_resid_v = _layer_state.get("o_resid_v")
                _k_resid_u = _layer_state.get("k_resid_u")
                _k_resid_v = _layer_state.get("k_resid_v")
                _v_resid_u = _layer_state.get("v_resid_u")
                _v_resid_v = _layer_state.get("v_resid_v")
                self._attn_share_layer_state[_lidx] = {
                    "group_id": _gid,
                    "head_perm": _head_perm,
                    "kv_head_perm": _kv_head_perm,
                    "q_resid_u": None if not torch.is_tensor(_q_resid_u) else self.loader.prepare_h2d_source(
                        _q_resid_u.to(dtype=_share_dtype).contiguous(),
                        dtype=_share_dtype,
                    ),
                    "q_resid_v": None if not torch.is_tensor(_q_resid_v) else self.loader.prepare_h2d_source(
                        _q_resid_v.to(dtype=_share_dtype).contiguous(),
                        dtype=_share_dtype,
                    ),
                    "o_resid_u": None if not torch.is_tensor(_o_resid_u) else self.loader.prepare_h2d_source(
                        _o_resid_u.to(dtype=_share_dtype).contiguous(),
                        dtype=_share_dtype,
                    ),
                    "o_resid_v": None if not torch.is_tensor(_o_resid_v) else self.loader.prepare_h2d_source(
                        _o_resid_v.to(dtype=_share_dtype).contiguous(),
                        dtype=_share_dtype,
                    ),
                    "k_resid_u": None if not torch.is_tensor(_k_resid_u) else self.loader.prepare_h2d_source(
                        _k_resid_u.to(dtype=_share_dtype).contiguous(),
                        dtype=_share_dtype,
                    ),
                    "k_resid_v": None if not torch.is_tensor(_k_resid_v) else self.loader.prepare_h2d_source(
                        _k_resid_v.to(dtype=_share_dtype).contiguous(),
                        dtype=_share_dtype,
                    ),
                    "v_resid_u": None if not torch.is_tensor(_v_resid_u) else self.loader.prepare_h2d_source(
                        _v_resid_u.to(dtype=_share_dtype).contiguous(),
                        dtype=_share_dtype,
                    ),
                    "v_resid_v": None if not torch.is_tensor(_v_resid_v) else self.loader.prepare_h2d_source(
                        _v_resid_v.to(dtype=_share_dtype).contiguous(),
                        dtype=_share_dtype,
                    ),
                }
                for _name in (
                    "q_resid_u_heads",
                    "q_resid_v_heads",
                    "o_resid_u_heads",
                    "o_resid_v_heads",
                    "k_resid_u_heads",
                    "k_resid_v_heads",
                    "v_resid_u_heads",
                    "v_resid_v_heads",
                ):
                    _tensor = _layer_state.get(_name)
                    self._attn_share_layer_state[_lidx][_name] = (
                        None
                        if not torch.is_tensor(_tensor)
                        else self.loader.prepare_h2d_source(
                            _tensor.to(dtype=_share_dtype).contiguous(),
                            dtype=_share_dtype,
                        )
                    )
            if self._attn_share_layer_state:
                print(
                    f"[attn_share] loaded q/o sharing for {len(self._attn_share_layer_state)}/{self.num_layers} layers "
                    f"| groups={len(self._attn_share_groups)} | exact={len(self._attn_share_exact_layers)} "
                    f"| prefill={self._attn_share_prefill_mode}",
                    flush=True,
                )

        if kv_basis_path and str(kv_basis_path).strip():
            _kv_payload = torch.load(str(kv_basis_path), map_location="cpu", weights_only=False)
            _kv_cfg = _kv_payload.get("config", {})
            self._kv_sparse_block_size = int(_kv_cfg.get("block_size", 32))
            _kv_hidden_size = int(_kv_cfg.get("hidden_size", getattr(self.config, "hidden_size", 16384)))
            self._kv_num_col_blocks = _kv_hidden_size // self._kv_sparse_block_size
            _kv_default_top_k = max(1, int(round(self._kv_num_col_blocks * 0.10)))
            self._kv_sparse_top_k = int(kv_sparse_top_k) if kv_sparse_top_k is not None else _kv_default_top_k
            _kv_basis_device = self.device if self.device.type == "cuda" else torch.device("cpu")
            for _lidx_s, _lstate in _kv_payload.get("layer_states", {}).items():
                _lidx = int(_lidx_s)
                if not (0 <= _lidx < self.num_layers):
                    continue
                _dec = _lstate["decoder_blocks"].to(device=_kv_basis_device, dtype=self.dtype).contiguous()
                _dec_norm_t = _dec.norm(dim=-1).transpose(0, 1).contiguous()
                self._kv_routing[_lidx] = {
                    "enc_w": _lstate["encoder_weight"].to(device=_kv_basis_device, dtype=self.dtype).contiguous(),
                    "enc_b": _lstate["encoder_bias"].to(device=_kv_basis_device, dtype=self.dtype).contiguous(),
                    "dec": _dec,
                    "dec_norm_t": _dec_norm_t,
                    "top_k": self._kv_sparse_top_k,
                }
                _blk_imp = _lstate.get("block_importance")
                if torch.is_tensor(_blk_imp):
                    _sorted_imp, _sorted_idx = _blk_imp.float().sort(descending=True)
                    _hot_k = max(1, int(round(self._kv_num_col_blocks * 0.10)))
                    self._kv_hot_blocks_by_layer[_lidx] = _sorted_idx[:_hot_k].contiguous()
                    self._kv_block_usage_ema[_lidx] = _blk_imp.float().clone()
                else:
                    self._kv_block_usage_ema[_lidx] = torch.zeros(self._kv_num_col_blocks, dtype=torch.float32)
                self._kv_block_usage_votes[_lidx] = torch.zeros(self._kv_num_col_blocks, dtype=torch.int32)
                self._kv_block_banked_until[_lidx] = torch.zeros(self._kv_num_col_blocks, dtype=torch.int32)
            if self._show_progress:
                print(
                    f"[kv_sparse] Loaded KV routing for {len(self._kv_routing)} layers, "
                    f"top_k={self._kv_sparse_top_k}, num_col_blocks={self._kv_num_col_blocks}",
                    flush=True,
                )

    def reset_caches(self) -> None:
        self._taylor_caches = [None for _ in range(self.num_layers)]
        self._dense_cache = DynamicCache(config=self.config) if DynamicCache is not None else None
        if self._token_archive is not None:
            self._token_archive.reset()
        for _layer_idx in list(self._session_sparse_route_layers):
            self._sparse_routing.pop(int(_layer_idx), None)
            self._sparse_top_k_by_layer.pop(int(_layer_idx), None)
            self._sparse_basis_top_k_by_layer.pop(int(_layer_idx), None)
            self._mlp_hot_blocks_by_layer.pop(int(_layer_idx), None)
        self._session_sparse_route_layers.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear_session_state(self) -> None:
        self._session_token_ids_cpu = None
        self._session_last_logits_cpu = None
        self.reset_caches()

    @staticmethod
    def _longest_common_prefix_len(prev_ids: torch.LongTensor, new_ids: torch.LongTensor) -> int:
        if prev_ids.ndim != 2 or new_ids.ndim != 2:
            return 0
        if int(prev_ids.shape[0]) != 1 or int(new_ids.shape[0]) != 1:
            return 0
        limit = min(int(prev_ids.shape[1]), int(new_ids.shape[1]))
        if limit <= 0:
            return 0
        prev_flat = prev_ids[0, :limit].to(device=torch.device("cpu"), dtype=torch.long)
        new_flat = new_ids[0, :limit].to(device=torch.device("cpu"), dtype=torch.long)
        mismatch = (prev_flat != new_flat).nonzero(as_tuple=False)
        if int(mismatch.numel()) == 0:
            return int(limit)
        return int(mismatch[0].item())

    def _crop_attention_caches(self, max_length: int) -> bool:
        target = max(0, int(max_length))
        if self.taylor_layer_set:
            return False
        if self._dense_cache is None:
            return target == 0
        if hasattr(self._dense_cache, "crop"):
            self._dense_cache.crop(target)
            return True
        return False

    def _release_dense_cache_for_retrieval_layers(self) -> None:
        if self._dense_cache is None or not self._retrieval_layers:
            return
        key_cache = getattr(self._dense_cache, "key_cache", None)
        value_cache = getattr(self._dense_cache, "value_cache", None)
        if not isinstance(key_cache, list) or not isinstance(value_cache, list):
            return
        released = 0
        for layer_idx in sorted(int(v) for v in self._retrieval_layers):
            if 0 <= layer_idx < len(key_cache):
                key_cache[layer_idx] = None
                value_cache[layer_idx] = None
                released += 1
        if released > 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _set_session_state(self, token_ids_cpu: torch.LongTensor, logits: Optional[torch.Tensor]) -> None:
        self._session_token_ids_cpu = token_ids_cpu.detach().to(device=torch.device("cpu"), dtype=torch.long).contiguous()
        if logits is None:
            self._session_last_logits_cpu = None
        else:
            self._session_last_logits_cpu = logits[:, -1:, :].detach().to(
                device=torch.device("cpu"),
                dtype=torch.float32,
            ).contiguous()

    def _ensure_lm_head_weight_cpu(self) -> Optional[torch.Tensor]:
        if not self._materialize_lm_head:
            return None
        if self._lm_head_weight_cpu is not None:
            return self._lm_head_weight_cpu
        weight = self.loader.load_parameter(self._lm_head_weight_name).to(dtype=self.dtype)
        self._lm_head_weight_cpu = weight
        if self._lm_head_weight_name == self._embed_weight_name:
            self._embed_weight_cpu = weight
        return self._lm_head_weight_cpu

    def _estimate_sparse_gpu_working_set_bytes(self) -> int:
        if not self._sparse_routing:
            return 0
        hidden_size = int(getattr(self.config, "hidden_size", 0))
        block_size = max(1, int(self._sparse_block_size))
        top_k = max(1, int(self._sparse_runtime_top_k or self._sparse_top_k or 1))
        quant_block_size = 64
        gate_or_up = top_k * (
            (hidden_size // 2) * block_size
            + (hidden_size // quant_block_size) * block_size * 4
        )
        down = top_k * (hidden_size * (block_size // 2) + hidden_size * 4)
        return int((gate_or_up * 2 + down) * 2)

    def _gpu_lm_head_reserve_bytes(self) -> int:
        safety_margin_bytes = int(256 * (1024 ** 2))
        hot_cache_reserve_bytes = int(self._vram_hot_cache_limit_bytes or 0)
        sparse_reserve_bytes = self._estimate_sparse_gpu_working_set_bytes()
        return int(max(safety_margin_bytes, self._vram_hot_cache_margin_bytes)) + hot_cache_reserve_bytes + sparse_reserve_bytes

    def _materialize_lm_head_on_gpu(self) -> None:
        if self._lm_head_gpu_attempted:
            return
        if not self._materialize_lm_head:
            return
        if not self._prefer_gpu_lm_head:
            self._lm_head_gpu_attempted = True
            return
        if self.device.type != "cuda":
            self._lm_head_gpu_attempted = True
            return
        if _is_windows_pre_ampere_cuda(self.device):
            self._lm_head_gpu_attempted = True
            return
        lm_head_weight_cpu = self._ensure_lm_head_weight_cpu()
        if lm_head_weight_cpu is None:
            return
        if self._lm_head_weight_gpu is not None:
            return
        self._lm_head_gpu_attempted = True

        required_bytes = int(lm_head_weight_cpu.numel()) * int(lm_head_weight_cpu.element_size())
        required_residual_bytes = self._gpu_lm_head_reserve_bytes()
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        except Exception:
            free_bytes, total_bytes = 0, 0
        if free_bytes > 0 and (int(free_bytes) - int(required_bytes)) < int(required_residual_bytes):
            print(
                f"[lm_head] staying on CPU; free={float(free_bytes) / (1024 ** 3):.2f} GB "
                f"required={float(required_bytes) / (1024 ** 3):.2f} GB "
                f"reserve={float(required_residual_bytes) / (1024 ** 3):.2f} GB",
                flush=True,
            )
            return

        try:
            torch.cuda.empty_cache()
            self._lm_head_weight_gpu = lm_head_weight_cpu.to(
                device=self.device,
                dtype=self.dtype,
                non_blocking=False,
            ).contiguous()
            if self._lm_head_weight_gpu.device.type == "cuda":
                try:
                    gpu_free_bytes, gpu_total_bytes = torch.cuda.mem_get_info(self.device)
                    print(
                        f"[lm_head] resident on GPU: "
                        f"{float(required_bytes) / (1024 ** 3):.2f} GB "
                        f"(free {float(gpu_free_bytes) / (1024 ** 3):.2f} / {float(gpu_total_bytes) / (1024 ** 3):.2f} GB)",
                        flush=True,
                    )
                except Exception:
                    print(
                        f"[lm_head] resident on GPU: {float(required_bytes) / (1024 ** 3):.2f} GB",
                        flush=True,
                    )
        except Exception as exc:
            self._lm_head_weight_gpu = None
            if _is_cuda_oom_error(exc):
                print(
                    f"[lm_head] GPU materialization failed; keeping CPU path: {type(exc).__name__}: {str(exc)[:200]}",
                    flush=True,
                )
            else:
                raise

    def _lm_head_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self._lm_head_weight_gpu is None:
            self._materialize_lm_head_on_gpu()
        if self._lm_head_weight_gpu is not None:
            return F.linear(hidden.to(dtype=self._lm_head_weight_gpu.dtype), self._lm_head_weight_gpu)
        lm_head_weight_cpu = self._ensure_lm_head_weight_cpu()
        if lm_head_weight_cpu is None:
            raise RuntimeError("StreamingLlamaRuntime.generate requires materialize_lm_head=True")
        hidden_cpu = hidden.to(device=torch.device("cpu"), dtype=lm_head_weight_cpu.dtype)
        return F.linear(hidden_cpu, lm_head_weight_cpu)

    def _reset_traffic_stats(self) -> None:
        self._traffic_current_phase = "idle"
        self._traffic_bytes_by_phase.clear()
        self._traffic_layer_visits_by_phase.clear()
        self._traffic_bytes_by_phase_layer.clear()
        self._traffic_layer_visits_by_phase_layer.clear()
        self._traffic_bytes_by_phase_tag.clear()
        self._last_traffic_report = None

    def _set_traffic_phase(self, phase: str) -> None:
        self._traffic_current_phase = str(phase or "other")

    def _record_layer_visit(self, layer_idx: int) -> None:
        phase = str(self._traffic_current_phase or "other")
        key = (phase, int(layer_idx))
        self._traffic_layer_visits_by_phase[phase] += 1
        self._traffic_layer_visits_by_phase_layer[key] += 1

    def _record_h2d_bytes(
        self,
        num_bytes: int,
        *,
        layer_idx: Optional[int],
        tag: str,
    ) -> None:
        if int(num_bytes) <= 0:
            return
        phase = str(self._traffic_current_phase or "other")
        self._traffic_bytes_by_phase[phase] += int(num_bytes)
        self._traffic_bytes_by_phase_tag[(phase, str(tag))] += int(num_bytes)
        if layer_idx is not None:
            self._traffic_bytes_by_phase_layer[(phase, int(layer_idx))] += int(num_bytes)

    @staticmethod
    def _build_phase_traffic_report(
        phase: str,
        *,
        bytes_by_phase: Dict[str, int],
        layer_visits_by_phase: Dict[str, int],
        bytes_by_phase_layer: Dict[Tuple[str, int], int],
        layer_visits_by_phase_layer: Dict[Tuple[str, int], int],
        bytes_by_phase_tag: Dict[Tuple[str, str], int],
    ) -> Dict[str, Any]:
        total_bytes = int(bytes_by_phase.get(phase, 0))
        layer_visits = int(layer_visits_by_phase.get(phase, 0))
        avg_bytes = float(total_bytes) / float(max(layer_visits, 1))
        layer_avgs: Dict[str, float] = {}
        for (phase_name, layer_idx), byte_count in bytes_by_phase_layer.items():
            if phase_name != phase:
                continue
            visits = int(layer_visits_by_phase_layer.get((phase_name, layer_idx), 0))
            layer_avgs[str(layer_idx)] = float(byte_count) / float(max(visits, 1)) / float(1024 ** 2)
        tag_totals: Dict[str, float] = {}
        for (phase_name, tag), byte_count in bytes_by_phase_tag.items():
            if phase_name != phase:
                continue
            tag_totals[str(tag)] = float(byte_count) / float(1024 ** 2)
        return {
            "phase": str(phase),
            "total_bytes": total_bytes,
            "total_mb": float(total_bytes) / float(1024 ** 2),
            "layer_visits": layer_visits,
            "avg_bytes_per_layer": avg_bytes,
            "avg_mb_per_layer": avg_bytes / float(1024 ** 2),
            "layer_avg_mb": dict(sorted(layer_avgs.items(), key=lambda item: int(item[0]))),
            "tag_totals_mb": dict(sorted(tag_totals.items())),
        }

    def _finalize_traffic_report(self) -> None:
        phases = ["prefill", "decode"]
        total_bytes = sum(int(self._traffic_bytes_by_phase.get(phase, 0)) for phase in phases)
        total_layer_visits = sum(int(self._traffic_layer_visits_by_phase.get(phase, 0)) for phase in phases)
        self._last_traffic_report = {
            "prefill": self._build_phase_traffic_report(
                "prefill",
                bytes_by_phase=self._traffic_bytes_by_phase,
                layer_visits_by_phase=self._traffic_layer_visits_by_phase,
                bytes_by_phase_layer=self._traffic_bytes_by_phase_layer,
                layer_visits_by_phase_layer=self._traffic_layer_visits_by_phase_layer,
                bytes_by_phase_tag=self._traffic_bytes_by_phase_tag,
            ),
            "decode": self._build_phase_traffic_report(
                "decode",
                bytes_by_phase=self._traffic_bytes_by_phase,
                layer_visits_by_phase=self._traffic_layer_visits_by_phase,
                bytes_by_phase_layer=self._traffic_bytes_by_phase_layer,
                layer_visits_by_phase_layer=self._traffic_layer_visits_by_phase_layer,
                bytes_by_phase_tag=self._traffic_bytes_by_phase_tag,
            ),
            "overall": {
                "total_bytes": int(total_bytes),
                "total_mb": float(total_bytes) / float(1024 ** 2),
                "layer_visits": int(total_layer_visits),
                "avg_bytes_per_layer": float(total_bytes) / float(max(total_layer_visits, 1)),
                "avg_mb_per_layer": float(total_bytes) / float(max(total_layer_visits, 1)) / float(1024 ** 2),
            },
        }

    def get_last_traffic_report(self) -> Optional[Dict[str, Any]]:
        return self._last_traffic_report

    def _schedule_prefetch_layer(self, layer_idx: int) -> None:
        if self._prefetch_executor is None:
            return
        if not self.loader._ram_cache_enabled:
            return
        if not self._layer_param_keys:
            return
        idx = int(layer_idx)
        if idx < 0 or idx >= self.num_layers:
            return
        with self._prefetch_lock:
            if idx in self._prefetch_pending_layers:
                return
            self._prefetch_pending_layers.add(idx)
        self._prefetch_executor.submit(self._prefetch_layer, idx)

    def _prefetch_layer(self, layer_idx: int) -> None:
        """Warm the RAM cache for *layer_idx* from disk on a background thread.

        Only loads parameters that are not yet cached — safe to call concurrently
        with the GPU forward pass because it touches only CPU memory and the
        thread-safe RAM cache dict (protected by _ram_cache_lock).
        """
        try:
            if not self.loader._ram_cache_enabled:
                return
            if not self._layer_param_keys:
                return
            prefix = f"model.layers.{int(layer_idx)}."
            for k in self._layer_param_keys:
                full_name = f"{prefix}{k}"
                with self.loader._ram_cache_lock:
                    already = full_name in self.loader._ram_cache
                if not already:
                    try:
                        self.loader._load_raw_for_param(full_name)
                    except Exception:
                        pass  # best-effort; main thread will retry
        finally:
            with self._prefetch_lock:
                self._prefetch_pending_layers.discard(int(layer_idx))

    @staticmethod
    def _coerce_block_score_vector(value: Any, *, num_blocks: int) -> Optional[torch.Tensor]:
        if value is None:
            return None
        vec: Optional[torch.Tensor] = None
        if torch.is_tensor(value):
            vec = value.detach().flatten().to(dtype=torch.float32, device="cpu")
        elif isinstance(value, (list, tuple)):
            if len(value) == 0:
                return None
            vec = torch.as_tensor(value, dtype=torch.float32).flatten().to(device="cpu")
        elif isinstance(value, dict):
            for key in (
                "block_importance",
                "block_importance_probs",
                "_block_importance_probs",
                "block_scores",
                "importance",
                "scores",
            ):
                if key in value:
                    nested = StreamingLlamaRuntime._coerce_block_score_vector(value[key], num_blocks=num_blocks)
                    if nested is not None:
                        return nested
            tmp = torch.zeros((num_blocks,), dtype=torch.float32)
            found = False
            for key, raw_val in value.items():
                try:
                    idx = int(key)
                except Exception:
                    continue
                if not (0 <= idx < num_blocks):
                    continue
                try:
                    tmp[idx] = float(raw_val)
                    found = True
                except Exception:
                    continue
            if found:
                vec = tmp
        if vec is None:
            return None
        if vec.numel() >= num_blocks:
            return vec[:num_blocks].contiguous()
        padded = torch.zeros((num_blocks,), dtype=torch.float32)
        padded[: int(vec.numel())] = vec
        return padded

    def _derive_hot_blocks_for_layer(
        self,
        *,
        layer_state: Dict[str, Any],
        layer_stats: Any,
        block_importance_by_layer: Any,
        layer_num_blocks: int,
    ) -> Optional[torch.Tensor]:
        candidates = [
            layer_state.get("block_importance"),
            layer_state.get("block_importance_probs"),
            layer_state.get("_block_importance_probs"),
            layer_stats,
            block_importance_by_layer,
        ]
        scores: Optional[torch.Tensor] = None
        for candidate in candidates:
            vec = self._coerce_block_score_vector(candidate, num_blocks=layer_num_blocks)
            if vec is not None and int(vec.numel()) == int(layer_num_blocks):
                scores = vec
                break

        if scores is None:
            decoder_bias = layer_state.get("decoder_bias")
            if torch.is_tensor(decoder_bias) and decoder_bias.ndim == 2:
                scores = decoder_bias.detach().abs().mean(dim=-1).to(dtype=torch.float32, device="cpu")
            else:
                decoder_blocks = layer_state.get("decoder_blocks")
                if torch.is_tensor(decoder_blocks) and decoder_blocks.ndim >= 2:
                    reduce_dims = tuple(range(1, decoder_blocks.ndim))
                    scores = decoder_blocks.detach().abs().mean(dim=reduce_dims).to(dtype=torch.float32, device="cpu")

        if scores is None or int(scores.numel()) == 0:
            return None

        scores = scores[:layer_num_blocks].to(dtype=torch.float32, device="cpu").contiguous()
        s_max = float(scores.max().item()) if scores.numel() > 0 else 0.0
        if s_max <= 0.0:
            hot_count = max(1, min(int(self._sparse_top_k), int(layer_num_blocks)))
            return torch.arange(hot_count, dtype=torch.long)

        norm_scores = scores / max(s_max, 1e-8)
        hot_idx = torch.nonzero(norm_scores >= self._hot_block_threshold, as_tuple=False).flatten().to(dtype=torch.long)
        if hot_idx.numel() == 0:
            hot_count = max(1, min(int(self._sparse_top_k), int(layer_num_blocks)))
            hot_idx = torch.topk(norm_scores, k=hot_count, largest=True).indices.to(dtype=torch.long)
        return hot_idx.sort().values.contiguous()

    def _order_blocks_for_layer_hot_cache(self, layer_idx: int, cpu_blocks: torch.Tensor) -> torch.Tensor:
        hot = self._mlp_hot_blocks_by_layer.get(int(layer_idx))
        if hot is None or int(hot.numel()) == 0 or int(cpu_blocks.numel()) <= 1:
            return cpu_blocks
        hot_set = {int(v) for v in hot.tolist()}
        ordered = [int(v) for v in cpu_blocks.tolist() if int(v) in hot_set]
        ordered.extend(int(v) for v in cpu_blocks.tolist() if int(v) not in hot_set)
        return torch.tensor(ordered, dtype=cpu_blocks.dtype, device=cpu_blocks.device)

    def _disable_vram_hot_cache(self, reason: str) -> None:
        self._vram_hot_cache_enabled = False
        self._vram_hot_cache_disable_reason = str(reason)
        self._vram_hot_cache_limit_bytes = 0
        self._vram_hot_cache_used_bytes = 0
        self._vram_nf4_cache.clear()
        self._attn_hot_head_cache.clear()
        for _param in self._sparse_param_cache.values():
            if isinstance(_param, dict):
                _param.pop("vram_hot", None)
                _param.pop("vram_hot_down", None)
        if self.device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        if not self._vram_hot_cache_oom_warned:
            self._vram_hot_cache_oom_warned = True
            print(
                f"[sparse] VRAM hot-cache disabled ({self._vram_hot_cache_disable_reason}); "
                "continuing with direct RAM->GPU sparse block streaming.",
                flush=True,
            )

    def _can_reserve_vram_hot_cache(self, required_bytes: int) -> bool:
        req = int(max(required_bytes, 0))
        if req <= 0:
            return True

        hard_limit = self._vram_hot_cache_limit_bytes
        if hard_limit is not None and self._vram_hot_cache_used_bytes + req > hard_limit:
            return False

        if self.device.type != "cuda":
            return True

        try:
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
        except Exception:
            return True

        # Effective limit is the tighter of:
        # 1) user/config budget, and
        # 2) currently free VRAM minus safety margin.
        dynamic_limit = self._vram_hot_cache_used_bytes + max(
            int(free_bytes) - int(self._vram_hot_cache_margin_bytes), 0
        )
        effective_limit = dynamic_limit
        if hard_limit is not None:
            effective_limit = min(int(hard_limit), int(dynamic_limit))

        if self._vram_hot_cache_used_bytes + req <= effective_limit:
            return True

        if not self._vram_hot_cache_pressure_warned:
            self._vram_hot_cache_pressure_warned = True
            print(
                "[sparse] VRAM hot-cache auto-clamp engaged: "
                f"used={self._vram_hot_cache_used_bytes / (1024 ** 3):.2f} GB, "
                f"free={int(free_bytes) / (1024 ** 3):.2f} GB, "
                f"margin={self._vram_hot_cache_margin_bytes / (1024 ** 3):.2f} GB.",
                flush=True,
            )
        return False

    def _maybe_cache_down_proj_hot_columns(
        self,
        *,
        full_name: str,
        param: Dict[str, Any],
        hot_blocks: torch.Tensor,
    ) -> bool:
        block_size = int(self._sparse_block_size)
        in_features = int(param["in_features"])
        out_features = int(param["out_features"])
        quant_block_size = int(param["quant_block_size"])
        total_col_blocks = in_features // max(block_size, 1)
        hot_blocks = hot_blocks[(hot_blocks >= 0) & (hot_blocks < total_col_blocks)].to(dtype=torch.long, device="cpu")
        hot_blocks = hot_blocks.unique(sorted=True)
        if int(hot_blocks.numel()) == 0:
            return False

        bytes_per_row = in_features // 2
        bytes_per_cblk = block_size // 2
        absmax_per_row = in_features // max(quant_block_size, 1)
        required_bytes = int(out_features * int(hot_blocks.numel()) * bytes_per_cblk)
        required_bytes += int(out_features * int(hot_blocks.numel()) * 4)
        if not self._can_reserve_vram_hot_cache(required_bytes):
            return False

        raw_2d = param["packed_weight"].view(out_features, bytes_per_row)
        col_starts = hot_blocks * bytes_per_cblk
        col_range = torch.arange(bytes_per_cblk, dtype=torch.long)
        col_idx = (col_starts.unsqueeze(-1) + col_range.unsqueeze(0)).reshape(-1)
        packed_cols_cpu = raw_2d[:, col_idx].reshape(out_features, int(hot_blocks.numel()), bytes_per_cblk).contiguous()

        absmax_2d = param["absmax"].view(out_features, absmax_per_row)
        abs_idx = (hot_blocks * block_size) // quant_block_size
        absmax_cols_cpu = absmax_2d[:, abs_idx].contiguous()

        try:
            packed_cols_gpu = self._copy_cpu_to_gpu(packed_cols_cpu, dtype=torch.uint8)
            absmax_cols_gpu = self._copy_cpu_to_gpu(absmax_cols_cpu, dtype=torch.float32)
        except Exception as _e:
            if _is_cuda_oom_error(_e):
                self._disable_vram_hot_cache("cuda_oom_during_down_hot_cache")
            return False

        lookup = torch.full((total_col_blocks,), -1, dtype=torch.int32)
        lookup[hot_blocks] = torch.arange(int(hot_blocks.numel()), dtype=torch.int32)
        hot_cache = {
            "block_ids_cpu": hot_blocks,
            "lookup_cpu": lookup,
            "packed_cols_gpu": packed_cols_gpu,
            "absmax_cols_gpu": absmax_cols_gpu,
        }
        param["vram_hot_down"] = hot_cache
        self._vram_nf4_cache[full_name] = hot_cache
        self._vram_hot_cache_used_bytes += required_bytes
        return True

    def _maybe_cache_sparse_param_hot_blocks(self, full_name: str, param: Dict[str, Any]) -> None:
        if full_name in self._vram_nf4_cache:
            return
        if str(self._traffic_current_phase or "idle") not in {"decode", "prefill"}:
            return
        if not self._vram_hot_cache_enabled:
            return
        if self.device.type != "cuda":
            return
        parts = str(full_name).split(".")
        if len(parts) < 4 or parts[0] != "model" or parts[1] != "layers":
            return
        try:
            layer_idx = int(parts[2])
        except Exception:
            return
        hot_blocks = self._mlp_hot_blocks_by_layer.get(layer_idx)
        if hot_blocks is None or int(hot_blocks.numel()) == 0:
            return
        if str(full_name).endswith(".mlp.down_proj.weight"):
            self._maybe_cache_down_proj_hot_columns(full_name=full_name, param=param, hot_blocks=hot_blocks)
            return

        total_blocks = int(param["packed_blocks"].shape[0])
        hot_blocks = hot_blocks[(hot_blocks >= 0) & (hot_blocks < total_blocks)].to(dtype=torch.long, device="cpu")
        hot_blocks = hot_blocks.unique(sorted=True)
        if int(hot_blocks.numel()) == 0:
            return

        packed_block_bytes = int(param["packed_blocks"].shape[1])
        absmax_block_bytes = int(param["absmax_blocks"].shape[1]) * 4
        required_bytes = int(hot_blocks.numel()) * (packed_block_bytes + absmax_block_bytes)
        if not self._can_reserve_vram_hot_cache(required_bytes):
            return

        try:
            packed_hot_gpu = self._copy_cpu_to_gpu(
                param["packed_blocks"][hot_blocks].contiguous(), dtype=torch.uint8
            )
            absmax_hot_gpu = self._copy_cpu_to_gpu(
                param["absmax_blocks"][hot_blocks].contiguous(), dtype=torch.float32
            )
        except Exception as _e:
            if _is_cuda_oom_error(_e):
                self._disable_vram_hot_cache("cuda_oom_during_hot_block_cache")
            return
        lookup = torch.full((total_blocks,), -1, dtype=torch.int32)
        lookup[hot_blocks] = torch.arange(int(hot_blocks.numel()), dtype=torch.int32)
        hot_cache = {
            "block_ids_cpu": hot_blocks,
            "lookup_cpu": lookup,
            "packed_blocks_gpu": packed_hot_gpu,
            "absmax_blocks_gpu": absmax_hot_gpu,
        }
        param["vram_hot"] = hot_cache
        self._vram_nf4_cache[full_name] = hot_cache
        self._vram_hot_cache_used_bytes += required_bytes

    # ── Sparse K/V param cache ────────────────────────────────────────────────

    def _get_sparse_4bit_kv_param(self, full_name: str) -> Dict[str, Any]:
        """Preprocess a K or V projection NF4 weight into column-block layout for sparse loading."""
        cached = self._kv_sparse_param_cache.get(full_name)
        if cached is not None:
            return cached

        raw_weight, quant_aux = self.loader._load_raw_for_param(full_name)
        if not quant_aux:
            raise RuntimeError(f"Sparse 4-bit KV path expected quantized weights for {full_name}")

        quant_state = bnb_functional.QuantState.from_dict(quant_aux, device=torch.device("cpu"))
        absmax = quant_state.absmax
        if quant_state.nested:
            absmax = bnb_functional.dequantize_blockwise(quant_state.absmax, quant_state.state2)
            absmax = absmax + quant_state.offset

        kv_hidden = int(quant_state.shape[0])    # e.g. 1024 (output dim of k_proj)
        hidden_size = int(quant_state.shape[1])  # e.g. 16384 (input dim)
        block_size = self._kv_sparse_block_size
        num_col_blocks = hidden_size // block_size
        quant_block_size = int(quant_state.blocksize)

        # packed_weight is [kv_hidden * hidden_size // 2] uint8 bytes
        bytes_per_col_step = block_size // 2  # bytes per column-block per row
        raw_2d = raw_weight.view(kv_hidden, hidden_size // 2)
        # col_blocks_packed: [num_col_blocks, kv_hidden * bytes_per_col_step]
        col_blocks_packed = (
            raw_2d.view(kv_hidden, num_col_blocks, bytes_per_col_step)
            .permute(1, 0, 2)
            .reshape(num_col_blocks, kv_hidden * bytes_per_col_step)
            .contiguous()
        )

        # Reshape absmax into column-block layout.
        # absmax is [kv_hidden * hidden_size // quant_block_size] float32.
        # Two cases:
        #   block_size >= quant_block_size: multiple quant-blocks per col-block,
        #       abs_per_col_block = block_size // quant_block_size, dequant uses quant_block_size.
        #   block_size < quant_block_size: one quant-block spans multiple col-blocks (e.g. 64 > 32),
        #       each quant-block absmax must be repeated for every col-block it spans,
        #       abs_per_col_block = 1, dequant uses block_size as the effective blocksize.
        abs_per_row = hidden_size // quant_block_size
        absmax_2d = absmax.to(dtype=torch.float32).view(kv_hidden, abs_per_row)
        if block_size < quant_block_size:
            # block_size (32) < quant_block_size (64): one quant-block spans multiple col-blocks.
            # bnb CUDA dequant requires blocksize >= 64, so we must dequant at quant_block_size (64).
            # Store absmax at quant-block granularity (not repeated per col-block).
            _qb_per_cb = quant_block_size // block_size  # e.g. 2
            # qblock_absmax: [num_quant_blocks, kv_hidden] — one absmax per 64-feature quant block per row
            qblock_absmax = absmax_2d.t().contiguous()  # [abs_per_row, kv_hidden] = [num_quant_blocks, kv_hidden]
            # col_blocks_absmax for hot-cache indexing (not used for dequant):
            # repeat so that col_block b maps to qblock_absmax[b // _qb_per_cb]
            col_blocks_absmax = qblock_absmax.repeat_interleave(_qb_per_cb, dim=0)  # [num_col_blocks, kv_hidden]
            abs_per_col_block = 1
            dequant_block_size = quant_block_size    # 64 — valid for bnb CUDA
            sub_per_quant = _qb_per_cb               # 2
        else:
            abs_per_col_block = block_size // quant_block_size
            dequant_block_size = quant_block_size    # original quant granularity
            sub_per_quant = 1
            qblock_absmax = None
            col_blocks_absmax = (
                absmax_2d.view(kv_hidden, num_col_blocks, abs_per_col_block)
                .permute(1, 0, 2)
                .reshape(num_col_blocks, kv_hidden * abs_per_col_block)
                .contiguous()
            )

        # Parse layer_idx from name
        parts = str(full_name).split(".")
        layer_idx = int(parts[2]) if len(parts) >= 4 and parts[1] == "layers" else -1

        result: Dict[str, Any] = {
            "packed_cols":      col_blocks_packed,   # [num_col_blocks, kv_hidden * bytes_per_col_step] uint8
            "absmax_cols":      col_blocks_absmax,   # [num_col_blocks, kv_hidden * abs_per_col_block] float32
            "qblock_absmax":    qblock_absmax,       # [num_quant_blocks, kv_hidden] or None
            "code":             quant_state.code.to(dtype=torch.float32).contiguous(),
            "code_gpu":         quant_state.code.to(device=self.device, dtype=torch.float32).contiguous()
                                if self.device.type == "cuda" else None,
            "kv_hidden":        kv_hidden,
            "hidden_size":      hidden_size,
            "block_size":       block_size,
            "num_col_blocks":   num_col_blocks,
            "quant_block_size":  quant_block_size,
            "dequant_block_size": dequant_block_size,  # blocksize to pass to _bnb_dequant_impl (>= 64)
            "sub_per_quant":    sub_per_quant,       # >1 when block_size < quant_block_size
            "quant_type":       str(quant_state.quant_type),
            "dtype":            quant_state.dtype,
            "layer_idx":        layer_idx,
        }

        self._maybe_cache_kv_hot_blocks(full_name, result)
        self._kv_sparse_param_cache[full_name] = result
        return result

    def _maybe_cache_kv_hot_blocks(self, full_name: str, param: Dict[str, Any]) -> None:
        """Pin hot K/V column-blocks to VRAM, using same budget as MLP/attn hot cache."""
        if not getattr(self, "_vram_hot_cache_enabled", False):
            return
        if self.device.type != "cuda":
            return
        if int(param.get("sub_per_quant", 1)) > 1:
            return  # sub-block pairing not supported in hot cache path
        layer_idx = int(param.get("layer_idx", -1))
        if layer_idx < 0:
            return
        if full_name in getattr(self, "_vram_nf4_cache", {}):
            return
        hot_blocks = self._kv_hot_blocks_by_layer.get(layer_idx)
        if hot_blocks is None or int(hot_blocks.numel()) == 0:
            return
        num_col_blocks = int(param["num_col_blocks"])
        hot_blocks = hot_blocks[(hot_blocks >= 0) & (hot_blocks < num_col_blocks)]
        hot_blocks = hot_blocks.unique(sorted=True)
        if int(hot_blocks.numel()) == 0:
            return
        packed_block_bytes = int(param["packed_cols"].shape[1])
        absmax_block_bytes = int(param["absmax_cols"].shape[1]) * 4
        required_bytes = int(hot_blocks.numel()) * (packed_block_bytes + absmax_block_bytes)
        if not self._can_reserve_vram_hot_cache(required_bytes):
            return
        try:
            packed_hot_gpu = self._copy_cpu_to_gpu(
                param["packed_cols"][hot_blocks].contiguous(), dtype=torch.uint8
            )
            absmax_hot_gpu = self._copy_cpu_to_gpu(
                param["absmax_cols"][hot_blocks].contiguous(), dtype=torch.float32
            )
        except Exception as _e:
            if _is_cuda_oom_error(_e):
                self._disable_vram_hot_cache("cuda_oom_kv_hot_blocks")
            return
        lookup = torch.full((num_col_blocks,), -1, dtype=torch.int32)
        lookup[hot_blocks] = torch.arange(int(hot_blocks.numel()), dtype=torch.int32)
        hot_cache = {
            "block_ids_cpu":   hot_blocks,
            "lookup_cpu":      lookup,
            "packed_cols_gpu": packed_hot_gpu,
            "absmax_cols_gpu": absmax_hot_gpu,
        }
        param["vram_hot_kv"] = hot_cache
        self._vram_nf4_cache[full_name] = hot_cache
        self._vram_hot_cache_used_bytes += required_bytes

    def _route_kv_blocks(
        self,
        hidden_norm: torch.Tensor,   # [B, S, hidden_size] post-input-layernorm
        layer_idx: int,
    ) -> Optional[torch.Tensor]:     # [N, top_k] long indices, or None
        """Return active K/V column-block indices for this layer, or None."""
        if int(layer_idx) in self._retrieval_layers and self._token_archive is not None:
            # Retrieval layers require full k_proj/v_proj for exact new-token K/V.
            return None
        if (
            str(self._traffic_current_phase or "idle") == "prefill"
            and self._sparse_kv_prefill_mode == "dense"
        ):
            return None
        if self._should_use_attn_share_for_layer(layer_idx):
            return None
        routing = self._kv_routing.get(layer_idx)
        if routing is None:
            return None
        enc_w = routing["enc_w"]
        enc_b = routing["enc_b"]
        dec_norm_t = routing["dec_norm_t"]
        top_k = int(routing.get("top_k", self._kv_sparse_top_k))
        N = hidden_norm.shape[0] * hidden_norm.shape[1]
        h = hidden_norm.reshape(N, -1).to(device=enc_w.device, dtype=enc_w.dtype)
        latent = F.silu(F.linear(h, enc_w, enc_b))
        scores = torch.matmul(latent.abs(), dec_norm_t)
        # Apply banked-block mask
        banked_until = self._kv_block_banked_until.get(layer_idx)
        if banked_until is not None:
            banked_mask = banked_until > self._kv_bank_step
            if banked_mask.any():
                scores[:, banked_mask.to(device=scores.device)] = float("-inf")
        top_k = max(1, min(top_k, int(scores.shape[-1])))
        return scores.topk(top_k, dim=-1).indices  # [N, top_k]

    def _load_sparse_kv(
        self,
        layer_idx: int,
        active_col_blocks: torch.Tensor,   # [K_blocks] long, sorted, CPU
        *,
        kv_hidden: int,
        hidden_size: int,  # noqa: ARG002
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load K and V partial weight matrices for active column blocks only.

        Returns (k_partial, v_partial), each [kv_hidden, K_blocks * block_size]
        float16 on self.device.
        """
        prefix = f"model.layers.{int(layer_idx)}.self_attn."
        block_size = self._kv_sparse_block_size

        def _gather_proj(proj_name: str) -> torch.Tensor:
            full_name = f"{prefix}{proj_name}.weight"
            param = self._get_sparse_4bit_kv_param(full_name)
            sub_per_quant = int(param.get("sub_per_quant", 1))

            code_gpu = param.get("code_gpu")
            if code_gpu is None:
                code_gpu = param["code"].to(device=self.device)

            if sub_per_quant > 1:
                # block_size (32) < quant_block_size (64): must dequant at quant_block_size granularity.
                # Map active 32-feature col_blocks to 64-feature quant_blocks, load pairs, dequant, extract.
                quant_block_size = int(param["quant_block_size"])
                num_col_blocks = int(param["num_col_blocks"])
                bytes_per_col_step = block_size // 2

                quant_blocks, qb_inverse = torch.unique(
                    active_col_blocks // sub_per_quant, sorted=True, return_inverse=True
                )
                K_qb = int(quant_blocks.numel())

                # For each quant_block, load both sub-blocks (cb*2 and cb*2+1) from packed_cols
                cb0_idx = quant_blocks * sub_per_quant                                      # [K_qb]
                cb1_idx = (quant_blocks * sub_per_quant + 1).clamp(max=num_col_blocks - 1)  # [K_qb]
                packed_cb0 = param["packed_cols"][cb0_idx]  # [K_qb, kv_hidden * bytes_per_col_step]
                packed_cb1 = param["packed_cols"][cb1_idx]  # [K_qb, kv_hidden * bytes_per_col_step]
                packed_pairs = torch.cat(
                    [packed_cb0.view(K_qb, kv_hidden, bytes_per_col_step),
                     packed_cb1.view(K_qb, kv_hidden, bytes_per_col_step)],
                    dim=2,
                ).contiguous()  # [K_qb, kv_hidden, 2*bytes_per_col_step]

                # Absmax at quant-block granularity: [K_qb, kv_hidden]
                qblock_absmax_cpu = param["qblock_absmax"][quant_blocks].contiguous()

                packed_gpu = self._copy_cpu_to_gpu(packed_pairs.reshape(-1), dtype=torch.uint8)
                absmax_gpu = self._copy_cpu_to_gpu(qblock_absmax_cpu.reshape(-1), dtype=torch.float32)
                self._wait_for_h2d_stream()

                out_size = K_qb * kv_hidden * quant_block_size
                out_fp16 = torch.empty(out_size, dtype=self.dtype, device=self.device)
                _bnb_dequant_impl(
                    packed_gpu.reshape(-1),
                    absmax_gpu.reshape(-1),
                    quant_block_size,          # 64 — valid for bnb CUDA
                    str(param["quant_type"]),
                    self.dtype,
                    out=out_fp16,
                )

                # Reshape: [K_qb, kv_hidden, quant_block_size]
                dequanted = out_fp16.view(K_qb, kv_hidden, quant_block_size)

                # Extract 32-feature sub-blocks for originally-active col_blocks
                # even col_block → first half ([:, :, :32]), odd → second half ([:, :, 32:])
                qb_inv_gpu = qb_inverse.to(device=self.device, dtype=torch.long)
                half0_mask = ((active_col_blocks % sub_per_quant) == 0).to(device=self.device)
                sel = dequanted[qb_inv_gpu]                             # [K_active, kv_hidden, 64]
                first_half  = sel[:, :, :block_size]                   # [K_active, kv_hidden, 32]
                second_half = sel[:, :, block_size:block_size * 2]     # [K_active, kv_hidden, 32]
                mask_exp = half0_mask[:, None, None].expand(-1, kv_hidden, block_size)
                result_3d = torch.where(mask_exp, first_half, second_half)   # [K_active, kv_hidden, 32]
                return result_3d.permute(1, 0, 2).reshape(kv_hidden, -1).contiguous()

            # Standard path: block_size >= quant_block_size (one absmax per col-block or multiple)
            hot_cache = param.get("vram_hot_kv")
            parts_packed: List[torch.Tensor] = []
            parts_absmax: List[torch.Tensor] = []
            cold_blocks = active_col_blocks

            if hot_cache is not None:
                lookup = hot_cache["lookup_cpu"]
                clamped = active_col_blocks.clamp(0, int(lookup.shape[0]) - 1)
                slots = lookup[clamped]
                hot_mask_cpu = slots >= 0
                if hot_mask_cpu.any():
                    hot_slots = slots[hot_mask_cpu].to(device=self.device, dtype=torch.long)
                    parts_packed.append(hot_cache["packed_cols_gpu"].index_select(0, hot_slots))
                    parts_absmax.append(hot_cache["absmax_cols_gpu"].index_select(0, hot_slots))
                cold_blocks = active_col_blocks[~hot_mask_cpu]

            if int(cold_blocks.numel()) > 0:
                cold_packed_cpu = param["packed_cols"][cold_blocks].contiguous()
                cold_absmax_cpu = param["absmax_cols"][cold_blocks].contiguous()
                cold_packed_gpu = self._copy_cpu_to_gpu(cold_packed_cpu, dtype=torch.uint8)
                cold_absmax_gpu = self._copy_cpu_to_gpu(cold_absmax_cpu, dtype=torch.float32)
                self._wait_for_h2d_stream()
                parts_packed.append(cold_packed_gpu)
                parts_absmax.append(cold_absmax_gpu)

            if not parts_packed:
                raise RuntimeError(f"No blocks loaded for {full_name}")

            packed_gpu = parts_packed[0] if len(parts_packed) == 1 else torch.cat(parts_packed, dim=0).contiguous()
            absmax_gpu = parts_absmax[0] if len(parts_absmax) == 1 else torch.cat(parts_absmax, dim=0).contiguous()

            K_blocks = int(packed_gpu.shape[0])
            out_size = K_blocks * kv_hidden * block_size
            out_fp16 = torch.empty(out_size, dtype=self.dtype, device=self.device)
            _bnb_dequant_impl(
                packed_gpu.reshape(-1),
                absmax_gpu.reshape(-1),
                int(param["dequant_block_size"]),
                str(param["quant_type"]),
                self.dtype,
                out=out_fp16,
            )
            return out_fp16.view(K_blocks, kv_hidden, block_size).permute(1, 0, 2).reshape(kv_hidden, -1).contiguous()

        k_partial = _gather_proj("k_proj")
        v_partial = _gather_proj("v_proj")
        return k_partial, v_partial

    def _clear_kv_skeleton(
        self,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
    ) -> None:
        """Zero out previously-written column blocks in K/V skeleton weights."""
        if self._kv_loaded_cols is not None and int(self._kv_loaded_cols.numel()) > 0:
            cols = self._kv_loaded_cols
            k_weight.index_fill_(1, cols, 0.0)
            v_weight.index_fill_(1, cols, 0.0)
        else:
            k_weight.zero_()
            v_weight.zero_()
        self._kv_loaded_cols = None

    def _populate_sparse_kv_skeleton(
        self,
        layer_idx: int,
        active_col_blocks: torch.Tensor,   # [N, top_k] long
        layer: "LlamaDecoderLayer",
    ) -> None:
        """Populate K/V skeleton weights with only the active column blocks.

        For prefill (N > 1): union of all tokens' active blocks.
        For decode (N == 1): that token's active blocks.
        The attention forward uses standard F.linear on the sparse skeleton;
        zero columns contribute zero to K/V output.
        """
        k_weight = layer.self_attn.k_proj.weight   # [kv_hidden, hidden_size] on GPU
        v_weight = layer.self_attn.v_proj.weight

        # Get union of active blocks across all tokens
        if active_col_blocks.shape[0] == 1:
            blocks_cpu = active_col_blocks[0].cpu()
        else:
            blocks_cpu = active_col_blocks.reshape(-1).cpu().unique(sorted=True)
        blocks_cpu = blocks_cpu.sort().values

        kv_hidden = int(k_weight.shape[0])
        hidden_size = int(k_weight.shape[1])
        block_size = self._kv_sparse_block_size

        # Clear previous skeleton columns
        self._clear_kv_skeleton(k_weight, v_weight)

        # Load active column blocks
        k_partial, v_partial = self._load_sparse_kv(
            layer_idx, blocks_cpu, kv_hidden=kv_hidden, hidden_size=hidden_size
        )

        # Compute column indices: block b → columns b*block_size .. (b+1)*block_size-1
        col_offsets = (
            blocks_cpu.unsqueeze(-1) * block_size
            + torch.arange(block_size, dtype=torch.long)
        ).reshape(-1).to(device=self.device, dtype=torch.long)

        # Scatter into skeleton
        k_weight.index_copy_(1, col_offsets, k_partial)
        v_weight.index_copy_(1, col_offsets, v_partial)
        self._kv_loaded_cols = col_offsets

    def _update_kv_block_banking(
        self,
        layer_idx: int,
        active_col_blocks: torch.Tensor,   # [N, top_k] or [top_k]
    ) -> None:
        """Update block usage EMA and banking state for a layer."""
        ema = self._kv_block_usage_ema.get(layer_idx)
        if ema is None:
            return
        decay = 0.95
        num_blocks = int(ema.numel())
        used = active_col_blocks.reshape(-1).cpu()
        used = used[(used >= 0) & (used < num_blocks)]
        current_usage = torch.zeros(num_blocks, dtype=torch.float32)
        if int(used.numel()) > 0:
            current_usage.scatter_add_(0, used.long(), torch.ones(int(used.numel()), dtype=torch.float32))
            current_usage = (current_usage > 0).float()
        ema.mul_(decay)
        ema.add_(current_usage * (1.0 - decay))
        votes = self._kv_block_usage_votes.get(layer_idx)
        banked_until = self._kv_block_banked_until.get(layer_idx)
        if votes is None or banked_until is None:
            return
        low_threshold = 0.001
        vote_threshold = 3
        cooldown = 64
        step = self._kv_bank_step
        low_mask = ema < low_threshold
        in_cooldown = banked_until > step
        vote_mask = low_mask & ~in_cooldown
        votes[vote_mask] += 1
        votes[~vote_mask] = 0
        can_bank = votes >= vote_threshold
        if can_bank.any():
            banked_until[can_bank] = step + cooldown
            votes[can_bank] = 0

    def _next_h2d_scratch_key(self, tag: str, dtype: torch.dtype) -> str:
        key = f"{str(tag)}:{str(dtype)}"
        slot = int(self._h2d_stage_slots[key] % 2)
        self._h2d_stage_slots[key] += 1
        return f"{key}:{slot}"

    def _copy_cpu_to_gpu(
        self,
        tensor: torch.Tensor,
        *,
        dtype: torch.dtype,
        layer_idx: Optional[int] = None,
        tag: str = "h2d",
    ) -> torch.Tensor:
        scratch_key = self._next_h2d_scratch_key(tag, dtype)
        prepared = self.loader._stage_h2d_source_via_scratch(
            tensor,
            dtype=dtype,
            scratch_key=scratch_key,
        )
        self._record_h2d_bytes(
            int(prepared.numel() * prepared.element_size()),
            layer_idx=layer_idx,
            tag=tag,
        )
        if self.device.type != "cuda":
            return prepared.to(device=self.device, dtype=dtype)
        try:
            out = torch.empty(tuple(prepared.shape), dtype=dtype, device=self.device)
            if self._h2d_stream is not None:
                with torch.cuda.stream(self._h2d_stream):
                    out.copy_(prepared, non_blocking=bool(prepared.is_pinned()))
                    self.loader._record_h2d_scratch_use(scratch_key, device=out.device)
            else:
                out.copy_(prepared, non_blocking=bool(prepared.is_pinned()))
                self.loader._record_h2d_scratch_use(scratch_key, device=out.device)
            return out
        except Exception as _e:
            if not _is_cuda_oom_error(_e):
                raise
            # GPU-only recovery path: drop persistent hot-cache tensors that may
            # have consumed VRAM headroom, then retry the same H2D copy once.
            self._disable_vram_hot_cache("cuda_oom_during_h2d_copy")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            out = torch.empty(tuple(prepared.shape), dtype=dtype, device=self.device)
            if self._h2d_stream is not None:
                with torch.cuda.stream(self._h2d_stream):
                    out.copy_(prepared, non_blocking=bool(prepared.is_pinned()))
                    self.loader._record_h2d_scratch_use(scratch_key, device=out.device)
            else:
                out.copy_(prepared, non_blocking=bool(prepared.is_pinned()))
                self.loader._record_h2d_scratch_use(scratch_key, device=out.device)
            return out

    def _copy_cpu_to_existing_gpu(
        self,
        dest: torch.Tensor,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        scratch_key = self._next_h2d_scratch_key("h2d_existing", dest.dtype)
        prepared = self.loader._stage_h2d_source_via_scratch(
            tensor,
            dtype=dest.dtype,
            scratch_key=scratch_key,
        )
        if self.device.type != "cuda":
            dest.copy_(prepared)
            return prepared
        if self._h2d_stream is not None:
            with torch.cuda.stream(self._h2d_stream):
                dest.copy_(prepared, non_blocking=bool(prepared.is_pinned()))
                self.loader._record_h2d_scratch_use(scratch_key, device=dest.device)
        else:
            dest.copy_(prepared, non_blocking=bool(prepared.is_pinned()))
            self.loader._record_h2d_scratch_use(scratch_key, device=dest.device)
        return prepared

    def _ensure_cpu_scratch(
        self,
        name: str,
        *,
        numel: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        buf = self._cpu_scratch.get(name)
        if buf is None or buf.dtype != dtype or int(buf.numel()) < int(numel):
            pin = bool(torch.cuda.is_available() and self.loader._pin_ram_cache)
            buf = torch.empty(int(numel), dtype=dtype, pin_memory=pin)
            self._cpu_scratch[name] = buf
        return buf[: int(numel)]

    def _wait_for_h2d_stream(self) -> None:
        if self.device.type != "cuda":
            return
        if self._h2d_stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self._h2d_stream)

    def _wait_h2d_stream_for_current(self) -> None:
        if self.device.type != "cuda":
            return
        if self._h2d_stream is not None:
            self._h2d_stream.wait_stream(torch.cuda.current_stream(self.device))

    def _clear_sparse_attn_qo_buffers(
        self,
        *,
        q_skel: torch.Tensor,
        o_skel: torch.Tensor,
        force_full: bool = False,
    ) -> None:
        need_full_clear = bool(force_full or self._attn_qo_state not in {"sparse", "zero"})
        if need_full_clear:
            q_skel.zero_()
            o_skel.zero_()
        else:
            if self._attn_loaded_q_rows is not None and int(self._attn_loaded_q_rows.numel()) > 0:
                q_skel.index_fill_(0, self._attn_loaded_q_rows, 0)
            if self._attn_loaded_o_cols is not None and int(self._attn_loaded_o_cols.numel()) > 0:
                o_skel.index_fill_(1, self._attn_loaded_o_cols, 0)
        self._attn_loaded_q_rows = None
        self._attn_loaded_o_cols = None
        self._attn_qo_state = "zero"

    def _maybe_cache_sparse_attn_hot_heads(
        self,
        *,
        layer_idx: int,
        q_name: str,
        o_name: str,
        meta_q: Dict[str, Any],
        meta_o: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        cached = self._attn_hot_head_cache.get(int(layer_idx))
        if cached is not None:
            return cached
        if str(self._traffic_current_phase or "idle") not in {"decode", "prefill"}:
            return None
        if not self._vram_hot_cache_enabled:
            return None
        if self.device.type != "cuda":
            return None

        static_heads = self._attn_active_head_indices.get(int(layer_idx))
        if static_heads is None or int(static_heads.numel()) <= 0:
            return None
        pool_heads = static_heads.to(dtype=torch.long, device="cpu").unique(sorted=True)
        if int(pool_heads.numel()) <= 0:
            return None

        head_dim = int(meta_q["head_dim"])
        num_heads_total = int(meta_q["num_heads_total"])
        in_features_q = int(meta_q["in_features"])
        q_block_size = int(meta_q["quant_block_size"])
        bytes_per_head_q = in_features_q // 2 * head_dim
        absmax_per_head_q = head_dim * in_features_q // q_block_size

        out_features_o = int(meta_o["out_features"])
        in_features_o = int(meta_o["in_features"])
        o_block_size = int(meta_o["quant_block_size"])
        bytes_per_row_o = in_features_o // 2
        bytes_per_head_col = head_dim // 2
        absmax_per_row_o = in_features_o // o_block_size
        absmax_per_head_col = head_dim // o_block_size
        pool_size = int(pool_heads.numel())

        required_bytes = 0
        required_bytes += int(pool_size * bytes_per_head_q)
        required_bytes += int(pool_size * absmax_per_head_q * 4)
        required_bytes += int(out_features_o * pool_size * bytes_per_head_col)
        required_bytes += int(out_features_o * pool_size * absmax_per_head_col * 4)
        if not self._can_reserve_vram_hot_cache(required_bytes):
            return None

        q_raw, _ = self.loader._load_raw_for_param(q_name)
        q_packed_cpu = q_raw.view(num_heads_total, bytes_per_head_q).index_select(0, pool_heads).contiguous()
        q_absmax_cpu = (
            meta_q["absmax_flat"]
            .view(num_heads_total, absmax_per_head_q)
            .index_select(0, pool_heads)
            .contiguous()
        )

        o_raw, _ = self.loader._load_raw_for_param(o_name)
        o_packed_2d = o_raw.view(out_features_o, bytes_per_row_o)
        o_col_offsets = (
            pool_heads.unsqueeze(-1) * bytes_per_head_col
            + torch.arange(bytes_per_head_col, dtype=torch.long)
        ).reshape(-1)
        o_packed_cpu = (
            o_packed_2d.index_select(1, o_col_offsets)
            .view(out_features_o, pool_size, bytes_per_head_col)
            .contiguous()
        )
        o_absmax_2d = meta_o["absmax_flat"].view(out_features_o, absmax_per_row_o)
        o_abs_offsets = (
            pool_heads.unsqueeze(-1) * absmax_per_head_col
            + torch.arange(absmax_per_head_col, dtype=torch.long)
        ).reshape(-1)
        o_absmax_cpu = (
            o_absmax_2d.index_select(1, o_abs_offsets)
            .view(out_features_o, pool_size, absmax_per_head_col)
            .contiguous()
        )

        try:
            q_packed_gpu = self._copy_cpu_to_gpu(
                q_packed_cpu,
                dtype=torch.uint8,
                layer_idx=layer_idx,
                tag="attn_hotcache_q_packed",
            )
            q_absmax_gpu = self._copy_cpu_to_gpu(
                q_absmax_cpu,
                dtype=torch.float32,
                layer_idx=layer_idx,
                tag="attn_hotcache_q_absmax",
            )
            o_packed_gpu = self._copy_cpu_to_gpu(
                o_packed_cpu,
                dtype=torch.uint8,
                layer_idx=layer_idx,
                tag="attn_hotcache_o_packed",
            )
            o_absmax_gpu = self._copy_cpu_to_gpu(
                o_absmax_cpu,
                dtype=torch.float32,
                layer_idx=layer_idx,
                tag="attn_hotcache_o_absmax",
            )
        except Exception as _e:
            if _is_cuda_oom_error(_e):
                self._disable_vram_hot_cache("cuda_oom_during_attn_head_hot_cache")
                return None
            raise

        if not self._vram_hot_cache_enabled:
            return None

        # Do NOT wait for the H2D stream here — the copies are queued on _h2d_stream
        # and will complete in the background while the compute stream runs
        # attention/MLP for this layer. A single sync is issued at the end of
        # _forward_prefill (before decode begins reading the cache).
        lookup = torch.full((num_heads_total,), -1, dtype=torch.int32)
        lookup[pool_heads] = torch.arange(pool_size, dtype=torch.int32)
        cached = {
            "pool_heads_cpu": pool_heads,
            "lookup_cpu": lookup,
            "q_packed_gpu": q_packed_gpu,
            "q_absmax_gpu": q_absmax_gpu,
            "o_packed_gpu": o_packed_gpu,
            "o_absmax_gpu": o_absmax_gpu,
        }
        self._attn_hot_head_cache[int(layer_idx)] = cached
        self._vram_hot_cache_used_bytes += required_bytes
        return cached

    def _prepare_sparse_blocks_for_param(
        self,
        param: Dict[str, Any],
        *,
        ordered_blocks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def _gather(*, allow_hot_cache: bool) -> Tuple[torch.Tensor, torch.Tensor]:
            hot_cache = param.get("vram_hot") if allow_hot_cache else None
            hot_packed = None
            hot_absmax = None
            cold_blocks = ordered_blocks
            if hot_cache is not None:
                lookup = hot_cache["lookup_cpu"].index_select(0, ordered_blocks.to(dtype=torch.long))
                hot_count = 0
                for idx in range(int(lookup.numel())):
                    if int(lookup[idx].item()) < 0:
                        break
                    hot_count += 1
                if hot_count > 0:
                    hot_slots = lookup[:hot_count].to(device=self.device, dtype=torch.long)
                    hot_packed = hot_cache["packed_blocks_gpu"].index_select(0, hot_slots)
                    hot_absmax = hot_cache["absmax_blocks_gpu"].index_select(0, hot_slots)
                cold_blocks = ordered_blocks[hot_count:]

            cold_packed = None
            cold_absmax = None
            if int(cold_blocks.numel()) > 0:
                cold_packed_cpu = param["packed_blocks"][cold_blocks].contiguous()
                cold_absmax_cpu = param["absmax_blocks"][cold_blocks].contiguous().to(dtype=torch.float32)
                cold_packed = self._copy_cpu_to_gpu(cold_packed_cpu, dtype=torch.uint8)
                cold_absmax = self._copy_cpu_to_gpu(cold_absmax_cpu, dtype=torch.float32)

            packed_parts = [t for t in (hot_packed, cold_packed) if t is not None]
            absmax_parts = [t for t in (hot_absmax, cold_absmax) if t is not None]
            if not packed_parts or not absmax_parts:
                raise RuntimeError("Sparse block gather produced no data")
            packed = packed_parts[0] if len(packed_parts) == 1 else torch.cat(packed_parts, dim=0).contiguous()
            absmax = absmax_parts[0] if len(absmax_parts) == 1 else torch.cat(absmax_parts, dim=0).contiguous()
            return packed, absmax

        try:
            return _gather(allow_hot_cache=True)
        except Exception as _e:
            if not _is_cuda_oom_error(_e):
                raise
            if torch.cuda.is_available():
                try:
                    _free, _total = torch.cuda.mem_get_info(self.device)
                    _alloc = torch.cuda.memory_allocated(self.device)
                    _reserved = torch.cuda.memory_reserved(self.device)
                    print(
                        f"[oom_diag] VRAM at OOM — total={_total/(1024**3):.2f}GB "
                        f"free(driver)={_free/(1024**3):.2f}GB "
                        f"allocated={_alloc/(1024**3):.2f}GB "
                        f"reserved={_reserved/(1024**3):.2f}GB "
                        f"hot_cache_used={self._vram_hot_cache_used_bytes/(1024**3):.2f}GB "
                        f"exc_type={type(_e).__name__!r} exc={str(_e)[:200]!r}",
                        flush=True,
                    )
                except Exception:
                    pass
            self._disable_vram_hot_cache("cuda_oom_during_sparse_block_prepare")
            try:
                return _gather(allow_hot_cache=False)
            except Exception as _e2:
                if not _is_cuda_oom_error(_e2):
                    raise
                torch.cuda.empty_cache()
                return _gather(allow_hot_cache=False)

    def _compute_sparse_basis_latent(
        self,
        flat_hidden: torch.Tensor,
        layer_idx: int,
        routing: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        routing = self._sparse_routing.get(layer_idx) if routing is None else routing
        if routing is None:
            raise RuntimeError(f"No sparse basis routing state loaded for layer {int(layer_idx)}")

        enc_w = routing["enc_w"]
        enc_b = routing["enc_b"]
        hidden_proj = flat_hidden.to(device=enc_w.device, dtype=enc_w.dtype)
        latent_dense = F.silu(F.linear(hidden_proj, enc_w, enc_b))
        latent_dim = int(latent_dense.shape[-1])
        if latent_dim <= 0:
            return latent_dense
        basis_top_k = int(
            routing.get(
                "basis_top_k",
                self._sparse_basis_top_k_by_layer.get(int(layer_idx), _DEFAULT_SPARSE_BASIS_TOP_K),
            )
        )
        basis_top_k = max(1, min(basis_top_k, latent_dim))
        if basis_top_k >= latent_dim:
            return latent_dense
        topk_idx = torch.topk(latent_dense.abs(), k=basis_top_k, dim=-1).indices
        latent_mask = torch.zeros_like(latent_dense)
        latent_mask.scatter_(1, topk_idx, 1.0)
        return latent_dense * latent_mask

    def _register_runtime_sparse_basis_layer(
        self,
        layer_idx: int,
        *,
        encoder_weight: torch.Tensor,
        encoder_bias: torch.Tensor,
        decoder_blocks: torch.Tensor,
        decoder_bias: Optional[torch.Tensor],
        scale: float,
        top_k: int,
        basis_top_k: int,
        session_local: bool,
    ) -> None:
        _basis_device = self.device if self.device.type == "cuda" else torch.device("cpu")
        _dec = decoder_blocks.to(device=_basis_device, dtype=self.dtype, non_blocking=False).contiguous()
        _layer_num_blocks = int(_dec.shape[0])
        _expected_output_blocks = int(self.config.hidden_size) // max(int(self._sparse_block_size), 1)
        if _layer_num_blocks != _expected_output_blocks:
            raise RuntimeError(
                f"Sparse basis layer {int(layer_idx)} stores {_layer_num_blocks} output blocks, "
                f"expected {_expected_output_blocks}. The learned-basis artifact is defined "
                "over hidden-size output blocks and cannot be mixed with FFN intermediate blocks."
            )
        _enc_w = encoder_weight.to(device=_basis_device, dtype=self.dtype, non_blocking=False).contiguous()
        _enc_b = encoder_bias.to(device=_basis_device, dtype=self.dtype, non_blocking=False).contiguous()
        _basis_rank = int(_enc_w.shape[0])
        _layer_top_k = int(max(1, min(int(top_k), _layer_num_blocks)))
        _basis_top_k = int(max(1, min(int(basis_top_k), _basis_rank)))
        _dec_bias = None
        if torch.is_tensor(decoder_bias):
            _dec_bias = decoder_bias.to(device=_basis_device, dtype=self.dtype, non_blocking=False).contiguous()
        _dec_norm_t = _dec.norm(dim=-1).transpose(0, 1).contiguous()
        if self._sparse_semantic_block_score_normalized:
            _dec_norm_t = F.normalize(_dec_norm_t, p=2.0, dim=-1, eps=1e-6)
        _dec_full_weight = None
        _dec_full_bias = None
        if self._sparse_basis_execution == "full_output_latent":
            _dec_full_weight = _dec.permute(0, 2, 1).reshape(
                _layer_num_blocks * int(self._sparse_block_size),
                _basis_rank,
            ).contiguous()
            if _dec_bias is not None and self._sparse_basis_bias_mode != "none":
                _dec_full_bias = _dec_bias.reshape(-1).contiguous()
        self._sparse_routing[int(layer_idx)] = {
            "enc_w": _enc_w,
            "enc_b": _enc_b,
            "dec": _dec,
            "dec_bias": _dec_bias,
            "dec_full_weight": _dec_full_weight,
            "dec_full_bias": _dec_full_bias,
            "dec_norm_t": _dec_norm_t,
            "scale": float(scale),
            "top_k": _layer_top_k,
            "basis_top_k": _basis_top_k,
        }
        self._sparse_top_k_by_layer[int(layer_idx)] = int(_layer_top_k)
        self._sparse_basis_top_k_by_layer[int(layer_idx)] = int(_basis_top_k)
        if session_local:
            self._session_sparse_route_layers.add(int(layer_idx))
            self._mlp_hot_blocks_by_layer.pop(int(layer_idx), None)

    def _maybe_fit_local_decode_guard_basis(
        self,
        layer_idx: int,
        mlp_input: torch.Tensor,
        mlp_out: torch.Tensor,
    ) -> None:
        if int(layer_idx) in self._sparse_routing:
            return
        if int(layer_idx) not in self._upper_decode_guard_layers:
            return
        rows = int(mlp_input.shape[0] * mlp_input.shape[1])
        if rows < 2:
            return

        fit_rank = int(min(max(int(self._sparse_checkpoint_basis_rank), 1), rows, int(mlp_out.shape[-1])))
        if fit_rank <= 0:
            return

        fitted = fit_layer_basis(
            x=mlp_input.reshape(rows, mlp_input.shape[-1]).detach().to(device="cpu", dtype=torch.float32),
            y=mlp_out.reshape(rows, mlp_out.shape[-1]).detach().to(device="cpu", dtype=torch.float32),
            basis_rank=int(fit_rank),
            block_size=int(self._sparse_block_size),
            pca_method="auto",
            pca_batch_rows=1024,
        )
        _scale = fitted.get("scale", 1.0)
        _scale_value = float(_scale.item()) if torch.is_tensor(_scale) and int(_scale.numel()) == 1 else float(_scale)
        self._register_runtime_sparse_basis_layer(
            int(layer_idx),
            encoder_weight=fitted["encoder_weight"],
            encoder_bias=fitted["encoder_bias"],
            decoder_blocks=fitted["decoder_blocks"],
            decoder_bias=fitted.get("decoder_bias"),
            scale=_scale_value,
            top_k=int(self._sparse_top_k),
            basis_top_k=int(min(_DEFAULT_SPARSE_BASIS_TOP_K, fit_rank)),
            session_local=True,
        )

    def _route_sparse_mlp(self, hidden: torch.Tensor, layer_idx: int) -> Optional[torch.Tensor]:
        """Return active block indices [N, top_k] for this layer, or None (→ dense).

        Uses the learned encoder/decoder from the basis checkpoint to predict
        which MLP output blocks will have the largest norm for this hidden state,
        then returns the top-K block indices.  All tensors moved to GPU lazily.
        """
        routing = self._sparse_routing.get(layer_idx)
        if routing is None:
            return None

        dec_norm_t = routing["dec_norm_t"]

        flat_hidden = hidden.view(hidden.shape[0] * hidden.shape[1], -1)
        latent = self._compute_sparse_basis_latent(flat_hidden, layer_idx, routing=routing)
        scores = torch.matmul(latent.abs(), dec_norm_t)
        k_runtime = int(
            routing.get(
                "top_k",
                self._sparse_runtime_top_k if int(self._sparse_runtime_top_k) > 0 else self._sparse_top_k,
            )
        )
        k_runtime = max(1, min(k_runtime, int(scores.shape[-1])))
        active_blocks = scores.topk(k_runtime, dim=-1).indices
        return active_blocks

    def _ensure_mlp_proj_staging(self) -> bool:
        if self._mlp_proj_staging is not None:
            return True
        if self.device.type != "cuda":
            return False
        try:
            self._mlp_proj_staging = torch.empty(
                int(self._mlp_proj_staging_numel),
                dtype=self.dtype,
                device=self.device,
            )
            return True
        except Exception as _e:
            if "out of memory" in str(_e).lower() or "alloc" in str(_e).lower():
                if not self._dense_mlp_staging_warned:
                    self._dense_mlp_staging_warned = True
                    print(
                        "[dense_mlp] Unable to allocate GPU staging for dense fallback; "
                        "dense calibration paths may use zero passthrough.",
                        flush=True,
                    )
                return False
            raise

    def _forward_learned_basis_mlp(
        self,
        layer_idx: int,
        hidden: torch.Tensor,
        active_blocks: torch.Tensor,
    ) -> torch.Tensor:
        routing = self._sparse_routing.get(layer_idx)
        if routing is None:
            return torch.zeros_like(hidden)

        dec = routing["dec"]
        dec_bias = routing.get("dec_bias")
        scale = float(routing.get("scale", 1.0))

        rows = int(hidden.shape[0] * hidden.shape[1])
        if rows <= 0:
            return torch.zeros_like(hidden)

        flat_hidden = hidden.view(rows, hidden.shape[-1])
        latent = self._compute_sparse_basis_latent(flat_hidden, layer_idx, routing=routing)
        num_blocks = int(dec.shape[0])
        block_size = int(dec.shape[-1])
        if int(hidden.shape[-1]) % max(block_size, 1) != 0:
            raise RuntimeError(
                f"Layer {int(layer_idx)} hidden size {int(hidden.shape[-1])} is not divisible by "
                f"learned-basis block_size {block_size}."
            )
        # Learned-basis reconstruction is sparse over selected output blocks only.
        # Inactive blocks stay at zero; decoder_bias is applied only to selected blocks.
        expected_output_blocks = int(hidden.shape[-1]) // max(block_size, 1)
        if num_blocks != expected_output_blocks:
            raise RuntimeError(
                f"Layer {int(layer_idx)} learned-basis decoder exposes {num_blocks} output blocks, "
                f"expected {expected_output_blocks} for hidden size {int(hidden.shape[-1])}. "
                "Learned-basis sparse routing is defined over output-space blocks and cannot "
                "be executed by the intermediate-space sparse 4-bit MLP path."
            )
        out_blocks = torch.zeros((rows, num_blocks, block_size), device=dec.device, dtype=dec.dtype)
        row_idx = torch.arange(rows, device=dec.device, dtype=torch.long)
        active_blocks = active_blocks.to(device=dec.device, dtype=torch.long)
        use_selected_bias = bool(torch.is_tensor(dec_bias) and self._sparse_basis_bias_mode == "selected")
        dec_bias_device = (
            dec_bias.to(device=dec.device, dtype=dec.dtype)
            if use_selected_bias
            else None
        )

        for slot in range(int(active_blocks.shape[1])):
            block_idx = active_blocks[:, slot]
            valid = (block_idx >= 0) & (block_idx < num_blocks)
            if not torch.any(valid):
                continue
            rows_valid = row_idx[valid]
            blocks_valid = block_idx[valid]
            coeff = latent.index_select(0, rows_valid)
            decoder_selected = dec.index_select(0, blocks_valid)
            contrib = torch.bmm(coeff.unsqueeze(1), decoder_selected).squeeze(1)
            if dec_bias_device is not None:
                contrib = contrib + dec_bias_device.index_select(0, blocks_valid)
            out_blocks[rows_valid, blocks_valid] += contrib

        out_flat = out_blocks.view(rows, num_blocks * block_size)
        if scale != 1.0:
            out_flat = out_flat * scale
        return out_flat.view_as(hidden).to(device=hidden.device, dtype=hidden.dtype)

    def _forward_learned_basis_mlp_full_output(
        self,
        layer_idx: int,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct the full MLP output from a sparse latent support.

        The learned-basis artifact already parameterizes the full hidden-size
        MLP output in output-space blocks. To preserve dense semantics while
        minimizing compute, this path keeps only the top-k latent coordinates
        and decodes the full output from those coordinates directly, without
        dropping output blocks at runtime.
        """
        routing = self._sparse_routing.get(layer_idx)
        if routing is None:
            return torch.zeros_like(hidden)

        dec = routing["dec"]
        dec_bias = routing.get("dec_bias")
        scale = float(routing.get("scale", 1.0))

        rows = int(hidden.shape[0] * hidden.shape[1])
        if rows <= 0:
            return torch.zeros_like(hidden)

        flat_hidden = hidden.view(rows, hidden.shape[-1])
        latent = self._compute_sparse_basis_latent(flat_hidden, layer_idx, routing=routing)
        latent = latent.to(device=dec.device, dtype=dec.dtype)

        num_blocks = int(dec.shape[0])
        block_size = int(dec.shape[-1])
        if int(hidden.shape[-1]) % max(block_size, 1) != 0:
            raise RuntimeError(
                f"Layer {int(layer_idx)} hidden size {int(hidden.shape[-1])} is not divisible by "
                f"learned-basis block_size {block_size}."
            )
        expected_output_blocks = int(hidden.shape[-1]) // max(block_size, 1)
        if num_blocks != expected_output_blocks:
            raise RuntimeError(
                f"Layer {int(layer_idx)} learned-basis decoder exposes {num_blocks} output blocks, "
                f"expected {expected_output_blocks} for hidden size {int(hidden.shape[-1])}. "
                "Learned-basis sparse routing is defined over output-space blocks and cannot "
                "be executed by the intermediate-space sparse 4-bit MLP path."
            )

        dec_full_weight = routing.get("dec_full_weight")
        if dec_full_weight is None:
            dec_full_weight = dec.permute(0, 2, 1).reshape(num_blocks * block_size, int(latent.shape[-1])).contiguous()
        dec_full_weight = dec_full_weight.to(device=dec.device, dtype=dec.dtype)

        bias_full = routing.get("dec_full_bias")
        if bias_full is None and torch.is_tensor(dec_bias) and self._sparse_basis_bias_mode != "none":
            bias_full = dec_bias.reshape(-1).contiguous()
        if torch.is_tensor(bias_full):
            bias_full = bias_full.to(device=dec.device, dtype=dec.dtype)

        out_flat = F.linear(latent, dec_full_weight, bias_full)
        if scale != 1.0:
            out_flat = out_flat * scale
        return out_flat.view_as(hidden).to(device=hidden.device, dtype=hidden.dtype)

    def _get_sparse_4bit_param(self, full_name: str) -> Dict[str, Any]:
        cached = self._sparse_param_cache.get(full_name)
        raw_weight: Optional[torch.Tensor] = None
        if cached is None:
            raw_weight, quant_aux = self.loader._load_raw_for_param(full_name)
            if not quant_aux:
                raise RuntimeError(f"Sparse 4-bit path expected quantized weights for {full_name}")

            quant_state = bnb_functional.QuantState.from_dict(quant_aux, device=torch.device("cpu"))
            absmax = quant_state.absmax
            if quant_state.nested:
                absmax = bnb_functional.dequantize_blockwise(quant_state.absmax, quant_state.state2)
                absmax = absmax + quant_state.offset

            in_features = int(quant_state.shape[1])
            out_features = int(quant_state.shape[0])
            block_size = int(self._sparse_block_size)
            bytes_per_row = in_features // 2
            bytes_per_block = bytes_per_row * block_size
            blocks_per_col = out_features // max(block_size, 1)
            absmax_cpu = absmax.to(dtype=torch.float32).contiguous()
            absmax_per_row = in_features // int(quant_state.blocksize)
            absmax_per_block = absmax_per_row * block_size

            cached = {
                "absmax_cpu": absmax_cpu,
                "code": quant_state.code.to(dtype=torch.float32).contiguous(),
                "code_gpu": quant_state.code.to(device=self.device, dtype=torch.float32, non_blocking=True).contiguous()
                if self.device.type == "cuda"
                else None,
                "out_features": out_features,
                "in_features": in_features,
                "bytes_per_row": bytes_per_row,
                "bytes_per_block": bytes_per_block,
                "blocks_per_col": blocks_per_col,
                "absmax_per_row": absmax_per_row,
                "absmax_per_block": absmax_per_block,
                "quant_block_size": int(quant_state.blocksize),
                "quant_type": str(quant_state.quant_type),
                "dtype": quant_state.dtype,
            }
            param_views = dict(cached)
            raw_flat = raw_weight.reshape(-1)
            param_views["packed_weight"] = raw_flat
            param_views["packed_blocks"] = raw_flat.view(blocks_per_col, bytes_per_block)
            param_views["absmax"] = absmax_cpu
            param_views["absmax_blocks"] = absmax_cpu.view(blocks_per_col, absmax_per_block)
            self._maybe_cache_sparse_param_hot_blocks(full_name, param_views)
            if "vram_hot" in param_views:
                cached["vram_hot"] = param_views["vram_hot"]
            if "vram_hot_down" in param_views:
                cached["vram_hot_down"] = param_views["vram_hot_down"]
            self._sparse_param_cache[full_name] = cached
            return param_views

        raw_weight, _quant_aux = self.loader._load_raw_for_param(full_name)
        raw_flat = raw_weight.reshape(-1)
        param = dict(cached)
        param["packed_weight"] = raw_flat
        param["packed_blocks"] = raw_flat.view(int(cached["blocks_per_col"]), int(cached["bytes_per_block"]))
        absmax_cpu = cached["absmax_cpu"]
        param["absmax"] = absmax_cpu
        param["absmax_blocks"] = absmax_cpu.view(int(cached["blocks_per_col"]), int(cached["absmax_per_block"]))
        return param

    def _load_optional_bias(
        self,
        full_name: str,
        bias_ref: Optional[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
        layer_idx: Optional[int] = None,
        tag: str = "bias",
    ) -> Optional[torch.Tensor]:
        if bias_ref is None:
            return None
        bias_name = full_name[:-7] + ".bias" if full_name.endswith(".weight") else ""
        if not bias_name:
            self._record_h2d_bytes(
                int(bias_ref.numel() * bias_ref.element_size()),
                layer_idx=layer_idx,
                tag=tag,
            )
            return bias_ref.to(device=device, dtype=dtype, non_blocking=True)
        bias = self.loader.load_parameter(bias_name)
        self._record_h2d_bytes(
            int(bias.numel() * bias.element_size()),
            layer_idx=layer_idx,
            tag=tag,
        )
        return bias.to(device=device, dtype=dtype, non_blocking=True)

    def _sparse_mlp_forward(
        self,
        layer_idx: int,
        mlp: nn.Module,
        hidden: torch.Tensor,
        active_blocks: torch.Tensor,
    ) -> torch.Tensor:
        """Run LlamaMLP on only the active neuron blocks; fall back if needed.

        active_blocks: [N, top_k] — block indices selected by _route_sparse_mlp.

        For each token we gather the rows of gate_proj/up_proj and the columns of
        down_proj that correspond to active neurons, run the SiLU-gated MLP on
        that small slice (~2% of weights for top_k=2%), and return the output.

        Falls back to dense MLP if the module doesn't expose the expected Linear
        sub-layers.
        """
        gate_proj = getattr(mlp, "gate_proj", None)
        up_proj   = getattr(mlp, "up_proj",   None)
        down_proj = getattr(mlp, "down_proj",  None)
        if gate_proj is None or up_proj is None or down_proj is None:
            return mlp(hidden)  # not a standard LlamaMLP — run dense
        if not (hasattr(gate_proj, "weight") and gate_proj.weight is not None):
            return mlp(hidden)  # quantised linear without accessible .weight

        block_size = self._sparse_block_size
        N, H = hidden.shape[0] * hidden.shape[1], hidden.shape[-1]
        h = hidden.view(N, H)

        # Expand block indices to individual neuron indices.
        # active_blocks: [N, K]  →  active_neurons: [N, K*block_size]
        neuron_offsets = torch.arange(block_size, device=active_blocks.device)  # [S]
        active_neurons = (
            active_blocks.unsqueeze(-1) * block_size + neuron_offsets
        ).reshape(N, -1)                                              # [N, K*S]

        # For single-token generation (N=1) the unique set == the full set.
        # For N>1 take the union so every token's active neurons are covered.
        if N == 1:
            unique_neurons = active_neurons.squeeze(0)                # [K*S]
        else:
            unique_neurons = active_neurons.unique()                  # [≤N*K*S]

        def _is_4bit_linear(linear: nn.Module) -> bool:
            weight = getattr(linear, "weight", None)
            return bool(torch.is_tensor(weight) and getattr(weight, "quant_state", None) is not None)

        def _load_quantized_weight_cpu(param_name: str) -> Optional[torch.Tensor]:
            try:
                raw_weight, quant_aux = self.loader._load_raw_for_param(param_name)
            except Exception:
                return None
            if not quant_aux:
                return None
            quant_state = bnb_functional.QuantState.from_dict(quant_aux, device=torch.device("cpu"))
            return bnb_functional.dequantize_4bit(raw_weight, quant_state=quant_state)

        def _project_active_inputs(linear: nn.Module, neurons: torch.Tensor) -> torch.Tensor:
            bias = getattr(linear, "bias", None)
            weight = getattr(linear, "weight", None)
            proj_weight = None
            linear_name = getattr(linear, "_sparse_param_name", "")
            if linear_name:
                dense_weight_cpu = _load_quantized_weight_cpu(linear_name)
                if dense_weight_cpu is not None:
                    proj_weight = dense_weight_cpu.index_select(0, neurons.cpu()).to(
                        device=h.device,
                        dtype=h.dtype,
                        non_blocking=True,
                    )
                    del dense_weight_cpu
            if proj_weight is None and _is_4bit_linear(linear):
                quant_state = weight.quant_state
                dense_weight = bnb_functional.dequantize_4bit(weight.t(), quant_state).t()
                proj_weight = dense_weight.index_select(0, neurons)
                del dense_weight
            if proj_weight is None:
                proj_weight = weight.index_select(0, neurons)
            if bias is not None:
                bias = bias.index_select(0, neurons)
            return F.linear(h, proj_weight, bias)

        def _project_active_outputs(x: torch.Tensor, linear: nn.Module, neurons: torch.Tensor) -> torch.Tensor:
            bias = getattr(linear, "bias", None)
            weight = getattr(linear, "weight", None)
            proj_weight = None
            linear_name = getattr(linear, "_sparse_param_name", "")
            if linear_name:
                dense_weight_cpu = _load_quantized_weight_cpu(linear_name)
                if dense_weight_cpu is not None:
                    proj_weight = dense_weight_cpu.index_select(1, neurons.cpu()).to(
                        device=x.device,
                        dtype=x.dtype,
                        non_blocking=True,
                    )
                    del dense_weight_cpu
            if proj_weight is None and _is_4bit_linear(linear):
                quant_state = weight.quant_state
                dense_weight = bnb_functional.dequantize_4bit(weight.t(), quant_state).t()
                proj_weight = dense_weight.index_select(1, neurons)
                del dense_weight
            if proj_weight is None:
                proj_weight = weight.index_select(1, neurons)
            return F.linear(x, proj_weight, bias)

        prefix = f"model.layers.{int(layer_idx)}.mlp."
        setattr(gate_proj, "_sparse_param_name", f"{prefix}gate_proj.weight")
        setattr(up_proj, "_sparse_param_name", f"{prefix}up_proj.weight")
        setattr(down_proj, "_sparse_param_name", f"{prefix}down_proj.weight")
        gate = _project_active_inputs(gate_proj, unique_neurons)    # [N, K*S]
        up   = _project_active_inputs(up_proj, unique_neurons)      # [N, K*S]
        act  = F.silu(gate) * up                                    # [N, K*S]
        del gate, up
        out  = _project_active_outputs(act, down_proj, unique_neurons)  # [N, H]
        return out.view_as(hidden)

    def _dense_mlp_forward_streaming(
        self,
        mlp: nn.Module,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        gate_proj = getattr(mlp, "gate_proj", None)
        up_proj = getattr(mlp, "up_proj", None)
        down_proj = getattr(mlp, "down_proj", None)
        if gate_proj is None or up_proj is None or down_proj is None:
            return mlp(hidden)

        flat_hidden = hidden.view(-1, hidden.shape[-1])

        def _linear_stream(x: torch.Tensor, linear: nn.Module) -> torch.Tensor:
            weight = getattr(linear, "weight", None)
            bias = getattr(linear, "bias", None)
            if weight is None:
                raise RuntimeError("Dense MLP streaming path requires linear.weight")
            weight_gpu = weight.to(device=x.device, dtype=x.dtype, non_blocking=True)
            bias_gpu = None if bias is None else bias.to(device=x.device, dtype=x.dtype, non_blocking=True)
            y = F.linear(x, weight_gpu, bias_gpu)
            del weight_gpu, bias_gpu
            return y

        gate = _linear_stream(flat_hidden, gate_proj)
        up = _linear_stream(flat_hidden, up_proj)
        act = F.silu(gate) * up
        del gate, up
        out = _linear_stream(act, down_proj)
        return out.view_as(hidden)

    def _build_sparse_active_layout(
        self,
        *,
        layer_idx: int,
        active_blocks: torch.Tensor,
        rows: int,
        block_size: int,
        max_valid_blocks: int,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        active_blocks_cpu = active_blocks.detach().to(device=torch.device("cpu"), dtype=torch.long).view(rows, -1)
        union_blocks = active_blocks_cpu.reshape(-1)
        if rows > 1:
            union_blocks = union_blocks.unique(sorted=True)
        ordered_blocks = self._order_blocks_for_layer_hot_cache(layer_idx, union_blocks)
        ordered_blocks = ordered_blocks[(ordered_blocks >= 0) & (ordered_blocks < int(max_valid_blocks))]
        ordered_blocks = ordered_blocks.unique(sorted=False).to(dtype=torch.long, device=torch.device("cpu")).contiguous()
        num_active_blocks = int(ordered_blocks.numel())
        if num_active_blocks <= 0:
            return (
                ordered_blocks,
                torch.empty((rows, active_blocks_cpu.shape[1]), dtype=torch.int32, device=self.device),
                torch.empty((rows, 0), dtype=dtype, device=self.device),
                torch.empty((0,), dtype=torch.long, device=torch.device("cpu")),
            )

        union_lookup = torch.full((int(max_valid_blocks),), -1, dtype=torch.int32)
        union_lookup[ordered_blocks] = torch.arange(num_active_blocks, dtype=torch.int32)
        active_local_cpu = torch.full(active_blocks_cpu.shape, -1, dtype=torch.int32)
        valid_slots = (active_blocks_cpu >= 0) & (active_blocks_cpu < int(max_valid_blocks))
        if bool(valid_slots.any()):
            active_local_cpu[valid_slots] = union_lookup[active_blocks_cpu[valid_slots]]

        active_local = active_local_cpu.to(device=self.device, dtype=torch.int32).contiguous()
        active_dim = num_active_blocks * int(block_size)
        flat_mask = torch.zeros((rows, active_dim), device=self.device, dtype=dtype)
        row_idx = torch.arange(rows, device=self.device, dtype=torch.long)
        neuron_offsets = torch.arange(int(block_size), device=self.device, dtype=torch.long)
        for slot in range(int(active_local.shape[1])):
            slot_idx = active_local[:, slot]
            valid = slot_idx >= 0
            if not bool(valid.any()):
                continue
            cols = slot_idx[valid].to(dtype=torch.long).unsqueeze(1) * int(block_size) + neuron_offsets.unsqueeze(0)
            flat_mask[row_idx[valid].unsqueeze(1), cols] = 1

        active_neurons = (
            ordered_blocks.unsqueeze(-1) * int(block_size)
            + torch.arange(int(block_size), device=torch.device("cpu"), dtype=torch.long)
        ).reshape(-1)
        return ordered_blocks, active_local, flat_mask.contiguous(), active_neurons.contiguous()

    def _sparse_mlp_forward_fast(
        self,
        layer_idx: int,
        mlp: nn.Module,
        hidden: torch.Tensor,
        active_blocks: torch.Tensor,
        _oom_retry_depth: int = 0,
    ) -> torch.Tensor:
        if layer_idx in self._sparse_routing:
            # Learned-basis sparse layers route over output-space blocks from the
            # checkpoint artifact, so they must stay on the basis decoder path.
            if str(getattr(self, "_sparse_basis_execution", "full_output_latent")) == "full_output_latent":
                return self._forward_learned_basis_mlp_full_output(layer_idx, hidden)
            return self._forward_learned_basis_mlp(layer_idx, hidden, active_blocks)

        gate_proj = getattr(mlp, "gate_proj", None)
        up_proj = getattr(mlp, "up_proj", None)
        down_proj = getattr(mlp, "down_proj", None)
        if gate_proj is None or up_proj is None or down_proj is None:
            return self._sparse_mlp_forward(layer_idx, mlp, hidden, active_blocks)

        block_size = self._sparse_block_size
        flat_hidden = hidden.view(-1, hidden.shape[-1])
        prefix = f"model.layers.{int(layer_idx)}.mlp."

        gate_param = self._get_sparse_4bit_param(f"{prefix}gate_proj.weight")
        up_param = self._get_sparse_4bit_param(f"{prefix}up_proj.weight")
        down_param = self._get_sparse_4bit_param(f"{prefix}down_proj.weight")
        max_valid_blocks = min(
            int(gate_param["packed_blocks"].shape[0]),
            int(up_param["packed_blocks"].shape[0]),
            int(down_param["in_features"]) // int(block_size),
        )
        ordered_blocks, active_local, flat_mask, active_neurons = self._build_sparse_active_layout(
            layer_idx=layer_idx,
            active_blocks=active_blocks,
            rows=int(flat_hidden.shape[0]),
            block_size=int(block_size),
            max_valid_blocks=int(max_valid_blocks),
            dtype=flat_hidden.dtype,
        )
        num_active_blocks = int(ordered_blocks.shape[0])
        if num_active_blocks <= 0:
            return torch.zeros_like(hidden)

        K_S = num_active_blocks * block_size
        H_in = int(gate_param["in_features"])

        gate_packed_gpu, gate_absmax_gpu = self._prepare_sparse_blocks_for_param(
            gate_param, ordered_blocks=ordered_blocks
        )
        up_packed_gpu, up_absmax_gpu = self._prepare_sparse_blocks_for_param(
            up_param, ordered_blocks=ordered_blocks
        )

        # Sync compute stream for gate/up transfers only.
        self._wait_for_h2d_stream()

        H_out = int(down_param["out_features"])
        I_in = int(down_param["in_features"])
        qbs = int(down_param["quant_block_size"])
        bytes_per_row = I_in // 2
        bytes_per_cblk = block_size // 2
        raw_2d = down_param["packed_weight"].view(H_out, bytes_per_row)
        col_b_starts = ordered_blocks * bytes_per_cblk
        col_b_range = torch.arange(bytes_per_cblk, dtype=torch.long)
        col_b_idx = (col_b_starts.unsqueeze(-1) + col_b_range.unsqueeze(0)).reshape(-1)
        gathered_down_packed_cpu = raw_2d[:, col_b_idx].reshape(H_out, num_active_blocks, bytes_per_cblk).contiguous()

        absmax_per_row = I_in // qbs
        absmax_2d = down_param["absmax"].view(H_out, absmax_per_row)
        down_abs_idx = (ordered_blocks * block_size) // qbs
        gathered_down_absmax_cpu = absmax_2d[:, down_abs_idx].contiguous()

        gathered_down_packed_gpu = None
        gathered_down_absmax_gpu = None
        if self._triton_fused_sparse_mlp:
            down_hot_cache = down_param.get("vram_hot_down")
            if down_hot_cache is not None:
                try:
                    lookup = down_hot_cache["lookup_cpu"].index_select(0, ordered_blocks.to(dtype=torch.long))
                    hot_count = 0
                    for idx in range(int(lookup.numel())):
                        if int(lookup[idx].item()) < 0:
                            break
                        hot_count += 1

                    hot_cols_packed = None
                    hot_cols_absmax = None
                    if hot_count > 0:
                        hot_slots = lookup[:hot_count].to(device=self.device, dtype=torch.long)
                        hot_cols_packed = down_hot_cache["packed_cols_gpu"].index_select(1, hot_slots)
                        hot_cols_absmax = down_hot_cache["absmax_cols_gpu"].index_select(1, hot_slots)

                    cold_cols_packed = None
                    cold_cols_absmax = None
                    cold_blocks = ordered_blocks[hot_count:]
                    if int(cold_blocks.numel()) > 0:
                        cold_starts = cold_blocks * bytes_per_cblk
                        cold_idx = (cold_starts.unsqueeze(-1) + col_b_range.unsqueeze(0)).reshape(-1)
                        cold_packed_cpu = raw_2d[:, cold_idx].reshape(H_out, int(cold_blocks.numel()), bytes_per_cblk).contiguous()
                        cold_abs_idx = (cold_blocks * block_size) // qbs
                        cold_absmax_cpu = absmax_2d[:, cold_abs_idx].contiguous()
                        cold_cols_packed = self._copy_cpu_to_gpu(cold_packed_cpu, dtype=torch.uint8)
                        cold_cols_absmax = self._copy_cpu_to_gpu(cold_absmax_cpu, dtype=torch.float32)

                    packed_parts = [t for t in (hot_cols_packed, cold_cols_packed) if t is not None]
                    absmax_parts = [t for t in (hot_cols_absmax, cold_cols_absmax) if t is not None]
                    if packed_parts and absmax_parts:
                        gathered_down_packed_gpu = (
                            packed_parts[0] if len(packed_parts) == 1 else torch.cat(packed_parts, dim=1).contiguous()
                        )
                        gathered_down_absmax_gpu = (
                            absmax_parts[0] if len(absmax_parts) == 1 else torch.cat(absmax_parts, dim=1).contiguous()
                        )
                except Exception as _e:
                    if not _is_cuda_oom_error(_e):
                        raise
                    self._disable_vram_hot_cache("cuda_oom_during_down_hot_prepare")
                    gathered_down_packed_gpu = None
                    gathered_down_absmax_gpu = None
            if gathered_down_packed_gpu is None or gathered_down_absmax_gpu is None:
                try:
                    gathered_down_packed_gpu = self._copy_cpu_to_gpu(gathered_down_packed_cpu, dtype=torch.uint8)
                    gathered_down_absmax_gpu = self._copy_cpu_to_gpu(gathered_down_absmax_cpu, dtype=torch.float32)
                except Exception as _e:
                    if not _is_cuda_oom_error(_e):
                        raise
                    self._disable_vram_hot_cache("cuda_oom_during_down_dma")
                    gathered_down_packed_gpu = self._copy_cpu_to_gpu(gathered_down_packed_cpu, dtype=torch.uint8)
                    gathered_down_absmax_gpu = self._copy_cpu_to_gpu(gathered_down_absmax_cpu, dtype=torch.float32)

        _triton_was_enabled = bool(self._triton_fused_sparse_mlp)
        triton_out = self._sparse_mlp_forward_fast_triton(
            hidden=hidden,
            flat_hidden=flat_hidden,
            active_local=active_local,
            flat_mask=flat_mask,
            active_neurons=active_neurons,
            gate_bias=getattr(gate_proj, "bias", None),
            up_bias=getattr(up_proj, "bias", None),
            down_bias=getattr(down_proj, "bias", None),
            gate_param=gate_param,
            up_param=up_param,
            down_param=down_param,
            gate_packed_gpu=gate_packed_gpu,
            gate_absmax_gpu=gate_absmax_gpu,
            up_packed_gpu=up_packed_gpu,
            up_absmax_gpu=up_absmax_gpu,
            down_packed_gpu=gathered_down_packed_gpu,
            down_absmax_gpu=gathered_down_absmax_gpu,
        )
        if triton_out is not None:
            return triton_out
        if _triton_was_enabled and (not self._triton_fused_sparse_mlp) and int(_oom_retry_depth) < 1:
            if self._h2d_stream is not None:
                try:
                    self._wait_for_h2d_stream()
                except Exception:
                    pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return self._sparse_mlp_forward_fast(
                layer_idx,
                mlp,
                hidden,
                active_blocks,
                _oom_retry_depth=int(_oom_retry_depth) + 1,
            )
        if self._h2d_stream is not None and (gathered_down_packed_gpu is not None or gathered_down_absmax_gpu is not None):
            try:
                self._wait_for_h2d_stream()
            except Exception as _e:
                if not _is_cuda_oom_error(_e):
                    raise
                self._disable_vram_hot_cache("cuda_oom_after_triton_fallback_wait")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        def _gather_and_dequant(
            param: Dict[str, Any],
            packed_gpu: torch.Tensor,
            absmax_gpu: torch.Tensor,
            bias_param: Optional[torch.Tensor],
        ) -> torch.Tensor:
            out_gpu = torch.empty((K_S, H_in), dtype=self.dtype, device=self.device)
            _bnb_dequant_impl(
                packed_gpu.reshape(-1),
                absmax_gpu.reshape(-1),
                param["quant_block_size"],
                param["quant_type"],
                out_gpu.dtype,
                out=out_gpu,
            )

            bias_gpu = None
            if bias_param is not None:
                bias_gpu = bias_param[active_neurons].to(device=self.device, dtype=self.dtype)

            return F.linear(flat_hidden, out_gpu, bias_gpu) * flat_mask

        try:
            gate = _gather_and_dequant(gate_param, gate_packed_gpu, gate_absmax_gpu, getattr(gate_proj, "bias", None))
            up = _gather_and_dequant(up_param, up_packed_gpu, up_absmax_gpu, getattr(up_proj, "bias", None))

            activated = F.silu(gate) * up
            del gate, up

            code_cpu = down_param["code"]
            absmax_col = gathered_down_absmax_cpu.unsqueeze(-1)

            hi = ((gathered_down_packed_cpu >> 4) & 0x0F).long()
            lo = (gathered_down_packed_cpu & 0x0F).long()
            decoded = torch.stack([code_cpu[hi], code_cpu[lo]], dim=-1).reshape(H_out, num_active_blocks, block_size)
            down_weight_active_cpu = (decoded * absmax_col).reshape(H_out, K_S).to(dtype=self.dtype)
            down_weight_active = down_weight_active_cpu.to(device=self.device, non_blocking=False)

            down_bias = getattr(down_proj, "bias", None)
            bias_gpu = None if down_bias is None else down_bias.to(device=self.device, dtype=self.dtype)

            out = F.linear(activated, down_weight_active, bias_gpu)
            return out.view_as(hidden)
        except Exception as _e:
            if not _is_cuda_oom_error(_e):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _k_dim = int(active_blocks.shape[-1]) if active_blocks.ndim > 0 else 1
            if int(_oom_retry_depth) < 2 and _k_dim > 1:
                _new_k = max(1, _k_dim // 2)
                if int(self._sparse_runtime_top_k) <= 0 or _new_k < int(self._sparse_runtime_top_k):
                    self._sparse_runtime_top_k = int(_new_k)
                    print(
                        f"[sparse] runtime top_k clamped to {int(self._sparse_runtime_top_k)} after CUDA OOM.",
                        flush=True,
                    )
                _reduced_blocks = active_blocks[..., :_new_k].contiguous()
                print(
                    f"[sparse] CUDA OOM in sparse MLP fallback; retrying with top_k={_new_k}.",
                    flush=True,
                )
                return self._sparse_mlp_forward_fast(
                    layer_idx,
                    mlp,
                    hidden,
                    _reduced_blocks,
                    _oom_retry_depth=int(_oom_retry_depth) + 1,
                )
            print(
                "[sparse] CUDA OOM in sparse MLP fallback at minimum retry budget; "
                "using zero MLP contribution for this layer.",
                flush=True,
            )
            return torch.zeros_like(hidden)

    def _sparse_mlp_forward_fast_triton(
        self,
        *,
        hidden: torch.Tensor,
        flat_hidden: torch.Tensor,
        active_local: torch.Tensor,
        flat_mask: torch.Tensor,
        active_neurons: torch.Tensor,
        gate_bias: Optional[torch.Tensor],
        up_bias: Optional[torch.Tensor],
        down_bias: Optional[torch.Tensor],
        gate_param: Dict[str, Any],
        up_param: Dict[str, Any],
        down_param: Dict[str, Any],
        gate_packed_gpu: torch.Tensor,
        gate_absmax_gpu: torch.Tensor,
        up_packed_gpu: torch.Tensor,
        up_absmax_gpu: torch.Tensor,
        down_packed_gpu: Optional[torch.Tensor],
        down_absmax_gpu: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self._triton_fused_sparse_mlp:
            return None
        if triton_sparse_output_linear_4bit is None or triton_sparse_input_linear_4bit is None:
            return None
        if down_packed_gpu is None or down_absmax_gpu is None:
            return None

        try:
            block_size = int(self._sparse_block_size)
            active_dim = int(flat_mask.shape[1])

            gate_bias_gpu = None
            if gate_bias is not None:
                gate_bias_gpu = gate_bias[active_neurons].to(device=self.device, dtype=flat_hidden.dtype)
            up_bias_gpu = None
            if up_bias is not None:
                up_bias_gpu = up_bias[active_neurons].to(device=self.device, dtype=flat_hidden.dtype)
            down_bias_gpu = None
            if down_bias is not None:
                down_bias_gpu = down_bias.to(device=self.device, dtype=flat_hidden.dtype)
            gate_code_gpu = gate_param.get("code_gpu")
            if gate_code_gpu is None:
                gate_code_gpu = gate_param["code"].to(device=self.device, dtype=torch.float32)
            up_code_gpu = up_param.get("code_gpu")
            if up_code_gpu is None:
                up_code_gpu = up_param["code"].to(device=self.device, dtype=torch.float32)
            down_code_gpu = down_param.get("code_gpu")
            if down_code_gpu is None:
                down_code_gpu = down_param["code"].to(device=self.device, dtype=torch.float32)

            gate = triton_sparse_output_linear_4bit(
                flat_hidden,
                active_local,
                flat_mask,
                packed_weight=gate_packed_gpu.reshape(-1),
                absmax=gate_absmax_gpu.reshape(-1),
                code=gate_code_gpu,
                input_dim=int(gate_param["in_features"]),
                quant_block_size=int(gate_param["quant_block_size"]),
                bias=gate_bias_gpu,
                block_size=block_size,
                quant_weight_ref=None,
            )
            up = triton_sparse_output_linear_4bit(
                flat_hidden,
                active_local,
                flat_mask,
                packed_weight=up_packed_gpu.reshape(-1),
                absmax=up_absmax_gpu.reshape(-1),
                code=up_code_gpu,
                input_dim=int(up_param["in_features"]),
                quant_block_size=int(up_param["quant_block_size"]),
                bias=up_bias_gpu,
                block_size=block_size,
                quant_weight_ref=None,
            )
            activated = F.silu(gate) * up

            # Ensure down projection NF4 bytes/scales have landed on device.
            self._wait_for_h2d_stream()
            down = triton_sparse_input_linear_4bit(
                activated,
                active_local,
                packed_weight=down_packed_gpu.reshape(-1),
                absmax=down_absmax_gpu.reshape(-1),
                code=down_code_gpu,
                out_features=int(down_param["out_features"]),
                in_features=int(active_dim),
                quant_block_size=int(block_size),
                bias=down_bias_gpu,
                block_size=block_size,
                quant_weight_ref=None,
            )
            return down.view_as(hidden)
        except Exception as exc:
            print(f"[sparse] Triton fused 4-bit path failed; falling back to dequant path: {exc}", flush=True)
            self._triton_fused_sparse_mlp = False
            return None
    def _dense_mlp_forward_streaming_fast(
        self,
        layer_idx: int,
        mlp: nn.Module,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        gate_proj = getattr(mlp, "gate_proj", None)
        up_proj = getattr(mlp, "up_proj", None)
        down_proj = getattr(mlp, "down_proj", None)
        if gate_proj is None or up_proj is None or down_proj is None:
            return self._dense_mlp_forward_streaming(mlp, hidden)
        if not self._ensure_mlp_proj_staging():
            return torch.zeros_like(hidden)

        flat_hidden = hidden.view(-1, hidden.shape[-1])
        prefix = f"model.layers.{int(layer_idx)}.mlp."

        # Reuse the pre-allocated projection staging buffer. This avoids a large
        # cudaMalloc/free per layer invocation and keeps the collector on a stable
        # memory footprint during long streaming runs.
        max_numel = max(
            gate_proj.weight.numel(),
            up_proj.weight.numel(),
            down_proj.weight.numel(),
        )
        if self._mlp_proj_staging is None or int(self._mlp_proj_staging.numel()) < int(max_numel):
            raise RuntimeError("MLP projection staging buffer is unavailable or undersized")
        staging = self._mlp_proj_staging[:max_numel]

        def _linear_stream(x: torch.Tensor, linear: nn.Module, param_name: str) -> torch.Tensor:
            weight = getattr(linear, "weight", None)
            if weight is None:
                raise RuntimeError("Dense MLP streaming path requires linear.weight")
            w0, w1 = weight.shape[0], weight.shape[1]
            weight_view = staging[: weight.numel()].view(w0, w1)
            raw = self.loader.load_parameter(param_name)
            self._record_h2d_bytes(
                int(raw.numel() * raw.element_size()),
                layer_idx=layer_idx,
                tag="mlp_dense_weight",
            )
            # copy_() does a direct CPU→GPU DMA — no intermediate GPU tensor needed.
            weight_view.copy_(raw.to(dtype=self.dtype))
            del raw
            bias_gpu = self._load_optional_bias(
                param_name,
                getattr(linear, "bias", None),
                device=x.device,
                dtype=x.dtype,
                layer_idx=layer_idx,
                tag="mlp_dense_bias",
            )
            y = F.linear(x, weight_view.to(dtype=x.dtype), bias_gpu)
            del bias_gpu
            return y

        gate = _linear_stream(flat_hidden, gate_proj, f"{prefix}gate_proj.weight")
        up = _linear_stream(flat_hidden, up_proj, f"{prefix}up_proj.weight")
        act = F.silu(gate) * up
        del gate, up
        out = _linear_stream(act, down_proj, f"{prefix}down_proj.weight")
        return out.view_as(hidden)

    def _dense_guard_mlp_forward_exact_chunked_4bit(
        self,
        layer_idx: int,
        mlp: nn.Module,
        hidden: torch.Tensor,
        *,
        chunk_blocks: Optional[int] = None,
        _oom_retry_depth: int = 0,
    ) -> torch.Tensor:
        gate_proj = getattr(mlp, "gate_proj", None)
        up_proj = getattr(mlp, "up_proj", None)
        down_proj = getattr(mlp, "down_proj", None)
        if gate_proj is None or up_proj is None or down_proj is None:
            return self._dense_mlp_forward_streaming(mlp, hidden)

        prefix = f"model.layers.{int(layer_idx)}.mlp."
        if self.device.type != "cuda":
            flat_hidden = hidden.view(-1, hidden.shape[-1]).to(device=self.device, dtype=self.dtype)
            gate_weight = self.loader.load_parameter(f"{prefix}gate_proj.weight").to(device=self.device, dtype=flat_hidden.dtype)
            up_weight = self.loader.load_parameter(f"{prefix}up_proj.weight").to(device=self.device, dtype=flat_hidden.dtype)
            down_weight = self.loader.load_parameter(f"{prefix}down_proj.weight").to(device=self.device, dtype=flat_hidden.dtype)
            gate_bias = self._load_optional_bias(
                f"{prefix}gate_proj.weight",
                getattr(gate_proj, "bias", None),
                device=self.device,
                dtype=flat_hidden.dtype,
                layer_idx=layer_idx,
                tag="mlp_guard_gate_bias",
            )
            up_bias = self._load_optional_bias(
                f"{prefix}up_proj.weight",
                getattr(up_proj, "bias", None),
                device=self.device,
                dtype=flat_hidden.dtype,
                layer_idx=layer_idx,
                tag="mlp_guard_up_bias",
            )
            down_bias = self._load_optional_bias(
                f"{prefix}down_proj.weight",
                getattr(down_proj, "bias", None),
                device=self.device,
                dtype=flat_hidden.dtype,
                layer_idx=layer_idx,
                tag="mlp_guard_down_bias",
            )
            gate = F.linear(flat_hidden, gate_weight, gate_bias)
            up = F.linear(flat_hidden, up_weight, up_bias)
            out = F.linear(F.silu(gate) * up, down_weight, down_bias)
            return out.view_as(hidden)

        block_size = int(self._sparse_block_size)
        flat_hidden = hidden.view(-1, hidden.shape[-1]).to(device=self.device, dtype=self.dtype)
        rows = int(flat_hidden.shape[0])
        if rows <= 0:
            return torch.zeros_like(hidden)

        gate_param = self._get_sparse_4bit_param(f"{prefix}gate_proj.weight")
        up_param = self._get_sparse_4bit_param(f"{prefix}up_proj.weight")
        down_param = self._get_sparse_4bit_param(f"{prefix}down_proj.weight")

        if int(gate_param["in_features"]) != int(hidden.shape[-1]) or int(up_param["in_features"]) != int(hidden.shape[-1]):
            raise RuntimeError(
                f"Layer {int(layer_idx)} dense guard input shapes do not match hidden size {int(hidden.shape[-1])}."
            )
        total_blocks = int(gate_param["blocks_per_col"])
        if int(down_param["out_features"]) != int(hidden.shape[-1]):
            raise RuntimeError(
                f"Layer {int(layer_idx)} dense guard down_proj out_features "
                f"{int(down_param['out_features'])} do not match hidden size {int(hidden.shape[-1])}."
            )
        if int(down_param["in_features"]) % max(block_size, 1) != 0:
            raise RuntimeError(
                f"Layer {int(layer_idx)} dense guard down_proj in_features "
                f"{int(down_param['in_features'])} are not divisible by block_size {block_size}."
            )
        expected_intermediate_blocks = int(down_param["in_features"]) // max(block_size, 1)
        if (
            total_blocks <= 0
            or total_blocks != int(up_param["blocks_per_col"])
            or total_blocks != expected_intermediate_blocks
        ):
            raise RuntimeError(
                f"Layer {int(layer_idx)} dense guard block layout is inconsistent: "
                f"gate={int(gate_param['blocks_per_col'])}, up={int(up_param['blocks_per_col'])}, "
                f"down={expected_intermediate_blocks}, block_size={block_size}."
            )

        chunk_blocks = int(chunk_blocks if chunk_blocks is not None else self._guard_mlp_chunk_blocks)
        chunk_blocks = max(1, min(chunk_blocks, total_blocks))

        gate_bias_full = self._load_optional_bias(
            f"{prefix}gate_proj.weight",
            getattr(gate_proj, "bias", None),
            device=self.device,
            dtype=flat_hidden.dtype,
            layer_idx=layer_idx,
            tag="mlp_guard_gate_bias",
        )
        up_bias_full = self._load_optional_bias(
            f"{prefix}up_proj.weight",
            getattr(up_proj, "bias", None),
            device=self.device,
            dtype=flat_hidden.dtype,
            layer_idx=layer_idx,
            tag="mlp_guard_up_bias",
        )
        down_bias = self._load_optional_bias(
            f"{prefix}down_proj.weight",
            getattr(down_proj, "bias", None),
            device=self.device,
            dtype=flat_hidden.dtype,
            layer_idx=layer_idx,
            tag="mlp_guard_down_bias",
        )

        try:
            out_flat = torch.zeros((rows, int(down_param["out_features"])), device=self.device, dtype=flat_hidden.dtype)
            neuron_offsets_gpu = torch.arange(block_size, device=self.device, dtype=torch.long)
            bytes_per_cblk = block_size // 2
            col_b_range = torch.arange(bytes_per_cblk, dtype=torch.long)
            down_quant_block_size = int(down_param["quant_block_size"])
            down_absmax_per_row = int(down_param["in_features"]) // max(down_quant_block_size, 1)
            down_raw_2d = down_param["packed_weight"].view(int(down_param["out_features"]), int(down_param["bytes_per_row"]))
            down_absmax_2d = down_param["absmax"].view(int(down_param["out_features"]), down_absmax_per_row)

            # These chunk indices are intermediate-space FFN blocks over the original
            # 4-bit gate/up/down weights. They are distinct from the learned-basis
            # output-space block ids used by _route_sparse_mlp / _forward_learned_basis_mlp.
            for block_start in range(0, total_blocks, chunk_blocks):
                block_stop = min(total_blocks, block_start + chunk_blocks)
                ordered_blocks = torch.arange(block_start, block_stop, dtype=torch.long)
                active_dim = int(ordered_blocks.numel()) * block_size
                active_neurons = (
                    ordered_blocks.to(device=self.device, dtype=torch.long).unsqueeze(-1) * block_size
                    + neuron_offsets_gpu.unsqueeze(0)
                ).reshape(-1)

                gate_packed_gpu, gate_absmax_gpu = self._prepare_sparse_blocks_for_param(
                    gate_param,
                    ordered_blocks=ordered_blocks,
                )
                up_packed_gpu, up_absmax_gpu = self._prepare_sparse_blocks_for_param(
                    up_param,
                    ordered_blocks=ordered_blocks,
                )
                self._wait_for_h2d_stream()

                gate_weight = torch.empty(
                    (active_dim, int(gate_param["in_features"])),
                    dtype=self.dtype,
                    device=self.device,
                )
                _bnb_dequant_impl(
                    gate_packed_gpu.reshape(-1),
                    gate_absmax_gpu.reshape(-1),
                    int(gate_param["quant_block_size"]),
                    gate_param["quant_type"],
                    gate_weight.dtype,
                    out=gate_weight,
                )
                gate_bias = (
                    None
                    if gate_bias_full is None
                    else gate_bias_full.index_select(0, active_neurons).to(dtype=flat_hidden.dtype)
                )
                gate = F.linear(flat_hidden, gate_weight.to(dtype=flat_hidden.dtype), gate_bias)
                del gate_weight, gate_packed_gpu, gate_absmax_gpu, gate_bias

                up_weight = torch.empty(
                    (active_dim, int(up_param["in_features"])),
                    dtype=self.dtype,
                    device=self.device,
                )
                _bnb_dequant_impl(
                    up_packed_gpu.reshape(-1),
                    up_absmax_gpu.reshape(-1),
                    int(up_param["quant_block_size"]),
                    up_param["quant_type"],
                    up_weight.dtype,
                    out=up_weight,
                )
                up_bias = (
                    None
                    if up_bias_full is None
                    else up_bias_full.index_select(0, active_neurons).to(dtype=flat_hidden.dtype)
                )
                up = F.linear(flat_hidden, up_weight.to(dtype=flat_hidden.dtype), up_bias)
                del up_weight, up_packed_gpu, up_absmax_gpu, up_bias

                activated = F.silu(gate) * up
                del gate, up

                down_col_starts = ordered_blocks * bytes_per_cblk
                down_col_idx = (down_col_starts.unsqueeze(-1) + col_b_range.unsqueeze(0)).reshape(-1)
                gathered_down_packed_cpu = (
                    down_raw_2d[:, down_col_idx]
                    .reshape(int(down_param["out_features"]), int(ordered_blocks.numel()), bytes_per_cblk)
                    .contiguous()
                )
                down_abs_idx = (ordered_blocks * block_size) // down_quant_block_size
                gathered_down_absmax_cpu = down_absmax_2d[:, down_abs_idx].contiguous()
                down_packed_gpu = self._copy_cpu_to_gpu(
                    gathered_down_packed_cpu,
                    dtype=torch.uint8,
                    layer_idx=layer_idx,
                    tag="mlp_guard_down_packed",
                )
                down_absmax_gpu = self._copy_cpu_to_gpu(
                    gathered_down_absmax_cpu,
                    dtype=torch.float32,
                    layer_idx=layer_idx,
                    tag="mlp_guard_down_absmax",
                )
                self._wait_for_h2d_stream()
                down_weight_chunk = torch.empty(
                    int(down_param["out_features"]),
                    active_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
                _bnb_dequant_impl(
                    down_packed_gpu.reshape(-1),
                    down_absmax_gpu.reshape(-1),
                    int(down_param["quant_block_size"]),
                    down_param["quant_type"],
                    down_weight_chunk.dtype,
                    out=down_weight_chunk,
                )
                out_flat += F.linear(activated, down_weight_chunk.to(dtype=flat_hidden.dtype), None)
                del activated, down_packed_gpu, down_absmax_gpu, down_weight_chunk

            if down_bias is not None:
                out_flat = out_flat + down_bias.unsqueeze(0)
            return out_flat.view_as(hidden).to(device=hidden.device, dtype=hidden.dtype)
        except Exception as exc:
            if not _is_cuda_oom_error(exc):
                raise
            self._disable_vram_hot_cache("cuda_oom_dense_guard_mlp")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if chunk_blocks <= 1:
                raise RuntimeError(
                    f"Dense guard MLP exact chunked 4-bit path OOM at minimum chunk size for layer {int(layer_idx)}."
                ) from exc
            next_chunk_blocks = max(1, chunk_blocks // 2)
            print(
                f"[dense_guard] CUDA OOM in layer {int(layer_idx)} with chunk_blocks={chunk_blocks}; "
                f"retrying with chunk_blocks={next_chunk_blocks}.",
                flush=True,
            )
            return self._dense_guard_mlp_forward_exact_chunked_4bit(
                layer_idx,
                mlp,
                hidden,
                chunk_blocks=next_chunk_blocks,
                _oom_retry_depth=int(_oom_retry_depth) + 1,
            )

    def _mlp_forward_dispatch(
        self,
        layer_idx: int,
        layer: LlamaDecoderLayer,
        mlp_input: torch.Tensor,
    ) -> torch.Tensor:
        if layer_idx in self._sparse_routing:
            if str(getattr(self, "_sparse_basis_execution", "full_output_latent")) == "full_output_latent":
                return self._forward_learned_basis_mlp_full_output(layer_idx, mlp_input)
            active_blocks = self._route_sparse_mlp(mlp_input, layer_idx)
            if active_blocks is None:
                raise RuntimeError(f"Sparse basis routing disappeared for layer {int(layer_idx)}")
            return self._sparse_mlp_forward_fast(layer_idx, layer.mlp, mlp_input, active_blocks)
        return self._dense_guard_mlp_forward_exact_chunked_4bit(layer_idx, layer.mlp, mlp_input)

    # ── Sparse Attention Head helpers ─────────────────────────────────────────

    def _should_use_attn_share_for_layer(self, layer_idx: int) -> bool:
        if int(layer_idx) not in self._attn_share_layer_state:
            return False
        if (
            str(self._traffic_current_phase or "idle") == "prefill"
            and self._attn_share_prefill_mode == "dense"
        ):
            return False
        return True

    def _load_shared_attn_qo(self, layer_idx: int) -> None:
        state = self._attn_share_layer_state.get(int(layer_idx))
        if state is None:
            raise RuntimeError(f"No attention-sharing state registered for layer {int(layer_idx)}")
        group = self._attn_share_groups.get(str(state["group_id"]))
        if group is None:
            raise RuntimeError(
                f"Layer {int(layer_idx)} references missing attention-sharing group {state['group_id']!r}"
            )

        head_dim = int(getattr(self.config, "head_dim", 0))
        q_skel = self._layer_skeleton.self_attn.q_proj.weight
        o_skel = self._layer_skeleton.self_attn.o_proj.weight
        head_perm = torch.as_tensor(state["head_perm"], dtype=torch.long, device=torch.device("cpu")).contiguous()

        if "q_base_u_heads" in group:
            q_base_u_cpu = _unpermute_headwise_tensor(group["q_base_u_heads"], head_perm)
            q_base_v_cpu = _unpermute_headwise_tensor(group["q_base_v_heads"], head_perm)
            q_base_u_gpu = self._copy_cpu_to_gpu(
                q_base_u_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_q_base_u_heads",
            )
            q_base_v_gpu = self._copy_cpu_to_gpu(
                q_base_v_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_q_base_v_heads",
            )
            self._wait_for_h2d_stream()
            q_recon = torch.bmm(q_base_u_gpu, q_base_v_gpu).reshape_as(q_skel)
            q_skel.copy_(q_recon)

            q_resid_u_heads_cpu = state.get("q_resid_u_heads")
            q_resid_v_heads_cpu = state.get("q_resid_v_heads")
            if torch.is_tensor(q_resid_u_heads_cpu) and torch.is_tensor(q_resid_v_heads_cpu):
                q_resid_u_heads_cpu = _unpermute_headwise_tensor(q_resid_u_heads_cpu, head_perm)
                q_resid_v_heads_cpu = _unpermute_headwise_tensor(q_resid_v_heads_cpu, head_perm)
                q_resid_u_heads_gpu = self._copy_cpu_to_gpu(
                    q_resid_u_heads_cpu,
                    dtype=self.dtype,
                    layer_idx=layer_idx,
                    tag="attn_share_q_resid_u_heads",
                )
                q_resid_v_heads_gpu = self._copy_cpu_to_gpu(
                    q_resid_v_heads_cpu,
                    dtype=self.dtype,
                    layer_idx=layer_idx,
                    tag="attn_share_q_resid_v_heads",
                )
                self._wait_for_h2d_stream()
                q_skel.add_(torch.bmm(q_resid_u_heads_gpu, q_resid_v_heads_gpu).reshape_as(q_skel))

            o_base_u_cpu = _unpermute_headwise_tensor(group["o_base_u_heads"], head_perm)
            o_base_v_cpu = _unpermute_headwise_tensor(group["o_base_v_heads"], head_perm)
            o_base_u_gpu = self._copy_cpu_to_gpu(
                o_base_u_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_o_base_u_heads",
            )
            o_base_v_gpu = self._copy_cpu_to_gpu(
                o_base_v_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_o_base_v_heads",
            )
            self._wait_for_h2d_stream()
            o_recon = torch.bmm(o_base_u_gpu, o_base_v_gpu).permute(1, 0, 2).reshape_as(o_skel)
            o_skel.copy_(o_recon)

            o_resid_u_heads_cpu = state.get("o_resid_u_heads")
            o_resid_v_heads_cpu = state.get("o_resid_v_heads")
            if torch.is_tensor(o_resid_u_heads_cpu) and torch.is_tensor(o_resid_v_heads_cpu):
                o_resid_u_heads_cpu = _unpermute_headwise_tensor(o_resid_u_heads_cpu, head_perm)
                o_resid_v_heads_cpu = _unpermute_headwise_tensor(o_resid_v_heads_cpu, head_perm)
                o_resid_u_heads_gpu = self._copy_cpu_to_gpu(
                    o_resid_u_heads_cpu,
                    dtype=self.dtype,
                    layer_idx=layer_idx,
                    tag="attn_share_o_resid_u_heads",
                )
                o_resid_v_heads_gpu = self._copy_cpu_to_gpu(
                    o_resid_v_heads_cpu,
                    dtype=self.dtype,
                    layer_idx=layer_idx,
                    tag="attn_share_o_resid_v_heads",
                )
                self._wait_for_h2d_stream()
                o_skel.add_(torch.bmm(o_resid_u_heads_gpu, o_resid_v_heads_gpu).permute(1, 0, 2).reshape_as(o_skel))

            self._attn_loaded_q_rows = None
            self._attn_loaded_o_cols = None
            self._attn_qo_state = "shared"
            return

        q_base_u_cpu = _unpermute_q_factor_rows(group["q_base_u"], head_perm, head_dim=head_dim)
        q_base_v_cpu = group["q_base_v"]
        q_base_u_gpu = self._copy_cpu_to_gpu(
            q_base_u_cpu,
            dtype=self.dtype,
            layer_idx=layer_idx,
            tag="attn_share_q_base_u",
        )
        q_base_v_gpu = self._copy_cpu_to_gpu(
            q_base_v_cpu,
            dtype=self.dtype,
            layer_idx=layer_idx,
            tag="attn_share_q_base_v",
        )
        self._wait_for_h2d_stream()
        torch.mm(q_base_u_gpu, q_base_v_gpu, out=q_skel)

        q_resid_u_cpu = state.get("q_resid_u")
        q_resid_v_cpu = state.get("q_resid_v")
        if torch.is_tensor(q_resid_u_cpu) and torch.is_tensor(q_resid_v_cpu):
            q_resid_u_cpu = _unpermute_q_factor_rows(q_resid_u_cpu, head_perm, head_dim=head_dim)
            q_resid_u_gpu = self._copy_cpu_to_gpu(
                q_resid_u_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_q_resid_u",
            )
            q_resid_v_gpu = self._copy_cpu_to_gpu(
                q_resid_v_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_q_resid_v",
            )
            self._wait_for_h2d_stream()
            q_skel.addmm_(q_resid_u_gpu, q_resid_v_gpu)

        o_base_u_cpu = group["o_base_u"]
        o_base_v_cpu = _unpermute_o_factor_cols(group["o_base_v"], head_perm, head_dim=head_dim)
        o_base_u_gpu = self._copy_cpu_to_gpu(
            o_base_u_cpu,
            dtype=self.dtype,
            layer_idx=layer_idx,
            tag="attn_share_o_base_u",
        )
        o_base_v_gpu = self._copy_cpu_to_gpu(
            o_base_v_cpu,
            dtype=self.dtype,
            layer_idx=layer_idx,
            tag="attn_share_o_base_v",
        )
        self._wait_for_h2d_stream()
        torch.mm(o_base_u_gpu, o_base_v_gpu, out=o_skel)

        o_resid_u_cpu = state.get("o_resid_u")
        o_resid_v_cpu = state.get("o_resid_v")
        if torch.is_tensor(o_resid_u_cpu) and torch.is_tensor(o_resid_v_cpu):
            o_resid_v_cpu = _unpermute_o_factor_cols(o_resid_v_cpu, head_perm, head_dim=head_dim)
            o_resid_u_gpu = self._copy_cpu_to_gpu(
                o_resid_u_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_o_resid_u",
            )
            o_resid_v_gpu = self._copy_cpu_to_gpu(
                o_resid_v_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_o_resid_v",
            )
            self._wait_for_h2d_stream()
            o_skel.addmm_(o_resid_u_gpu, o_resid_v_gpu)

        self._attn_loaded_q_rows = None
        self._attn_loaded_o_cols = None
        self._attn_qo_state = "shared"

    def _load_shared_attn_kv(self, layer_idx: int) -> None:
        state = self._attn_share_layer_state.get(int(layer_idx))
        if state is None:
            raise RuntimeError(f"No attention-sharing state registered for layer {int(layer_idx)}")
        group = self._attn_share_groups.get(str(state["group_id"]))
        if group is None:
            raise RuntimeError(
                f"Layer {int(layer_idx)} references missing attention-sharing group {state['group_id']!r}"
            )

        head_dim = int(getattr(self.config, "head_dim", 0))
        k_skel = self._layer_skeleton.self_attn.k_proj.weight
        v_skel = self._layer_skeleton.self_attn.v_proj.weight
        kv_head_perm = torch.as_tensor(state["kv_head_perm"], dtype=torch.long, device=torch.device("cpu")).contiguous()

        if "k_base_u_heads" in group:
            k_base_u_cpu = _unpermute_headwise_tensor(group["k_base_u_heads"], kv_head_perm)
            k_base_v_cpu = _unpermute_headwise_tensor(group["k_base_v_heads"], kv_head_perm)
            k_base_u_gpu = self._copy_cpu_to_gpu(
                k_base_u_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_k_base_u_heads",
            )
            k_base_v_gpu = self._copy_cpu_to_gpu(
                k_base_v_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_k_base_v_heads",
            )
            self._wait_for_h2d_stream()
            k_skel.copy_(torch.bmm(k_base_u_gpu, k_base_v_gpu).reshape_as(k_skel))

            k_resid_u_heads_cpu = state.get("k_resid_u_heads")
            k_resid_v_heads_cpu = state.get("k_resid_v_heads")
            if torch.is_tensor(k_resid_u_heads_cpu) and torch.is_tensor(k_resid_v_heads_cpu):
                k_resid_u_heads_cpu = _unpermute_headwise_tensor(k_resid_u_heads_cpu, kv_head_perm)
                k_resid_v_heads_cpu = _unpermute_headwise_tensor(k_resid_v_heads_cpu, kv_head_perm)
                k_resid_u_heads_gpu = self._copy_cpu_to_gpu(
                    k_resid_u_heads_cpu,
                    dtype=self.dtype,
                    layer_idx=layer_idx,
                    tag="attn_share_k_resid_u_heads",
                )
                k_resid_v_heads_gpu = self._copy_cpu_to_gpu(
                    k_resid_v_heads_cpu,
                    dtype=self.dtype,
                    layer_idx=layer_idx,
                    tag="attn_share_k_resid_v_heads",
                )
                self._wait_for_h2d_stream()
                k_skel.add_(torch.bmm(k_resid_u_heads_gpu, k_resid_v_heads_gpu).reshape_as(k_skel))

            v_base_u_cpu = _unpermute_headwise_tensor(group["v_base_u_heads"], kv_head_perm)
            v_base_v_cpu = _unpermute_headwise_tensor(group["v_base_v_heads"], kv_head_perm)
            v_base_u_gpu = self._copy_cpu_to_gpu(
                v_base_u_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_v_base_u_heads",
            )
            v_base_v_gpu = self._copy_cpu_to_gpu(
                v_base_v_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_v_base_v_heads",
            )
            self._wait_for_h2d_stream()
            v_skel.copy_(torch.bmm(v_base_u_gpu, v_base_v_gpu).reshape_as(v_skel))

            v_resid_u_heads_cpu = state.get("v_resid_u_heads")
            v_resid_v_heads_cpu = state.get("v_resid_v_heads")
            if torch.is_tensor(v_resid_u_heads_cpu) and torch.is_tensor(v_resid_v_heads_cpu):
                v_resid_u_heads_cpu = _unpermute_headwise_tensor(v_resid_u_heads_cpu, kv_head_perm)
                v_resid_v_heads_cpu = _unpermute_headwise_tensor(v_resid_v_heads_cpu, kv_head_perm)
                v_resid_u_heads_gpu = self._copy_cpu_to_gpu(
                    v_resid_u_heads_cpu,
                    dtype=self.dtype,
                    layer_idx=layer_idx,
                    tag="attn_share_v_resid_u_heads",
                )
                v_resid_v_heads_gpu = self._copy_cpu_to_gpu(
                    v_resid_v_heads_cpu,
                    dtype=self.dtype,
                    layer_idx=layer_idx,
                    tag="attn_share_v_resid_v_heads",
                )
                self._wait_for_h2d_stream()
                v_skel.add_(torch.bmm(v_resid_u_heads_gpu, v_resid_v_heads_gpu).reshape_as(v_skel))

            self._kv_loaded_cols = None
            return

        k_base_u_cpu = _unpermute_q_factor_rows(group["k_base_u"], kv_head_perm, head_dim=head_dim)
        k_base_v_cpu = group["k_base_v"]
        k_base_u_gpu = self._copy_cpu_to_gpu(
            k_base_u_cpu,
            dtype=self.dtype,
            layer_idx=layer_idx,
            tag="attn_share_k_base_u",
        )
        k_base_v_gpu = self._copy_cpu_to_gpu(
            k_base_v_cpu,
            dtype=self.dtype,
            layer_idx=layer_idx,
            tag="attn_share_k_base_v",
        )
        self._wait_for_h2d_stream()
        torch.mm(k_base_u_gpu, k_base_v_gpu, out=k_skel)

        k_resid_u_cpu = state.get("k_resid_u")
        k_resid_v_cpu = state.get("k_resid_v")
        if torch.is_tensor(k_resid_u_cpu) and torch.is_tensor(k_resid_v_cpu):
            k_resid_u_cpu = _unpermute_q_factor_rows(k_resid_u_cpu, kv_head_perm, head_dim=head_dim)
            k_resid_u_gpu = self._copy_cpu_to_gpu(
                k_resid_u_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_k_resid_u",
            )
            k_resid_v_gpu = self._copy_cpu_to_gpu(
                k_resid_v_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_k_resid_v",
            )
            self._wait_for_h2d_stream()
            k_skel.addmm_(k_resid_u_gpu, k_resid_v_gpu)

        v_base_u_cpu = _unpermute_q_factor_rows(group["v_base_u"], kv_head_perm, head_dim=head_dim)
        v_base_v_cpu = group["v_base_v"]
        v_base_u_gpu = self._copy_cpu_to_gpu(
            v_base_u_cpu,
            dtype=self.dtype,
            layer_idx=layer_idx,
            tag="attn_share_v_base_u",
        )
        v_base_v_gpu = self._copy_cpu_to_gpu(
            v_base_v_cpu,
            dtype=self.dtype,
            layer_idx=layer_idx,
            tag="attn_share_v_base_v",
        )
        self._wait_for_h2d_stream()
        torch.mm(v_base_u_gpu, v_base_v_gpu, out=v_skel)

        v_resid_u_cpu = state.get("v_resid_u")
        v_resid_v_cpu = state.get("v_resid_v")
        if torch.is_tensor(v_resid_u_cpu) and torch.is_tensor(v_resid_v_cpu):
            v_resid_u_cpu = _unpermute_q_factor_rows(v_resid_u_cpu, kv_head_perm, head_dim=head_dim)
            v_resid_u_gpu = self._copy_cpu_to_gpu(
                v_resid_u_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_v_resid_u",
            )
            v_resid_v_gpu = self._copy_cpu_to_gpu(
                v_resid_v_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_v_resid_v",
            )
            self._wait_for_h2d_stream()
            v_skel.addmm_(v_resid_u_gpu, v_resid_v_gpu)

        self._kv_loaded_cols = None

    def _get_attn_active_heads(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Return sorted active head indices [K] for this layer, or None (dense).

        Blends static calibration importance with a live Taylor state_z signal.
        state_z[0, g, :].norm() reflects how much KV group g has accumulated
        context - groups that have not fired much are down-weighted.
        """
        if self._attn_sparse_disabled_reason is not None:
            return None

        static_indices = self._attn_active_head_indices.get(layer_idx)
        if static_indices is None:
            return None

        static_imp = self._attn_head_importance.get(layer_idx)
        if static_imp is None:
            k0 = max(1, min(int(self._attn_active_heads), int(static_indices.numel())))
            return static_indices[:k0].sort().values

        taylor_cache = self._taylor_caches[layer_idx]
        if taylor_cache is None:
            k0 = max(1, min(int(self._attn_active_heads), int(static_indices.numel())))
            return static_indices[:k0].sort().values

        state_z = taylor_cache.state_z
        num_kv_heads = int(state_z.shape[1])
        group_norms = state_z[0].norm(dim=-1).float().to(device=torch.device("cpu"))

        heads_per_group = max(1, int(static_imp.shape[0]) // max(num_kv_heads, 1))
        head_norms = group_norms.repeat_interleave(heads_per_group)
        if int(head_norms.numel()) < int(static_imp.shape[0]):
            pad = int(static_imp.shape[0]) - int(head_norms.numel())
            head_norms = torch.cat([head_norms, head_norms.new_zeros((pad,))], dim=0)
        elif int(head_norms.numel()) > int(static_imp.shape[0]):
            head_norms = head_norms[: int(static_imp.shape[0])]

        static_indices_cpu = static_indices.to(device=torch.device("cpu"), dtype=torch.long)
        static_imp_pool = static_imp.to(device=torch.device("cpu")).index_select(0, static_indices_cpu)
        head_norms_pool = head_norms.index_select(0, static_indices_cpu)
        norm_s = static_imp_pool / static_imp_pool.max().clamp_min(1e-8)
        norm_d = head_norms_pool / head_norms_pool.max().clamp_min(1e-8)
        combined = norm_s * norm_d

        total_heads = int(combined.shape[0])
        max_heads = int(self._attn_max_active_heads) if int(self._attn_max_active_heads) > 0 else total_heads
        max_heads = max(1, min(max_heads, total_heads))
        min_heads = max(1, min(int(self._attn_min_active_heads), max_heads))

        if self._attn_dynamic_threshold > 0.0:
            dynamic_floor = combined.max().clamp_min(1e-8) * float(self._attn_dynamic_threshold)
            dynamic_count = int((combined >= dynamic_floor).sum().item())
            target_k = max(min_heads, min(max_heads, dynamic_count))
        else:
            target_k = max(min_heads, min(max_heads, int(self._attn_active_heads)))

        self._attn_runtime_head_counts[layer_idx] = int(target_k)
        local_top = torch.topk(combined, k=target_k, largest=True).indices
        return static_indices_cpu.index_select(0, local_top).sort().values

    def _shrink_attn_sparse_budget_on_oom(self, *, head_dim: int, hidden_size: int) -> bool:
        """Reduce sparse-attention head budget to recover VRAM on CUDA OOM."""
        cur_max = int(self._attn_max_active_heads) if int(self._attn_max_active_heads) > 0 else int(self._attn_active_heads)
        min_heads = max(1, int(self._attn_min_active_heads))
        if cur_max <= min_heads:
            return False

        new_max = max(min_heads, cur_max // 2)
        if new_max >= cur_max:
            return False

        self._attn_max_active_heads = int(new_max)
        self._attn_active_heads = max(1, min(int(self._attn_active_heads), int(new_max)))
        for _lidx, _indices in list(self._attn_active_head_indices.items()):
            if int(_indices.numel()) > int(new_max):
                self._attn_active_head_indices[_lidx] = _indices[: int(new_max)].contiguous()

        if self.device.type == "cuda":
            try:
                self._attn_q_head_staging = torch.empty(
                    int(new_max) * int(head_dim) * int(hidden_size),
                    dtype=self.dtype,
                    device=self.device,
                )
            except Exception:
                return False
            torch.cuda.empty_cache()

        print(
            f"[sparse_attn] CUDA OOM recovery: reducing max active heads to {int(new_max)} and retrying.",
            flush=True,
        )
        return True
    def _get_sparse_4bit_attn_meta(self, full_name: str, *, head_dim: int) -> Dict[str, Any]:
        """Cache NF4 metadata for an attention projection weight.

        Stores only the dequantised absmax (flat float32, ~16 MB) and the NF4
        codebook (16 floats).  Raw packed bytes are never stored here — they are
        fetched O(1) from loader._ram_cache on each call, keeping the per-entry
        overhead to ~64 MB instead of doubling the 128 MB byte buffer.
        """
        cached = self._attn_sparse_param_meta.get(full_name)
        if cached is not None:
            return cached

        raw_weight, quant_aux = self.loader._load_raw_for_param(full_name)
        if not quant_aux:
            raise RuntimeError(f"Sparse 4-bit attention path: expected NF4 weights for {full_name!r}")

        quant_state = bnb_functional.QuantState.from_dict(quant_aux, device=torch.device("cpu"))
        absmax = quant_state.absmax
        if quant_state.nested:
            absmax = bnb_functional.dequantize_blockwise(quant_state.absmax, quant_state.state2)
            absmax = absmax + quant_state.offset

        out_features = int(quant_state.shape[0])
        in_features  = int(quant_state.shape[1])

        cached = {
            "out_features":    out_features,
            "in_features":     in_features,
            "head_dim":        int(head_dim),
            "num_heads_total": out_features // int(head_dim),
            "quant_block_size": int(quant_state.blocksize),
            "quant_type":      str(quant_state.quant_type),
            "code":            self.loader.prepare_h2d_source(
                quant_state.code.to(dtype=torch.float32).contiguous(),
                dtype=torch.float32,
                pin_override=False,
            ),  # [16]
            # Flat dequantised absmax — ~16 MB for 405B q_proj.  Caching this avoids
            # repeating the nested dequant on every token (expensive CPU op).
            "absmax_flat":     self.loader.prepare_h2d_source(
                absmax.to(dtype=torch.float32).contiguous(),
                dtype=torch.float32,
                pin_override=False,
            ),
        }
        self._attn_sparse_param_meta[full_name] = cached
        return cached

    def _load_sparse_attn_heads(
        self,
        layer_idx: int,
        active_heads: torch.Tensor,
        *,
        head_dim: int,
        hidden_size: int,
    ) -> None:
        """Load only the active K heads' NF4 bytes for q_proj and o_proj.

        q_proj (row gather → GPU dequant):
            Gathers K head-row slices of NF4 bytes on CPU, transfers to the
            pre-allocated _nf4_staging buffer (no cudaMalloc), dequantises with
            _bnb_dequant_impl into _attn_q_head_staging, then scatters into the
            zeroed skeleton q_proj.weight at the active row positions.

        o_proj (column gather → CPU decode → GPU scatter):
            Adapts the down_proj column-gather pattern from _sparse_mlp_forward_fast.
            Gathers K head-column slices on CPU, decodes NF4 nibbles via the code
            table, scales by absmax, transfers FP16 result, and scatters into the
            zeroed skeleton o_proj.weight.

        k_proj and v_proj are loaded normally by _load_layer() — they are small
        (8.4 MB each for 405B GQA with 8 KV heads) and always needed in full.
        """
        if self._nf4_staging is None or self._attn_q_head_staging is None:
            return  # CPU-only mode: caller falls back to full dense load

        prefix = f"model.layers.{int(layer_idx)}."
        q_name = f"{prefix}self_attn.q_proj.weight"
        o_name = f"{prefix}self_attn.o_proj.weight"

        meta_q = self._get_sparse_4bit_attn_meta(q_name, head_dim=head_dim)
        meta_o = self._get_sparse_4bit_attn_meta(o_name, head_dim=head_dim)

        active_cpu = active_heads.to(device=torch.device("cpu"), dtype=torch.long)
        active_list = [int(v) for v in active_cpu.tolist()]
        K = int(active_cpu.shape[0])

        # ── q_proj: Row-block gather ──────────────────────────────────────────
        q_skel = self._layer_skeleton.self_attn.q_proj.weight  # [num_heads*head_dim, hidden]
        o_skel = self._layer_skeleton.self_attn.o_proj.weight  # [hidden, num_heads*head_dim]
        self._clear_sparse_attn_qo_buffers(q_skel=q_skel, o_skel=o_skel)

        hot_cache = self._maybe_cache_sparse_attn_hot_heads(
            layer_idx=layer_idx,
            q_name=q_name,
            o_name=o_name,
            meta_q=meta_q,
            meta_o=meta_o,
        )
        if hot_cache is not None:
            lookup = hot_cache["lookup_cpu"].index_select(0, active_cpu)
            if bool(torch.all(lookup >= 0).item()):
                hot_slots = lookup.to(device=self.device, dtype=torch.long)
                active_gpu = active_heads.to(device=self.device, dtype=torch.long)

                q_packed_gpu = hot_cache["q_packed_gpu"].index_select(0, hot_slots).reshape(-1)
                q_absmax_gpu = hot_cache["q_absmax_gpu"].index_select(0, hot_slots).reshape(-1)
                q_partial = self._attn_q_head_staging[: K * head_dim * hidden_size].view(K * head_dim, hidden_size)
                _bnb_dequant_impl(
                    q_packed_gpu,
                    q_absmax_gpu,
                    meta_q["quant_block_size"],
                    meta_q["quant_type"],
                    self.dtype,
                    out=q_partial,
                )
                row_idx = (
                    active_gpu.unsqueeze(-1) * head_dim
                    + torch.arange(head_dim, device=self.device)
                ).reshape(-1)
                q_skel.index_copy_(0, row_idx, q_partial)
                self._attn_loaded_q_rows = row_idx.detach().clone()

                o_packed_gpu = hot_cache["o_packed_gpu"].index_select(1, hot_slots).reshape(-1)
                o_absmax_gpu = hot_cache["o_absmax_gpu"].index_select(1, hot_slots).reshape(-1)
                o_partial = self._attn_q_head_staging[: hidden_size * K * head_dim].view(hidden_size, K * head_dim)
                _bnb_dequant_impl(
                    o_packed_gpu,
                    o_absmax_gpu,
                    meta_o["quant_block_size"],
                    meta_o["quant_type"],
                    self.dtype,
                    out=o_partial,
                )
                col_idx_gpu = (
                    active_gpu.unsqueeze(-1) * head_dim
                    + torch.arange(head_dim, device=self.device)
                ).reshape(-1)
                o_skel.index_copy_(1, col_idx_gpu, o_partial)
                self._attn_loaded_o_cols = col_idx_gpu.detach().clone()
                self._attn_qo_state = "sparse"
                return

        q_raw, _ = self.loader._load_raw_for_param(q_name)       # RAM cache hit
        bytes_per_head_q = meta_q["in_features"] // 2 * head_dim  # 1.048 MB for 405B
        packed_2d_q = q_raw.view(meta_q["num_heads_total"], bytes_per_head_q)  # view, no copy
        q_rows_cpu = packed_2d_q.index_select(0, active_cpu).contiguous()

        absmax_per_head_q = head_dim * meta_q["in_features"] // meta_q["quant_block_size"]
        absmax_2d_q = meta_q["absmax_flat"].view(meta_q["num_heads_total"], absmax_per_head_q)
        q_abs_cpu = absmax_2d_q.index_select(0, active_cpu).contiguous()

        gathered_q = self._copy_cpu_to_existing_gpu(self._nf4_staging[: q_rows_cpu.numel()], q_rows_cpu.reshape(-1))
        n_q = gathered_q.numel()
        self._record_h2d_bytes(
            int(gathered_q.numel() * gathered_q.element_size()),
            layer_idx=layer_idx,
            tag="attn_sparse_q_packed",
        )
        absmax_q_gpu = self._copy_cpu_to_gpu(
            q_abs_cpu.reshape(-1),
            dtype=torch.float32,
            layer_idx=layer_idx,
            tag="attn_sparse_q_absmax",
        )
        q_partial = self._attn_q_head_staging[: K * head_dim * hidden_size].view(K * head_dim, hidden_size)
        _bnb_dequant_impl(
            self._nf4_staging[:n_q], absmax_q_gpu,
            meta_q["quant_block_size"], meta_q["quant_type"], self.dtype, out=q_partial,
        )
        active_gpu = active_heads.to(device=self.device, dtype=torch.long)
        row_idx = (
            active_gpu.unsqueeze(-1) * head_dim
            + torch.arange(head_dim, device=self.device)
        ).reshape(-1)   # [K*head_dim]
        q_skel.index_copy_(0, row_idx, q_partial)
        self._attn_loaded_q_rows = row_idx.detach().clone()

        # ── o_proj: Column-block gather → GPU NF4 decode ─────────────────────
        # Previously decoded NF4 on CPU, creating ~2 GB of int64 intermediates
        # per layer (.long() expansion of [H_out, K, 64] uint8 → 512 MB each).
        # 126 layers × ~2 GB = minutes of CPU work per prefill.
        # Fix: same GPU staging path as q_proj above.
        o_raw, _ = self.loader._load_raw_for_param(o_name)
        H_out     = meta_o["out_features"]                     # hidden_size = 16384
        H_in      = meta_o["in_features"]                      # num_heads * head_dim = 16384
        qbs       = meta_o["quant_block_size"]                 # 64

        bytes_per_row_o    = H_in // 2                         # 8192
        bytes_per_head_col = head_dim // 2                     # 64 bytes per head-col per row
        raw_2d_o = o_raw.view(H_out, bytes_per_row_o)          # [H_out, 8192] — no copy
        o_col_offsets = (
            active_cpu.unsqueeze(-1) * bytes_per_head_col
            + torch.arange(bytes_per_head_col, dtype=torch.long)
        ).reshape(-1)
        o_cols_cpu = raw_2d_o.index_select(1, o_col_offsets).contiguous()

        # Absmax: head_dim=128 spans 2 quant groups (qbs=64) per row.
        absmax_per_head_col = head_dim // qbs                  # 2
        absmax_per_row_o    = H_in // qbs                      # 256
        absmax_2d_o = meta_o["absmax_flat"].view(H_out, absmax_per_row_o)
        o_abs_offsets = (
            active_cpu.unsqueeze(-1) * absmax_per_head_col
            + torch.arange(absmax_per_head_col, dtype=torch.long)
        ).reshape(-1)
        o_abs_cpu = absmax_2d_o.index_select(1, o_abs_offsets).contiguous()

        # GPU staging + GPU NF4 decode.  gathered_o is [H_out, K*head_dim//2] in
        # row-major order — exactly what bitsandbytes expects for a [H_out, K*head_dim] weight.
        gathered_o = self._copy_cpu_to_existing_gpu(self._nf4_staging[: o_cols_cpu.numel()], o_cols_cpu.reshape(-1))
        n_o = gathered_o.numel()                               # H_out * K * head_dim // 2
        self._record_h2d_bytes(
            int(gathered_o.numel() * gathered_o.element_size()),
            layer_idx=layer_idx,
            tag="attn_sparse_o_packed",
        )
        absmax_o_gpu = self._copy_cpu_to_gpu(
            o_abs_cpu.reshape(-1),
            dtype=torch.float32,
            layer_idx=layer_idx,
            tag="attn_sparse_o_absmax",
        )
        self._wait_for_h2d_stream()
        # Reuse _attn_q_head_staging — same numel as K*head_dim*hidden_size, different view.
        o_partial = self._attn_q_head_staging[: H_out * K * head_dim].view(H_out, K * head_dim)
        _bnb_dequant_impl(
            self._nf4_staging[:n_o], absmax_o_gpu,
            qbs, meta_o["quant_type"], self.dtype, out=o_partial,
        )
        if self._debug_sync_cuda:
            torch.cuda.synchronize()
        col_idx_gpu = (
            active_gpu.unsqueeze(-1) * head_dim
            + torch.arange(head_dim, device=self.device)
        ).reshape(-1)   # [K*head_dim]
        o_skel.index_copy_(1, col_idx_gpu, o_partial)
        self._attn_loaded_o_cols = col_idx_gpu.detach().clone()
        self._attn_qo_state = "sparse"

    def _batch_preload_layer(self, layer_idx: int) -> None:
        """Batch-load ALL tensors for a layer into RAM cache with a single safe_open per shard.

        On Windows, repeated safe_open calls on the same large shard crash after ~5 opens
        because Windows does not reliably reclaim the mmap VA region between opens of a
        5 GB file.  By grouping all reads from the same shard into one _load_exact_tensors
        call we reduce shard 1 opens from N (one per weight) to 1 — avoiding the crash.
        """
        if not self.loader._ram_cache_enabled:
            return
        prefix = f"model.layers.{int(layer_idx)}."

        # Collect base weight names for this layer that are not yet RAM-cached.
        uncached_bases: List[str] = []
        for k, _dest, is_fp in self._layer_state_items:
            if not is_fp:
                continue
            full_name = f"{prefix}{k}"
            with self.loader._ram_cache_lock:
                already = full_name in self.loader._ram_cache
            if not already:
                uncached_bases.append(full_name)

        if not uncached_bases:
            return

        # Expand each base name with its quant-aux tensor names so a single
        # _load_exact_tensors call fetches everything in one pass per shard.
        all_names: List[str] = []
        for base in uncached_bases:
            all_names.append(base)
            all_names.extend(self.loader._quant_aux_by_base.get(base, []))

        try:
            tensors = self.loader._load_exact_tensors(all_names)
        except Exception:
            return  # fall back to per-weight loading in the normal loop

        # Populate RAM cache in the same format _load_raw_for_param would.
        for base in uncached_bases:
            if base not in tensors:
                continue
            aux_names_for_base = self.loader._quant_aux_by_base.get(base, [])
            weight = tensors[base]
            quant_aux = {n: tensors[n] for n in aux_names_for_base if n in tensors}

            weight = self.loader._maybe_pin_cpu_tensor(weight.contiguous())
            quant_aux_pinned = {
                n: self.loader._maybe_pin_cpu_tensor(v.contiguous())
                for n, v in quant_aux.items()
            }

            with self.loader._ram_cache_lock:
                if base in self.loader._ram_cache:
                    continue
                self.loader._ram_cache[base] = (weight, quant_aux_pinned)
                if self.loader._ram_cache_limit_bytes is not None:
                    self.loader._ram_cache_lru[base] = None
                    self.loader._ram_cache_lru.move_to_end(base, last=True)
                    nbytes = ShardedSafetensorLoader._entry_nbytes(weight, quant_aux_pinned)
                    self.loader._ram_cache_entry_bytes[base] = nbytes
                    self.loader._ram_cache_current_bytes += nbytes
                    self.loader._evict_ram_cache_locked()

    def _load_layer(self, layer_idx: int) -> LlamaDecoderLayer:
        self._layer_skeleton.layer_idx = int(layer_idx)
        if hasattr(self._layer_skeleton, "self_attn"):
            self._layer_skeleton.self_attn.layer_idx = int(layer_idx)
        self._record_layer_visit(layer_idx)
        # Pre-warm RAM cache for all layer tensors in a single batch so we
        # never open the same shard more than once per layer (Windows crash fix).
        if self._enable_windows_batch_preload:
            self._batch_preload_layer(layer_idx)

        if self._show_progress and torch.cuda.is_available():
            try:
                _free_vram, _total_vram = torch.cuda.mem_get_info(self.device)
                _alloc_vram = torch.cuda.memory_allocated(self.device)
                print(
                    f"  [vram] layer {int(layer_idx):03d}/{self.num_layers}"
                    f"  free={_free_vram / 1e9:.2f} GB"
                    f"  alloc={_alloc_vram / 1e9:.2f} GB"
                    f"  total={_total_vram / 1e9:.2f} GB"
                    f"  phase={self._traffic_current_phase}",
                    flush=True,
                )
            except Exception:
                pass

        layer_state_items = self._layer_state_items

        # Determine if sparse attention is active for this layer.
        # _get_attn_active_heads returns None when no importance data is loaded
        # or for layers not profiled — both fall back to full dense loading.
        _allow_sparse_attn = not (
            str(self._traffic_current_phase or "idle") == "prefill"
            and self._sparse_attn_prefill_mode == "dense"
        )
        _use_shared_attn = self._should_use_attn_share_for_layer(layer_idx)
        _active_attn_heads = (
            None
            if _use_shared_attn
            else self._get_attn_active_heads(layer_idx) if _allow_sparse_attn else None
        )
        _head_dim = int(getattr(self.config, "head_dim", 128))
        _hidden   = int(getattr(self.config, "hidden_size", 16384))
        _allow_sparse_kv = not (
            str(self._traffic_current_phase or "idle") == "prefill"
            and self._sparse_kv_prefill_mode == "dense"
        )
        # Keys to skip in the main loop (handled by _load_sparse_attn_heads below).
        _skip_attn: set = set()
        if _active_attn_heads is not None:
            _skip_attn = _skip_attn | {"self_attn.q_proj.weight", "self_attn.o_proj.weight"}
        if _use_shared_attn:
            _skip_attn = _skip_attn | {
                "self_attn.q_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
            }
        # Skip K/V weights when sparse KV routing is active — loaded on-demand
        # by _populate_sparse_kv_skeleton() inside forward_token/_forward_prefill.
        # Retrieval layers always load full k/v weights so new-token keys are exact.
        if layer_idx in self._kv_routing and _allow_sparse_kv and not _use_shared_attn and layer_idx not in self._retrieval_layers:
            _skip_attn = _skip_attn | {"self_attn.k_proj.weight", "self_attn.v_proj.weight"}
            # Zero the skeleton so stale data from the previous layer doesn't bleed in.
            # _populate_sparse_kv_skeleton will then fill only the active col-blocks.
            self._layer_skeleton.self_attn.k_proj.weight.zero_()
            self._layer_skeleton.self_attn.v_proj.weight.zero_()
            self._kv_loaded_cols = None  # invalidate previous layer's col offsets

        # ── Tier 1: RAM cache → GPU  (miss falls through to SSD on first pass) ──
        prefix = f"model.layers.{int(layer_idx)}."
        for k, dest, is_fp in layer_state_items:
            full_name = f"{prefix}{k}"
            if k.startswith("mlp.") and is_fp:
                continue
            if k in _skip_attn:
                continue  # handled below by _load_sparse_attn_heads
            if is_fp:
                try:
                    self.loader.load_parameter_into(
                        name=full_name,
                        out=dest,
                        dtype=self.dtype,
                        staging=self._nf4_staging,
                        absmax_staging=self._absmax_staging,
                        nested_absmax_staging=self._nested_absmax_staging,
                        state2_absmax_staging=self._state2_absmax_staging,
                        code_staging=self._code_staging,
                        byte_counter=lambda n, _layer_idx=layer_idx, _k=k: self._record_h2d_bytes(
                            int(n),
                            layer_idx=_layer_idx,
                            tag=f"layer_load:{_k}",
                        ),
                    )
                except Exception as _e:
                    if not _is_cuda_oom_error(_e):
                        raise
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    try:
                        self.loader.load_parameter_into(
                            name=full_name,
                            out=dest,
                            dtype=self.dtype,
                            staging=self._nf4_staging,
                            absmax_staging=self._absmax_staging,
                            nested_absmax_staging=self._nested_absmax_staging,
                            state2_absmax_staging=self._state2_absmax_staging,
                            code_staging=self._code_staging,
                            byte_counter=lambda n, _layer_idx=layer_idx, _k=k: self._record_h2d_bytes(
                                int(n),
                                layer_idx=_layer_idx,
                                tag=f"layer_load:{_k}",
                            ),
                        )
                    except Exception as _e2:
                        if not _is_cuda_oom_error(_e2):
                            raise
                        print(
                            f"[layer_load] CUDA OOM loading {full_name}; zeroing tensor and continuing.",
                            flush=True,
                        )
                        dest.zero_()
            else:
                raw = self.loader.load_parameter(full_name)
                dest.copy_(raw)

        if _active_attn_heads is None and not _use_shared_attn:
            self._attn_loaded_q_rows = None
            self._attn_loaded_o_cols = None
            self._attn_qo_state = "dense"

        if _use_shared_attn:
            self._load_shared_attn_qo(layer_idx)
            self._load_shared_attn_kv(layer_idx)

        # Sparse attention head loading: only transfer NF4 bytes for K active heads.
        # q_proj and o_proj are zeroed first; only the active head rows/columns are filled.
        if _active_attn_heads is not None and not _use_shared_attn:
            if int(layer_idx) in self._attn_zero_only_layers:
                self._clear_sparse_attn_qo_buffers(
                    q_skel=self._layer_skeleton.self_attn.q_proj.weight,
                    o_skel=self._layer_skeleton.self_attn.o_proj.weight,
                )
            else:
                _attn_retry = _active_attn_heads
                while True:
                    try:
                        self._wait_h2d_stream_for_current()
                        self._load_sparse_attn_heads(
                            layer_idx, _attn_retry, head_dim=_head_dim, hidden_size=_hidden,
                        )
                        break
                    except Exception as _e:
                        if not _is_cuda_oom_error(_e):
                            raise
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if self._shrink_attn_sparse_budget_on_oom(head_dim=_head_dim, hidden_size=_hidden):
                            _attn_retry = self._get_attn_active_heads(layer_idx)
                            if _attn_retry is not None and int(_attn_retry.numel()) > 0:
                                continue

                        # Last-resort fallback: keep sparse path for other layers,
                        # zero q/o for this layer only to avoid dense q/o OOM.
                        self._attn_runtime_head_counts[int(layer_idx)] = 0
                        self._attn_zero_only_layers.add(int(layer_idx))
                        print(
                            f"[sparse_attn] CUDA OOM at min head budget on layer {layer_idx}; "
                            f"zeroing q/o for this layer and continuing.",
                            flush=True,
                        )
                        self._clear_sparse_attn_qo_buffers(
                            q_skel=self._layer_skeleton.self_attn.q_proj.weight,
                            o_skel=self._layer_skeleton.self_attn.o_proj.weight,
                        )
                        break

        # Kick off RAM prefetch for the next layer on the background thread.
        next_idx = layer_idx + 1
        if next_idx < self.num_layers:
            self._schedule_prefetch_layer(next_idx)

        return self._layer_skeleton

    def _release_modules(self, *modules: nn.Module) -> None:
        # Avoid aggressive GC/empty_cache in the hot path
        pass

    def _embed_tokens_cpu(self, token_ids: torch.LongTensor) -> torch.Tensor:
        token_ids_cpu = token_ids.to(device=torch.device("cpu"), dtype=torch.long)
        if self._embed_weight_cpu is not None:
            return F.embedding(token_ids_cpu, self._embed_weight_cpu)

        flat_ids = token_ids_cpu.reshape(-1)
        unique_ids = flat_ids.unique(sorted=False)
        missing_ids = []
        with self._embed_row_cache_lock:
            for token_id in unique_ids.tolist():
                if int(token_id) not in self._embed_row_cache:
                    missing_ids.append(int(token_id))
        if missing_ids:
            fetched_rows = self.loader.load_rows(self._embed_weight_name, missing_ids).to(dtype=self.dtype)
            with self._embed_row_cache_lock:
                for idx, token_id in enumerate(missing_ids):
                    self._embed_row_cache[int(token_id)] = fetched_rows[idx].contiguous()

        row_tensors = []
        with self._embed_row_cache_lock:
            for token_id in flat_ids.tolist():
                row_tensors.append(self._embed_row_cache[int(token_id)])
        return torch.stack(row_tensors, dim=0).view(int(token_ids_cpu.shape[0]), int(token_ids_cpu.shape[1]), -1)

    def _lm_head_forward_cpu(self, hidden: torch.Tensor) -> torch.Tensor:
        lm_head_weight_cpu = self._ensure_lm_head_weight_cpu()
        if lm_head_weight_cpu is None:
            raise RuntimeError("StreamingLlamaRuntime.generate requires materialize_lm_head=True")
        hidden_cpu = hidden.to(device=torch.device("cpu"), dtype=lm_head_weight_cpu.dtype)
        return F.linear(hidden_cpu, lm_head_weight_cpu)

    @staticmethod
    def _sample_next_token(
        logits: torch.Tensor,
        *,
        do_sample: bool,
        temperature: float,
        top_k: Optional[int],
        top_p: float,
    ) -> torch.Tensor:
        if (not do_sample) or temperature <= 0.0:
            return torch.argmax(logits, dim=-1)

        scores = logits / max(float(temperature), 1e-5)
        if top_k is not None and top_k > 0:
            top_k = min(int(top_k), scores.shape[-1])
            topk_vals = torch.topk(scores, top_k, dim=-1).values
            min_topk = topk_vals[:, -1].unsqueeze(-1)
            scores = scores.masked_fill(scores < min_topk, float("-inf"))

        if 0.0 < float(top_p) < 1.0:
            sorted_scores, sorted_idx = torch.sort(scores, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_scores, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumulative > float(top_p)
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            sorted_scores = sorted_scores.masked_fill(sorted_mask, float("-inf"))
            scores = torch.full_like(scores, float("-inf"))
            scores.scatter_(dim=-1, index=sorted_idx, src=sorted_scores)

        probs = torch.softmax(scores, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def forward_token(
        self,
        token_ids: torch.LongTensor,
        *,
        position_index: int,
        capture_layers: Optional[Sequence[int]] = None,
        use_attention_cache: bool = True,
    ) -> tuple[torch.Tensor, Dict[int, Dict[str, torch.Tensor]]]:
        if token_ids.ndim != 2 or int(token_ids.shape[0]) != 1 or int(token_ids.shape[1]) != 1:
            raise RuntimeError("StreamingLlamaRuntime.forward_token currently supports batch_size=1 and seq_len=1 only")

        capture_set = {int(idx) for idx in capture_layers or []}
        captures: Dict[int, Dict[str, torch.Tensor]] = {}
        printed_progress = False
        # Index on CPU (avoids 3.9 GiB GPU allocation), move the resulting 32 KB vector to GPU.
        hidden = self._embed_tokens_cpu(token_ids).to(device=self.device, dtype=self.dtype)
        position_ids = torch.tensor([[int(position_index)]], device=self.device, dtype=torch.long)
        rope = self.rotary_emb(hidden, position_ids)

        for layer_idx in range(self.num_layers):
            if self._show_progress:
                printed_progress = True
            if self._show_progress and torch.cuda.is_available():
                alloc_gb = torch.cuda.memory_allocated() / 1e9
                reserv_gb = torch.cuda.memory_reserved() / 1e9
                print(
                    f"  [layer {layer_idx + 1}/{self.num_layers}] VRAM {alloc_gb:.2f}/{reserv_gb:.2f} GB (alloc/reserv)",
                    end="\r",
                    flush=True,
                )
            elif self._show_progress:
                print(f"  [layer {layer_idx + 1}/{self.num_layers}] loading...", end="\r", flush=True)
            if self._debug_steps:
                print(f"[debug] layer {layer_idx}: load", flush=True)
            layer = self._load_layer(layer_idx)
            taylor_attn: Optional[GQATaylorSSDSelfAttention] = None
            residual = hidden
            hidden_norm = layer.input_layernorm(hidden)
            _use_shared_attn = self._should_use_attn_share_for_layer(layer_idx)
            _active_attn_heads = None if _use_shared_attn else self._get_attn_active_heads(layer_idx)

            # Sparse K/V routing: predict active column-blocks and populate skeleton.
            # Retrieval layers bypass sparse K/V and keep full k_proj/v_proj.
            _active_kv_blocks = None
            if layer_idx not in self._retrieval_layers:
                _active_kv_blocks = self._route_kv_blocks(hidden_norm, layer_idx)
            if _active_kv_blocks is not None:
                self._populate_sparse_kv_skeleton(layer_idx, _active_kv_blocks, layer)

            if layer_idx in self.taylor_layer_set and use_attention_cache:
                if self._debug_steps:
                    print(f"[debug] layer {layer_idx}: build_taylor", flush=True)
                taylor_attn = self._shared_taylor_attn
                taylor_attn.layer_idx = layer_idx
                if self._debug_steps:
                    print(f"[debug] layer {layer_idx}: attn_forward_taylor", flush=True)
                attn_out, _attn_weights, present = taylor_attn(
                    hidden_states=hidden_norm,
                    position_ids=position_ids,
                    position_embeddings=rope,
                    past_key_value=self._taylor_caches[layer_idx],
                    use_cache=True,
                )
                self._taylor_caches[layer_idx] = present
            elif (
                layer_idx in self._retrieval_layers
                and self._token_archive is not None
                and use_attention_cache
            ):
                if self._debug_steps:
                    print(f"[debug] layer {layer_idx}: attn_forward_retrieval", flush=True)
                attn_out = self._forward_retrieval_attn(
                    layer_idx=layer_idx,
                    layer=layer,
                    hidden_norm=hidden_norm,
                    rope=rope,
                    position_index=int(position_ids.view(-1)[0].item()),
                    active_heads=_active_attn_heads,
                )
            else:
                if self._debug_steps:
                    print(f"[debug] layer {layer_idx}: attn_forward_dense", flush=True)
                # No-cache path: used by collect_dense_mlp_rows (use_attention_cache=False)
                # to avoid accumulating ~4 MB Taylor-SSD state per layer × 126 layers
                # which would exhaust the 200 MB pool headroom by layer ~48 and crash.
                attn_out, _attn_weights = layer.self_attn(
                    hidden_states=hidden_norm,
                    position_embeddings=rope,
                    attention_mask=None,
                    past_key_values=self._dense_cache if use_attention_cache else None,
                    cache_position=position_ids.view(-1),
                )

            hidden = residual + attn_out
            residual = hidden
            mlp_input = layer.post_attention_layernorm(hidden)

            if self._debug_steps:
                print(f"[debug] layer {layer_idx}: route", flush=True)
            mlp_out = self._mlp_forward_dispatch(layer_idx, layer, mlp_input)

            if layer_idx in capture_set:
                captures[layer_idx] = {
                    "mlp_input": mlp_input.detach().cpu(),
                    "dense_mlp_out": mlp_out.detach().cpu(),
                }
            hidden = residual + mlp_out
            # Update KV block banking EMA after each decode layer
            if _active_kv_blocks is not None:
                self._update_kv_block_banking(layer_idx, _active_kv_blocks)
            self._release_modules(layer, *( [taylor_attn] if taylor_attn is not None else [] ))

        # Advance block-banking step counter once per decode token
        if self._kv_routing:
            self._kv_bank_step += 1

        if printed_progress:
            print("", flush=True)
        hidden = self.norm(hidden)
        if not self._materialize_lm_head:
            logits = torch.zeros((hidden.shape[0], hidden.shape[1], 1), dtype=torch.float32)
        else:
            logits = self._lm_head_forward(hidden).float()
        return logits, captures

    def forward_sequence(
        self,
        token_ids: torch.LongTensor,
        *,
        selected_layers_set: Optional[set] = None,
    ) -> None:
        """Run a full-sequence calibration pass with a layer-first loop.

        Each of the 126 layers is loaded from RAM→GPU exactly once per sequence.
        Within each layer, tokens are processed one at a time using the same
        call signature as forward_token() (seq_len=1, no KV cache).
        No KV cache means each token attends only to itself — acceptable for head
        importance calibration where relative per-head magnitudes matter, not
        autoregressive quality. Dense MLP rows are still collected through the
        staging-based calibration path in this mode.

        Speedup vs forward_token(): seq_len× fewer PCIe layer loads.

        selected_layers_set: if provided, layers not in this set are skipped entirely
        (no PCIe load, hidden states pass through unchanged). Saves ~(1 - K/N)× bandwidth.
        """
        if token_ids.ndim != 2 or int(token_ids.shape[0]) != 1:
            raise RuntimeError("forward_sequence requires batch_size=1")

        seq_len = int(token_ids.shape[1])
        all_hidden = self._embed_tokens_cpu(token_ids).to(device=self.device, dtype=self.dtype)
        position_ids = torch.arange(seq_len, device=self.device, dtype=torch.long)
        # Pre-allocate second buffer once — avoids cudaMalloc on every layer iteration.
        next_hidden = torch.empty_like(all_hidden)

        for layer_idx in range(self.num_layers):
            # Skip PCIe load entirely for layers not being profiled.
            if selected_layers_set is not None and layer_idx not in selected_layers_set:
                continue

            if torch.cuda.is_available() and layer_idx % 10 == 0:
                alloc_gb = torch.cuda.memory_allocated() / 1e9
                reserv_gb = torch.cuda.memory_reserved() / 1e9
                print(
                    f"  [seq layer {layer_idx + 1}/{self.num_layers}] VRAM {alloc_gb:.2f}/{reserv_gb:.2f} GB",
                    end="\r", flush=True,
                )
            layer = self._load_layer(layer_idx)

            # Process one token at a time (batch=1, seq=1).
            # The batch-as-tokens trick (batch=seq_len) causes the Taylor local-attention
            # to allocate [seq_len, 128_heads, local_window, head_dim] = ~268 MB tensors,
            # overflowing VRAM and causing Windows WDDM to page to system RAM.
            # Single-token processing keeps all Taylor state tensors at [1, ...] = tiny.
            for pos in range(seq_len):
                h = all_hidden[:, pos : pos + 1, :]                     # [1, 1, hidden]
                position_ids_tok = position_ids[pos : pos + 1].unsqueeze(0)  # [1, 1]
                rope_tok = self.rotary_emb(h, position_ids_tok)
                h_norm = layer.input_layernorm(h)
                attn_out, _ = layer.self_attn(
                    hidden_states=h_norm,
                    position_embeddings=rope_tok,
                    attention_mask=None,
                    past_key_values=None,
                )
                next_hidden[:, pos : pos + 1, :] = h + attn_out

            all_hidden, next_hidden = next_hidden, all_hidden  # swap — no allocation
            self._release_modules(layer)

        print("", flush=True)

    def _forward_prefill(self, token_ids: torch.LongTensor, *, position_offset: int = 0) -> torch.Tensor:
        """Process all prompt tokens with each of the 126 layers loaded only once.

        Instead of the naive loop (load layer N for token 1, load layer N for
        token 2, …), we invert the loops: load layer N once, then run all prompt
        tokens through it sequentially before unloading.  For a P-token prompt
        this reduces layer loads from P×126 to 126.

        Attention and MLP are both processed over the whole prompt per layer.
        Sparse q/o heads are still loaded once per layer, and sparse K/V routing
        already unions active column blocks across prompt tokens, so batching the
        actual attention call collapses the hot-path Python/CUDA launch overhead
        without changing the streamed-weight footprint. Taylor-SSD still walks
        the sequence recurrently internally, but doing that inside one module
        call is much cheaper than re-entering the full attention stack token by
        token from Python.
        """
        seq_len = int(token_ids.shape[1])
        all_hidden = self._embed_tokens_cpu(token_ids).to(device=self.device, dtype=self.dtype)
        # [1, seq_len, hidden]
        position_start = int(position_offset)
        position_ids_all = torch.arange(
            position_start,
            position_start + seq_len,
            device=self.device,
            dtype=torch.long,
        )
        position_ids_batch = position_ids_all.unsqueeze(0)

        for layer_idx in range(self.num_layers):
            if self._show_progress and torch.cuda.is_available():
                alloc_gb = torch.cuda.memory_allocated() / 1e9
                reserv_gb = torch.cuda.memory_reserved() / 1e9
                print(
                    f"  [prefill layer {layer_idx + 1}/{self.num_layers}] VRAM {alloc_gb:.2f}/{reserv_gb:.2f} GB",
                    end="\r", flush=True,
                )
            layer = self._load_layer(layer_idx)

            # ── Attention: one token at a time ────────────────────────────
            use_taylor = layer_idx in self.taylor_layer_set
            if use_taylor:
                taylor_attn = self._shared_taylor_attn
                taylor_attn.layer_idx = layer_idx
            h_norm = layer.input_layernorm(all_hidden)
            _active_kv_blocks_prefill = self._route_kv_blocks(h_norm, layer_idx)
            if _active_kv_blocks_prefill is not None:
                self._populate_sparse_kv_skeleton(layer_idx, _active_kv_blocks_prefill, layer)
            rope_all = self.rotary_emb(all_hidden, position_ids_batch)

            if use_taylor:
                attn_out, _, present = taylor_attn(
                    hidden_states=h_norm,
                    position_ids=position_ids_batch,
                    position_embeddings=rope_all,
                    past_key_value=self._taylor_caches[layer_idx],
                    use_cache=True,
                )
                self._taylor_caches[layer_idx] = present
            else:
                attn_out, _ = layer.self_attn(
                    hidden_states=h_norm,
                    position_embeddings=rope_all,
                    attention_mask=None,
                    past_key_values=self._dense_cache,
                    cache_position=position_ids_all.view(-1),
                )
            all_hidden = all_hidden + attn_out

            # all_hidden is now the post-attention residual for all tokens

            # ── Flush deferred CUDA errors from attention before MLP ──────
            # On Windows WDDM a failed async kernel sets a sticky error that
            # surfaces as AcceleratorError on the *next* synchronous CUDA op
            # (even a tiny torch.empty call).  Synchronising here ensures any
            # such error is caught and logged at the right place — and the
            # error flag is cleared — before the MLP path begins.
            if torch.cuda.is_available() and self._debug_sync_cuda:
                try:
                    torch.cuda.synchronize(self.device)
                except Exception as _sync_e:
                    if _is_cuda_oom_error(_sync_e):
                        print(
                            f"[prefill_sync] CUDA error after attention at layer {layer_idx}: "
                            f"{type(_sync_e).__name__!r}: {str(_sync_e)[:200]!r}",
                            flush=True,
                        )
                        # Error is now surfaced and cleared; continue with MLP.
                    else:
                        raise

            # ── MLP: batched over all tokens ──────────────────────────────
            residual = all_hidden
            mlp_input = layer.post_attention_layernorm(all_hidden)
            mlp_out = self._mlp_forward_dispatch(layer_idx, layer, mlp_input)
            self._maybe_fit_local_decode_guard_basis(layer_idx, mlp_input, mlp_out)
            all_hidden = residual + mlp_out

            self._release_modules(layer)

        if self._show_progress:
            print("", flush=True)
        # Flush any hot-cache H2D copies that were queued on _h2d_stream during
        # this prefill pass. All 126 layers' copies run in parallel with their
        # respective attention/MLP compute above; we sync once here before the
        # first decode token reads the hot-cache data.
        self._wait_for_h2d_stream()

        # Populate the token-posting archive from the dense KV cache so that
        # the very first decode token has access to the full prompt context.
        if self._token_archive is not None and self._dense_cache is not None:
            self._token_archive.warm_up_from_dense_cache(
                self._dense_cache, seq_len=int(token_ids.shape[1])
            )
            self._release_dense_cache_for_retrieval_layers()

        all_hidden = self.norm(all_hidden)
        # Only the last token's logits are needed to start generation.
        logits = self._lm_head_forward(all_hidden[:, -1:, :]).float()
        return logits

    # ─────────────────────────────────────────────────────────────────────────
    # Token-posting retrieval attention
    # ─────────────────────────────────────────────────────────────────────────

    def _forward_retrieval_attn(
        self,
        layer_idx: int,
        layer: "LlamaDecoderLayer",
        hidden_norm: torch.Tensor,      # [1, 1, hidden_size]
        rope: tuple,                    # (cos, sin) from self.rotary_emb
        position_index: int,
        active_heads: Optional[torch.Tensor],   # [K] sorted head indices, or None
    ) -> torch.Tensor:
        """Exact attention on a sparse shortlist of archived tokens.

        Replaces the dense full-context ``layer.self_attn(past_key_values=...)``
        call for retrieval layers.  The shortlist is assembled by the
        ``TokenPostingArchive``:
          • sink tokens  — first num_sinks, always on GPU
          • archive candidates — top-M from posting-list probe
          • ring tokens  — most recent ring_size, always on GPU

        The new token's exact K/V is computed with the fully-loaded skeleton
        weights and appended to the archive before returning.
        """
        archive = self._token_archive
        cfg = self.config
        head_dim = int(getattr(cfg, "head_dim", 128))
        num_heads = int(getattr(cfg, "num_attention_heads", 128))
        num_kv = int(getattr(cfg, "num_key_value_heads", 8))
        queries_per_group = num_heads // num_kv   # e.g. 16 for 405B

        # ── Step 1: Compute exact q, k, v for the new token ──────────────────
        cos, sin = rope
        q_raw = layer.self_attn.q_proj(hidden_norm)    # [1, 1, H*D]
        k_raw = layer.self_attn.k_proj(hidden_norm)    # [1, 1, G*D]
        v_raw = layer.self_attn.v_proj(hidden_norm)    # [1, 1, G*D]

        # Reshape to [B, heads, S, D] for apply_rotary_pos_emb.
        q = q_raw.view(1, 1, num_heads, head_dim).transpose(1, 2)    # [1, H, 1, D]
        k = k_raw.view(1, 1, num_kv, head_dim).transpose(1, 2)       # [1, G, 1, D]
        v = v_raw.view(1, 1, num_kv, head_dim).transpose(1, 2)       # [1, G, 1, D]

        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        # q: [1, H, 1, D]   k: [1, G, 1, D]  (post-RoPE)

        # ── Step 2: Determine active query heads and their KV group mapping ───
        if active_heads is not None:
            active_h_list: List[int] = active_heads.tolist()
        else:
            active_h_list = list(range(num_heads))

        # Map each active head to its KV group.
        from collections import defaultdict as _defaultdict
        group_to_heads: Dict[int, List[int]] = _defaultdict(list)
        for h in active_h_list:
            group_to_heads[h // queries_per_group].append(h)

        # ── Step 3: Probe archive per KV group and run exact attention ────────
        step = archive.step
        archive.step += 1

        out_by_head: Dict[int, torch.Tensor] = {}

        for g, heads_in_group in group_to_heads.items():
            # Probe with all active head queries in this KV group and union candidates.
            h_indices = torch.tensor(heads_in_group, device=self.device, dtype=torch.long)
            q_rep = q[0, h_indices, 0, :]    # [H_group, D] FP16 GPU

            # Fetch shortlist K/V from archive (sinks + archive candidates + ring).
            k_ctx, v_ctx = archive.fetch_shortlist_kv(
                layer_idx, g, q_rep, step, M=self._retrieval_candidates,
            )    # [T, D] GPU FP16

            # Also attend to the current new token itself (causal self-inclusion).
            k_new_g = k[0, g, 0:1, :]    # [1, D]
            v_new_g = v[0, g, 0:1, :]

            if k_ctx.shape[0] > 0:
                k_all = torch.cat([k_ctx, k_new_g], dim=0)    # [T+1, D]
                v_all = torch.cat([v_ctx, v_new_g], dim=0)
            else:
                k_all = k_new_g
                v_all = v_new_g

            # Expand to [1, 1, T+1, D] for scaled_dot_product_attention.
            k_4d = k_all.unsqueeze(0).unsqueeze(0)    # [1, 1, T+1, D]
            v_4d = v_all.unsqueeze(0).unsqueeze(0)

            # Per-head exact attention over shared shortlist.
            for h in heads_in_group:
                q_h = q[0:1, h:h+1, :, :]    # [1, 1, 1, D]
                out_h = F.scaled_dot_product_attention(
                    q_h, k_4d, v_4d,
                    scale=head_dim ** -0.5,
                )    # [1, 1, 1, D]
                out_by_head[h] = out_h[0, 0, 0, :]    # [D]

        # ── Step 4: Scatter head outputs and apply o_proj ─────────────────────
        # Build dense output tensor; inactive heads remain zero (o_proj cols
        # for those heads are also zero in sparse mode, so output is correct).
        out_flat = torch.zeros(
            1, 1, num_heads * head_dim,
            dtype=self.dtype, device=self.device,
        )
        for h, out_h in out_by_head.items():
            out_flat[0, 0, h * head_dim : (h + 1) * head_dim] = out_h

        attn_out = layer.self_attn.o_proj(out_flat)    # [1, 1, hidden_size]

        # ── Step 5: Append new token K/V to archive ───────────────────────────
        k_new_cpu = k[0, :, 0, :].detach().cpu()    # [G, D]
        v_new_cpu = v[0, :, 0, :].detach().cpu()
        archive.append_token(layer_idx, position_index, k_new_cpu, v_new_cpu)

        return attn_out

    def generate(
        self,
        input_ids: torch.LongTensor,
        *,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: float = 1.0,
        token_callback: Optional[Callable[[torch.Tensor, torch.LongTensor], None]] = None,
        reuse_session_cache: bool = False,
    ) -> torch.LongTensor:
        if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            raise RuntimeError("StreamingLlamaRuntime.generate currently supports batch_size=1 only")
        if not self._materialize_lm_head:
            raise RuntimeError("StreamingLlamaRuntime.generate requires materialize_lm_head=True")

        generated = input_ids.to(device=self.device)
        prompt_ids_cpu = input_ids.detach().to(device=torch.device("cpu"), dtype=torch.long).contiguous()
        processed_ids_cpu = prompt_ids_cpu
        self._reset_traffic_stats()
        logits: Optional[torch.Tensor] = None
        reuse_prefix_len = 0
        prev_session_len = 0
        with torch.no_grad():
            prompt_len = int(generated.shape[1])
            if prompt_len <= 0:
                raise RuntimeError("No prompt tokens were provided")

            if bool(reuse_session_cache) and self._session_token_ids_cpu is not None:
                prev_session_len = int(self._session_token_ids_cpu.shape[-1])
                reuse_prefix_len = self._longest_common_prefix_len(self._session_token_ids_cpu, prompt_ids_cpu)
                if reuse_prefix_len > 0:
                    if self.taylor_layer_set and reuse_prefix_len < prev_session_len:
                        print("[session] Taylor cache cannot crop on divergence; falling back to cold prefill", flush=True)
                        reuse_prefix_len = 0
                    elif reuse_prefix_len < prev_session_len and not self._crop_attention_caches(reuse_prefix_len):
                        print("[session] cache crop unavailable; falling back to cold prefill", flush=True)
                        reuse_prefix_len = 0

            if reuse_prefix_len <= 0:
                self.reset_caches()
                self._set_traffic_phase("prefill")
                if prompt_len > 1:
                    print(f"[prompt] batched prefill: {prompt_len} tokens × 1 layer pass", flush=True)
                    logits = self._forward_prefill(generated)
                else:
                    logits, _ = self.forward_token(generated[:, 0:1], position_index=0)
            else:
                self._set_traffic_phase("prefill")
                suffix_len = int(prompt_len - reuse_prefix_len)
                if suffix_len > 0:
                    print(
                        f"[prompt] delta prefill: reused {reuse_prefix_len}/{prompt_len} tokens; "
                        f"streaming {suffix_len} new tokens",
                        flush=True,
                    )
                    suffix_ids = generated[:, reuse_prefix_len:]
                    if suffix_len > 1:
                        logits = self._forward_prefill(suffix_ids, position_offset=reuse_prefix_len)
                    else:
                        logits, _ = self.forward_token(suffix_ids[:, 0:1], position_index=reuse_prefix_len)
                elif prev_session_len == prompt_len and self._session_last_logits_cpu is not None:
                    print(f"[prompt] prefix cache hit: reused {reuse_prefix_len}/{prompt_len} tokens", flush=True)
                    logits = self._session_last_logits_cpu.to(device=self.device, dtype=torch.float32)
                else:
                    replay_prefix_len = max(0, int(prompt_len - 1))
                    if not self._crop_attention_caches(replay_prefix_len):
                        self.reset_caches()
                        self._set_traffic_phase("prefill")
                        print(f"[prompt] replay prefill: {prompt_len} tokens × 1 layer pass", flush=True)
                        logits = self._forward_prefill(generated)
                    else:
                        print(
                            f"[prompt] replay tail: reused {replay_prefix_len}/{prompt_len} tokens; "
                            f"replaying final prompt token",
                            flush=True,
                        )
                        logits, _ = self.forward_token(
                            generated[:, replay_prefix_len:replay_prefix_len + 1],
                            position_index=replay_prefix_len,
                        )
            if logits is None:
                raise RuntimeError("No prompt tokens were provided")

            self._set_traffic_phase("decode")
            for _step_idx in range(int(max_new_tokens)):
                next_token = self._sample_next_token(
                    logits[:, -1, :],
                    do_sample=bool(do_sample),
                    temperature=float(temperature),
                    top_k=top_k,
                    top_p=float(top_p),
                ).view(1, 1)
                generated = torch.cat([generated, next_token.to(device=self.device, dtype=generated.dtype)], dim=-1)
                if token_callback is not None:
                    token_callback(next_token, generated)
                processed_ids_cpu = torch.cat(
                    [
                        processed_ids_cpu,
                        next_token.detach().to(device=torch.device("cpu"), dtype=torch.long).contiguous(),
                    ],
                    dim=-1,
                )
                if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
                    break
                if _step_idx + 1 >= int(max_new_tokens):
                    break
                logits, _ = self.forward_token(next_token, position_index=int(generated.shape[1]) - 1)
            self._set_session_state(processed_ids_cpu, logits)
        self._set_traffic_phase("idle")
        self._finalize_traffic_report()
        return generated

    def collect_dense_mlp_rows(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        selected_layers: Sequence[int],
        layer_x: Dict[int, List[torch.Tensor]],
        layer_y: Dict[int, List[torch.Tensor]],
        layer_rows: Dict[int, int],
        max_rows: int,
        # Optional KV co-collection: if provided, h_norm (input_layernorm output)
        # is captured into layer_kv_x during the same forward pass, at zero extra cost.
        layer_kv_x: Optional[Dict[int, List[torch.Tensor]]] = None,
        layer_kv_rows: Optional[Dict[int, int]] = None,
        kv_max_rows: Optional[int] = None,
    ) -> None:
        if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            raise RuntimeError(
                "StreamingLlamaRuntime.collect_dense_mlp_rows currently supports batch_size=1 only"
            )
        if not selected_layers:
            return
        if self._mlp_proj_staging is None:
            raise RuntimeError(
                "collect_dense_mlp_rows: _mlp_proj_staging is None — VRAM was insufficient to "
                "allocate the MLP staging buffer at startup. All collected dense_mlp_out values "
                "would be zeros, producing a useless checkpoint. Free VRAM or reduce other "
                "pre-allocated buffers before collecting."
            )

        valid_len = int(attention_mask[0].sum().item()) if attention_mask is not None else int(input_ids.shape[1])
        self.reset_caches()
        with torch.no_grad():
            selected_set = {int(idx) for idx in selected_layers}
            all_hidden = self._embed_tokens_cpu(input_ids[:, :valid_len]).to(device=self.device, dtype=self.dtype)
            next_hidden = torch.empty_like(all_hidden)
            attn_hidden = torch.empty_like(all_hidden)
            position_ids = torch.arange(valid_len, device=self.device, dtype=torch.long)
            printed_progress = False

            for layer_idx in range(self.num_layers):
                pending_mlp = [idx for idx in selected_layers if int(layer_rows[int(idx)]) < int(max_rows)]
                _kv_max = int(kv_max_rows or max_rows)
                pending_kv: List[int] = (
                    [idx for idx in layer_kv_x
                     if layer_kv_rows is not None
                     and int(layer_kv_rows.get(int(idx), 0)) < _kv_max]
                    if layer_kv_x is not None
                    else []
                )
                # Keep processing layers as long as either MLP or KV data is still needed.
                pending_layers = sorted(set(int(i) for i in pending_mlp) | set(int(i) for i in pending_kv))
                if not pending_layers:
                    break
                if int(layer_idx) > int(max(pending_layers)):
                    break
                rows_done = sum(1 for idx in selected_layers if int(layer_rows[int(idx)]) >= int(max_rows))
                printed_progress = True
                status = (
                    f"[collect] layer {layer_idx+1}/{self.num_layers} "
                    f"({rows_done}/{len(selected_layers)} mlp done,"
                    f" {len(pending_mlp)} mlp / {len(pending_kv)} kv pending)"
                )
                print(status.ljust(120), end="\r", flush=True)
                layer = self._load_layer(layer_idx)

                for pos in range(valid_len):
                    h = all_hidden[:, pos : pos + 1, :]
                    position_ids_tok = position_ids[pos : pos + 1].unsqueeze(0)
                    rope_tok = self.rotary_emb(h, position_ids_tok)
                    h_norm = layer.input_layernorm(h)
                    attn_out, _ = layer.self_attn(
                        hidden_states=h_norm,
                        position_embeddings=rope_tok,
                        attention_mask=None,
                        past_key_values=None,
                    )
                    attn_hidden[:, pos : pos + 1, :] = h + attn_out

                mlp_input = layer.post_attention_layernorm(attn_hidden)
                mlp_out = self._dense_mlp_forward_streaming_fast(layer_idx, layer.mlp, mlp_input)

                if int(layer_idx) in selected_set:
                    remain = int(max_rows) - int(layer_rows[int(layer_idx)])
                    if remain > 0:
                        x = mlp_input.reshape(-1, mlp_input.shape[-1]).float()
                        y = mlp_out.reshape(-1, mlp_out.shape[-1]).float()
                        take = min(int(remain), int(x.shape[0]))
                        if take > 0:
                            layer_x[int(layer_idx)].append(x[:take].detach().cpu())
                            layer_y[int(layer_idx)].append(y[:take].detach().cpu())
                            layer_rows[int(layer_idx)] += int(take)

                # KV co-collection: re-apply input_layernorm to the layer's input
                # (now in next_hidden after the swap) to get h_norm for all tokens at once.
                if (
                    layer_kv_x is not None
                    and layer_kv_rows is not None
                    and int(layer_idx) in layer_kv_x
                ):
                    _kv_remain = int(kv_max_rows or max_rows) - int(layer_kv_rows.get(int(layer_idx), 0))
                    if _kv_remain > 0:
                        # all_hidden is still the pre-attention layer input (swap is below).
                        _h_norm_all = layer.input_layernorm(all_hidden[:, :valid_len, :])
                        _kv_x = _h_norm_all.reshape(-1, _h_norm_all.shape[-1]).float()
                        _kv_take = min(_kv_remain, int(_kv_x.shape[0]))
                        if _kv_take > 0:
                            layer_kv_x[int(layer_idx)].append(_kv_x[:_kv_take].detach().cpu())
                            layer_kv_rows[int(layer_idx)] = layer_kv_rows.get(int(layer_idx), 0) + _kv_take

                next_hidden.copy_(attn_hidden + mlp_out)
                all_hidden, next_hidden = next_hidden, all_hidden
                self._release_modules(layer)
            if printed_progress:
                print("", flush=True)
