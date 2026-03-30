from __future__ import annotations

import concurrent.futures
import gc
import json
import os
import threading
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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

from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding

try:
    from ..gqa_taylor_ssd import GQATaylorSSDSelfAttention, TaylorSSDLayerCache
except ImportError:  # pragma: no cover
    from gqa_taylor_ssd import GQATaylorSSDSelfAttention, TaylorSSDLayerCache

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

                with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                    for key in shard_keys:
                        tensor = handle.get_tensor(key)
                        if os.name == "nt":
                            tensor = tensor.clone().contiguous()
                        out[key] = tensor
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
        attn_active_heads: Optional[int] = None,
        attn_head_activity_threshold: float = 0.10,
        attn_min_active_heads: int = 16,
        attn_max_active_heads: Optional[int] = None,
        enable_triton_fused_sparse_mlp: bool = True,
        enable_cuda_h2d_overlap: bool = True,
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
        self._show_progress = _resolve_show_progress_default()
        self._prefer_gpu_lm_head = _resolve_gpu_lm_head_default()
        self.loader = ShardedSafetensorLoader(self.snapshot_dir, pin_ram_cache=ram_cache_pinned)
        self.loader._ram_cache_enabled = bool(ram_cache)
        self._enable_background_prefetch = bool(ram_cache) and _resolve_background_prefetch_default()
        self._enable_cuda_h2d_overlap = bool(enable_cuda_h2d_overlap and torch.cuda.is_available())
        self._h2d_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=self.device) if self._enable_cuda_h2d_overlap else None
        )
        self.num_layers = int(getattr(self.config, "num_hidden_layers", 0))
        if self.num_layers <= 0:
            raise RuntimeError("Invalid llama config: num_hidden_layers must be > 0")

        self.taylor_layer_set = (
            set(range(self.num_layers))
            if taylor_layers is None
            else {int(idx) for idx in taylor_layers if 0 <= int(idx) < self.num_layers}
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
                    int(float(_vram_total) * 0.20),
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
        _target_layer_mb_raw = os.getenv("STREAMING_TARGET_LAYER_MB", "").strip()
        try:
            self._target_layer_traffic_mb = float(_target_layer_mb_raw) if _target_layer_mb_raw else 30.0
        except ValueError:
            self._target_layer_traffic_mb = 30.0

        # Keep embed_tokens as a direct CPU tensor — avoid an extra multi-GB module copy at startup.
        self._embed_weight_cpu = self.loader.load_parameter("model.embed_tokens.weight").to(dtype=self.dtype)
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
        self._lm_head_weight_cpu: Optional[torch.Tensor] = None
        self._lm_head_weight_gpu: Optional[torch.Tensor] = None
        if self._materialize_lm_head:
            lm_head_weight_name = (
                "lm_head.weight"
                if "lm_head.weight" in self.loader.weight_map
                else "model.embed_tokens.weight"
            )
            if lm_head_weight_name == "model.embed_tokens.weight":
                self._lm_head_weight_cpu = self._embed_weight_cpu
            else:
                self._lm_head_weight_cpu = self.loader.load_parameter(lm_head_weight_name).to(dtype=self.dtype)

        self.rotary_emb = LlamaRotaryEmbedding(self.config, device=self.device)
        self._taylor_caches: List[Optional[TaylorSSDLayerCache]] = [None for _ in range(self.num_layers)]
        self._dense_cache = DynamicCache(config=self.config) if DynamicCache is not None else None
        self._layer_skeleton = LlamaDecoderLayer(self.config, layer_idx=0).to(device=self.device, dtype=self.dtype)
        for p in self._layer_skeleton.parameters():
            p.requires_grad = False
        self._layer_skeleton.eval()
        # Keep the large MLP weights on CPU. The 8 GB target only has enough VRAM
        # for attention + activations + one streamed projection at a time.
        self._layer_skeleton.mlp.to(device=torch.device("cpu"), dtype=self.dtype)
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
        # Try to pre-allocate MLP staging at init. On constrained VRAM setups
        # (e.g. Windows where DWM reserves several GB of the 8 GB card), this
        # allocation may fail. If it does, dense-MLP fallback stays unavailable.
        # Sparse-basis inference now honors the checkpoint's explicit layer
        # selection and zero-passes non-selected layers instead of depending on
        # this buffer to preserve behavior.
        _ffn = int(getattr(self.config, "intermediate_size", _h * 4))
        self._mlp_proj_staging: Optional[torch.Tensor] = None
        # Skip the ~1.6 GB MLP staging buffer when sparse routing is active —
        # sparse inference never uses _dense_mlp_forward_streaming_fast, and
        # skipping this allocation saves critical VRAM on 8 GB cards.
        _skip_mlp_staging = bool(sparse_basis_path and str(sparse_basis_path).strip())
        if torch.cuda.is_available() and not _skip_mlp_staging:
            try:
                self._mlp_proj_staging = torch.empty(
                    _ffn * _h, dtype=self.dtype, device=self.device
                )
            except Exception as _e:
                if "out of memory" in str(_e).lower() or "alloc" in str(_e).lower():
                    import warnings
                    warnings.warn(
                        f"[StreamingLlamaRuntime] Insufficient VRAM for MLP staging "
                        f"({_ffn * _h * 2 // 1024 // 1024} MB); dense MLP layers will "
                        f"use zero-passthrough (acceptable for attn calibration; "
                        f"inference should use --sparse-basis-path)."
                    )
                else:
                    raise

        # ── Sparse MLP routing weights ────────────────────────────────────────
        # Loaded from the learned-basis checkpoint produced by
        # init_learned_basis_from_dense_mlp.py.  Kept in CPU RAM (each layer's
        # routing tensors are ~3 MB — trivial compared to the 1.6 GB MLP weights)
        # and moved to GPU lazily when a layer is processed.
        #
        # For each sparse layer, the checkpoint stores:
        #   encoder_weight [basis_rank, hidden_size] — projects hidden → latent
        #   encoder_bias   [basis_rank]
        #   decoder_blocks [num_blocks, basis_rank, block_size] — predicts block outputs
        #
        # Routing:  latent = enc_w @ h + enc_b
        #           scores = ||einsum('nr,brs->nbs', latent, dec)||  per block
        #           active  = topk(scores) → active neuron indices
        #
        # Execution: only the rows of gate/up and columns of down corresponding
        #            to active neurons are gathered and matmul'd → ~sparse_top_k/
        #            num_blocks fraction of the MLP compute.
        self._sparse_routing: Dict[int, Dict[str, torch.Tensor]] = {}
        self._sparse_top_k: int = 0
        self._sparse_runtime_top_k: int = 0
        self._sparse_block_size: int = 32
        self._sparse_num_blocks: int = 0
        self._sparse_top_k_by_layer: Dict[int, int] = {}
        self._sparse_param_cache: Dict[str, Dict[str, Any]] = {}
        self._sparse_explicit_layer_selection: Set[int] = set()

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
            # top_k: honour explicit override, else use 2 % of checkpoint blocks as default
            _num_blocks = max(1, int(self._sparse_num_blocks))
            _default_top_k = max(1, int(round(_num_blocks * 0.02)))
            self._sparse_top_k = int(sparse_top_k) if sparse_top_k is not None else _default_top_k
            self._sparse_runtime_top_k = int(self._sparse_top_k)
            _layer_states = _payload.get("layer_states", {})
            _stats = _payload.get("stats", {})
            _top_k_by_layer = _cfg.get("basis_top_k_by_layer", {}) or _cfg.get("top_k_by_layer", {}) or {}
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
                _layer_top_k = int(_top_k_by_layer.get(str(_lidx), self._sparse_top_k))
                _layer_top_k = int(max(1, min(_layer_top_k, _layer_num_blocks)))
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
                self._sparse_routing[_lidx] = {
                    # Keep on CPU — each layer's routing tensors are ~12 MB total
                    # (enc_w [96,16384] + dec [1664,96,32]). Storing all 83 layers
                    # on GPU would consume ~1 GB of VRAM permanently. Transfer is
                    # done lazily per-layer in _route_sparse_mlp (~12 MB/layer).
                    "enc_w": _state["encoder_weight"].to(dtype=self.dtype),  # [R, H] CPU
                    "enc_b": _state["encoder_bias"].to(dtype=self.dtype),    # [R]    CPU
                    "dec":   _state["decoder_blocks"].to(dtype=self.dtype),  # [B, R, S] CPU
                }
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
                    "dec_norm_t": _dec.norm(dim=-1).transpose(0, 1).contiguous(),
                    "scale": float(_scale.item()) if torch.is_tensor(_scale) and int(_scale.numel()) == 1 else 1.0,
                    "top_k": _layer_top_k,
                }
                self._sparse_top_k_by_layer[_lidx] = int(_layer_top_k)
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
                  f"| block_size={self._sparse_block_size}", flush=True)
            if self._sparse_explicit_layer_selection:
                print(
                    f"[sparse] explicit layer selection: {len(self._sparse_explicit_layer_selection)}/{self.num_layers} "
                    f"MLP layers; non-selected layers use zero-pass-through",
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
        # _attn_active_head_indices[layer_idx] = Tensor[K] of sorted head indices
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

        if attn_head_importance_path and str(attn_head_importance_path).strip():
            _attn_payload = torch.load(
                str(attn_head_importance_path), map_location="cpu", weights_only=False
            )
            _attn_cfg = _attn_payload.get("config", {})
            _H   = int(_attn_cfg.get("num_heads",    getattr(self.config, "num_attention_heads", 128)))
            _D   = int(_attn_cfg.get("head_dim",     getattr(self.config, "head_dim", 128)))
            _Hid = int(_attn_cfg.get("hidden_size",  getattr(self.config, "hidden_size", 16384)))
            K_default = int(attn_active_heads) if attn_active_heads is not None else max(1, min(self._attn_min_active_heads, _H))
            if self._attn_max_active_heads <= 0:
                _num_kv = int(_attn_cfg.get("num_kv_heads", getattr(self.config, "num_key_value_heads", max(1, _H // 16))))
                _per_head_weight_bytes = (_Hid * _D // 2) + (_Hid * _D // 64 * 4)
                _per_head_qo_mb = float(2 * _per_head_weight_bytes) / float(1024 ** 2)
                _kv_weight_bytes = (_num_kv * _D * _Hid // 2) + (_num_kv * _D * _Hid // 64 * 4)
                _kv_total_mb = float(2 * _kv_weight_bytes) / float(1024 ** 2)
                _budget_heads = max(
                    1,
                    int((float(self._target_layer_traffic_mb) - float(_kv_total_mb)) // max(_per_head_qo_mb, 1e-6)),
                )
                self._attn_max_active_heads = min(K_default, _budget_heads)
            self._attn_max_active_heads = max(1, min(self._attn_max_active_heads, _H))
            self._attn_min_active_heads = max(1, min(self._attn_min_active_heads, self._attn_max_active_heads))
            K = max(1, min(K_default, self._attn_max_active_heads))
            self._attn_active_heads = K

            for _lidx_s, _state in _attn_payload.get("layer_states", {}).items():
                _lidx = int(_lidx_s)
                if not (0 <= _lidx < self.num_layers):
                    continue
                imp = _state["importance"].float()          # [num_heads]
                self._attn_head_importance[_lidx] = imp
                top_k = torch.topk(imp, k=min(self._attn_max_active_heads, _H), largest=True).indices.sort().values
                self._attn_active_head_indices[_lidx] = top_k  # sorted for contiguous gather

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
                f" | target_layer_mb={self._target_layer_traffic_mb:.1f}",
                flush=True,
            )

        self._materialize_lm_head_on_gpu()

    def reset_caches(self) -> None:
        self._taylor_caches = [None for _ in range(self.num_layers)]
        self._dense_cache = DynamicCache(config=self.config) if DynamicCache is not None else None
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

    def _set_session_state(self, token_ids_cpu: torch.LongTensor, logits: Optional[torch.Tensor]) -> None:
        self._session_token_ids_cpu = token_ids_cpu.detach().to(device=torch.device("cpu"), dtype=torch.long).contiguous()
        if logits is None:
            self._session_last_logits_cpu = None
        else:
            self._session_last_logits_cpu = logits[:, -1:, :].detach().to(
                device=torch.device("cpu"),
                dtype=torch.float32,
            ).contiguous()

    def _materialize_lm_head_on_gpu(self) -> None:
        if not self._materialize_lm_head:
            return
        if not self._prefer_gpu_lm_head:
            return
        if self.device.type != "cuda":
            return
        if self._lm_head_weight_cpu is None:
            return
        if self._lm_head_weight_gpu is not None:
            return

        required_bytes = int(self._lm_head_weight_cpu.numel()) * int(self._lm_head_weight_cpu.element_size())
        safety_margin_bytes = int(256 * (1024 ** 2))
        hot_cache_reserve_bytes = 0
        if self._vram_hot_cache_limit_bytes is not None and self._attn_active_head_indices:
            hot_cache_reserve_bytes = int(self._vram_hot_cache_limit_bytes)
        required_residual_bytes = int(max(safety_margin_bytes, self._vram_hot_cache_margin_bytes)) + int(
            hot_cache_reserve_bytes
        )
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
            self._lm_head_weight_gpu = self._lm_head_weight_cpu.to(
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
        if self._lm_head_weight_gpu is not None:
            return F.linear(hidden.to(dtype=self._lm_head_weight_gpu.dtype), self._lm_head_weight_gpu)
        if self._lm_head_weight_cpu is None:
            raise RuntimeError("StreamingLlamaRuntime.generate requires materialize_lm_head=True")
        hidden_cpu = hidden.to(device=torch.device("cpu"), dtype=self._lm_head_weight_cpu.dtype)
        return F.linear(hidden_cpu, self._lm_head_weight_cpu)

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
        if str(self._traffic_current_phase or "idle") != "decode":
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

    def _copy_cpu_to_gpu(
        self,
        tensor: torch.Tensor,
        *,
        dtype: torch.dtype,
        layer_idx: Optional[int] = None,
        tag: str = "h2d",
    ) -> torch.Tensor:
        prepared = self.loader.prepare_h2d_source(tensor, dtype=dtype, pin_override=True)
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
            else:
                out.copy_(prepared, non_blocking=bool(prepared.is_pinned()))
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
            else:
                out.copy_(prepared, non_blocking=bool(prepared.is_pinned()))
            return out

    def _copy_cpu_to_existing_gpu(
        self,
        dest: torch.Tensor,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        prepared = self.loader.prepare_h2d_source(tensor, dtype=dest.dtype, pin_override=True)
        if self.device.type != "cuda":
            dest.copy_(prepared)
            return prepared
        if self._h2d_stream is not None:
            self._h2d_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self._h2d_stream):
                dest.copy_(prepared, non_blocking=bool(prepared.is_pinned()))
        else:
            dest.copy_(prepared, non_blocking=bool(prepared.is_pinned()))
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
        if str(self._traffic_current_phase or "idle") != "decode":
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

        self._wait_for_h2d_stream()
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

    def _route_sparse_mlp(self, hidden: torch.Tensor, layer_idx: int) -> Optional[torch.Tensor]:
        """Return active block indices [N, top_k] for this layer, or None (→ dense).

        Uses the learned encoder/decoder from the basis checkpoint to predict
        which MLP output blocks will have the largest norm for this hidden state,
        then returns the top-K block indices.  All tensors moved to GPU lazily.
        """
        routing = self._sparse_routing.get(layer_idx)
        if routing is None:
            return None

        enc_w = routing["enc_w"]
        enc_b = routing["enc_b"]
        dec_norm_t = routing["dec_norm_t"]

        N = hidden.shape[0] * hidden.shape[1]
        h = hidden.view(N, -1).to(device=enc_w.device, dtype=enc_w.dtype)

        latent = F.linear(h, enc_w, enc_b)
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

    def _skip_dense_mlp_for_layer(self, layer_idx: int) -> bool:
        if not self._sparse_explicit_layer_selection:
            return False
        return int(layer_idx) not in self._sparse_explicit_layer_selection

    def _forward_learned_basis_mlp(
        self,
        layer_idx: int,
        hidden: torch.Tensor,
        active_blocks: torch.Tensor,
    ) -> torch.Tensor:
        routing = self._sparse_routing.get(layer_idx)
        if routing is None:
            return torch.zeros_like(hidden)

        enc_w = routing["enc_w"]
        enc_b = routing["enc_b"]
        dec = routing["dec"]
        dec_bias = routing.get("dec_bias")
        scale = float(routing.get("scale", 1.0))

        rows = int(hidden.shape[0] * hidden.shape[1])
        if rows <= 0:
            return torch.zeros_like(hidden)

        flat_hidden = hidden.view(rows, hidden.shape[-1]).to(device=enc_w.device, dtype=enc_w.dtype)
        latent = F.linear(flat_hidden, enc_w, enc_b)
        num_blocks = int(dec.shape[0])
        block_size = int(dec.shape[-1])
        out_blocks = torch.zeros((rows, num_blocks, block_size), device=dec.device, dtype=dec.dtype)
        row_idx = torch.arange(rows, device=dec.device, dtype=torch.long)
        active_blocks = active_blocks.to(device=dec.device, dtype=torch.long)

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
            if torch.is_tensor(dec_bias):
                contrib = contrib + dec_bias.index_select(0, blocks_valid)
            out_blocks[rows_valid, blocks_valid] += contrib

        out_flat = out_blocks.view(rows, num_blocks * block_size)
        if scale != 1.0:
            out_flat = out_flat * scale
        return out_flat.view_as(hidden).to(device=hidden.device, dtype=hidden.dtype)

    def _get_sparse_4bit_param(self, full_name: str) -> Dict[str, Any]:
        cached = self._sparse_param_cache.get(full_name)
        if cached is not None:
            return cached

        raw_weight, quant_aux = self.loader._load_raw_for_param(full_name)
        if not quant_aux:
            raise RuntimeError(f"Sparse 4-bit path expected quantized weights for {full_name}")

        quant_state = bnb_functional.QuantState.from_dict(quant_aux, device=torch.device("cpu"))
        absmax = quant_state.absmax
        if quant_state.nested:
            absmax = bnb_functional.dequantize_blockwise(quant_state.absmax, quant_state.state2)
            absmax = absmax + quant_state.offset

        # Precompute block-major views for CPU gather
        in_features = int(quant_state.shape[1])
        out_features = int(quant_state.shape[0])
        block_size = self._sparse_block_size
        
        bytes_per_row = in_features // 2
        bytes_per_block = bytes_per_row * block_size
        blocks_per_col = out_features // block_size
        packed_blocks = raw_weight.view(blocks_per_col, bytes_per_block).contiguous()
        
        absmax_per_row = in_features // int(quant_state.blocksize)
        absmax_per_block = absmax_per_row * block_size
        absmax_blocks = absmax.view(blocks_per_col, absmax_per_block).contiguous()

        cached = {
            "packed_weight": raw_weight.reshape(-1).contiguous(),
            "packed_blocks": packed_blocks,
            "absmax": absmax.to(dtype=torch.float32).contiguous(),
            "absmax_blocks": absmax_blocks.to(dtype=torch.float32).contiguous(),
            "code": quant_state.code.to(dtype=torch.float32).contiguous(),
            "code_gpu": quant_state.code.to(device=self.device, dtype=torch.float32, non_blocking=True).contiguous()
            if self.device.type == "cuda"
            else None,
            "out_features": out_features,
            "in_features": in_features,
            "quant_block_size": int(quant_state.blocksize),
            "quant_type": str(quant_state.quant_type),
            "dtype": quant_state.dtype,
        }
        self._maybe_cache_sparse_param_hot_blocks(full_name, cached)
        self._sparse_param_cache[full_name] = cached
        return cached

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

    def _sparse_mlp_forward_fast(
        self,
        layer_idx: int,
        mlp: nn.Module,
        hidden: torch.Tensor,
        active_blocks: torch.Tensor,
        _oom_retry_depth: int = 0,
    ) -> torch.Tensor:
        if layer_idx in self._sparse_routing:
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

        cpu_blocks = active_blocks.cpu().view(-1)
        if flat_hidden.shape[0] > 1:
            cpu_blocks = cpu_blocks.unique()
        ordered_blocks = self._order_blocks_for_layer_hot_cache(layer_idx, cpu_blocks)
        max_valid_blocks = min(
            int(gate_param["packed_blocks"].shape[0]),
            int(up_param["packed_blocks"].shape[0]),
            int(down_param["in_features"]) // int(block_size),
        )
        ordered_blocks = ordered_blocks[(ordered_blocks >= 0) & (ordered_blocks < max_valid_blocks)]
        num_active_blocks = int(ordered_blocks.shape[0])
        if num_active_blocks <= 0:
            return torch.zeros_like(hidden)

        K_S = num_active_blocks * block_size
        H_in = int(gate_param["in_features"])
        neuron_offsets = torch.arange(block_size, device="cpu")
        active_neurons = (ordered_blocks.unsqueeze(-1) * block_size + neuron_offsets).reshape(-1)

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
            ordered_blocks=ordered_blocks,
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

            return F.linear(flat_hidden, out_gpu, bias_gpu)

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
        ordered_blocks: torch.Tensor,
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
            rows = int(flat_hidden.shape[0])
            num_active_blocks = int(ordered_blocks.numel())
            block_size = int(self._sparse_block_size)
            active_local = torch.arange(num_active_blocks, device=self.device, dtype=torch.int32).view(1, -1)
            active_local = active_local.expand(rows, num_active_blocks).contiguous()
            active_dim = num_active_blocks * block_size
            flat_mask = torch.ones((rows, active_dim), device=self.device, dtype=flat_hidden.dtype)

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
        if self._mlp_proj_staging is None:
            # GPU-only mode: if dense staging is unavailable, skip dense MLP
            # compute for this layer instead of falling back to CPU.
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

    # ── Sparse Attention Head helpers ─────────────────────────────────────────

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
            return static_indices[:k0]

        taylor_cache = self._taylor_caches[layer_idx]
        if taylor_cache is None:
            k0 = max(1, min(int(self._attn_active_heads), int(static_indices.numel())))
            return static_indices[:k0]

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
        q_rows_cpu = self._ensure_cpu_scratch(
            "attn_q_rows",
            numel=K * bytes_per_head_q,
            dtype=torch.uint8,
        ).view(K, bytes_per_head_q)
        for _slot, _head_idx in enumerate(active_list):
            q_rows_cpu[_slot].copy_(packed_2d_q[_head_idx], non_blocking=False)

        absmax_per_head_q = head_dim * meta_q["in_features"] // meta_q["quant_block_size"]
        absmax_2d_q = meta_q["absmax_flat"].view(meta_q["num_heads_total"], absmax_per_head_q)
        q_abs_cpu = self._ensure_cpu_scratch(
            "attn_q_abs",
            numel=K * absmax_per_head_q,
            dtype=torch.float32,
        ).view(K, absmax_per_head_q)
        for _slot, _head_idx in enumerate(active_list):
            q_abs_cpu[_slot].copy_(absmax_2d_q[_head_idx], non_blocking=False)

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
        self._wait_for_h2d_stream()
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
        o_cols_cpu = self._ensure_cpu_scratch(
            "attn_o_cols",
            numel=H_out * K * bytes_per_head_col,
            dtype=torch.uint8,
        ).view(H_out, K * bytes_per_head_col)
        for _slot, _head_idx in enumerate(active_list):
            _dst_start = _slot * bytes_per_head_col
            _src_start = _head_idx * bytes_per_head_col
            o_cols_cpu[:, _dst_start : _dst_start + bytes_per_head_col].copy_(
                raw_2d_o[:, _src_start : _src_start + bytes_per_head_col],
                non_blocking=False,
            )

        # Absmax: head_dim=128 spans 2 quant groups (qbs=64) per row.
        absmax_per_head_col = head_dim // qbs                  # 2
        absmax_per_row_o    = H_in // qbs                      # 256
        absmax_2d_o = meta_o["absmax_flat"].view(H_out, absmax_per_row_o)
        o_abs_cpu = self._ensure_cpu_scratch(
            "attn_o_abs",
            numel=H_out * K * absmax_per_head_col,
            dtype=torch.float32,
        ).view(H_out, K * absmax_per_head_col)
        for _slot, _head_idx in enumerate(active_list):
            _dst_start = _slot * absmax_per_head_col
            _src_start = _head_idx * absmax_per_head_col
            o_abs_cpu[:, _dst_start : _dst_start + absmax_per_head_col].copy_(
                absmax_2d_o[:, _src_start : _src_start + absmax_per_head_col],
                non_blocking=False,
            )

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

    def _load_layer(self, layer_idx: int) -> LlamaDecoderLayer:
        self._layer_skeleton.layer_idx = int(layer_idx)
        if hasattr(self._layer_skeleton, "self_attn"):
            self._layer_skeleton.self_attn.layer_idx = int(layer_idx)
        self._record_layer_visit(layer_idx)

        layer_state_items = self._layer_state_items

        # Determine if sparse attention is active for this layer.
        # _get_attn_active_heads returns None when no importance data is loaded
        # or for layers not profiled — both fall back to full dense loading.
        _active_attn_heads = self._get_attn_active_heads(layer_idx)
        _head_dim = int(getattr(self.config, "head_dim", 128))
        _hidden   = int(getattr(self.config, "hidden_size", 16384))
        # Keys to skip in the main loop (handled by _load_sparse_attn_heads below).
        _skip_attn: set = (
            {"self_attn.q_proj.weight", "self_attn.o_proj.weight"}
            if _active_attn_heads is not None
            else set()
        )

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

        if _active_attn_heads is None:
            self._attn_loaded_q_rows = None
            self._attn_loaded_o_cols = None
            self._attn_qo_state = "dense"

        # Sparse attention head loading: only transfer NF4 bytes for K active heads.
        # q_proj and o_proj are zeroed first; only the active head rows/columns are filled.
        if _active_attn_heads is not None:
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
        return F.embedding(token_ids.to(device=torch.device("cpu")), self._embed_weight_cpu)

    def _lm_head_forward_cpu(self, hidden: torch.Tensor) -> torch.Tensor:
        if self._lm_head_weight_cpu is None:
            raise RuntimeError("StreamingLlamaRuntime.generate requires materialize_lm_head=True")
        hidden_cpu = hidden.to(device=torch.device("cpu"), dtype=self._lm_head_weight_cpu.dtype)
        return F.linear(hidden_cpu, self._lm_head_weight_cpu)

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
            _active_blocks = self._route_sparse_mlp(mlp_input, layer_idx)
            if _active_blocks is not None:
                if self._debug_steps:
                    print(f"[debug] layer {layer_idx}: sparse_mlp", flush=True)
                mlp_out = self._sparse_mlp_forward_fast(layer_idx, layer.mlp, mlp_input, _active_blocks)
            else:
                if self._skip_dense_mlp_for_layer(layer_idx):
                    if self._debug_steps:
                        print(f"[debug] layer {layer_idx}: sparse_skip_mlp", flush=True)
                    mlp_out = torch.zeros_like(mlp_input)
                elif self._debug_steps:
                    print(f"[debug] layer {layer_idx}: dense_mlp", flush=True)
                    mlp_out = self._dense_mlp_forward_streaming_fast(layer_idx, layer.mlp, mlp_input)
                else:
                    mlp_out = self._dense_mlp_forward_streaming_fast(layer_idx, layer.mlp, mlp_input)

            if layer_idx in capture_set:
                captures[layer_idx] = {
                    "mlp_input": mlp_input.detach().cpu(),
                    "dense_mlp_out": mlp_out.detach().cpu(),
                }
            hidden = residual + mlp_out
            self._release_modules(layer, *( [taylor_attn] if taylor_attn is not None else [] ))

        if printed_progress:
            print("", flush=True)
        hidden = self.norm(hidden)
        if self._lm_head_weight_cpu is None:
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
        autoregressive quality.  MLP residual is zero (already handled by
        zero-passthrough fallback when _mlp_proj_staging is None).

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

        Attention is still processed one token at a time (Taylor-SSD is a
        recurrent operator; running seq_len>1 through it allocates ~268 MB
        local-attention tensors per layer and overflows 8 GB VRAM).  MLP is
        batched over all tokens at once — the sparse path transfers only the
        active-block NF4 bytes regardless of seq_len.
        """
        seq_len = int(token_ids.shape[1])
        all_hidden = self._embed_tokens_cpu(token_ids).to(device=self.device, dtype=self.dtype)
        # [1, seq_len, hidden]
        next_hidden = torch.empty_like(all_hidden)
        position_start = int(position_offset)
        position_ids_all = torch.arange(
            position_start,
            position_start + seq_len,
            device=self.device,
            dtype=torch.long,
        )

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

            for pos in range(seq_len):
                h = all_hidden[:, pos : pos + 1, :]           # [1, 1, hidden]
                pos_ids = position_ids_all[pos : pos + 1].unsqueeze(0)  # [1, 1]
                rope_tok = self.rotary_emb(h, pos_ids)
                h_norm = layer.input_layernorm(h)

                if use_taylor:
                    attn_out, _, present = taylor_attn(
                        hidden_states=h_norm,
                        position_ids=pos_ids,
                        position_embeddings=rope_tok,
                        past_key_value=self._taylor_caches[layer_idx],
                        use_cache=True,
                    )
                    self._taylor_caches[layer_idx] = present
                else:
                    attn_out, _ = layer.self_attn(
                        hidden_states=h_norm,
                        position_embeddings=rope_tok,
                        attention_mask=None,
                        past_key_values=self._dense_cache,
                        cache_position=pos_ids.view(-1),
                    )

                next_hidden[:, pos : pos + 1, :] = h + attn_out

            all_hidden, next_hidden = next_hidden, all_hidden  # swap — no allocation
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
            _active_blocks = self._route_sparse_mlp(mlp_input, layer_idx)
            if _active_blocks is not None:
                mlp_out = self._sparse_mlp_forward_fast(layer_idx, layer.mlp, mlp_input, _active_blocks)
            else:
                if self._skip_dense_mlp_for_layer(layer_idx):
                    mlp_out = torch.zeros_like(mlp_input)
                else:
                    mlp_out = self._dense_mlp_forward_streaming_fast(layer_idx, layer.mlp, mlp_input)
            all_hidden = residual + mlp_out

            self._release_modules(layer)

        if self._show_progress:
            print("", flush=True)
        all_hidden = self.norm(all_hidden)
        # Only the last token's logits are needed to start generation.
        logits = self._lm_head_forward(all_hidden[:, -1:, :]).float()
        return logits

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
        if self._lm_head_weight_cpu is None:
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
                if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
                    break
                logits, _ = self.forward_token(next_token, position_index=int(generated.shape[1]) - 1)
                processed_ids_cpu = torch.cat(
                    [
                        processed_ids_cpu,
                        next_token.detach().to(device=torch.device("cpu"), dtype=torch.long).contiguous(),
                    ],
                    dim=-1,
                )
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
                pending_layers = [idx for idx in selected_layers if int(layer_rows[int(idx)]) < int(max_rows)]
                if not pending_layers:
                    break
                if int(layer_idx) > int(max(pending_layers)):
                    break
                rows_done = sum(1 for idx in selected_layers if int(layer_rows[int(idx)]) >= int(max_rows))
                printed_progress = True
                status = (
                    f"[collect] layer {layer_idx+1}/{self.num_layers} "
                    f"({rows_done}/{len(selected_layers)} layers saturated, {len(pending_layers)} still collecting)"
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

                next_hidden.copy_(attn_hidden + mlp_out)
                all_hidden, next_hidden = next_hidden, all_hidden
                self._release_modules(layer)
            if printed_progress:
                print("", flush=True)
