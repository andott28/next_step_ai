"""ShardedSafetensorLoader: RAM-cached, NF4-aware safetensors loader."""
from __future__ import annotations

import contextlib
import json
import os
import struct as _struct
import threading
from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import bitsandbytes.functional as bnb_functional
import torch
from bitsandbytes.backends.cuda.ops import _dequantize_4bit_impl as _bnb_dequant_impl
from safetensors import safe_open

from ._helpers import (
    _SAFETENSORS_DTYPE_TO_TORCH,
    _assert_bnb_ready,
    _readinto_cpu_tensor,
    _resolve_pin_ram_cache_default,
    _resolve_ram_cache_limit_bytes,
    _torch_dtype_itemsize,
)


def _load_safetensors_direct(
    shard_path: Path, keys: Sequence[str]
) -> dict[str, torch.Tensor]:
    """Load specific tensors from a safetensors file via direct I/O (no mmap).

    On Windows, repeated safe_open (mmap) calls on a 5 GB shard produce an
    access-violation in torch/storage.py after a few opens — even when each
    context manager is properly closed — because Windows does not reliably
    release the VA region between opens of a large file.  Reading each tensor
    directly with a seek + read sidesteps the issue entirely while producing
    a regular (non-mmap) CPU tensor indistinguishable from the mmap path.
    """
    keys_set = set(keys)
    out: dict[str, torch.Tensor] = {}
    with open(str(shard_path), "rb") as f:
        header_size = _struct.unpack("<Q", f.read(8))[0]
        header: dict[str, Any] = json.loads(f.read(header_size).decode("utf-8"))
        data_base = 8 + header_size

        for key, meta in header.items():
            if key == "__metadata__" or key not in keys_set:
                continue
            if not isinstance(meta, dict):
                continue
            dtype_str = str(meta.get("dtype", ""))
            torch_dtype = _SAFETENSORS_DTYPE_TO_TORCH.get(dtype_str)
            if torch_dtype is None:
                continue
            shape: list[int] = list(meta.get("shape", []))
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


class ShardedSafetensorLoader:
    def __init__(
        self,
        snapshot_dir: Path,
        *,
        cache_shard_handles: bool | None = None,
        pin_ram_cache: bool | None = None,
    ) -> None:
        self.snapshot_dir = Path(snapshot_dir)
        index_path = self.snapshot_dir / "model.safetensors.index.json"
        self.weight_map: dict[str, str] = {}
        if index_path.exists():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            raw_map = payload.get("weight_map", {})
            self.weight_map = {str(k): str(v) for k, v in raw_map.items()}
        else:
            safetensor_files = sorted(self.snapshot_dir.glob("*.safetensors"))
            if not safetensor_files:
                raise RuntimeError(f"No safetensors shards found in {self.snapshot_dir}")
            with safe_open(str(safetensor_files[0]), framework="pt", device="cpu") as handle:
                for name in handle:
                    self.weight_map[str(name)] = safetensor_files[0].name

        self._available_names = set(self.weight_map.keys())
        self._quant_aux_by_base: dict[str, list[str]] = defaultdict(list)
        for name in self.weight_map:
            if ".weight." not in name:
                continue
            base = name.split(".weight.", 1)[0] + ".weight"
            self._quant_aux_by_base[base].append(name)



        if cache_shard_handles is None:
            cache_shard_handles = os.name != "nt"
        self._cache_shard_handles = bool(cache_shard_handles)
        self._shard_handles: dict[str, Any] = {}




        self._ram_cache: dict[str, tuple[torch.Tensor, dict[str, torch.Tensor]]] = {}
        self._ram_cache_lru: OrderedDict[str, None] = OrderedDict()
        self._ram_cache_entry_bytes: dict[str, int] = {}
        self._ram_cache_current_bytes: int = 0
        self._ram_cache_limit_bytes: int | None = _resolve_ram_cache_limit_bytes()
        self._ram_cache_enabled: bool = True
        self._pin_ram_cache: bool = _resolve_pin_ram_cache_default() if pin_ram_cache is None else bool(pin_ram_cache)
        self._ram_cache_lock = threading.Lock()
        self._tensor_load_lock = threading.Lock()
        self._quant_meta_cache: dict[str, dict[str, Any]] = {}
        self._quant_meta_lock = threading.Lock()
        self._h2d_copy_scratch: dict[str, torch.Tensor] = {}
        self._h2d_copy_events: dict[str, Any] = {}
        self._direct_header_cache: dict[str, tuple[int, dict[str, Any]]] = {}
        self._direct_header_lock = threading.Lock()

    @staticmethod
    def _entry_nbytes(weight: torch.Tensor, quant_aux: dict[str, torch.Tensor]) -> int:
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
        dtype: torch.dtype | None = None,
        pin_override: bool | None = None,
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
        dtype: torch.dtype | None = None,
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
            with contextlib.suppress(Exception):
                previous.synchronize()

        numel = int(tensor.numel())
        scratch = self._h2d_copy_scratch.get(scratch_key)
        if scratch is None or scratch.dtype != tensor.dtype or int(scratch.numel()) < numel:
            scratch = torch.empty(numel, dtype=tensor.dtype, pin_memory=True)
            self._h2d_copy_scratch[scratch_key] = scratch

        view = scratch[:numel].view(tuple(tensor.shape))
        view.copy_(tensor, non_blocking=False)
        return view

    def _record_h2d_scratch_use(self, scratch_key: str | None, *, device: torch.device) -> None:
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

    def _get_direct_shard_header(self, shard_name: str) -> tuple[int, dict[str, Any]]:
        cached = self._direct_header_cache.get(shard_name)
        if cached is not None:
            return cached
        shard_path = self.snapshot_dir / str(shard_name)
        with open(str(shard_path), "rb") as f:
            header_size = _struct.unpack("<Q", f.read(8))[0]
            header: dict[str, Any] = json.loads(f.read(header_size).decode("utf-8"))
        payload = (8 + int(header_size), header)
        with self._direct_header_lock:
            existing = self._direct_header_cache.get(shard_name)
            if existing is not None:
                return existing
            self._direct_header_cache[shard_name] = payload
        return payload

    def _get_tensor_direct_meta(self, full_name: str) -> tuple[Path, int, dict[str, Any]]:
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

    def load_nf4_packed_blocks(
        self,
        name: str,
        block_indices: torch.Tensor,
        *,
        bytes_per_block: int,
    ) -> torch.Tensor:
        """Load specific NF4 packed blocks directly from disk without loading the full weight.

        NF4 weights are stored as flat uint8 in safetensors (e.g. shape [N, 1]).
        Each logical block of `block_size` output rows occupies `bytes_per_block` contiguous
        bytes.  This method reads only those byte ranges, reducing SSD I/O from ~436 MB to
        ~13 MB per projection for 51 active blocks out of 1664.

        Falls back to full-weight load when the weight is already in the RAM cache (free).
        Returns (n_blocks, bytes_per_block) uint8 tensor on CPU.
        """
        full_name = str(name)
        bpb = int(bytes_per_block)
        idx = block_indices.to(dtype=torch.long).reshape(-1)

        # RAM cache hit: extract blocks from the already-loaded weight (free, no I/O)
        if self._ram_cache_enabled:
            with self._ram_cache_lock:
                cached = self._ram_cache.get(full_name)
            if cached is not None:
                weight, _ = cached
                raw_flat = weight.reshape(-1)
                packed_blocks = raw_flat.view(-1, bpb)
                return packed_blocks[idx].contiguous()

        if os.name != "nt":
            # Non-Windows: load full tensor and extract (no direct seek path needed)
            weight, _ = self._load_raw_for_param(full_name, store_in_ram_cache=False)
            raw_flat = weight.reshape(-1)
            return raw_flat.view(-1, bpb)[idx].contiguous()

        # Windows: direct seek-and-read for only the needed blocks
        shard_path, data_base, meta = self._get_tensor_direct_meta(full_name)
        start_offset, _ = meta["data_offsets"]

        idx_sorted, inverse = idx.sort()
        n_blocks = int(idx_sorted.numel())
        if n_blocks == 0:
            return torch.empty((0, bpb), dtype=torch.uint8)

        out_sorted = torch.empty((n_blocks, bpb), dtype=torch.uint8)
        file_base = int(data_base) + int(start_offset)
        with open(str(shard_path), "rb") as f:
            cursor = 0
            while cursor < n_blocks:
                run_start = cursor
                first_block = int(idx_sorted[cursor].item())
                cursor += 1
                while cursor < n_blocks and int(idx_sorted[cursor].item()) == first_block + (cursor - run_start):
                    cursor += 1
                f.seek(file_base + first_block * bpb)
                _readinto_cpu_tensor(f, out_sorted[run_start:cursor])

        return out_sorted.index_select(0, inverse)

    def _load_exact_tensors(self, names: Sequence[str]) -> dict[str, torch.Tensor]:
        requested = [str(name) for name in names if str(name) in self._available_names]
        by_shard: dict[str, list[str]] = defaultdict(list)
        for name in requested:
            by_shard[self.weight_map[name]].append(name)

        out: dict[str, torch.Tensor] = {}
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
        quant_aux: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
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

    def _load_raw_for_param(
        self,
        full_name: str,
        *,
        store_in_ram_cache: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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

        if self._ram_cache_enabled and bool(store_in_ram_cache):
            if os.name == "nt" and self._cache_shard_handles:
                weight = weight.clone().contiguous()
                quant_aux = {k: v.clone().contiguous() for k, v in quant_aux.items()}
            weight = self._maybe_pin_cpu_tensor(weight.contiguous())
            quant_aux = {k: self._maybe_pin_cpu_tensor(v.contiguous()) for k, v in quant_aux.items()}
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
        staging: torch.Tensor | None = None,
        absmax_staging: torch.Tensor | None = None,
        nested_absmax_staging: torch.Tensor | None = None,
        state2_absmax_staging: torch.Tensor | None = None,
        code_staging: torch.Tensor | None = None,
        byte_counter: Any | None = None,
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
            dtype: torch.dtype | None = None,
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
                    f"Shape mismatch for '{full_name}': {n} NF4 bytes -> {n * 2} elements "
                    f"but out has {out.numel()} elements (shape {out.shape})"
                )

            _assert_bnb_ready(weight_gpu, f"{full_name}/weight_gpu", expected_device=out.device)
            _assert_bnb_ready(absmax, f"{full_name}/absmax", expected_device=out.device)
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

        quant_state = bnb_functional.QuantState.from_dict(quant_aux, device=out.device)
        traffic_bytes += int(quant_state.absmax.numel() * quant_state.absmax.element_size())
        if quant_state.nested:
            traffic_bytes += int(quant_state.state2.absmax.numel() * quant_state.state2.absmax.element_size())
            traffic_bytes += int(quant_state.state2.code.numel() * quant_state.state2.code.element_size())

        if n * 2 != out.numel():
            raise RuntimeError(
                f"Shape mismatch for '{full_name}': {n} NF4 bytes -> {n * 2} elements "
                f"but out has {out.numel()} elements (shape {out.shape})"
            )







        if quant_state.nested:
            absmax = bnb_functional.dequantize_blockwise(quant_state.absmax, quant_state.state2)
            absmax = absmax.float() + quant_state.offset
        else:
            absmax = quant_state.absmax.float()
        _assert_bnb_ready(weight_gpu, f"{full_name}/weight_gpu", expected_device=out.device)
        _assert_bnb_ready(absmax, f"{full_name}/absmax_gpu", expected_device=out.device)
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
    ) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for key in expected_keys:
            full_name = f"{prefix}{key}"
            tensor = self.load_parameter(full_name)
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                tensor = tensor.to(dtype=dtype)
            state[str(key)] = tensor
        return state
