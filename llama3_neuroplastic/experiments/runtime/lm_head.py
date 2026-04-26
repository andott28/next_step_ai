from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F

try:
    import bitsandbytes.functional as bnb_functional
except ImportError:
    bnb_functional = None

try:
    from ...triton_sparse_mlp import triton_sparse_input_linear_4bit
except Exception:
    try:
        from triton_sparse_mlp import triton_sparse_input_linear_4bit
    except Exception:
        triton_sparse_input_linear_4bit = None

from ._helpers import _is_cuda_oom_error


class RuntimeLmHeadMixin:
    @staticmethod
    def _resolve_lm_head_block_size(in_features: int) -> int:
        raw_override = os.getenv("STREAMING_GPU_LM_HEAD_BLOCK_SIZE", "").strip()
        candidates: list[int] = []
        if raw_override:
            try:
                candidates.append(max(1, int(raw_override)))
            except ValueError:
                pass
        candidates.extend([256, 128, 64, 32, 16, 8, 4, 2, 1])
        for candidate in candidates:
            if int(in_features) % int(candidate) == 0:
                return int(candidate)
        return 1

    @staticmethod
    def _lm_head_quantized_weight_gb(meta: dict[str, Any]) -> float:
        total_bytes = 0
        for key in ("packed_weight", "absmax", "code"):
            tensor = meta.get(key)
            if torch.is_tensor(tensor):
                total_bytes += int(tensor.numel()) * int(tensor.element_size())
        return float(total_bytes) / float(1024 ** 3)

    def get_lm_head_status(self) -> dict[str, Any]:
        if not bool(getattr(self, "_materialize_lm_head", True)):
            return {
                "enabled": False,
                "mode": "disabled",
                "on_gpu": False,
                "weight_name": str(getattr(self, "_lm_head_weight_name", "")),
                "dtype": None,
                "gpu_attempted": bool(getattr(self, "_lm_head_gpu_attempted", False)),
                "gpu_preferred": bool(getattr(self, "_prefer_gpu_lm_head", False)),
                "last_failure": str(getattr(self, "_lm_head_gpu_last_failure", "") or ""),
                "weight_gb": 0.0,
            }
        nf4_meta = getattr(self, "_lm_head_nf4_meta_gpu", None)
        gpu_weight = getattr(self, "_lm_head_weight_gpu", None)
        cpu_weight = getattr(self, "_lm_head_weight_cpu", None)
        weight_name = str(getattr(self, "_lm_head_weight_name", ""))
        last_failure = str(getattr(self, "_lm_head_gpu_last_failure", "") or "")
        if isinstance(nf4_meta, dict):
            mode = "gpu_nf4"
            dtype_name = str(getattr(self, "dtype", None))
            weight_gb = self._lm_head_quantized_weight_gb(nf4_meta)
        elif gpu_weight is not None:
            mode = "gpu_dense"
            dtype_name = str(gpu_weight.dtype)
            weight_gb = float(gpu_weight.numel() * gpu_weight.element_size()) / float(1024 ** 3)
        elif cpu_weight is not None:
            mode = "cpu_dense"
            dtype_name = str(cpu_weight.dtype)
            weight_gb = float(cpu_weight.numel() * cpu_weight.element_size()) / float(1024 ** 3)
        elif bool(getattr(self, "_prefer_gpu_lm_head", False)) and getattr(getattr(self, "device", None), "type", None) == "cuda":
            mode = "gpu_pending"
            dtype_name = None
            weight_gb = 0.0
        else:
            mode = "cpu_pending"
            dtype_name = None
            weight_gb = 0.0
        return {
            "enabled": True,
            "mode": str(mode),
            "on_gpu": bool(isinstance(nf4_meta, dict) or gpu_weight is not None),
            "weight_name": weight_name,
            "dtype": dtype_name,
            "gpu_attempted": bool(getattr(self, "_lm_head_gpu_attempted", False)),
            "gpu_preferred": bool(getattr(self, "_prefer_gpu_lm_head", False)),
            "last_failure": last_failure,
            "weight_gb": float(weight_gb),
        }

    def _ensure_lm_head_weight_cpu(self) -> torch.Tensor | None:
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
        raw_override_gb = os.getenv("STREAMING_GPU_LM_HEAD_RESERVE_GB", "").strip()
        if raw_override_gb:
            try:
                override_gb = float(raw_override_gb)
            except ValueError:
                override_gb = -1.0
            if override_gb >= 0.0:
                return int(override_gb * (1024 ** 3))
        if bool(getattr(self, "_explicit_gpu_lm_head", False)):
            return int(256 * (1024 ** 2))
        safety_margin_bytes = int(256 * (1024 ** 2))
        sparse_reserve_bytes = self._estimate_sparse_gpu_working_set_bytes()
        # The hot cache is self-limiting via _can_reserve_vram_hot_cache (which
        # uses the 1-GB margin + driver free-bytes).  Do NOT add hot_cache_limit
        # to the residual here — that would double-count it and prevent the NF4
        # LM head from loading on 8 GB GPUs (2 GB NF4 + 5.25 GB cache > 8 GB).
        return int(max(safety_margin_bytes, self._vram_hot_cache_margin_bytes)) + sparse_reserve_bytes

    @staticmethod
    def _resolve_lm_head_nf4_quant_rows(out_features: int) -> int:
        raw_override = os.getenv("STREAMING_GPU_LM_HEAD_QUANT_ROWS", "").strip()
        if raw_override:
            try:
                value = int(raw_override)
            except ValueError:
                value = 0
            if value > 0:
                return int(min(max(1, value), int(out_features)))
        return int(min(max(1, 1024), int(out_features)))

    @staticmethod
    def _resolve_lm_head_nf4_quant_block_size(in_features: int) -> int:
        raw_override = os.getenv("STREAMING_GPU_LM_HEAD_NF4_BLOCK_SIZE", "").strip()
        if raw_override:
            try:
                value = int(raw_override)
            except ValueError:
                value = 0
            if value > 0 and int(in_features) % int(value) == 0:
                return int(value)
        for candidate in (64, 128, 256, 32):
            if int(in_features) % int(candidate) == 0:
                return int(candidate)
        raise RuntimeError(
            f"Unable to select an NF4 quant block size for in_features={int(in_features)}; "
            "set STREAMING_GPU_LM_HEAD_NF4_BLOCK_SIZE to a positive divisor."
        )

    def _quantize_dense_lm_head_nf4_cpu(
        self,
        dense_weight_cpu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        if bnb_functional is None or not hasattr(bnb_functional, "quantize_4bit"):
            raise RuntimeError("bitsandbytes.quantize_4bit is unavailable")
        if dense_weight_cpu.ndim != 2:
            raise RuntimeError(
                f"Expected a rank-2 dense LM head weight, got shape {tuple(dense_weight_cpu.shape)}"
            )
        if dense_weight_cpu.device.type != "cpu":
            dense_weight_cpu = dense_weight_cpu.to(device=torch.device("cpu"))
        dense_weight_cpu = dense_weight_cpu.contiguous()

        out_features = int(dense_weight_cpu.shape[0])
        in_features = int(dense_weight_cpu.shape[1])
        quant_block_size = self._resolve_lm_head_nf4_quant_block_size(in_features)
        total_elements = int(out_features) * int(in_features)
        if total_elements % 2 != 0:
            raise RuntimeError(
                f"Dense LM head element count must be even for packed NF4 bytes, got {total_elements}"
            )
        if total_elements % int(quant_block_size) != 0:
            raise RuntimeError(
                f"Dense LM head element count {total_elements} must be divisible by quant block size {int(quant_block_size)}"
            )

        packed_cpu = torch.empty(total_elements // 2, dtype=torch.uint8, device=torch.device("cpu"))
        absmax_cpu = torch.empty(total_elements // int(quant_block_size), dtype=torch.float32, device=torch.device("cpu"))
        code_cpu: torch.Tensor | None = None

        rows_per_chunk = self._resolve_lm_head_nf4_quant_rows(out_features)
        packed_offset = 0
        absmax_offset = 0
        for row_start in range(0, out_features, rows_per_chunk):
            row_stop = min(row_start + rows_per_chunk, out_features)
            chunk = dense_weight_cpu[row_start:row_stop]
            if chunk.dtype not in {torch.float16, torch.float32, torch.bfloat16}:
                chunk = chunk.to(dtype=torch.float16)
            elif chunk.dtype == torch.bfloat16:
                chunk = chunk.to(dtype=torch.float16)
            if not chunk.is_contiguous():
                chunk = chunk.contiguous()

            chunk_packed, chunk_state = bnb_functional.quantize_4bit(
                chunk,
                blocksize=int(quant_block_size),
                compress_statistics=False,
                quant_type="nf4",
                quant_storage=torch.uint8,
            )
            chunk_packed_flat = chunk_packed.reshape(-1).to(device=torch.device("cpu"), dtype=torch.uint8).contiguous()
            packed_next = packed_offset + int(chunk_packed_flat.numel())
            packed_cpu[packed_offset:packed_next].copy_(chunk_packed_flat, non_blocking=False)
            packed_offset = packed_next

            chunk_absmax = chunk_state.absmax
            if bool(getattr(chunk_state, "nested", False)):
                chunk_absmax = bnb_functional.dequantize_blockwise(chunk_state.absmax, chunk_state.state2)
                chunk_absmax = chunk_absmax + chunk_state.offset
            chunk_absmax_flat = chunk_absmax.reshape(-1).to(device=torch.device("cpu"), dtype=torch.float32).contiguous()
            absmax_next = absmax_offset + int(chunk_absmax_flat.numel())
            absmax_cpu[absmax_offset:absmax_next].copy_(chunk_absmax_flat, non_blocking=False)
            absmax_offset = absmax_next

            chunk_code = chunk_state.code.reshape(-1).to(device=torch.device("cpu"), dtype=torch.float32).contiguous()
            if code_cpu is None:
                code_cpu = chunk_code
            elif int(code_cpu.numel()) != int(chunk_code.numel()) or not torch.equal(code_cpu, chunk_code):
                raise RuntimeError("Inconsistent NF4 codebook across dense LM-head quantization chunks")

        if packed_offset != int(packed_cpu.numel()):
            raise RuntimeError(
                f"Packed NF4 write underflow: wrote {packed_offset}, expected {int(packed_cpu.numel())}"
            )
        if absmax_offset != int(absmax_cpu.numel()):
            raise RuntimeError(
                f"Absmax write underflow: wrote {absmax_offset}, expected {int(absmax_cpu.numel())}"
            )
        if code_cpu is None:
            raise RuntimeError("Dense LM-head quantization produced no NF4 codebook")
        return (
            packed_cpu,
            absmax_cpu,
            code_cpu,
            int(out_features),
            int(in_features),
            int(quant_block_size),
        )

    def _materialize_lm_head_nf4_on_gpu(self) -> bool:
        if getattr(self, "_lm_head_nf4_meta_gpu", None) is not None:
            return True
        if self.device.type != "cuda":
            self._lm_head_gpu_last_failure = "non_cuda_device"
            return False
        if triton_sparse_input_linear_4bit is None or bnb_functional is None:
            self._lm_head_gpu_last_failure = "triton_or_bitsandbytes_unavailable"
            return False
        try:
            raw_weight, quant_aux = self.loader._load_raw_for_param(
                self._lm_head_weight_name,
                store_in_ram_cache=True,
            )
        except Exception as exc:
            self._lm_head_gpu_last_failure = f"{type(exc).__name__}: {str(exc)[:200]}"
            return False

        packed_cpu: torch.Tensor
        absmax_cpu: torch.Tensor
        code_cpu: torch.Tensor
        out_features: int
        in_features: int
        quant_block_size: int
        quantized_from_dense = False
        if quant_aux:
            quant_state = bnb_functional.QuantState.from_dict(quant_aux, device=torch.device("cpu"))
            absmax = quant_state.absmax
            if quant_state.nested:
                absmax = bnb_functional.dequantize_blockwise(quant_state.absmax, quant_state.state2)
                absmax = absmax + quant_state.offset
            out_features = int(quant_state.shape[0])
            in_features = int(quant_state.shape[1])
            quant_block_size = int(quant_state.blocksize)
            packed_cpu = raw_weight.reshape(-1).to(dtype=torch.uint8).contiguous()
            absmax_cpu = absmax.to(dtype=torch.float32).contiguous()
            code_cpu = quant_state.code.to(dtype=torch.float32).contiguous()
        else:
            try:
                packed_cpu, absmax_cpu, code_cpu, out_features, in_features, quant_block_size = (
                    self._quantize_dense_lm_head_nf4_cpu(
                        raw_weight.to(device=torch.device("cpu"))
                    )
                )
                quantized_from_dense = True
            except Exception as exc:
                self._lm_head_gpu_last_failure = f"dense_lm_head_nf4_quantization_failed:{type(exc).__name__}:{str(exc)[:160]}"
                return False

        block_size = self._resolve_lm_head_block_size(in_features)
        num_blocks = max(1, int(in_features) // int(block_size))
        required_bytes = (
            int(packed_cpu.numel()) * int(packed_cpu.element_size())
            + int(absmax_cpu.numel()) * int(absmax_cpu.element_size())
            + int(code_cpu.numel()) * int(code_cpu.element_size())
        )
        required_residual_bytes = self._gpu_lm_head_reserve_bytes()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        except Exception:
            free_bytes, total_bytes = 0, 0
        if free_bytes > 0 and (int(free_bytes) - int(required_bytes)) < int(required_residual_bytes):
            self._lm_head_gpu_last_failure = "insufficient_free_vram_for_nf4"
            print(
                f"[lm_head] staying off dense GPU load; free={float(free_bytes) / (1024 ** 3):.2f} GiB "
                f"nf4={float(required_bytes) / (1024 ** 3):.2f} GiB "
                f"reserve_after_load={float(required_residual_bytes) / (1024 ** 3):.2f} GiB "
                f"total={float(total_bytes) / (1024 ** 3):.2f} GiB",
                flush=True,
            )
            return False

        try:
            packed_gpu = packed_cpu.to(device=self.device, dtype=torch.uint8, non_blocking=False).contiguous()
            absmax_gpu = absmax_cpu.to(device=self.device, dtype=torch.float32, non_blocking=False).contiguous()
            code_gpu = code_cpu.to(device=self.device, dtype=torch.float32, non_blocking=False).contiguous()
            active_idx = torch.arange(num_blocks, device=self.device, dtype=torch.int32).view(1, num_blocks).contiguous()
            self._lm_head_nf4_meta_gpu = {
                "packed_weight": packed_gpu,
                "absmax": absmax_gpu,
                "code": code_gpu,
                "out_features": int(out_features),
                "in_features": int(in_features),
                "quant_block_size": int(quant_block_size),
                "block_size": int(block_size),
                "active_idx": active_idx,
            }
            self._lm_head_gpu_last_failure = None
            if quantized_from_dense:
                print(
                    "[lm_head] source weight had no NF4 metadata; quantized dense LM head to NF4 on CPU before GPU upload",
                    flush=True,
                )
            print(
                f"[lm_head] resident on GPU as NF4: {self._lm_head_quantized_weight_gb(self._lm_head_nf4_meta_gpu):.2f} GiB",
                flush=True,
            )
            return True
        except Exception as exc:
            self._lm_head_nf4_meta_gpu = None
            self._lm_head_gpu_last_failure = f"{type(exc).__name__}: {str(exc)[:200]}"
            if _is_cuda_oom_error(exc):
                if bool(getattr(self, "_explicit_gpu_lm_head", False)):
                    raise RuntimeError(
                        "STREAMING_GPU_LM_HEAD was explicitly enabled, but GPU NF4 LM-head materialization OOMed."
                    ) from exc
                print(
                    f"[lm_head] GPU NF4 materialization failed; continuing without quantized hot path: "
                    f"{type(exc).__name__}: {str(exc)[:200]}",
                    flush=True,
                )
                return False
            raise

    def _materialize_lm_head_on_gpu(self) -> None:
        if self._lm_head_gpu_attempted:
            return
        if not self._materialize_lm_head:
            return
        if not self._prefer_gpu_lm_head:
            self._lm_head_gpu_last_failure = "gpu_lm_head_not_preferred"
            self._lm_head_gpu_attempted = True
            return
        if self.device.type != "cuda":
            self._lm_head_gpu_last_failure = "non_cuda_device"
            self._lm_head_gpu_attempted = True
            return
        if self.device.type == "cuda":
            cap = torch.cuda.get_device_capability(self.device)
            if cap < (7, 0):
                self._lm_head_gpu_last_failure = f"unsupported_cuda_capability_{int(cap[0])}_{int(cap[1])}"
                self._lm_head_gpu_attempted = True
                return
        if bool(getattr(self, "_prefer_gpu_quant_lm_head", True)) and self._materialize_lm_head_nf4_on_gpu():
            self._lm_head_gpu_attempted = True
            self._lm_head_gpu_last_failure = None
            return

        self._lm_head_gpu_attempted = True
        if not bool(getattr(self, "_allow_dense_gpu_lm_head", True)):
            if not str(getattr(self, "_lm_head_gpu_last_failure", "") or "").strip():
                self._lm_head_gpu_last_failure = "dense_gpu_lm_head_blocked_on_pre_ampere"
            return

        lm_head_weight_cpu = self._ensure_lm_head_weight_cpu()
        if lm_head_weight_cpu is None:
            return
        if self._lm_head_weight_gpu is not None:
            return

        required_bytes = int(lm_head_weight_cpu.numel()) * int(lm_head_weight_cpu.element_size())
        required_residual_bytes = self._gpu_lm_head_reserve_bytes()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        except Exception:
            free_bytes, total_bytes = 0, 0
        if free_bytes > 0 and (int(free_bytes) - int(required_bytes)) < int(required_residual_bytes):
            self._lm_head_gpu_last_failure = "insufficient_free_vram"
            print(
                f"[lm_head] staying on CPU; free={float(free_bytes) / (1024 ** 3):.2f} GiB "
                f"lm_head={float(required_bytes) / (1024 ** 3):.2f} GiB "
                f"reserve_after_load={float(required_residual_bytes) / (1024 ** 3):.2f} GiB "
                f"need_total={float(required_bytes + required_residual_bytes) / (1024 ** 3):.2f} GiB",
                flush=True,
            )
            return

        try:
            self._lm_head_weight_gpu = lm_head_weight_cpu.to(
                device=self.device,
                dtype=self.dtype,
                non_blocking=False,
            ).contiguous()
            self._lm_head_gpu_last_failure = None
            if self._lm_head_weight_gpu.device.type == "cuda":
                try:
                    gpu_free_bytes, gpu_total_bytes = torch.cuda.mem_get_info(self.device)
                    print(
                        f"[lm_head] resident on GPU: "
                        f"{float(required_bytes) / (1024 ** 3):.2f} GiB "
                        f"(free {float(gpu_free_bytes) / (1024 ** 3):.2f} / {float(gpu_total_bytes) / (1024 ** 3):.2f} GiB)",
                        flush=True,
                    )
                except Exception:
                    print(
                        f"[lm_head] resident on GPU: {float(required_bytes) / (1024 ** 3):.2f} GiB",
                        flush=True,
                    )
        except Exception as exc:
            self._lm_head_weight_gpu = None
            self._lm_head_gpu_last_failure = f"{type(exc).__name__}: {str(exc)[:200]}"
            if _is_cuda_oom_error(exc):
                if bool(getattr(self, "_explicit_gpu_lm_head", False)):
                    raise RuntimeError(
                        "STREAMING_GPU_LM_HEAD was explicitly enabled, but GPU LM-head materialization OOMed."
                    ) from exc
                print(
                    f"[lm_head] GPU materialization failed; keeping CPU path: {type(exc).__name__}: {str(exc)[:200]}",
                    flush=True,
                )
            else:
                raise

    def _lm_head_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_lm_head_nf4_meta_gpu", None) is None and self._lm_head_weight_gpu is None:
            self._materialize_lm_head_on_gpu()
        nf4_meta = getattr(self, "_lm_head_nf4_meta_gpu", None)
        if isinstance(nf4_meta, dict):
            hidden_flat = hidden.view(-1, hidden.shape[-1]).to(device=self.device, dtype=self.dtype).contiguous()
            rows = int(hidden_flat.shape[0])
            active_idx = nf4_meta["active_idx"]
            if rows != int(active_idx.shape[0]):
                active_idx = active_idx.expand(rows, -1).contiguous()
            logits_flat = triton_sparse_input_linear_4bit(
                hidden_flat,
                active_idx,
                packed_weight=nf4_meta["packed_weight"],
                absmax=nf4_meta["absmax"],
                code=nf4_meta["code"],
                out_features=int(nf4_meta["out_features"]),
                in_features=int(nf4_meta["in_features"]),
                quant_block_size=int(nf4_meta["quant_block_size"]),
                bias=None,
                block_size=int(nf4_meta["block_size"]),
                quant_weight_ref=None,
            )
            return logits_flat.view(int(hidden.shape[0]), int(hidden.shape[1]), int(nf4_meta["out_features"]))
        if self._lm_head_weight_gpu is not None:
            return F.linear(hidden.to(dtype=self._lm_head_weight_gpu.dtype), self._lm_head_weight_gpu)
        lm_head_weight_cpu = self._ensure_lm_head_weight_cpu()
        if lm_head_weight_cpu is None:
            raise RuntimeError("StreamingLlamaRuntime.generate requires materialize_lm_head=True")
        hidden_cpu = hidden.to(device=torch.device("cpu"), dtype=lm_head_weight_cpu.dtype)
        return F.linear(hidden_cpu, lm_head_weight_cpu)
