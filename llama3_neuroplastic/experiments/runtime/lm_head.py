from __future__ import annotations

import torch
import torch.nn.functional as F

from ._helpers import _is_cuda_oom_error


class RuntimeLmHeadMixin:
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


        if self.device.type == "cuda":
            cap = torch.cuda.get_device_capability(self.device)
            if cap < (7, 0):
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
            free_bytes, _total_bytes = 0, 0
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
