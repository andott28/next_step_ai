from __future__ import annotations

import gc
from contextlib import suppress
from typing import Any

import torch

try:
    from transformers.cache_utils import DynamicCache
except Exception:
    DynamicCache = None


class RuntimeSessionMixin:
    def _record_runtime_event(self, name: str, **fields: Any) -> None:
        events = getattr(self, "_runtime_events", None)
        if not isinstance(events, list):
            self._runtime_events = []
            events = self._runtime_events
        events.append({"name": str(name), **fields})
        max_events = int(getattr(self, "_runtime_events_max", 128) or 128)
        if len(events) > max_events:
            del events[:-max_events]

    def get_runtime_events(self) -> list[dict[str, Any]]:
        return list(getattr(self, "_runtime_events", []) or [])

    def reset_decode_profiler(self) -> None:
        if hasattr(self, "_decode_profile_steps"):
            self._decode_profile_steps.clear()
        self._last_decode_profile_report = None

    def enable_decode_profiler(self, enabled: bool = True, *, max_steps: int | None = None) -> None:
        self._decode_profile_enabled = bool(enabled)
        if max_steps is not None:
            self._decode_profile_max_steps = max(0, int(max_steps))
        self.reset_decode_profiler()

    def _begin_decode_profile_step(self, *, position_index: int) -> dict[str, Any] | None:
        if not bool(getattr(self, "_decode_profile_enabled", False)):
            return None
        if str(getattr(self, "_traffic_current_phase", "idle")) != "decode":
            return None
        return {
            "position_index": int(position_index),
            "layers": [],
        }

    def _record_decode_profile_layer(
        self,
        step: dict[str, Any] | None,
        *,
        layer_idx: int,
        cpu_ms: float,
        events: tuple[Any, Any, Any, Any] | None,
    ) -> None:
        if step is None:
            return
        step["layers"].append(
            {
                "layer_idx": int(layer_idx),
                "cpu_ms": float(cpu_ms),
                "_events": events,
            }
        )

    def _finalize_decode_profile_step(self, step: dict[str, Any] | None) -> dict[str, Any] | None:
        if step is None:
            return None
        if self.device.type == "cuda" and any(layer.get("_events") is not None for layer in step["layers"]):
            torch.cuda.synchronize(self.device)
        finalized_layers: list[dict[str, float | int]] = []
        total_attn_ms = 0.0
        total_mlp_ms = 0.0
        total_other_ms = 0.0
        total_layer_ms = 0.0
        total_cpu_ms = 0.0
        total_python_overhead_ms = 0.0
        for layer in step["layers"]:
            events = layer.pop("_events", None)
            if events is not None:
                ev_start, ev_attn_end, ev_mlp_end, ev_end = events
                attn_ms = float(ev_start.elapsed_time(ev_attn_end))
                mlp_ms = float(ev_attn_end.elapsed_time(ev_mlp_end))
                other_ms = float(ev_mlp_end.elapsed_time(ev_end))
                total_ms = float(ev_start.elapsed_time(ev_end))
            else:
                attn_ms = 0.0
                mlp_ms = 0.0
                other_ms = 0.0
                total_ms = float(layer["cpu_ms"])
            total_attn_ms += attn_ms
            total_mlp_ms += mlp_ms
            total_other_ms += other_ms
            total_layer_ms += total_ms
            total_cpu_ms += float(layer["cpu_ms"])
            python_overhead_ms = max(0.0, float(layer["cpu_ms"]) - float(total_ms))
            total_python_overhead_ms += python_overhead_ms
            finalized_layers.append(
                {
                    "layer_idx": int(layer["layer_idx"]),
                    "load_attn_ms": float(attn_ms),
                    "mlp_ms": float(mlp_ms),
                    "other_ms": float(other_ms),
                    "total_ms": float(total_ms),
                    "cpu_ms": float(layer["cpu_ms"]),
                    "python_overhead_ms": float(python_overhead_ms),
                }
            )
        finalized = {
            "position_index": int(step["position_index"]),
            "layers": finalized_layers,
            "summary": {
                "layers": int(len(finalized_layers)),
                "load_attn_ms": float(total_attn_ms),
                "mlp_ms": float(total_mlp_ms),
                "other_ms": float(total_other_ms),
                "total_ms": float(total_layer_ms),
                "cpu_ms": float(total_cpu_ms),
                "python_overhead_ms": float(total_python_overhead_ms),
                "mean_layer_total_ms": float(total_layer_ms) / float(max(len(finalized_layers), 1)),
                "mean_layer_python_overhead_ms": float(total_python_overhead_ms) / float(max(len(finalized_layers), 1)),
            },
        }
        steps = getattr(self, "_decode_profile_steps", None)
        if isinstance(steps, list):
            steps.append(finalized)
            max_steps = int(getattr(self, "_decode_profile_max_steps", 0) or 0)
            if max_steps > 0 and len(steps) > max_steps:
                del steps[:-max_steps]
        self._last_decode_profile_report = self.get_decode_profile_report()
        return finalized

    def get_decode_profile_report(self) -> dict[str, Any] | None:
        steps = list(getattr(self, "_decode_profile_steps", []) or [])
        if not steps:
            return None
        mean_total_ms = sum(float(step["summary"]["total_ms"]) for step in steps) / float(max(len(steps), 1))
        mean_attn_ms = sum(float(step["summary"]["load_attn_ms"]) for step in steps) / float(max(len(steps), 1))
        mean_mlp_ms = sum(float(step["summary"]["mlp_ms"]) for step in steps) / float(max(len(steps), 1))
        mean_other_ms = sum(float(step["summary"]["other_ms"]) for step in steps) / float(max(len(steps), 1))
        mean_python_overhead_ms = (
            sum(float(step["summary"].get("python_overhead_ms", 0.0)) for step in steps)
            / float(max(len(steps), 1))
        )
        return {
            "enabled": bool(getattr(self, "_decode_profile_enabled", False)),
            "steps_recorded": int(len(steps)),
            "summary": {
                "mean_total_ms": float(mean_total_ms),
                "mean_load_attn_ms": float(mean_attn_ms),
                "mean_mlp_ms": float(mean_mlp_ms),
                "mean_other_ms": float(mean_other_ms),
                "mean_python_overhead_ms": float(mean_python_overhead_ms),
            },
            "steps": steps,
        }

    def clear_sparse_transfer_caches(self, *, release_cuda: bool = False) -> None:
        if hasattr(self, "_sparse_block_transfer_cache"):
            self._sparse_block_transfer_cache.clear()
        if hasattr(self, "_downproj_transfer_cache"):
            self._downproj_transfer_cache.clear()
        if hasattr(self, "_mlp_prefetch_active_blocks"):
            self._mlp_prefetch_active_blocks.clear()
        if release_cuda and bool(getattr(self, "_hard_cuda_cache_flush", False)) and torch.cuda.is_available():
            torch.cuda.empty_cache()
            self._record_runtime_event("cuda_empty_cache", reason="clear_sparse_transfer_caches")

    def reset_caches(self) -> None:
        self._taylor_caches = [None for _ in range(self.num_layers)]
        self._dense_cache = DynamicCache(config=self.config) if DynamicCache is not None else None
        if hasattr(self, "_compact_attn_cache"):
            self._compact_attn_cache.clear()
        if self._token_archive is not None:
            self._token_archive.reset()
        if hasattr(self, "_single_kernel_mlp_out_accum"):
            self._single_kernel_mlp_out_accum = None
        if hasattr(self, "_single_kernel_mlp_tile_state"):
            self._single_kernel_mlp_tile_state = None
        if hasattr(self, "_single_kernel_mlp_tile_done"):
            self._single_kernel_mlp_tile_done = None
        if hasattr(self, "_single_kernel_mlp_epoch"):
            self._single_kernel_mlp_epoch = 0
        if hasattr(self, "_smc_valid"):
            self._smc_valid.fill_(False)
        if hasattr(self, "_smc_attn_valid"):
            self._smc_attn_valid.fill_(False)
        for _layer_idx in list(self._session_sparse_route_layers):
            self._sparse_routing.pop(int(_layer_idx), None)
            self._sparse_top_k_by_layer.pop(int(_layer_idx), None)
            self._sparse_basis_top_k_by_layer.pop(int(_layer_idx), None)
            self._mlp_hot_blocks_by_layer.pop(int(_layer_idx), None)
        self._session_sparse_route_layers.clear()
        if hasattr(self, "_down_proj_col_cache"):
            self._down_proj_col_cache.clear()
        self.clear_sparse_transfer_caches()
        gc.collect()
        if bool(getattr(self, "_hard_cuda_cache_flush", False)) and torch.cuda.is_available():
            torch.cuda.empty_cache()
            self._record_runtime_event("cuda_empty_cache", reason="reset_caches")

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
            dense_ok = target == 0
        elif hasattr(self._dense_cache, "crop"):
            self._dense_cache.crop(target)
            dense_ok = True
        else:
            dense_ok = False
        compact_cache = getattr(self, "_compact_attn_cache", None)
        if isinstance(compact_cache, dict):
            for layer_idx, entry in list(compact_cache.items()):
                if not isinstance(entry, dict):
                    compact_cache.pop(layer_idx, None)
                    continue
                k_tensor = entry.get("k")
                v_tensor = entry.get("v")
                if torch.is_tensor(k_tensor) and int(k_tensor.ndim) == 4:
                    entry["k"] = k_tensor[:, :, :target, :].contiguous()
                elif k_tensor is not None:
                    entry["k"] = None
                if torch.is_tensor(v_tensor) and int(v_tensor.ndim) == 4:
                    entry["v"] = v_tensor[:, :, :target, :].contiguous()
                elif v_tensor is not None:
                    entry["v"] = None
        return bool(dense_ok)

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

    def _set_session_state(self, token_ids_cpu: torch.LongTensor, logits: torch.Tensor | None) -> None:
        self._session_token_ids_cpu = token_ids_cpu.detach().to(device=torch.device("cpu"), dtype=torch.long).contiguous()
        if logits is None:
            self._session_last_logits_cpu = None
        else:
            self._session_last_logits_cpu = logits[:, -1:, :].detach().to(
                device=torch.device("cpu"),
                dtype=torch.float32,
            ).contiguous()

    def _reset_traffic_stats(self) -> None:
        self._traffic_current_phase = "idle"
        self._traffic_bytes_by_phase.clear()
        self._traffic_layer_visits_by_phase.clear()
        self._traffic_bytes_by_phase_layer.clear()
        self._traffic_layer_visits_by_phase_layer.clear()
        self._traffic_bytes_by_phase_tag.clear()
        self._last_traffic_report = None
        self._first_decode_t = None
        self._decode_done_t = None
        self._decode_mlp_hot_blocks_hit = 0
        self._decode_mlp_cold_blocks_streamed = 0
        self._decode_down_hot_blocks_hit = 0
        self._decode_down_cold_blocks_streamed = 0
        self._h2d_pinned_bytes = 0
        self._h2d_unpinned_bytes = 0
        self.reset_decode_profiler()

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
        layer_idx: int | None,
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
        bytes_by_phase: dict[str, int],
        layer_visits_by_phase: dict[str, int],
        bytes_by_phase_layer: dict[tuple[str, int], int],
        layer_visits_by_phase_layer: dict[tuple[str, int], int],
        bytes_by_phase_tag: dict[tuple[str, str], int],
    ) -> dict[str, Any]:
        total_bytes = int(bytes_by_phase.get(phase, 0))
        layer_visits = int(layer_visits_by_phase.get(phase, 0))
        avg_bytes = float(total_bytes) / float(max(layer_visits, 1))
        layer_avgs: dict[str, float] = {}
        for (phase_name, layer_idx), byte_count in bytes_by_phase_layer.items():
            if phase_name != phase:
                continue
            visits = int(layer_visits_by_phase_layer.get((phase_name, layer_idx), 0))
            layer_avgs[str(layer_idx)] = float(byte_count) / float(max(visits, 1)) / float(1024 ** 2)
        tag_totals: dict[str, float] = {}
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
        decode_report = self._build_phase_traffic_report(
            "decode",
            bytes_by_phase=self._traffic_bytes_by_phase,
            layer_visits_by_phase=self._traffic_layer_visits_by_phase,
            bytes_by_phase_layer=self._traffic_bytes_by_phase_layer,
            layer_visits_by_phase_layer=self._traffic_layer_visits_by_phase_layer,
            bytes_by_phase_tag=self._traffic_bytes_by_phase_tag,
        )
        decode_avg_mb = float(decode_report.get("avg_mb_per_layer", 0.0))
        decode_budget_mb = 20.0
        self._last_traffic_report = {
            "prefill": self._build_phase_traffic_report(
                "prefill",
                bytes_by_phase=self._traffic_bytes_by_phase,
                layer_visits_by_phase=self._traffic_layer_visits_by_phase,
                bytes_by_phase_layer=self._traffic_bytes_by_phase_layer,
                layer_visits_by_phase_layer=self._traffic_layer_visits_by_phase_layer,
                bytes_by_phase_tag=self._traffic_bytes_by_phase_tag,
            ),
            "decode": decode_report,
            "overall": {
                "total_bytes": int(total_bytes),
                "total_mb": float(total_bytes) / float(1024 ** 2),
                "layer_visits": int(total_layer_visits),
                "avg_bytes_per_layer": float(total_bytes) / float(max(total_layer_visits, 1)),
                "avg_mb_per_layer": float(total_bytes) / float(max(total_layer_visits, 1)) / float(1024 ** 2),
            },
            "h2d_budget": {
                "target_mb_per_layer": float(decode_budget_mb),
                "decode_avg_mb_per_layer": float(decode_avg_mb),
                "within_budget": bool(decode_avg_mb <= decode_budget_mb),
            },
        }
        if hasattr(self, "get_runtime_status"):
            with suppress(Exception):
                self._last_traffic_report["runtime_status"] = self.get_runtime_status()

    def get_last_traffic_report(self) -> dict[str, Any] | None:
        return self._last_traffic_report

    def begin_traffic_phase(self, phase: str) -> None:
        self._set_traffic_phase(str(phase))

    def finalize_traffic_report(self) -> None:
        self._finalize_traffic_report()

    def materialize_lm_head(self) -> None:
        self._materialize_lm_head_on_gpu()

    def prefill_logits(self, token_ids: torch.LongTensor, *, position_offset: int = 0) -> torch.Tensor:
        if token_ids.ndim != 2 or int(token_ids.shape[0]) != 1:
            raise RuntimeError("StreamingLlamaRuntime.prefill_logits currently supports batch_size=1 only")
        if int(token_ids.shape[1]) <= 0:
            raise RuntimeError("No prompt tokens were provided")
        if int(token_ids.shape[1]) > 1:
            return self._forward_prefill(token_ids.to(self.device), position_offset=int(position_offset))
        logits, _captures = self.forward_token(
            token_ids[:, 0:1].to(self.device),
            position_index=int(position_offset),
        )
        return logits

    def score_window_logits(self, token_ids: torch.LongTensor) -> torch.Tensor:
        if token_ids.ndim != 2 or int(token_ids.shape[0]) != 1:
            raise RuntimeError("score_window_logits currently supports batch_size=1 only")
        if int(token_ids.shape[1]) <= 1:
            raise RuntimeError("Need at least 2 tokens for window scoring")
        self.reset_caches()
        self.begin_traffic_phase("prefill")
        return self.prefill_logits(token_ids.to(self.device))

    def decode_token_logits(
        self,
        token_ids: torch.LongTensor,
        *,
        position_index: int,
    ) -> torch.Tensor:
        logits, _captures = self.forward_token(
            token_ids.to(self.device),
            position_index=int(position_index),
        )
        return logits

    def sample_next_token(
        self,
        logits: torch.Tensor,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = 50,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        return self._sample_next_token(
            logits,
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_k=top_k,
            top_p=float(top_p),
        )
