from __future__ import annotations

import gc
from typing import Any

import torch

try:
    from transformers.cache_utils import DynamicCache
except Exception:
    DynamicCache = None


class RuntimeSessionMixin:
    def reset_caches(self) -> None:
        self._taylor_caches = [None for _ in range(self.num_layers)]
        self._dense_cache = DynamicCache(config=self.config) if DynamicCache is not None else None
        if self._token_archive is not None:
            self._token_archive.reset()
        if hasattr(self, "_smc_valid"):
            self._smc_valid.fill_(False)
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
