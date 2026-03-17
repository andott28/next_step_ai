from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


def page_for_token(token_index: int, page_size_tokens: int) -> int:
    if page_size_tokens <= 0:
        raise ValueError("page_size_tokens must be > 0")
    return int(token_index) // int(page_size_tokens)


def page_token_span(page_index: int, page_size_tokens: int) -> Tuple[int, int]:
    if page_size_tokens <= 0:
        raise ValueError("page_size_tokens must be > 0")
    start = int(page_index) * int(page_size_tokens)
    end = start + int(page_size_tokens)
    return start, end


def gqa_query_head_to_kv_group(query_head_idx: int, num_attention_heads: int, num_key_value_heads: int) -> int:
    if num_attention_heads <= 0 or num_key_value_heads <= 0:
        return 0
    group = max(1, int(num_attention_heads) // int(num_key_value_heads))
    return min(int(query_head_idx) // group, int(num_key_value_heads) - 1)


@dataclass
class SparseAttentionConfig:
    enabled: bool = False
    local_window_tokens: int = 2048
    sink_tokens: int = 8
    page_size_tokens: int = 256
    retrieval_top_k_pages: int = 8
    retrieval_head_group_ids: Tuple[int, ...] = (0,)
    retrieval_start_layer: Optional[int] = None
    archive_cpu_dtype: str = "int4"
    hot_archive_gpu_pages: int = 0
    disable_ssd_fetch_in_decode: bool = True
    force_single_model_runtime: bool = True
    strict_fully_sparse: bool = False

    def validate(self) -> None:
        if self.local_window_tokens <= 0:
            raise ValueError("local_window_tokens must be > 0")
        if self.sink_tokens < 0:
            raise ValueError("sink_tokens must be >= 0")
        if self.page_size_tokens <= 0:
            raise ValueError("page_size_tokens must be > 0")
        if self.retrieval_top_k_pages <= 0:
            raise ValueError("retrieval_top_k_pages must be > 0")
        if self.archive_cpu_dtype not in ("int4", "fp16"):
            raise ValueError("archive_cpu_dtype must be one of {'int4', 'fp16'}")


@dataclass
class SparseAttentionStepStats:
    step_index: int
    selected_pages_by_layer: Dict[int, List[int]] = field(default_factory=dict)
    bytes_cpu_to_gpu: int = 0
    archive_tokens: int = 0
    fallback_triggered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": int(self.step_index),
            "selected_pages_by_layer": {str(k): [int(x) for x in v] for k, v in self.selected_pages_by_layer.items()},
            "bytes_cpu_to_gpu": int(self.bytes_cpu_to_gpu),
            "archive_tokens": int(self.archive_tokens),
            "fallback_triggered": bool(self.fallback_triggered),
        }


def _quantize_to_int4(x: torch.Tensor) -> Dict[str, Any]:
    flat = x.reshape(-1).float()
    max_abs = float(flat.abs().max().item()) if flat.numel() > 0 else 1.0
    scale = max(max_abs / 7.0, 1e-8)
    q = torch.clamp(torch.round(flat / scale), min=-8, max=7).to(torch.int16)
    q_u4 = (q + 8).to(torch.uint8)
    if q_u4.numel() % 2 == 1:
        q_u4 = torch.cat([q_u4, torch.zeros((1,), dtype=torch.uint8, device=q_u4.device)], dim=0)
    lo = q_u4[0::2]
    hi = q_u4[1::2]
    packed = (lo | (hi << 4)).contiguous().cpu()
    return {
        "packed": packed,
        "scale": float(scale),
        "shape": tuple(x.shape),
        "numel": int(x.numel()),
        "dtype": str(x.dtype),
    }


def _dequantize_from_int4(payload: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    packed = payload["packed"].to(device=device)
    scale = float(payload["scale"])
    numel = int(payload["numel"])
    lo = (packed & 0x0F).to(torch.int16)
    hi = ((packed >> 4) & 0x0F).to(torch.int16)
    vals = torch.empty((packed.numel() * 2,), dtype=torch.int16, device=device)
    vals[0::2] = lo
    vals[1::2] = hi
    vals = vals[:numel] - 8
    out = (vals.float() * scale).to(dtype=dtype)
    return out.view(*payload["shape"])


@dataclass
class _HeadArchiveState:
    key_pages: List[Any] = field(default_factory=list)
    value_pages: List[Any] = field(default_factory=list)
    summary_pages: List[torch.Tensor] = field(default_factory=list)
    current_key_tokens: List[torch.Tensor] = field(default_factory=list)
    current_value_tokens: List[torch.Tensor] = field(default_factory=list)
    total_tokens: int = 0


class LongRangePageArchive:
    def __init__(self, config: SparseAttentionConfig, head_dim: int) -> None:
        self.config = config
        self.head_dim = int(head_dim)
        self._layers: Dict[int, Dict[int, _HeadArchiveState]] = {}

    def reset(self) -> None:
        self._layers.clear()

    def _head_state(self, layer_idx: int, head_idx: int) -> _HeadArchiveState:
        layer = self._layers.setdefault(int(layer_idx), {})
        return layer.setdefault(int(head_idx), _HeadArchiveState())

    def _serialize_page(self, x: torch.Tensor) -> Any:
        cpu = x.detach().to(device=torch.device("cpu"), dtype=torch.float16).contiguous()
        if self.config.archive_cpu_dtype == "int4":
            return _quantize_to_int4(cpu)
        return cpu

    def _deserialize_page(self, x: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(x, dict) and "packed" in x:
            return _dequantize_from_int4(x, device=device, dtype=dtype)
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype, non_blocking=True)
        raise TypeError("Unsupported page payload")

    def append_token(
        self,
        layer_idx: int,
        head_idx: int,
        key_token: torch.Tensor,
        value_token: torch.Tensor,
    ) -> None:
        state = self._head_state(layer_idx=layer_idx, head_idx=head_idx)
        key_cpu = key_token.detach().to(device=torch.device("cpu"), dtype=torch.float16).view(1, -1).contiguous()
        value_cpu = value_token.detach().to(device=torch.device("cpu"), dtype=torch.float16).view(1, -1).contiguous()
        state.current_key_tokens.append(key_cpu)
        state.current_value_tokens.append(value_cpu)
        state.total_tokens += 1
        if len(state.current_key_tokens) >= int(self.config.page_size_tokens):
            self.flush_partial_page(layer_idx=layer_idx, head_idx=head_idx)

    def append_sequence(
        self,
        layer_idx: int,
        head_idx: int,
        key_tokens: torch.Tensor,
        value_tokens: torch.Tensor,
    ) -> None:
        if key_tokens.numel() == 0:
            return
        k = key_tokens.detach().to(device=torch.device("cpu"), dtype=torch.float16).contiguous()
        v = value_tokens.detach().to(device=torch.device("cpu"), dtype=torch.float16).contiguous()
        if k.dim() != 2 or v.dim() != 2:
            raise ValueError("append_sequence expects [T, D] tensors")
        if k.shape != v.shape:
            raise ValueError("key/value sequence shape mismatch")
        state = self._head_state(layer_idx=layer_idx, head_idx=head_idx)
        state.total_tokens += int(k.shape[0])
        page_size = int(self.config.page_size_tokens)
        cursor = 0
        while cursor < int(k.shape[0]):
            end = min(cursor + page_size, int(k.shape[0]))
            key_page = k[cursor:end]
            value_page = v[cursor:end]
            summary = key_page.mean(dim=0).contiguous().to(dtype=torch.float32)
            state.key_pages.append(self._serialize_page(key_page))
            state.value_pages.append(self._serialize_page(value_page))
            state.summary_pages.append(summary.cpu())
            cursor = end

    def flush_partial_page(self, layer_idx: int, head_idx: int) -> None:
        state = self._head_state(layer_idx=layer_idx, head_idx=head_idx)
        if not state.current_key_tokens:
            return
        key_page = torch.cat(state.current_key_tokens, dim=0).contiguous()
        value_page = torch.cat(state.current_value_tokens, dim=0).contiguous()
        summary = key_page.mean(dim=0).contiguous().to(dtype=torch.float32)
        state.key_pages.append(self._serialize_page(key_page))
        state.value_pages.append(self._serialize_page(value_page))
        state.summary_pages.append(summary.cpu())
        state.current_key_tokens.clear()
        state.current_value_tokens.clear()

    def num_pages(self, layer_idx: int, head_idx: int) -> int:
        state = self._head_state(layer_idx=layer_idx, head_idx=head_idx)
        return len(state.summary_pages)

    def get_summaries(self, layer_idx: int, head_idx: int, device: torch.device) -> torch.Tensor:
        state = self._head_state(layer_idx=layer_idx, head_idx=head_idx)
        if not state.summary_pages:
            return torch.empty((0, self.head_dim), device=device, dtype=torch.float32)
        return torch.stack(state.summary_pages, dim=0).to(device=device, dtype=torch.float32, non_blocking=True)

    def fetch_pages(
        self,
        layer_idx: int,
        head_idx: int,
        page_indices: Sequence[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        state = self._head_state(layer_idx=layer_idx, head_idx=head_idx)
        if not page_indices:
            empty = torch.empty((0, self.head_dim), device=device, dtype=dtype)
            return empty, empty, 0
        keys: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        bytes_copied = 0
        for idx in page_indices:
            i = int(idx)
            if i < 0 or i >= len(state.key_pages):
                continue
            key_page = self._deserialize_page(state.key_pages[i], device=device, dtype=dtype)
            value_page = self._deserialize_page(state.value_pages[i], device=device, dtype=dtype)
            keys.append(key_page)
            values.append(value_page)
            bytes_copied += int(key_page.numel() + value_page.numel()) * int(torch.tensor([], dtype=dtype).element_size())
        if not keys:
            empty = torch.empty((0, self.head_dim), device=device, dtype=dtype)
            return empty, empty, 0
        return torch.cat(keys, dim=0), torch.cat(values, dim=0), bytes_copied

    def token_count(self) -> int:
        total = 0
        for layer in self._layers.values():
            for state in layer.values():
                total += int(state.total_tokens)
        return total


class LongRangeSummaryTable:
    def __init__(self) -> None:
        self._cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def update(self, layer_idx: int, head_idx: int, summaries: torch.Tensor) -> None:
        self._cache[(int(layer_idx), int(head_idx))] = summaries.detach()

    def get(self, layer_idx: int, head_idx: int) -> Optional[torch.Tensor]:
        return self._cache.get((int(layer_idx), int(head_idx)))

    def reset(self) -> None:
        self._cache.clear()


class TwoStagePageRetriever:
    def __init__(self, config: SparseAttentionConfig, archive: LongRangePageArchive, table: LongRangeSummaryTable) -> None:
        self.config = config
        self.archive = archive
        self.table = table

    def retrieve(
        self,
        layer_idx: int,
        head_idx: int,
        query_vec: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[List[int], torch.Tensor, torch.Tensor, int]:
        summaries = self.table.get(layer_idx=layer_idx, head_idx=head_idx)
        if summaries is None or summaries.numel() == 0:
            empty = torch.empty((0, int(query_vec.numel())), device=device, dtype=dtype)
            return [], empty, empty, 0
        q = query_vec.detach().to(device=summaries.device, dtype=torch.float32).view(1, -1)
        scores = torch.matmul(summaries, q.t()).view(-1)
        k = min(int(self.config.retrieval_top_k_pages), int(scores.numel()))
        if k <= 0:
            empty = torch.empty((0, int(query_vec.numel())), device=device, dtype=dtype)
            return [], empty, empty, 0
        top = torch.topk(scores, k=k, dim=0, largest=True, sorted=True).indices.detach().cpu().tolist()
        keys, values, bytes_copied = self.archive.fetch_pages(
            layer_idx=layer_idx,
            head_idx=head_idx,
            page_indices=top,
            device=device,
            dtype=dtype,
        )
        return [int(x) for x in top], keys, values, int(bytes_copied)


class SparseAttentionRuntime:
    def __init__(self, config: SparseAttentionConfig, num_layers: int, num_attention_heads: int, num_key_value_heads: int, head_dim: int) -> None:
        self.config = config
        self.num_layers = int(num_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.config.validate()
        self.archive = LongRangePageArchive(config=config, head_dim=self.head_dim)
        self.summary_table = LongRangeSummaryTable()
        self.retriever = TwoStagePageRetriever(config=config, archive=self.archive, table=self.summary_table)
        self._steps: List[SparseAttentionStepStats] = []
        self._step_counter = 0

    def reset(self) -> None:
        self.archive.reset()
        self.summary_table.reset()
        self._steps.clear()
        self._step_counter = 0

    def _retrieval_start_layer(self) -> int:
        if self.config.retrieval_start_layer is None:
            return max(0, int(self.num_layers * 2 // 3))
        return max(0, min(int(self.config.retrieval_start_layer), self.num_layers - 1))

    def _layer_uses_retrieval(self, layer_idx: int) -> bool:
        return int(layer_idx) >= self._retrieval_start_layer()

    def _head_uses_retrieval(self, kv_head_idx: int) -> bool:
        return int(kv_head_idx) in set(int(x) for x in self.config.retrieval_head_group_ids)

    def _local_keep_positions(self, total_tokens: int) -> List[int]:
        if total_tokens <= 0:
            return []
        sink = min(int(self.config.sink_tokens), total_tokens)
        start = max(0, total_tokens - int(self.config.local_window_tokens))
        keep = set(range(sink))
        keep.update(range(start, total_tokens))
        return sorted(int(x) for x in keep)

    def _strict_validate(self, sparse_mlp_diagnostics: Optional[Dict[str, Any]]) -> None:
        if not self.config.strict_fully_sparse:
            return
        if not self.config.disable_ssd_fetch_in_decode:
            raise RuntimeError("strict_fully_sparse: disable_ssd_fetch_in_decode must be true")
        if sparse_mlp_diagnostics is not None:
            fast_fallback_detected = False
            for layer in sparse_mlp_diagnostics.get("layers", []):
                if float(layer.get("dense_fallback_rate", 0.0)) > 0.0:
                    fast_fallback_detected = True
                    break
            if fast_fallback_detected:
                raise RuntimeError("strict_fully_sparse: sparse MLP fallback rate must be zero")

    def update_and_compress_cache(
        self,
        legacy_cache: Sequence[Sequence[Any]],
        sparse_mlp_diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tuple[tuple, ...], SparseAttentionStepStats]:
        self._strict_validate(sparse_mlp_diagnostics=sparse_mlp_diagnostics)
        step = SparseAttentionStepStats(step_index=int(self._step_counter))
        self._step_counter += 1
        compressed_layers: List[tuple] = []

        for layer_idx, layer_cache in enumerate(legacy_cache):
            layer_tuple = tuple(layer_cache)
            if len(layer_tuple) < 2:
                compressed_layers.append(layer_tuple)
                continue
            key_states, value_states = layer_tuple[0], layer_tuple[1]
            if not (torch.is_tensor(key_states) and torch.is_tensor(value_states)):
                compressed_layers.append(layer_tuple)
                continue
            if key_states.dim() != 4:
                compressed_layers.append(layer_tuple)
                continue
            bsz, kv_heads, seq_len, head_dim = key_states.shape
            if bsz != 1 or head_dim != self.head_dim:
                compressed_layers.append(layer_tuple)
                continue

            # Bootstrap from full cache once; afterwards append only newest token.
            layer_needs_bootstrap = int(seq_len) > 1 and self.archive.num_pages(layer_idx=layer_idx, head_idx=0) == 0
            if layer_needs_bootstrap:
                seq_k = key_states[0]
                seq_v = value_states[0]
                for h in range(int(kv_heads)):
                    self.archive.append_sequence(layer_idx=layer_idx, head_idx=h, key_tokens=seq_k[h], value_tokens=seq_v[h])
                    summaries = self.archive.get_summaries(layer_idx=layer_idx, head_idx=h, device=seq_k.device)
                    self.summary_table.update(layer_idx=layer_idx, head_idx=h, summaries=summaries)
                last_k = seq_k[:, -1, :].detach()
            else:
                last_k = key_states[:, :, -1, :].detach().squeeze(0)
                last_v = value_states[:, :, -1, :].detach().squeeze(0)
                for h in range(int(kv_heads)):
                    self.archive.append_token(layer_idx=layer_idx, head_idx=h, key_token=last_k[h], value_token=last_v[h])
                    summaries = self.archive.get_summaries(layer_idx=layer_idx, head_idx=h, device=last_k.device)
                    self.summary_table.update(layer_idx=layer_idx, head_idx=h, summaries=summaries)

            keep_positions = set(self._local_keep_positions(total_tokens=int(seq_len)))
            selected_pages_layer: List[int] = []
            retrieval_needed = int(seq_len) > int(self.config.local_window_tokens + self.config.sink_tokens)
            if self._layer_uses_retrieval(layer_idx) and retrieval_needed:
                for h in range(int(kv_heads)):
                    if not self._head_uses_retrieval(kv_head_idx=h):
                        continue
                    query = last_k[h].detach().to(dtype=torch.float32)
                    top_pages, _, _, bytes_copied = self.retriever.retrieve(
                        layer_idx=layer_idx,
                        head_idx=h,
                        query_vec=query,
                        device=key_states.device,
                        dtype=key_states.dtype,
                    )
                    step.bytes_cpu_to_gpu += int(bytes_copied)
                    selected_pages_layer.extend(int(p) for p in top_pages)
                    for page_idx in top_pages:
                        start, end = page_token_span(page_index=int(page_idx), page_size_tokens=int(self.config.page_size_tokens))
                        keep_positions.update(range(start, min(end, int(seq_len))))
                if selected_pages_layer:
                    step.selected_pages_by_layer[int(layer_idx)] = sorted(set(int(x) for x in selected_pages_layer))

            if not keep_positions:
                keep_positions.add(max(0, int(seq_len) - 1))
            keep = torch.tensor(sorted(x for x in keep_positions if 0 <= x < int(seq_len)), device=key_states.device, dtype=torch.long)
            new_k = torch.index_select(key_states, dim=-2, index=keep)
            new_v = torch.index_select(value_states, dim=-2, index=keep)

            if len(layer_tuple) == 2:
                compressed_layers.append((new_k, new_v))
            else:
                compressed_layers.append((new_k, new_v, *layer_tuple[2:]))

        step.archive_tokens = int(self.archive.token_count())
        self._steps.append(step)
        return tuple(compressed_layers), step

    def diagnostics(self) -> Dict[str, Any]:
        total_bytes = sum(int(s.bytes_cpu_to_gpu) for s in self._steps)
        total_pages = sum(sum(len(v) for v in s.selected_pages_by_layer.values()) for s in self._steps)
        steps = len(self._steps)
        return {
            "enabled": bool(self.config.enabled),
            "steps": int(steps),
            "total_bytes_cpu_to_gpu": int(total_bytes),
            "mean_bytes_cpu_to_gpu_per_step": float(total_bytes / max(steps, 1)),
            "total_selected_pages": int(total_pages),
            "mean_selected_pages_per_step": float(total_pages / max(steps, 1)),
            "last_step": self._steps[-1].to_dict() if self._steps else {},
            "config": {
                "local_window_tokens": int(self.config.local_window_tokens),
                "sink_tokens": int(self.config.sink_tokens),
                "page_size_tokens": int(self.config.page_size_tokens),
                "retrieval_top_k_pages": int(self.config.retrieval_top_k_pages),
                "retrieval_head_group_ids": [int(x) for x in self.config.retrieval_head_group_ids],
                "retrieval_start_layer": self._retrieval_start_layer(),
                "archive_cpu_dtype": str(self.config.archive_cpu_dtype),
                "strict_fully_sparse": bool(self.config.strict_fully_sparse),
            },
        }
