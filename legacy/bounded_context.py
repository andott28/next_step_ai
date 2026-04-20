from __future__ import annotations

import ast
import math
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import torch

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


def _estimate_tokens_from_text(text: str) -> int:


    return max(1, int(math.ceil(len(text) / 4.0)))


def gqa_query_head_to_kv_group(query_head_idx: int, num_attention_heads: int, num_key_value_heads: int) -> int:
    if num_attention_heads <= 0 or num_key_value_heads <= 0:
        return 0
    group = max(1, int(num_attention_heads) // int(num_key_value_heads))
    return min(int(query_head_idx) // group, int(num_key_value_heads) - 1)


def token_to_page_index(token_index: int, page_size_tokens: int) -> int:
    if page_size_tokens <= 0:
        raise ValueError("page_size_tokens must be > 0")
    return int(token_index) // int(page_size_tokens)


def page_index_to_token_span(page_index: int, page_size_tokens: int) -> tuple[int, int]:
    if page_size_tokens <= 0:
        raise ValueError("page_size_tokens must be > 0")
    start = int(page_index) * int(page_size_tokens)
    return start, start + int(page_size_tokens)


@dataclass(frozen=True)
class BoundedContextMemoryEstimate:
    bytes_per_token_all_layers: int
    local_cache_bytes: int
    global_cache_bytes: int
    sink_cache_bytes: int
    total_bytes: int
    budget_bytes: int

    @property
    def total_gib(self) -> float:
        return float(self.total_bytes) / (1024.0**3)

    @property
    def budget_gib(self) -> float:
        return float(self.budget_bytes) / (1024.0**3)

    @property
    def remaining_gib(self) -> float:
        return float(self.budget_bytes - self.total_bytes) / (1024.0**3)


@dataclass
class BoundedContextConfig:
    enabled: bool = False
    sink_tokens: int = 8
    local_window_tokens: int = 10_000
    global_window_tokens: int = 128_000
    global_group_id: int = 0
    global_start_layer: int = 84
    vram_budget_gib: float = 3.5
    kv_cache_bits: int = 4
    summary_interval_min_tokens: int = 4_000
    summary_interval_max_tokens: int = 8_000
    retrieval_chunk_min_tokens: int = 300
    retrieval_chunk_max_tokens: int = 800
    retrieval_top_k_min: int = 8
    retrieval_top_k_max: int = 20

    def validate(self) -> None:
        if self.sink_tokens < 0:
            raise ValueError("sink_tokens must be >= 0")
        if self.local_window_tokens <= 0:
            raise ValueError("local_window_tokens must be > 0")
        if self.global_window_tokens <= 0:
            raise ValueError("global_window_tokens must be > 0")
        if self.summary_interval_min_tokens <= 0:
            raise ValueError("summary_interval_min_tokens must be > 0")
        if self.summary_interval_max_tokens < self.summary_interval_min_tokens:
            raise ValueError("summary_interval_max_tokens must be >= summary_interval_min_tokens")
        if self.retrieval_chunk_min_tokens <= 0:
            raise ValueError("retrieval_chunk_min_tokens must be > 0")
        if self.retrieval_chunk_max_tokens < self.retrieval_chunk_min_tokens:
            raise ValueError("retrieval_chunk_max_tokens must be >= retrieval_chunk_min_tokens")
        if self.retrieval_top_k_min <= 0:
            raise ValueError("retrieval_top_k_min must be > 0")
        if self.retrieval_top_k_max < self.retrieval_top_k_min:
            raise ValueError("retrieval_top_k_max must be >= retrieval_top_k_min")
        if self.kv_cache_bits not in (2, 4, 8, 16):
            raise ValueError("kv_cache_bits must be one of {2, 4, 8, 16}")

    def validate_sparse_attention_runtime(
        self,
        *,
        strict_fully_sparse: bool,
        sca_use_cuda: bool,
        sca_spmm_impl: str,
        fast_fallback_threshold: float,
        disable_ssd_fetch_in_decode: bool,
        allow_noncuda_spmm_for_diagnostics: bool = False,
    ) -> None:
        if not bool(strict_fully_sparse):
            return
        if not bool(sca_use_cuda):
            raise ValueError("strict_fully_sparse requires sca_use_cuda=True")
        if str(sca_spmm_impl) != "cuda_spmm" and not bool(allow_noncuda_spmm_for_diagnostics):
            raise ValueError("strict_fully_sparse requires sca_spmm_impl='cuda_spmm'")
        if float(fast_fallback_threshold) > 0.0:
            raise ValueError("strict_fully_sparse requires fast fallback disabled (threshold <= 0)")
        if not bool(disable_ssd_fetch_in_decode):
            raise ValueError("strict_fully_sparse requires disable_ssd_fetch_in_decode=True")

    def resolve_global_start_layer(self, num_hidden_layers: int) -> int:
        if num_hidden_layers <= 0:
            return 0
        return _clamp_int(self.global_start_layer, 0, max(num_hidden_layers - 1, 0))

    def kv_bytes_per_value(self) -> float:
        return float(self.kv_cache_bits) / 8.0

    def estimate_memory(
        self,
        num_hidden_layers: int,
        num_key_value_heads: int,
        head_dim: int,
    ) -> BoundedContextMemoryEstimate:
        bytes_per_token_all_layers = int(
            2.0
            * float(num_hidden_layers)
            * float(num_key_value_heads)
            * float(head_dim)
            * self.kv_bytes_per_value()
        )

        local_cache_bytes = int(
            2.0
            * float(num_hidden_layers)
            * float(num_key_value_heads)
            * float(head_dim)
            * self.kv_bytes_per_value()
            * float(self.local_window_tokens)
        )

        global_layers = max(0, int(num_hidden_layers) - self.resolve_global_start_layer(num_hidden_layers))
        global_cache_bytes = int(
            2.0
            * float(global_layers)
            * 1.0
            * float(head_dim)
            * self.kv_bytes_per_value()
            * float(self.global_window_tokens)
        )

        sink_cache_bytes = int(
            2.0
            * float(num_hidden_layers)
            * float(num_key_value_heads)
            * float(head_dim)
            * self.kv_bytes_per_value()
            * float(self.sink_tokens)
        )

        total = local_cache_bytes + global_cache_bytes + sink_cache_bytes
        budget = int(float(self.vram_budget_gib) * (1024.0**3))
        return BoundedContextMemoryEstimate(
            bytes_per_token_all_layers=bytes_per_token_all_layers,
            local_cache_bytes=local_cache_bytes,
            global_cache_bytes=global_cache_bytes,
            sink_cache_bytes=sink_cache_bytes,
            total_bytes=total,
            budget_bytes=budget,
        )


@dataclass
class TierLayerSelection:
    layer_idx: int
    seq_len_before: int
    sink_positions: torch.Tensor
    local_positions: torch.Tensor
    global_positions: torch.Tensor | None
    retained_positions: torch.Tensor
    uses_global: bool


@dataclass
class TierCacheMetadata:
    layers: list[TierLayerSelection] = field(default_factory=list)

    def get_layer(self, layer_idx: int) -> TierLayerSelection | None:
        if layer_idx < 0 or layer_idx >= len(self.layers):
            return None
        return self.layers[layer_idx]


class TieredContextPolicy:
    def __init__(
        self,
        config: BoundedContextConfig,
        num_hidden_layers: int,
        num_key_value_heads: int,
        num_attention_heads: int,
    ) -> None:
        self.config = config
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_key_value_heads = int(num_key_value_heads)
        self.num_attention_heads = int(num_attention_heads)
        self.global_start_layer = self.config.resolve_global_start_layer(self.num_hidden_layers)

    def layer_uses_global(self, layer_idx: int) -> bool:
        return int(layer_idx) >= self.global_start_layer

    def _sink_positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
        n = min(int(seq_len), max(0, int(self.config.sink_tokens)))
        if n <= 0:
            return torch.empty((0,), dtype=torch.long, device=device)
        return torch.arange(n, device=device, dtype=torch.long)

    def _local_positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len <= 0:
            return torch.empty((0,), dtype=torch.long, device=device)
        start = max(0, int(seq_len) - int(self.config.local_window_tokens))
        return torch.arange(start, int(seq_len), device=device, dtype=torch.long)

    def _global_positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len <= 0:
            return torch.empty((0,), dtype=torch.long, device=device)
        start = max(0, int(seq_len) - int(self.config.global_window_tokens))
        return torch.arange(start, int(seq_len), device=device, dtype=torch.long)

    @staticmethod
    def _merge_positions(*positions: torch.Tensor) -> torch.Tensor:
        valid = [p for p in positions if p is not None and p.numel() > 0]
        if not valid:
            fallback_device = positions[0].device if positions else torch.device("cpu")
            return torch.empty((0,), dtype=torch.long, device=fallback_device)
        merged = torch.cat(valid, dim=0)
        merged = torch.unique(merged, sorted=True)
        return merged

    def select_layer_positions(
        self,
        layer_idx: int,
        seq_len: int,
        device: torch.device,
    ) -> TierLayerSelection:
        sink = self._sink_positions(seq_len=seq_len, device=device)
        local = self._local_positions(seq_len=seq_len, device=device)
        uses_global = self.layer_uses_global(layer_idx)
        global_pos = self._global_positions(seq_len=seq_len, device=device) if uses_global else None
        retained = self._merge_positions(sink, local, global_pos if global_pos is not None else torch.empty(0, dtype=torch.long, device=device))
        return TierLayerSelection(
            layer_idx=int(layer_idx),
            seq_len_before=int(seq_len),
            sink_positions=sink,
            local_positions=local,
            global_positions=global_pos,
            retained_positions=retained,
            uses_global=uses_global,
        )

    def _kv_group_for_query_head(self, head_idx: int) -> int:
        if self.num_key_value_heads <= 0:
            return 0
        group_size = max(1, self.num_attention_heads // self.num_key_value_heads)
        return min(int(head_idx) // group_size, self.num_key_value_heads - 1)

    def build_visibility_mask(
        self,
        layer_idx: int,
        kv_length: int,
        num_attention_heads: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        h = int(num_attention_heads) if num_attention_heads is not None else self.num_attention_heads
        device = device if device is not None else torch.device("cpu")
        if kv_length <= 0 or h <= 0:
            return torch.empty((h, 0), dtype=torch.bool, device=device)

        selection = self.select_layer_positions(layer_idx=layer_idx, seq_len=int(kv_length), device=device)
        local_visible = torch.zeros((kv_length,), dtype=torch.bool, device=device)
        if selection.sink_positions.numel() > 0:
            local_visible.index_fill_(0, selection.sink_positions, True)
        if selection.local_positions.numel() > 0:
            local_visible.index_fill_(0, selection.local_positions, True)

        global_visible = torch.zeros((kv_length,), dtype=torch.bool, device=device)
        if selection.global_positions is not None and selection.global_positions.numel() > 0:
            global_visible.index_fill_(0, selection.global_positions, True)
        else:
            global_visible = local_visible

        visibility = torch.zeros((h, kv_length), dtype=torch.bool, device=device)
        if not selection.uses_global:
            visibility[:, :] = local_visible
            return visibility

        for head_idx in range(h):
            kv_group = self._kv_group_for_query_head(head_idx)
            if kv_group == int(self.config.global_group_id):
                visibility[head_idx] = global_visible
            else:
                visibility[head_idx] = local_visible
        return visibility


class TieredAttentionMaskBundle:
    def __init__(self, policy: TieredContextPolicy) -> None:
        self.policy = policy

    def build_layer_mask(
        self,
        layer_idx: int,
        base_mask: torch.Tensor | None,
        kv_length: int,
        num_attention_heads: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if kv_length <= 0:
            return base_mask

        visibility = self.policy.build_visibility_mask(
            layer_idx=layer_idx,
            kv_length=kv_length,
            num_attention_heads=num_attention_heads,
            device=device,
        )
        if visibility.numel() == 0:
            return base_mask

        disallowed = ~visibility
        additive = torch.zeros((1, num_attention_heads, 1, kv_length), dtype=dtype, device=device)
        additive = additive.masked_fill(disallowed.view(1, num_attention_heads, 1, kv_length), torch.finfo(dtype).min)

        if base_mask is None:
            return additive

        mask = base_mask
        if mask.dim() != 4:
            return mask
        if mask.shape[1] == 1 and num_attention_heads > 1:
            mask = mask.expand(mask.shape[0], num_attention_heads, mask.shape[2], mask.shape[3])
        mask = mask[..., :kv_length].to(device=device)
        return mask + additive


class TieredCacheCompressor:
    def __init__(self, policy: TieredContextPolicy) -> None:
        self.policy = policy

    def compress_legacy_cache(
        self,
        legacy_cache: Sequence[Sequence[object]],
    ) -> tuple[tuple[tuple, ...], TierCacheMetadata]:
        compressed_layers: list[tuple] = []
        metadata = TierCacheMetadata()

        for layer_idx, layer_cache in enumerate(legacy_cache):
            layer_tuple = tuple(layer_cache)
            if len(layer_tuple) < 2:
                compressed_layers.append(layer_tuple)
                metadata.layers.append(
                    TierLayerSelection(
                        layer_idx=layer_idx,
                        seq_len_before=0,
                        sink_positions=torch.empty((0,), dtype=torch.long),
                        local_positions=torch.empty((0,), dtype=torch.long),
                        global_positions=None,
                        retained_positions=torch.empty((0,), dtype=torch.long),
                        uses_global=self.policy.layer_uses_global(layer_idx),
                    )
                )
                continue

            key_states, value_states = layer_tuple[0], layer_tuple[1]
            if not (torch.is_tensor(key_states) and torch.is_tensor(value_states)):
                compressed_layers.append(layer_tuple)
                metadata.layers.append(
                    TierLayerSelection(
                        layer_idx=layer_idx,
                        seq_len_before=0,
                        sink_positions=torch.empty((0,), dtype=torch.long),
                        local_positions=torch.empty((0,), dtype=torch.long),
                        global_positions=None,
                        retained_positions=torch.empty((0,), dtype=torch.long),
                        uses_global=self.policy.layer_uses_global(layer_idx),
                    )
                )
                continue

            seq_len = int(key_states.shape[-2])
            select = self.policy.select_layer_positions(
                layer_idx=layer_idx,
                seq_len=seq_len,
                device=key_states.device,
            )
            keep = select.retained_positions

            if keep.numel() == 0:
                new_k = key_states[..., :0, :]
                new_v = value_states[..., :0, :]
            elif keep.numel() == seq_len:
                new_k, new_v = key_states, value_states
            else:
                keep = keep.to(device=key_states.device, dtype=torch.long)
                new_k = torch.index_select(key_states, dim=-2, index=keep)
                new_v = torch.index_select(value_states, dim=-2, index=keep)

            compressed = (new_k, new_v) if len(layer_tuple) == 2 else (new_k, new_v, *layer_tuple[2:])
            compressed_layers.append(compressed)
            metadata.layers.append(select)

        return tuple(compressed_layers), metadata


@dataclass
class RollingSummarySnapshot:
    total_tokens: int
    summary_text: str
    summary_token_ids: torch.Tensor | None


class RollingSummaryManager:
    def __init__(
        self,
        min_interval_tokens: int,
        max_interval_tokens: int,
    ) -> None:
        self.min_interval_tokens = int(min_interval_tokens)
        self.max_interval_tokens = int(max_interval_tokens)
        self.total_tokens = 0
        self.last_summary_at = 0
        self.summary_text = ""
        self.summary_token_ids: torch.Tensor | None = None
        self.pending_refresh = False

    def observe(self, total_tokens: int) -> bool:
        self.total_tokens = int(total_tokens)
        since = self.total_tokens - self.last_summary_at
        if since >= self.max_interval_tokens:
            self.pending_refresh = True
            return True
        if since >= self.min_interval_tokens:
            self.pending_refresh = True
            return True
        return False

    def update_summary(self, summary_text: str, summary_token_ids: torch.Tensor | None = None) -> None:
        self.summary_text = str(summary_text)
        self.summary_token_ids = summary_token_ids.detach().clone() if summary_token_ids is not None else None
        self.last_summary_at = int(self.total_tokens)
        self.pending_refresh = False

    def snapshot(self) -> RollingSummarySnapshot:
        token_ids = self.summary_token_ids.detach().clone() if self.summary_token_ids is not None else None
        return RollingSummarySnapshot(
            total_tokens=int(self.total_tokens),
            summary_text=self.summary_text,
            summary_token_ids=token_ids,
        )


@dataclass
class RepoChunk:
    chunk_id: str
    path: str
    start_line: int
    end_line: int
    text: str
    token_estimate: int
    symbols: tuple[str, ...]
    imports: tuple[str, ...]
    terms: dict[str, int]


class CodeRepoMemoryIndex:
    def __init__(
        self,
        root_dir: str,
        chunk_min_tokens: int = 300,
        chunk_max_tokens: int = 800,
    ) -> None:
        self.root_dir = os.path.abspath(root_dir)
        self.chunk_min_tokens = int(chunk_min_tokens)
        self.chunk_max_tokens = int(chunk_max_tokens)
        self.chunks: list[RepoChunk] = []

    def _iter_files(self) -> Iterable[str]:
        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "node_modules", ".venv", "verification_env"}]
            for fn in files:
                if fn.endswith((".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".sh", ".ps1", ".ts", ".tsx", ".js", ".jsx", ".java", ".rs", ".go", ".cpp", ".c", ".h")):
                    yield os.path.join(root, fn)

    @staticmethod
    def _term_histogram(text: str) -> dict[str, int]:
        terms: dict[str, int] = {}
        for term in _TOKEN_RE.findall(text.lower()):
            terms[term] = terms.get(term, 0) + 1
        return terms

    @staticmethod
    def _python_symbols_and_imports(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        symbols: list[str] = []
        imports: list[str] = []
        try:
            tree = ast.parse(text)
        except Exception:
            return tuple(), tuple()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.append(str(node.name))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(str(alias.name))
            elif isinstance(node, ast.ImportFrom):
                module = str(node.module) if node.module is not None else ""
                if module:
                    imports.append(module)
        return tuple(sorted(set(symbols))), tuple(sorted(set(imports)))

    def _chunk_text(
        self,
        path: str,
        text: str,
    ) -> list[RepoChunk]:
        lines = text.splitlines()
        if not lines:
            return []

        chunks: list[RepoChunk] = []
        start = 0
        current_chars = 0
        chunk_idx = 0
        min_chars = self.chunk_min_tokens * 4
        max_chars = self.chunk_max_tokens * 4

        for idx, line in enumerate(lines):
            current_chars += len(line) + 1
            should_cut = current_chars >= max_chars
            if not should_cut and current_chars >= min_chars:
                stripped = line.strip()
                if stripped.startswith(("def ", "class ", "async def ", "# ", "## ", "### ")):
                    should_cut = True
            if not should_cut:
                continue

            chunk_text = "\n".join(lines[start : idx + 1]).strip()
            if chunk_text:
                token_est = _estimate_tokens_from_text(chunk_text)
                symbols, imports = self._python_symbols_and_imports(chunk_text) if path.endswith(".py") else (tuple(), tuple())
                rel_path = os.path.relpath(path, self.root_dir)
                chunk_id = f"{rel_path}:{start+1}-{idx+1}:{chunk_idx}"
                chunks.append(
                    RepoChunk(
                        chunk_id=chunk_id,
                        path=rel_path,
                        start_line=start + 1,
                        end_line=idx + 1,
                        text=chunk_text,
                        token_estimate=token_est,
                        symbols=symbols,
                        imports=imports,
                        terms=self._term_histogram(chunk_text),
                    )
                )
                chunk_idx += 1
            start = idx + 1
            current_chars = 0

        if start < len(lines):
            chunk_text = "\n".join(lines[start:]).strip()
            if chunk_text:
                token_est = _estimate_tokens_from_text(chunk_text)
                symbols, imports = self._python_symbols_and_imports(chunk_text) if path.endswith(".py") else (tuple(), tuple())
                rel_path = os.path.relpath(path, self.root_dir)
                chunk_id = f"{rel_path}:{start+1}-{len(lines)}:{chunk_idx}"
                chunks.append(
                    RepoChunk(
                        chunk_id=chunk_id,
                        path=rel_path,
                        start_line=start + 1,
                        end_line=len(lines),
                        text=chunk_text,
                        token_estimate=token_est,
                        symbols=symbols,
                        imports=imports,
                        terms=self._term_histogram(chunk_text),
                    )
                )
        return chunks

    def build(self) -> int:
        self.chunks = []
        for path in self._iter_files():
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                try:
                    with open(path, encoding="latin-1") as f:
                        text = f.read()
                except Exception:
                    continue
            except Exception:
                continue
            self.chunks.extend(self._chunk_text(path=path, text=text))
        return len(self.chunks)

    @staticmethod
    def _score_terms(query_terms: dict[str, int], chunk_terms: dict[str, int]) -> float:
        if not query_terms or not chunk_terms:
            return 0.0
        score = 0.0
        for term, q_count in query_terms.items():
            c_count = chunk_terms.get(term, 0)
            if c_count <= 0:
                continue
            score += float(min(q_count, c_count))
        norm = math.sqrt(float(sum(v * v for v in query_terms.values()))) * math.sqrt(
            float(sum(v * v for v in chunk_terms.values()))
        )
        if norm <= 0:
            return 0.0
        return score / norm

    def retrieve(self, query: str, top_k: int = 8, min_k: int | None = None, max_k: int | None = None) -> list[RepoChunk]:
        if not self.chunks:
            return []
        q_terms = self._term_histogram(query)
        k_lo = int(min_k) if min_k is not None else 1
        k_hi = int(max_k) if max_k is not None else len(self.chunks)
        k = _clamp_int(int(top_k), k_lo, k_hi)

        scored: list[tuple[float, RepoChunk]] = []
        for chunk in self.chunks:
            score = self._score_terms(q_terms, chunk.terms)
            if score <= 0:
                continue
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:k]]


def apply_retrieved_chunks_to_prompt(
    prompt_ids: torch.LongTensor,
    retrieved_chunk_ids: Sequence[torch.LongTensor],
    local_window_tokens: int,
) -> torch.LongTensor:
    if prompt_ids.dim() != 2:
        raise ValueError("prompt_ids must have shape [batch, seq]")
    if prompt_ids.shape[0] != 1:

        return prompt_ids

    additions: list[torch.Tensor] = []
    for chunk in retrieved_chunk_ids:
        if chunk is None:
            continue
        if chunk.dim() == 1:
            additions.append(chunk.view(1, -1).to(device=prompt_ids.device, dtype=prompt_ids.dtype))
        elif chunk.dim() == 2 and chunk.shape[0] == 1:
            additions.append(chunk.to(device=prompt_ids.device, dtype=prompt_ids.dtype))
    if not additions:
        return prompt_ids

    merged = torch.cat([prompt_ids] + additions, dim=-1)
    if merged.shape[-1] <= int(local_window_tokens):
        return merged
    return merged[:, -int(local_window_tokens) :]
