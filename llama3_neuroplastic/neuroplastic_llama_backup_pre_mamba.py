from __future__ import annotations

from dataclasses import dataclass
import json
import os
import types
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

try:
    from .bounded_context import (
        BoundedContextConfig,
        CodeRepoMemoryIndex,
        RollingSummaryManager,
        TierCacheMetadata,
        TieredAttentionMaskBundle,
        TieredCacheCompressor,
        TieredContextPolicy,
        apply_retrieved_chunks_to_prompt,
    )
    from .cuda.sca_sparse_loader import SCACUDAExtensionError, SCACUDAKernels
    from .sca_sparse_adapter import compute_active_blocks
    from .sca_sparse_config import SCASparseConfig, build_block_centers, build_inhibition_matrix
    from .sca_sparse_mlp import SparseLlamaMLP, SparseRouteSelection
except ImportError:  # Script-mode fallback
    from bounded_context import (
        BoundedContextConfig,
        CodeRepoMemoryIndex,
        RollingSummaryManager,
        TierCacheMetadata,
        TieredAttentionMaskBundle,
        TieredCacheCompressor,
        TieredContextPolicy,
        apply_retrieved_chunks_to_prompt,
    )
    from cuda.sca_sparse_loader import SCACUDAExtensionError, SCACUDAKernels
    from sca_sparse_adapter import compute_active_blocks
    from sca_sparse_config import SCASparseConfig, build_block_centers, build_inhibition_matrix
    from sca_sparse_mlp import SparseLlamaMLP, SparseRouteSelection


@dataclass
class _PackedInt4Tensor:
    packed: torch.Tensor
    shape: Tuple[int, ...]
    scale: float
    numel: int
    source_device: str
    target_dtype: torch.dtype


@dataclass
class _CPUOffloadedTensor:
    tensor: torch.Tensor
    source_device: str


class NeuroplasticLlama(nn.Module):
    def __init__(
        self,
        model_name: str = "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        num_tasks: int = 10,
        neuroplasticity_enabled: bool = True,
        sca_block_size: int = 32,
        sca_block_rank: int = 4,
        sca_top_k: int = 3,
        sca_sigma: float = 1.0,
        sca_refractory_steps: int = 100,
        sca_inhibition_lambda: float = 0.0,
        sca_grid_size: int = 16,
        sca_use_cuda: bool = True,
        sca_spmm_impl: str = "dense",
        sca_soft_mask: bool = True,
        sca_grouped_row_gemm: bool = False,
        sca_grouped_row_min_bucket: int = 2,
        sca_grouped_row_allow_4bit_dequant: bool = False,
        kv_int4_quantization: bool = True,
        kv_cpu_offload: bool = True,
        bounded_context_enabled: bool = False,
        bounded_sink_tokens: int = 8,
        bounded_local_window_tokens: int = 10_000,
        bounded_global_window_tokens: int = 128_000,
        bounded_global_start_layer: int = 84,
        bounded_global_group_id: int = 0,
        bounded_vram_budget_gib: float = 3.5,
        summary_interval_min_tokens: int = 4_000,
        summary_interval_max_tokens: int = 8_000,
        retrieval_chunk_min_tokens: int = 300,
        retrieval_chunk_max_tokens: int = 800,
        retrieval_top_k_min: int = 8,
        retrieval_top_k_max: int = 20,
        **_: object,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_tasks = int(num_tasks)
        self.neuroplasticity_enabled = bool(neuroplasticity_enabled)
        self.buffer_layers = 2
        self.collect_bio_gate_telemetry = False  # compatibility flag
        self.kv_int4_quantization = bool(kv_int4_quantization)
        self.kv_cpu_offload = bool(kv_cpu_offload or kv_int4_quantization)
        self.bounded_context_config = BoundedContextConfig(
            enabled=bool(bounded_context_enabled),
            sink_tokens=int(bounded_sink_tokens),
            local_window_tokens=int(bounded_local_window_tokens),
            global_window_tokens=int(bounded_global_window_tokens),
            global_group_id=int(bounded_global_group_id),
            global_start_layer=int(bounded_global_start_layer),
            vram_budget_gib=float(bounded_vram_budget_gib),
            summary_interval_min_tokens=int(summary_interval_min_tokens),
            summary_interval_max_tokens=int(summary_interval_max_tokens),
            retrieval_chunk_min_tokens=int(retrieval_chunk_min_tokens),
            retrieval_chunk_max_tokens=int(retrieval_chunk_max_tokens),
            retrieval_top_k_min=int(retrieval_top_k_min),
            retrieval_top_k_max=int(retrieval_top_k_max),
        )
        self.bounded_context_config.validate()
        self._tier_policy: Optional[TieredContextPolicy] = None
        self._tier_cache_compressor: Optional[TieredCacheCompressor] = None
        self._tier_mask_bundle: Optional[TieredAttentionMaskBundle] = None
        self._latest_tier_metadata: Optional[TierCacheMetadata] = None
        self._summary_manager = RollingSummaryManager(
            min_interval_tokens=self.bounded_context_config.summary_interval_min_tokens,
            max_interval_tokens=self.bounded_context_config.summary_interval_max_tokens,
        )
        self._summary_provider: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
        self._repo_index: Optional[CodeRepoMemoryIndex] = None

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.model = LlamaForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        for p in self.model.parameters():
            p.requires_grad = False
        if self.bounded_context_config.enabled:
            self.model.config._attn_implementation = "eager"

        self.config = self.model.config
        self.hidden_size = int(self.config.hidden_size)
        num_hidden_layers = int(getattr(self.config, "num_hidden_layers", len(self.model.model.layers)))
        num_key_value_heads = int(getattr(self.config, "num_key_value_heads", getattr(self.config, "num_attention_heads", 1)))
        num_attention_heads = int(getattr(self.config, "num_attention_heads", num_key_value_heads))
        self._tier_policy = TieredContextPolicy(
            config=self.bounded_context_config,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=num_key_value_heads,
            num_attention_heads=num_attention_heads,
        )
        self._tier_cache_compressor = TieredCacheCompressor(self._tier_policy)
        self._tier_mask_bundle = TieredAttentionMaskBundle(self._tier_policy)
        self._bounded_memory_estimate = self.bounded_context_config.estimate_memory(
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=num_key_value_heads,
            head_dim=int(getattr(self.config, "head_dim", self.hidden_size // max(num_attention_heads, 1))),
        )
        self.sca_config = SCASparseConfig(
            hidden_size=self.hidden_size,
            block_size=sca_block_size,
            block_rank=sca_block_rank,
            top_k=sca_top_k,
            sigma=sca_sigma,
            refractory_steps=sca_refractory_steps,
            inhibition_lambda=sca_inhibition_lambda,
            use_cuda=sca_use_cuda,
            grid_size=sca_grid_size,
            spmm_impl=sca_spmm_impl,
            soft_mask=sca_soft_mask,
            grouped_row_gemm=sca_grouped_row_gemm,
            grouped_row_min_bucket=sca_grouped_row_min_bucket,
            grouped_row_allow_4bit_dequant=sca_grouped_row_allow_4bit_dequant,
        )

        self.task_embedding = nn.Embedding(self.num_tasks, self.hidden_size)
        self.spatial_proj = nn.Linear(self.hidden_size, 3, bias=True)
        nn.init.xavier_uniform_(self.spatial_proj.weight)
        nn.init.zeros_(self.spatial_proj.bias)

        centers = build_block_centers(self.sca_config)
        inhibition = build_inhibition_matrix(centers, radius=self.sca_config.inhibition_radius)
        self.register_buffer("block_centers", centers.float(), persistent=True)
        self.register_buffer("inhibition_matrix", inhibition.float(), persistent=True)

        self.sca_adapters = nn.ModuleList()  # backward-compat placeholder (legacy checkpoints/scripts)
        self.adapters = self.sca_adapters
        self.adapter_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float16), requires_grad=False)
        self.score = nn.Linear(self.hidden_size, 1)  # compatibility for old scripts
        self.sca_sparse_mlps: list[SparseLlamaMLP] = []

        self.sca_cuda_kernels = None
        if self.sca_config.use_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("sca_use_cuda=True requires CUDA runtime")
            try:
                self.sca_cuda_kernels = SCACUDAKernels.from_build(verbose=False)
            except SCACUDAExtensionError as exc:
                raise RuntimeError(str(exc)) from exc

        num_layers = len(self.model.model.layers)
        self.register_buffer(
            "refractory_until",
            torch.zeros((num_layers, self.sca_config.num_blocks), dtype=torch.int32),
            persistent=False,
        )
        self._routing_decode_mode = False
        self._routing_step = 0

        self._replace_mlp_with_sparse_wrappers()
        self._install_bounded_context_attention_hooks()

        self._current_task_id = 0
        self._cached_task_emb: Optional[torch.Tensor] = None

        # Keep trainable parts on model device.
        dev = self.device
        self.task_embedding.to(device=dev, dtype=torch.float16)
        self.spatial_proj.to(device=dev, dtype=torch.float16)
        self.score.to(device=dev, dtype=torch.float16)
        self.adapter_scale.data = self.adapter_scale.data.to(device=dev, dtype=torch.float16)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def set_kv_cache_mode(self, int4_quantization: Optional[bool] = None, cpu_offload: Optional[bool] = None) -> None:
        if int4_quantization is not None:
            self.kv_int4_quantization = bool(int4_quantization)
        if cpu_offload is not None:
            self.kv_cpu_offload = bool(cpu_offload)
        if self.kv_int4_quantization:
            self.kv_cpu_offload = True

    def set_bounded_context_mode(
        self,
        enabled: Optional[bool] = None,
        sink_tokens: Optional[int] = None,
        local_window_tokens: Optional[int] = None,
        global_window_tokens: Optional[int] = None,
        global_start_layer: Optional[int] = None,
        global_group_id: Optional[int] = None,
    ) -> None:
        if enabled is not None:
            self.bounded_context_config.enabled = bool(enabled)
        if sink_tokens is not None:
            self.bounded_context_config.sink_tokens = int(sink_tokens)
        if local_window_tokens is not None:
            self.bounded_context_config.local_window_tokens = int(local_window_tokens)
        if global_window_tokens is not None:
            self.bounded_context_config.global_window_tokens = int(global_window_tokens)
        if global_start_layer is not None:
            self.bounded_context_config.global_start_layer = int(global_start_layer)
        if global_group_id is not None:
            self.bounded_context_config.global_group_id = int(global_group_id)
        self.bounded_context_config.validate()

        if self._tier_policy is not None:
            self._tier_policy = TieredContextPolicy(
                config=self.bounded_context_config,
                num_hidden_layers=self._tier_policy.num_hidden_layers,
                num_key_value_heads=self._tier_policy.num_key_value_heads,
                num_attention_heads=self._tier_policy.num_attention_heads,
            )
            self._tier_cache_compressor = TieredCacheCompressor(self._tier_policy)
            self._tier_mask_bundle = TieredAttentionMaskBundle(self._tier_policy)

    def register_summary_provider(self, provider: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]) -> None:
        self._summary_provider = provider

    def get_bounded_context_summary_state(self) -> Dict[str, Any]:
        snap = self._summary_manager.snapshot()
        token_ids = snap.summary_token_ids.tolist() if snap.summary_token_ids is not None else None
        return {
            "total_tokens": int(snap.total_tokens),
            "summary_text": snap.summary_text,
            "summary_token_ids": token_ids,
            "pending_refresh": bool(self._summary_manager.pending_refresh),
        }

    def get_bounded_context_memory_estimate(self) -> Dict[str, float]:
        estimate = self._bounded_memory_estimate
        return {
            "bytes_per_token_all_layers": float(estimate.bytes_per_token_all_layers),
            "local_cache_gib": float(estimate.local_cache_bytes) / (1024.0**3),
            "global_cache_gib": float(estimate.global_cache_bytes) / (1024.0**3),
            "sink_cache_gib": float(estimate.sink_cache_bytes) / (1024.0**3),
            "total_cache_gib": float(estimate.total_bytes) / (1024.0**3),
            "budget_gib": float(estimate.budget_gib),
            "remaining_gib": float(estimate.remaining_gib),
        }

    def build_repo_memory_index(self, root_dir: str) -> int:
        self._repo_index = CodeRepoMemoryIndex(
            root_dir=root_dir,
            chunk_min_tokens=self.bounded_context_config.retrieval_chunk_min_tokens,
            chunk_max_tokens=self.bounded_context_config.retrieval_chunk_max_tokens,
        )
        return int(self._repo_index.build())

    def retrieve_repo_chunks(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if self._repo_index is None:
            return []
        top_k = int(top_k) if top_k is not None else self.bounded_context_config.retrieval_top_k_min
        chunks = self._repo_index.retrieve(
            query=query,
            top_k=top_k,
            min_k=self.bounded_context_config.retrieval_top_k_min,
            max_k=self.bounded_context_config.retrieval_top_k_max,
        )
        return [
            {
                "chunk_id": c.chunk_id,
                "path": c.path,
                "start_line": int(c.start_line),
                "end_line": int(c.end_line),
                "token_estimate": int(c.token_estimate),
                "symbols": list(c.symbols),
                "imports": list(c.imports),
                "text": c.text,
            }
            for c in chunks
        ]

    def _normalize_active_scores(self, active_idx: torch.Tensor, active_score: torch.Tensor) -> torch.Tensor:
        valid = active_idx >= 0
        neg_inf = torch.full_like(active_score, float("-inf"))
        masked = torch.where(valid, active_score, neg_inf)
        weights = torch.softmax(masked, dim=-1)
        return torch.where(valid, weights, torch.zeros_like(weights))

    def _kv_optimized_generation_enabled(self) -> bool:
        return bool(self.kv_int4_quantization or self.kv_cpu_offload)

    @staticmethod
    def _to_legacy_cache(past_key_values: Any) -> Any:
        if past_key_values is None:
            return None
        if hasattr(past_key_values, "to_legacy_cache"):
            return past_key_values.to_legacy_cache()
        return past_key_values

    def _resolve_device(self, source_device: str) -> torch.device:
        try:
            resolved = torch.device(source_device)
        except Exception:
            resolved = self.device
        if resolved.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return resolved

    @staticmethod
    def _normalize_eos_token_ids(
        eos_token_id: Optional[Union[int, List[int], Tuple[int, ...]]],
    ) -> Optional[List[int]]:
        if eos_token_id is None:
            return None
        if isinstance(eos_token_id, int):
            return [int(eos_token_id)]
        return [int(token_id) for token_id in eos_token_id]

    @staticmethod
    def _pack_int4_tensor(tensor: torch.Tensor) -> _PackedInt4Tensor:
        source = tensor.detach().to(device="cpu", dtype=torch.float32)
        max_abs = float(source.abs().amax().item()) if source.numel() > 0 else 0.0
        scale = max(max_abs / 7.0, 1e-8)
        quant = torch.round(source / scale).clamp(-8, 7).to(torch.int16) + 8
        flat = quant.reshape(-1).to(torch.uint8)
        numel = int(flat.numel())
        if numel % 2 != 0:
            flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8)], dim=0)
        packed = (flat[0::2] | (flat[1::2] << 4)).contiguous()
        return _PackedInt4Tensor(
            packed=packed,
            shape=tuple(source.shape),
            scale=scale,
            numel=numel,
            source_device=str(tensor.device),
            target_dtype=tensor.dtype,
        )

    def _unpack_int4_tensor(self, packed_tensor: _PackedInt4Tensor) -> torch.Tensor:
        packed = packed_tensor.packed
        low = (packed & 0x0F).to(torch.int16)
        high = ((packed >> 4) & 0x0F).to(torch.int16)
        flat = torch.empty((packed.numel() * 2,), dtype=torch.int16)
        flat[0::2] = low
        flat[1::2] = high
        flat = flat[: packed_tensor.numel] - 8
        restored = (flat.to(torch.float32) * packed_tensor.scale).reshape(packed_tensor.shape)
        return restored.to(
            device=self._resolve_device(packed_tensor.source_device),
            dtype=packed_tensor.target_dtype,
            non_blocking=True,
        )

    @staticmethod
    def _offload_tensor_to_cpu(tensor: torch.Tensor) -> _CPUOffloadedTensor:
        return _CPUOffloadedTensor(
            tensor=tensor.detach().to(device="cpu"),
            source_device=str(tensor.device),
        )

    def _restore_offloaded_tensor(self, offloaded_tensor: _CPUOffloadedTensor) -> torch.Tensor:
        return offloaded_tensor.tensor.to(
            device=self._resolve_device(offloaded_tensor.source_device),
            dtype=offloaded_tensor.tensor.dtype,
            non_blocking=True,
        )

    def _bounded_context_enabled(self) -> bool:
        return bool(
            self.bounded_context_config.enabled
            and self._tier_policy is not None
            and self._tier_cache_compressor is not None
            and self._tier_mask_bundle is not None
        )

    def _get_tier_mask_bundle(self) -> Optional[TieredAttentionMaskBundle]:
        if not self._bounded_context_enabled():
            return None
        return self._tier_mask_bundle

    def _install_bounded_context_attention_hooks(self) -> None:
        for layer in self.model.model.layers:
            attn = layer.self_attn
            if getattr(attn, "_bounded_context_hooked", False):
                continue

            def _forward_with_tier_mask(
                attn_self,
                hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor],
                attention_mask: Optional[torch.Tensor],
                past_key_values: Optional[Any] = None,
                cache_position: Optional[torch.LongTensor] = None,
                **kwargs: Any,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                tier_mask_bundle = kwargs.pop("tier_mask_bundle", None)
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, attn_self.head_dim)

                query_states = attn_self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                key_states = attn_self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                value_states = attn_self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                if past_key_values is not None:
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    key_states, value_states = past_key_values.update(
                        key_states,
                        value_states,
                        attn_self.layer_idx,
                        cache_kwargs,
                    )

                layer_attention_mask = attention_mask
                if isinstance(tier_mask_bundle, TieredAttentionMaskBundle):
                    layer_attention_mask = tier_mask_bundle.build_layer_mask(
                        layer_idx=int(attn_self.layer_idx),
                        base_mask=attention_mask,
                        kv_length=int(key_states.shape[-2]),
                        num_attention_heads=int(query_states.shape[1]),
                        dtype=query_states.dtype,
                        device=query_states.device,
                    )

                attention_interface: Callable = eager_attention_forward
                if attn_self.config._attn_implementation != "eager":
                    attention_interface = ALL_ATTENTION_FUNCTIONS[attn_self.config._attn_implementation]

                attn_output, attn_weights = attention_interface(
                    attn_self,
                    query_states,
                    key_states,
                    value_states,
                    layer_attention_mask,
                    dropout=0.0 if not attn_self.training else attn_self.attention_dropout,
                    scaling=attn_self.scaling,
                    **kwargs,
                )

                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = attn_self.o_proj(attn_output)
                return attn_output, attn_weights

            attn.forward = types.MethodType(_forward_with_tier_mask, attn)
            attn._bounded_context_hooked = True

    def _maybe_update_summary(self, generated: torch.LongTensor) -> None:
        if generated is None:
            return
        total_tokens = int(generated.shape[-1])
        if not self._summary_manager.observe(total_tokens):
            return
        if self._summary_provider is None:
            return
        payload = self._summary_provider(
            {
                "total_tokens": total_tokens,
                "latest_tier_metadata": self._latest_tier_metadata,
            }
        )
        if not isinstance(payload, dict):
            return
        summary_text = str(payload.get("summary_text", ""))
        summary_token_ids = payload.get("summary_token_ids", None)
        if summary_token_ids is not None and torch.is_tensor(summary_token_ids):
            if summary_token_ids.dim() == 1:
                summary_token_ids = summary_token_ids.view(1, -1)
            elif summary_token_ids.dim() != 2:
                summary_token_ids = None
        else:
            summary_token_ids = None
        self._summary_manager.update_summary(summary_text=summary_text, summary_token_ids=summary_token_ids)

    def _pack_past_key_values(self, past_key_values: Any) -> Any:
        legacy_cache = self._to_legacy_cache(past_key_values)
        if legacy_cache is None:
            return legacy_cache

        if self._bounded_context_enabled() and self._tier_cache_compressor is not None:
            legacy_cache, self._latest_tier_metadata = self._tier_cache_compressor.compress_legacy_cache(legacy_cache)

        if not self._kv_optimized_generation_enabled():
            return legacy_cache
        packed_layers = []
        for layer_cache in legacy_cache:
            packed_entries = []
            for entry_idx, entry in enumerate(layer_cache):
                if not torch.is_tensor(entry):
                    packed_entries.append(entry)
                    continue
                if self.kv_int4_quantization and entry_idx < 2 and entry.is_floating_point():
                    packed_entries.append(self._pack_int4_tensor(entry))
                elif self.kv_cpu_offload:
                    packed_entries.append(self._offload_tensor_to_cpu(entry))
                else:
                    packed_entries.append(entry)
            packed_layers.append(tuple(packed_entries))
        return tuple(packed_layers)

    def _restore_past_key_values(self, packed_past_key_values: Any) -> Any:
        if packed_past_key_values is None or not self._kv_optimized_generation_enabled():
            return packed_past_key_values
        restored_layers = []
        for layer_cache in packed_past_key_values:
            restored_entries = []
            for entry in layer_cache:
                if isinstance(entry, _PackedInt4Tensor):
                    restored_entries.append(self._unpack_int4_tensor(entry))
                elif isinstance(entry, _CPUOffloadedTensor):
                    restored_entries.append(self._restore_offloaded_tensor(entry))
                else:
                    restored_entries.append(entry)
            restored_layers.append(tuple(restored_entries))
        return tuple(restored_layers)

    @staticmethod
    def _sample_next_token(
        logits: torch.Tensor,
        do_sample: bool,
        temperature: float,
        top_k: Optional[int],
        top_p: float,
    ) -> torch.Tensor:
        if (not do_sample) or temperature <= 0.0:
            return torch.argmax(logits, dim=-1)

        scores = logits / max(float(temperature), 1e-5)

        if top_k is not None and top_k > 0:
            top_k = min(int(top_k), scores.size(-1))
            topk_values = torch.topk(scores, top_k, dim=-1).values
            min_topk = topk_values[:, -1].unsqueeze(-1)
            scores = scores.masked_fill(scores < min_topk, float("-inf"))

        if 0.0 < top_p < 1.0:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_scores, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumulative_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            sorted_scores = sorted_scores.masked_fill(sorted_mask, float("-inf"))
            scores = torch.full_like(scores, float("-inf"))
            scores.scatter_(dim=-1, index=sorted_indices, src=sorted_scores)

        probs = torch.softmax(scores, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs_sum = probs.sum(dim=-1, keepdim=True)
        fallback = probs_sum.squeeze(-1) <= 0
        probs = probs / probs_sum.clamp_min(1e-12)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        if fallback.any():
            greedy = torch.argmax(logits, dim=-1)
            sampled = torch.where(fallback, greedy, sampled)
        return sampled

    def register_hooks(self) -> None:
        # Backward-compat no-op: routing now lives inside SparseLlamaMLP wrappers.
        return None

    def _replace_mlp_with_sparse_wrappers(self) -> None:
        self.sca_sparse_mlps = []
        for layer_idx, layer in enumerate(self.model.model.layers):
            if isinstance(layer.mlp, SparseLlamaMLP):
                wrapped = layer.mlp
            else:
                wrapped = SparseLlamaMLP(
                    base_mlp=layer.mlp,
                    config=self.sca_config,
                    layer_idx=layer_idx,
                    route_fn=self._route_sparse_feature_banks,
                    enabled_fn=self._sparse_layer_enabled,
                )
                layer.mlp = wrapped
            self.sca_sparse_mlps.append(wrapped)

    def _sparse_layer_enabled(self, layer_idx: int) -> bool:
        if not self.neuroplasticity_enabled:
            return False
        return layer_idx < (len(self.model.model.layers) - self.buffer_layers)

    def _prepare_routing_state(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.FloatTensor],
        past_key_values: Optional[List[torch.FloatTensor]],
    ) -> None:
        if not self.neuroplasticity_enabled:
            self._routing_decode_mode = False
            return

        seq_len = 0
        if inputs_embeds is not None:
            seq_len = int(inputs_embeds.shape[1])
        elif input_ids is not None:
            seq_len = int(input_ids.shape[1])

        decode_mode = bool(past_key_values is not None and seq_len == 1)
        self._routing_decode_mode = decode_mode
        if not decode_mode:
            self._routing_step = 0
            if self.refractory_until.numel() > 0:
                self.refractory_until.zero_()

    def _update_refractory_state(self, layer_idx: int, active_idx: torch.Tensor) -> None:
        if self.sca_config.refractory_steps <= 0:
            return
        valid = active_idx[active_idx >= 0]
        if valid.numel() == 0:
            return

        target = self.refractory_until[layer_idx]
        block_ids = torch.unique(valid).to(device=target.device, dtype=torch.long)
        value = int(self._routing_step + self.sca_config.refractory_steps)
        values = torch.full((block_ids.shape[0],), value, device=target.device, dtype=target.dtype)
        target[block_ids] = values

    def _route_sparse_feature_banks(self, hidden_states: torch.Tensor, layer_idx: int) -> SparseRouteSelection:
        flat_hidden = hidden_states.reshape(-1, self.hidden_size)
        query_in = F.layer_norm(flat_hidden.float(), (self.hidden_size,))
        spatial_device = self.spatial_proj.weight.device
        spatial_dtype = self.spatial_proj.weight.dtype
        query_raw = self.spatial_proj(query_in.to(device=spatial_device, dtype=spatial_dtype))
        query = (torch.sigmoid(query_raw) * float(self.sca_config.grid_size - 1)).reshape(-1, 3)
        query = query.to(device=hidden_states.device, dtype=torch.float32)

        use_cuda_kernel = (
            (not self.training)
            and self.sca_config.use_cuda
            and (self.sca_cuda_kernels is not None)
            and query.is_cuda
        )
        inhibition = self.inhibition_matrix.to(device=query.device)
        block_centers = self.block_centers.to(device=query.device)
        decode_mode = bool(self._routing_decode_mode and self.sca_config.refractory_steps > 0)
        refractory_until = None
        if decode_mode:
            refractory_until = self.refractory_until[layer_idx].to(device=query.device)

        active_idx, active_score = compute_active_blocks(
            query=query,
            block_centers=block_centers,
            config=self.sca_config,
            refractory_until=refractory_until,
            step=int(self._routing_step),
            decode_mode=decode_mode,
            inhibition_matrix=inhibition,
            use_cuda_kernel=use_cuda_kernel,
            cuda_kernels=self.sca_cuda_kernels,
        )
        if decode_mode:
            self._update_refractory_state(layer_idx, active_idx)

        score_weights = self._normalize_active_scores(active_idx, active_score)
        return SparseRouteSelection(active_idx=active_idx.long(), score_weights=score_weights)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        task_id: int = 0,
    ):
        self._current_task_id = int(task_id)
        self._cached_task_emb = self.task_embedding(torch.tensor(self._current_task_id, device=self.device))

        # Keep clean baseline path when disabled: no task embedding injection.
        if self.neuroplasticity_enabled and inputs_embeds is None and input_ids is not None:
            embeds = self.model.get_input_embeddings()(input_ids).to(dtype=torch.float16)
            task_bias = (self._cached_task_emb * 0.01).to(dtype=embeds.dtype).view(1, 1, -1)
            inputs_embeds = embeds + task_bias
            input_ids = None

        self._prepare_routing_state(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
        )
        tier_mask_bundle = self._get_tier_mask_bundle()
        model_kwargs: Dict[str, Any] = {}
        # LlamaForCausalLM forwards unknown kwargs into loss_function when labels are set.
        # Keep tier-specific kwargs on generation path only.
        if tier_mask_bundle is not None and labels is None:
            model_kwargs["tier_mask_bundle"] = tier_mask_bundle

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **model_kwargs,
        )
        if self.neuroplasticity_enabled and self._routing_decode_mode:
            self._routing_step += 1
        return outputs

    def forward_dual_stream(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        task_id: int = 0,
    ) -> dict[str, Any]:
        if input_ids is None:
            raise ValueError("input_ids must be provided.")

        prev_enabled = self.neuroplasticity_enabled

        self.neuroplasticity_enabled = False
        with torch.no_grad():
            base_out = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=use_cache,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                task_id=task_id,
            )

        self.neuroplasticity_enabled = prev_enabled
        adapted_out = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            task_id=task_id,
        )

        self.neuroplasticity_enabled = prev_enabled

        num_blocks = max(int(getattr(self.sca_config, "num_blocks", 1)), 1)
        top_k = int(getattr(self.sca_config, "top_k", 0))
        active_ratio = float(top_k) / float(num_blocks)
        bio_stats = {
            "active_block_ratio": active_ratio,
            "biological_active_ratio": active_ratio,
            "refractory_violation_rate": 0.0,
            "pattern_instability": 0.0,
            "active_score_mean": 0.0,
        }
        return {
            "base_logits": base_out.logits,
            "adapted_logits": adapted_out.logits,
            "bio_stats": bio_stats,
        }

    def _generate_with_kv_optimizations(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: float = 1.0,
        use_cache: bool = True,
        eos_token_id: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
        pad_token_id: Optional[int] = None,
        task_id: int = 0,
        retrieved_chunk_ids: Optional[List[torch.LongTensor]] = None,
    ) -> torch.LongTensor:
        input_ids = input_ids.to(device=self.device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
        else:
            attention_mask = attention_mask.to(device=input_ids.device)

        if self._bounded_context_enabled() and retrieved_chunk_ids:
            input_ids = apply_retrieved_chunks_to_prompt(
                prompt_ids=input_ids,
                retrieved_chunk_ids=retrieved_chunk_ids,
                local_window_tokens=int(self.bounded_context_config.local_window_tokens),
            )
            attention_mask = torch.ones_like(input_ids, dtype=attention_mask.dtype, device=attention_mask.device)

        if self._bounded_context_enabled():
            summary_snapshot = self._summary_manager.snapshot()
            summary_ids = summary_snapshot.summary_token_ids
            if summary_ids is not None and summary_ids.numel() > 0:
                summary_ids = summary_ids.to(device=input_ids.device, dtype=input_ids.dtype)
                if summary_ids.dim() == 1:
                    summary_ids = summary_ids.view(1, -1)
                if summary_ids.shape[0] == 1 and input_ids.shape[0] > 1:
                    summary_ids = summary_ids.expand(input_ids.shape[0], -1)
                if summary_ids.shape[0] == input_ids.shape[0]:
                    sink = min(int(self.bounded_context_config.sink_tokens), int(input_ids.shape[-1]))
                    prefix = input_ids[:, :sink]
                    suffix = input_ids[:, sink:]
                    input_ids = torch.cat([prefix, summary_ids, suffix], dim=-1)
                    summary_mask = torch.ones(
                        (attention_mask.shape[0], summary_ids.shape[-1]),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    attention_mask = torch.cat([attention_mask[:, :sink], summary_mask, attention_mask[:, sink:]], dim=-1)

        generation_cfg = getattr(self.model, "generation_config", None)
        if max_new_tokens is None:
            cfg_default = getattr(generation_cfg, "max_new_tokens", None) if generation_cfg is not None else None
            max_new_tokens = int(cfg_default) if cfg_default is not None else 50
        max_new_tokens = int(max_new_tokens)
        if max_new_tokens <= 0:
            return input_ids

        eos_ids = self._normalize_eos_token_ids(eos_token_id)
        if eos_ids is None and generation_cfg is not None:
            eos_ids = self._normalize_eos_token_ids(getattr(generation_cfg, "eos_token_id", None))

        if pad_token_id is None and generation_cfg is not None:
            cfg_pad = getattr(generation_cfg, "pad_token_id", None)
            if cfg_pad is not None:
                pad_token_id = int(cfg_pad)
        if pad_token_id is None and eos_ids is not None and len(eos_ids) > 0:
            pad_token_id = int(eos_ids[0])

        generated = input_ids
        unfinished = torch.ones((generated.shape[0],), dtype=torch.bool, device=generated.device)
        cached_state = None
        prefill_outputs = None

        stream_prefill = bool(use_cache and self._bounded_context_enabled() and generated.shape[-1] > 1)
        if stream_prefill:
            prefill_tokens = generated[:, :1]
            prefill_mask = attention_mask[:, :1]
            for pos in range(int(generated.shape[-1]) - 1):
                model_inputs = prefill_tokens if cached_state is None else prefill_tokens[:, -1:]
                past_key_values = self._restore_past_key_values(cached_state) if cached_state is not None else None
                outputs = self(
                    input_ids=model_inputs,
                    attention_mask=prefill_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    return_dict=True,
                    task_id=task_id,
                )
                cached_state = self._pack_past_key_values(outputs.past_key_values)
                next_prompt_token = generated[:, pos + 1 : pos + 2]
                next_prompt_mask = attention_mask[:, pos + 1 : pos + 2]
                prefill_tokens = torch.cat([prefill_tokens, next_prompt_token], dim=-1)
                prefill_mask = torch.cat([prefill_mask, next_prompt_mask], dim=-1)
                self._maybe_update_summary(prefill_tokens)

            past_key_values = self._restore_past_key_values(cached_state) if cached_state is not None else None
            prefill_outputs = self(
                input_ids=prefill_tokens[:, -1:],
                attention_mask=prefill_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                return_dict=True,
                task_id=task_id,
            )
            cached_state = self._pack_past_key_values(prefill_outputs.past_key_values)
            generated = prefill_tokens
            attention_mask = prefill_mask

        for step_idx in range(max_new_tokens):
            if step_idx == 0 and prefill_outputs is not None:
                outputs = prefill_outputs
            else:
                model_inputs = generated[:, -1:] if (cached_state is not None and use_cache) else generated
                past_key_values = self._restore_past_key_values(cached_state) if (cached_state is not None and use_cache) else None
                outputs = self(
                    input_ids=model_inputs,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    return_dict=True,
                    task_id=task_id,
                )
            next_token_logits = outputs.logits[:, -1, :].float()
            next_tokens = self._sample_next_token(
                logits=next_token_logits,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            if eos_ids is not None:
                if pad_token_id is None:
                    raise ValueError("pad_token_id is required when eos_token_id is set.")
                pad_tokens = torch.full_like(next_tokens, int(pad_token_id))
                next_tokens = torch.where(unfinished, next_tokens, pad_tokens)

            generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=-1)
            mask_step = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, mask_step], dim=-1)

            if use_cache:
                cached_state = self._pack_past_key_values(outputs.past_key_values)

            self._maybe_update_summary(generated)

            if eos_ids is not None:
                is_eos = torch.zeros_like(unfinished)
                for eos in eos_ids:
                    is_eos |= next_tokens.eq(eos)
                unfinished = unfinished & (~is_eos)
                if not unfinished.any():
                    break

        return generated

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: float = 1.0,
        use_cache: bool = True,
        eos_token_id: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
        pad_token_id: Optional[int] = None,
        task_id: int = 0,
        retrieved_chunk_ids: Optional[List[torch.LongTensor]] = None,
        **kwargs: Any,
    ):
        if input_ids is None:
            raise ValueError("input_ids must be provided for generation.")

        max_length = kwargs.pop("max_length", None)
        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise ValueError(f"Unsupported generation kwargs in NeuroplasticLlama.generate: {unknown}")
        if max_new_tokens is None and max_length is not None:
            max_new_tokens = max(int(max_length) - int(input_ids.shape[-1]), 0)

        return self._generate_with_kv_optimizations(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_cache=use_cache,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            task_id=task_id,
            retrieved_chunk_ids=retrieved_chunk_ids,
        )

    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)
        state_dict = {
            "sca_config": self.sca_config.to_dict(),
            "spatial_proj": self.spatial_proj.state_dict(),
            "block_centers": self.block_centers.detach().cpu(),
            "adapter_scale": self.adapter_scale.detach().cpu(),
            "task_embedding": self.task_embedding.state_dict(),
        }
        torch.save(state_dict, os.path.join(save_directory, "neuroplastic_llama_sca_v2.bin"))

        config = self.model.config.to_dict()
        config["architecture"] = "NeuroplasticLlamaSCAv2"
        config["num_tasks"] = self.num_tasks
        config["base_model_name"] = self.model_name
        config["sca_config"] = self.sca_config.to_dict()
        config["kv_cache"] = {
            "int4_quantization": self.kv_int4_quantization,
            "cpu_offload": self.kv_cpu_offload,
        }
        config["bounded_context"] = {
            "enabled": self.bounded_context_config.enabled,
            "sink_tokens": self.bounded_context_config.sink_tokens,
            "local_window_tokens": self.bounded_context_config.local_window_tokens,
            "global_window_tokens": self.bounded_context_config.global_window_tokens,
            "global_start_layer": self.bounded_context_config.global_start_layer,
            "global_group_id": self.bounded_context_config.global_group_id,
            "vram_budget_gib": self.bounded_context_config.vram_budget_gib,
            "summary_interval_min_tokens": self.bounded_context_config.summary_interval_min_tokens,
            "summary_interval_max_tokens": self.bounded_context_config.summary_interval_max_tokens,
            "retrieval_chunk_min_tokens": self.bounded_context_config.retrieval_chunk_min_tokens,
            "retrieval_chunk_max_tokens": self.bounded_context_config.retrieval_chunk_max_tokens,
            "retrieval_top_k_min": self.bounded_context_config.retrieval_top_k_min,
            "retrieval_top_k_max": self.bounded_context_config.retrieval_top_k_max,
        }
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            model_name = config.get("base_model_name", "unsloth/Meta-Llama-3.1-8B-bnb-4bit")
            num_tasks = int(config.get("num_tasks", 10))
            sca_cfg = config.get("sca_config", {})
            kv_cfg = config.get("kv_cache", {})
            bounded_cfg = config.get("bounded_context", {})
        else:
            model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
            num_tasks = 10
            sca_cfg = {}
            kv_cfg = {}
            bounded_cfg = {}

        kv_int4_arg = kwargs.pop("kv_int4_quantization", kv_cfg.get("int4_quantization", True))
        kv_cpu_arg = kwargs.pop("kv_cpu_offload", kv_cfg.get("cpu_offload", True))
        if kv_int4_arg is None:
            kv_int4_arg = kv_cfg.get("int4_quantization", True)
        if kv_cpu_arg is None:
            kv_cpu_arg = kv_cfg.get("cpu_offload", True)

        model = cls(
            model_name=model_name,
            num_tasks=num_tasks,
            neuroplasticity_enabled=kwargs.pop("neuroplasticity_enabled", True),
            sca_block_size=int(sca_cfg.get("block_size", 32)),
            sca_block_rank=int(sca_cfg.get("block_rank", 4)),
            sca_top_k=int(sca_cfg.get("top_k", 3)),
            sca_sigma=float(sca_cfg.get("sigma", 1.0)),
            sca_refractory_steps=int(sca_cfg.get("refractory_steps", 100)),
            sca_inhibition_lambda=float(sca_cfg.get("inhibition_lambda", 0.0)),
            sca_grid_size=int(sca_cfg.get("grid_size", 16)),
            sca_use_cuda=bool(sca_cfg.get("use_cuda", True)),
            sca_spmm_impl=str(sca_cfg.get("spmm_impl", "dense")),
            sca_soft_mask=bool(sca_cfg.get("soft_mask", True)),
            sca_grouped_row_gemm=bool(sca_cfg.get("grouped_row_gemm", False)),
            sca_grouped_row_min_bucket=int(sca_cfg.get("grouped_row_min_bucket", 2)),
            sca_grouped_row_allow_4bit_dequant=bool(sca_cfg.get("grouped_row_allow_4bit_dequant", False)),
            kv_int4_quantization=bool(kv_int4_arg),
            kv_cpu_offload=bool(kv_cpu_arg),
            bounded_context_enabled=bool(kwargs.pop("bounded_context_enabled", bounded_cfg.get("enabled", False))),
            bounded_sink_tokens=int(kwargs.pop("bounded_sink_tokens", bounded_cfg.get("sink_tokens", 8))),
            bounded_local_window_tokens=int(
                kwargs.pop("bounded_local_window_tokens", bounded_cfg.get("local_window_tokens", 10_000))
            ),
            bounded_global_window_tokens=int(
                kwargs.pop("bounded_global_window_tokens", bounded_cfg.get("global_window_tokens", 128_000))
            ),
            bounded_global_start_layer=int(
                kwargs.pop("bounded_global_start_layer", bounded_cfg.get("global_start_layer", 84))
            ),
            bounded_global_group_id=int(kwargs.pop("bounded_global_group_id", bounded_cfg.get("global_group_id", 0))),
            bounded_vram_budget_gib=float(kwargs.pop("bounded_vram_budget_gib", bounded_cfg.get("vram_budget_gib", 3.5))),
            summary_interval_min_tokens=int(
                kwargs.pop("summary_interval_min_tokens", bounded_cfg.get("summary_interval_min_tokens", 4_000))
            ),
            summary_interval_max_tokens=int(
                kwargs.pop("summary_interval_max_tokens", bounded_cfg.get("summary_interval_max_tokens", 8_000))
            ),
            retrieval_chunk_min_tokens=int(
                kwargs.pop("retrieval_chunk_min_tokens", bounded_cfg.get("retrieval_chunk_min_tokens", 300))
            ),
            retrieval_chunk_max_tokens=int(
                kwargs.pop("retrieval_chunk_max_tokens", bounded_cfg.get("retrieval_chunk_max_tokens", 800))
            ),
            retrieval_top_k_min=int(kwargs.pop("retrieval_top_k_min", bounded_cfg.get("retrieval_top_k_min", 8))),
            retrieval_top_k_max=int(kwargs.pop("retrieval_top_k_max", bounded_cfg.get("retrieval_top_k_max", 20))),
        )

        checkpoint_path = os.path.join(pretrained_model_name_or_path, "neuroplastic_llama_sca_v2.bin")
        if not os.path.exists(checkpoint_path):
            return model

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "spatial_proj" in state_dict:
            model.spatial_proj.load_state_dict(state_dict["spatial_proj"])
        if "block_centers" in state_dict:
            model.block_centers.copy_(state_dict["block_centers"].to(device=model.block_centers.device, dtype=torch.float32))
        if "adapter_scale" in state_dict:
            model.adapter_scale.data = state_dict["adapter_scale"].to(device=model.adapter_scale.device, dtype=model.adapter_scale.dtype)
        if "task_embedding" in state_dict:
            model.task_embedding.load_state_dict(state_dict["task_embedding"])
        return model
