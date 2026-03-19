from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import types
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes.functional as bnb_functional
from transformers import BitsAndBytesConfig, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
try:
    from transformers.cache_utils import DynamicCache
except Exception:  # pragma: no cover
    DynamicCache = None
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
    from .sca_sparse_adapter import compute_active_blocks
    from .sca_decoder_mirror import DecoderMirrorConfig, SparseDecoderMirrorSCA
    from .sca_sparse_config import SCASparseConfig, build_block_centers, build_inhibition_matrix
    from .sca_sparse_mlp import SparseLlamaMLP, SparseRouteSelection
    from .triton_sparse_mlp import materialize_linear_weight
    from .gqa_mamba_rank_collapse import (
        GQALayout,
        GQAMambaRankCollapseBlock,
        collapse_llama_gqa_attention_layer,
        transplant_svd_group_to_mamba_projection,
    )
    from .paged_sparse_attention import SparseAttentionConfig, SparseAttentionRuntime
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
    from sca_sparse_adapter import compute_active_blocks
    from sca_decoder_mirror import DecoderMirrorConfig, SparseDecoderMirrorSCA
    from sca_sparse_config import SCASparseConfig, build_block_centers, build_inhibition_matrix
    from sca_sparse_mlp import SparseLlamaMLP, SparseRouteSelection
    from triton_sparse_mlp import materialize_linear_weight
    from gqa_mamba_rank_collapse import (
        GQALayout,
        GQAMambaRankCollapseBlock,
        collapse_llama_gqa_attention_layer,
        transplant_svd_group_to_mamba_projection,
    )
    from paged_sparse_attention import SparseAttentionConfig, SparseAttentionRuntime


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


def _default_middle_layer_band(num_layers: int) -> list[int]:
    if num_layers <= 0:
        return []
    start = int(num_layers * 0.25)
    stop = int(num_layers * 0.75)
    if stop <= start:
        return list(range(num_layers))
    return list(range(start, stop))


def _normalize_layer_indices(layer_indices: Optional[List[int]], num_layers: int) -> list[int]:
    if layer_indices is None:
        return _default_middle_layer_band(num_layers)
    normalized = sorted({int(idx) for idx in layer_indices if 0 <= int(idx) < num_layers})
    return normalized


def _infer_module_device(module: nn.Module, fallback: torch.device) -> torch.device:
    for param in module.parameters(recurse=True):
        if torch.is_tensor(param):
            return param.device
    for buffer in module.buffers(recurse=True):
        if torch.is_tensor(buffer):
            return buffer.device
    return fallback


class _CollapsedMambaSelfAttention(nn.Module):
    def __init__(
        self,
        mamba_block: GQAMambaRankCollapseBlock,
        layer_idx: int,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.mamba_block = mamba_block
        self.layer_idx = int(layer_idx)
        self._dtype = dtype

    @classmethod
    def _materialize_proj_weight(cls, linear: nn.Module) -> torch.Tensor:
        weight = getattr(linear, "weight", None)
        qstate = getattr(weight, "quant_state", None)
        qshape = tuple(getattr(qstate, "shape", ())) if qstate is not None else ()
        qnumel = int(qshape[0] * qshape[1]) if len(qshape) == 2 else 0

        def _accept_if_valid(candidate: Any) -> Optional[torch.Tensor]:
            if not torch.is_tensor(candidate):
                return None
            dense = candidate.detach()
            if len(qshape) == 2:
                if tuple(dense.shape) == qshape:
                    return dense.float().cpu().contiguous()
                if dense.numel() == qnumel:
                    return dense.reshape(qshape).float().cpu().contiguous()
                if dense.ndim == 2 and tuple(dense.t().shape) == qshape:
                    return dense.t().float().cpu().contiguous()
                return None
            if dense.ndim == 2:
                return dense.float().cpu().contiguous()
            return None

        # Quantized path: require dequantization to resolve exact dense shape.
        if torch.is_tensor(weight) and qstate is not None:
            # Known-good path in this environment.
            try:
                for packed in (weight, weight.t() if weight.ndim == 2 else weight):
                    deq = _accept_if_valid(bnb_functional.dequantize_4bit(packed, quant_state=qstate))
                    if deq is not None:
                        return deq
            except Exception as e:
                print(f"bnb dequantize warning: {e}")

            # Fallback path for some bitsandbytes builds.
            if hasattr(weight, "dequantize"):
                try:
                    dense = _accept_if_valid(weight.dequantize())
                    if dense is not None:
                        return dense
                except Exception:
                    pass

            raise RuntimeError(
                f"Failed to dequantize quantized projection for {type(linear).__name__}; "
                f"packed_shape={tuple(weight.shape)} quant_shape={qshape}"
            )

        # Dense path.
        if torch.is_tensor(weight) and weight.dim() == 2 and weight.is_floating_point():
            return weight.detach().float().cpu().contiguous()

        if torch.is_tensor(weight):
            dev = weight.device
        else:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            dense = materialize_linear_weight(linear, device=dev, dtype=torch.float32)
            return dense.detach().float().cpu().contiguous()
        except Exception as exc:
            qstate = getattr(weight, "quant_state", None)
            qshape = getattr(qstate, "shape", None)
            raise RuntimeError(
                f"Failed to materialize projection weight for {type(linear).__name__}. "
                f"weight_shape={tuple(weight.shape) if torch.is_tensor(weight) else None}, "
                f"quant_shape={tuple(qshape) if qshape is not None else None}"
            ) from exc

    @classmethod
    def from_llama_attention(
        cls,
        source_attn: nn.Module,
        layer_idx: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        target_rank: Optional[int],
        variance_threshold: float,
        state_dim: Optional[int],
        dtype: torch.dtype,
    ) -> "_CollapsedMambaSelfAttention":
        layout = GQALayout(
            num_attention_heads=int(num_attention_heads),
            num_key_value_heads=int(num_key_value_heads),
            head_dim=int(head_dim),
        )
        collapsed = collapse_llama_gqa_attention_layer(
            w_q=cls._materialize_proj_weight(source_attn.q_proj),
            w_k=cls._materialize_proj_weight(source_attn.k_proj),
            w_v=cls._materialize_proj_weight(source_attn.v_proj),
            w_o=cls._materialize_proj_weight(source_attn.o_proj),
            layout=layout,
            target_rank=target_rank,
            variance_threshold=variance_threshold,
        )
        projections = [
            transplant_svd_group_to_mamba_projection(
                result=item,
                query_heads_per_group=layout.query_heads_per_group,
                head_dim=layout.head_dim,
                state_dim=state_dim,
            )
            for item in collapsed
        ]
        block = GQAMambaRankCollapseBlock(projections)
        return cls(mamba_block=block, layer_idx=layer_idx, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> Any:
        del attention_mask, position_ids, cache_position, position_embeddings, kwargs
        attn_output = self.mamba_block(hidden_states.to(dtype=self._dtype)).to(dtype=hidden_states.dtype)
        attn_weights = None
        if use_cache:
            return attn_output, attn_weights, past_key_value
        if output_attentions:
            return attn_output, attn_weights
        return attn_output, None


class HybridCollapsedMambaSelfAttention(nn.Module):
    def __init__(
        self,
        original_attn: nn.Module,
        mamba_block: GQAMambaRankCollapseBlock,
        layer_idx: int,
        mix_init: float = 0.05,
        dtype: torch.dtype = torch.float16,
        enable_output_bias: bool = False,
    ) -> None:
        super().__init__()
        self.original_attn = original_attn
        self.mamba_block = mamba_block
        self.layer_idx = int(layer_idx)
        self._dtype = dtype
        mix_value = float(max(min(mix_init, 1.0 - 1e-6), 1e-6))
        self.mix_logit = nn.Parameter(torch.logit(torch.tensor(mix_value, dtype=torch.float32)))
        self.output_gain = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.output_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.enable_output_bias = bool(enable_output_bias)
        self.capture_alignment = False
        self._last_alignment: Optional[Dict[str, torch.Tensor]] = None

    @property
    def mix_value(self) -> torch.Tensor:
        return torch.sigmoid(self.mix_logit)

    def set_mix_value(self, value: float) -> None:
        mix_value = float(max(min(value, 1.0 - 1e-6), 1e-6))
        with torch.no_grad():
            self.mix_logit.copy_(torch.logit(torch.tensor(mix_value, dtype=self.mix_logit.dtype, device=self.mix_logit.device)))

    @classmethod
    def from_llama_attention(
        cls,
        source_attn: nn.Module,
        layer_idx: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        target_rank: Optional[int],
        variance_threshold: float,
        state_dim: Optional[int],
        mix_init: float,
        dtype: torch.dtype,
    ) -> "HybridCollapsedMambaSelfAttention":
        collapsed = _CollapsedMambaSelfAttention.from_llama_attention(
            source_attn=source_attn,
            layer_idx=layer_idx,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            target_rank=target_rank,
            variance_threshold=variance_threshold,
            state_dim=state_dim,
            dtype=dtype,
        )
        return cls(
            original_attn=source_attn,
            mamba_block=collapsed.mamba_block,
            layer_idx=layer_idx,
            mix_init=mix_init,
            dtype=dtype,
        )

    def set_alignment_capture(self, enabled: bool) -> None:
        self.capture_alignment = bool(enabled)
        if not enabled:
            self._last_alignment = None

    def get_last_alignment(self) -> Optional[Dict[str, torch.Tensor]]:
        return self._last_alignment

    def export_hybrid_state(self) -> Dict[str, Any]:
        return {
            "mamba_block": {k: v.detach().cpu() for k, v in self.mamba_block.state_dict().items()},
            "mix_logit": self.mix_logit.detach().cpu(),
            "output_gain": self.output_gain.detach().cpu(),
            "output_bias": self.output_bias.detach().cpu(),
            "enable_output_bias": bool(self.enable_output_bias),
        }

    def load_hybrid_state(self, payload: Dict[str, Any], strict: bool = True) -> None:
        if "mamba_block" in payload:
            self.mamba_block.load_state_dict(payload["mamba_block"], strict=True)
            if "mix_logit" in payload:
                self.mix_logit.data.copy_(payload["mix_logit"].to(device=self.mix_logit.device, dtype=self.mix_logit.dtype))
            elif strict:
                raise RuntimeError("Missing mix_logit in hybrid attention payload")
            if "output_gain" in payload:
                self.output_gain.data.copy_(payload["output_gain"].to(device=self.output_gain.device, dtype=self.output_gain.dtype))
            elif strict:
                raise RuntimeError("Missing output_gain in hybrid attention payload")
            if "output_bias" in payload:
                self.output_bias.data.copy_(payload["output_bias"].to(device=self.output_bias.device, dtype=self.output_bias.dtype))
            self.enable_output_bias = bool(payload.get("enable_output_bias", self.enable_output_bias))
            return

        # Backward compatibility: accept old collapsed-Mamba state dicts.
        self.mamba_block.load_state_dict(payload, strict=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> Any:
        original_result = self.original_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if not isinstance(original_result, tuple):
            raise RuntimeError("Expected original attention to return a tuple")
        if use_cache and len(original_result) >= 3:
            original_out, attn_weights, present = original_result[:3]
        elif len(original_result) >= 2:
            original_out, attn_weights = original_result[:2]
            present = past_key_value
        else:
            raise RuntimeError("Unexpected original attention return shape")

        mamba_out = self.mamba_block(hidden_states.to(dtype=self._dtype)).to(dtype=hidden_states.dtype)
        mamba_out = mamba_out * self.output_gain.to(device=hidden_states.device, dtype=hidden_states.dtype)
        if self.enable_output_bias:
            mamba_out = mamba_out + self.output_bias.to(device=hidden_states.device, dtype=hidden_states.dtype)

        mix = self.mix_value.to(device=hidden_states.device, dtype=hidden_states.dtype).view(1, 1, 1)
        attn_output = ((1.0 - mix) * original_out) + (mix * mamba_out)

        if self.capture_alignment:
            self._last_alignment = {
                "original_out": original_out,
                "mamba_out": mamba_out,
                "mix": mix,
            }

        if use_cache:
            return attn_output, attn_weights, present
        if output_attentions:
            return attn_output, attn_weights
        return attn_output, None


class NeuroplasticLlama(nn.Module):
    def __init__(
        self,
        model_name: str = "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        num_tasks: int = 10,
        neuroplasticity_enabled: bool = True,
        sca_block_size: int = 32,
        sca_block_rank: int = 4,
        sca_basis_rank: int = 32,
        sca_basis_top_k: int = 8,
        sca_top_k: int = 3,
        sca_routing_mode: str = "spatial_grid",
        sca_semantic_block_score_normalized: bool = False,
        sca_adaptive_top_k: bool = False,
        sca_adaptive_top_k_min: Optional[int] = None,
        sca_adaptive_top_k_max: Optional[int] = None,
        sca_adaptive_top_k_min_score_ratio: float = 0.15,
        sca_sigma: float = 1.0,
        sca_refractory_steps: int = 100,
        sca_inhibition_lambda: float = 0.0,
        sca_grid_size: int = 16,
        sca_use_cuda: bool = True,
        sca_spmm_impl: str = "dense",
        sca_sparse_placement: str = "input_mask",
        sca_soft_mask: bool = True,
        sca_grouped_row_gemm: bool = False,
        sca_grouped_row_min_bucket: int = 2,
        sca_grouped_row_allow_4bit_dequant: bool = False,
        sca_stability_dense_fallback_threshold: float = 0.0,
        sca_bottom_buffer_layers: int = 2,
        sca_decode_guard_layers: int = 12,
        sca_basis_rank_by_layer: Optional[Dict[int, int]] = None,
        sca_basis_top_k_by_layer: Optional[Dict[int, int]] = None,
        sca_top_k_by_layer: Optional[Dict[int, int]] = None,
        strict_decode_repetition_penalty: float = 1.35,
        strict_decode_penalty_window: int = 64,
        strict_decode_enable_repetition_penalty: bool = False,
        strict_decode_upper_layer_cap_enabled: bool = True,
        strict_runtime_allow_noncuda_spmm_diagnostic: bool = False,
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
        attention_sparse_mode: bool = False,
        attention_local_window_tokens: int = 2048,
        attention_sink_tokens: int = 8,
        attention_page_size_tokens: int = 256,
        attention_retrieval_top_k_pages: int = 8,
        attention_retrieval_head_group_ids: Optional[List[int]] = None,
        attention_retrieval_start_layer: Optional[int] = None,
        attention_archive_cpu_dtype: str = "int4",
        attention_hot_archive_gpu_pages: int = 0,
        attention_disable_ssd_fetch_in_decode: bool = True,
        attention_force_single_model_runtime: bool = True,
        attention_gqa_mamba_enabled: bool = False,
        attention_gqa_target_rank: Optional[int] = None,
        attention_gqa_variance_threshold: float = 0.90,
        attention_gqa_state_dim: Optional[int] = None,
        attention_gqa_max_layers: Optional[int] = None,
        attention_gqa_verbose: bool = False,
        attention_hybrid_enabled: bool = False,
        attention_hybrid_layers: Optional[List[int]] = None,
        attention_hybrid_mix_init: float = 0.05,
        attention_hybrid_target_rank: Optional[int] = 16,
        attention_hybrid_variance_threshold: float = 0.90,
        attention_hybrid_state_dim: Optional[int] = None,
        attention_hybrid_force_no_cache: bool = True,
        decoder_mirror_enabled: bool = False,
        decoder_mirror_top_k: Optional[int] = None,
        decoder_mirror_rank: int = 4,
        decoder_mirror_route_conditioned: bool = True,
        decoder_mirror_source_layers: Optional[List[int]] = None,
        decoder_mirror_route_prior_scale_init: float = 0.25,
        decoder_mirror_residual_scale_init: float = 0.0,
        **_: object,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_tasks = int(num_tasks)
        self.neuroplasticity_enabled = bool(neuroplasticity_enabled)
        self.buffer_layers = 2
        self.bottom_buffer_layers = max(int(sca_bottom_buffer_layers), 0)
        self.decode_guard_layers = max(int(sca_decode_guard_layers), 0)
        self.strict_decode_repetition_penalty = float(max(strict_decode_repetition_penalty, 1.0))
        self.strict_decode_penalty_window = int(max(strict_decode_penalty_window, 0))
        self.strict_decode_enable_repetition_penalty = bool(strict_decode_enable_repetition_penalty)
        self.strict_decode_upper_layer_cap_enabled = bool(strict_decode_upper_layer_cap_enabled)
        self.strict_runtime_allow_noncuda_spmm_diagnostic = bool(strict_runtime_allow_noncuda_spmm_diagnostic)
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
        retrieval_groups = attention_retrieval_head_group_ids if attention_retrieval_head_group_ids is not None else [0]
        self.sparse_attention_config = SparseAttentionConfig(
            enabled=bool(attention_sparse_mode),
            local_window_tokens=int(attention_local_window_tokens),
            sink_tokens=int(attention_sink_tokens),
            page_size_tokens=int(attention_page_size_tokens),
            retrieval_top_k_pages=int(attention_retrieval_top_k_pages),
            retrieval_head_group_ids=tuple(int(x) for x in retrieval_groups),
            retrieval_start_layer=None if attention_retrieval_start_layer is None else int(attention_retrieval_start_layer),
            archive_cpu_dtype=str(attention_archive_cpu_dtype),
            hot_archive_gpu_pages=int(attention_hot_archive_gpu_pages),
            disable_ssd_fetch_in_decode=bool(attention_disable_ssd_fetch_in_decode),
            force_single_model_runtime=bool(attention_force_single_model_runtime),
            strict_fully_sparse=False,
        )
        self.sparse_attention_config.validate()
        self._sparse_attention_runtime: Optional[SparseAttentionRuntime] = None
        self._last_sparse_attention_step: Dict[str, Any] = {}

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
        self._sparse_attention_runtime = SparseAttentionRuntime(
            config=self.sparse_attention_config,
            num_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=int(getattr(self.config, "head_dim", self.hidden_size // max(num_attention_heads, 1))),
        )
        self.sca_config = SCASparseConfig(
            hidden_size=self.hidden_size,
            block_size=sca_block_size,
            block_rank=sca_block_rank,
            basis_rank=sca_basis_rank,
            basis_top_k=sca_basis_top_k,
            top_k=sca_top_k,
            routing_mode=str(sca_routing_mode),
            semantic_block_score_normalized=bool(sca_semantic_block_score_normalized),
            basis_rank_by_layer=dict(sca_basis_rank_by_layer or {}),
            basis_top_k_by_layer=dict(sca_basis_top_k_by_layer or {}),
            top_k_by_layer=dict(sca_top_k_by_layer or {}),
            adaptive_top_k=bool(sca_adaptive_top_k),
            adaptive_top_k_min=int(sca_adaptive_top_k_min if sca_adaptive_top_k_min is not None else sca_top_k),
            adaptive_top_k_max=int(sca_adaptive_top_k_max if sca_adaptive_top_k_max is not None else sca_top_k),
            adaptive_top_k_min_score_ratio=float(sca_adaptive_top_k_min_score_ratio),
            sigma=sca_sigma,
            refractory_steps=sca_refractory_steps,
            inhibition_lambda=sca_inhibition_lambda,
            use_cuda=sca_use_cuda,
            grid_size=sca_grid_size,
            spmm_impl=sca_spmm_impl,
            sparse_placement=str(sca_sparse_placement),
            soft_mask=sca_soft_mask,
            grouped_row_gemm=sca_grouped_row_gemm,
            grouped_row_min_bucket=sca_grouped_row_min_bucket,
            grouped_row_allow_4bit_dequant=sca_grouped_row_allow_4bit_dequant,
            stability_dense_fallback_threshold=float(sca_stability_dense_fallback_threshold),
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
        num_layers = len(self.model.model.layers)
        self.sca_layer_output_scale = nn.Parameter(torch.ones((num_layers,), dtype=torch.float32), requires_grad=False)
        # Mildly dampen upper decode-critical layers by default to reduce lexical attractor collapse.
        decode_band_start = max(int(num_layers * 0.75), 0)
        if decode_band_start < num_layers:
            with torch.no_grad():
                self.sca_layer_output_scale[decode_band_start:] = 0.9

        if self.sca_config.use_cuda and not torch.cuda.is_available():
            raise RuntimeError("sca_use_cuda=True requires CUDA runtime")
        # Runtime is extension-free by default; routing uses Triton gate when available,
        # otherwise torch fallback, while sparse SpMM stays on CUDA path.
        self.sca_cuda_kernels = None

        num_layers = len(self.model.model.layers)
        self.register_buffer(
            "refractory_until",
            torch.zeros((num_layers, self.sca_config.num_blocks), dtype=torch.int32),
            persistent=False,
        )
        self._routing_decode_mode = False
        self._routing_step = 0

        self._replace_mlp_with_sparse_wrappers()
        self.attention_hybrid_enabled = bool(attention_hybrid_enabled)
        self.attention_hybrid_layers = _normalize_layer_indices(attention_hybrid_layers, num_layers)
        self.attention_hybrid_force_no_cache = bool(attention_hybrid_force_no_cache)
        self.attention_hybrid_mix_init = float(attention_hybrid_mix_init)
        self.attention_hybrid_target_rank = attention_hybrid_target_rank
        self.attention_hybrid_variance_threshold = float(attention_hybrid_variance_threshold)
        self.attention_hybrid_state_dim = attention_hybrid_state_dim
        self._sparse_mlp_bank_manifest_path: Optional[str] = None
        self.attention_gqa_mamba_enabled = bool(attention_gqa_mamba_enabled)
        if self.attention_hybrid_enabled:
            self._install_bounded_context_attention_hooks()
            self._replace_attention_with_hybrid_gqa_mamba(
                target_rank=self.attention_hybrid_target_rank,
                variance_threshold=self.attention_hybrid_variance_threshold,
                state_dim=self.attention_hybrid_state_dim,
                mix_init=self.attention_hybrid_mix_init,
                layer_indices=self.attention_hybrid_layers,
                verbose=attention_gqa_verbose,
            )
        elif self.attention_gqa_mamba_enabled:
            self._replace_attention_with_gqa_mamba(
                target_rank=attention_gqa_target_rank,
                variance_threshold=float(attention_gqa_variance_threshold),
                state_dim=attention_gqa_state_dim,
                max_layers=attention_gqa_max_layers,
                verbose=attention_gqa_verbose,
            )
        else:
            self._install_bounded_context_attention_hooks()

        self._current_task_id = 0
        self._cached_task_emb: Optional[torch.Tensor] = None
        self.disable_task_bias_injection: bool = False
        self._sca_recalibration_layer_indices: List[int] = []
        self._sca_recalibration_active_sparse_layer_indices: List[int] = []
        self._sca_recalibration_mode: str = "local_mlp_geometry"
        self._sca_recalibration_trainable_modules: List[str] = []
        self._sca_recalibration_hybrid_checkpoint_path: str = ""
        self._sca_disable_task_bias_injection_from_artifact: Optional[bool] = None
        self._sca_sparse_layer_override: Optional[set[int]] = None
        self.decoder_mirror_config: Optional[DecoderMirrorConfig] = None
        self.decoder_mirror: Optional[SparseDecoderMirrorSCA] = None
        self._decoder_mirror_enabled: bool = bool(decoder_mirror_enabled)
        self._last_decoder_mirror_diagnostics: Dict[str, Any] = {}
        self._decoder_mirror_calibration_mode: str = "decoder_co_warp"
        self._decoder_mirror_trainable_modules: List[str] = []
        self._decoder_mirror_hybrid_checkpoint_path: str = ""
        self._decoder_mirror_sca_checkpoint_path: str = ""
        self._decoder_mirror_source_layers: List[int] = []
        self._decoder_mirror_route_prior_missing: bool = True
        self._decoder_mirror_source_layers_used: List[int] = []
        self._decoder_mirror_init_kwargs: Dict[str, Any] = {
            "top_k": None if decoder_mirror_top_k is None else int(decoder_mirror_top_k),
            "rank": int(decoder_mirror_rank),
            "route_conditioned": bool(decoder_mirror_route_conditioned),
            "source_layers": None if decoder_mirror_source_layers is None else [int(v) for v in decoder_mirror_source_layers],
            "route_prior_scale_init": float(decoder_mirror_route_prior_scale_init),
            "residual_scale_init": float(decoder_mirror_residual_scale_init),
        }

        # Keep trainable parts on model device.
        dev = self.device
        self.task_embedding.to(device=dev, dtype=torch.float16)
        self.spatial_proj.to(device=dev, dtype=torch.float16)
        self.score.to(device=dev, dtype=torch.float16)
        self.adapter_scale.data = self.adapter_scale.data.to(device=dev, dtype=torch.float16)
        if bool(decoder_mirror_enabled):
            self._ensure_decoder_mirror_module()
            self.set_decoder_mirror_enabled(True)

    def _layer_output_scale(self, layer_idx: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not hasattr(self, "sca_layer_output_scale"):
            return torch.tensor(1.0, device=device, dtype=dtype)
        idx = int(max(min(layer_idx, int(self.sca_layer_output_scale.shape[0]) - 1), 0))
        scale = self.sca_layer_output_scale[idx].to(device=device, dtype=dtype)
        scale = scale.clamp(min=0.0, max=1.5)
        # In strict sparse decode, cap upper-layer SCA contribution to reduce repetitive lexical attractors.
        if bool(self.strict_decode_upper_layer_cap_enabled) and bool(getattr(self, "sparse_attention_config", None) is not None) and bool(
            getattr(self.sparse_attention_config, "strict_fully_sparse", False)
        ):
            decode_band_start = max(int(self.sca_layer_output_scale.shape[0] * 0.75), 0)
            if idx >= decode_band_start:
                cap = torch.tensor(0.7, device=device, dtype=dtype)
                scale = torch.minimum(scale, cap)
        return scale

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

    def set_sparse_attention_mode(
        self,
        enabled: Optional[bool] = None,
        local_window_tokens: Optional[int] = None,
        sink_tokens: Optional[int] = None,
        page_size_tokens: Optional[int] = None,
        retrieval_top_k_pages: Optional[int] = None,
        retrieval_head_group_ids: Optional[List[int]] = None,
        retrieval_start_layer: Optional[int] = None,
        archive_cpu_dtype: Optional[str] = None,
        hot_archive_gpu_pages: Optional[int] = None,
        disable_ssd_fetch_in_decode: Optional[bool] = None,
        force_single_model_runtime: Optional[bool] = None,
        strict_fully_sparse: Optional[bool] = None,
    ) -> None:
        if enabled is not None:
            self.sparse_attention_config.enabled = bool(enabled)
        if local_window_tokens is not None:
            self.sparse_attention_config.local_window_tokens = int(local_window_tokens)
        if sink_tokens is not None:
            self.sparse_attention_config.sink_tokens = int(sink_tokens)
        if page_size_tokens is not None:
            self.sparse_attention_config.page_size_tokens = int(page_size_tokens)
        if retrieval_top_k_pages is not None:
            self.sparse_attention_config.retrieval_top_k_pages = int(retrieval_top_k_pages)
        if retrieval_head_group_ids is not None:
            self.sparse_attention_config.retrieval_head_group_ids = tuple(int(x) for x in retrieval_head_group_ids)
        if retrieval_start_layer is not None:
            self.sparse_attention_config.retrieval_start_layer = int(retrieval_start_layer)
        if archive_cpu_dtype is not None:
            self.sparse_attention_config.archive_cpu_dtype = str(archive_cpu_dtype)
        if hot_archive_gpu_pages is not None:
            self.sparse_attention_config.hot_archive_gpu_pages = int(hot_archive_gpu_pages)
        if disable_ssd_fetch_in_decode is not None:
            self.sparse_attention_config.disable_ssd_fetch_in_decode = bool(disable_ssd_fetch_in_decode)
        if force_single_model_runtime is not None:
            self.sparse_attention_config.force_single_model_runtime = bool(force_single_model_runtime)
        if strict_fully_sparse is not None:
            self.sparse_attention_config.strict_fully_sparse = bool(strict_fully_sparse)
        self.sparse_attention_config.validate()
        if self._sparse_attention_runtime is not None:
            self._sparse_attention_runtime.config = self.sparse_attention_config

    def get_sparse_attention_diagnostics(self) -> Dict[str, Any]:
        if self._sparse_attention_runtime is None:
            return {
                "enabled": False,
                "steps": 0,
                "last_step": {},
            }
        out = self._sparse_attention_runtime.diagnostics()
        if self._last_sparse_attention_step:
            out["last_step"] = dict(self._last_sparse_attention_step)
        return out

    def reset_sparse_attention_state(self) -> None:
        if self._sparse_attention_runtime is not None:
            self._sparse_attention_runtime.reset()
        self._last_sparse_attention_step = {}

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
        return bool(self.kv_int4_quantization or self.kv_cpu_offload or self.sparse_attention_config.enabled)

    def _should_disable_generation_cache(self, requested_use_cache: bool) -> bool:
        if not requested_use_cache:
            return True
        # Sparse-attention decode requires cache and provides its own bounded behavior.
        if bool(getattr(self, "sparse_attention_config", None) is not None) and bool(
            getattr(self.sparse_attention_config, "enabled", False)
        ):
            return False
        if bool(getattr(self, "attention_hybrid_enabled", False)) and bool(getattr(self, "attention_hybrid_force_no_cache", True)):
            return True
        if bool(getattr(self, "attention_gqa_mamba_enabled", False)):
            return True
        return False

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

    def _replace_attention_with_gqa_mamba(
        self,
        target_rank: Optional[int] = None,
        variance_threshold: float = 0.90,
        state_dim: Optional[int] = None,
        max_layers: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        num_attention_heads = int(getattr(self.config, "num_attention_heads", 1))
        num_key_value_heads = int(getattr(self.config, "num_key_value_heads", num_attention_heads))
        head_dim = int(getattr(self.config, "head_dim", self.hidden_size // max(num_attention_heads, 1)))
        max_layers_int = int(max_layers) if max_layers is not None else None
        total_layers = len(self.model.model.layers)

        for layer_idx, layer in enumerate(self.model.model.layers):
            if max_layers_int is not None and layer_idx >= max_layers_int:
                if verbose:
                    print(f"[gqa-mamba] keeping original self_attn for layer {layer_idx}/{total_layers-1}", flush=True)
                continue
            if verbose:
                print(f"[gqa-mamba] replacing self_attn layer {layer_idx}/{total_layers-1}", flush=True)
            source_device = _infer_module_device(layer.self_attn, self.device)
            replacement = _CollapsedMambaSelfAttention.from_llama_attention(
                source_attn=layer.self_attn,
                layer_idx=layer_idx,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                target_rank=target_rank,
                variance_threshold=variance_threshold,
                state_dim=state_dim,
                dtype=torch.float16,
            ).to(device=source_device, dtype=torch.float16)
            layer.self_attn = replacement

    def _replace_attention_with_hybrid_gqa_mamba(
        self,
        target_rank: Optional[int] = 16,
        variance_threshold: float = 0.90,
        state_dim: Optional[int] = None,
        mix_init: float = 0.05,
        layer_indices: Optional[List[int]] = None,
        verbose: bool = False,
    ) -> None:
        num_attention_heads = int(getattr(self.config, "num_attention_heads", 1))
        num_key_value_heads = int(getattr(self.config, "num_key_value_heads", num_attention_heads))
        head_dim = int(getattr(self.config, "head_dim", self.hidden_size // max(num_attention_heads, 1)))
        selected = set(_normalize_layer_indices(layer_indices, len(self.model.model.layers)))
        total_layers = len(self.model.model.layers)

        for layer_idx, layer in enumerate(self.model.model.layers):
            if layer_idx not in selected:
                if verbose:
                    print(f"[hybrid-gqa-mamba] keeping original self_attn for layer {layer_idx}/{total_layers-1}", flush=True)
                continue
            if verbose:
                print(f"[hybrid-gqa-mamba] wrapping self_attn layer {layer_idx}/{total_layers-1}", flush=True)
            original_attn = layer.self_attn
            source_device = _infer_module_device(original_attn, self.device)
            replacement = HybridCollapsedMambaSelfAttention.from_llama_attention(
                source_attn=original_attn,
                layer_idx=layer_idx,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                target_rank=target_rank,
                variance_threshold=variance_threshold,
                state_dim=state_dim,
                mix_init=mix_init,
                dtype=torch.float16,
            ).to(device=source_device)
            replacement.mamba_block.to(device=source_device, dtype=torch.float16)
            layer.self_attn = replacement

    def iter_hybrid_attention_modules(self) -> List[Tuple[int, HybridCollapsedMambaSelfAttention]]:
        modules: List[Tuple[int, HybridCollapsedMambaSelfAttention]] = []
        for layer_idx, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn
            if isinstance(attn, HybridCollapsedMambaSelfAttention):
                modules.append((layer_idx, attn))
        return modules

    def get_effective_hybrid_layers(self) -> List[int]:
        return [layer_idx for layer_idx, _attn in self.iter_hybrid_attention_modules()]

    def set_hybrid_alignment_capture(self, enabled: bool) -> None:
        for _layer_idx, attn in self.iter_hybrid_attention_modules():
            attn.set_alignment_capture(enabled)

    def set_mlp_alignment_capture(self, enabled: bool, layer_indices: Optional[List[int]] = None) -> None:
        selected = set(layer_indices) if layer_indices is not None else None
        for layer_idx, wrapper in enumerate(self.sca_sparse_mlps):
            wrapper.set_alignment_capture(bool(enabled) and (selected is None or layer_idx in selected))

    def _default_decoder_mirror_source_layers(self) -> List[int]:
        total_layers = len(getattr(self, "sca_sparse_mlps", []))
        if total_layers <= 0:
            return []
        usable_limit = max(total_layers - int(getattr(self, "buffer_layers", 0)), 0)
        candidates = list(range(usable_limit)) if usable_limit > 0 else list(range(total_layers))
        if not candidates:
            candidates = list(range(total_layers))
        return candidates[-4:]

    def _normalize_decoder_mirror_source_layers(self, layer_indices: Optional[List[int]]) -> List[int]:
        total_layers = len(getattr(self, "sca_sparse_mlps", []))
        selected = self._default_decoder_mirror_source_layers() if layer_indices is None else sorted({int(v) for v in layer_indices})
        invalid = [idx for idx in selected if idx < 0 or idx >= total_layers]
        if invalid:
            raise RuntimeError(f"Invalid decoder mirror source layers: {invalid}")
        return selected

    def _ensure_decoder_mirror_module(
        self,
        *,
        source_layers: Optional[List[int]] = None,
        top_k: Optional[int] = None,
        rank: Optional[int] = None,
        route_conditioned: Optional[bool] = None,
        route_prior_scale_init: Optional[float] = None,
        residual_scale_init: Optional[float] = None,
    ) -> SparseDecoderMirrorSCA:
        init_kwargs = dict(getattr(self, "_decoder_mirror_init_kwargs", {}))
        resolved_source_layers = self._normalize_decoder_mirror_source_layers(
            source_layers if source_layers is not None else init_kwargs.get("source_layers")
        )
        resolved_top_k = int(top_k if top_k is not None else init_kwargs.get("top_k") or self.sca_config.top_k)
        resolved_rank = int(rank if rank is not None else init_kwargs.get("rank") or 4)
        resolved_route_conditioned = bool(
            route_conditioned if route_conditioned is not None else init_kwargs.get("route_conditioned", True)
        )
        resolved_route_prior_scale_init = float(
            route_prior_scale_init
            if route_prior_scale_init is not None
            else init_kwargs.get("route_prior_scale_init", 0.25)
        )
        resolved_residual_scale_init = float(
            residual_scale_init
            if residual_scale_init is not None
            else init_kwargs.get("residual_scale_init", 0.0)
        )

        if int(self.sca_config.num_blocks) != int(self.hidden_size // self.sca_config.block_size):
            raise RuntimeError("Main SCA config has inconsistent block sizing")

        config = DecoderMirrorConfig(
            hidden_size=int(self.hidden_size),
            block_size=int(self.sca_config.block_size),
            num_blocks=int(self.sca_config.num_blocks),
            top_k=resolved_top_k,
            rank=resolved_rank,
            grid_size=int(self.sca_config.grid_size),
            sigma=float(self.sca_config.sigma),
            route_prior_scale_init=resolved_route_prior_scale_init,
            residual_scale_init=resolved_residual_scale_init,
            source_layer_indices=resolved_source_layers,
            enabled=True,
            route_conditioned=resolved_route_conditioned,
        )

        replace_module = True
        if self.decoder_mirror is not None and self.decoder_mirror_config is not None:
            current = self.decoder_mirror_config
            replace_module = any(
                [
                    int(current.hidden_size) != int(config.hidden_size),
                    int(current.block_size) != int(config.block_size),
                    int(current.num_blocks) != int(config.num_blocks),
                    int(current.top_k) != int(config.top_k),
                    int(current.rank) != int(config.rank),
                ]
            )

        if replace_module:
            self.decoder_mirror = SparseDecoderMirrorSCA(config).to(device=self.device, dtype=torch.float16)
        self.decoder_mirror_config = config
        self._decoder_mirror_source_layers = list(resolved_source_layers)
        self._decoder_mirror_init_kwargs = {
            "top_k": int(config.top_k),
            "rank": int(config.rank),
            "route_conditioned": bool(config.route_conditioned),
            "source_layers": list(config.source_layer_indices),
            "route_prior_scale_init": float(config.route_prior_scale_init),
            "residual_scale_init": float(config.residual_scale_init),
        }
        return self.decoder_mirror

    def set_decoder_mirror_enabled(self, enabled: bool) -> None:
        if bool(enabled):
            self._ensure_decoder_mirror_module()
        self._decoder_mirror_enabled = bool(enabled)
        if self.decoder_mirror is not None:
            self.decoder_mirror.config.enabled = bool(enabled)
        layer_indices = None if self.decoder_mirror_config is None else self.decoder_mirror_config.source_layer_indices
        self.set_decoder_mirror_route_capture(bool(enabled), layer_indices=layer_indices)

    def set_decoder_mirror_route_capture(self, enabled: bool, layer_indices: Optional[List[int]] = None) -> None:
        selected = set(layer_indices) if layer_indices is not None else None
        for layer_idx, wrapper in enumerate(getattr(self, "sca_sparse_mlps", [])):
            wrapper.set_route_capture(bool(enabled) and (selected is None or layer_idx in selected))

    def _build_decoder_mirror_route_prior(
        self,
        source_layers: Optional[List[int]] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        selected = self._normalize_decoder_mirror_source_layers(source_layers)
        inferred_batch = int(batch_size) if batch_size is not None else None
        inferred_seq = int(seq_len) if seq_len is not None else None
        rows: Optional[int] = None
        route_prior: Optional[torch.Tensor] = None
        used_layers: List[int] = []

        target_device = device if device is not None else self.device
        for layer_idx in selected:
            wrapper = self.sca_sparse_mlps[layer_idx]
            snapshot = wrapper.get_last_route_snapshot()
            if snapshot is None:
                continue
            layer_rows = int(snapshot["rows"].item()) if torch.is_tensor(snapshot.get("rows")) else int(snapshot["rows"])
            layer_batch = (
                int(snapshot["batch_size"].item()) if torch.is_tensor(snapshot.get("batch_size")) else int(snapshot["batch_size"])
            )
            layer_seq = int(snapshot["seq_len"].item()) if torch.is_tensor(snapshot.get("seq_len")) else int(snapshot["seq_len"])
            if rows is None:
                rows = layer_rows
                inferred_batch = layer_batch
                inferred_seq = layer_seq
                route_prior = torch.zeros((rows, self.sca_config.num_blocks), device=target_device, dtype=torch.float32)
            elif rows != layer_rows:
                continue
            active_idx = snapshot["active_idx"].to(device=target_device, dtype=torch.long)
            score_weights = snapshot.get("score_weights")
            if score_weights is None:
                valid = active_idx >= 0
                score_weights = torch.zeros_like(active_idx, dtype=torch.float32, device=target_device)
                if torch.any(valid):
                    counts = valid.sum(dim=-1, keepdim=True).clamp_min(1)
                    score_weights = torch.where(
                        valid,
                        1.0 / counts.to(dtype=torch.float32),
                        torch.zeros_like(score_weights),
                    )
            else:
                score_weights = score_weights.to(device=target_device, dtype=torch.float32)
            row_index = torch.arange(active_idx.shape[0], device=target_device, dtype=torch.long)
            for slot in range(active_idx.shape[1]):
                block_idx = active_idx[:, slot]
                valid = block_idx >= 0
                if not torch.any(valid):
                    continue
                route_prior[row_index[valid], block_idx[valid]] += score_weights[valid, slot]
            used_layers.append(int(layer_idx))

        if rows is None:
            rows = int((inferred_batch or 0) * (inferred_seq or 0))
            route_prior = torch.zeros((rows, self.sca_config.num_blocks), device=target_device, dtype=torch.float32)

        if route_prior.numel() > 0:
            route_prior = route_prior / route_prior.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        self._decoder_mirror_route_prior_missing = len(used_layers) == 0
        self._decoder_mirror_source_layers_used = used_layers
        self._decoder_mirror_source_layers = list(selected)
        final_batch = int(inferred_batch or 0)
        final_seq = int(inferred_seq or 0)
        if final_batch * final_seq != rows:
            final_batch = max(rows, 0)
            final_seq = 1 if rows > 0 else 0
        return route_prior.view(final_batch, final_seq, self.sca_config.num_blocks)

    def get_decoder_mirror_diagnostics(self) -> Dict[str, Any]:
        out = dict(getattr(self, "_last_decoder_mirror_diagnostics", {}))
        if not out and self.decoder_mirror is not None:
            out = self.decoder_mirror.get_last_diagnostics()
        out.setdefault("enabled", bool(getattr(self, "_decoder_mirror_enabled", False)))
        out.setdefault("route_prior_missing", bool(getattr(self, "_decoder_mirror_route_prior_missing", True)))
        out.setdefault("source_layers_used", list(getattr(self, "_decoder_mirror_source_layers_used", [])))
        out.setdefault("configured_source_layers", list(getattr(self, "_decoder_mirror_source_layers", [])))
        return out

    def prepare_sca_local_recalibration(
        self,
        include_task_embedding: bool = False,
        include_spatial_proj: bool = True,
        include_adapter_scale: bool = False,
        include_layer_output_scale: bool = True,
        layer_indices: Optional[List[int]] = None,
        active_sparse_layer_indices: Optional[List[int]] = None,
        recalibration_mode: str = "local_mlp_geometry",
        hybrid_checkpoint_path: str = "",
    ) -> List[nn.Parameter]:
        for param in self.parameters():
            param.requires_grad = False

        decode_manifold_mode = str(recalibration_mode) == "decode_manifold_alignment"
        total_layers = len(getattr(self, "sca_sparse_mlps", []))
        self._sca_recalibration_layer_indices = (
            sorted(set(int(v) for v in layer_indices if 0 <= int(v) < total_layers))
            if layer_indices is not None
            else list(range(total_layers))
        )
        if active_sparse_layer_indices is not None:
            active_sparse = sorted(set(int(v) for v in active_sparse_layer_indices if 0 <= int(v) < total_layers))
        elif layer_indices is not None:
            active_sparse = list(self._sca_recalibration_layer_indices)
        else:
            active_sparse = []
        self._sca_recalibration_active_sparse_layer_indices = list(active_sparse)
        self._sca_recalibration_mode = str(recalibration_mode)
        self._sca_recalibration_hybrid_checkpoint_path = str(hybrid_checkpoint_path or "")
        self.set_sparse_layer_override(active_sparse if active_sparse else None)
        self.set_mlp_alignment_capture(True, self._sca_recalibration_layer_indices)
        if decode_manifold_mode:
            include_spatial_proj = False
            include_task_embedding = False
            self._decoder_mirror_enabled = False

        trainable: List[nn.Parameter] = []
        trainable_names: List[str] = []
        if include_spatial_proj:
            self.spatial_proj.to(device=self.device, dtype=torch.float32)
            for name, param in self.spatial_proj.named_parameters():
                param.requires_grad = True
                trainable.append(param)
                trainable_names.append(f"spatial_proj.{name}")
        if include_task_embedding:
            self.task_embedding.to(device=self.device, dtype=torch.float32)
            for name, param in self.task_embedding.named_parameters():
                param.requires_grad = True
                trainable.append(param)
                trainable_names.append(f"task_embedding.{name}")
        if include_adapter_scale and hasattr(self, "adapter_scale"):
            self.adapter_scale.data = self.adapter_scale.data.to(device=self.adapter_scale.device, dtype=torch.float32)
            self.adapter_scale.requires_grad = True
            trainable.append(self.adapter_scale)
            trainable_names.append("adapter_scale")
        if include_layer_output_scale and hasattr(self, "sca_layer_output_scale"):
            self.sca_layer_output_scale.data = self.sca_layer_output_scale.data.to(
                device=self.sca_layer_output_scale.device,
                dtype=torch.float32,
            )
            self.sca_layer_output_scale.requires_grad = True
            trainable.append(self.sca_layer_output_scale)
            trainable_names.append("sca_layer_output_scale")
        if str(getattr(self.sca_config, "sparse_placement", "input_mask")) == "learned_basis":
            selected_layers = set(self._sca_recalibration_layer_indices)
            for layer_idx, wrapper in enumerate(self.sca_sparse_mlps):
                if selected_layers and layer_idx not in selected_layers:
                    continue
                for name, param in wrapper.iter_sparse_basis_parameters():
                    param.data = param.data.to(device=param.device, dtype=torch.float32)
                    param.requires_grad = True
                    trainable.append(param)
                    trainable_names.append(f"sca_sparse_mlps.{layer_idx}.{name}")

        self._sca_recalibration_trainable_modules = trainable_names
        return trainable

    def set_sparse_layer_override(self, layer_indices: Optional[List[int]]) -> None:
        if layer_indices is None:
            self._sca_sparse_layer_override = None
            return
        total_layers = len(getattr(self, "sca_sparse_mlps", []))
        self._sca_sparse_layer_override = {
            int(v) for v in layer_indices if 0 <= int(v) < total_layers
        }

    def compute_sca_local_recalibration_loss(
        self,
        loss_mode: str = "mse_plus_norm",
        norm_weight: float = 0.01,
        layer_weight_map: Optional[Dict[int, float]] = None,
        delta_norm_cap: float = 0.25,
        delta_norm_cap_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        allowed_modes = {"mse", "cosine", "mse_plus_norm"}
        if loss_mode not in allowed_modes:
            raise ValueError(f"Unsupported loss_mode '{loss_mode}', expected one of {sorted(allowed_modes)}")

        selected = set(self._sca_recalibration_layer_indices) if self._sca_recalibration_layer_indices else None
        total_losses: List[torch.Tensor] = []
        mse_terms: List[torch.Tensor] = []
        cosine_terms: List[torch.Tensor] = []
        norm_terms: List[torch.Tensor] = []
        delta_norm_ratio_terms: List[torch.Tensor] = []
        delta_norm_cap_terms: List[torch.Tensor] = []
        per_layer_loss: Dict[str, float] = {}
        per_layer_norm: Dict[str, float] = {}
        per_layer_delta_norm_ratio: Dict[str, float] = {}
        fallback_by_layer: Dict[str, float] = {}

        for layer_idx, wrapper in enumerate(self.sca_sparse_mlps):
            if selected is not None and layer_idx not in selected:
                continue
            alignment = wrapper.get_last_alignment()
            if alignment is None:
                continue

            dense_out = alignment["dense_mlp_out"].float()
            sparse_out = alignment["sparse_mlp_out"].float()
            if not torch.isfinite(dense_out).all() or not torch.isfinite(sparse_out).all():
                raise RuntimeError(f"Non-finite alignment activations captured for layer {layer_idx}")
            mse = F.mse_loss(sparse_out, dense_out)
            cosine = 1.0 - F.cosine_similarity(
                sparse_out.reshape(1, -1),
                dense_out.reshape(1, -1),
                dim=-1,
                eps=1e-6,
            ).mean()
            dense_norm = dense_out.norm(dim=-1).mean()
            sparse_norm = sparse_out.norm(dim=-1).mean()
            norm_penalty = (sparse_norm - dense_norm).pow(2)
            delta = sparse_out - dense_out
            delta_norm_ratio = delta.pow(2).mean() / dense_out.pow(2).mean().clamp_min(1e-6)
            delta_cap_penalty = torch.relu(delta_norm_ratio - float(delta_norm_cap)).pow(2)

            if loss_mode == "mse":
                layer_loss = mse
            elif loss_mode == "cosine":
                layer_loss = cosine
            else:
                layer_loss = mse + (float(norm_weight) * norm_penalty)
            if float(delta_norm_cap_weight) > 0.0:
                layer_loss = layer_loss + (float(delta_norm_cap_weight) * delta_cap_penalty)
            if layer_weight_map is not None:
                layer_loss = layer_loss * float(layer_weight_map.get(int(layer_idx), 1.0))

            total_losses.append(layer_loss)
            mse_terms.append(mse.detach())
            cosine_terms.append(cosine.detach())
            norm_terms.append(norm_penalty.detach())
            delta_norm_ratio_terms.append(delta_norm_ratio.detach())
            delta_norm_cap_terms.append(delta_cap_penalty.detach())
            per_layer_loss[str(layer_idx)] = float(layer_loss.detach().cpu().item())
            per_layer_norm[str(layer_idx)] = float(norm_penalty.detach().cpu().item())
            per_layer_delta_norm_ratio[str(layer_idx)] = float(delta_norm_ratio.detach().cpu().item())
            fallback_by_layer[str(layer_idx)] = float(
                alignment.get(
                    "fallback_triggered",
                    torch.tensor(0.0, device=sparse_out.device, dtype=sparse_out.dtype),
                )
                .float()
                .detach()
                .mean()
                .cpu()
                .item()
            )

        if not total_losses:
            raise RuntimeError("No sparse MLP alignment activations were captured; enable capture before forward")

        total = torch.stack(total_losses).mean()
        fallback_values = list(fallback_by_layer.values())
        metrics: Dict[str, Any] = {
            "layers_captured": float(len(total_losses)),
            "loss_mse": float(torch.stack(mse_terms).mean().cpu().item()),
            "loss_cosine": float(torch.stack(cosine_terms).mean().cpu().item()),
            "loss_norm": float(torch.stack(norm_terms).mean().cpu().item()),
            "loss_delta_norm_ratio": float(torch.stack(delta_norm_ratio_terms).mean().cpu().item()),
            "loss_delta_norm_cap": float(torch.stack(delta_norm_cap_terms).mean().cpu().item()),
            "fallback_rate": float(sum(fallback_values) / max(len(fallback_values), 1)),
            "fallback_rate_by_layer": fallback_by_layer,
            "per_layer_loss": per_layer_loss,
            "per_layer_norm": per_layer_norm,
            "per_layer_delta_norm_ratio": per_layer_delta_norm_ratio,
        }
        return total, metrics

    def export_sca_recalibration_state(self) -> Dict[str, Any]:
        active_sparse_layers = [
            int(v) for v in getattr(self, "_sca_recalibration_active_sparse_layer_indices", []) if int(v) >= 0
        ]
        export_layers = sorted(set(active_sparse_layers or self._sca_recalibration_layer_indices))
        export_layer_set = set(export_layers)
        payload: Dict[str, Any] = {
            "model_name": self.model_name,
            "hybrid_checkpoint_path": self._sca_recalibration_hybrid_checkpoint_path,
            "layer_selection": list(export_layers),
            "active_sparse_layer_selection": list(active_sparse_layers),
            "sca_config": self.sca_config.to_dict(),
            "recalibration_mode": self._sca_recalibration_mode,
            "trainable_modules": list(self._sca_recalibration_trainable_modules),
            "spatial_proj_state_dict": self.spatial_proj.state_dict(),
            "hybrid_config_summary": {
                "enabled": bool(self.attention_hybrid_enabled),
                "layers": list(self.get_effective_hybrid_layers()),
                "target_rank": self.attention_hybrid_target_rank,
                "variance_threshold": self.attention_hybrid_variance_threshold,
                "state_dim": self.attention_hybrid_state_dim,
                "force_no_cache": bool(self.attention_hybrid_force_no_cache),
            },
            "router_metadata_snapshot": {
                "sparse_mlp_bank_status": self.get_sparse_mlp_bank_status(),
                "disable_task_bias_injection": bool(self.disable_task_bias_injection),
                "strict_decode_upper_layer_cap_enabled": bool(self.strict_decode_upper_layer_cap_enabled),
                "bottom_buffer_layers": int(getattr(self, "bottom_buffer_layers", 0)),
            },
        }
        if any(name.startswith("task_embedding.") for name in self._sca_recalibration_trainable_modules):
            payload["task_embedding_state_dict"] = self.task_embedding.state_dict()
        if "adapter_scale" in self._sca_recalibration_trainable_modules:
            payload["adapter_scale"] = self.adapter_scale.detach().cpu()
        if "sca_layer_output_scale" in self._sca_recalibration_trainable_modules and hasattr(self, "sca_layer_output_scale"):
            payload["sca_layer_output_scale"] = self.sca_layer_output_scale.detach().cpu()
        sparse_wrapper_state: Dict[str, Dict[str, torch.Tensor]] = {}
        for layer_idx, wrapper in enumerate(self.sca_sparse_mlps):
            if int(layer_idx) in export_layer_set and hasattr(wrapper, "export_sparse_recalibration_state"):
                sparse_wrapper_state[str(layer_idx)] = wrapper.export_sparse_recalibration_state()
        if sparse_wrapper_state:
            payload["sparse_mlp_wrapper_state"] = sparse_wrapper_state
        return payload

    def load_sca_recalibration_state(
        self,
        checkpoint_path: str,
        strict: bool = True,
    ) -> Dict[str, int]:
        blob = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(blob, dict):
            raise RuntimeError("SCA recalibration checkpoint must be a dict")

        sca_cfg = blob.get("sca_config")
        if isinstance(sca_cfg, dict):
            if "sparse_placement" in sca_cfg:
                self.sca_config.sparse_placement = str(sca_cfg["sparse_placement"])
            if "routing_mode" in sca_cfg:
                self.sca_config.routing_mode = str(sca_cfg["routing_mode"])
            if "basis_rank_by_layer" in sca_cfg:
                self.sca_config.basis_rank_by_layer = self.sca_config._normalize_layer_map(sca_cfg["basis_rank_by_layer"])
            if "basis_top_k_by_layer" in sca_cfg:
                self.sca_config.basis_top_k_by_layer = self.sca_config._normalize_layer_map(sca_cfg["basis_top_k_by_layer"])
            if "top_k_by_layer" in sca_cfg:
                self.sca_config.top_k_by_layer = self.sca_config._normalize_layer_map(sca_cfg["top_k_by_layer"])

        loaded = 0
        missing = 0
        spatial_state = blob.get("spatial_proj_state_dict")
        if spatial_state is None:
            if strict:
                raise RuntimeError("SCA recalibration checkpoint missing 'spatial_proj_state_dict'")
            missing += 1
        else:
            self.spatial_proj.load_state_dict(spatial_state, strict=True)
            loaded += 1

        task_state = blob.get("task_embedding_state_dict")
        if task_state is not None:
            self.task_embedding.load_state_dict(task_state, strict=True)
            loaded += 1
        else:
            missing += 1

        adapter_scale = blob.get("adapter_scale")
        if adapter_scale is not None and hasattr(self, "adapter_scale"):
            self.adapter_scale.data = adapter_scale.to(device=self.adapter_scale.device, dtype=self.adapter_scale.dtype)
            loaded += 1
        layer_output_scale = blob.get("sca_layer_output_scale")
        if layer_output_scale is not None and hasattr(self, "sca_layer_output_scale"):
            self.sca_layer_output_scale.data = layer_output_scale.to(
                device=self.sca_layer_output_scale.device,
                dtype=self.sca_layer_output_scale.dtype,
            )
            loaded += 1
        sparse_wrapper_state = blob.get("sparse_mlp_wrapper_state")
        if isinstance(sparse_wrapper_state, dict):
            for key, state in sparse_wrapper_state.items():
                layer_idx = int(key)
                if layer_idx < 0 or layer_idx >= len(self.sca_sparse_mlps):
                    if strict:
                        raise RuntimeError(f"Sparse wrapper state references invalid layer {layer_idx}")
                    continue
                info = self.sca_sparse_mlps[layer_idx].load_sparse_recalibration_state(state, strict=bool(strict))
                loaded += int(info.get("loaded_items", 0))
                missing += int(info.get("missing_items", 0))

        layer_selection = blob.get("layer_selection")
        if isinstance(layer_selection, list):
            self._sca_recalibration_layer_indices = [int(v) for v in layer_selection]
        active_sparse_layer_selection = blob.get("active_sparse_layer_selection")
        if isinstance(active_sparse_layer_selection, list):
            self._sca_recalibration_active_sparse_layer_indices = [int(v) for v in active_sparse_layer_selection]
        self._sca_recalibration_mode = str(blob.get("recalibration_mode", self._sca_recalibration_mode))
        self._sca_recalibration_trainable_modules = [
            str(v) for v in blob.get("trainable_modules", self._sca_recalibration_trainable_modules)
        ]
        self._sca_recalibration_hybrid_checkpoint_path = str(
            blob.get("hybrid_checkpoint_path", self._sca_recalibration_hybrid_checkpoint_path)
        )
        router_metadata = blob.get("router_metadata_snapshot")
        if isinstance(router_metadata, dict) and "disable_task_bias_injection" in router_metadata:
            disable_bias = bool(router_metadata["disable_task_bias_injection"])
            self.disable_task_bias_injection = disable_bias
            self._sca_disable_task_bias_injection_from_artifact = disable_bias
        if isinstance(router_metadata, dict) and "strict_decode_upper_layer_cap_enabled" in router_metadata:
            self.strict_decode_upper_layer_cap_enabled = bool(router_metadata["strict_decode_upper_layer_cap_enabled"])
        if isinstance(router_metadata, dict) and "bottom_buffer_layers" in router_metadata:
            self.bottom_buffer_layers = max(int(router_metadata["bottom_buffer_layers"]), 0)
        return {"loaded_items": int(loaded), "missing_items": int(missing)}

    def load_learned_basis_init(self, checkpoint_path: str, strict: bool = True) -> Dict[str, int]:
        blob = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(blob, dict):
            raise RuntimeError("learned-basis init checkpoint must be a dict")
        config = blob.get("config", {})
        if not isinstance(config, dict):
            config = {}
        if "sparse_placement" in config:
            self.sca_config.sparse_placement = str(config.get("sparse_placement"))
        if "routing_mode" in config:
            self.sca_config.routing_mode = str(config.get("routing_mode"))
        if "basis_rank_by_layer" in config:
            self.sca_config.basis_rank_by_layer = self.sca_config._normalize_layer_map(config.get("basis_rank_by_layer", {}))
        if "basis_top_k_by_layer" in config:
            self.sca_config.basis_top_k_by_layer = self.sca_config._normalize_layer_map(config.get("basis_top_k_by_layer", {}))
        if "top_k_by_layer" in config:
            self.sca_config.top_k_by_layer = self.sca_config._normalize_layer_map(config.get("top_k_by_layer", {}))
        if self.sca_config.sparse_placement != "learned_basis":
            raise RuntimeError(
                "learned-basis init requires sparse_placement='learned_basis'; "
                f"got '{self.sca_config.sparse_placement}'"
            )
        expected_rank = int(getattr(self.sca_config, "basis_rank", 0))
        ckpt_rank = int(config.get("basis_rank", expected_rank))
        if expected_rank > 0 and ckpt_rank > 0 and ckpt_rank != expected_rank:
            raise RuntimeError(f"basis_rank mismatch: model={expected_rank}, checkpoint={ckpt_rank}")

        layer_states = blob.get("layer_states")
        if not isinstance(layer_states, dict):
            raise RuntimeError("learned-basis init checkpoint missing 'layer_states'")

        loaded = 0
        missing = 0
        for key, payload in layer_states.items():
            layer_idx = int(key)
            if layer_idx < 0 or layer_idx >= len(self.sca_sparse_mlps):
                if strict:
                    raise RuntimeError(f"learned-basis init references invalid layer {layer_idx}")
                missing += 1
                continue
            if not isinstance(payload, dict):
                if strict:
                    raise RuntimeError(f"learned-basis init payload for layer {layer_idx} must be a dict")
                missing += 1
                continue
            info = self.sca_sparse_mlps[layer_idx].initialize_sparse_basis_from_dense_init(payload, strict=bool(strict))
            loaded += int(info.get("loaded_items", 0))
            missing += int(info.get("missing_items", 0))
        return {"loaded_items": int(loaded), "missing_items": int(missing)}

    def prepare_decoder_mirror_calibration(
        self,
        source_layer_indices: Optional[List[int]] = None,
        top_k: Optional[int] = None,
        rank: int = 4,
        route_conditioned: bool = True,
        calibration_mode: str = "decoder_co_warp",
        hybrid_checkpoint_path: str = "",
        sca_recalibrated_checkpoint_path: str = "",
    ) -> List[nn.Parameter]:
        for param in self.parameters():
            param.requires_grad = False

        mirror = self._ensure_decoder_mirror_module(
            source_layers=source_layer_indices,
            top_k=top_k,
            rank=rank,
            route_conditioned=route_conditioned,
        )
        mirror.to(device=self.device, dtype=torch.float32)
        self._decoder_mirror_enabled = True
        self._decoder_mirror_calibration_mode = str(calibration_mode)
        self._decoder_mirror_hybrid_checkpoint_path = str(hybrid_checkpoint_path or "")
        self._decoder_mirror_sca_checkpoint_path = str(sca_recalibrated_checkpoint_path or "")
        self.set_decoder_mirror_route_capture(True, layer_indices=mirror.config.source_layer_indices)

        trainable: List[nn.Parameter] = []
        trainable_names: List[str] = []
        for name, param in mirror.named_parameters():
            param.requires_grad = True
            trainable.append(param)
            trainable_names.append(f"decoder_mirror.{name}")
        self._decoder_mirror_trainable_modules = trainable_names
        return trainable

    def compute_decoder_mirror_calibration_loss(
        self,
        mirror_logits: torch.Tensor,
        dense_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        logits_kl_weight: float = 1.0,
        ce_weight: float = 0.25,
        delta_penalty_weight: float = 0.01,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.decoder_mirror is None:
            raise RuntimeError("Decoder mirror is not initialized")
        if not torch.isfinite(mirror_logits).all() or not torch.isfinite(dense_logits).all():
            raise RuntimeError("Non-finite logits passed into decoder mirror calibration loss")

        mirror_logp = F.log_softmax(mirror_logits.float(), dim=-1)
        dense_probs = F.softmax(dense_logits.float(), dim=-1)
        logits_kl = F.kl_div(mirror_logp, dense_probs, reduction="batchmean")
        ce = mirror_logits.new_zeros(())
        if labels is not None:
            vocab_size = int(getattr(getattr(self, "config", None), "vocab_size", mirror_logits.shape[-1]))
            ce = self.model.loss_function(logits=mirror_logits, labels=labels, vocab_size=vocab_size)
        delta_ratio_term = self.decoder_mirror.get_last_delta_norm_ratio_term()
        if delta_ratio_term is None:
            delta_ratio_term = mirror_logits.new_zeros(())
        total = (
            float(logits_kl_weight) * logits_kl
            + float(ce_weight) * ce
            + float(delta_penalty_weight) * delta_ratio_term
        )
        if not torch.isfinite(total):
            raise RuntimeError("Non-finite total decoder mirror calibration loss")

        fallback_rate_by_layer = {
            str(idx): float(bool(getattr(wrapper, "_last_fallback_triggered", False)))
            for idx, wrapper in enumerate(getattr(self, "sca_sparse_mlps", []))
        }
        diagnostics = self.get_decoder_mirror_diagnostics()
        metrics: Dict[str, Any] = {
            "loss_total": float(total.detach().cpu().item()),
            "loss_logits_kl": float(logits_kl.detach().cpu().item()),
            "loss_ce": float(ce.detach().cpu().item()),
            "loss_delta_norm": float(delta_ratio_term.detach().cpu().item()),
            "mean_active_blocks": float(diagnostics.get("mean_active_blocks", 0.0)),
            "mean_touched_weight_fraction": float(diagnostics.get("touched_weight_fraction", 0.0)),
            "fallback_rate_by_layer": fallback_rate_by_layer,
            "route_prior_missing": bool(diagnostics.get("route_prior_missing", True)),
        }
        return total, metrics

    def export_decoder_mirror_state(self) -> Dict[str, Any]:
        if self.decoder_mirror is None or self.decoder_mirror_config is None:
            raise RuntimeError("Decoder mirror is not initialized")
        return {
            "model_name": self.model_name,
            "hybrid_checkpoint_path": self._decoder_mirror_hybrid_checkpoint_path,
            "sca_recalibrated_checkpoint_path": self._decoder_mirror_sca_checkpoint_path,
            "mirror_config": self.decoder_mirror_config.to_dict(),
            "source_layer_indices": list(self.decoder_mirror_config.source_layer_indices),
            "route_conditioned": bool(self.decoder_mirror_config.route_conditioned),
            "trainable_modules": list(self._decoder_mirror_trainable_modules),
            "decoder_mirror_state_dict": self.decoder_mirror.state_dict(),
            "hybrid_config_summary": {
                "enabled": bool(self.attention_hybrid_enabled),
                "layers": list(self.get_effective_hybrid_layers()),
                "target_rank": self.attention_hybrid_target_rank,
                "variance_threshold": self.attention_hybrid_variance_threshold,
                "state_dim": self.attention_hybrid_state_dim,
            },
            "sca_config_summary": self.sca_config.to_dict(),
        }

    def load_decoder_mirror_state(self, checkpoint_path: str, strict: bool = True) -> Dict[str, int]:
        blob = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(blob, dict):
            raise RuntimeError("Decoder mirror checkpoint must be a dict")
        mirror_config_payload = blob.get("mirror_config")
        if not isinstance(mirror_config_payload, dict):
            raise RuntimeError("Decoder mirror checkpoint missing 'mirror_config'")
        if int(mirror_config_payload.get("block_size", self.sca_config.block_size)) != int(self.sca_config.block_size):
            raise RuntimeError("Decoder mirror block_size does not match main SCA config")
        if int(mirror_config_payload.get("num_blocks", self.sca_config.num_blocks)) != int(self.sca_config.num_blocks):
            raise RuntimeError("Decoder mirror num_blocks does not match main SCA config")

        self._ensure_decoder_mirror_module(
            source_layers=[int(v) for v in mirror_config_payload.get("source_layer_indices", [])],
            top_k=int(mirror_config_payload.get("top_k", self.sca_config.top_k)),
            rank=int(mirror_config_payload.get("rank", 4)),
            route_conditioned=bool(mirror_config_payload.get("route_conditioned", True)),
            route_prior_scale_init=float(mirror_config_payload.get("route_prior_scale_init", 0.25)),
            residual_scale_init=float(mirror_config_payload.get("residual_scale_init", 0.0)),
        )
        state = blob.get("decoder_mirror_state_dict")
        if not isinstance(state, dict):
            raise RuntimeError("Decoder mirror checkpoint missing 'decoder_mirror_state_dict'")
        self.decoder_mirror.load_state_dict(state, strict=bool(strict))
        self.decoder_mirror.to(device=self.device, dtype=torch.float16)
        self._decoder_mirror_hybrid_checkpoint_path = str(blob.get("hybrid_checkpoint_path", ""))
        self._decoder_mirror_sca_checkpoint_path = str(blob.get("sca_recalibrated_checkpoint_path", ""))
        self._decoder_mirror_trainable_modules = [str(v) for v in blob.get("trainable_modules", [])]
        self.set_decoder_mirror_enabled(True)
        return {"loaded_items": 1, "missing_items": 0}

    def prepare_local_geometry_calibration(self, include_output_bias: bool = False) -> List[nn.Parameter]:
        for param in self.parameters():
            param.requires_grad = False
        trainable: List[nn.Parameter] = []
        self.set_hybrid_alignment_capture(True)
        for _layer_idx, attn in self.iter_hybrid_attention_modules():
            params = [attn.mix_logit, attn.output_gain]
            for group in attn.mamba_block.groups:
                params.extend([group.dynamics.a_logit, group.dynamics.delta_log])
            if include_output_bias:
                attn.enable_output_bias = True
                params.append(attn.output_bias)
            for param in params:
                param.requires_grad = True
                trainable.append(param)
        return trainable

    def compute_local_geometry_loss(
        self,
        mix_regularization: float = 0.01,
        norm_regularization: float = 0.01,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses: List[torch.Tensor] = []
        mix_values: List[float] = []
        norm_terms: List[float] = []

        for _layer_idx, attn in self.iter_hybrid_attention_modules():
            alignment = attn.get_last_alignment()
            if alignment is None:
                continue
            original_out = torch.nan_to_num(alignment["original_out"].float(), nan=0.0, posinf=1e3, neginf=-1e3)
            mamba_out = torch.nan_to_num(alignment["mamba_out"].float(), nan=0.0, posinf=1e3, neginf=-1e3)
            diff = torch.nan_to_num(mamba_out - original_out, nan=0.0, posinf=1e3, neginf=-1e3)
            mse = (diff.square()).mean()
            orig_norm = torch.clamp(original_out.norm(dim=-1).mean(), max=1e3)
            mamba_norm = torch.clamp(mamba_out.norm(dim=-1).mean(), max=1e3)
            norm_penalty = (mamba_norm - orig_norm).pow(2)
            mix_penalty = attn.mix_value.float().pow(2)
            losses.append(mse + (float(mix_regularization) * mix_penalty) + (float(norm_regularization) * norm_penalty))
            mix_values.append(float(torch.nan_to_num(attn.mix_value.detach().cpu(), nan=torch.tensor(0.0)).item()))
            norm_terms.append(float(norm_penalty.detach().cpu().item()))

        if not losses:
            raise RuntimeError("No hybrid alignment activations were captured; ensure capture is enabled before forward")

        total = torch.stack(losses).mean()
        metrics = {
            "hybrid_layers": float(len(losses)),
            "mean_mix": float(sum(mix_values) / max(len(mix_values), 1)),
            "mean_norm_penalty": float(sum(norm_terms) / max(len(norm_terms), 1)),
        }
        return total, metrics

    def sanitize_hybrid_parameters(self) -> Dict[str, int]:
        repaired = 0
        for _layer_idx, attn in self.iter_hybrid_attention_modules():
            with torch.no_grad():
                before = attn.mix_logit.clone()
                attn.mix_logit.copy_(torch.nan_to_num(attn.mix_logit, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0))
                attn.output_gain.copy_(torch.nan_to_num(attn.output_gain, nan=1.0, posinf=2.0, neginf=0.0).clamp(0.0, 2.0))
                attn.output_bias.copy_(torch.nan_to_num(attn.output_bias, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0))
                for group in attn.mamba_block.groups:
                    group.dynamics.a_logit.copy_(
                        torch.nan_to_num(group.dynamics.a_logit, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)
                    )
                    group.dynamics.delta_log.copy_(
                        torch.nan_to_num(group.dynamics.delta_log, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)
                    )
                if not torch.equal(before, attn.mix_logit):
                    repaired += 1
        return {"repaired_layers": int(repaired)}

    def export_hybrid_attention_state(self) -> Dict[str, Any]:
        state_dict: Dict[str, Any] = {}
        mix_values: Dict[str, float] = {}
        layer_selection: List[int] = []
        for layer_idx, attn in self.iter_hybrid_attention_modules():
            key = f"layers.{layer_idx}.self_attn"
            state_dict[key] = attn.export_hybrid_state()
            mix_values[key] = float(attn.mix_value.detach().cpu().item())
            layer_selection.append(layer_idx)
        return {
            "model_name": self.model_name,
            "layer_selection": layer_selection,
            "target_rank": self.attention_hybrid_target_rank,
            "variance_threshold": self.attention_hybrid_variance_threshold,
            "state_dim": self.attention_hybrid_state_dim,
            "hybrid_attention_state_dict": state_dict,
            "mix_values_by_layer": mix_values,
        }

    def export_collapsed_attention_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        if self.attention_hybrid_enabled:
            return self.export_hybrid_attention_state()
        payload: Dict[str, Dict[str, torch.Tensor]] = {}
        for layer_idx, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn
            if hasattr(attn, "mamba_block"):
                payload[f"layers.{layer_idx}.self_attn"] = attn.state_dict()
        return payload

    def load_hybrid_attention_state(
        self,
        checkpoint_path: str,
        strict: bool = True,
    ) -> Dict[str, int]:
        blob = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(blob, dict):
            raise RuntimeError("hybrid attention checkpoint must be a dict")
        state_map = blob.get("hybrid_attention_state_dict")
        if state_map is None:
            state_map = blob.get("collapsed_attention_state_dict", blob)
        if not isinstance(state_map, dict):
            raise RuntimeError("checkpoint must contain 'hybrid_attention_state_dict' or 'collapsed_attention_state_dict'")

        loaded = 0
        missing = 0
        for layer_idx, layer in enumerate(self.model.model.layers):
            key = f"layers.{layer_idx}.self_attn"
            attn = layer.self_attn
            layer_state = state_map.get(key)
            if layer_state is None:
                if strict and isinstance(attn, HybridCollapsedMambaSelfAttention):
                    raise RuntimeError(f"Missing hybrid state for {key}")
                missing += 1
                continue
            if isinstance(attn, HybridCollapsedMambaSelfAttention):
                attn.load_hybrid_state(layer_state, strict=strict)
                loaded += 1
                continue
            if hasattr(attn, "mamba_block"):
                attn.load_state_dict(layer_state, strict=True)
                loaded += 1
                continue
            if strict:
                raise RuntimeError(f"Layer {layer_idx} self_attn is not hybrid or collapsed-Mamba")
            missing += 1
        return {"loaded_layers": int(loaded), "missing_layers": int(missing)}

    def load_collapsed_attention_state(
        self,
        checkpoint_path: str,
        strict: bool = True,
    ) -> Dict[str, int]:
        return self.load_hybrid_attention_state(checkpoint_path=checkpoint_path, strict=strict)

    def clear_sparse_mlp_bank_manifest(self) -> None:
        for wrapper in self.sca_sparse_mlps:
            if hasattr(wrapper, "clear_block_bank"):
                wrapper.clear_block_bank()
        self._sparse_mlp_bank_manifest_path = None

    def load_sparse_mlp_bank_manifest(
        self,
        manifest_path: str,
        strict: bool = True,
    ) -> Dict[str, int]:
        manifest_file = Path(manifest_path)
        if not manifest_file.exists():
            raise RuntimeError(f"Sparse MLP manifest does not exist: {manifest_path}")
        blob = json.loads(manifest_file.read_text(encoding="utf-8"))
        if not isinstance(blob, dict):
            raise RuntimeError("Sparse MLP manifest must be a JSON object")
        layer_entries = blob.get("layers", [])
        if not isinstance(layer_entries, list):
            raise RuntimeError("Sparse MLP manifest missing 'layers' list")

        by_layer: Dict[int, Dict[str, Any]] = {}
        for entry in layer_entries:
            if not isinstance(entry, dict):
                continue
            if "layer_idx" not in entry:
                continue
            by_layer[int(entry["layer_idx"])] = entry

        self.clear_sparse_mlp_bank_manifest()
        loaded = 0
        missing = 0
        for layer_idx, wrapper in enumerate(self.sca_sparse_mlps):
            entry = by_layer.get(layer_idx)
            if entry is None:
                missing += 1
                if strict:
                    raise RuntimeError(f"Sparse MLP manifest missing layer {layer_idx}")
                continue
            layer_file_name = entry.get("file")
            if not layer_file_name:
                if strict:
                    raise RuntimeError(f"Sparse MLP manifest entry for layer {layer_idx} has no file")
                missing += 1
                continue
            layer_file = manifest_file.parent / str(layer_file_name)
            if not layer_file.exists():
                if strict:
                    raise RuntimeError(f"Sparse MLP bank layer file is missing: {layer_file}")
                missing += 1
                continue
            payload = torch.load(layer_file, map_location="cpu")
            if not isinstance(payload, dict):
                if strict:
                    raise RuntimeError(f"Sparse MLP bank layer payload must be dict: {layer_file}")
                missing += 1
                continue
            wrapper.load_block_bank(payload, strict=strict)
            loaded += 1

        self._sparse_mlp_bank_manifest_path = str(manifest_file)
        return {"loaded_layers": int(loaded), "missing_layers": int(missing)}

    def get_sparse_mlp_bank_status(self) -> Dict[str, Any]:
        loaded_layers: List[int] = []
        for layer_idx, wrapper in enumerate(self.sca_sparse_mlps):
            has_bank = bool(hasattr(wrapper, "has_block_bank") and wrapper.has_block_bank())
            if has_bank:
                loaded_layers.append(layer_idx)
        return {
            "loaded_layers": int(len(loaded_layers)),
            "layer_indices": loaded_layers,
            "manifest_path": self._sparse_mlp_bank_manifest_path,
        }

    def get_sparse_mlp_diagnostics(self) -> Dict[str, Any]:
        layers: List[Dict[str, Any]] = []
        total_bytes = 0.0
        mean_fraction = 0.0
        counted = 0
        latent_norms: List[float] = []
        latent_active_fractions: List[float] = []
        latent_coords_used: List[float] = []
        latent_usage_entropies: List[float] = []
        latent_support_overlaps: List[float] = []
        latent_support_unique_fractions: List[float] = []
        max_latent_importance_fractions: List[float] = []
        max_latent_load_fractions: List[float] = []
        max_block_importance_fractions: List[float] = []
        max_block_load_fractions: List[float] = []
        for wrapper in self.sca_sparse_mlps:
            diag = wrapper.get_last_diagnostics()
            if diag is None:
                continue
            row = diag.to_dict()
            learned_basis = wrapper.get_last_learned_basis_stats()
            if learned_basis is not None:
                row["learned_basis"] = learned_basis
                latent_norms.append(float(learned_basis.get("mean_latent_norm", 0.0)))
                latent_active_fractions.append(float(learned_basis.get("active_latent_fraction", 0.0)))
                latent_coords_used.append(float(learned_basis.get("active_latent_coords_used", 0.0)))
                latent_usage_entropies.append(float(learned_basis.get("latent_usage_entropy", 0.0)))
                latent_support_overlaps.append(float(learned_basis.get("support_overlap_mean", 0.0)))
                latent_support_unique_fractions.append(float(learned_basis.get("support_unique_fraction", 0.0)))
                max_latent_importance_fractions.append(float(learned_basis.get("max_latent_importance_fraction", 0.0)))
                max_latent_load_fractions.append(float(learned_basis.get("max_latent_load_fraction", 0.0)))
                max_block_importance_fractions.append(float(learned_basis.get("max_block_importance_fraction", 0.0)))
                max_block_load_fractions.append(float(learned_basis.get("max_block_load_fraction", 0.0)))
            row.update(wrapper.get_fallback_stats())
            layers.append(row)
            total_bytes += float(diag.estimated_bytes_fetched_per_token)
            mean_fraction += float(diag.touched_weight_fraction)
            counted += 1
        mean_fallback = 0.0
        if layers:
            mean_fallback = float(sum(float(layer.get("dense_fallback_rate", 0.0)) for layer in layers) / len(layers))
        return {
            "layers": layers,
            "mean_touched_weight_fraction": float(mean_fraction / max(counted, 1)),
            "estimated_bytes_fetched_per_token": float(total_bytes),
            "mean_dense_fallback_rate": float(mean_fallback),
            "mean_latent_norm": float(sum(latent_norms) / max(len(latent_norms), 1)),
            "mean_active_latent_fraction": float(sum(latent_active_fractions) / max(len(latent_active_fractions), 1)),
            "mean_active_latent_coords_used": float(sum(latent_coords_used) / max(len(latent_coords_used), 1)),
            "mean_latent_usage_entropy": float(sum(latent_usage_entropies) / max(len(latent_usage_entropies), 1)),
            "mean_latent_support_overlap": float(sum(latent_support_overlaps) / max(len(latent_support_overlaps), 1)),
            "mean_latent_support_unique_fraction": float(
                sum(latent_support_unique_fractions) / max(len(latent_support_unique_fractions), 1)
            ),
            "mean_max_latent_importance_fraction": float(
                sum(max_latent_importance_fractions) / max(len(max_latent_importance_fractions), 1)
            ),
            "mean_max_latent_load_fraction": float(
                sum(max_latent_load_fractions) / max(len(max_latent_load_fractions), 1)
            ),
            "mean_max_block_importance_fraction": float(
                sum(max_block_importance_fractions) / max(len(max_block_importance_fractions), 1)
            ),
            "mean_max_block_load_fraction": float(sum(max_block_load_fractions) / max(len(max_block_load_fractions), 1)),
        }

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

        if self.sparse_attention_config.enabled and self._sparse_attention_runtime is not None:
            sparse_diag = self.get_sparse_mlp_diagnostics()
            legacy_cache, step_stats = self._sparse_attention_runtime.update_and_compress_cache(
                legacy_cache=legacy_cache,
                sparse_mlp_diagnostics=sparse_diag,
            )
            self._last_sparse_attention_step = step_stats.to_dict()

        if (not self.sparse_attention_config.enabled) and self._bounded_context_enabled() and self._tier_cache_compressor is not None:
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
        legacy = tuple(restored_layers)
        if DynamicCache is not None:
            try:
                return DynamicCache.from_legacy_cache(legacy)
            except Exception:
                return legacy
        return legacy

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

    @staticmethod
    def _apply_repetition_penalty_to_scores(
        scores: torch.Tensor,
        generated: torch.Tensor,
        penalty: float,
        window: int,
    ) -> torch.Tensor:
        if penalty <= 1.0 or window <= 0 or generated.numel() == 0:
            return scores
        out = scores.clone()
        tail = generated[:, -int(window) :]
        for row in range(out.shape[0]):
            seen = torch.unique(tail[row])
            if seen.numel() == 0:
                continue
            selected = out[row, seen]
            penalized = torch.where(selected > 0, selected / float(penalty), selected * float(penalty))
            out[row, seen] = penalized
        return out

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
                    output_scale_fn=self._layer_output_scale,
                )
                try:
                    target_device = next(layer.parameters()).device
                except StopIteration:
                    target_device = self.device
                wrapped.to(device=target_device)
                layer.mlp = wrapped
            self.sca_sparse_mlps.append(wrapped)

    def _sparse_layer_enabled(self, layer_idx: int) -> bool:
        if not self.neuroplasticity_enabled:
            return False
        num_layers = len(self.model.model.layers)
        lower = max(int(getattr(self, "bottom_buffer_layers", 0)), 0)
        upper = num_layers - int(getattr(self, "buffer_layers", 0)) - int(getattr(self, "decode_guard_layers", 0))
        enabled = lower <= int(layer_idx) < max(upper, 0)
        if not enabled:
            return False
        override = getattr(self, "_sca_sparse_layer_override", None)
        if override is None:
            return True
        return int(layer_idx) in override

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

        use_cuda_kernel = bool(self.sca_config.use_cuda and query.is_cuda)
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
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        task_id: int = 0,
        **kwargs: Any,
    ):
        self._current_task_id = int(task_id)
        self._cached_task_emb = self.task_embedding(torch.tensor(self._current_task_id, device=self.device))
        self._last_decoder_mirror_diagnostics = {}

        # Keep clean baseline path when disabled: no task embedding injection.
        if (
            self.neuroplasticity_enabled
            and (not self.disable_task_bias_injection)
            and inputs_embeds is None
            and input_ids is not None
        ):
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
        model_kwargs: Dict[str, Any] = dict(kwargs)
        # LlamaForCausalLM forwards unknown kwargs into loss_function when labels are set.
        # Keep tier-specific kwargs on generation path only.
        if tier_mask_bundle is not None and labels is None:
            model_kwargs["tier_mask_bundle"] = tier_mask_bundle

        use_decoder_mirror = bool(
            getattr(self, "_decoder_mirror_enabled", False)
            and self.neuroplasticity_enabled
            and getattr(self, "decoder_mirror", None) is not None
        )
        outputs: Any
        if not use_decoder_mirror:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **model_kwargs,
            )
        else:
            inner_outputs = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                cache_position=cache_position,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                **model_kwargs,
            )
            hidden_states_out = inner_outputs.last_hidden_state
            configured_layers = None if self.decoder_mirror_config is None else self.decoder_mirror_config.source_layer_indices
            route_prior = self._build_decoder_mirror_route_prior(
                source_layers=configured_layers,
                batch_size=int(hidden_states_out.shape[0]),
                seq_len=int(hidden_states_out.shape[1]),
                device=hidden_states_out.device,
            )
            warped_hidden, mirror_diag = self.decoder_mirror(
                hidden_states_out,
                route_prior=route_prior,
                source_layers_used=list(self._decoder_mirror_source_layers_used),
            )
            self._last_decoder_mirror_diagnostics = dict(mirror_diag)
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.model.lm_head(warped_hidden[:, slice_indices, :])

            loss = None
            if labels is not None:
                loss = self.model.loss_function(logits=logits, labels=labels, vocab_size=self.model.config.vocab_size, **model_kwargs)

            hidden_states_tuple = inner_outputs.hidden_states
            if hidden_states_tuple is not None and len(hidden_states_tuple) > 0:
                hidden_states_list = list(hidden_states_tuple)
                hidden_states_list[-1] = warped_hidden
                hidden_states_tuple = tuple(hidden_states_list)

            outputs = CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=inner_outputs.past_key_values,
                hidden_states=hidden_states_tuple,
                attentions=inner_outputs.attentions,
            )
            effective_return_dict = return_dict if return_dict is not None else getattr(self.model.config, "use_return_dict", True)
            if not effective_return_dict:
                values = (loss, logits, inner_outputs.past_key_values, hidden_states_tuple, inner_outputs.attentions)
                outputs = tuple(v for v in values if v is not None)
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
        use_cache = not self._should_disable_generation_cache(bool(use_cache))
        if self.sparse_attention_config.enabled:
            self.reset_sparse_attention_state()
        if self.sparse_attention_config.strict_fully_sparse:
            self.bounded_context_config.validate_sparse_attention_runtime(
                strict_fully_sparse=True,
                sca_use_cuda=bool(self.sca_config.use_cuda),
                sca_spmm_impl=str(self.sca_config.spmm_impl),
                fast_fallback_threshold=float(self.sca_config.stability_dense_fallback_threshold),
                disable_ssd_fetch_in_decode=bool(self.sparse_attention_config.disable_ssd_fetch_in_decode),
                allow_noncuda_spmm_for_diagnostics=bool(self.strict_runtime_allow_noncuda_spmm_diagnostic),
            )

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
            if bool(self.sparse_attention_config.strict_fully_sparse) and bool(self.strict_decode_enable_repetition_penalty):
                next_token_logits = self._apply_repetition_penalty_to_scores(
                    next_token_logits,
                    generated=generated,
                    penalty=float(self.strict_decode_repetition_penalty),
                    window=int(self.strict_decode_penalty_window),
                )
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
        config["sca_bottom_buffer_layers"] = int(getattr(self, "bottom_buffer_layers", 0))
        config["sca_decode_guard_layers"] = int(self.decode_guard_layers)
        config["strict_decode_repetition_penalty"] = float(self.strict_decode_repetition_penalty)
        config["strict_decode_penalty_window"] = int(self.strict_decode_penalty_window)
        config["strict_decode_enable_repetition_penalty"] = bool(self.strict_decode_enable_repetition_penalty)
        config["strict_decode_upper_layer_cap_enabled"] = bool(self.strict_decode_upper_layer_cap_enabled)
        config["strict_runtime_allow_noncuda_spmm_diagnostic"] = bool(self.strict_runtime_allow_noncuda_spmm_diagnostic)
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
        config["hybrid_attention"] = {
            "enabled": bool(self.attention_hybrid_enabled),
            "layers": list(self.get_effective_hybrid_layers()),
            "force_no_cache": bool(self.attention_hybrid_force_no_cache),
            "mix_init": float(self.attention_hybrid_mix_init),
            "target_rank": self.attention_hybrid_target_rank,
            "variance_threshold": float(self.attention_hybrid_variance_threshold),
            "state_dim": self.attention_hybrid_state_dim,
        }
        config["sparse_attention"] = {
            "enabled": bool(self.sparse_attention_config.enabled),
            "local_window_tokens": int(self.sparse_attention_config.local_window_tokens),
            "sink_tokens": int(self.sparse_attention_config.sink_tokens),
            "page_size_tokens": int(self.sparse_attention_config.page_size_tokens),
            "retrieval_top_k_pages": int(self.sparse_attention_config.retrieval_top_k_pages),
            "retrieval_head_group_ids": [int(x) for x in self.sparse_attention_config.retrieval_head_group_ids],
            "retrieval_start_layer": self.sparse_attention_config.retrieval_start_layer,
            "archive_cpu_dtype": str(self.sparse_attention_config.archive_cpu_dtype),
            "hot_archive_gpu_pages": int(self.sparse_attention_config.hot_archive_gpu_pages),
            "disable_ssd_fetch_in_decode": bool(self.sparse_attention_config.disable_ssd_fetch_in_decode),
            "force_single_model_runtime": bool(self.sparse_attention_config.force_single_model_runtime),
            "strict_fully_sparse": bool(self.sparse_attention_config.strict_fully_sparse),
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
            sca_bottom_buffer_layers = int(config.get("sca_bottom_buffer_layers", 2))
            sca_decode_guard_layers = int(config.get("sca_decode_guard_layers", 12))
            strict_decode_repetition_penalty = float(config.get("strict_decode_repetition_penalty", 1.35))
            strict_decode_penalty_window = int(config.get("strict_decode_penalty_window", 64))
            strict_decode_enable_repetition_penalty = bool(config.get("strict_decode_enable_repetition_penalty", False))
            strict_decode_upper_layer_cap_enabled = bool(config.get("strict_decode_upper_layer_cap_enabled", True))
            strict_runtime_allow_noncuda_spmm_diagnostic = bool(
                config.get("strict_runtime_allow_noncuda_spmm_diagnostic", False)
            )
            kv_cfg = config.get("kv_cache", {})
            bounded_cfg = config.get("bounded_context", {})
            hybrid_cfg = config.get("hybrid_attention", {})
            sparse_attn_cfg = config.get("sparse_attention", {})
        else:
            model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
            num_tasks = 10
            sca_cfg = {}
            sca_bottom_buffer_layers = 2
            sca_decode_guard_layers = 12
            strict_decode_repetition_penalty = 1.35
            strict_decode_penalty_window = 64
            strict_decode_enable_repetition_penalty = False
            strict_decode_upper_layer_cap_enabled = True
            strict_runtime_allow_noncuda_spmm_diagnostic = False
            kv_cfg = {}
            bounded_cfg = {}
            hybrid_cfg = {}
            sparse_attn_cfg = {}

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
            sca_basis_rank=int(sca_cfg.get("basis_rank", 32)),
            sca_basis_top_k=int(sca_cfg.get("basis_top_k", 8)),
            sca_top_k=int(sca_cfg.get("top_k", 3)),
            sca_routing_mode=str(sca_cfg.get("routing_mode", "spatial_grid")),
            sca_adaptive_top_k=bool(sca_cfg.get("adaptive_top_k", False)),
            sca_adaptive_top_k_min=int(sca_cfg.get("adaptive_top_k_min", sca_cfg.get("top_k", 3))),
            sca_adaptive_top_k_max=int(sca_cfg.get("adaptive_top_k_max", sca_cfg.get("top_k", 3))),
            sca_adaptive_top_k_min_score_ratio=float(sca_cfg.get("adaptive_top_k_min_score_ratio", 0.15)),
            sca_sigma=float(sca_cfg.get("sigma", 1.0)),
            sca_refractory_steps=int(sca_cfg.get("refractory_steps", 100)),
            sca_inhibition_lambda=float(sca_cfg.get("inhibition_lambda", 0.0)),
            sca_grid_size=int(sca_cfg.get("grid_size", 16)),
            sca_use_cuda=bool(sca_cfg.get("use_cuda", True)),
            sca_spmm_impl=str(sca_cfg.get("spmm_impl", "dense")),
            sca_sparse_placement=str(sca_cfg.get("sparse_placement", "input_mask")),
            sca_soft_mask=bool(sca_cfg.get("soft_mask", True)),
            sca_grouped_row_gemm=bool(sca_cfg.get("grouped_row_gemm", False)),
            sca_grouped_row_min_bucket=int(sca_cfg.get("grouped_row_min_bucket", 2)),
            sca_grouped_row_allow_4bit_dequant=bool(sca_cfg.get("grouped_row_allow_4bit_dequant", False)),
            sca_stability_dense_fallback_threshold=float(sca_cfg.get("stability_dense_fallback_threshold", 0.0)),
            sca_bottom_buffer_layers=int(kwargs.pop("sca_bottom_buffer_layers", sca_bottom_buffer_layers)),
            sca_basis_rank_by_layer=kwargs.pop("sca_basis_rank_by_layer", sca_cfg.get("basis_rank_by_layer", {})),
            sca_basis_top_k_by_layer=kwargs.pop("sca_basis_top_k_by_layer", sca_cfg.get("basis_top_k_by_layer", {})),
            sca_top_k_by_layer=kwargs.pop("sca_top_k_by_layer", sca_cfg.get("top_k_by_layer", {})),
            sca_decode_guard_layers=int(kwargs.pop("sca_decode_guard_layers", sca_decode_guard_layers)),
            strict_decode_repetition_penalty=float(
                kwargs.pop("strict_decode_repetition_penalty", strict_decode_repetition_penalty)
            ),
            strict_decode_penalty_window=int(kwargs.pop("strict_decode_penalty_window", strict_decode_penalty_window)),
            strict_decode_enable_repetition_penalty=bool(
                kwargs.pop("strict_decode_enable_repetition_penalty", strict_decode_enable_repetition_penalty)
            ),
            strict_decode_upper_layer_cap_enabled=bool(
                kwargs.pop("strict_decode_upper_layer_cap_enabled", strict_decode_upper_layer_cap_enabled)
            ),
            strict_runtime_allow_noncuda_spmm_diagnostic=bool(
                kwargs.pop(
                    "strict_runtime_allow_noncuda_spmm_diagnostic",
                    strict_runtime_allow_noncuda_spmm_diagnostic,
                )
            ),
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
            attention_hybrid_enabled=bool(kwargs.pop("attention_hybrid_enabled", hybrid_cfg.get("enabled", False))),
            attention_hybrid_layers=kwargs.pop("attention_hybrid_layers", hybrid_cfg.get("layers")),
            attention_hybrid_mix_init=float(kwargs.pop("attention_hybrid_mix_init", hybrid_cfg.get("mix_init", 0.05))),
            attention_hybrid_target_rank=kwargs.pop("attention_hybrid_target_rank", hybrid_cfg.get("target_rank", 16)),
            attention_hybrid_variance_threshold=float(
                kwargs.pop("attention_hybrid_variance_threshold", hybrid_cfg.get("variance_threshold", 0.90))
            ),
            attention_hybrid_state_dim=kwargs.pop("attention_hybrid_state_dim", hybrid_cfg.get("state_dim")),
            attention_hybrid_force_no_cache=bool(
                kwargs.pop("attention_hybrid_force_no_cache", hybrid_cfg.get("force_no_cache", True))
            ),
            attention_sparse_mode=bool(kwargs.pop("attention_sparse_mode", sparse_attn_cfg.get("enabled", False))),
            attention_local_window_tokens=int(
                kwargs.pop("attention_local_window_tokens", sparse_attn_cfg.get("local_window_tokens", 2048))
            ),
            attention_sink_tokens=int(kwargs.pop("attention_sink_tokens", sparse_attn_cfg.get("sink_tokens", 8))),
            attention_page_size_tokens=int(
                kwargs.pop("attention_page_size_tokens", sparse_attn_cfg.get("page_size_tokens", 256))
            ),
            attention_retrieval_top_k_pages=int(
                kwargs.pop("attention_retrieval_top_k_pages", sparse_attn_cfg.get("retrieval_top_k_pages", 8))
            ),
            attention_retrieval_head_group_ids=kwargs.pop(
                "attention_retrieval_head_group_ids", sparse_attn_cfg.get("retrieval_head_group_ids", [0])
            ),
            attention_retrieval_start_layer=kwargs.pop(
                "attention_retrieval_start_layer", sparse_attn_cfg.get("retrieval_start_layer")
            ),
            attention_archive_cpu_dtype=str(
                kwargs.pop("attention_archive_cpu_dtype", sparse_attn_cfg.get("archive_cpu_dtype", "int4"))
            ),
            attention_hot_archive_gpu_pages=int(
                kwargs.pop("attention_hot_archive_gpu_pages", sparse_attn_cfg.get("hot_archive_gpu_pages", 0))
            ),
            attention_disable_ssd_fetch_in_decode=bool(
                kwargs.pop(
                    "attention_disable_ssd_fetch_in_decode",
                    sparse_attn_cfg.get("disable_ssd_fetch_in_decode", True),
                )
            ),
            attention_force_single_model_runtime=bool(
                kwargs.pop(
                    "attention_force_single_model_runtime",
                    sparse_attn_cfg.get("force_single_model_runtime", True),
                )
            ),
        )
        model.set_sparse_attention_mode(
            strict_fully_sparse=bool(kwargs.pop("attention_strict_fully_sparse", sparse_attn_cfg.get("strict_fully_sparse", False)))
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
