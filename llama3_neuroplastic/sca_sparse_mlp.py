from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .sca_sparse_config import SCASparseConfig
    from .triton_sparse_mlp import (
        linear_has_4bit_weight,
        materialize_linear_4bit_params,
        materialize_linear_bias,
        materialize_linear_weight,
        triton_sparse_input_linear,
        triton_sparse_input_linear_4bit,
        triton_sparse_mlp_available,
        triton_sparse_output_linear,
        triton_sparse_output_linear_4bit,
    )
except ImportError:
    from sca_sparse_config import SCASparseConfig
    from triton_sparse_mlp import (
        linear_has_4bit_weight,
        materialize_linear_4bit_params,
        materialize_linear_bias,
        materialize_linear_weight,
        triton_sparse_input_linear,
        triton_sparse_input_linear_4bit,
        triton_sparse_mlp_available,
        triton_sparse_output_linear,
        triton_sparse_output_linear_4bit,
    )


@dataclass
class SparseRouteSelection:
    active_idx: torch.Tensor
    score_weights: Optional[torch.Tensor] = None
    latent_idx: Optional[torch.Tensor] = None
    latent_weights: Optional[torch.Tensor] = None
    block_scores: Optional[torch.Tensor] = None


@dataclass
class SparseMLPDiagnostics:
    layer_idx: int
    rows: int
    mean_active_blocks: float
    unique_active_blocks: int
    touched_weight_fraction: float
    estimated_bytes_fetched_per_token: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class SparseLlamaMLP(nn.Module):
    def __init__(
        self,
        base_mlp: nn.Module,
        config: SCASparseConfig,
        layer_idx: int,
        route_fn: Callable[[torch.Tensor, int], SparseRouteSelection],
        enabled_fn: Callable[[int], bool],
        output_scale_fn: Optional[Callable[[int, torch.device, torch.dtype], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.base_mlp = base_mlp
        self.config = config
        self.layer_idx = int(layer_idx)
        self.route_fn = route_fn
        self.enabled_fn = enabled_fn
        self.output_scale_fn = output_scale_fn

        self.hidden_size = int(config.hidden_size)
        self.block_size = int(config.block_size)
        self.num_blocks = int(config.num_blocks)
        self._triton_linear_cache: dict[str, dict[str, object]] = {}
        self._last_diagnostics: Optional[SparseMLPDiagnostics] = None
        self._block_bank: Optional[Dict[str, Any]] = None
        self.capture_alignment: bool = False
        self._last_alignment: Optional[Dict[str, torch.Tensor]] = None
        self._last_fallback_triggered: bool = False
        self._fallback_total_steps: int = 0
        self._fallback_triggered_steps: int = 0
        self.capture_route_snapshot: bool = False
        self._last_route_snapshot: Optional[Dict[str, torch.Tensor]] = None
        self._last_learned_basis_stats: Optional[Dict[str, Any]] = None
        self.curriculum_enabled: bool = False
        self.curriculum_alpha: float = 1.0
        self.block_banking_enabled: bool = False
        self.block_banking_low_usage_threshold: float = 0.001
        self.block_banking_vote_threshold: int = 3
        self.block_banking_cooldown_steps: int = 64
        self.block_banking_max_fraction: float = 0.25
        self.block_banking_usage_ema_decay: float = 0.95
        self._bank_step: int = 0
        self.register_buffer("_block_usage_ema", torch.zeros(self.num_blocks, dtype=torch.float32), persistent=False)
        self.register_buffer("_block_low_usage_votes", torch.zeros(self.num_blocks, dtype=torch.int32), persistent=False)
        self.register_buffer("_block_banked_until", torch.zeros(self.num_blocks, dtype=torch.int32), persistent=False)
        self.sparse_basis_encoder = nn.Linear(self.hidden_size, int(config.basis_rank), bias=True)
        self.sparse_basis_decoder = nn.Parameter(torch.empty(self.num_blocks, int(config.basis_rank), self.block_size))
        self.sparse_basis_bias = nn.Parameter(torch.zeros(self.num_blocks, self.block_size))
        self.sparse_output_compensation_bias = nn.Parameter(torch.zeros(self.hidden_size))
        self.sparse_basis_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self._reset_sparse_basis_parameters()

    def _reset_sparse_basis_parameters(self) -> None:
        nn.init.xavier_uniform_(self.sparse_basis_encoder.weight)
        nn.init.zeros_(self.sparse_basis_encoder.bias)
        nn.init.xavier_uniform_(self.sparse_basis_decoder)
        nn.init.zeros_(self.sparse_basis_bias)
        nn.init.zeros_(self.sparse_output_compensation_bias)
        with torch.no_grad():
            self.sparse_basis_scale.fill_(1.0)

    def iter_sparse_basis_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [
            ("sparse_basis_encoder.weight", self.sparse_basis_encoder.weight),
            ("sparse_basis_encoder.bias", self.sparse_basis_encoder.bias),
            ("sparse_basis_decoder", self.sparse_basis_decoder),
            ("sparse_basis_bias", self.sparse_basis_bias),
            ("sparse_output_compensation_bias", self.sparse_output_compensation_bias),
            ("sparse_basis_scale", self.sparse_basis_scale),
        ]

    def export_sparse_recalibration_state(self) -> Dict[str, torch.Tensor]:
        return {
            name: param.detach().cpu()
            for name, param in self.iter_sparse_basis_parameters()
        }

    def load_sparse_recalibration_state(self, payload: Dict[str, torch.Tensor], strict: bool = True) -> Dict[str, int]:
        loaded = 0
        missing = 0
        for name, param in self.iter_sparse_basis_parameters():
            tensor = payload.get(name)
            if tensor is None:
                if name == "sparse_output_compensation_bias":
                    param.data.zero_()
                    missing += 1
                    continue
                if strict:
                    raise RuntimeError(f"Sparse recalibration state missing '{name}'")
                missing += 1
                continue
            param.data = tensor.to(device=param.device, dtype=param.dtype)
            loaded += 1
        return {"loaded_items": int(loaded), "missing_items": int(missing)}

    def initialize_sparse_basis_from_dense_init(self, payload: Dict[str, torch.Tensor], strict: bool = True) -> Dict[str, int]:
        expected = {
            "encoder_weight": self.sparse_basis_encoder.weight,
            "encoder_bias": self.sparse_basis_encoder.bias,
            "decoder_blocks": self.sparse_basis_decoder,
            "decoder_bias": self.sparse_basis_bias,
        }
        loaded = 0
        missing = 0
        for key, param in expected.items():
            tensor = payload.get(key)
            if tensor is None:
                if strict:
                    raise RuntimeError(f"Dense-init payload missing '{key}' for layer {self.layer_idx}")
                missing += 1
                continue
            if tuple(tensor.shape) != tuple(param.shape):
                if strict:
                    raise RuntimeError(
                        f"Dense-init payload shape mismatch for '{key}' at layer {self.layer_idx}: "
                        f"expected {tuple(param.shape)}, got {tuple(tensor.shape)}"
                    )
                missing += 1
                continue
            param.data = tensor.to(device=param.device, dtype=param.dtype)
            loaded += 1
        scale = payload.get("scale")
        if scale is not None:
            self.sparse_basis_scale.data = scale.to(
                device=self.sparse_basis_scale.device,
                dtype=self.sparse_basis_scale.dtype,
            )
            loaded += 1
        else:
            with torch.no_grad():
                self.sparse_basis_scale.fill_(1.0)
        with torch.no_grad():
            self.sparse_output_compensation_bias.zero_()
        return {"loaded_items": int(loaded), "missing_items": int(missing)}

    def get_last_learned_basis_stats(self) -> Optional[Dict[str, float]]:
        if self._last_learned_basis_stats is None:
            return None
        return {
            str(key): float(value)
            for key, value in self._last_learned_basis_stats.items()
            if not str(key).startswith("_") and not torch.is_tensor(value)
        }

    def get_last_learned_basis_regularizer_tensors(self) -> Optional[Dict[str, torch.Tensor]]:
        if self._last_learned_basis_stats is None:
            return None
        out: Dict[str, torch.Tensor] = {}
        for key, value in self._last_learned_basis_stats.items():
            if torch.is_tensor(value):
                out[str(key)] = value
        return out or None

    def _effective_basis_rank(self) -> int:
        if hasattr(self.config, "basis_rank_for_layer"):
            return int(self.config.basis_rank_for_layer(self.layer_idx))
        return int(getattr(self.config, "basis_rank", self.sparse_basis_encoder.out_features))

    def _effective_basis_top_k(self) -> int:
        if hasattr(self.config, "basis_top_k_for_layer"):
            return int(self.config.basis_top_k_for_layer(self.layer_idx))
        return int(getattr(self.config, "basis_top_k", 1))

    def _effective_block_top_k(self) -> int:
        if hasattr(self.config, "top_k_for_layer"):
            return int(self.config.top_k_for_layer(self.layer_idx))
        return int(getattr(self.config, "route_top_k", getattr(self.config, "top_k", 1)))

    @staticmethod
    def _normalize_selected_weights(active_idx: torch.Tensor, active_score: torch.Tensor) -> torch.Tensor:
        valid_mask = active_idx >= 0
        neg_inf = torch.full_like(active_score, float("-inf"))
        masked = torch.where(valid_mask, active_score, neg_inf)
        weights = torch.softmax(masked, dim=-1)
        return torch.where(valid_mask, weights, torch.zeros_like(weights))

    def _apply_adaptive_block_top_k(
        self,
        active_idx: torch.Tensor,
        active_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(getattr(self.config, "adaptive_top_k", False)):
            return active_idx, active_score
        if active_idx.numel() == 0:
            return active_idx, active_score
        valid = active_idx >= 0
        if not torch.any(valid):
            return active_idx, active_score

        keep = valid.clone()
        best = active_score[:, :1].clamp_min(1e-8)
        relative = active_score / best
        keep = keep & (relative >= float(getattr(self.config, "adaptive_top_k_min_score_ratio", 0.15)))

        min_k = int(
            min(
                getattr(self.config, "adaptive_top_k_min", active_idx.shape[1]),
                active_idx.shape[1],
            )
        )
        if min_k > 0:
            keep[:, :min_k] = valid[:, :min_k]

        pruned_idx = active_idx.masked_fill(~keep, -1)
        pruned_score = active_score.masked_fill(~keep, 0.0)
        return pruned_idx, pruned_score

    def _compute_latent_dense(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        rows = int(hidden_states.shape[0] * hidden_states.shape[1])
        flat_hidden = hidden_states.reshape(rows, self.hidden_size)
        latent_dense = F.silu(self.sparse_basis_encoder(flat_hidden.to(dtype=self.sparse_basis_encoder.weight.dtype)))
        effective_rank = int(min(max(self._effective_basis_rank(), 1), latent_dense.shape[-1]))
        return latent_dense[:, :effective_rank], effective_rank

    def _select_latent_support(
        self,
        latent_dense: torch.Tensor,
        *,
        preset_idx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = int(latent_dense.shape[0])
        effective_top_k = int(min(max(self._effective_basis_top_k(), 1), latent_dense.shape[-1]))
        latent_abs = latent_dense.abs()
        if preset_idx is not None:
            topk_idx = preset_idx.to(device=latent_dense.device, dtype=torch.long)
            if topk_idx.shape[0] != rows:
                raise RuntimeError("preset latent_idx rows must match latent rows")
            if topk_idx.numel() > 0:
                topk_idx = topk_idx.clamp(min=0, max=max(latent_dense.shape[-1] - 1, 0))
            latent_mask = torch.zeros_like(latent_dense)
            if topk_idx.numel() > 0:
                latent_mask.scatter_(1, topk_idx, 1.0)
        elif effective_top_k < int(latent_dense.shape[-1]):
            topk_idx = torch.topk(latent_abs, k=effective_top_k, dim=-1).indices
            latent_mask = torch.zeros_like(latent_dense)
            latent_mask.scatter_(1, topk_idx, 1.0)
        else:
            topk_idx = torch.arange(
                latent_dense.shape[-1],
                device=latent_dense.device,
                dtype=torch.long,
            ).view(1, -1).expand(rows, -1)
            latent_mask = torch.ones_like(latent_dense)
        latent = latent_dense * latent_mask
        selected_abs = torch.gather(latent_abs, 1, topk_idx) if topk_idx.numel() > 0 else torch.empty_like(topk_idx, dtype=latent_abs.dtype)
        denom = selected_abs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        latent_weights = selected_abs / denom if selected_abs.numel() > 0 else selected_abs
        return latent, latent_mask, topk_idx, latent_weights

    def _route_learned_basis_semantic(self, hidden_states: torch.Tensor) -> SparseRouteSelection:
        latent_dense, effective_rank = self._compute_latent_dense(hidden_states)
        latent, _latent_mask, latent_idx, latent_weights = self._select_latent_support(latent_dense)
        decoder = self.sparse_basis_decoder[:, :effective_rank, :].to(device=hidden_states.device, dtype=latent.dtype)
        decoder_norm = decoder.norm(dim=-1)
        if bool(getattr(self.config, "semantic_block_score_normalized", False)):
            decoder_norm = F.normalize(decoder_norm, p=2.0, dim=0, eps=1e-6)
        block_scores = latent.abs() @ decoder_norm.transpose(0, 1)
        if self.block_banking_enabled:
            banked_mask = (self._block_banked_until > int(self._bank_step)).to(device=block_scores.device)
            if bool(torch.any(banked_mask)):
                available = int((~banked_mask).sum().item())
                if available > 0:
                    block_scores = block_scores.masked_fill(banked_mask.unsqueeze(0), float("-inf"))

        block_top_k = int(min(max(self._effective_block_top_k(), 1), block_scores.shape[-1]))
        active_score, active_idx = torch.topk(block_scores, k=block_top_k, dim=-1, largest=True, sorted=True)
        active_idx, active_score = self._apply_adaptive_block_top_k(active_idx=active_idx, active_score=active_score)
        score_weights = self._normalize_selected_weights(active_idx=active_idx, active_score=active_score)
        return SparseRouteSelection(
            active_idx=active_idx.long(),
            score_weights=score_weights,
            latent_idx=latent_idx.long(),
            latent_weights=latent_weights.to(dtype=torch.float32),
            block_scores=block_scores.to(dtype=torch.float32),
        )

    def _update_block_banking(self, active_idx: torch.Tensor) -> None:
        if not self.block_banking_enabled:
            return
        self._bank_step += 1
        rows = int(max(active_idx.shape[0], 1))
        valid = active_idx >= 0
        if not torch.any(valid):
            usage = torch.zeros_like(self._block_usage_ema)
        else:
            blocks = active_idx[valid].to(dtype=torch.long)
            usage = torch.zeros_like(self._block_usage_ema)
            usage.scatter_add_(0, blocks, torch.ones_like(blocks, dtype=torch.float32))
            usage = usage / float(rows)
        decay = float(self.block_banking_usage_ema_decay)
        self._block_usage_ema.mul_(decay).add_(usage * (1.0 - decay))

        in_cooldown = self._block_banked_until > int(self._bank_step)
        low_usage = self._block_usage_ema < float(self.block_banking_low_usage_threshold)
        vote_mask = low_usage & (~in_cooldown)
        self._block_low_usage_votes = torch.where(
            vote_mask,
            self._block_low_usage_votes + 1,
            torch.zeros_like(self._block_low_usage_votes),
        )

        max_banked = int(max(0, min(self.num_blocks - 1, int(round(self.num_blocks * self.block_banking_max_fraction)))))
        currently_banked = int(in_cooldown.sum().item())
        slots_left = int(max(0, max_banked - currently_banked))
        if slots_left <= 0:
            return

        can_bank = self._block_low_usage_votes >= int(self.block_banking_vote_threshold)
        if not torch.any(can_bank):
            return
        active_blocks = torch.unique(active_idx[valid].to(dtype=torch.long)) if torch.any(valid) else torch.empty(0, dtype=torch.long, device=active_idx.device)
        if active_blocks.numel() > 0:
            can_bank[active_blocks] = False
        candidates = torch.nonzero(can_bank, as_tuple=False).flatten()
        if candidates.numel() == 0:
            return
        chosen = candidates[:slots_left]
        self._block_banked_until[chosen] = int(self._bank_step + self.block_banking_cooldown_steps)
        self._block_low_usage_votes[chosen] = 0

    def _apply_output_scale(self, out: torch.Tensor) -> torch.Tensor:
        if self.output_scale_fn is None:
            return out
        scale = self.output_scale_fn(self.layer_idx, out.device, out.dtype)
        if not torch.is_tensor(scale):
            return out
        if scale.ndim == 0:
            return out * scale
        return out * scale.reshape(1, 1, 1)

    def _apply_curriculum_blend(self, out: torch.Tensor, dense_ref: Optional[torch.Tensor]) -> torch.Tensor:
        if dense_ref is None or not self.curriculum_enabled:
            return out
        alpha = float(max(0.0, min(1.0, self.curriculum_alpha)))
        if alpha >= 1.0:
            return out
        dense = dense_ref.to(device=out.device, dtype=out.dtype)
        return (dense * (1.0 - alpha)) + (out * alpha)

    @staticmethod
    def _storage_ptr(param: Optional[torch.Tensor]) -> int:
        if torch.is_tensor(param):
            return int(param.data_ptr())
        return -1

    @staticmethod
    def _storage_version(param: Optional[torch.Tensor]) -> int:
        if torch.is_tensor(param):
            return int(getattr(param, "_version", 0))
        return -1

    def _resolve_triton_weight_dtype(self, hidden_dtype: torch.dtype, device: torch.device) -> torch.dtype:
        if device.type == "cuda":
            index = device.index if device.index is not None else torch.cuda.current_device()
            major, _minor = torch.cuda.get_device_capability(index)
            if major < 8 and hidden_dtype == torch.bfloat16:
                return torch.float16
        if hidden_dtype in (torch.float16, torch.bfloat16, torch.float32):
            return hidden_dtype
        return torch.float16

    def _get_cached_triton_linear_params(
        self,
        cache_key: str,
        linear: nn.Module,
        target_device: torch.device,
        target_weight_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        weight = getattr(linear, "weight", None)
        bias = getattr(linear, "bias", None)
        weight_ptr = self._storage_ptr(weight)
        bias_ptr = self._storage_ptr(bias)
        weight_version = self._storage_version(weight)
        bias_version = self._storage_version(bias)
        cached = self._triton_linear_cache.get(cache_key)

        if cached is not None:
            cached_weight = cached["weight"]
            cached_bias = cached["bias"]
            if (
                int(cached["weight_ptr"]) == weight_ptr
                and int(cached["bias_ptr"]) == bias_ptr
                and int(cached["weight_version"]) == weight_version
                and int(cached["bias_version"]) == bias_version
                and cached_weight.device == target_device
                and cached_weight.dtype == target_weight_dtype
            ):
                return cached_weight, cached_bias

        mat_weight = materialize_linear_weight(
            linear,
            device=target_device,
            dtype=target_weight_dtype,
        )
        mat_bias = materialize_linear_bias(
            linear,
            device=target_device,
            dtype=target_weight_dtype,
        )
        self._triton_linear_cache[cache_key] = {
            "weight": mat_weight,
            "bias": mat_bias,
            "weight_ptr": weight_ptr,
            "bias_ptr": bias_ptr,
            "weight_version": weight_version,
            "bias_version": bias_version,
        }
        return mat_weight, mat_bias

    def _get_cached_triton_4bit_linear_params(
        self,
        cache_key: str,
        linear: nn.Module,
        target_device: torch.device,
        target_bias_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, str, Optional[torch.Tensor]]:
        weight = getattr(linear, "weight", None)
        bias = getattr(linear, "bias", None)
        quant_state = getattr(weight, "quant_state", None)
        absmax_tensor = getattr(quant_state, "absmax", None) if quant_state is not None else None
        code_tensor = getattr(quant_state, "code", None) if quant_state is not None else None
        weight_ptr = self._storage_ptr(weight)
        bias_ptr = self._storage_ptr(bias)
        absmax_ptr = self._storage_ptr(absmax_tensor)
        code_ptr = self._storage_ptr(code_tensor)
        weight_version = self._storage_version(weight)
        bias_version = self._storage_version(bias)
        absmax_version = self._storage_version(absmax_tensor)
        code_version = self._storage_version(code_tensor)
        cached = self._triton_linear_cache.get(cache_key)

        if cached is not None:
            packed_weight = cached["packed_weight"]
            absmax = cached["absmax"]
            code = cached["code"]
            cached_bias = cached["bias"]
            if (
                int(cached["weight_ptr"]) == weight_ptr
                and int(cached["bias_ptr"]) == bias_ptr
                and int(cached["absmax_ptr"]) == absmax_ptr
                and int(cached["code_ptr"]) == code_ptr
                and int(cached["weight_version"]) == weight_version
                and int(cached["bias_version"]) == bias_version
                and int(cached["absmax_version"]) == absmax_version
                and int(cached["code_version"]) == code_version
                and packed_weight.device == target_device
                and absmax.device == target_device
                and code.device == target_device
                and (cached_bias is None or cached_bias.dtype == target_bias_dtype)
            ):
                return (
                    packed_weight,
                    absmax,
                    code,
                    int(cached["out_features"]),
                    int(cached["in_features"]),
                    int(cached["quant_block_size"]),
                    str(cached["quant_type"]),
                    cached_bias,
                )

        packed_weight, absmax, code, out_features, in_features, quant_block_size, quant_type = materialize_linear_4bit_params(
            linear,
            device=target_device,
        )
        mat_bias = materialize_linear_bias(
            linear,
            device=target_device,
            dtype=target_bias_dtype,
        )
        self._triton_linear_cache[cache_key] = {
            "packed_weight": packed_weight,
            "absmax": absmax,
            "code": code,
            "bias": mat_bias,
            "out_features": int(out_features),
            "in_features": int(in_features),
            "quant_block_size": int(quant_block_size),
            "quant_type": str(quant_type),
            "weight_ptr": weight_ptr,
            "bias_ptr": bias_ptr,
            "absmax_ptr": absmax_ptr,
            "code_ptr": code_ptr,
            "weight_version": weight_version,
            "bias_version": bias_version,
            "absmax_version": absmax_version,
            "code_version": code_version,
        }
        return packed_weight, absmax, code, out_features, in_features, quant_block_size, quant_type, mat_bias

    def _build_feature_mask(
        self,
        active_idx: torch.Tensor,
        score_weights: Optional[torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        rows = int(active_idx.shape[0])
        mask_blocks = torch.zeros((rows, self.num_blocks), device=device, dtype=dtype)
        row_idx = torch.arange(rows, device=device)

        for slot in range(active_idx.shape[1]):
            block_idx = active_idx[:, slot]
            valid = block_idx >= 0
            if not torch.any(valid):
                continue

            rows_valid = row_idx[valid]
            blocks_valid = block_idx[valid]
            if score_weights is None:
                values = torch.ones((rows_valid.shape[0],), device=device, dtype=dtype)
            else:
                values = score_weights[valid, slot].to(device=device, dtype=dtype) * float(active_idx.shape[1])
            mask_blocks[rows_valid, blocks_valid] += values

        if score_weights is None:
            mask_blocks.clamp_(max=1.0)

        mask = (
            mask_blocks.unsqueeze(-1)
            .expand(rows, self.num_blocks, self.block_size)
            .reshape(rows, self.hidden_size)
        )
        return mask

    @staticmethod
    def _linear_io_features(linear: nn.Module) -> tuple[int, int]:
        out_features = getattr(linear, "out_features", None)
        in_features = getattr(linear, "in_features", None)
        weight = getattr(linear, "weight", None)
        quant_state = getattr(weight, "quant_state", None)
        quant_shape = tuple(getattr(quant_state, "shape", ())) if quant_state is not None else ()
        if out_features is None and len(quant_shape) == 2:
            out_features = int(quant_shape[0])
        if in_features is None and len(quant_shape) == 2:
            in_features = int(quant_shape[1])
        if out_features is None and torch.is_tensor(weight) and weight.ndim == 2:
            out_features = int(weight.shape[0])
        if in_features is None and torch.is_tensor(weight) and weight.ndim == 2:
            in_features = int(weight.shape[1])
        if out_features is None or in_features is None:
            raise RuntimeError(f"Unable to infer linear shape for {type(linear).__name__}")
        return int(out_features), int(in_features)

    @staticmethod
    def _linear_weight_bytes_per_scalar(linear: nn.Module) -> float:
        weight = getattr(linear, "weight", None)
        if torch.is_tensor(weight):
            if getattr(weight, "quant_state", None) is not None:
                return 0.5
            return float(torch.tensor([], dtype=weight.dtype).element_size())
        return 2.0

    def _update_diagnostics(self, active_idx: torch.Tensor) -> None:
        valid = active_idx >= 0
        rows = int(active_idx.shape[0])
        if rows <= 0:
            self._last_diagnostics = SparseMLPDiagnostics(
                layer_idx=self.layer_idx,
                rows=0,
                mean_active_blocks=0.0,
                unique_active_blocks=0,
                touched_weight_fraction=0.0,
                estimated_bytes_fetched_per_token=0.0,
            )
            return

        active_counts = valid.sum(dim=-1).to(dtype=torch.float32)
        mean_active_blocks = float(active_counts.mean().item())
        unique_active_blocks = int(torch.unique(active_idx[valid]).numel()) if torch.any(valid) else 0
        touched_fraction = float(mean_active_blocks / max(self.num_blocks, 1))

        if self._block_bank is None:
            gate_out, _gate_in = self._linear_io_features(self.base_mlp.gate_proj)
            _up_out, _up_in = self._linear_io_features(self.base_mlp.up_proj)
            _down_out, down_in = self._linear_io_features(self.base_mlp.down_proj)
            gate_bytes = self._linear_weight_bytes_per_scalar(self.base_mlp.gate_proj)
            up_bytes = self._linear_weight_bytes_per_scalar(self.base_mlp.up_proj)
            down_bytes = self._linear_weight_bytes_per_scalar(self.base_mlp.down_proj)
        else:
            gate_blocks = self._block_bank["gate_proj_blocks"]
            down_blocks = self._block_bank["down_proj_blocks"]
            gate_out = int(gate_blocks.shape[1])
            down_in = int(down_blocks.shape[2])
            gate_bytes = float(torch.tensor([], dtype=gate_blocks.dtype).element_size())
            up_bytes = float(torch.tensor([], dtype=self._block_bank["up_proj_blocks"].dtype).element_size())
            down_bytes = float(torch.tensor([], dtype=down_blocks.dtype).element_size())
        bytes_per_active_block = (
            float(gate_out * self.block_size) * gate_bytes
            + float(gate_out * self.block_size) * up_bytes
            + float(down_in * self.block_size) * down_bytes
        )
        est_bytes_per_token = float(mean_active_blocks * bytes_per_active_block)

        self._last_diagnostics = SparseMLPDiagnostics(
            layer_idx=self.layer_idx,
            rows=rows,
            mean_active_blocks=mean_active_blocks,
            unique_active_blocks=unique_active_blocks,
            touched_weight_fraction=touched_fraction,
            estimated_bytes_fetched_per_token=est_bytes_per_token,
        )

    def get_last_diagnostics(self) -> Optional[SparseMLPDiagnostics]:
        return self._last_diagnostics

    def set_alignment_capture(self, enabled: bool) -> None:
        self.capture_alignment = bool(enabled)
        if not self.capture_alignment:
            self._last_alignment = None
            self._last_fallback_triggered = False

    def get_last_alignment(self) -> Optional[Dict[str, torch.Tensor]]:
        return self._last_alignment

    def set_route_capture(self, enabled: bool) -> None:
        self.capture_route_snapshot = bool(enabled)
        if not self.capture_route_snapshot:
            self._last_route_snapshot = None

    def set_dense_sparse_curriculum(self, enabled: bool, alpha: float) -> None:
        self.curriculum_enabled = bool(enabled)
        self.curriculum_alpha = float(max(0.0, min(1.0, alpha)))

    def configure_block_banking(
        self,
        *,
        enabled: bool,
        low_usage_threshold: float = 0.001,
        vote_threshold: int = 3,
        cooldown_steps: int = 64,
        max_fraction: float = 0.25,
        usage_ema_decay: float = 0.95,
    ) -> None:
        self.block_banking_enabled = bool(enabled)
        self.block_banking_low_usage_threshold = float(max(0.0, low_usage_threshold))
        self.block_banking_vote_threshold = int(max(1, vote_threshold))
        self.block_banking_cooldown_steps = int(max(1, cooldown_steps))
        self.block_banking_max_fraction = float(min(max(max_fraction, 0.0), 0.95))
        self.block_banking_usage_ema_decay = float(min(max(usage_ema_decay, 0.0), 0.999))
        if not self.block_banking_enabled:
            self._bank_step = 0
            self._block_usage_ema.zero_()
            self._block_low_usage_votes.zero_()
            self._block_banked_until.zero_()

    def get_block_banking_stats(self) -> Dict[str, float]:
        banked = (self._block_banked_until > int(self._bank_step)).float()
        return {
            "bank_step": float(self._bank_step),
            "banked_fraction": float(banked.mean().detach().cpu().item()),
            "usage_ema_mean": float(self._block_usage_ema.mean().detach().cpu().item()),
            "usage_ema_max": float(self._block_usage_ema.max().detach().cpu().item()) if self._block_usage_ema.numel() > 0 else 0.0,
        }

    def get_last_route_snapshot(self) -> Optional[Dict[str, torch.Tensor]]:
        return self._last_route_snapshot

    def get_fallback_stats(self) -> Dict[str, float]:
        total = int(self._fallback_total_steps)
        triggered = int(self._fallback_triggered_steps)
        return {
            "total_steps": float(total),
            "triggered_steps": float(triggered),
            "dense_fallback_rate": float(triggered / max(total, 1)),
            "last_fallback_triggered": float(1.0 if self._last_fallback_triggered else 0.0),
        }

    def has_block_bank(self) -> bool:
        return self._block_bank is not None

    def clear_block_bank(self) -> None:
        self._block_bank = None

    def load_block_bank(self, payload: Dict[str, Any], strict: bool = True) -> None:
        gate_blocks = payload.get("gate_proj_blocks")
        up_blocks = payload.get("up_proj_blocks")
        down_blocks = payload.get("down_proj_blocks")
        if not torch.is_tensor(gate_blocks) or not torch.is_tensor(up_blocks) or not torch.is_tensor(down_blocks):
            raise RuntimeError("Sparse block bank payload must contain gate_proj_blocks, up_proj_blocks, and down_proj_blocks")

        gate_blocks = gate_blocks.detach().cpu().contiguous()
        up_blocks = up_blocks.detach().cpu().contiguous()
        down_blocks = down_blocks.detach().cpu().contiguous()
        if gate_blocks.ndim != 3 or up_blocks.ndim != 3 or down_blocks.ndim != 3:
            raise RuntimeError("Sparse block bank tensors must be rank-3")

        num_blocks = int(payload.get("num_blocks", gate_blocks.shape[0]))
        block_size = int(payload.get("block_size", gate_blocks.shape[2]))
        if strict and num_blocks != self.num_blocks:
            raise RuntimeError(f"Bank num_blocks mismatch: expected {self.num_blocks}, got {num_blocks}")
        if strict and block_size != self.block_size:
            raise RuntimeError(f"Bank block_size mismatch: expected {self.block_size}, got {block_size}")
        if strict and int(gate_blocks.shape[0]) != num_blocks:
            raise RuntimeError("gate_proj_blocks first dimension must equal num_blocks")
        if strict and int(up_blocks.shape[0]) != num_blocks:
            raise RuntimeError("up_proj_blocks first dimension must equal num_blocks")
        if strict and int(down_blocks.shape[0]) != num_blocks:
            raise RuntimeError("down_proj_blocks first dimension must equal num_blocks")
        if strict and int(gate_blocks.shape[2]) != block_size:
            raise RuntimeError("gate_proj_blocks last dimension must equal block_size")
        if strict and int(up_blocks.shape[2]) != block_size:
            raise RuntimeError("up_proj_blocks last dimension must equal block_size")
        if strict and int(down_blocks.shape[1]) != block_size:
            raise RuntimeError("down_proj_blocks middle dimension must equal block_size")

        intermediate_size = int(gate_blocks.shape[1])
        if strict and int(up_blocks.shape[1]) != intermediate_size:
            raise RuntimeError("up_proj_blocks intermediate dimension mismatch")
        if strict and int(down_blocks.shape[2]) != intermediate_size:
            raise RuntimeError("down_proj_blocks input dimension mismatch")

        gate_bias = payload.get("gate_bias")
        up_bias = payload.get("up_bias")
        down_bias = payload.get("down_bias")
        if gate_bias is not None and not torch.is_tensor(gate_bias):
            raise RuntimeError("gate_bias must be a tensor or None")
        if up_bias is not None and not torch.is_tensor(up_bias):
            raise RuntimeError("up_bias must be a tensor or None")
        if down_bias is not None and not torch.is_tensor(down_bias):
            raise RuntimeError("down_bias must be a tensor or None")

        gate_bias = gate_bias.detach().cpu().contiguous() if torch.is_tensor(gate_bias) else None
        up_bias = up_bias.detach().cpu().contiguous() if torch.is_tensor(up_bias) else None
        down_bias = down_bias.detach().cpu().contiguous() if torch.is_tensor(down_bias) else None
        if strict and gate_bias is not None and int(gate_bias.shape[0]) != intermediate_size:
            raise RuntimeError("gate_bias size mismatch")
        if strict and up_bias is not None and int(up_bias.shape[0]) != intermediate_size:
            raise RuntimeError("up_bias size mismatch")
        if strict and down_bias is not None and int(down_bias.shape[0]) != self.hidden_size:
            raise RuntimeError("down_bias size mismatch")

        self._block_bank = {
            "num_blocks": int(num_blocks),
            "block_size": int(block_size),
            "intermediate_size": int(intermediate_size),
            "gate_proj_blocks": gate_blocks,
            "up_proj_blocks": up_blocks,
            "down_proj_blocks": down_blocks,
            "gate_bias": gate_bias,
            "up_bias": up_bias,
            "down_bias": down_bias,
        }

    @staticmethod
    def _linear_has_dense_weight(linear: nn.Module) -> bool:
        weight = getattr(linear, "weight", None)
        return bool(torch.is_tensor(weight) and weight.dim() == 2 and weight.is_floating_point())

    @staticmethod
    def _linear_has_4bit_weight(linear: nn.Module) -> bool:
        return bool(linear_has_4bit_weight(linear))

    def _can_use_torch_block_sparse(self) -> bool:
        return (
            self._linear_has_dense_weight(self.base_mlp.gate_proj)
            and self._linear_has_dense_weight(self.base_mlp.up_proj)
            and self._linear_has_dense_weight(self.base_mlp.down_proj)
        )

    def _can_use_triton_4bit(self) -> bool:
        return (
            self._linear_has_4bit_weight(self.base_mlp.gate_proj)
            and self._linear_has_4bit_weight(self.base_mlp.up_proj)
            and self._linear_has_4bit_weight(self.base_mlp.down_proj)
        )

    def _can_use_triton_sparse(self) -> bool:
        return self._can_use_torch_block_sparse() or self._can_use_triton_4bit()

    def _triton_4bit_trainable_weights(self) -> bool:
        if not self._can_use_triton_4bit():
            return False
        weights = [
            getattr(self.base_mlp.gate_proj, "weight", None),
            getattr(self.base_mlp.up_proj, "weight", None),
            getattr(self.base_mlp.down_proj, "weight", None),
        ]
        return any(bool(torch.is_tensor(w) and getattr(w, "requires_grad", False)) for w in weights)

    def _project_input_blocks(self, x_flat: torch.Tensor, linear: nn.Module, active_idx: torch.Tensor) -> torch.Tensor:
        rows = int(x_flat.shape[0])
        out_features = int(linear.weight.shape[0])
        out = torch.zeros((rows, out_features), device=x_flat.device, dtype=x_flat.dtype)
        if getattr(linear, "bias", None) is not None:
            out += linear.bias.to(device=x_flat.device, dtype=x_flat.dtype).unsqueeze(0)

        x_blocks = x_flat.view(rows, self.num_blocks, self.block_size)
        weight = linear.weight.to(device=x_flat.device, dtype=x_flat.dtype).contiguous()
        weight_blocks = weight.view(out_features, self.num_blocks, self.block_size)
        row_idx = torch.arange(rows, device=x_flat.device)

        for slot in range(active_idx.shape[1]):
            block_idx = active_idx[:, slot]
            valid = block_idx >= 0
            if not torch.any(valid):
                continue

            rows_valid = row_idx[valid]
            blocks_valid = block_idx[valid]
            x_selected = x_blocks[rows_valid, blocks_valid]
            w_selected = weight_blocks[:, blocks_valid, :].permute(1, 0, 2).contiguous()
            out[rows_valid] += torch.einsum("rb,rob->ro", x_selected, w_selected)
        return out

    def _project_output_blocks(
        self,
        x_flat: torch.Tensor,
        linear: nn.Module,
        active_idx: torch.Tensor,
        score_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        rows = int(x_flat.shape[0])
        input_dim = int(x_flat.shape[1])
        out_blocks = torch.zeros(
            (rows, self.num_blocks, self.block_size),
            device=x_flat.device,
            dtype=x_flat.dtype,
        )
        weight = materialize_linear_weight(linear, device=x_flat.device, dtype=x_flat.dtype).contiguous()
        weight_blocks = weight.view(self.num_blocks, self.block_size, input_dim)
        bias_blocks = None
        if getattr(linear, "bias", None) is not None:
            bias = materialize_linear_bias(linear, device=x_flat.device, dtype=x_flat.dtype)
            bias_blocks = bias.view(self.num_blocks, self.block_size)
        row_idx = torch.arange(rows, device=x_flat.device)

        for slot in range(active_idx.shape[1]):
            block_idx = active_idx[:, slot]
            valid = block_idx >= 0
            if not torch.any(valid):
                continue

            rows_valid = row_idx[valid]
            blocks_valid = block_idx[valid]
            x_selected = x_flat[rows_valid]
            w_selected = weight_blocks[blocks_valid]
            contrib = torch.bmm(w_selected, x_selected.unsqueeze(-1)).squeeze(-1)
            if bias_blocks is not None:
                contrib = contrib + bias_blocks[blocks_valid]
            if score_weights is not None:
                slot_weights = score_weights[valid, slot].to(device=contrib.device, dtype=contrib.dtype)
                contrib = contrib * slot_weights.unsqueeze(-1) * float(active_idx.shape[1])
            out_blocks[rows_valid, blocks_valid] += contrib

        return out_blocks.view(rows, self.hidden_size)

    def _forward_dense_masked(
        self,
        hidden_states: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> torch.Tensor:
        masked_hidden = hidden_states * feature_mask
        gate = self.base_mlp.gate_proj(masked_hidden)
        up = self.base_mlp.up_proj(masked_hidden)
        activated = self.base_mlp.act_fn(gate) * up
        down = self.base_mlp.down_proj(activated)
        return down * feature_mask

    def _forward_output_sparse(
        self,
        hidden_states: torch.Tensor,
        active_idx: torch.Tensor,
        score_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        flat_hidden = hidden_states.reshape(-1, self.hidden_size)
        gate = self.base_mlp.gate_proj(flat_hidden)
        up = self.base_mlp.up_proj(flat_hidden)
        activated = self.base_mlp.act_fn(gate) * up
        down = self._project_output_blocks(activated, self.base_mlp.down_proj, active_idx, score_weights=score_weights)
        return down.view_as(hidden_states)

    def _build_intermediate_group_mask(
        self,
        active_idx: torch.Tensor,
        rows: int,
        intermediate_size: int,
        device: torch.device,
        dtype: torch.dtype,
        score_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if intermediate_size % self.num_blocks != 0:
            raise RuntimeError("intermediate_size must be divisible by num_blocks for intermediate_group placement")
        group_size = intermediate_size // self.num_blocks
        mask_groups = torch.zeros((rows, self.num_blocks), device=device, dtype=dtype)
        row_idx = torch.arange(rows, device=device)
        for slot in range(active_idx.shape[1]):
            block_idx = active_idx[:, slot]
            valid = block_idx >= 0
            if not torch.any(valid):
                continue
            if score_weights is not None:
                values = score_weights[valid, slot].to(device=device, dtype=dtype)
                mask_groups[row_idx[valid], block_idx[valid]] += values
            else:
                mask_groups[row_idx[valid], block_idx[valid]] = 1.0
        if score_weights is None:
            mask_groups.clamp_(max=1.0)
        return mask_groups.unsqueeze(-1).expand(rows, self.num_blocks, group_size).reshape(rows, intermediate_size)

    def _forward_intermediate_group_sparse(
        self,
        hidden_states: torch.Tensor,
        active_idx: torch.Tensor,
        score_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        flat_hidden = hidden_states.reshape(-1, self.hidden_size)
        gate = self.base_mlp.gate_proj(flat_hidden)
        up = self.base_mlp.up_proj(flat_hidden)
        activated = self.base_mlp.act_fn(gate) * up
        inter_mask = self._build_intermediate_group_mask(
            active_idx=active_idx,
            rows=int(flat_hidden.shape[0]),
            intermediate_size=int(activated.shape[-1]),
            device=activated.device,
            dtype=activated.dtype,
            score_weights=score_weights if self.config.soft_mask else None,
        )
        activated = activated * inter_mask
        down = self.base_mlp.down_proj(activated)
        return down.view_as(hidden_states)

    def _forward_learned_basis(
        self,
        hidden_states: torch.Tensor,
        route: SparseRouteSelection,
    ) -> torch.Tensor:
        active_idx = route.active_idx
        score_weights = route.score_weights if self.config.soft_mask else None
        rows = int(hidden_states.shape[0] * hidden_states.shape[1])
        latent_dense, effective_rank = self._compute_latent_dense(hidden_states)
        latent, latent_mask, topk_idx, latent_weights = self._select_latent_support(
            latent_dense,
            preset_idx=route.latent_idx,
        )
        out_blocks = torch.zeros(
            (rows, self.num_blocks, self.block_size),
            device=hidden_states.device,
            dtype=latent.dtype,
        )
        decoder = self.sparse_basis_decoder[:, :effective_rank, :].to(device=hidden_states.device, dtype=latent.dtype)
        bias = self.sparse_basis_bias.to(device=hidden_states.device, dtype=latent.dtype)
        row_idx = torch.arange(rows, device=hidden_states.device)
        for slot in range(active_idx.shape[1]):
            block_idx = active_idx[:, slot]
            valid = block_idx >= 0
            if not torch.any(valid):
                continue
            rows_valid = row_idx[valid]
            blocks_valid = block_idx[valid].to(dtype=torch.long)
            coeff = latent[rows_valid]
            decoder_selected = decoder.index_select(0, blocks_valid)
            contrib = torch.bmm(coeff.unsqueeze(1), decoder_selected).squeeze(1) + bias.index_select(0, blocks_valid)
            if score_weights is not None:
                slot_weights = score_weights[valid, slot].to(device=contrib.device, dtype=contrib.dtype)
                contrib = contrib * slot_weights.unsqueeze(-1) * float(active_idx.shape[1])
            out_blocks[rows_valid, blocks_valid] += contrib
        out_flat = out_blocks.view(rows, self.hidden_size)
        out_flat = out_flat * self.sparse_basis_scale.to(device=out_flat.device, dtype=out_flat.dtype)
        if bool(getattr(self.config, "output_compensation_bias_enabled", False)):
            comp_scale = torch.zeros((rows, 1), device=out_flat.device, dtype=out_flat.dtype)
            if route.block_scores is not None and route.block_scores.numel() > 0:
                block_scores = route.block_scores.to(device=out_flat.device, dtype=out_flat.dtype)
                total_mass = block_scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                valid = active_idx >= 0
                gather_idx = active_idx.clamp_min(0)
                selected_mass = torch.gather(block_scores, 1, gather_idx)
                selected_mass = torch.where(
                    valid,
                    selected_mass,
                    torch.zeros_like(selected_mass),
                ).sum(dim=-1, keepdim=True)
                omitted_ratio = (1.0 - (selected_mass / total_mass)).clamp_min(0.0)
                latent_mean_abs = latent.abs().mean(dim=-1, keepdim=True)
                comp_scale = omitted_ratio * latent_mean_abs.to(device=out_flat.device, dtype=out_flat.dtype)
            out_flat = out_flat + (
                comp_scale
                * self.sparse_output_compensation_bias.to(device=out_flat.device, dtype=out_flat.dtype).unsqueeze(0)
            )
        if rows > 0:
            mean_latent_norm = float(latent.norm(dim=-1).mean().detach().cpu().item())
            latent_load = latent_mask.float().mean(dim=0)
            active_fraction = float(latent_load.mean().detach().cpu().item())
            coords_used = float((latent_load > 0).float().sum().detach().cpu().item())
            latent_importance = latent.abs().sum(dim=0)
            latent_importance = latent_importance / latent_importance.sum().clamp_min(1e-6)
            latent_load_probs = latent_load / latent_load.sum().clamp_min(1e-6)
            row_probs = latent.abs() / latent.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
            usage_entropy = float((-(latent_importance * latent_importance.clamp_min(1e-6).log()).sum()).detach().cpu().item())
            support_overlap = float(
                ((latent_mask @ latent_mask.transpose(0, 1)).float() / max(int(topk_idx.shape[1]), 1))
                .mean()
                .detach()
                .cpu()
                .item()
            )
            unique_fraction = 0.0
            if rows > 1 and int(topk_idx.shape[1]) <= 16:
                unique_fraction = float(torch.unique(topk_idx.detach().to(dtype=torch.int64), dim=0).shape[0] / rows)

            valid_blocks = active_idx >= 0
            if score_weights is None:
                block_selected = valid_blocks.to(dtype=latent.dtype)
                denom = block_selected.sum(dim=-1, keepdim=True).clamp_min(1.0)
                block_selected = block_selected / denom
            else:
                block_selected = score_weights.to(device=latent.device, dtype=latent.dtype)
            block_importance = torch.zeros((self.num_blocks,), device=latent.device, dtype=latent.dtype)
            block_load = torch.zeros((self.num_blocks,), device=latent.device, dtype=latent.dtype)
            if torch.any(valid_blocks):
                block_importance.scatter_add_(0, active_idx[valid_blocks], block_selected[valid_blocks])
                block_load.scatter_add_(
                    0,
                    active_idx[valid_blocks],
                    torch.ones_like(active_idx[valid_blocks], dtype=latent.dtype),
                )
            block_importance = block_importance / block_importance.sum().clamp_min(1e-6)
            block_load_probs = block_load / block_load.sum().clamp_min(1e-6)
            self._last_learned_basis_stats = {
                "mean_latent_norm": mean_latent_norm,
                "active_latent_fraction": active_fraction,
                "active_latent_coords_used": coords_used,
                "latent_usage_entropy": usage_entropy,
                "support_overlap_mean": support_overlap,
                "support_unique_fraction": unique_fraction,
                "max_latent_importance_fraction": float(latent_importance.max().detach().cpu().item()),
                "max_latent_load_fraction": float(latent_load_probs.max().detach().cpu().item()),
                "max_block_importance_fraction": float(block_importance.max().detach().cpu().item()),
                "max_block_load_fraction": float(block_load_probs.max().detach().cpu().item()),
                "_coord_probs": latent_importance,
                "_row_probs": row_probs,
                "_latent_importance_probs": latent_importance,
                "_latent_load_probs": latent_load_probs,
                "_block_importance_probs": block_importance,
                "_block_load_probs": block_load_probs,
                "_latent_selected_weights": latent_weights,
            }
        else:
            self._last_learned_basis_stats = {
                "mean_latent_norm": 0.0,
                "active_latent_fraction": 0.0,
                "active_latent_coords_used": 0.0,
                "latent_usage_entropy": 0.0,
                "support_overlap_mean": 0.0,
                "support_unique_fraction": 0.0,
                "max_latent_importance_fraction": 0.0,
                "max_latent_load_fraction": 0.0,
                "max_block_importance_fraction": 0.0,
                "max_block_load_fraction": 0.0,
                "_coord_probs": torch.empty((0,), device=hidden_states.device, dtype=torch.float32),
                "_row_probs": torch.empty((0, 0), device=hidden_states.device, dtype=torch.float32),
                "_latent_importance_probs": torch.empty((0,), device=hidden_states.device, dtype=torch.float32),
                "_latent_load_probs": torch.empty((0,), device=hidden_states.device, dtype=torch.float32),
                "_block_importance_probs": torch.empty((0,), device=hidden_states.device, dtype=torch.float32),
                "_block_load_probs": torch.empty((0,), device=hidden_states.device, dtype=torch.float32),
                "_latent_selected_weights": torch.empty((0, 0), device=hidden_states.device, dtype=torch.float32),
            }
        return out_flat.view_as(hidden_states).to(dtype=hidden_states.dtype)

    def _build_grouped_block_columns(
        self,
        pattern_blocks: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        blocks = pattern_blocks[pattern_blocks >= 0].to(device=device, dtype=torch.long)
        if blocks.numel() == 0:
            return torch.empty((0,), device=device, dtype=torch.long)
        offsets = torch.arange(self.block_size, device=device, dtype=torch.long)
        cols = blocks[:, None] * self.block_size + offsets[None, :]
        return cols.reshape(-1).contiguous()

    def _forward_grouped_row_gemm(
        self,
        hidden_states: torch.Tensor,
        active_idx: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> torch.Tensor:
        flat_hidden = hidden_states.reshape(-1, self.hidden_size)
        flat_mask = feature_mask.reshape(-1, self.hidden_size)
        masked_flat = flat_hidden * flat_mask
        rows = int(masked_flat.shape[0])

        if rows == 0:
            return torch.zeros_like(hidden_states)

        use_dense_weights = self._can_use_torch_block_sparse()
        if not use_dense_weights and not self.config.grouped_row_allow_4bit_dequant:
            raise RuntimeError("grouped_row_gemm requires dense weights or grouped_row_allow_4bit_dequant=true")

        target_weight_dtype = self._resolve_triton_weight_dtype(masked_flat.dtype, masked_flat.device)
        if use_dense_weights:
            gate_weight, gate_bias = self._get_cached_triton_linear_params(
                cache_key="grouped_gate_proj_dense",
                linear=self.base_mlp.gate_proj,
                target_device=masked_flat.device,
                target_weight_dtype=target_weight_dtype,
            )
            up_weight, up_bias = self._get_cached_triton_linear_params(
                cache_key="grouped_up_proj_dense",
                linear=self.base_mlp.up_proj,
                target_device=masked_flat.device,
                target_weight_dtype=target_weight_dtype,
            )
            down_weight, down_bias = self._get_cached_triton_linear_params(
                cache_key="grouped_down_proj_dense",
                linear=self.base_mlp.down_proj,
                target_device=masked_flat.device,
                target_weight_dtype=target_weight_dtype,
            )
        else:
            gate_weight = materialize_linear_weight(
                self.base_mlp.gate_proj,
                device=masked_flat.device,
                dtype=target_weight_dtype,
            )
            gate_bias = materialize_linear_bias(
                self.base_mlp.gate_proj,
                device=masked_flat.device,
                dtype=target_weight_dtype,
            )
            up_weight = materialize_linear_weight(
                self.base_mlp.up_proj,
                device=masked_flat.device,
                dtype=target_weight_dtype,
            )
            up_bias = materialize_linear_bias(
                self.base_mlp.up_proj,
                device=masked_flat.device,
                dtype=target_weight_dtype,
            )
            down_weight = materialize_linear_weight(
                self.base_mlp.down_proj,
                device=masked_flat.device,
                dtype=target_weight_dtype,
            )
            down_bias = materialize_linear_bias(
                self.base_mlp.down_proj,
                device=masked_flat.device,
                dtype=target_weight_dtype,
            )

        out_flat = torch.zeros((rows, self.hidden_size), device=masked_flat.device, dtype=target_weight_dtype)
        unique_patterns, inverse = torch.unique(active_idx.to(dtype=torch.int32), dim=0, return_inverse=True)
        all_rows = torch.arange(rows, device=masked_flat.device, dtype=torch.long)

        for pattern_idx in range(int(unique_patterns.shape[0])):
            row_mask = inverse == pattern_idx
            bucket_rows = all_rows[row_mask]
            if bucket_rows.numel() == 0:
                continue
            pattern = unique_patterns[pattern_idx]
            cols = self._build_grouped_block_columns(pattern, device=masked_flat.device)
            if cols.numel() == 0:
                continue

            if bucket_rows.numel() < int(self.config.grouped_row_min_bucket):
                row_x = masked_flat[bucket_rows]
                row_out = self.base_mlp.down_proj(
                    self.base_mlp.act_fn(self.base_mlp.gate_proj(row_x)) * self.base_mlp.up_proj(row_x)
                )
                row_out = row_out * flat_mask[bucket_rows].to(dtype=row_out.dtype, device=row_out.device)
                out_flat[bucket_rows] = row_out.to(dtype=out_flat.dtype)
                continue

            x_group = masked_flat[bucket_rows][:, cols]
            gate_local = x_group @ gate_weight[:, cols].transpose(0, 1)
            if gate_bias is not None:
                gate_local = gate_local + gate_bias.unsqueeze(0)
            up_local = x_group @ up_weight[:, cols].transpose(0, 1)
            if up_bias is not None:
                up_local = up_local + up_bias.unsqueeze(0)
            activated_local = self.base_mlp.act_fn(gate_local) * up_local
            down_local = activated_local @ down_weight[cols, :].transpose(0, 1)
            if down_bias is not None:
                down_local = down_local + down_bias[cols].unsqueeze(0)
            out_flat[bucket_rows[:, None], cols.unsqueeze(0)] = down_local

        out_flat = out_flat * flat_mask.to(dtype=out_flat.dtype, device=out_flat.device)
        return out_flat.view_as(hidden_states).to(dtype=hidden_states.dtype)

    def _forward_torch_block_sparse(
        self,
        hidden_states: torch.Tensor,
        active_idx: torch.Tensor,
        feature_mask: torch.Tensor,
        score_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        flat_hidden = hidden_states.reshape(-1, self.hidden_size)
        flat_mask = feature_mask.reshape(-1, self.hidden_size)
        masked_flat = flat_hidden * flat_mask

        gate = self._project_input_blocks(masked_flat, self.base_mlp.gate_proj, active_idx)
        up = self._project_input_blocks(masked_flat, self.base_mlp.up_proj, active_idx)
        activated = self.base_mlp.act_fn(gate) * up
        down = self._project_output_blocks(activated, self.base_mlp.down_proj, active_idx, score_weights=score_weights)
        down = down * flat_mask
        return down.view_as(hidden_states)

    def _forward_materialized_block_sparse(
        self,
        hidden_states: torch.Tensor,
        active_idx: torch.Tensor,
        feature_mask: torch.Tensor,
        score_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        flat_hidden = hidden_states.reshape(-1, self.hidden_size)
        flat_mask = feature_mask.reshape(-1, self.hidden_size)
        masked_flat = flat_hidden * flat_mask
        rows = int(masked_flat.shape[0])
        row_idx = torch.arange(rows, device=masked_flat.device)

        w_dtype = masked_flat.dtype
        gate_weight = materialize_linear_weight(self.base_mlp.gate_proj, device=masked_flat.device, dtype=w_dtype).contiguous()
        up_weight = materialize_linear_weight(self.base_mlp.up_proj, device=masked_flat.device, dtype=w_dtype).contiguous()
        down_weight = materialize_linear_weight(self.base_mlp.down_proj, device=masked_flat.device, dtype=w_dtype).contiguous()
        gate_bias = materialize_linear_bias(self.base_mlp.gate_proj, device=masked_flat.device, dtype=w_dtype)
        up_bias = materialize_linear_bias(self.base_mlp.up_proj, device=masked_flat.device, dtype=w_dtype)
        down_bias = materialize_linear_bias(self.base_mlp.down_proj, device=masked_flat.device, dtype=w_dtype)

        intermediate_size = int(gate_weight.shape[0])
        x_blocks = masked_flat.view(rows, self.num_blocks, self.block_size)
        gate_weight_blocks = gate_weight.view(intermediate_size, self.num_blocks, self.block_size)
        up_weight_blocks = up_weight.view(intermediate_size, self.num_blocks, self.block_size)
        down_weight_blocks = down_weight.view(self.num_blocks, self.block_size, intermediate_size)

        gate = torch.zeros((rows, intermediate_size), device=masked_flat.device, dtype=w_dtype)
        up = torch.zeros((rows, intermediate_size), device=masked_flat.device, dtype=w_dtype)
        if gate_bias is not None:
            gate = gate + gate_bias.unsqueeze(0)
        if up_bias is not None:
            up = up + up_bias.unsqueeze(0)

        for slot in range(active_idx.shape[1]):
            block_idx = active_idx[:, slot]
            valid = block_idx >= 0
            if not torch.any(valid):
                continue
            rows_valid = row_idx[valid]
            blocks_valid = block_idx[valid].to(dtype=torch.long)
            x_selected = x_blocks[rows_valid, blocks_valid]
            gw = gate_weight_blocks[:, blocks_valid, :].permute(1, 0, 2).contiguous()
            uw = up_weight_blocks[:, blocks_valid, :].permute(1, 0, 2).contiguous()
            gate[rows_valid] += torch.einsum("rb,rob->ro", x_selected, gw)
            up[rows_valid] += torch.einsum("rb,rob->ro", x_selected, uw)

        activated = self.base_mlp.act_fn(gate) * up
        out_blocks = torch.zeros((rows, self.num_blocks, self.block_size), device=masked_flat.device, dtype=w_dtype)

        down_bias_blocks = None
        if down_bias is not None:
            down_bias_blocks = down_bias.view(self.num_blocks, self.block_size)

        for slot in range(active_idx.shape[1]):
            block_idx = active_idx[:, slot]
            valid = block_idx >= 0
            if not torch.any(valid):
                continue
            rows_valid = row_idx[valid]
            blocks_valid = block_idx[valid].to(dtype=torch.long)
            dw = down_weight_blocks[blocks_valid]
            contrib = torch.bmm(dw, activated[rows_valid].unsqueeze(-1)).squeeze(-1)
            if down_bias_blocks is not None:
                contrib = contrib + down_bias_blocks[blocks_valid]
            if score_weights is not None:
                slot_weights = score_weights[valid, slot].to(device=contrib.device, dtype=contrib.dtype)
                contrib = contrib * slot_weights.unsqueeze(-1) * float(active_idx.shape[1])
            out_blocks[rows_valid, blocks_valid] += contrib

        out_flat = out_blocks.view(rows, self.hidden_size)
        out_flat = out_flat * flat_mask
        return out_flat.view_as(hidden_states).to(dtype=hidden_states.dtype)

    def _forward_block_bank_sparse(
        self,
        hidden_states: torch.Tensor,
        active_idx: torch.Tensor,
        feature_mask: torch.Tensor,
        score_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._block_bank is None:
            raise RuntimeError("Sparse block bank is not loaded")

        bank = self._block_bank
        rows = int(hidden_states.shape[0] * hidden_states.shape[1])
        flat_hidden = hidden_states.reshape(rows, self.hidden_size)
        flat_mask = feature_mask.reshape(rows, self.hidden_size)
        masked_flat = flat_hidden * flat_mask
        x_blocks = masked_flat.view(rows, self.num_blocks, self.block_size)
        row_idx = torch.arange(rows, device=hidden_states.device)

        gate_blocks = bank["gate_proj_blocks"]
        up_blocks = bank["up_proj_blocks"]
        down_blocks = bank["down_proj_blocks"]
        intermediate_size = int(bank["intermediate_size"])

        gate = torch.zeros((rows, intermediate_size), device=hidden_states.device, dtype=hidden_states.dtype)
        up = torch.zeros((rows, intermediate_size), device=hidden_states.device, dtype=hidden_states.dtype)

        gate_bias = bank["gate_bias"]
        up_bias = bank["up_bias"]
        if gate_bias is not None:
            gate = gate + gate_bias.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(0)
        if up_bias is not None:
            up = up + up_bias.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(0)

        for slot in range(active_idx.shape[1]):
            block_idx = active_idx[:, slot]
            valid = block_idx >= 0
            if not torch.any(valid):
                continue
            rows_valid = row_idx[valid]
            blocks_valid = block_idx[valid].to(dtype=torch.long)
            gather_idx = blocks_valid.to(device=gate_blocks.device)

            x_selected = x_blocks[rows_valid, blocks_valid]
            gate_selected = gate_blocks.index_select(0, gather_idx).to(device=hidden_states.device, dtype=hidden_states.dtype)
            up_selected = up_blocks.index_select(0, gather_idx).to(device=hidden_states.device, dtype=hidden_states.dtype)
            gate[rows_valid] += torch.einsum("rb,rob->ro", x_selected, gate_selected)
            up[rows_valid] += torch.einsum("rb,rob->ro", x_selected, up_selected)

        activated = self.base_mlp.act_fn(gate) * up
        out_blocks = torch.zeros(
            (rows, self.num_blocks, self.block_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        down_bias = bank["down_bias"]
        down_bias_blocks = None
        if down_bias is not None:
            down_bias_blocks = down_bias.to(device=hidden_states.device, dtype=hidden_states.dtype).view(self.num_blocks, self.block_size)

        for slot in range(active_idx.shape[1]):
            block_idx = active_idx[:, slot]
            valid = block_idx >= 0
            if not torch.any(valid):
                continue
            rows_valid = row_idx[valid]
            blocks_valid = block_idx[valid].to(dtype=torch.long)
            gather_idx = blocks_valid.to(device=down_blocks.device)

            down_selected = down_blocks.index_select(0, gather_idx).to(device=hidden_states.device, dtype=hidden_states.dtype)
            contrib = torch.bmm(down_selected, activated[rows_valid].unsqueeze(-1)).squeeze(-1)
            if down_bias_blocks is not None:
                contrib = contrib + down_bias_blocks[blocks_valid]
            if score_weights is not None:
                slot_weights = score_weights[valid, slot].to(device=contrib.device, dtype=contrib.dtype)
                contrib = contrib * slot_weights.unsqueeze(-1) * float(active_idx.shape[1])
            out_blocks[rows_valid, blocks_valid] += contrib

        out_flat = out_blocks.view(rows, self.hidden_size)
        out_flat = out_flat * flat_mask
        return out_flat.view_as(hidden_states)

    def _forward_triton_sparse(
        self,
        hidden_states: torch.Tensor,
        active_idx: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Prefer the dequantized dense Triton path for inference stability on 4-bit weights.
        # It avoids numerically fragile direct 4-bit sparse kernels while keeping sparse routing.
        if self._can_use_torch_block_sparse() or self._can_use_triton_4bit():
            return self._forward_triton_sparse_dense(hidden_states, active_idx, feature_mask)
        return self._forward_triton_sparse_4bit(hidden_states, active_idx, feature_mask)

    def _forward_triton_sparse_dense(
        self,
        hidden_states: torch.Tensor,
        active_idx: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> torch.Tensor:
        flat_hidden = hidden_states.reshape(-1, self.hidden_size)
        flat_mask = feature_mask.reshape(-1, self.hidden_size)
        masked_flat = flat_hidden * flat_mask
        target_weight_dtype = self._resolve_triton_weight_dtype(masked_flat.dtype, masked_flat.device)

        active_idx_i32 = active_idx.to(dtype=torch.int32)
        gate_weight, gate_bias = self._get_cached_triton_linear_params(
            cache_key="gate_proj",
            linear=self.base_mlp.gate_proj,
            target_device=masked_flat.device,
            target_weight_dtype=target_weight_dtype,
        )
        up_weight, up_bias = self._get_cached_triton_linear_params(
            cache_key="up_proj",
            linear=self.base_mlp.up_proj,
            target_device=masked_flat.device,
            target_weight_dtype=target_weight_dtype,
        )
        down_weight, down_bias = self._get_cached_triton_linear_params(
            cache_key="down_proj",
            linear=self.base_mlp.down_proj,
            target_device=masked_flat.device,
            target_weight_dtype=target_weight_dtype,
        )

        gate = triton_sparse_input_linear(
            masked_flat,
            active_idx_i32,
            weight=gate_weight,
            bias=gate_bias,
            block_size=self.block_size,
        )
        up = triton_sparse_input_linear(
            masked_flat,
            active_idx_i32,
            weight=up_weight,
            bias=up_bias,
            block_size=self.block_size,
        )
        activated = self.base_mlp.act_fn(gate) * up
        down = triton_sparse_output_linear(
            activated,
            active_idx_i32,
            flat_mask=flat_mask,
            weight=down_weight,
            bias=down_bias,
            block_size=self.block_size,
        )
        return down.view_as(hidden_states).to(dtype=hidden_states.dtype)

    def _forward_triton_sparse_4bit(
        self,
        hidden_states: torch.Tensor,
        active_idx: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> torch.Tensor:
        flat_hidden = hidden_states.reshape(-1, self.hidden_size)
        flat_mask = feature_mask.reshape(-1, self.hidden_size)
        masked_flat = flat_hidden * flat_mask

        active_idx_i32 = active_idx.to(dtype=torch.int32)
        (
            gate_packed,
            gate_absmax,
            gate_code,
            gate_out_features,
            gate_in_features,
            gate_quant_block_size,
            gate_quant_type,
            gate_bias,
        ) = self._get_cached_triton_4bit_linear_params(
            cache_key="gate_proj_4bit",
            linear=self.base_mlp.gate_proj,
            target_device=masked_flat.device,
            target_bias_dtype=masked_flat.dtype,
        )
        (
            up_packed,
            up_absmax,
            up_code,
            up_out_features,
            up_in_features,
            up_quant_block_size,
            up_quant_type,
            up_bias,
        ) = self._get_cached_triton_4bit_linear_params(
            cache_key="up_proj_4bit",
            linear=self.base_mlp.up_proj,
            target_device=masked_flat.device,
            target_bias_dtype=masked_flat.dtype,
        )
        (
            down_packed,
            down_absmax,
            down_code,
            _down_out_features,
            down_in_features,
            down_quant_block_size,
            down_quant_type,
            down_bias,
        ) = self._get_cached_triton_4bit_linear_params(
            cache_key="down_proj_4bit",
            linear=self.base_mlp.down_proj,
            target_device=masked_flat.device,
            target_bias_dtype=masked_flat.dtype,
        )

        if gate_quant_type not in {"nf4", "fp4"} or up_quant_type not in {"nf4", "fp4"} or down_quant_type not in {"nf4", "fp4"}:
            raise RuntimeError(
                f"Unsupported 4-bit quant_type for Triton sparse path: gate={gate_quant_type}, up={up_quant_type}, down={down_quant_type}"
            )

        gate = triton_sparse_input_linear_4bit(
            masked_flat,
            active_idx_i32,
            packed_weight=gate_packed,
            absmax=gate_absmax,
            code=gate_code,
            out_features=gate_out_features,
            in_features=gate_in_features,
            quant_block_size=gate_quant_block_size,
            bias=gate_bias,
            block_size=self.block_size,
            quant_weight_ref=getattr(self.base_mlp.gate_proj, "weight", None),
        )
        up = triton_sparse_input_linear_4bit(
            masked_flat,
            active_idx_i32,
            packed_weight=up_packed,
            absmax=up_absmax,
            code=up_code,
            out_features=up_out_features,
            in_features=up_in_features,
            quant_block_size=up_quant_block_size,
            bias=up_bias,
            block_size=self.block_size,
            quant_weight_ref=getattr(self.base_mlp.up_proj, "weight", None),
        )
        activated = self.base_mlp.act_fn(gate) * up
        down = triton_sparse_output_linear_4bit(
            activated,
            active_idx_i32,
            flat_mask=flat_mask,
            packed_weight=down_packed,
            absmax=down_absmax,
            code=down_code,
            input_dim=down_in_features,
            quant_block_size=down_quant_block_size,
            bias=down_bias,
            block_size=self.block_size,
            quant_weight_ref=getattr(self.base_mlp.down_proj, "weight", None),
        )
        return down.view_as(hidden_states).to(dtype=hidden_states.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self._last_alignment = None
        self._last_fallback_triggered = False
        self._last_route_snapshot = None
        self._last_learned_basis_stats = None

        def _mark_fallback(triggered: bool) -> None:
            self._last_fallback_triggered = bool(triggered)
            self._fallback_total_steps += 1
            if triggered:
                self._fallback_triggered_steps += 1

        capture = bool(self.capture_alignment)
        dense_ref: Optional[torch.Tensor] = None
        need_dense_blend = bool(self.curriculum_enabled and float(self.curriculum_alpha) < 1.0)
        if capture or need_dense_blend:
            with torch.no_grad():
                dense_ref = self.base_mlp(hidden_states)
        semantic_learned_basis = bool(
            self.config.sparse_placement == "learned_basis"
            and getattr(self.config, "routing_mode", "spatial_grid") == "semantic_latent"
        )
        route_slots_upper = int(self._effective_block_top_k() if semantic_learned_basis else max(int(self.config.route_top_k), 1))

        def _record_alignment(
            sparse_out: torch.Tensor,
            feature_mask: Optional[torch.Tensor],
            route: Optional[SparseRouteSelection],
            fallback_triggered: bool,
        ) -> None:
            if dense_ref is None:
                _mark_fallback(fallback_triggered)
                return
            _mark_fallback(fallback_triggered)
            fallback_tensor = torch.tensor(
                1.0 if fallback_triggered else 0.0,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            active_idx = None if route is None else route.active_idx
            self._last_alignment = {
                "mlp_input": hidden_states.detach(),
                "dense_mlp_out": dense_ref.detach(),
                "sparse_mlp_out": sparse_out,
                "feature_mask": (
                    feature_mask.detach()
                    if feature_mask is not None
                    else torch.ones_like(hidden_states, device=hidden_states.device, dtype=hidden_states.dtype)
                ),
                "active_idx": (
                    active_idx.detach()
                    if active_idx is not None
                    else torch.full(
                        (hidden_states.shape[0] * hidden_states.shape[1], route_slots_upper),
                        -1,
                        device=hidden_states.device,
                        dtype=torch.long,
                    )
                ),
                "fallback_triggered": fallback_tensor,
            }
            if route is not None:
                if route.latent_idx is not None:
                    self._last_alignment["latent_idx"] = route.latent_idx.detach()
                if route.latent_weights is not None:
                    self._last_alignment["latent_weights"] = route.latent_weights.detach().to(dtype=torch.float32)
                if route.block_scores is not None:
                    self._last_alignment["block_scores"] = route.block_scores.detach().to(dtype=torch.float32)
                self._last_alignment["effective_basis_rank"] = torch.tensor(
                    int(self._effective_basis_rank()),
                    device=hidden_states.device,
                    dtype=torch.int32,
                )
                self._last_alignment["effective_basis_top_k"] = torch.tensor(
                    int(self._effective_basis_top_k()),
                    device=hidden_states.device,
                    dtype=torch.int32,
                )
                self._last_alignment["effective_block_top_k"] = torch.tensor(
                    int(self._effective_block_top_k()),
                    device=hidden_states.device,
                    dtype=torch.int32,
                )

        if not self.enabled_fn(self.layer_idx):
            self._last_diagnostics = SparseMLPDiagnostics(
                layer_idx=self.layer_idx,
                rows=int(hidden_states.shape[0] * hidden_states.shape[1]),
                mean_active_blocks=0.0,
                unique_active_blocks=0,
                touched_weight_fraction=0.0,
                estimated_bytes_fetched_per_token=0.0,
            )
            out = self.base_mlp(hidden_states)
            out = self._apply_output_scale(out)
            out = self._apply_curriculum_blend(out, dense_ref)
            # Decode-guard/buffer-disabled layers are intentional dense path, not sparse fallback.
            _record_alignment(out, None, None, fallback_triggered=False)
            return out

        threshold = float(getattr(self.config, "stability_dense_fallback_threshold", 0.0))
        touched_ratio_upper = float(route_slots_upper) / float(max(self.num_blocks, 1))
        if threshold > 0.0 and touched_ratio_upper < threshold:
            rows = int(hidden_states.shape[0] * hidden_states.shape[1])
            slots = int(max(route_slots_upper, 1))
            active_idx = (
                torch.arange(slots, device=hidden_states.device, dtype=torch.long)
                .view(1, -1)
                .expand(rows, -1)
                .contiguous()
            )
            fallback_route = SparseRouteSelection(active_idx=active_idx)
            self._update_diagnostics(active_idx)
            out = self.base_mlp(hidden_states)
            out = self._apply_output_scale(out)
            out = self._apply_curriculum_blend(out, dense_ref)
            _record_alignment(
                out,
                torch.ones_like(hidden_states, device=hidden_states.device, dtype=hidden_states.dtype),
                fallback_route,
                fallback_triggered=True,
            )
            return out

        if semantic_learned_basis:
            route = self._route_learned_basis_semantic(hidden_states)
        else:
            route = self.route_fn(hidden_states, self.layer_idx)
        active_idx = route.active_idx
        if semantic_learned_basis:
            self._update_block_banking(active_idx)
        score_weights = route.score_weights if self.config.soft_mask else None
        if self.capture_route_snapshot:
            rows = int(hidden_states.shape[0] * hidden_states.shape[1])
            snapshot_weights = score_weights
            if snapshot_weights is None:
                valid = active_idx >= 0
                snapshot_weights = torch.zeros_like(active_idx, dtype=torch.float32, device=active_idx.device)
                if torch.any(valid):
                    counts = valid.sum(dim=-1, keepdim=True).clamp_min(1)
                    snapshot_weights = torch.where(
                        valid,
                        1.0 / counts.to(dtype=torch.float32),
                        torch.zeros_like(snapshot_weights),
                    )
            self._last_route_snapshot = {
                "active_idx": active_idx.detach(),
                "score_weights": snapshot_weights.detach().to(dtype=torch.float32),
                "rows": torch.tensor(rows, device=hidden_states.device, dtype=torch.int32),
                "batch_size": torch.tensor(int(hidden_states.shape[0]), device=hidden_states.device, dtype=torch.int32),
                "seq_len": torch.tensor(int(hidden_states.shape[1]), device=hidden_states.device, dtype=torch.int32),
                "layer_idx": torch.tensor(int(self.layer_idx), device=hidden_states.device, dtype=torch.int32),
                "effective_basis_rank": torch.tensor(int(self._effective_basis_rank()), device=hidden_states.device, dtype=torch.int32),
                "effective_basis_top_k": torch.tensor(int(self._effective_basis_top_k()), device=hidden_states.device, dtype=torch.int32),
                "effective_block_top_k": torch.tensor(int(self._effective_block_top_k()), device=hidden_states.device, dtype=torch.int32),
            }
            if route.latent_idx is not None:
                self._last_route_snapshot["latent_idx"] = route.latent_idx.detach()
            if route.latent_weights is not None:
                self._last_route_snapshot["latent_weights"] = route.latent_weights.detach().to(dtype=torch.float32)
            if route.block_scores is not None:
                self._last_route_snapshot["block_scores"] = route.block_scores.detach().to(dtype=torch.float32)
        self._update_diagnostics(active_idx)
        if threshold > 0.0 and self._last_diagnostics is not None:
            if float(self._last_diagnostics.touched_weight_fraction) < threshold:
                out = self.base_mlp(hidden_states)
                out = self._apply_output_scale(out)
                out = self._apply_curriculum_blend(out, dense_ref)
                _record_alignment(
                    out,
                    torch.ones_like(hidden_states, device=hidden_states.device, dtype=hidden_states.dtype),
                    route,
                    fallback_triggered=True,
                )
                return out

        if active_idx.numel() == 0:
            out = torch.zeros_like(hidden_states)
            out = self._apply_output_scale(out)
            out = self._apply_curriculum_blend(out, dense_ref)
            _record_alignment(out, torch.zeros_like(hidden_states), route, fallback_triggered=False)
            return out

        flat_mask = self._build_feature_mask(
            active_idx=active_idx,
            score_weights=score_weights,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        feature_mask = flat_mask.view_as(hidden_states)
        if self._block_bank is not None:
            if self.config.sparse_placement == "learned_basis":
                out = self._forward_learned_basis(hidden_states, route)
                out = self._apply_output_scale(out)
                out = self._apply_curriculum_blend(out, dense_ref)
                _record_alignment(out, feature_mask, route, fallback_triggered=False)
                return out
            if self.config.sparse_placement != "input_mask":
                out = self._forward_output_sparse(hidden_states, active_idx, score_weights=score_weights)
                out = self._apply_output_scale(out)
                out = self._apply_curriculum_blend(out, dense_ref)
                _record_alignment(out, feature_mask, route, fallback_triggered=False)
                return out
            out = self._forward_block_bank_sparse(
                hidden_states=hidden_states,
                active_idx=active_idx,
                feature_mask=feature_mask,
                score_weights=score_weights,
            )
            out = self._apply_output_scale(out)
            out = self._apply_curriculum_blend(out, dense_ref)
            _record_alignment(out, feature_mask, route, fallback_triggered=False)
            return out
        if self.config.grouped_row_gemm:
            if self.config.sparse_placement == "learned_basis":
                out = self._forward_learned_basis(hidden_states, route)
                out = self._apply_output_scale(out)
                out = self._apply_curriculum_blend(out, dense_ref)
                _record_alignment(out, feature_mask, route, fallback_triggered=False)
                return out
            if self.config.sparse_placement != "input_mask":
                out = self._forward_output_sparse(hidden_states, active_idx, score_weights=score_weights)
                out = self._apply_output_scale(out)
                out = self._apply_curriculum_blend(out, dense_ref)
                _record_alignment(out, feature_mask, route, fallback_triggered=False)
                return out
            out = self._forward_grouped_row_gemm(hidden_states, active_idx, feature_mask)
            out = self._apply_output_scale(out)
            out = self._apply_curriculum_blend(out, dense_ref)
            _record_alignment(out, feature_mask, route, fallback_triggered=False)
            return out

        if self.config.spmm_impl == "dense":
            if self.config.sparse_placement == "input_mask":
                out = self._forward_dense_masked(hidden_states, feature_mask)
            elif self.config.sparse_placement == "learned_basis":
                out = self._forward_learned_basis(hidden_states, route)
            elif self.config.sparse_placement == "output_sparse":
                out = self._forward_output_sparse(hidden_states, active_idx, score_weights=score_weights)
            else:
                out = self._forward_intermediate_group_sparse(hidden_states, active_idx, score_weights=score_weights)
            out = self._apply_output_scale(out)
            out = self._apply_curriculum_blend(out, dense_ref)
            _record_alignment(out, feature_mask, route, fallback_triggered=False)
            return out
        if self.config.spmm_impl == "cuda_spmm":
            if not hidden_states.is_cuda or not active_idx.is_cuda:
                raise RuntimeError("cuda_spmm requires CUDA tensors")
            if self.config.sparse_placement == "learned_basis":
                out = self._forward_learned_basis(hidden_states, route)
                out = self._apply_output_scale(out)
                out = self._apply_curriculum_blend(out, dense_ref)
                _record_alignment(out, feature_mask, route, fallback_triggered=False)
                return out
            if self.config.sparse_placement != "input_mask":
                out = self._forward_output_sparse(hidden_states, active_idx, score_weights=score_weights)
                out = self._apply_output_scale(out)
                out = self._apply_curriculum_blend(out, dense_ref)
                _record_alignment(out, feature_mask, route, fallback_triggered=False)
                return out
            if self._can_use_triton_4bit():
                out = self._forward_materialized_block_sparse(
                    hidden_states=hidden_states,
                    active_idx=active_idx,
                    feature_mask=feature_mask,
                    score_weights=score_weights,
                )
                out = self._apply_output_scale(out)
                out = self._apply_curriculum_blend(out, dense_ref)
                _record_alignment(out, feature_mask, route, fallback_triggered=False)
                return out
            if not triton_sparse_mlp_available():
                raise RuntimeError("cuda_spmm requires Triton")
            if not self._can_use_triton_sparse():
                raise RuntimeError("cuda_spmm requires Triton-compatible dense or 4-bit linear weights")
            if self._can_use_triton_4bit() and self.training and torch.is_grad_enabled() and self._triton_4bit_trainable_weights():
                raise RuntimeError("cuda_spmm 4-bit path does not support quantized weight gradients")
            out = self._forward_triton_sparse(hidden_states, active_idx, feature_mask)
            out = self._apply_output_scale(out)
            out = self._apply_curriculum_blend(out, dense_ref)
            _record_alignment(out, feature_mask, route, fallback_triggered=False)
            return out
        if self.config.spmm_impl == "torch_block_sparse":
            if not self._can_use_torch_block_sparse():
                raise RuntimeError("torch_block_sparse requires dense floating-point linear weights")
            if self.config.sparse_placement == "input_mask":
                out = self._forward_torch_block_sparse(hidden_states, active_idx, feature_mask, score_weights=score_weights)
            elif self.config.sparse_placement == "learned_basis":
                out = self._forward_learned_basis(hidden_states, route)
            elif self.config.sparse_placement == "output_sparse":
                out = self._forward_output_sparse(hidden_states, active_idx, score_weights=score_weights)
            else:
                out = self._forward_intermediate_group_sparse(hidden_states, active_idx, score_weights=score_weights)
            out = self._apply_output_scale(out)
            out = self._apply_curriculum_blend(out, dense_ref)
            _record_alignment(out, feature_mask, route, fallback_triggered=False)
            return out
        raise ValueError(f"Unsupported spmm_impl: {self.config.spmm_impl}")
