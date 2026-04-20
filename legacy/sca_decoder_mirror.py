from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .sca_sparse_adapter import SCABlockSparseAdapter
    from .sca_sparse_config import SCASparseConfig, build_block_centers
except ImportError:
    from sca_sparse_adapter import SCABlockSparseAdapter
    from sca_sparse_config import SCASparseConfig, build_block_centers


@dataclass
class DecoderMirrorConfig:
    hidden_size: int
    block_size: int
    num_blocks: int
    top_k: int
    rank: int
    grid_size: int
    sigma: float
    route_prior_scale_init: float
    residual_scale_init: float
    source_layer_indices: list[int]
    enabled: bool = True
    route_conditioned: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DecoderMirrorDiagnostics:
    mean_active_blocks: float
    touched_weight_fraction: float
    delta_norm_ratio: float
    route_prior_nonzero_fraction: float
    route_prior_missing: bool
    mean_route_prior_entropy: float
    source_layers_used: list[int]
    residual_scale: float
    route_prior_scale: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SparseDecoderMirrorSCA(nn.Module):
    def __init__(self, config: DecoderMirrorConfig) -> None:
        super().__init__()
        if int(config.hidden_size) <= 0:
            raise ValueError("hidden_size must be > 0")
        if int(config.block_size) <= 0:
            raise ValueError("block_size must be > 0")
        if int(config.hidden_size) % int(config.block_size) != 0:
            raise ValueError("hidden_size must be divisible by block_size")
        if int(config.num_blocks) != int(config.hidden_size // config.block_size):
            raise ValueError("num_blocks must match hidden_size // block_size")
        if int(config.top_k) <= 0 or int(config.top_k) > int(config.num_blocks):
            raise ValueError("top_k must be in [1, num_blocks]")
        if int(config.rank) <= 0:
            raise ValueError("rank must be > 0")
        self.config = config
        self.hidden_size = int(config.hidden_size)
        self.block_size = int(config.block_size)
        self.num_blocks = int(config.num_blocks)
        self.top_k = int(config.top_k)

        self.spatial_proj = nn.Linear(self.hidden_size, 3, bias=True)
        nn.init.xavier_uniform_(self.spatial_proj.weight)
        nn.init.zeros_(self.spatial_proj.bias)

        adapter_config = SCASparseConfig(
            hidden_size=self.hidden_size,
            block_size=self.block_size,
            block_rank=int(config.rank),
            top_k=self.top_k,
            sigma=float(config.sigma),
            refractory_steps=0,
            inhibition_lambda=0.0,
            use_cuda=False,
            grid_size=int(config.grid_size),
            spmm_impl="dense",
            soft_mask=True,
            grouped_row_gemm=False,
            stability_dense_fallback_threshold=0.0,
        )
        self.adapter = SCABlockSparseAdapter(adapter_config)
        self.route_prior_scale = nn.Parameter(torch.tensor(float(config.route_prior_scale_init), dtype=torch.float32))
        self.residual_scale = nn.Parameter(torch.tensor(float(config.residual_scale_init), dtype=torch.float32))

        block_centers = build_block_centers(adapter_config).float()
        self.register_buffer("block_centers", block_centers, persistent=True)

        self._last_diagnostics: DecoderMirrorDiagnostics | None = None
        self._last_delta_norm_ratio_term: torch.Tensor | None = None
        self._last_route_prior: torch.Tensor | None = None
        self._last_active_idx: torch.Tensor | None = None

    def get_last_diagnostics(self) -> dict[str, Any]:
        if self._last_diagnostics is None:
            return {
                "mean_active_blocks": 0.0,
                "touched_weight_fraction": 0.0,
                "delta_norm_ratio": 0.0,
                "route_prior_nonzero_fraction": 0.0,
                "route_prior_missing": True,
                "mean_route_prior_entropy": 0.0,
                "source_layers_used": [],
                "residual_scale": float(self.residual_scale.detach().cpu().item()),
                "route_prior_scale": float(self.route_prior_scale.detach().cpu().item()),
            }
        return self._last_diagnostics.to_dict()

    def get_last_delta_norm_ratio_term(self) -> torch.Tensor | None:
        return self._last_delta_norm_ratio_term

    def get_last_active_idx(self) -> torch.Tensor | None:
        return self._last_active_idx

    def _spatial_scores(self, flat_hidden: torch.Tensor) -> torch.Tensor:
        query_in = F.layer_norm(flat_hidden.float(), (self.hidden_size,))
        proj_device = self.spatial_proj.weight.device
        proj_dtype = self.spatial_proj.weight.dtype
        query_raw = self.spatial_proj(query_in.to(device=proj_device, dtype=proj_dtype))
        query = (torch.sigmoid(query_raw) * float(self.config.grid_size - 1)).to(
            device=flat_hidden.device,
            dtype=torch.float32,
        )
        centers = self.block_centers.to(device=flat_hidden.device, dtype=torch.float32)
        d2 = torch.sum((query[:, None, :] - centers[None, :, :]) ** 2, dim=-1)
        return torch.exp(-d2 / (2.0 * float(self.config.sigma) * float(self.config.sigma)))

    def _normalize_route_prior(
        self,
        route_prior: torch.Tensor | None,
        rows: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, bool, float]:
        if route_prior is None or route_prior.numel() == 0:
            return torch.zeros((rows, self.num_blocks), device=device, dtype=torch.float32), True, 0.0
        prior = route_prior.reshape(rows, self.num_blocks).to(device=device, dtype=torch.float32)
        prior = torch.clamp(prior, min=0.0)
        prior = prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        entropy = -(prior * prior.clamp_min(1e-6).log()).sum(dim=-1).mean()
        return prior, False, float(entropy.detach().cpu().item())

    def forward(
        self,
        hidden_states: torch.Tensor,
        route_prior: torch.Tensor | None = None,
        source_layers_used: list[int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        if hidden_size != self.hidden_size:
            raise RuntimeError(
                f"Decoder mirror hidden size mismatch: expected {self.hidden_size}, got {hidden_size}"
            )
        rows = int(batch_size * seq_len)
        if rows == 0 or not bool(self.config.enabled):
            self._last_delta_norm_ratio_term = hidden_states.new_zeros(())
            self._last_active_idx = torch.empty((0, self.top_k), device=hidden_states.device, dtype=torch.long)
            self._last_route_prior = None
            self._last_diagnostics = DecoderMirrorDiagnostics(
                mean_active_blocks=0.0,
                touched_weight_fraction=0.0,
                delta_norm_ratio=0.0,
                route_prior_nonzero_fraction=0.0,
                route_prior_missing=True,
                mean_route_prior_entropy=0.0,
                source_layers_used=list(source_layers_used or []),
                residual_scale=float(self.residual_scale.detach().cpu().item()),
                route_prior_scale=float(self.route_prior_scale.detach().cpu().item()),
            )
            return hidden_states, self._last_diagnostics.to_dict()

        flat_hidden = hidden_states.reshape(rows, self.hidden_size)
        spatial_scores = self._spatial_scores(flat_hidden)
        prior, prior_missing, entropy = self._normalize_route_prior(route_prior, rows, hidden_states.device)
        self._last_route_prior = prior.detach()

        fused_scores = spatial_scores
        if bool(self.config.route_conditioned) and not prior_missing:
            prior_bias = torch.clamp(torch.log(prior + 1e-6), min=-6.0, max=0.0)
            fused_scores = fused_scores + (
                self.route_prior_scale.to(device=hidden_states.device, dtype=fused_scores.dtype) * prior_bias
            )

        active_score, active_idx = torch.topk(fused_scores, k=self.top_k, dim=-1, largest=True, sorted=True)
        score_weights = torch.softmax(active_score, dim=-1)
        self._last_active_idx = active_idx.detach()

        adapter_dtype = self.adapter.down_w.dtype
        adapter_hidden = hidden_states.to(dtype=adapter_dtype)
        delta = self.adapter.forward_sparse(
            hidden_states=adapter_hidden,
            active_idx=active_idx.long(),
            score_weights=score_weights,
            use_cuda_kernel=False,
            cuda_kernels=None,
        )
        delta = delta.to(dtype=hidden_states.dtype)
        residual_scale = self.residual_scale.to(device=hidden_states.device, dtype=hidden_states.dtype).view(1, 1, 1)
        warped = hidden_states + (residual_scale * delta)

        mean_active_blocks = float((active_idx >= 0).sum(dim=-1).float().mean().detach().cpu().item())
        touched_weight_fraction = float(mean_active_blocks / max(self.num_blocks, 1))
        delta_norm_ratio_term = (
            delta.float().pow(2).sum(dim=-1) / hidden_states.float().pow(2).sum(dim=-1).clamp_min(1e-6)
        ).mean()
        self._last_delta_norm_ratio_term = delta_norm_ratio_term
        route_prior_nonzero_fraction = float((prior > 0).float().mean().detach().cpu().item()) if not prior_missing else 0.0

        self._last_diagnostics = DecoderMirrorDiagnostics(
            mean_active_blocks=mean_active_blocks,
            touched_weight_fraction=touched_weight_fraction,
            delta_norm_ratio=float(delta_norm_ratio_term.detach().cpu().item()),
            route_prior_nonzero_fraction=route_prior_nonzero_fraction,
            route_prior_missing=bool(prior_missing),
            mean_route_prior_entropy=float(entropy),
            source_layers_used=list(source_layers_used or []),
            residual_scale=float(self.residual_scale.detach().cpu().item()),
            route_prior_scale=float(self.route_prior_scale.detach().cpu().item()),
        )
        return warped, self._last_diagnostics.to_dict()
