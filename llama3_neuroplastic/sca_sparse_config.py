from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict

import torch


@dataclass
class SCASparseConfig:
    hidden_size: int = 4096
    block_size: int = 32
    block_rank: int = 4
    basis_rank: int = 32
    basis_top_k: int = 8
    top_k: int = 3
    adaptive_top_k: bool = False
    adaptive_top_k_min: int = 3
    adaptive_top_k_max: int = 3
    adaptive_top_k_min_score_ratio: float = 0.15
    sigma: float = 1.0
    refractory_steps: int = 100
    inhibition_lambda: float = 0.0
    inhibition_radius: float = 1.5
    use_cuda: bool = True
    grid_size: int = 16
    spmm_impl: str = "dense"
    sparse_placement: str = "input_mask"
    routing_mode: str = "spatial_grid"
    semantic_block_score_normalized: bool = False
    basis_rank_by_layer: Dict[int, int] = field(default_factory=dict)
    basis_top_k_by_layer: Dict[int, int] = field(default_factory=dict)
    top_k_by_layer: Dict[int, int] = field(default_factory=dict)
    soft_mask: bool = True
    grouped_row_gemm: bool = False
    grouped_row_min_bucket: int = 2
    grouped_row_allow_4bit_dequant: bool = False
    stability_dense_fallback_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if self.hidden_size % self.block_size != 0:
            raise ValueError("hidden_size must be divisible by block_size")
        if self.block_rank <= 0:
            raise ValueError("block_rank must be > 0")
        if self.basis_rank <= 0:
            raise ValueError("basis_rank must be > 0")
        if self.basis_top_k <= 0:
            raise ValueError("basis_top_k must be > 0")
        if self.basis_top_k > self.basis_rank:
            raise ValueError("basis_top_k must be <= basis_rank")
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if self.top_k > self.num_blocks:
            raise ValueError("top_k must be <= num_blocks")
        if self.adaptive_top_k_min <= 0:
            raise ValueError("adaptive_top_k_min must be > 0")
        if self.adaptive_top_k_max <= 0:
            raise ValueError("adaptive_top_k_max must be > 0")
        if self.adaptive_top_k_max < self.adaptive_top_k_min:
            raise ValueError("adaptive_top_k_max must be >= adaptive_top_k_min")
        if self.adaptive_top_k_max > self.num_blocks:
            raise ValueError("adaptive_top_k_max must be <= num_blocks")
        if self.adaptive_top_k_min_score_ratio <= 0.0 or self.adaptive_top_k_min_score_ratio > 1.0:
            raise ValueError("adaptive_top_k_min_score_ratio must be in (0, 1]")
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0")
        if self.refractory_steps < 0:
            raise ValueError("refractory_steps must be >= 0")
        if self.inhibition_lambda < 0:
            raise ValueError("inhibition_lambda must be >= 0")
        if self.inhibition_radius < 0:
            raise ValueError("inhibition_radius must be >= 0")
        if self.grid_size <= 0:
            raise ValueError("grid_size must be > 0")
        if self.grid_size < 2:
            raise ValueError("grid_size must be >= 2")
        if self.spmm_impl not in {"dense", "torch_block_sparse", "cuda_spmm"}:
            raise ValueError("spmm_impl must be one of: dense, torch_block_sparse, cuda_spmm")
        if self.sparse_placement not in {"input_mask", "output_sparse", "intermediate_group", "learned_basis"}:
            raise ValueError("sparse_placement must be one of: input_mask, output_sparse, intermediate_group, learned_basis")
        if self.routing_mode not in {"spatial_grid", "semantic_latent"}:
            raise ValueError("routing_mode must be one of: spatial_grid, semantic_latent")
        if self.grouped_row_min_bucket <= 0:
            raise ValueError("grouped_row_min_bucket must be > 0")
        if self.stability_dense_fallback_threshold < 0.0 or self.stability_dense_fallback_threshold > 1.0:
            raise ValueError("stability_dense_fallback_threshold must be in [0, 1]")

        self.basis_rank_by_layer = self._normalize_layer_map(self.basis_rank_by_layer)
        self.basis_top_k_by_layer = self._normalize_layer_map(self.basis_top_k_by_layer)
        self.top_k_by_layer = self._normalize_layer_map(self.top_k_by_layer)

        for layer_idx, value in self.basis_rank_by_layer.items():
            if value <= 0:
                raise ValueError(f"basis_rank_by_layer[{layer_idx}] must be > 0")
            if value > self.basis_rank:
                raise ValueError(f"basis_rank_by_layer[{layer_idx}] must be <= basis_rank")
        for layer_idx, value in self.basis_top_k_by_layer.items():
            if value <= 0:
                raise ValueError(f"basis_top_k_by_layer[{layer_idx}] must be > 0")
            max_rank = self.basis_rank_for_layer(layer_idx)
            if value > max_rank:
                raise ValueError(f"basis_top_k_by_layer[{layer_idx}] must be <= effective basis rank")
        for layer_idx, value in self.top_k_by_layer.items():
            if value <= 0:
                raise ValueError(f"top_k_by_layer[{layer_idx}] must be > 0")
            if value > self.num_blocks:
                raise ValueError(f"top_k_by_layer[{layer_idx}] must be <= num_blocks")

    @staticmethod
    def _normalize_layer_map(raw: Dict[int, int] | Dict[str, int]) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for key, value in dict(raw).items():
            layer_idx = int(key)
            if layer_idx < 0:
                raise ValueError("layer override keys must be >= 0")
            out[layer_idx] = int(value)
        return out

    @property
    def num_blocks(self) -> int:
        return self.hidden_size // self.block_size

    @property
    def route_top_k(self) -> int:
        if self.adaptive_top_k:
            return int(self.adaptive_top_k_max)
        return int(self.top_k)

    def basis_rank_for_layer(self, layer_idx: int) -> int:
        return int(max(1, min(self.basis_rank_by_layer.get(int(layer_idx), self.basis_rank), self.basis_rank)))

    def basis_top_k_for_layer(self, layer_idx: int) -> int:
        default = self.basis_top_k_by_layer.get(int(layer_idx), self.basis_top_k)
        effective_rank = self.basis_rank_for_layer(layer_idx)
        return int(max(1, min(int(default), effective_rank)))

    def top_k_for_layer(self, layer_idx: int) -> int:
        default = self.top_k_by_layer.get(int(layer_idx), self.route_top_k)
        return int(max(1, min(int(default), self.num_blocks)))

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "SCASparseConfig":
        return cls(**payload)


def build_block_centers(config: SCASparseConfig) -> torch.Tensor:
    """
    Build fixed 3D centers for each block.
    For hidden sizes that form an exact cube (e.g. 4096 = 16^3), we map dims
    onto that base lattice and rescale to the configured grid_size domain.
    """
    base_grid = round(config.hidden_size ** (1.0 / 3.0))
    if base_grid ** 3 != config.hidden_size:
        raise ValueError(
            f"hidden_size must be a perfect cube for 3D mapping, got hidden_size={config.hidden_size}"
        )

    grid = float(config.grid_size)
    base = float(base_grid)
    idx = torch.arange(config.hidden_size, dtype=torch.float32)
    x = idx % base
    y = torch.floor(idx / base) % base
    z = torch.floor(idx / (base * base))
    coords = torch.stack((x, y, z), dim=-1).float()
    if base > 1.0:
        coords = coords * ((grid - 1.0) / (base - 1.0))
    centers = coords.view(config.num_blocks, config.block_size, 3).mean(dim=1)
    return centers.contiguous()


def build_inhibition_matrix(block_centers: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Neighbor matrix for optional lateral inhibition.
    """
    if radius <= 0:
        return torch.zeros(
            (block_centers.shape[0], block_centers.shape[0]),
            device=block_centers.device,
            dtype=torch.float32,
        )

    dists = torch.cdist(block_centers.float(), block_centers.float(), p=2)
    neighbors = (dists <= radius) & (dists > 0)
    matrix = neighbors.float()

    denom = matrix.sum(dim=1, keepdim=True).clamp_min(1.0)
    matrix = matrix / denom
    return matrix.contiguous()
