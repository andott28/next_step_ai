from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

try:
    from sklearn.decomposition import IncrementalPCA
except ImportError:  # pragma: no cover
    IncrementalPCA = None


def _select_pca_method(
    *,
    rows: int,
    hidden_size: int,
    rank_eff: int,
    pca_method: str,
) -> str:
    requested = str(pca_method).strip().lower()
    if requested in {"lowrank", "incremental"}:
        return requested
    if requested != "auto":
        raise ValueError(f"Unsupported PCA method: {pca_method}")
    element_count = int(rows) * int(hidden_size)
    if int(rows) >= max(4096, int(rank_eff) * 8) or element_count >= 32_000_000:
        return "incremental"
    return "lowrank"


def _fit_lowrank_pca(y_centered: torch.Tensor, *, rank_eff: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
    _u, singular_values, v = torch.pca_lowrank(y_centered, q=int(rank_eff), center=False, niter=2)
    v = v[:, : int(rank_eff)].contiguous()
    coeff = y_centered @ v
    singular_sq = singular_values[: int(rank_eff)].pow(2)
    total_sq = float(y_centered.pow(2).sum().clamp_min(1e-8).item())
    explained = float(singular_sq.sum().item() / total_sq)
    return v, coeff, explained


def _fit_incremental_pca(
    y_centered: torch.Tensor,
    *,
    rank_eff: int,
    batch_rows: int,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    if IncrementalPCA is None:
        raise RuntimeError("scikit-learn is required for incremental PCA")
    batch_size = int(max(batch_rows, rank_eff, 1))
    ipca = IncrementalPCA(n_components=int(rank_eff), batch_size=batch_size)
    y_cpu = y_centered.detach().to(device="cpu", dtype=torch.float32)
    for start in range(0, int(y_cpu.shape[0]), batch_size):
        chunk = y_cpu[start : start + batch_size]
        if int(chunk.shape[0]) < int(rank_eff):
            break
        ipca.partial_fit(chunk.contiguous().numpy())
    components = torch.from_numpy(ipca.components_.copy()).to(dtype=torch.float32)
    v = components.transpose(0, 1).contiguous()
    coeff_chunks = []
    for start in range(0, int(y_cpu.shape[0]), batch_size):
        chunk = y_cpu[start : start + batch_size]
        coeff_chunks.append(chunk @ v)
    coeff = torch.cat(coeff_chunks, dim=0) if coeff_chunks else y_cpu.new_zeros((0, int(rank_eff)))
    explained = float(sum(ipca.explained_variance_ratio_))
    return v, coeff, explained


def _fit_encoder_from_coeff(
    *,
    x_cpu: torch.Tensor,
    coeff: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ones = torch.ones((x_cpu.shape[0], 1), dtype=x_cpu.dtype)
    x_aug = torch.cat([x_cpu, ones], dim=-1)
    proj = torch.linalg.lstsq(x_aug, coeff).solution
    enc_w_eff = proj[:-1, :].transpose(0, 1).contiguous()
    enc_b_eff = proj[-1, :].contiguous()
    return enc_w_eff, enc_b_eff


def _pad_basis_rows(
    *,
    basis: torch.Tensor,
    enc_w_eff: torch.Tensor,
    enc_b_eff: torch.Tensor,
    basis_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(basis.shape[0]) >= int(basis_rank):
        return basis, enc_w_eff, enc_b_eff
    pad_rows = int(basis_rank) - int(basis.shape[0])
    basis = torch.cat([basis, torch.zeros((pad_rows, basis.shape[1]), dtype=basis.dtype)], dim=0)
    enc_w_eff = torch.cat(
        [enc_w_eff, torch.zeros((pad_rows, enc_w_eff.shape[1]), dtype=enc_w_eff.dtype)],
        dim=0,
    )
    enc_b_eff = torch.cat([enc_b_eff, torch.zeros((pad_rows,), dtype=enc_b_eff.dtype)], dim=0)
    return basis, enc_w_eff, enc_b_eff


def fit_layer_basis(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    basis_rank: int,
    block_size: int,
    pca_method: str = "auto",
    pca_batch_rows: int = 1024,
) -> Dict[str, Any]:
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise RuntimeError("x/y must be 2D with matching rows")
    hidden_size = int(y.shape[1])
    num_blocks = hidden_size // int(block_size)
    if num_blocks * int(block_size) != hidden_size:
        raise RuntimeError("hidden_size must be divisible by block_size")

    x_cpu = x.detach().to(device="cpu", dtype=torch.float32)
    y_cpu = y.detach().to(device="cpu", dtype=torch.float32)
    y_mean = y_cpu.mean(dim=0)
    y_centered = y_cpu - y_mean
    rows = int(y_centered.shape[0])
    rank_eff = int(max(min(int(basis_rank), rows, hidden_size), 1))
    selected_method = _select_pca_method(
        rows=rows,
        hidden_size=hidden_size,
        rank_eff=rank_eff,
        pca_method=pca_method,
    )
    if selected_method == "incremental":
        v, coeff, explained = _fit_incremental_pca(y_centered, rank_eff=rank_eff, batch_rows=int(pca_batch_rows))
    else:
        v, coeff, explained = _fit_lowrank_pca(y_centered, rank_eff=rank_eff)

    enc_w_eff, enc_b_eff = _fit_encoder_from_coeff(x_cpu=x_cpu, coeff=coeff)
    basis = v.transpose(0, 1).contiguous()
    basis, enc_w_eff, enc_b_eff = _pad_basis_rows(
        basis=basis,
        enc_w_eff=enc_w_eff,
        enc_b_eff=enc_b_eff,
        basis_rank=int(basis_rank),
    )

    decoder_blocks = basis.view(int(basis_rank), num_blocks, int(block_size)).permute(1, 0, 2).contiguous()
    decoder_bias = y_mean.view(num_blocks, int(block_size)).contiguous()

    return {
        "encoder_weight": enc_w_eff.detach().cpu().float(),
        "encoder_bias": enc_b_eff.detach().cpu().float(),
        "decoder_blocks": decoder_blocks.detach().cpu().float(),
        "decoder_bias": decoder_bias.detach().cpu().float(),
        "scale": torch.tensor(1.0, dtype=torch.float32),
        "samples": int(x_cpu.shape[0]),
        "explained_variance_ratio": float(explained),
        "rank_effective": int(rank_eff),
        "pca_method": selected_method,
    }


def fit_block_score_basis(
    x: torch.Tensor,
    block_scores: torch.Tensor,
    *,
    basis_rank: int,
    pca_method: str = "auto",
    pca_batch_rows: int = 1024,
) -> Dict[str, Any]:
    if x.ndim != 2 or block_scores.ndim != 2 or x.shape[0] != block_scores.shape[0]:
        raise RuntimeError("x/block_scores must be 2D with matching rows")

    x_cpu = x.detach().to(device="cpu", dtype=torch.float32)
    scores_cpu = block_scores.detach().to(device="cpu", dtype=torch.float32)
    num_blocks = int(scores_cpu.shape[1])
    if num_blocks <= 0:
        raise RuntimeError("block_scores must have at least one column")

    score_mean = scores_cpu.mean(dim=0)
    score_centered = scores_cpu - score_mean
    rows = int(score_centered.shape[0])
    rank_eff = int(max(min(int(basis_rank), rows, num_blocks), 1))
    selected_method = _select_pca_method(
        rows=rows,
        hidden_size=num_blocks,
        rank_eff=rank_eff,
        pca_method=pca_method,
    )
    if selected_method == "incremental":
        v, coeff, explained = _fit_incremental_pca(
            score_centered,
            rank_eff=rank_eff,
            batch_rows=int(pca_batch_rows),
        )
    else:
        v, coeff, explained = _fit_lowrank_pca(score_centered, rank_eff=rank_eff)

    enc_w_eff, enc_b_eff = _fit_encoder_from_coeff(x_cpu=x_cpu, coeff=coeff)
    basis = v.transpose(0, 1).contiguous()
    basis, enc_w_eff, enc_b_eff = _pad_basis_rows(
        basis=basis,
        enc_w_eff=enc_w_eff,
        enc_b_eff=enc_b_eff,
        basis_rank=int(basis_rank),
    )
    score_weight = basis.transpose(0, 1).contiguous()
    block_importance = scores_cpu.mean(dim=0).contiguous()

    return {
        "encoder_weight": enc_w_eff.detach().cpu().float(),
        "encoder_bias": enc_b_eff.detach().cpu().float(),
        "score_weight": score_weight.detach().cpu().float(),
        "score_bias": score_mean.detach().cpu().float(),
        "block_importance": block_importance.detach().cpu().float(),
        "scale": torch.tensor(1.0, dtype=torch.float32),
        "samples": int(x_cpu.shape[0]),
        "explained_variance_ratio": float(explained),
        "rank_effective": int(rank_eff),
        "pca_method": selected_method,
    }
