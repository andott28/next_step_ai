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


def _fit_lowrank_pca(y_centered: torch.Tensor, *, rank_eff: int) -> Tuple[torch.Tensor, torch.Tensor]:
    _u, _s, v = torch.pca_lowrank(y_centered, q=int(rank_eff), center=False, niter=2)
    v = v[:, : int(rank_eff)].contiguous()
    coeff = y_centered @ v
    return v, coeff


def _fit_incremental_pca(
    y_centered: torch.Tensor,
    *,
    rank_eff: int,
    batch_rows: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    return v, coeff


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
        v, coeff = _fit_incremental_pca(y_centered, rank_eff=rank_eff, batch_rows=int(pca_batch_rows))
    else:
        v, coeff = _fit_lowrank_pca(y_centered, rank_eff=rank_eff)

    ones = torch.ones((x_cpu.shape[0], 1), dtype=x_cpu.dtype)
    x_aug = torch.cat([x_cpu, ones], dim=-1)
    lsq = torch.linalg.lstsq(x_aug, coeff)
    proj = lsq.solution
    enc_w_eff = proj[:-1, :].transpose(0, 1).contiguous()
    enc_b_eff = proj[-1, :].contiguous()

    basis = v.transpose(0, 1).contiguous()
    if rank_eff < int(basis_rank):
        pad_rows = int(basis_rank) - rank_eff
        basis = torch.cat([basis, torch.zeros((pad_rows, hidden_size), dtype=basis.dtype)], dim=0)
        enc_w_eff = torch.cat(
            [enc_w_eff, torch.zeros((pad_rows, enc_w_eff.shape[1]), dtype=enc_w_eff.dtype)],
            dim=0,
        )
        enc_b_eff = torch.cat([enc_b_eff, torch.zeros((pad_rows,), dtype=enc_b_eff.dtype)], dim=0)

    decoder_blocks = basis.view(int(basis_rank), num_blocks, int(block_size)).permute(1, 0, 2).contiguous()
    decoder_bias = y_mean.view(num_blocks, int(block_size)).contiguous()

    total_var = y_centered.pow(2).sum().clamp_min(1e-8)
    captured_var = coeff.pow(2).sum()
    explained = float((captured_var / total_var).item())
    return {
        "encoder_weight": enc_w_eff.detach().cpu().float(),
        "encoder_bias": enc_b_eff.detach().cpu().float(),
        "decoder_blocks": decoder_blocks.detach().cpu().float(),
        "decoder_bias": decoder_bias.detach().cpu().float(),
        "scale": torch.tensor(1.0, dtype=torch.float32),
        "samples": int(x_cpu.shape[0]),
        "explained_variance_ratio": explained,
        "rank_effective": int(rank_eff),
        "pca_method": selected_method,
    }
