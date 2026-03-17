from __future__ import annotations

from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def triton_sca_gate_available() -> bool:
    return bool(_TRITON_AVAILABLE)


def _next_pow2(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


if _TRITON_AVAILABLE:

    @triton.jit
    def _triton_spatial_topk_kernel(
        q_ptr,
        centers_ptr,
        refractory_ptr,
        out_idx_ptr,
        out_score_ptr,
        rows: tl.constexpr,
        num_blocks: tl.constexpr,
        top_k: tl.constexpr,
        sigma_coeff,
        step,
        use_refractory: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= rows:
            return

        offs = tl.arange(0, BLOCK)
        valid = offs < num_blocks

        qx = tl.load(q_ptr + row * 3 + 0)
        qy = tl.load(q_ptr + row * 3 + 1)
        qz = tl.load(q_ptr + row * 3 + 2)

        cx = tl.load(centers_ptr + offs * 3 + 0, mask=valid, other=0.0)
        cy = tl.load(centers_ptr + offs * 3 + 1, mask=valid, other=0.0)
        cz = tl.load(centers_ptr + offs * 3 + 2, mask=valid, other=0.0)

        dx = qx - cx
        dy = qy - cy
        dz = qz - cz
        dist2 = dx * dx + dy * dy + dz * dz
        score = tl.exp(-dist2 * sigma_coeff)
        score = tl.where(valid, score, -float("inf"))

        if use_refractory:
            refr = tl.load(refractory_ptr + offs, mask=valid, other=0)
            blocked = refr > step
            score = tl.where(blocked & valid, -float("inf"), score)

        # Deterministic tie-break toward lower block index.
        score = score - tl.where(valid, offs.to(tl.float32) * 1e-6, 0.0)

        for k in tl.static_range(0, top_k):
            best_score = tl.max(score, axis=0)
            best_idx = tl.argmax(score, axis=0, tie_break_left=True)
            valid_pick = best_score > -1e20
            out_off = row * top_k + k
            tl.store(out_idx_ptr + out_off, tl.where(valid_pick, best_idx, -1))
            tl.store(out_score_ptr + out_off, tl.where(valid_pick, best_score, 0.0))
            score = tl.where(offs == best_idx, -float("inf"), score)


def triton_compute_active_blocks_topk(
    query: torch.Tensor,
    block_centers: torch.Tensor,
    *,
    sigma: float,
    top_k: int,
    refractory_until: Optional[torch.Tensor] = None,
    step: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not triton_sca_gate_available():
        raise RuntimeError("Triton is not available")
    if not query.is_cuda or not block_centers.is_cuda:
        raise RuntimeError("Triton SCA gate requires CUDA tensors")
    if query.ndim != 2 or query.shape[-1] != 3:
        raise ValueError(f"Expected query [N,3], got {tuple(query.shape)}")
    if block_centers.ndim != 2 or block_centers.shape[-1] != 3:
        raise ValueError(f"Expected block_centers [B,3], got {tuple(block_centers.shape)}")

    rows = int(query.shape[0])
    num_blocks = int(block_centers.shape[0])
    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if top_k > num_blocks:
        raise ValueError("top_k must be <= num_blocks")
    if rows == 0:
        return (
            torch.empty((0, top_k), device=query.device, dtype=torch.long),
            torch.empty((0, top_k), device=query.device, dtype=torch.float32),
        )
    if num_blocks > 1024:
        raise RuntimeError(f"Triton SCA gate currently supports <=1024 blocks, got {num_blocks}")

    q = query.contiguous().to(dtype=torch.float32)
    centers = block_centers.contiguous().to(dtype=torch.float32)
    out_idx = torch.empty((rows, top_k), device=q.device, dtype=torch.int32)
    out_score = torch.empty((rows, top_k), device=q.device, dtype=torch.float32)

    use_refractory = refractory_until is not None and int(step) >= 0
    if use_refractory:
        refractory = refractory_until.contiguous().to(device=q.device, dtype=torch.int32)
    else:
        refractory = torch.zeros((num_blocks,), device=q.device, dtype=torch.int32)

    block = _next_pow2(num_blocks)
    block = min(block, 1024)
    grid = (rows,)
    sigma_coeff = float(1.0 / max(2.0 * float(sigma) * float(sigma), 1e-12))
    _triton_spatial_topk_kernel[grid](
        q_ptr=q,
        centers_ptr=centers,
        refractory_ptr=refractory,
        out_idx_ptr=out_idx,
        out_score_ptr=out_score,
        rows=rows,
        num_blocks=num_blocks,
        top_k=top_k,
        sigma_coeff=sigma_coeff,
        step=int(step),
        use_refractory=bool(use_refractory),
        BLOCK=block,
    )
    return out_idx.to(dtype=torch.long), out_score
