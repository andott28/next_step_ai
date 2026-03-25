#!/usr/bin/env python3
"""Fit learned-basis routing weights from an existing resume file.

Usage:
    python fit_basis_from_resume.py \
        --resume  results/learned_basis_init_405b_96r.resume.pt \
        --output  results/learned_basis_init_405b_96r.pt \
        --basis-rank 96 \
        --block-size 32
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch


def _fit_layer_basis(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    basis_rank: int,
    block_size: int,
) -> Dict[str, Any]:
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise RuntimeError("x/y must be 2D with matching rows")
    hidden_size = int(y.shape[1])
    num_blocks = hidden_size // int(block_size)
    if num_blocks * int(block_size) != hidden_size:
        raise RuntimeError("hidden_size must be divisible by block_size")

    y_mean = y.mean(dim=0)
    y_centered = y - y_mean
    rows = int(y_centered.shape[0])
    rank_eff = int(max(min(int(basis_rank), rows, hidden_size), 1))

    _u, s, v = torch.pca_lowrank(y_centered, q=rank_eff, center=False, niter=2)
    v = v[:, :rank_eff].contiguous()
    coeff = y_centered @ v

    ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
    x_aug = torch.cat([x, ones], dim=-1)
    lsq = torch.linalg.lstsq(x_aug, coeff)
    proj = lsq.solution
    enc_w_eff = proj[:-1, :].transpose(0, 1).contiguous()
    enc_b_eff = proj[-1, :].contiguous()

    basis = v.transpose(0, 1).contiguous()
    if rank_eff < int(basis_rank):
        pad_rows = int(basis_rank) - rank_eff
        basis = torch.cat([basis, torch.zeros((pad_rows, hidden_size), dtype=basis.dtype)], dim=0)
        enc_w_eff = torch.cat(
            [enc_w_eff, torch.zeros((pad_rows, enc_w_eff.shape[1]), dtype=enc_w_eff.dtype)], dim=0
        )
        enc_b_eff = torch.cat([enc_b_eff, torch.zeros((pad_rows,), dtype=enc_b_eff.dtype)], dim=0)

    decoder_blocks = basis.view(int(basis_rank), num_blocks, int(block_size)).permute(1, 0, 2).contiguous()
    decoder_bias = y_mean.view(num_blocks, int(block_size)).contiguous()

    total_var = y_centered.pow(2).sum().clamp_min(1e-8)
    captured_var = coeff.pow(2).sum()
    explained = float((captured_var / total_var).detach().cpu().item())
    return {
        "encoder_weight": enc_w_eff.detach().cpu().float(),
        "encoder_bias": enc_b_eff.detach().cpu().float(),
        "decoder_blocks": decoder_blocks.detach().cpu().float(),
        "decoder_bias": decoder_bias.detach().cpu().float(),
        "scale": torch.tensor(1.0, dtype=torch.float32),
        "samples": int(x.shape[0]),
        "explained_variance_ratio": explained,
        "rank_effective": int(rank_eff),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--resume", required=True, help="Path to .resume.pt file")
    p.add_argument("--output", required=True, help="Path to write final .pt file")
    p.add_argument("--basis-rank", type=int, default=96)
    p.add_argument("--block-size", type=int, default=32)
    args = p.parse_args()

    resume_path = Path(args.resume)
    output_path = Path(args.output)

    print(f"Loading resume from {resume_path} ...")
    data = torch.load(str(resume_path), map_location="cpu", weights_only=False)
    layer_x: Dict = data.get("layer_x", {})
    layer_y: Dict = data.get("layer_y", {})

    if not layer_x:
        print("ERROR: no layer_x data found in resume file.")
        return

    layer_states: Dict[str, Any] = {}
    layers = sorted(int(k) for k in layer_x.keys())
    print(f"Fitting {len(layers)} layers (rank={args.basis_rank}, block_size={args.block_size}) ...")

    for i, layer_idx in enumerate(layers):
        xs = layer_x.get(layer_idx) or layer_x.get(str(layer_idx))
        ys = layer_y.get(layer_idx) or layer_y.get(str(layer_idx))
        if not xs or not ys:
            print(f"  [{i+1}/{len(layers)}] layer {layer_idx}: SKIP (no data)")
            continue

        x = torch.cat([t.view(1, -1) if t.ndim == 1 else t.view(t.shape[0], -1) for t in xs], dim=0).float()
        y = torch.cat([t.view(1, -1) if t.ndim == 1 else t.view(t.shape[0], -1) for t in ys], dim=0).float()

        try:
            fitted = _fit_layer_basis(x, y, basis_rank=args.basis_rank, block_size=args.block_size)
            layer_states[str(layer_idx)] = fitted
            ev = fitted["explained_variance_ratio"]
            print(f"  [{i+1}/{len(layers)}] layer {layer_idx}: rows={x.shape[0]} rank_eff={fitted['rank_effective']} explained={ev:.3f}")
        except Exception as e:
            print(f"  [{i+1}/{len(layers)}] layer {layer_idx}: ERROR {e}")

    num_blocks_example = None
    block_size_example = None
    if layer_states:
        ex = next(iter(layer_states.values()))
        num_blocks_example = ex["decoder_blocks"].shape[0]
        block_size_example = ex["decoder_blocks"].shape[2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "layer_states": layer_states,
        "config": {
            "sparse_placement": "learned_basis",
            "basis_rank": args.basis_rank,
            "block_size": args.block_size,
            "num_blocks": num_blocks_example,
        },
    }
    torch.save(payload, str(output_path))
    print(f"\nSaved {len(layer_states)} fitted layers to {output_path}")


if __name__ == "__main__":
    main()
