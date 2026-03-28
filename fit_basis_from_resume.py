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

from llama3_neuroplastic.basis_fitting import fit_layer_basis


def _fit_layer_basis(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    basis_rank: int,
    block_size: int,
    pca_method: str,
    pca_batch_rows: int,
) -> Dict[str, Any]:
    return fit_layer_basis(
        x=x,
        y=y,
        basis_rank=int(basis_rank),
        block_size=int(block_size),
        pca_method=str(pca_method),
        pca_batch_rows=int(pca_batch_rows),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--resume", required=True, help="Path to .resume.pt file")
    p.add_argument("--output", required=True, help="Path to write final .pt file")
    p.add_argument("--basis-rank", type=int, default=96)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--pca-method", type=str, default="auto", choices=["auto", "lowrank", "incremental"])
    p.add_argument("--pca-batch-rows", type=int, default=1024)
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
            fitted = _fit_layer_basis(
                x,
                y,
                basis_rank=args.basis_rank,
                block_size=args.block_size,
                pca_method=args.pca_method,
                pca_batch_rows=args.pca_batch_rows,
            )
            layer_states[str(layer_idx)] = fitted
            ev = fitted["explained_variance_ratio"]
            method = fitted.get("pca_method", args.pca_method)
            print(
                f"  [{i+1}/{len(layers)}] layer {layer_idx}: "
                f"rows={x.shape[0]} rank_eff={fitted['rank_effective']} pca={method} explained={ev:.3f}"
            )
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
            "pca_method": args.pca_method,
            "pca_batch_rows": args.pca_batch_rows,
            "num_blocks": num_blocks_example,
        },
    }
    torch.save(payload, str(output_path))
    print(f"\nSaved {len(layer_states)} fitted layers to {output_path}")


if __name__ == "__main__":
    main()
