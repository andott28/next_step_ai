from __future__ import annotations

import argparse

import torch

try:
    import bitsandbytes.nn as bnn
except ImportError:
    bnn = None

from triton_sparse_mlp import (
    materialize_linear_4bit_params,
    materialize_linear_bias,
    triton_sparse_input_linear,
    triton_sparse_input_linear_4bit,
    triton_sparse_output_linear,
    triton_sparse_output_linear_4bit,
)


def _build_mask(rows: int, in_features: int, top_k: int, block_size: int, active_idx: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(rows, in_features, device=active_idx.device, dtype=torch.float16)
    for row in range(rows):
        for slot in range(top_k):
            block = int(active_idx[row, slot].item())
            start = block * block_size
            mask[row, start : start + block_size] = 1
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dense_input", "dense_output", "4bit_input", "4bit_output"], required=True)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.float16
    rows = 128
    in_features = 4096
    out_features = 14336
    top_k = 3
    block_size = 32
    num_blocks = in_features // block_size

    torch.manual_seed(0)
    x = torch.randn(rows, in_features, device=device, dtype=dtype)
    active_idx = torch.randint(0, num_blocks, (rows, top_k), device=device, dtype=torch.int32)
    flat_mask = _build_mask(rows, in_features, top_k, block_size, active_idx)

    if args.mode == "dense_input":
        weight = torch.randn(out_features, in_features, device=device, dtype=dtype)
        bias = torch.randn(out_features, device=device, dtype=dtype)
        fn = lambda: triton_sparse_input_linear(x, active_idx, weight=weight, bias=bias, block_size=block_size)
    elif args.mode == "dense_output":
        hidden = torch.randn(rows, out_features, device=device, dtype=dtype)
        weight = torch.randn(in_features, out_features, device=device, dtype=dtype)
        bias = torch.randn(in_features, device=device, dtype=dtype)
        fn = lambda: triton_sparse_output_linear(hidden, active_idx, flat_mask, weight=weight, bias=bias, block_size=block_size)
    else:
        if bnn is None:
            raise RuntimeError("bitsandbytes is required for 4-bit profiling")
        lin_in = bnn.Linear4bit(in_features, out_features, bias=True, quant_type="nf4", compute_dtype=dtype).to(device)
        lin_out = bnn.Linear4bit(out_features, in_features, bias=True, quant_type="nf4", compute_dtype=dtype).to(device)
        _ = lin_in(x)
        hidden = torch.randn(rows, out_features, device=device, dtype=dtype)
        _ = lin_out(hidden)
        in_packed, in_absmax, in_code, in_out_features, in_in_features, in_quant_block_size, _ = materialize_linear_4bit_params(
            lin_in, device=x.device
        )
        in_bias = materialize_linear_bias(lin_in, device=x.device, dtype=x.dtype)
        out_packed, out_absmax, out_code, _, out_input_dim, out_quant_block_size, _ = materialize_linear_4bit_params(
            lin_out, device=x.device
        )
        out_bias = materialize_linear_bias(lin_out, device=x.device, dtype=x.dtype)
        if args.mode == "4bit_input":
            fn = lambda: triton_sparse_input_linear_4bit(
                x,
                active_idx,
                packed_weight=in_packed,
                absmax=in_absmax,
                code=in_code,
                out_features=in_out_features,
                in_features=in_in_features,
                quant_block_size=in_quant_block_size,
                bias=in_bias,
                block_size=block_size,
                quant_weight_ref=getattr(lin_in, "weight", None),
            )
        else:
            fn = lambda: triton_sparse_output_linear_4bit(
                hidden,
                active_idx,
                flat_mask,
                packed_weight=out_packed,
                absmax=out_absmax,
                code=out_code,
                input_dim=out_input_dim,
                quant_block_size=out_quant_block_size,
                bias=out_bias,
                block_size=block_size,
                quant_weight_ref=getattr(lin_out, "weight", None),
            )

    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()
    for _ in range(args.iters):
        fn()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
