#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import bitsandbytes.functional as bnb_functional
import torch

try:
    from .neuroplastic_llama_gqa_mamba import NeuroplasticLlama
except ImportError:  # pragma: no cover
    from neuroplastic_llama_gqa_mamba import NeuroplasticLlama


def _materialize_weight(linear: torch.nn.Module) -> torch.Tensor:
    weight = getattr(linear, "weight", None)
    if torch.is_tensor(weight) and weight.is_floating_point() and weight.ndim == 2:
        return weight.detach().float().cpu().contiguous()
    if torch.is_tensor(weight) and getattr(weight, "quant_state", None) is not None:
        quant_state = weight.quant_state
        dense = bnb_functional.dequantize_4bit(weight, quant_state=quant_state)
        target_shape = tuple(getattr(quant_state, "shape", ()))
        if len(target_shape) == 2 and tuple(dense.shape) != target_shape:
            dense = dense.reshape(target_shape)
        return dense.detach().float().cpu().contiguous()
    raise RuntimeError(f"Unsupported linear weight type for export: {type(weight)!r}")


def _materialize_bias(linear: torch.nn.Module) -> torch.Tensor | None:
    bias = getattr(linear, "bias", None)
    if bias is None:
        return None
    return bias.detach().float().cpu().contiguous()


def _parse_layer_selection(spec: Optional[str]) -> Optional[List[int]]:
    if spec is None or spec.strip() == "":
        return None
    out: set[int] = set()
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid layer range: {token}")
            out.update(range(start, end + 1))
        else:
            out.add(int(token))
    return sorted(out)


def _resolve_dtype(name: str) -> torch.dtype:
    normalized = str(name).strip().lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported export dtype: {name}")


def main() -> None:
    p = argparse.ArgumentParser(description="Export sparse MLP block banks and manifest without changing base weights.")
    p.add_argument("--model-name", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="results/sparse_mlp_bank")
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--layers", type=str, default="", help="Layer selection, e.g. '8-23' or '8,9,10'")
    p.add_argument("--dtype", type=str, default="float16", help="Export dtype: float16|bfloat16|float32")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_layers = _parse_layer_selection(args.layers)
    selected_set = set(selected_layers) if selected_layers is not None else None
    export_dtype = _resolve_dtype(args.dtype)

    model = NeuroplasticLlama(
        model_name=args.model_name,
        neuroplasticity_enabled=False,
        sca_block_size=args.block_size,
        sca_top_k=args.top_k,
        sca_use_cuda=False,
    )

    manifest: Dict[str, Any] = {
        "model_name": args.model_name,
        "sca_config": model.sca_config.to_dict(),
        "export_dtype": str(export_dtype).replace("torch.", ""),
        "layer_selection": selected_layers,
        "layers": [],
    }

    for layer_idx, wrapper in enumerate(model.sca_sparse_mlps):
        if selected_set is not None and layer_idx not in selected_set:
            continue

        gate_weight = _materialize_weight(wrapper.base_mlp.gate_proj)
        up_weight = _materialize_weight(wrapper.base_mlp.up_proj)
        down_weight = _materialize_weight(wrapper.base_mlp.down_proj)
        gate_bias = _materialize_bias(wrapper.base_mlp.gate_proj)
        up_bias = _materialize_bias(wrapper.base_mlp.up_proj)
        down_bias = _materialize_bias(wrapper.base_mlp.down_proj)

        num_blocks = int(wrapper.num_blocks)
        block_size = int(wrapper.block_size)
        intermediate = int(gate_weight.shape[0])

        gate_blocks = gate_weight.reshape(intermediate, num_blocks, block_size).permute(1, 0, 2).contiguous().to(dtype=export_dtype)
        up_blocks = up_weight.reshape(intermediate, num_blocks, block_size).permute(1, 0, 2).contiguous().to(dtype=export_dtype)
        down_blocks = down_weight.reshape(num_blocks, block_size, down_weight.shape[1]).contiguous().to(dtype=export_dtype)
        if gate_bias is not None:
            gate_bias = gate_bias.to(dtype=export_dtype)
        if up_bias is not None:
            up_bias = up_bias.to(dtype=export_dtype)
        if down_bias is not None:
            down_bias = down_bias.to(dtype=export_dtype)

        layer_file = out_dir / f"layer_{layer_idx:03d}_mlp_blocks.pt"
        torch.save(
            {
                "layer_idx": layer_idx,
                "num_blocks": num_blocks,
                "block_size": block_size,
                "gate_proj_blocks": gate_blocks,
                "up_proj_blocks": up_blocks,
                "down_proj_blocks": down_blocks,
                "gate_bias": gate_bias,
                "up_bias": up_bias,
                "down_bias": down_bias,
            },
            layer_file,
        )

        manifest["layers"].append(
            {
                "layer_idx": layer_idx,
                "file": layer_file.name,
                "num_blocks": num_blocks,
                "block_size": block_size,
                "intermediate_size": intermediate,
                "gate_blocks_shape": list(gate_blocks.shape),
                "up_blocks_shape": list(up_blocks.shape),
                "down_blocks_shape": list(down_blocks.shape),
            }
        )

    manifest_path = out_dir / "mlp_bank_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "layers": len(manifest["layers"]),
                "manifest_path": str(manifest_path),
                "export_dtype": manifest["export_dtype"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
