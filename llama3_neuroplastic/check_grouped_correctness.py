#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

try:
    from .neuroplastic_llama import NeuroplasticLlama
    from .sca_sparse_config import SCASparseConfig
    from .sca_sparse_mlp import SparseLlamaMLP, SparseRouteSelection
except ImportError:  # Script-mode fallback
    from neuroplastic_llama import NeuroplasticLlama
    from sca_sparse_config import SCASparseConfig
    from sca_sparse_mlp import SparseLlamaMLP, SparseRouteSelection


LOGGER = logging.getLogger("check_grouped_correctness")


@dataclass
class ShapeCase:
    name: str
    rows: int
    hidden_size: int
    intermediate_size: int
    top_k: int
    block_size: int


class _ToyMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=True, device=device, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=True, device=device, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True, device=device, dtype=dtype)
        self.act_fn = nn.SiLU()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_active_idx(rows: int, num_blocks: int, top_k: int, device: torch.device) -> torch.Tensor:
    base = torch.arange(rows, device=device, dtype=torch.long).unsqueeze(1)
    offs = torch.arange(top_k, device=device, dtype=torch.long).unsqueeze(0)
    return ((base + offs) % max(num_blocks, 1)).contiguous()


def _error_metrics(ref: torch.Tensor, got: torch.Tensor) -> Dict[str, float]:
    diff = (got - ref).abs().float()
    return {
        "max_abs_error": float(diff.max().item()) if diff.numel() > 0 else 0.0,
        "mean_abs_error": float(diff.mean().item()) if diff.numel() > 0 else 0.0,
        "ref_max_abs": float(ref.abs().float().max().item()) if ref.numel() > 0 else 0.0,
    }


def _is_pass(metrics: Dict[str, float], atol: float, rtol: float) -> bool:
    return bool(metrics["max_abs_error"] <= (atol + (rtol * metrics["ref_max_abs"])))


def _run_dense_case(case: ShapeCase, device: torch.device, dtype: torch.dtype, seed: int, atol: float, rtol: float) -> Dict[str, Any]:
    _set_seed(seed)
    cfg = SCASparseConfig(
        hidden_size=case.hidden_size,
        block_size=case.block_size,
        top_k=case.top_k,
        spmm_impl="dense",
        soft_mask=False,
        grouped_row_gemm=True,
        grouped_row_min_bucket=2,
        grouped_row_allow_4bit_dequant=False,
    )
    mlp = _ToyMLP(case.hidden_size, case.intermediate_size, device=device, dtype=dtype)
    route = SparseRouteSelection(active_idx=torch.empty((0, case.top_k), device=device, dtype=torch.long))
    wrapper = SparseLlamaMLP(
        base_mlp=mlp,
        config=cfg,
        layer_idx=0,
        route_fn=lambda _h, _l: route,
        enabled_fn=lambda _l: True,
    )

    hidden = torch.randn((case.rows, case.hidden_size), device=device, dtype=dtype)
    active_idx = _build_active_idx(case.rows, cfg.num_blocks, case.top_k, device=device)
    feature_mask = wrapper._build_feature_mask(
        active_idx=active_idx,
        score_weights=None,
        dtype=hidden.dtype,
        device=hidden.device,
    )

    with torch.inference_mode():
        ref = wrapper._forward_dense_masked(hidden, feature_mask)
        got = wrapper._forward_grouped_row_gemm(hidden, active_idx, feature_mask)

    metrics = _error_metrics(ref, got)
    passed = _is_pass(metrics, atol=atol, rtol=rtol)
    return {
        "group": "dense_grouped_vs_dense_masked",
        "case": case.name,
        "shape": {
            "rows": int(case.rows),
            "in_features": int(case.hidden_size),
            "out_features": int(case.intermediate_size),
            "top_k": int(case.top_k),
            "block_size": int(case.block_size),
        },
        **metrics,
        "atol": float(atol),
        "rtol": float(rtol),
        "pass": bool(passed),
    }


def _run_4bit_case(
    wrapper: SparseLlamaMLP,
    *,
    rows: int,
    top_k: int,
    grouped_min_bucket: int,
    atol: float,
    rtol: float,
    seed: int,
) -> Dict[str, Any]:
    _set_seed(seed)
    device = wrapper.base_mlp.gate_proj.weight.device
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    hidden_size = int(wrapper.hidden_size)
    intermediate_size = int(getattr(wrapper.base_mlp.gate_proj.weight, "shape", [0])[0])

    wrapper.config.top_k = int(max(1, min(top_k, wrapper.config.num_blocks)))
    wrapper.config.grouped_row_gemm = True
    wrapper.config.grouped_row_min_bucket = int(max(1, grouped_min_bucket))
    wrapper.config.grouped_row_allow_4bit_dequant = True

    hidden = torch.randn((rows, hidden_size), device=device, dtype=dtype)
    active_idx = _build_active_idx(rows, wrapper.config.num_blocks, wrapper.config.top_k, device=device)
    feature_mask = wrapper._build_feature_mask(
        active_idx=active_idx,
        score_weights=None,
        dtype=hidden.dtype,
        device=hidden.device,
    )

    with torch.inference_mode():
        ref = wrapper._forward_dense_masked(hidden, feature_mask)
        got = wrapper._forward_grouped_row_gemm(hidden, active_idx, feature_mask)

    metrics = _error_metrics(ref, got)
    passed = _is_pass(metrics, atol=atol, rtol=rtol)
    return {
        "group": "grouped_4bit_dequant_vs_bitsandbytes_dense_reference",
        "case": f"rows_{rows}",
        "shape": {
            "rows": int(rows),
            "in_features": int(hidden_size),
            "out_features": int(intermediate_size),
            "top_k": int(wrapper.config.top_k),
            "block_size": int(wrapper.config.block_size),
        },
        **metrics,
        "atol": float(atol),
        "rtol": float(rtol),
        "pass": bool(passed),
    }


def _resolve_checkpoint_dir(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    marker = os.path.join(path, "neuroplastic_llama_sca_v2.bin")
    if os.path.isfile(marker):
        return path
    return None


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Correctness checks for grouped-row GEMM paths.")
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--model-name", type=str, default="unsloth/Meta-Llama-3.1-8B-bnb-4bit")
    p.add_argument("--layer-idx", type=int, default=0)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--grouped-min-bucket", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dense-atol", type=float, default=2e-3)
    p.add_argument("--dense-rtol", type=float, default=2e-2)
    p.add_argument("--q4-atol", type=float, default=8e-3)
    p.add_argument("--q4-rtol", type=float, default=5e-2)
    p.add_argument("--skip-4bit", action="store_true")
    p.add_argument("--output-json", type=str, default=None)
    return p


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = _build_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    dense_cases = [
        ShapeCase("small_a", rows=4, hidden_size=128, intermediate_size=256, top_k=2, block_size=32),
        ShapeCase("small_b", rows=8, hidden_size=256, intermediate_size=512, top_k=3, block_size=32),
        ShapeCase("real_shape", rows=128, hidden_size=4096, intermediate_size=14336, top_k=3, block_size=32),
    ]

    results: List[Dict[str, Any]] = []
    for idx, case in enumerate(dense_cases):
        LOGGER.info("Dense correctness case: %s", case.name)
        row = _run_dense_case(
            case=case,
            device=device,
            dtype=dtype,
            seed=int(args.seed) + idx,
            atol=float(args.dense_atol),
            rtol=float(args.dense_rtol),
        )
        results.append(row)

    if not args.skip_4bit:
        resolved_ckpt = _resolve_checkpoint_dir(args.checkpoint_dir)
        if resolved_ckpt is not None:
            LOGGER.info("Loading checkpoint model from: %s", resolved_ckpt)
            model = NeuroplasticLlama.from_pretrained(resolved_ckpt, neuroplasticity_enabled=True)
        else:
            LOGGER.info("Loading base model from: %s", args.model_name)
            model = NeuroplasticLlama(model_name=args.model_name, neuroplasticity_enabled=True)
        model.eval()
        wrapper = model.sca_sparse_mlps[int(args.layer_idx)]

        for rows in (2, 8, 128):
            LOGGER.info("4-bit correctness case: rows=%d", rows)
            row = _run_4bit_case(
                wrapper=wrapper,
                rows=rows,
                top_k=int(args.top_k),
                grouped_min_bucket=int(args.grouped_min_bucket),
                atol=float(args.q4_atol),
                rtol=float(args.q4_rtol),
                seed=int(args.seed) + rows,
            )
            results.append(row)

    failed = [r for r in results if not bool(r["pass"])]
    summary = {
        "total_cases": int(len(results)),
        "passed_cases": int(len(results) - len(failed)),
        "failed_cases": int(len(failed)),
        "all_passed": bool(len(failed) == 0),
    }
    payload = {"summary": summary, "results": results}
    print(json.dumps(payload, indent=2))

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        LOGGER.info("Wrote correctness JSON: %s", args.output_json)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
