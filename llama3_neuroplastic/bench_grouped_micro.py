#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

try:
    import triton
except ImportError:  # pragma: no cover
    triton = None

try:
    from .neuroplastic_llama import NeuroplasticLlama
    from .sca_sparse_config import SCASparseConfig
    from .sca_sparse_mlp import SparseLlamaMLP, SparseRouteSelection
    from .triton_sparse_mlp import materialize_linear_bias, materialize_linear_weight
except ImportError:  # Script-mode fallback
    from neuroplastic_llama import NeuroplasticLlama
    from sca_sparse_config import SCASparseConfig
    from sca_sparse_mlp import SparseLlamaMLP, SparseRouteSelection
    from triton_sparse_mlp import materialize_linear_bias, materialize_linear_weight


LOGGER = logging.getLogger("bench_grouped_micro")


@dataclass
class MicroConfig:
    run_name: str
    checkpoint_dir: Optional[str]
    model_name: str
    seed: int
    top_k: int
    block_size: int
    grouped_min_bucket: int
    warmup: int
    rep: int
    output_json: Optional[str]


class _DenseMLP(nn.Module):
    def __init__(
        self,
        gate_proj: nn.Linear,
        up_proj: nn.Linear,
        down_proj: nn.Linear,
    ) -> None:
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.act_fn = nn.SiLU()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_checkpoint_dir(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    marker = os.path.join(path, "neuroplastic_llama_sca_v2.bin")
    if os.path.isfile(marker):
        return path
    return None


def _build_active_idx(rows: int, top_k: int, device: torch.device) -> torch.Tensor:
    pattern = torch.arange(top_k, device=device, dtype=torch.long).unsqueeze(0)
    return pattern.repeat(rows, 1).contiguous()


def _materialize_dense_linear(linear: nn.Module, *, device: torch.device, dtype: torch.dtype) -> nn.Linear:
    weight = materialize_linear_weight(linear, device=device, dtype=dtype)
    bias = materialize_linear_bias(linear, device=device, dtype=dtype)
    dense = nn.Linear(int(weight.shape[1]), int(weight.shape[0]), bias=bias is not None)
    with torch.inference_mode():
        dense.weight.copy_(weight)
        if bias is not None and dense.bias is not None:
            dense.bias.copy_(bias)
    return dense.to(device=device, dtype=dtype)


def _build_dense_grouped_wrapper(src_wrapper: SparseLlamaMLP, dtype: torch.dtype) -> SparseLlamaMLP:
    device = src_wrapper.base_mlp.gate_proj.weight.device
    gate = _materialize_dense_linear(src_wrapper.base_mlp.gate_proj, device=device, dtype=dtype)
    up = _materialize_dense_linear(src_wrapper.base_mlp.up_proj, device=device, dtype=dtype)
    down = _materialize_dense_linear(src_wrapper.base_mlp.down_proj, device=device, dtype=dtype)
    dense_mlp = _DenseMLP(gate_proj=gate, up_proj=up, down_proj=down)
    cfg = SCASparseConfig(
        hidden_size=int(src_wrapper.hidden_size),
        block_size=int(src_wrapper.block_size),
        top_k=int(src_wrapper.config.top_k),
        block_rank=int(src_wrapper.config.block_rank),
        sigma=float(src_wrapper.config.sigma),
        refractory_steps=int(src_wrapper.config.refractory_steps),
        inhibition_lambda=float(src_wrapper.config.inhibition_lambda),
        use_cuda=bool(src_wrapper.config.use_cuda),
        grid_size=int(src_wrapper.config.grid_size),
        spmm_impl="dense",
        soft_mask=False,
        grouped_row_gemm=True,
        grouped_row_min_bucket=int(src_wrapper.config.grouped_row_min_bucket),
        grouped_row_allow_4bit_dequant=False,
    )
    route = SparseRouteSelection(active_idx=torch.empty((0, cfg.top_k), device=device, dtype=torch.long))
    return SparseLlamaMLP(
        base_mlp=dense_mlp,
        config=cfg,
        layer_idx=0,
        route_fn=lambda _h, _l: route,
        enabled_fn=lambda _l: True,
    ).to(device=device, dtype=dtype)


def _bench_quantiles_ms(fn: Any, warmup: int, rep: int) -> Dict[str, float]:
    p20, p50, p80 = triton.testing.do_bench(
        fn,
        warmup=warmup,
        rep=rep,
        quantiles=[0.2, 0.5, 0.8],
    )
    return {
        "p20_ms": float(p20),
        "median_ms": float(p50),
        "p80_ms": float(p80),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Microbench grouped-row kernels vs current 4-bit Triton sparse path.")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--model-name", type=str, default="unsloth/Meta-Llama-3.1-8B-bnb-4bit")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--grouped-min-bucket", type=int, default=2)
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--rep", type=int, default=100)
    p.add_argument("--output-json", type=str, default=None)
    return p


def _to_cfg(args: argparse.Namespace) -> MicroConfig:
    run_name = args.run_name
    if run_name is None:
        run_name = f"grouped_micro_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return MicroConfig(
        run_name=str(run_name),
        checkpoint_dir=args.checkpoint_dir,
        model_name=str(args.model_name),
        seed=int(args.seed),
        top_k=int(args.top_k),
        block_size=int(args.block_size),
        grouped_min_bucket=int(args.grouped_min_bucket),
        warmup=int(args.warmup),
        rep=int(args.rep),
        output_json=args.output_json,
    )


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = _build_parser().parse_args()
    cfg = _to_cfg(args)
    _set_seed(cfg.seed)

    if triton is None:
        raise RuntimeError("triton is required for bench_grouped_micro.py")
    if not torch.cuda.is_available():
        raise RuntimeError("bench_grouped_micro.py requires CUDA")

    resolved_ckpt = _resolve_checkpoint_dir(cfg.checkpoint_dir)
    if resolved_ckpt is not None:
        LOGGER.info("Loading checkpoint model from: %s", resolved_ckpt)
        model = NeuroplasticLlama.from_pretrained(resolved_ckpt, neuroplasticity_enabled=True)
    else:
        LOGGER.info("Loading base model from: %s", cfg.model_name)
        model = NeuroplasticLlama(model_name=cfg.model_name, neuroplasticity_enabled=True)
    model.eval()
    model.collect_bio_gate_telemetry = False
    model.sca_config.top_k = int(max(1, min(cfg.top_k, model.sca_config.num_blocks)))
    model.sca_config.grouped_row_min_bucket = int(max(1, cfg.grouped_min_bucket))
    model.sca_config.grouped_row_allow_4bit_dequant = True
    model.sca_config.grouped_row_gemm = True

    if int(model.sca_config.block_size) != int(cfg.block_size):
        raise ValueError(
            f"Requested block_size={cfg.block_size} but loaded model uses block_size={model.sca_config.block_size}. "
            "Changing block size post-load is unsupported for initialized sparse wrappers."
        )

    wrapper_4bit = model.sca_sparse_mlps[0]
    dense_wrapper = _build_dense_grouped_wrapper(wrapper_4bit, dtype=torch.float16)
    dense_wrapper.config.top_k = int(model.sca_config.top_k)
    dense_wrapper.config.grouped_row_min_bucket = int(model.sca_config.grouped_row_min_bucket)
    dense_wrapper.config.grouped_row_gemm = True
    dense_wrapper.config.grouped_row_allow_4bit_dequant = False

    row_sweep = [1, 2, 4, 8, 32, 128]
    out_rows: List[Dict[str, Any]] = []

    for rows in row_sweep:
        LOGGER.info("Benchmark rows=%d", rows)
        hidden = torch.randn((rows, 4096), device=model.device, dtype=torch.float16)
        active_idx = _build_active_idx(rows, int(model.sca_config.top_k), device=model.device)
        feature_mask = wrapper_4bit._build_feature_mask(
            active_idx=active_idx,
            score_weights=None,
            dtype=hidden.dtype,
            device=hidden.device,
        )

        def _baseline() -> torch.Tensor:
            return wrapper_4bit._forward_triton_sparse_4bit(hidden, active_idx, feature_mask)

        def _grouped_dense() -> torch.Tensor:
            return dense_wrapper._forward_grouped_row_gemm(hidden, active_idx, feature_mask)

        def _grouped_q4() -> torch.Tensor:
            return wrapper_4bit._forward_grouped_row_gemm(hidden, active_idx, feature_mask)

        baseline = _bench_quantiles_ms(_baseline, warmup=cfg.warmup, rep=cfg.rep)
        grouped_dense = _bench_quantiles_ms(_grouped_dense, warmup=cfg.warmup, rep=cfg.rep)
        grouped_q4 = _bench_quantiles_ms(_grouped_q4, warmup=cfg.warmup, rep=cfg.rep)

        mode_rows = [
            ("current_4bit_triton_sparse_path", baseline),
            ("grouped_dense_gemm_materialized_columns", grouped_dense),
            ("grouped_4bit_dequant_to_dense", grouped_q4),
        ]
        baseline_median = max(float(baseline["median_ms"]), 1e-9)
        for mode, stats in mode_rows:
            out_rows.append(
                {
                    "mode": mode,
                    "rows": int(rows),
                    "median_ms": float(stats["median_ms"]),
                    "p20_ms": float(stats["p20_ms"]),
                    "p80_ms": float(stats["p80_ms"]),
                    "estimated_speedup_vs_baseline": float(baseline_median / max(float(stats["median_ms"]), 1e-9)),
                }
            )

    payload = {
        "run_name": cfg.run_name,
        "checkpoint_dir": resolved_ckpt,
        "seed": int(cfg.seed),
        "shape": {
            "rows": -1,
            "in_features": 4096,
            "out_features": 14336,
            "top_k": int(model.sca_config.top_k),
            "block_size": int(model.sca_config.block_size),
        },
        "results": out_rows,
    }
    print(json.dumps(payload, indent=2))

    if cfg.output_json:
        os.makedirs(os.path.dirname(cfg.output_json) or ".", exist_ok=True)
        with open(cfg.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        LOGGER.info("Wrote microbench JSON: %s", cfg.output_json)


if __name__ == "__main__":
    run()
