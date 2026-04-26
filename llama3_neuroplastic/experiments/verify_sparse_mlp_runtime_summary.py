from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

try:
    from .streaming_llama_runtime import StreamingLlamaRuntime
except ImportError:
    from streaming_llama_runtime import StreamingLlamaRuntime


def _parse_device(device_name: str) -> torch.device:
    name = str(device_name).strip().lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _parse_dtype(dtype_name: str) -> torch.dtype:
    name = str(dtype_name).strip().lower()
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load StreamingLlamaRuntime with a sparse MLP checkpoint and validate routing summary metadata."
    )
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--sparse-basis-path", type=str, required=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32", "fp16", "bf16", "fp32"])
    parser.add_argument("--sparse-top-k", type=int, default=None)
    parser.add_argument("--sparse-basis-top-k", type=int, default=None)
    parser.add_argument(
        "--sparse-mlp-execution",
        type=str,
        default="auto",
        choices=[
            "auto",
            "exact_blockwise_sparse",
            "exact_intermediate_sparse",
        ],
    )
    parser.add_argument("--sparse-mlp-prefill-mode", type=str, default="dense", choices=["dense", "sparse"])
    parser.add_argument("--expect-execution", type=str, default="", help="Require every sparse layer to resolve to this mode.")
    parser.add_argument("--expect-target", type=str, default="", help="Require every sparse layer to use this artifact target.")
    parser.add_argument("--expect-block-domain", type=str, default="", help="Require every sparse layer to use this block domain.")
    parser.add_argument("--expect-layers", type=int, default=0, help="Require exactly this many loaded sparse layers when > 0.")
    parser.add_argument("--expect-num-blocks", type=int, default=0, help="Require every sparse layer to expose this block count.")
    parser.add_argument("--min-layers", type=int, default=1)
    parser.add_argument("--max-layers-print", type=int, default=8)
    parser.add_argument("--json-out", type=str, default="")
    return parser


def _count_matches(counts: dict[str, int], key: str, expected_total: int) -> bool:
    return int(counts.get(str(key), 0)) == int(expected_total)


def main() -> int:
    args = _build_parser().parse_args()
    runtime = StreamingLlamaRuntime(
        model_name_or_path=str(args.model_name),
        device=_parse_device(str(args.device)),
        dtype=_parse_dtype(str(args.dtype)),
        taylor_layers=[],
        local_files_only=bool(args.local_files_only),
        ram_cache=False,
        materialize_lm_head=False,
        sparse_basis_path=str(args.sparse_basis_path),
        sparse_top_k=int(args.sparse_top_k) if args.sparse_top_k is not None else None,
        sparse_basis_top_k=int(args.sparse_basis_top_k) if args.sparse_basis_top_k is not None else None,
        sparse_mlp_execution=str(args.sparse_mlp_execution),
        sparse_mlp_prefill_mode=str(args.sparse_mlp_prefill_mode),
        enable_triton_fused_sparse_mlp=False,
        enable_cuda_h2d_overlap=False,
    )

    summary = runtime.get_sparse_mlp_summary()
    layer_infos: list[dict[str, Any]] = []
    for layer_idx in sorted(runtime._sparse_routing.keys()):
        info = runtime.get_sparse_mlp_layer_info(int(layer_idx))
        if info is not None:
            layer_infos.append(info)

    errors: list[str] = []
    loaded_layers = int(summary.get("layers", 0))
    if loaded_layers < int(args.min_layers):
        errors.append(f"loaded sparse layers {loaded_layers} < required minimum {int(args.min_layers)}")
    if int(args.expect_layers) > 0 and loaded_layers != int(args.expect_layers):
        errors.append(f"loaded sparse layers {loaded_layers} != expected {int(args.expect_layers)}")
    if args.expect_execution and not _count_matches(summary.get("execution_counts", {}), str(args.expect_execution), loaded_layers):
        errors.append(
            f"execution_counts={summary.get('execution_counts', {})} does not contain only {args.expect_execution!r}"
        )
    if args.expect_target and not _count_matches(summary.get("artifact_target_counts", {}), str(args.expect_target), loaded_layers):
        errors.append(
            f"artifact_target_counts={summary.get('artifact_target_counts', {})} does not contain only {args.expect_target!r}"
        )
    if args.expect_block_domain and not _count_matches(
        summary.get("block_domain_counts", {}),
        str(args.expect_block_domain),
        loaded_layers,
    ):
        errors.append(
            f"block_domain_counts={summary.get('block_domain_counts', {})} does not contain only "
            f"{args.expect_block_domain!r}"
        )
    unique_num_blocks = sorted({int(info["num_blocks"]) for info in layer_infos if "num_blocks" in info})
    if int(args.expect_num_blocks) > 0 and unique_num_blocks != [int(args.expect_num_blocks)]:
        errors.append(f"unique_num_blocks={unique_num_blocks} != expected [{int(args.expect_num_blocks)}]")

    report: dict[str, Any] = {
        "model_name": str(args.model_name),
        "sparse_basis_path": str(args.sparse_basis_path),
        "runtime_summary": summary,
        "unique_num_blocks": unique_num_blocks,
        "sample_layers": layer_infos[: max(0, int(args.max_layers_print))],
        "errors": errors,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
