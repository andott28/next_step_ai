from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

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
        description="Compare dense streaming MLP generation against sparse MLP generation for one prompt."
    )
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--sparse-basis-path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32", "fp16", "bf16", "fp32"])
    parser.add_argument("--ram-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sparse-top-k", type=int, default=None)
    parser.add_argument("--sparse-basis-top-k", type=int, default=None)
    parser.add_argument(
        "--sparse-mlp-execution",
        type=str,
        default="exact_intermediate_sparse",
        choices=[
            "auto",
            "output_basis_surrogate",
            "routed_output_blocks",
            "exact_intermediate_sparse",
            "exact_intermediate_sparse_oracle",
        ],
    )
    parser.add_argument("--sparse-mlp-prefill-mode", type=str, default="dense", choices=["dense", "sparse"])
    parser.add_argument("--require-identical", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json-out", type=str, default="")
    return parser


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_once(
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    sparse: bool,
) -> dict[str, Any]:
    runtime = StreamingLlamaRuntime(
        model_name_or_path=str(args.model_name),
        device=_parse_device(str(args.device)),
        dtype=_parse_dtype(str(args.dtype)),
        taylor_layers=[],
        local_files_only=bool(args.local_files_only),
        ram_cache=bool(args.ram_cache),
        materialize_lm_head=True,
        sparse_basis_path=str(args.sparse_basis_path) if sparse else None,
        sparse_top_k=int(args.sparse_top_k) if sparse and args.sparse_top_k is not None else None,
        sparse_basis_top_k=int(args.sparse_basis_top_k) if sparse and args.sparse_basis_top_k is not None else None,
        sparse_mlp_execution=str(args.sparse_mlp_execution) if sparse else None,
        sparse_mlp_prefill_mode=str(args.sparse_mlp_prefill_mode),
        enable_triton_fused_sparse_mlp=True,
        enable_cuda_h2d_overlap=True,
    )
    encoded = tokenizer(str(args.prompt), return_tensors="pt")
    input_ids = encoded["input_ids"].to(dtype=torch.long)
    generated = runtime.generate(
        input_ids,
        max_new_tokens=int(args.max_new_tokens),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        do_sample=False,
        top_k=None,
    )
    generated_cpu = generated.detach().to(device=torch.device("cpu"), dtype=torch.long)
    prompt_len = int(input_ids.shape[-1])
    generated_ids = [int(value) for value in generated_cpu[0].tolist()]
    completion_ids = generated_ids[prompt_len:]
    result = {
        "mode": "sparse" if sparse else "dense",
        "generated_ids": generated_ids,
        "completion_ids": completion_ids,
        "text": tokenizer.decode(generated_ids, skip_special_tokens=False),
        "completion_text": tokenizer.decode(completion_ids, skip_special_tokens=False),
        "sparse_summary": runtime.get_sparse_mlp_summary() if sparse else {},
    }
    del runtime
    _release_cuda()
    return result


def main() -> int:
    args = _build_parser().parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_name),
        use_fast=True,
        local_files_only=bool(args.local_files_only),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dense = _run_once(args=args, tokenizer=tokenizer, sparse=False)
    sparse = _run_once(args=args, tokenizer=tokenizer, sparse=True)
    identical = dense["generated_ids"] == sparse["generated_ids"]
    completion_identical = dense["completion_ids"] == sparse["completion_ids"]
    report: dict[str, Any] = {
        "model_name": str(args.model_name),
        "prompt": str(args.prompt),
        "max_new_tokens": int(args.max_new_tokens),
        "dense": dense,
        "sparse": sparse,
        "identical": bool(identical),
        "completion_identical": bool(completion_identical),
        "errors": [] if identical or not bool(args.require_identical) else ["dense and sparse generated token ids differ"],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
    return 0 if not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
