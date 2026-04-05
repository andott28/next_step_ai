#!/usr/bin/env python3
from __future__ import annotations

import os

if os.getenv("STREAMING_DEBUG_SYNC_CUDA", "").strip().lower() in {"1", "true", "yes", "on"}:
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

try:
    from .streaming_llama_runtime import StreamingLlamaRuntime
except ImportError:  # pragma: no cover
    from streaming_llama_runtime import StreamingLlamaRuntime


def _parse_layer_selection(spec: str | None) -> Optional[List[int]]:
    if spec is None:
        return None
    stripped = str(spec).strip()
    if stripped == "" or stripped.lower() == "all":
        return None
    if stripped.lower() in {"none", "off", "disable", "disabled"}:
        return []
    out: set[int] = set()
    for part in stripped.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid layer range: {token}")
            out.update(range(start, end + 1))
        else:
            out.add(int(token))
    return sorted(out)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run greedy decode with layer-by-layer streamed Llama weights.")
    p.add_argument("--model-name", type=str, required=True, help="HF repo id or local snapshot directory")
    p.add_argument("--prompt", action="append", default=[], help="Prompt to run; may be repeated")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--taylor-layers",
        type=str,
        default="",
        help="Taylor-attention layer selection, for example '0-31'. Omit or use 'all' for the runtime default, use 'none' to disable Taylor attention.",
    )
    p.add_argument(
        "--taylor-feature-map",
        type=str,
        default="hybrid_performer",
        choices=["hybrid_performer", "elu", "taylor"],
    )
    p.add_argument("--taylor-local-window", type=int, default=64)
    p.add_argument("--taylor-feature-dim", type=int, default=64)
    p.add_argument("--taylor-state-decay", type=float, default=1.0)
    p.add_argument("--dump-json", type=str, default="")
    p.add_argument(
        "--max-runtime-layers",
        type=int,
        default=0,
        help="Optional smoke-test cap for runtime.num_layers. Keeps the exact command surface but limits layer passes.",
    )
    p.add_argument(
        "--no-ram-cache",
        action="store_true",
        default=False,
        help="Disable the RAM weight cache (re-reads from SSD every token). Use only when RAM is severely constrained.",
    )
    p.add_argument(
        "--no-ram-cache-pinned",
        action="store_true",
        default=False,
        help="Keep RAM cache pageable (disables pin_memory). Default keeps cache pinned for faster PCIe DMA.",
    )
    p.add_argument(
        "--sparse-basis-path",
        type=str,
        default="",
        help="Path to learned-basis checkpoint (.pt) from init_learned_basis_from_dense_mlp.py. "
             "Enables sparse MLP execution (~2%% of neurons active per layer).",
    )
    p.add_argument(
        "--sparse-top-k",
        type=int,
        default=None,
        help="Override number of active MLP blocks per token (default: 2%% of blocks from checkpoint config).",
    )
    p.add_argument(
        "--vram-hot-cache-gb",
        type=float,
        default=None,
        help="VRAM budget (GB) for persistent NF4 hot-block cache. Default uses STREAMING_VRAM_HOT_CACHE_GB or 6.4 GB.",
    )
    p.add_argument(
        "--hot-block-threshold",
        type=float,
        default=0.80,
        help="Layer-local score threshold for selecting deterministic hot MLP blocks for VRAM cache.",
    )
    p.add_argument(
        "--attn-head-importance-path",
        type=str,
        default="",
        help="Path to attention head importance checkpoint (.pt) from "
             "init_learned_attn_head_importance.py. Enables sparse attention head loading "
             "(only top-K heads' NF4 bytes transferred per token, reducing attention PCIe traffic).",
    )
    p.add_argument(
        "--attn-share-path",
        type=str,
        default="",
        help="Path to a no-training cross-layer attention sharing checkpoint (.pt). "
             "When present, decode q_proj/o_proj are reconstructed from shared low-rank "
             "group factors plus per-layer residuals instead of per-layer dense q/o loads.",
    )
    p.add_argument(
        "--attn-active-heads",
        type=int,
        default=None,
        help="Default number of active attention heads per layer before dynamic thresholding.",
    )
    p.add_argument(
        "--attn-head-activity-threshold",
        type=float,
        default=0.10,
        help="Natural sparsity threshold over combined static importance × Taylor state_z norm.",
    )
    p.add_argument(
        "--attn-min-active-heads",
        type=int,
        default=16,
        help="Lower bound for dynamic active attention heads.",
    )
    p.add_argument(
        "--attn-max-active-heads",
        type=int,
        default=None,
        help="Upper bound for dynamic active attention heads (default: same as active-head default).",
    )
    p.add_argument(
        "--disable-triton-fused-sparse-mlp",
        action="store_true",
        default=False,
        help="Disable Triton fused NF4 sparse MLP kernels and use dequant + F.linear fallback.",
    )
    p.add_argument(
        "--disable-cuda-h2d-overlap",
        action="store_true",
        default=False,
        help="Disable dedicated H2D CUDA stream overlap for sparse weight transfers.",
    )
    p.add_argument(
        "--kv-basis-path",
        type=str,
        default="",
        help="Path to K/V routing basis checkpoint (.pt) from init_learned_basis_from_dense_mlp.py. "
             "Enables sparse K/V projection loading (~10%% of column blocks transferred per token).",
    )
    p.add_argument(
        "--kv-sparse-top-k",
        type=int,
        default=None,
        help="Override number of active K/V column blocks per token (default: from checkpoint config).",
    )
    p.add_argument(
        "--no-stream-output",
        action="store_true",
        default=False,
        help="Disable incremental token streaming to stdout and only print the final text after generation.",
    )
    return p


def _make_token_callback(tokenizer: Any, args: Any) -> tuple[Dict[str, bool], Any]:
    state = {"started": False}

    def _token_callback(next_token: torch.Tensor, _generated: torch.LongTensor) -> None:
        if bool(args.no_stream_output):
            return
        token_id = int(next_token.view(-1)[0].item())
        if tokenizer.eos_token_id is not None and token_id == int(tokenizer.eos_token_id):
            return
        piece = tokenizer.decode(
            [token_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not piece:
            return
        if not state["started"]:
            print("[stream] ", end="", flush=True)
            state["started"] = True
        print(piece, end="", flush=True)

    return state, _token_callback


def _run_single_prompt(
    prompt: str,
    *,
    runtime: "StreamingLlamaRuntime",
    tokenizer: Any,
    args: Any,
    prompt_idx: int = 0,
    reuse_session_cache: bool = False,
) -> Dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]
    stream_state, token_callback = _make_token_callback(tokenizer, args)

    t0 = time.perf_counter()
    with torch.no_grad():
        generated = runtime.generate(
            input_ids=input_ids,
            max_new_tokens=int(args.max_new_tokens),
            eos_token_id=tokenizer.eos_token_id,
            do_sample=bool(args.do_sample),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            token_callback=token_callback,
            reuse_session_cache=bool(reuse_session_cache),
        )
    elapsed = time.perf_counter() - t0
    if stream_state["started"]:
        print("", flush=True)
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    new_tokens = max(int(generated.shape[-1] - input_ids.shape[-1]), 1)
    row = {
        "prompt_idx": int(prompt_idx),
        "latency_s": float(elapsed),
        "tok_s": float(new_tokens / max(elapsed, 1e-9)),
        "text": text,
    }
    traffic = runtime.get_last_traffic_report()
    if traffic is not None:
        row["traffic"] = traffic
        decode = traffic.get("decode", {})
        overall = traffic.get("overall", {})
        print(
            "[traffic] "
            f"decode={float(decode.get('avg_mb_per_layer', 0.0)):.2f} MB/layer "
            f"overall={float(overall.get('avg_mb_per_layer', 0.0)):.2f} MB/layer",
            flush=True,
        )
    print(f"\n[latency {elapsed:.1f}s | {row['tok_s']:.2f} tok/s]\n{text}\n")
    return row


def main() -> None:
    args = _build_arg_parser().parse_args()
    prompts = list(args.prompt) if args.prompt else []
    taylor_layers = _parse_layer_selection(args.taylor_layers)

    runtime = StreamingLlamaRuntime(
        model_name_or_path=str(args.model_name),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.float16,
        taylor_layers=taylor_layers,
        taylor_feature_map=str(args.taylor_feature_map),
        taylor_local_window=int(args.taylor_local_window),
        taylor_feature_dim=int(args.taylor_feature_dim),
        taylor_state_decay=float(args.taylor_state_decay),
        local_files_only=bool(args.local_files_only),
        ram_cache=not bool(args.no_ram_cache),
        ram_cache_pinned=False if bool(args.no_ram_cache_pinned) else None,
        sparse_basis_path=str(args.sparse_basis_path) if args.sparse_basis_path else None,
        sparse_top_k=int(args.sparse_top_k) if args.sparse_top_k is not None else None,
        vram_hot_cache_gb=float(args.vram_hot_cache_gb) if args.vram_hot_cache_gb is not None else None,
        hot_block_threshold=float(args.hot_block_threshold),
        attn_head_importance_path=str(args.attn_head_importance_path) if args.attn_head_importance_path else None,
        attn_share_path=str(args.attn_share_path) if args.attn_share_path else None,
        attn_active_heads=int(args.attn_active_heads) if args.attn_active_heads is not None else None,
        attn_head_activity_threshold=float(args.attn_head_activity_threshold),
        attn_min_active_heads=int(args.attn_min_active_heads),
        attn_max_active_heads=int(args.attn_max_active_heads) if args.attn_max_active_heads is not None else None,
        enable_triton_fused_sparse_mlp=not bool(args.disable_triton_fused_sparse_mlp),
        enable_cuda_h2d_overlap=not bool(args.disable_cuda_h2d_overlap),
        kv_basis_path=str(args.kv_basis_path) if args.kv_basis_path else None,
        kv_sparse_top_k=int(args.kv_sparse_top_k) if args.kv_sparse_top_k is not None else None,
    )
    if int(args.max_runtime_layers) > 0:
        runtime.num_layers = min(int(runtime.num_layers), int(args.max_runtime_layers))
        print(f"[smoke] runtime capped to {int(runtime.num_layers)} layers", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_name), use_fast=True, local_files_only=bool(args.local_files_only)
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows: List[Dict[str, Any]] = []

    if prompts:
        # Scripted mode: run the given --prompt args and exit.
        for idx, prompt in enumerate(prompts):
            rows.append(_run_single_prompt(prompt, runtime=runtime, tokenizer=tokenizer, args=args, prompt_idx=idx))
    else:
        print("[interactive] Model loaded. Type a prompt. Use /reset to clear cached prefix state. Ctrl+C or blank line to quit.")
        idx = 0
        while True:
            try:
                prompt = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not prompt:
                break
            if prompt.lower() in {"exit", "quit"}:
                break
            if prompt.lower() in {"/reset", "reset", "new"}:
                runtime.clear_session_state()
                print("[session] cleared", flush=True)
                continue
            try:
                rows.append(
                    _run_single_prompt(
                        prompt,
                        runtime=runtime,
                        tokenizer=tokenizer,
                        args=args,
                        prompt_idx=idx,
                        reuse_session_cache=True,
                    )
                )
            except Exception:
                import traceback
                traceback.print_exc()
            idx += 1

    if args.dump_json:
        out_path = Path(args.dump_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": str(args.model_name),
            "taylor_layers": list(taylor_layers or []),
            "rows": rows,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote diagnostics: {out_path}")


if __name__ == "__main__":
    main()
