from __future__ import annotations

import os

if os.getenv("STREAMING_DEBUG_SYNC_CUDA", "").strip().lower() in {"1", "true", "yes", "on"}:
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from llama3_neuroplastic.layer_selection import parse_layer_selection

try:
    from .streaming_llama_runtime import StreamingLlamaRuntime
except ImportError:
    from streaming_llama_runtime import StreamingLlamaRuntime


def _parse_layer_selection(spec: str | None) -> list[int] | None:
    return parse_layer_selection(spec, all_as_none=True, allow_none_token=True)


_THROUGHPUT_PROBE_DEFAULTS: dict[str, Any] = {
    "sparse_top_k": 208,
    "sparse_basis_top_k": 64,
    "sparse_mlp_execution": "exact_blockwise_sparse",
    "sparse_mlp_prefill_mode": "hot_cache",
    "vram_hot_cache_gb": 5.25,
    "pre_warm": True,
    "calibrate_hot_cache": True,
    "hot_cache_calibration_tokens": 64,
    "attn_active_heads": 5,
    "attn_min_active_heads": 5,
    "attn_max_active_heads": 5,
    "sparse_attn_prefill_mode": "sparse",
    "sparse_kv_prefill_mode": "sparse",
}

_THROUGHPUT_CONTRACT_TARGETS: dict[str, float] = {
    "decode_tok_s": 3.30,
    "mean_decode_ms_per_token": 303.0,
    "decode_avg_mb_per_layer": 23.0,
}


def _apply_throughput_probe_defaults(args: Any) -> None:
    for key, value in _THROUGHPUT_PROBE_DEFAULTS.items():
        setattr(args, key, value)


def _validate_required_path(path_value: str | None, *, flag: str) -> None:
    if not str(path_value or "").strip():
        raise RuntimeError(f"{flag} is required for --throughput-probe")


def _validate_runtime_for_throughput_probe(runtime: StreamingLlamaRuntime) -> None:
    errors = runtime.validate_throughput_probe()
    if errors:
        formatted = "; ".join(errors)
        raise RuntimeError(f"throughput probe is not on the 3.3 fast path: {formatted}")


def _normalize_throughput_contract(raw_value: str | None) -> str:
    value = str(raw_value or "off").strip().lower() or "off"
    if value in {"off", "none", "false", "0"}:
        return "off"
    if value in {"probe", "fast_path"}:
        return "probe"
    if value in {"strict", "enforce"}:
        return "strict"
    raise RuntimeError("throughput-contract must be one of: off, probe, strict")


def _build_contract_check(*, name: str, passed: bool, actual: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": str(name),
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
    }


def _build_throughput_contract_report(row: dict[str, Any]) -> dict[str, Any]:
    runtime_status = dict(row.get("runtime_status", {}) or {})
    traffic = dict(row.get("traffic", {}) or {})
    decode_traffic = dict(traffic.get("decode", {}) or {})
    new_tokens = int(row.get("new_tokens", 0))
    expected_layer_visits = int(runtime_status.get("num_layers", 0)) * new_tokens
    actual_layer_visits = int(decode_traffic.get("layer_visits", 0))
    checks = [
        _build_contract_check(
            name="decode_tok_s",
            passed=float(row.get("decode_tok_s", 0.0)) >= _THROUGHPUT_CONTRACT_TARGETS["decode_tok_s"],
            actual=float(row.get("decode_tok_s", 0.0)),
            expected=f">={_THROUGHPUT_CONTRACT_TARGETS['decode_tok_s']}",
        ),
        _build_contract_check(
            name="mean_decode_ms_per_token",
            passed=float(row.get("mean_decode_ms_per_token", float("inf"))) <= _THROUGHPUT_CONTRACT_TARGETS["mean_decode_ms_per_token"],
            actual=float(row.get("mean_decode_ms_per_token", 0.0)),
            expected=f"<={_THROUGHPUT_CONTRACT_TARGETS['mean_decode_ms_per_token']}",
        ),
        _build_contract_check(
            name="decode_avg_mb_per_layer",
            passed=float(row.get("decode_avg_mb_per_layer", float("inf"))) <= _THROUGHPUT_CONTRACT_TARGETS["decode_avg_mb_per_layer"],
            actual=float(row.get("decode_avg_mb_per_layer", 0.0)),
            expected=f"<={_THROUGHPUT_CONTRACT_TARGETS['decode_avg_mb_per_layer']}",
        ),
        _build_contract_check(
            name="decode_layer_visits",
            passed=expected_layer_visits > 0 and actual_layer_visits == expected_layer_visits,
            actual=actual_layer_visits,
            expected=expected_layer_visits,
        ),
        _build_contract_check(
            name="lm_head_on_gpu",
            passed=bool(runtime_status.get("lm_head_on_gpu", False)),
            actual=bool(runtime_status.get("lm_head_on_gpu", False)),
            expected=True,
        ),
        _build_contract_check(
            name="lm_head_mode",
            passed=str(runtime_status.get("lm_head_mode", "")) == "gpu_nf4",
            actual=str(runtime_status.get("lm_head_mode", "")),
            expected="gpu_nf4",
        ),
        _build_contract_check(
            name="decode_backend",
            passed=str(runtime_status.get("decode_backend", "")) == "single_kernel_sparse_decode_sm75",
            actual=str(runtime_status.get("decode_backend", "")),
            expected="single_kernel_sparse_decode_sm75",
        ),
        _build_contract_check(
            name="attn_backend_decode",
            passed=str(runtime_status.get("attn_backend_decode", "")) == "compact_sparse_v1",
            actual=str(runtime_status.get("attn_backend_decode", "")),
            expected="compact_sparse_v1",
        ),
        _build_contract_check(
            name="compact_sparse_attention_steps",
            passed=int(runtime_status.get("compact_sparse_attention_steps", 0)) > 0,
            actual=int(runtime_status.get("compact_sparse_attention_steps", 0)),
            expected=">0",
        ),
        _build_contract_check(
            name="vram_hot_cache_live_calibrated",
            passed=bool(runtime_status.get("vram_hot_cache_live_calibrated", False)),
            actual=bool(runtime_status.get("vram_hot_cache_live_calibrated", False)),
            expected=True,
        ),
        _build_contract_check(
            name="decode_mlp_cold_blocks_streamed",
            passed=int(runtime_status.get("decode_mlp_cold_blocks_streamed", 0)) == 0,
            actual=int(runtime_status.get("decode_mlp_cold_blocks_streamed", 0)),
            expected=0,
        ),
        _build_contract_check(
            name="decode_down_cold_blocks_streamed",
            passed=int(runtime_status.get("decode_down_cold_blocks_streamed", 0)) == 0,
            actual=int(runtime_status.get("decode_down_cold_blocks_streamed", 0)),
            expected=0,
        ),
    ]
    failed = [check["name"] for check in checks if not bool(check["passed"])]
    return {
        "contract": "strict",
        "passed": len(failed) == 0,
        "failed_checks": failed,
        "checks": checks,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run greedy decode with layer-by-layer streamed Llama weights.")
    p.add_argument("--model-name", type=str, required=True, help="HF repo id or local snapshot directory")
    p.add_argument("--prompt", action="append", default=[], help="Prompt to run; may be repeated")
    p.add_argument(
        "--prompt-format",
        type=str,
        default="raw",
        choices=["raw", "chat"],
        help="Prompt encoding mode: raw tokenizer call or tokenizer chat template with generation prompt.",
    )
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
        "--throughput-contract",
        type=str,
        default="off",
        choices=["off", "probe", "strict"],
        help="Validate the 3.3 tok/s fast path configuration only ('probe') or configuration plus measured decode metrics ('strict').",
    )
    p.add_argument(
        "--throughput-probe",
        action="store_true",
        default=False,
        help=(
            "Force the README 3.3 tok/s probe regime: sparse-top-k 51, attn heads 5, "
            "sparse K/V, VRAM hot cache, pre-warm, calibration, and Triton sparse MLP."
        ),
    )
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
             "Enables sparse MLP routing for exact blockwise intermediate-block execution.",
    )
    p.add_argument(
        "--sparse-top-k",
        type=int,
        default=None,
        help="Override number of active MLP blocks per token (default: 2%% of blocks from checkpoint config).",
    )
    p.add_argument(
        "--sparse-basis-top-k",
        type=int,
        default=None,
        help="Override latent basis support at runtime. Applies to both surrogate and router artifacts.",
    )
    p.add_argument(
        "--sparse-mlp-execution",
        type=str,
        default=None,
        choices=["auto", "exact_blockwise_sparse", "exact_intermediate_sparse"],
        help="Sparse MLP execution mode. auto selects exact_blockwise_sparse for intermediate-router artifacts.",
    )
    p.add_argument(
        "--sparse-mlp-prefill-mode",
        type=str,
        default="dense",
        choices=["dense", "sparse", "hot_cache"],
        help="Whether prompt prefill uses dense MLPs, sparse MLP, or VRAM hot-cache blocks.",
    )
    p.add_argument(
        "--sparse-mlp-prefill-top-k",
        type=int,
        default=None,
        help="Optional prompt-prefill-only MLP block count. Decode still uses --sparse-top-k.",
    )
    p.add_argument(
        "--vram-hot-cache-gb",
        type=float,
        default=None,
        help="VRAM budget (GB) for persistent NF4 hot-block cache. Default is disabled unless STREAMING_VRAM_HOT_CACHE_GB or --pre-warm is set.",
    )
    p.add_argument(
        "--hot-block-threshold",
        type=float,
        default=0.80,
        help="Layer-local score threshold for selecting deterministic hot MLP blocks for VRAM cache.",
    )
    p.add_argument(
        "--pre-warm",
        action="store_true",
        default=False,
        help=(
            "Load VRAM hot-cache at startup before first inference (paid once; persists across all queries). "
            "Automatically sets --sparse-mlp-prefill-mode hot_cache so prefill MLP reads come from VRAM "
            "instead of SSD, reducing per-query prefill I/O to attention weights only (~34 GB vs ~199 GB)."
        ),
    )
    p.add_argument(
        "--calibrate-hot-cache",
        action="store_true",
        default=False,
        help=(
            "After the initial VRAM hot-cache pre-warm, run a short prompt-based routing pass, "
            "replace MLP hot blocks with live-route top hits, and rebuild the VRAM hot cache."
        ),
    )
    p.add_argument(
        "--hot-cache-calibration-tokens",
        type=int,
        default=64,
        help="Maximum prompt tokens to use for live-route hot-cache calibration.",
    )
    p.add_argument(
        "--hot-cache-calibration-prompt",
        type=str,
        default="",
        help="Optional prompt used only for hot-cache calibration. Defaults to the first --prompt.",
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
        "--sparse-attn-prefill-mode",
        type=str,
        default="dense",
        choices=["dense", "sparse"],
        help="Whether prompt prefill uses dense or sparse attention Q/O projection loading.",
    )
    p.add_argument(
        "--sparse-kv-prefill-mode",
        type=str,
        default="dense",
        choices=["dense", "sparse"],
        help="Whether prompt prefill uses dense or sparse K/V projection loading.",
    )
    p.add_argument(
        "--attn-share-prefill-mode",
        type=str,
        default="dense",
        choices=["dense", "shared"],
        help="Whether prompt prefill uses dense attention or cross-layer shared attention reconstruction.",
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
    p.add_argument(
        "--profile-decode",
        action="store_true",
        default=False,
        help="Capture per-layer decode timings into runtime status and JSON diagnostics.",
    )
    p.add_argument(
        "--profile-max-steps",
        type=int,
        default=0,
        help="Optional cap on retained decode profile steps. 0 keeps all measured decode steps.",
    )
    return p


def _make_token_callback(tokenizer: Any, args: Any) -> tuple[dict[str, bool], Any]:
    state = {"started": False}
    stream_encoding = getattr(sys.stdout, "encoding", None) or "utf-8"

    def _safe_piece(text: str) -> str:
        try:
            text.encode(stream_encoding)
            return text
        except Exception:
            return text.encode(stream_encoding, errors="replace").decode(stream_encoding, errors="replace")

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
        piece = _safe_piece(piece)
        if not state["started"]:
            print("[stream] ", end="", flush=True)
            state["started"] = True
        print(piece, end="", flush=True)

    return state, _token_callback


def _run_single_prompt(
    prompt: str,
    *,
    runtime: StreamingLlamaRuntime,
    tokenizer: Any,
    args: Any,
    prompt_idx: int = 0,
    reuse_session_cache: bool = False,
) -> dict[str, Any]:
    if str(getattr(args, "prompt_format", "raw")) == "chat":
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    else:
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
    full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    completion_tokens = generated[0, input_ids.shape[-1] :]
    completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=True)
    stream_encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        full_text.encode(stream_encoding)
    except Exception:
        full_text = full_text.encode(stream_encoding, errors="replace").decode(stream_encoding, errors="replace")
    try:
        completion_text.encode(stream_encoding)
    except Exception:
        completion_text = completion_text.encode(stream_encoding, errors="replace").decode(stream_encoding, errors="replace")
    new_tokens = max(int(generated.shape[-1] - input_ids.shape[-1]), 1)
    first_decode_t = getattr(runtime, "_first_decode_t", None)
    decode_done_t = getattr(runtime, "_decode_done_t", None)
    if first_decode_t is not None and decode_done_t is not None and float(decode_done_t) >= float(first_decode_t):
        prefill_elapsed = max(float(first_decode_t) - float(t0), 0.0)
        decode_elapsed = max(float(decode_done_t) - float(first_decode_t), 1e-9)
    else:
        prefill_elapsed = 0.0
        decode_elapsed = max(float(elapsed), 1e-9)
    total_tok_s = float(new_tokens / max(elapsed, 1e-9))
    decode_tok_s = float(new_tokens / max(decode_elapsed, 1e-9))
    prefill_tok_s = float(int(input_ids.shape[-1]) / max(prefill_elapsed, 1e-9)) if prefill_elapsed > 0.0 else 0.0
    row = {
        "prompt_idx": int(prompt_idx),
        "latency_s": float(elapsed),
        "prefill_latency_s": float(prefill_elapsed),
        "decode_latency_s": float(decode_elapsed),
        "tok_s": decode_tok_s,
        "decode_tok_s": decode_tok_s,
        "total_tok_s": total_tok_s,
        "prefill_tok_s": prefill_tok_s,
        "new_tokens": int(new_tokens),
        "mean_decode_ms_per_token": float(decode_elapsed * 1000.0) / float(max(new_tokens, 1)),
        "text": full_text,
        "completion_text": completion_text,
    }
    row["runtime_status"] = runtime.get_runtime_status()
    decode_profile_report = runtime.get_decode_profile_report()
    if decode_profile_report is not None:
        row["decode_profile_report"] = decode_profile_report
    traffic = runtime.get_last_traffic_report()
    if traffic is not None:
        row["traffic"] = traffic
        decode = traffic.get("decode", {})
        overall = traffic.get("overall", {})
        decode_layer_visits = int(decode.get("layer_visits", 0))
        row["decode_avg_mb_per_layer"] = float(decode.get("avg_mb_per_layer", 0.0))
        row["decode_ms_per_layer"] = float(decode_elapsed * 1000.0) / float(max(decode_layer_visits, 1))
        print(
            "[traffic] "
            f"decode={float(decode.get('avg_mb_per_layer', 0.0)):.2f} MB/layer "
            f"overall={float(overall.get('avg_mb_per_layer', 0.0)):.2f} MB/layer",
            flush=True,
        )
    print(
        f"\n[latency total={elapsed:.1f}s prefill={prefill_elapsed:.1f}s "
        f"decode={decode_elapsed:.3f}s | {row['tok_s']:.2f} tok/s "
        f"| total={total_tok_s:.4f} tok/s]\n{completion_text}\n"
    )
    if str(getattr(args, "throughput_contract", "off")) == "strict":
        row["throughput_contract"] = _build_throughput_contract_report(row)
    else:
        row["throughput_contract"] = {
            "contract": str(getattr(args, "throughput_contract", "off")),
            "passed": True,
            "failed_checks": [],
            "checks": [],
        }
    return row


def main() -> None:
    args = _build_arg_parser().parse_args()
    args.throughput_contract = _normalize_throughput_contract(getattr(args, "throughput_contract", "off"))
    if bool(args.throughput_probe):
        _apply_throughput_probe_defaults(args)
        _validate_required_path(args.sparse_basis_path, flag="--sparse-basis-path")
        _validate_required_path(args.attn_head_importance_path, flag="--attn-head-importance-path")
        _validate_required_path(args.kv_basis_path, flag="--kv-basis-path")
        args.disable_triton_fused_sparse_mlp = False
        if str(args.throughput_contract) == "off":
            args.throughput_contract = "probe"
    if bool(args.pre_warm):
        args.sparse_mlp_prefill_mode = "hot_cache"
        if args.vram_hot_cache_gb is None:
            args.vram_hot_cache_gb = 6.4
    prompts = list(args.prompt) if args.prompt else []
    taylor_layers = _parse_layer_selection(args.taylor_layers)
    runtime_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime_dtype = torch.bfloat16
    if runtime_device.type == "cuda":
        major, _minor = torch.cuda.get_device_capability(runtime_device)
        if int(major) < 8:
            runtime_dtype = torch.float16

    runtime = StreamingLlamaRuntime(
        model_name_or_path=str(args.model_name),
        device=runtime_device,
        dtype=runtime_dtype,
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
        sparse_basis_top_k=int(args.sparse_basis_top_k) if args.sparse_basis_top_k is not None else None,
        sparse_mlp_execution=str(args.sparse_mlp_execution) if args.sparse_mlp_execution else None,
        sparse_mlp_prefill_mode=str(args.sparse_mlp_prefill_mode),
        sparse_mlp_prefill_top_k=(
            int(args.sparse_mlp_prefill_top_k)
            if args.sparse_mlp_prefill_top_k is not None
            else None
        ),
        vram_hot_cache_gb=float(args.vram_hot_cache_gb) if args.vram_hot_cache_gb is not None else None,
        hot_block_threshold=float(args.hot_block_threshold),
        attn_head_importance_path=str(args.attn_head_importance_path) if args.attn_head_importance_path else None,
        attn_share_path=str(args.attn_share_path) if args.attn_share_path else None,
        attn_active_heads=int(args.attn_active_heads) if args.attn_active_heads is not None else None,
        attn_head_activity_threshold=float(args.attn_head_activity_threshold),
        attn_min_active_heads=int(args.attn_min_active_heads),
        attn_max_active_heads=int(args.attn_max_active_heads) if args.attn_max_active_heads is not None else None,
        sparse_attn_prefill_mode=str(args.sparse_attn_prefill_mode),
        sparse_kv_prefill_mode=str(args.sparse_kv_prefill_mode),
        attn_share_prefill_mode=str(args.attn_share_prefill_mode),
        enable_triton_fused_sparse_mlp=not bool(args.disable_triton_fused_sparse_mlp),
        enable_cuda_h2d_overlap=not bool(args.disable_cuda_h2d_overlap),
        kv_basis_path=str(args.kv_basis_path) if args.kv_basis_path else None,
        kv_sparse_top_k=int(args.kv_sparse_top_k) if args.kv_sparse_top_k is not None else None,
    )
    if bool(args.profile_decode):
        runtime.enable_decode_profiler(True, max_steps=int(args.profile_max_steps))
    if str(args.throughput_contract) in {"probe", "strict"}:
        _validate_runtime_for_throughput_probe(runtime)
    if int(args.max_runtime_layers) > 0:
        runtime.num_layers = min(int(runtime.num_layers), int(args.max_runtime_layers))
        print(f"[smoke] runtime capped to {int(runtime.num_layers)} layers", flush=True)
        if int(runtime.num_layers) <= 16:
            print(
                "[smoke] coherence warning: <=16-layer cap is for runtime regression only and usually produces low-quality text.",
                flush=True,
            )

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_name), use_fast=True, local_files_only=bool(args.local_files_only)
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if bool(args.pre_warm):
        import time as _prewarm_time
        _t0 = _prewarm_time.perf_counter()
        runtime.pre_warm_vram_hot_cache()
        if bool(args.calibrate_hot_cache):
            calibration_prompt = str(args.hot_cache_calibration_prompt or (prompts[0] if prompts else "")).strip()
            if not calibration_prompt:
                calibration_prompt = "Answer with one word. What is the capital of France?"
            if str(getattr(args, "prompt_format", "raw")) == "chat":
                calibration_encoded = tokenizer.apply_chat_template(
                    [{"role": "user", "content": calibration_prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
            else:
                calibration_encoded = tokenizer(calibration_prompt, return_tensors="pt")
            runtime.calibrate_vram_hot_cache(
                calibration_encoded["input_ids"],
                max_tokens=int(args.hot_cache_calibration_tokens),
                rebuild_cache=True,
            )
            if str(args.throughput_contract) in {"probe", "strict"}:
                _validate_runtime_for_throughput_probe(runtime)
        print(f"[pre-warm] startup completed in {_prewarm_time.perf_counter() - _t0:.1f}s", flush=True)

    rows: list[dict[str, Any]] = []

    if prompts:

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
            "throughput_probe": bool(args.throughput_probe),
            "throughput_contract": str(args.throughput_contract),
            "runtime_status": runtime.get_runtime_status(),
            "rows": rows,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote diagnostics: {out_path}")
    strict_failures = [
        row for row in rows
        if str((row.get("throughput_contract", {}) or {}).get("contract", "off")) == "strict"
        and not bool((row.get("throughput_contract", {}) or {}).get("passed", False))
    ]
    if strict_failures:
        failed = strict_failures[0]["throughput_contract"].get("failed_checks", [])
        raise RuntimeError(f"strict throughput contract failed: {', '.join(str(x) for x in failed)}")


if __name__ == "__main__":
    main()
