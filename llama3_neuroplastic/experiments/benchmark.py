"""benchmark.py — Reproducible throughput benchmark for StreamingLlamaRuntime.

Measures:
  - Prefill latency (ms / token)
  - Decode throughput (tokens / second)
  - Per-layer H2D traffic (GB / token, from traffic stats)
  - VRAM high-water mark (MB)
  - First-token latency (ms)

Usage:
    python -m llama3_neuroplastic.experiments.benchmark \
        --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" \
        --sparse-basis-path results/mlp_basis.pt \
        --prompt "The theory of relativity states that" \
        --max-new-tokens 20 \
        --warmup-tokens 5 \
        --output-json results/benchmark.json

The script runs a warmup pass (to fill caches) then a timed pass.
All timing uses torch.cuda.Event for GPU-accurate measurement.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from llama3_neuroplastic.layer_selection import parse_layer_selection

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

try:
    from .contracts import (
        apply_throughput_probe_defaults as _apply_throughput_probe_defaults,
    )
    from .contracts import (
        build_throughput_contract_report,
    )
    from .contracts import (
        normalize_throughput_contract as _normalize_throughput_contract,
    )
    from .streaming_llama_runtime import StreamingLlamaRuntime
except ImportError:
    from contracts import (
        apply_throughput_probe_defaults as _apply_throughput_probe_defaults,
    )
    from contracts import (
        build_throughput_contract_report,
    )
    from contracts import (
        normalize_throughput_contract as _normalize_throughput_contract,
    )
    from streaming_llama_runtime import StreamingLlamaRuntime


_DEFAULT_PROMPT = (
    "In the beginning, there was a vast and silent void. Then, slowly, "
    "the first light emerged from the darkness."
)

def _cuda_time_ms(start_event: torch.cuda.Event, end_event: torch.cuda.Event) -> float:
    """Return elapsed time in milliseconds between two CUDA events."""
    torch.cuda.synchronize()
    return float(start_event.elapsed_time(end_event))


def _record_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.memory_allocated()) / (1024 ** 2)


def _runtime_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        major, _minor = torch.cuda.get_device_capability(device)
        if int(major) < 8:
            return torch.float16
    return torch.bfloat16


def _validate_required_path(path_value: str | None, *, flag: str) -> None:
    if not str(path_value or "").strip():
        raise RuntimeError(f"{flag} is required for --throughput-probe")


def _validate_runtime_for_throughput_probe(runtime: StreamingLlamaRuntime) -> None:
    errors = runtime.validate_throughput_probe()
    if errors:
        raise RuntimeError(f"throughput probe is not on the 3.3 fast path: {'; '.join(errors)}")


def _build_throughput_contract_report(results: dict[str, Any]) -> dict[str, Any]:
    return build_throughput_contract_report(results, include_sparse_kv_checks=True)


def run_benchmark(
    runtime: StreamingLlamaRuntime,
    input_ids: torch.LongTensor,
    *,
    max_new_tokens: int = 20,
    warmup_tokens: int = 5,
) -> dict[str, Any]:
    device = runtime.device
    tokens_generated: list[int] = []
    token_times_ms: list[float] = []
    first_token_latency_ms: float | None = None


    print(f"[bench] Warmup: generating {warmup_tokens} tokens ...", flush=True)
    runtime._reset_traffic_stats()
    runtime.reset_caches()
    warmup_generated = runtime.generate(
        input_ids.to(device),
        max_new_tokens=warmup_tokens,
        do_sample=False,
    )
    if getattr(runtime, "_vram_hot_cache_enabled", False) and getattr(runtime, "_sparse_routing", None):
        # Calibrate from the full warmup-generated sequence so the selected hot
        # blocks see the same absolute positions and first decode-token context
        # as the timed run.
        _calib_ids = warmup_generated.detach().to(device=torch.device("cpu"), dtype=torch.long)
        runtime.calibrate_vram_hot_cache(
            _calib_ids,
            max_tokens=int(_calib_ids.shape[1]),
            rebuild_cache=True,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("[bench] Warmup done. Starting timed run ...", flush=True)
    runtime._reset_traffic_stats()


    runtime.reset_caches()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


    prefill_start = time.perf_counter()
    if device.type == "cuda":
        ev_prefill_start = torch.cuda.Event(enable_timing=True)
        ev_prefill_end = torch.cuda.Event(enable_timing=True)
        ev_prefill_start.record()

    prompt_len = int(input_ids.shape[1])
    runtime.reset_caches()
    runtime.begin_traffic_phase("prefill")
    logits = runtime.prefill_logits(input_ids.to(device))

    if device.type == "cuda":
        ev_prefill_end.record()
        torch.cuda.synchronize()
        prefill_gpu_ms = float(ev_prefill_start.elapsed_time(ev_prefill_end))
    else:
        prefill_gpu_ms = (time.perf_counter() - prefill_start) * 1000.0

    prefill_ms_per_token = prefill_gpu_ms / max(prompt_len, 1)


    generated = input_ids.to(device).clone()
    runtime.begin_traffic_phase("decode")
    runtime.materialize_lm_head()

    for step in range(max_new_tokens):
        step_start = time.perf_counter()
        if device.type == "cuda":
            ev_step_start = torch.cuda.Event(enable_timing=True)
            ev_step_end = torch.cuda.Event(enable_timing=True)
            ev_step_start.record()

        next_token = runtime.sample_next_token(
            logits[:, -1, :],
            do_sample=False,
            temperature=1.0,
            top_k=None,
            top_p=1.0,
        ).view(1, 1).to(device=device)
        generated = torch.cat([generated, next_token], dim=-1)
        logits = runtime.decode_token_logits(next_token, position_index=int(generated.shape[1]) - 1)

        if device.type == "cuda":
            ev_step_end.record()
            torch.cuda.synchronize()
            step_ms = float(ev_step_start.elapsed_time(ev_step_end))
        else:
            step_ms = (time.perf_counter() - step_start) * 1000.0

        tokens_generated.append(int(next_token.item()))
        token_times_ms.append(step_ms)
        if first_token_latency_ms is None:
            first_token_latency_ms = step_ms

        print(
            f"[bench] token {step + 1}/{max_new_tokens}: {step_ms:.1f} ms "
            f"({1000.0 / step_ms:.2f} tok/s)",
            flush=True,
        )


    vram_peak_mb = 0.0
    if device.type == "cuda":
        vram_peak_mb = float(torch.cuda.max_memory_allocated(device)) / (1024 ** 2)

    mean_decode_ms = sum(token_times_ms) / len(token_times_ms) if token_times_ms else 0.0
    decode_tps = 1000.0 / mean_decode_ms if mean_decode_ms > 0 else 0.0

    runtime.begin_traffic_phase("idle")
    runtime.finalize_traffic_report()
    traffic_report = runtime.get_last_traffic_report()
    decode_layer_visits = int(((traffic_report or {}).get("decode", {}) or {}).get("layer_visits", 0))
    decode_avg_mb_per_layer = float(((traffic_report or {}).get("decode", {}) or {}).get("avg_mb_per_layer", 0.0))
    decode_ms_per_layer = mean_decode_ms / float(max(decode_layer_visits // max(max_new_tokens, 1), 1))

    return {
        "prompt_tokens": prompt_len,
        "new_tokens": max_new_tokens,
        "prefill_total_ms": prefill_gpu_ms,
        "prefill_ms_per_token": prefill_ms_per_token,
        "first_token_latency_ms": first_token_latency_ms,
        "mean_decode_ms_per_token": mean_decode_ms,
        "decode_tok_s": decode_tps,
        "decode_tokens_per_second": decode_tps,
        "decode_ms_per_layer": decode_ms_per_layer,
        "decode_avg_mb_per_layer": decode_avg_mb_per_layer,
        "vram_peak_mb": vram_peak_mb,
        "token_times_ms": token_times_ms,
        "traffic": traffic_report,
        "traffic_report": traffic_report,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark streaming runtime throughput")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--sparse-basis-path", default=None)
    parser.add_argument("--sparse-top-k", type=int, default=None)
    parser.add_argument("--sparse-basis-top-k", type=int, default=None)
    parser.add_argument("--sparse-mlp-execution", default=None)
    parser.add_argument("--sparse-mlp-prefill-mode", default="dense", choices=["dense", "sparse", "hot_cache"])
    parser.add_argument("--attn-head-importance-path", default=None)
    parser.add_argument("--attn-active-heads", type=int, default=None)
    parser.add_argument("--attn-min-active-heads", type=int, default=16)
    parser.add_argument("--attn-max-active-heads", type=int, default=None)
    parser.add_argument("--sparse-attn-prefill-mode", default="dense", choices=["dense", "sparse"])
    parser.add_argument("--kv-basis-path", default=None)
    parser.add_argument("--kv-sparse-top-k", type=int, default=None)
    parser.add_argument("--sparse-kv-prefill-mode", default="dense", choices=["dense", "sparse"])
    parser.add_argument("--mlp-skip-mask-path", default=None)
    parser.add_argument("--prompt", default=_DEFAULT_PROMPT)
    parser.add_argument("--prompt-format", default="raw", choices=["raw", "chat"])
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--warmup-tokens", type=int, default=5,
                        help="Tokens to generate in warmup pass (fills RAM cache, not timed)")
    parser.add_argument("--vram-hot-cache-gb", type=float, default=5.25)
    parser.add_argument("--pre-warm", action="store_true", default=False)
    parser.add_argument("--calibrate-hot-cache", action="store_true", default=False)
    parser.add_argument("--hot-cache-calibration-tokens", type=int, default=64)
    parser.add_argument("--taylor-layers", type=str, default=None,
                        help="Taylor-attention layer selection, e.g. '0-31'. Use 'none' to disable.")
    parser.add_argument("--disable-triton-fused-sparse-mlp", action="store_true", default=False)
    parser.add_argument("--disable-cuda-h2d-overlap", action="store_true", default=False)
    parser.add_argument("--hard-cuda-cache-flush", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ram-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--throughput-probe", action="store_true", default=False)
    parser.add_argument(
        "--throughput-contract",
        default="off",
        choices=["off", "probe", "strict"],
        help="Validate the 3.3 tok/s fast path configuration only ('probe') or configuration plus measured decode metrics ('strict').",
    )
    parser.add_argument(
        "--profile-decode",
        action="store_true",
        default=False,
        help="Capture per-layer decode timings into the output JSON/runtime status.",
    )
    parser.add_argument(
        "--profile-max-steps",
        type=int,
        default=0,
        help="Optional cap on retained decode profile steps. 0 keeps all measured decode steps.",
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args(argv)
    args.throughput_contract = _normalize_throughput_contract(getattr(args, "throughput_contract", "off"))
    if bool(args.throughput_probe):
        _apply_throughput_probe_defaults(args)
        _validate_required_path(args.sparse_basis_path, flag="--sparse-basis-path")
        _validate_required_path(args.attn_head_importance_path, flag="--attn-head-importance-path")
        _validate_required_path(args.kv_basis_path, flag="--kv-basis-path")
        args.disable_triton_fused_sparse_mlp = False
        if str(args.throughput_contract) == "off":
            args.throughput_contract = "probe"

    if AutoTokenizer is None:
        print("ERROR: transformers not installed", file=sys.stderr)
        sys.exit(1)

    print("[bench] Tokenizing prompt ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=bool(args.local_files_only))
    if args.prompt_format == "chat":
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": str(args.prompt)}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = encoded["input_ids"]
    else:
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
    print(f"[bench] Prompt: {int(input_ids.shape[1])} tokens", flush=True)

    print("[bench] Initializing StreamingLlamaRuntime ...", flush=True)
    _taylor_layers: list[int] | None = None
    if args.taylor_layers is not None:
        _tl = args.taylor_layers.strip().lower()
        if _tl in ("none", "off", ""):
            _taylor_layers = []
        else:
            _taylor_layers = parse_layer_selection(_tl, all_as_none=True, allow_none_token=True)

    runtime_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = StreamingLlamaRuntime(
        model_name_or_path=args.model_name,
        device=runtime_device,
        dtype=_runtime_dtype(runtime_device),
        local_files_only=bool(args.local_files_only),
        ram_cache=bool(args.ram_cache),
        sparse_basis_path=args.sparse_basis_path,
        sparse_top_k=int(args.sparse_top_k) if args.sparse_top_k is not None else None,
        sparse_basis_top_k=int(args.sparse_basis_top_k) if args.sparse_basis_top_k is not None else None,
        sparse_mlp_execution=str(args.sparse_mlp_execution) if args.sparse_mlp_execution else None,
        sparse_mlp_prefill_mode=str(args.sparse_mlp_prefill_mode),
        attn_head_importance_path=args.attn_head_importance_path,
        attn_active_heads=int(args.attn_active_heads) if args.attn_active_heads is not None else None,
        attn_min_active_heads=int(args.attn_min_active_heads),
        attn_max_active_heads=int(args.attn_max_active_heads) if args.attn_max_active_heads is not None else None,
        sparse_attn_prefill_mode=str(args.sparse_attn_prefill_mode),
        kv_basis_path=args.kv_basis_path,
        kv_sparse_top_k=int(args.kv_sparse_top_k) if args.kv_sparse_top_k is not None else None,
        sparse_kv_prefill_mode=str(args.sparse_kv_prefill_mode),
        mlp_skip_mask_path=args.mlp_skip_mask_path,
        vram_hot_cache_gb=args.vram_hot_cache_gb,
        taylor_layers=_taylor_layers,
        enable_triton_fused_sparse_mlp=not bool(args.disable_triton_fused_sparse_mlp),
        enable_cuda_h2d_overlap=not bool(args.disable_cuda_h2d_overlap),
        hard_cuda_cache_flush=bool(args.hard_cuda_cache_flush),
    )
    if bool(args.profile_decode):
        runtime.enable_decode_profiler(True, max_steps=int(args.profile_max_steps))
    if str(args.throughput_contract) in {"probe", "strict"}:
        _validate_runtime_for_throughput_probe(runtime)
    if bool(args.pre_warm):
        runtime.pre_warm_vram_hot_cache()
    if bool(args.calibrate_hot_cache):
        runtime.calibrate_vram_hot_cache(
            input_ids.detach().to(device=torch.device("cpu"), dtype=torch.long),
            max_tokens=int(args.hot_cache_calibration_tokens),
            rebuild_cache=True,
        )
        if str(args.throughput_contract) in {"probe", "strict"}:
            _validate_runtime_for_throughput_probe(runtime)

    results = run_benchmark(
        runtime,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        warmup_tokens=args.warmup_tokens,
    )
    results["runtime_status"] = runtime.get_runtime_status()
    results["throughput_probe"] = bool(args.throughput_probe)
    results["throughput_contract"] = (
        _build_throughput_contract_report(results)
        if str(args.throughput_contract) == "strict"
        else {"contract": str(args.throughput_contract), "passed": True, "failed_checks": [], "checks": []}
    )
    decode_profile_report = runtime.get_decode_profile_report()
    if decode_profile_report is not None:
        results["decode_profile_report"] = decode_profile_report

    print(
        f"\n[bench] === RESULTS ===\n"
        f"  Prompt tokens:           {results['prompt_tokens']}\n"
        f"  Prefill total:           {results['prefill_total_ms']:.1f} ms\n"
        f"  Prefill per token:       {results['prefill_ms_per_token']:.1f} ms/tok\n"
        f"  First-token latency:     {results['first_token_latency_ms']:.1f} ms\n"
        f"  Mean decode latency:     {results['mean_decode_ms_per_token']:.1f} ms/tok\n"
        f"  Mean decode / layer:     {results['decode_ms_per_layer']:.2f} ms/layer\n"
        f"  Decode throughput:       {results['decode_tokens_per_second']:.2f} tok/s\n"
        f"  Decode traffic:          {results['decode_avg_mb_per_layer']:.2f} MiB/layer\n"
        f"  VRAM peak:               {results['vram_peak_mb']:.1f} MB\n"
        f"  Contract:                {results['throughput_contract']['contract']} "
        f"({'pass' if results['throughput_contract']['passed'] else 'fail'})",
        flush=True,
    )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[bench] Results written to {out_path}", flush=True)
    if str(args.throughput_contract) == "strict" and not bool(results["throughput_contract"]["passed"]):
        failed = ", ".join(str(name) for name in results["throughput_contract"].get("failed_checks", []))
        raise RuntimeError(f"strict throughput contract failed: {failed}")


if __name__ == "__main__":
    main()
