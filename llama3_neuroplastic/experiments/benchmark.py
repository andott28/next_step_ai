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
from typing import Any, Dict, List, Optional

import torch

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None  # type: ignore

try:
    from .streaming_llama_runtime import StreamingLlamaRuntime
except ImportError:
    from streaming_llama_runtime import StreamingLlamaRuntime  # type: ignore


_DEFAULT_PROMPT = (
    "In the beginning, there was a vast and silent void. Then, slowly, "
    "the first light emerged from the darkness."
)


def _cuda_time_ms(start_event: "torch.cuda.Event", end_event: "torch.cuda.Event") -> float:
    """Return elapsed time in milliseconds between two CUDA events."""
    torch.cuda.synchronize()
    return float(start_event.elapsed_time(end_event))


def _record_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.memory_allocated()) / (1024 ** 2)


def run_benchmark(
    runtime: StreamingLlamaRuntime,
    input_ids: torch.LongTensor,
    *,
    max_new_tokens: int = 20,
    warmup_tokens: int = 5,
) -> Dict[str, Any]:
    device = runtime.device
    tokens_generated: List[int] = []
    token_times_ms: List[float] = []
    first_token_latency_ms: Optional[float] = None

    # ── Warmup pass ──────────────────────────────────────────────────────────
    print(f"[bench] Warmup: generating {warmup_tokens} tokens ...", flush=True)
    runtime.reset_caches()
    _ = runtime.generate(
        input_ids.to(device),
        max_new_tokens=warmup_tokens,
        do_sample=False,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("[bench] Warmup done. Starting timed run ...", flush=True)

    # ── Timed pass ───────────────────────────────────────────────────────────
    runtime.reset_caches()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Prefill timing
    prefill_start = time.perf_counter()
    if device.type == "cuda":
        ev_prefill_start = torch.cuda.Event(enable_timing=True)
        ev_prefill_end = torch.cuda.Event(enable_timing=True)
        ev_prefill_start.record()

    prompt_len = int(input_ids.shape[1])
    runtime.reset_caches()
    runtime._set_traffic_phase("prefill")
    if prompt_len > 1:
        logits = runtime._forward_prefill(input_ids.to(device))
    else:
        logits, _ = runtime.forward_token(input_ids[:, 0:1].to(device), position_index=0)

    if device.type == "cuda":
        ev_prefill_end.record()
        torch.cuda.synchronize()
        prefill_gpu_ms = float(ev_prefill_start.elapsed_time(ev_prefill_end))
    else:
        prefill_gpu_ms = (time.perf_counter() - prefill_start) * 1000.0

    prefill_ms_per_token = prefill_gpu_ms / max(prompt_len, 1)

    # Decode timing
    generated = input_ids.to(device).clone()
    runtime._set_traffic_phase("decode")
    runtime._materialize_lm_head_on_gpu()

    for step in range(max_new_tokens):
        step_start = time.perf_counter()
        if device.type == "cuda":
            ev_step_start = torch.cuda.Event(enable_timing=True)
            ev_step_end = torch.cuda.Event(enable_timing=True)
            ev_step_start.record()

        next_token = runtime._sample_next_token(
            logits[:, -1, :],
            do_sample=False,
            temperature=1.0,
            top_k=None,
            top_p=1.0,
        ).view(1, 1)
        generated = torch.cat([generated, next_token], dim=-1)
        logits, _ = runtime.forward_token(next_token, position_index=int(generated.shape[1]) - 1)

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

    # ── Collect metrics ───────────────────────────────────────────────────────
    vram_peak_mb = 0.0
    if device.type == "cuda":
        vram_peak_mb = float(torch.cuda.max_memory_allocated(device)) / (1024 ** 2)

    mean_decode_ms = sum(token_times_ms) / len(token_times_ms) if token_times_ms else 0.0
    decode_tps = 1000.0 / mean_decode_ms if mean_decode_ms > 0 else 0.0

    # Extract traffic stats from runtime if available.
    traffic_report = None
    try:
        traffic_report = runtime._last_traffic_report
    except AttributeError:
        pass

    return {
        "prompt_tokens": prompt_len,
        "new_tokens": max_new_tokens,
        "prefill_total_ms": prefill_gpu_ms,
        "prefill_ms_per_token": prefill_ms_per_token,
        "first_token_latency_ms": first_token_latency_ms,
        "mean_decode_ms_per_token": mean_decode_ms,
        "decode_tokens_per_second": decode_tps,
        "vram_peak_mb": vram_peak_mb,
        "token_times_ms": token_times_ms,
        "traffic_report": traffic_report,
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark streaming runtime throughput")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--sparse-basis-path", default=None)
    parser.add_argument("--attn-head-importance-path", default=None)
    parser.add_argument("--kv-basis-path", default=None)
    parser.add_argument("--prompt", default=_DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--warmup-tokens", type=int, default=5,
                        help="Tokens to generate in warmup pass (fills RAM cache, not timed)")
    parser.add_argument("--vram-hot-cache-gb", type=float, default=4.0)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args(argv)

    if AutoTokenizer is None:
        print("ERROR: transformers not installed", file=sys.stderr)
        sys.exit(1)

    print(f"[bench] Tokenizing prompt ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
    print(f"[bench] Prompt: {int(input_ids.shape[1])} tokens", flush=True)

    print(f"[bench] Initializing StreamingLlamaRuntime ...", flush=True)
    runtime = StreamingLlamaRuntime(
        model_name=args.model_name,
        sparse_basis_path=args.sparse_basis_path,
        attn_head_importance_path=args.attn_head_importance_path,
        kv_basis_path=args.kv_basis_path,
        vram_hot_cache_gb=args.vram_hot_cache_gb,
    )

    results = run_benchmark(
        runtime,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        warmup_tokens=args.warmup_tokens,
    )

    print(
        f"\n[bench] === RESULTS ===\n"
        f"  Prompt tokens:           {results['prompt_tokens']}\n"
        f"  Prefill total:           {results['prefill_total_ms']:.1f} ms\n"
        f"  Prefill per token:       {results['prefill_ms_per_token']:.1f} ms/tok\n"
        f"  First-token latency:     {results['first_token_latency_ms']:.1f} ms\n"
        f"  Mean decode latency:     {results['mean_decode_ms_per_token']:.1f} ms/tok\n"
        f"  Decode throughput:       {results['decode_tokens_per_second']:.2f} tok/s\n"
        f"  VRAM peak:               {results['vram_peak_mb']:.1f} MB",
        flush=True,
    )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[bench] Results written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
