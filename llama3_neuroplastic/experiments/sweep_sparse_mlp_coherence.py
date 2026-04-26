from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

try:
    from .streaming_llama_runtime import StreamingLlamaRuntime
except ImportError:
    from streaming_llama_runtime import StreamingLlamaRuntime


_THROUGHPUT_PROBE_DEFAULTS: dict[str, Any] = {
    "sparse_mlp_prefill_mode": "hot_cache",
    "vram_hot_cache_gb": 5.25,
    "hot_cache_gb_list": "5.25",
    "attn_active_heads": 5,
    "attn_min_active_heads": 5,
    "attn_max_active_heads": 5,
    "active_heads_list": "5",
    "sparse_attn_prefill_mode": "sparse",
    "sparse_kv_prefill_mode": "sparse",
}


def _parse_int_list(raw: str, *, name: str, minimum: int = 1) -> list[int]:
    values: list[int] = []
    for part in str(raw).replace(";", ",").split(","):
        text = part.strip()
        if not text:
            continue
        value = int(text)
        if value < int(minimum):
            raise ValueError(f"{name} values must be >= {int(minimum)}, got {value}")
        values.append(int(value))
    if not values:
        raise ValueError(f"At least one {name} value is required")
    return values


def _parse_float_list(raw: str, *, name: str, minimum: float = 0.0) -> list[float]:
    values: list[float] = []
    for part in str(raw).replace(";", ",").split(","):
        text = part.strip()
        if not text:
            continue
        value = float(text)
        if value < float(minimum):
            raise ValueError(f"{name} values must be >= {float(minimum)}, got {value}")
        values.append(float(value))
    if not values:
        raise ValueError(f"At least one {name} value is required")
    return values


def _runtime_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        major, _minor = torch.cuda.get_device_capability(device)
        if int(major) < 8:
            return torch.float16
    return torch.bfloat16


def _encode_prompt(tokenizer: Any, prompt: str, prompt_format: str) -> torch.LongTensor:
    if str(prompt_format).strip().lower() == "chat":
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": str(prompt)}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        encoded = tokenizer(str(prompt), return_tensors="pt")
    return encoded["input_ids"].to(dtype=torch.long)


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _apply_throughput_probe_defaults(args: argparse.Namespace) -> None:
    for key, value in _THROUGHPUT_PROBE_DEFAULTS.items():
        setattr(args, key, value)


def _run_generation(
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    input_ids: torch.LongTensor,
    sparse_top_k: int | None,
    use_sparse_attention: bool,
    attn_active_heads: int | None,
    vram_hot_cache_gb: float,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = StreamingLlamaRuntime(
        model_name_or_path=str(args.model_name),
        device=device,
        dtype=_runtime_dtype(device),
        taylor_layers=[],
        local_files_only=bool(args.local_files_only),
        ram_cache=bool(args.ram_cache),
        materialize_lm_head=True,
        sparse_basis_path=str(args.sparse_basis_path) if sparse_top_k is not None else None,
        sparse_top_k=int(sparse_top_k) if sparse_top_k is not None else None,
        sparse_basis_top_k=int(args.sparse_basis_top_k) if sparse_top_k is not None else None,
        sparse_mlp_execution=str(args.sparse_mlp_execution) if sparse_top_k is not None else None,
        sparse_mlp_prefill_mode=str(args.sparse_mlp_prefill_mode),
        vram_hot_cache_gb=float(vram_hot_cache_gb),
        attn_head_importance_path=str(args.attn_head_importance_path) if use_sparse_attention else None,
        attn_active_heads=int(attn_active_heads) if use_sparse_attention and attn_active_heads is not None else None,
        attn_min_active_heads=(
            int(attn_active_heads)
            if use_sparse_attention and attn_active_heads is not None
            else int(args.attn_min_active_heads)
        ),
        attn_max_active_heads=(
            int(attn_active_heads)
            if use_sparse_attention and attn_active_heads is not None
            else (int(args.attn_max_active_heads) if use_sparse_attention else None)
        ),
        sparse_attn_prefill_mode=str(args.sparse_attn_prefill_mode) if use_sparse_attention else "dense",
        kv_basis_path=str(args.kv_basis_path) if use_sparse_attention and str(args.kv_basis_path).strip() else None,
        sparse_kv_prefill_mode=str(args.sparse_kv_prefill_mode) if use_sparse_attention else "dense",
        enable_triton_fused_sparse_mlp=not bool(args.disable_triton_fused_sparse_mlp),
        enable_cuda_h2d_overlap=not bool(args.disable_cuda_h2d_overlap),
    )
    runtime.enable_decode_profiler(True, max_steps=max(0, int(args.max_new_tokens)))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        generated = runtime.generate(
            input_ids=input_ids,
            max_new_tokens=int(args.max_new_tokens),
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=1.0,
            top_k=None,
            top_p=1.0,
        )
    elapsed = time.perf_counter() - t0
    first_decode_t = getattr(runtime, "_first_decode_t", None)
    decode_done_t = getattr(runtime, "_decode_done_t", None)
    if first_decode_t is not None and decode_done_t is not None and float(decode_done_t) >= float(first_decode_t):
        prefill_elapsed = max(float(first_decode_t) - float(t0), 0.0)
        decode_elapsed = max(float(decode_done_t) - float(first_decode_t), 1e-9)
    else:
        prefill_elapsed = 0.0
        decode_elapsed = max(float(elapsed), 1e-9)
    vram_peak_mb = (
        float(torch.cuda.max_memory_allocated(device)) / float(1024 ** 2)
        if device.type == "cuda"
        else 0.0
    )
    generated_cpu = generated.detach().to(device=torch.device("cpu"), dtype=torch.long)
    prompt_len = int(input_ids.shape[-1])
    generated_ids = [int(value) for value in generated_cpu[0].tolist()]
    completion_ids = generated_ids[prompt_len:]
    runtime_status = runtime.get_runtime_status()
    decode_profile_summary = dict(runtime_status.get("decode_profile_summary", {}))
    result = {
        "mode": "dense" if sparse_top_k is None else "sparse",
        "sparse_top_k": None if sparse_top_k is None else int(sparse_top_k),
        "vram_hot_cache_gb": float(vram_hot_cache_gb),
        "use_sparse_attention": bool(use_sparse_attention),
        "attn_active_heads": int(attn_active_heads) if attn_active_heads is not None else None,
        "generated_ids": generated_ids,
        "completion_ids": completion_ids,
        "text": tokenizer.decode(generated_ids, skip_special_tokens=False),
        "completion_text": tokenizer.decode(completion_ids, skip_special_tokens=False),
        "latency_s": float(elapsed),
        "prefill_latency_s": float(prefill_elapsed),
        "decode_latency_s": float(decode_elapsed),
        "decode_tok_s": float(max(len(completion_ids), 1) / max(decode_elapsed, 1e-9)),
        "total_tok_s": float(max(len(completion_ids), 1) / max(elapsed, 1e-9)),
        "mean_mlp_ms": float(decode_profile_summary.get("mean_mlp_ms", 0.0) or 0.0),
        "mean_load_attn_ms": float(decode_profile_summary.get("mean_load_attn_ms", 0.0) or 0.0),
        "decode_mlp_hot_blocks_hit": int(runtime_status.get("decode_mlp_hot_blocks_hit", 0)),
        "decode_mlp_cold_blocks_streamed": int(runtime_status.get("decode_mlp_cold_blocks_streamed", 0)),
        "decode_down_hot_blocks_hit": int(runtime_status.get("decode_down_hot_blocks_hit", 0)),
        "decode_down_cold_blocks_streamed": int(runtime_status.get("decode_down_cold_blocks_streamed", 0)),
        "vram_peak_mb": float(vram_peak_mb),
        "sparse_summary": runtime.get_sparse_mlp_summary() if sparse_top_k is not None else {},
        "runtime_status": runtime_status,
        "traffic": runtime.get_last_traffic_report(),
    }
    del runtime
    _release_cuda()
    return result


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_key(run: dict[str, Any]) -> tuple[int | None, bool, int | None, float, str, bool, str]:
    top_k = run.get("sparse_top_k")
    return (
        None if top_k is None else int(top_k),
        bool(run.get("use_sparse_attention", False)),
        None if run.get("attn_active_heads") is None else int(run.get("attn_active_heads")),
        float(run.get("vram_hot_cache_gb", 0.0)),
        str(run.get("sparse_attn_prefill_mode", "dense")),
        bool(run.get("use_sparse_kv", False)),
        str(run.get("sparse_kv_prefill_mode", "dense")),
    )


def _requested_run_key(
    *,
    top_k: int,
    use_sparse_attention: bool,
    attn_active_heads: int | None,
    vram_hot_cache_gb: float,
    sparse_attn_prefill_mode: str,
    use_sparse_kv: bool,
    sparse_kv_prefill_mode: str,
) -> tuple[int, bool, int | None, float, str, bool, str]:
    return (
        int(top_k),
        bool(use_sparse_attention),
        None if attn_active_heads is None else int(attn_active_heads),
        float(vram_hot_cache_gb),
        str(sparse_attn_prefill_mode) if use_sparse_attention else "dense",
        bool(use_sparse_kv),
        str(sparse_kv_prefill_mode) if use_sparse_kv else "dense",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep sparse MLP/attention/hot-cache settings against a dense decode baseline."
    )
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--sparse-basis-path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--prompt-format", type=str, default="raw", choices=["raw", "chat"])
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--top-k-list", type=str, default="128,160,192,208,256")
    parser.add_argument("--hot-cache-gb-list", type=str, default="0")
    parser.add_argument("--active-heads-list", type=str, default="")
    parser.add_argument("--sparse-basis-top-k", type=int, default=64)
    parser.add_argument("--sparse-mlp-execution", type=str, default="exact_blockwise_sparse")
    parser.add_argument("--sparse-mlp-prefill-mode", type=str, default="dense", choices=["dense", "sparse", "hot_cache"])
    parser.add_argument("--vram-hot-cache-gb", type=float, default=0.0)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ram-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-sparse-attention", action="store_true", default=False)
    parser.add_argument("--attn-head-importance-path", type=str, default="")
    parser.add_argument("--attn-active-heads", type=int, default=5)
    parser.add_argument("--attn-min-active-heads", type=int, default=5)
    parser.add_argument("--attn-max-active-heads", type=int, default=5)
    parser.add_argument("--sparse-attn-prefill-mode", type=str, default="dense", choices=["dense", "sparse"])
    parser.add_argument("--kv-basis-path", type=str, default="")
    parser.add_argument("--sparse-kv-prefill-mode", type=str, default="dense", choices=["dense", "sparse"])
    parser.add_argument("--disable-triton-fused-sparse-mlp", action="store_true", default=False)
    parser.add_argument("--disable-cuda-h2d-overlap", action="store_true", default=False)
    parser.add_argument("--throughput-probe", action="store_true", default=False)
    parser.add_argument("--stop-after-first-match", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-dense", action="store_true", default=False)
    parser.add_argument("--dense-json", type=str, default="")
    parser.add_argument("--output-json", type=str, required=True)
    return parser


def _update_best_coherent(report: dict[str, Any]) -> None:
    best_run = None
    best_tps = -1.0
    for run in report.get("runs", []):
        if not isinstance(run, dict):
            continue
        coherent = bool(run.get("completion_identical", False))
        if not coherent:
            continue
        tps = float(run.get("decode_tok_s", 0.0))
        if tps > best_tps:
            best_tps = tps
            best_run = {
                "sparse_top_k": int(run["sparse_top_k"]),
                "vram_hot_cache_gb": float(run.get("vram_hot_cache_gb", 0.0)),
                "attn_active_heads": run.get("attn_active_heads"),
                "decode_tok_s": tps,
                "mean_mlp_ms": float(run.get("mean_mlp_ms", 0.0)),
                "mean_load_attn_ms": float(run.get("mean_load_attn_ms", 0.0)),
            }
    report["best_completion_identical_run"] = best_run


def main() -> int:
    args = _build_parser().parse_args()
    if bool(args.throughput_probe):
        _apply_throughput_probe_defaults(args)
    output_path = Path(args.output_json)
    top_k_values = _parse_int_list(str(args.top_k_list), name="top-k", minimum=1)
    hot_cache_values = _parse_float_list(str(args.hot_cache_gb_list), name="hot-cache-gb", minimum=0.0)

    use_sparse_attention = bool(args.use_sparse_attention)
    if use_sparse_attention:
        if str(args.active_heads_list).strip():
            active_head_values = _parse_int_list(str(args.active_heads_list), name="active-heads", minimum=1)
        else:
            active_head_values = [int(args.attn_active_heads)]
    else:
        active_head_values = [None]

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_name),
        use_fast=True,
        local_files_only=bool(args.local_files_only),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    input_ids = _encode_prompt(tokenizer, str(args.prompt), str(args.prompt_format))

    report: dict[str, Any] = {
        "model_name": str(args.model_name),
        "prompt": str(args.prompt),
        "prompt_format": str(args.prompt_format),
        "max_new_tokens": int(args.max_new_tokens),
        "top_k_values": top_k_values,
        "hot_cache_gb_values": hot_cache_values,
        "active_head_values": active_head_values,
        "dense": None,
        "runs": [],
        "first_identical_config": None,
        "first_completion_identical_config": None,
        "best_completion_identical_run": None,
    }
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if isinstance(existing, dict):
            report.update(existing)

    dense_result: dict[str, Any] | None = None
    if str(args.dense_json).strip():
        dense_path = Path(args.dense_json)
        if dense_path.exists():
            loaded = json.loads(dense_path.read_text(encoding="utf-8"))
            dense_result = loaded.get("dense") if isinstance(loaded, dict) and "dense" in loaded else loaded
    if dense_result is None and isinstance(report.get("dense"), dict):
        dense_result = report["dense"]
    if dense_result is None and not bool(args.skip_dense):
        print("[sweep] running dense baseline", flush=True)
        dense_result = _run_generation(
            args=args,
            tokenizer=tokenizer,
            input_ids=input_ids,
            sparse_top_k=None,
            use_sparse_attention=False,
            attn_active_heads=None,
            vram_hot_cache_gb=0.0,
        )
        report["dense"] = dense_result
        _write_report(output_path, report)
    if dense_result is None:
        raise RuntimeError("Dense baseline is required. Provide --dense-json or omit --skip-dense.")
    report["dense"] = dense_result

    completed_run_keys = {
        _run_key(run)
        for run in report.get("runs", [])
        if isinstance(run, dict) and run.get("sparse_top_k") is not None
    }
    dense_ids = list(dense_result["generated_ids"])
    dense_completion_ids = list(dense_result["completion_ids"])
    use_sparse_kv = use_sparse_attention and bool(str(args.kv_basis_path).strip())
    stop_requested = False

    for top_k in top_k_values:
        for hot_cache_gb in hot_cache_values:
            for active_heads in active_head_values:
                run_key = _requested_run_key(
                    top_k=int(top_k),
                    use_sparse_attention=use_sparse_attention,
                    attn_active_heads=active_heads,
                    vram_hot_cache_gb=float(hot_cache_gb),
                    sparse_attn_prefill_mode=str(args.sparse_attn_prefill_mode),
                    use_sparse_kv=use_sparse_kv,
                    sparse_kv_prefill_mode=str(args.sparse_kv_prefill_mode),
                )
                if run_key in completed_run_keys:
                    continue
                print(
                    f"[sweep] running sparse top_k={int(top_k)} hot_cache_gb={float(hot_cache_gb):.2f} "
                    f"active_heads={active_heads}",
                    flush=True,
                )
                sparse_result = _run_generation(
                    args=args,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    sparse_top_k=int(top_k),
                    use_sparse_attention=use_sparse_attention,
                    attn_active_heads=active_heads,
                    vram_hot_cache_gb=float(hot_cache_gb),
                )
                sparse_result["sparse_attn_prefill_mode"] = (
                    str(args.sparse_attn_prefill_mode) if use_sparse_attention else "dense"
                )
                sparse_result["use_sparse_kv"] = bool(use_sparse_kv)
                sparse_result["sparse_kv_prefill_mode"] = (
                    str(args.sparse_kv_prefill_mode) if bool(use_sparse_kv) else "dense"
                )
                sparse_result["identical"] = list(sparse_result["generated_ids"]) == dense_ids
                sparse_result["completion_identical"] = list(sparse_result["completion_ids"]) == dense_completion_ids

                report.setdefault("runs", []).append(sparse_result)
                if sparse_result["identical"] and report.get("first_identical_config") is None:
                    report["first_identical_config"] = {
                        "sparse_top_k": int(top_k),
                        "vram_hot_cache_gb": float(hot_cache_gb),
                        "attn_active_heads": active_heads,
                    }
                if sparse_result["completion_identical"] and report.get("first_completion_identical_config") is None:
                    report["first_completion_identical_config"] = {
                        "sparse_top_k": int(top_k),
                        "vram_hot_cache_gb": float(hot_cache_gb),
                        "attn_active_heads": active_heads,
                    }

                _update_best_coherent(report)
                _write_report(output_path, report)
                print(
                    "[sweep] top_k="
                    f"{int(top_k)} hot_cache_gb={float(hot_cache_gb):.2f} active_heads={active_heads} "
                    f"completion_identical={bool(sparse_result['completion_identical'])} "
                    f"decode_tok_s={float(sparse_result['decode_tok_s']):.4f} "
                    f"mean_mlp_ms={float(sparse_result['mean_mlp_ms']):.2f} "
                    f"mean_load_attn_ms={float(sparse_result['mean_load_attn_ms']):.2f}",
                    flush=True,
                )
                if bool(args.stop_after_first_match) and bool(sparse_result["completion_identical"]):
                    stop_requested = True
                    break
            if stop_requested:
                break
        if stop_requested:
            break

    _update_best_coherent(report)
    _write_report(output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
