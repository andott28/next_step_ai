#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

try:
    from .neuroplastic_llama_gqa_mamba import NeuroplasticLlama
except ImportError:  # pragma: no cover
    from neuroplastic_llama_gqa_mamba import NeuroplasticLlama


SMOKE_PROMPTS = [
    "Write one sentence about Norway.",
    "Explain why recurrent models can reduce memory traffic.",
    "Summarize grouped-query attention in two sentences.",
    "Describe a sparse MLP routing system.",
    "Give one short fact about Mamba-2.",
]


def _load_artifact(path: str) -> Dict[str, Any]:
    blob = torch.load(path, map_location="cpu")
    if not isinstance(blob, dict):
        raise RuntimeError("Hybrid attention artifact must be a dict")
    return blob


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run hybrid GQA-Mamba inference and print performance counters.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to hybrid_attention_state.pt")
    p.add_argument("--model-name", type=str, default="", help="Optional override for base model name")
    p.add_argument("--prompt", action="append", default=[], help="Prompt to run; may be repeated")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--sparse-mlp-manifest", type=str, default="", help="Optional MLP bank manifest to load at runtime")
    p.add_argument("--require-honest-sparse", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--sca-recalibrated-checkpoint",
        type=str,
        default="",
        help="Optional sca_recalibrated_state.pt to load after hybrid checkpoint",
    )
    p.add_argument(
        "--decoder-mirror-checkpoint",
        type=str,
        default="",
        help="Optional decoder_mirror_sca_state.pt to load after hybrid/SCA checkpoints",
    )
    p.add_argument("--allow-cache", action="store_true", help="Allow generation cache in hybrid mode")
    p.add_argument("--enable-sparse-mlp", action="store_true", help="Enable sparse-MLP routing path")
    p.add_argument("--sca-decode-guard-layers", type=int, default=12, help="Number of top decode layers to keep dense.")
    p.add_argument("--enable-sparse-attention", action="store_true", help="Enable paged sparse attention decode path")
    p.add_argument("--local-window-tokens", type=int, default=2048)
    p.add_argument("--sink-tokens", type=int, default=8)
    p.add_argument("--page-size-tokens", type=int, default=256)
    p.add_argument("--retrieval-top-k-pages", type=int, default=8)
    p.add_argument(
        "--retrieval-head-groups",
        type=str,
        default="0",
        help="Comma-separated KV group ids for long-range retrieval (example: 0,1)",
    )
    p.add_argument("--retrieval-start-layer", type=int, default=-1)
    p.add_argument("--archive-cpu-dtype", type=str, choices=["int4", "fp16"], default="int4")
    p.add_argument("--hot-archive-gpu-pages", type=int, default=0)
    p.add_argument("--strict-fully-sparse", action="store_true")
    p.add_argument("--sca-top-k", type=int, default=3)
    p.add_argument("--sca-basis-rank", type=int, default=32)
    p.add_argument("--sca-basis-top-k", type=int, default=8)
    p.add_argument(
        "--sca-spmm-impl",
        type=str,
        choices=["dense", "cuda_spmm", "torch_block_sparse"],
        default="",
        help="Override sparse MLP spmm backend. Empty uses strict/non-strict default behavior.",
    )
    p.add_argument(
        "--sca-sparse-placement",
        type=str,
        choices=["input_mask", "output_sparse", "intermediate_group", "learned_basis"],
        default="input_mask",
        help="Where to apply sparse masking inside the MLP.",
    )
    p.add_argument(
        "--sca-fallback-threshold",
        type=float,
        default=None,
        help="Override sparse MLP dense fallback threshold. Empty defaults to 0.0 (no dense fallback).",
    )
    p.add_argument("--sca-adaptive-top-k", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sca-adaptive-top-k-min", type=int, default=3)
    p.add_argument("--sca-adaptive-top-k-max", type=int, default=12)
    p.add_argument("--sca-adaptive-top-k-min-score-ratio", type=float, default=0.15)
    p.add_argument(
        "--allow-strict-noncuda-spmm",
        action="store_true",
        help="Allow strict runtime checks with non-cuda spmm backend for diagnosis only.",
    )
    p.add_argument(
        "--enable-strict-repetition-penalty",
        action="store_true",
        help="Re-enable strict decode repetition penalty; disabled by default because it destabilizes sparse decode.",
    )
    p.add_argument("--dump-attention-diagnostics-json", type=str, default="")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    allow_cache_effective = bool(args.allow_cache) or (bool(args.strict_fully_sparse) and bool(args.enable_sparse_attention))
    sca_spmm_impl = str(args.sca_spmm_impl).strip() if str(args.sca_spmm_impl).strip() else (
        "cuda_spmm" if bool(args.strict_fully_sparse) else "dense"
    )
    fallback_threshold = 0.0 if args.sca_fallback_threshold is None else float(args.sca_fallback_threshold)

    artifact = _load_artifact(args.checkpoint)
    model_name = args.model_name or str(artifact.get("model_name", "unsloth/Meta-Llama-3.1-8B-bnb-4bit"))
    layer_selection = artifact.get("layer_selection")
    target_rank = artifact.get("target_rank", 16)
    variance_threshold = artifact.get("variance_threshold", 0.90)
    state_dim = artifact.get("state_dim")

    retrieval_groups = [int(x.strip()) for x in str(args.retrieval_head_groups).split(",") if x.strip()]
    retrieval_start_layer = None if int(args.retrieval_start_layer) < 0 else int(args.retrieval_start_layer)
    model = NeuroplasticLlama(
        model_name=model_name,
        neuroplasticity_enabled=bool(args.enable_sparse_mlp),
        sca_use_cuda=True,
        sca_top_k=int(args.sca_top_k),
        sca_basis_rank=int(args.sca_basis_rank),
        sca_basis_top_k=int(args.sca_basis_top_k),
        sca_spmm_impl=sca_spmm_impl,
        sca_sparse_placement=str(args.sca_sparse_placement),
        sca_stability_dense_fallback_threshold=float(fallback_threshold),
        sca_adaptive_top_k=bool(args.sca_adaptive_top_k),
        sca_adaptive_top_k_min=int(args.sca_adaptive_top_k_min),
        sca_adaptive_top_k_max=int(args.sca_adaptive_top_k_max),
        sca_adaptive_top_k_min_score_ratio=float(args.sca_adaptive_top_k_min_score_ratio),
        sca_decode_guard_layers=int(args.sca_decode_guard_layers),
        attention_hybrid_enabled=True,
        attention_hybrid_layers=layer_selection,
        attention_hybrid_target_rank=target_rank,
        attention_hybrid_variance_threshold=variance_threshold,
        attention_hybrid_state_dim=state_dim,
        attention_hybrid_force_no_cache=not bool(allow_cache_effective),
        attention_sparse_mode=bool(args.enable_sparse_attention),
        attention_local_window_tokens=int(args.local_window_tokens),
        attention_sink_tokens=int(args.sink_tokens),
        attention_page_size_tokens=int(args.page_size_tokens),
        attention_retrieval_top_k_pages=int(args.retrieval_top_k_pages),
        attention_retrieval_head_group_ids=retrieval_groups,
        attention_retrieval_start_layer=retrieval_start_layer,
        attention_archive_cpu_dtype=str(args.archive_cpu_dtype),
        attention_hot_archive_gpu_pages=int(args.hot_archive_gpu_pages),
        attention_disable_ssd_fetch_in_decode=True,
        attention_force_single_model_runtime=True,
        strict_decode_enable_repetition_penalty=bool(args.enable_strict_repetition_penalty),
        strict_runtime_allow_noncuda_spmm_diagnostic=bool(args.allow_strict_noncuda_spmm),
    )
    model.set_sparse_attention_mode(strict_fully_sparse=bool(args.strict_fully_sparse))
    load_info = model.load_hybrid_attention_state(args.checkpoint, strict=True)
    sca_recalibration_info: Dict[str, Any] = {}
    decoder_mirror_info: Dict[str, Any] = {}
    if args.sca_recalibrated_checkpoint:
        sca_recalibration_info = model.load_sca_recalibration_state(args.sca_recalibrated_checkpoint, strict=True)
    if args.decoder_mirror_checkpoint:
        decoder_mirror_info = model.load_decoder_mirror_state(args.decoder_mirror_checkpoint, strict=True)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = list(args.prompt) if args.prompt else list(SMOKE_PROMPTS)
    manifest_summary: Dict[str, Any] = {}
    manifest_load_info: Dict[str, Any] = {}
    if args.sparse_mlp_manifest:
        manifest_path = Path(args.sparse_mlp_manifest)
        manifest_summary = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_load_info = model.load_sparse_mlp_bank_manifest(str(manifest_path), strict=False)

    print(f"load_info={load_info}")
    if sca_recalibration_info:
        print(f"sca_recalibration_load_info={sca_recalibration_info}")
    if decoder_mirror_info:
        print(f"decoder_mirror_load_info={decoder_mirror_info}")
    print(f"effective_hybrid_layers={model.get_effective_hybrid_layers()}")
    print(f"allow_cache={bool(allow_cache_effective)} enable_sparse_mlp={bool(args.enable_sparse_mlp)}")
    print(f"enable_sparse_attention={bool(args.enable_sparse_attention)} strict_fully_sparse={bool(args.strict_fully_sparse)}")
    print(
        "runtime_flags="
        + json.dumps(
            {
                "disable_task_bias_injection": bool(model.disable_task_bias_injection),
                "strict_decode_enable_repetition_penalty": bool(model.strict_decode_enable_repetition_penalty),
                "strict_runtime_allow_noncuda_spmm_diagnostic": bool(model.strict_runtime_allow_noncuda_spmm_diagnostic),
                "sca_spmm_impl": str(model.sca_config.spmm_impl),
                "sca_sparse_placement": str(model.sca_config.sparse_placement),
                "sca_top_k": int(model.sca_config.top_k),
                "sca_basis_rank": int(model.sca_config.basis_rank),
                "sca_basis_top_k": int(model.sca_config.basis_top_k),
                "sca_fallback_threshold": float(model.sca_config.stability_dense_fallback_threshold),
                "sca_adaptive_top_k": bool(model.sca_config.adaptive_top_k),
                "sca_route_top_k": int(model.sca_config.route_top_k),
                "sca_decode_guard_layers": int(model.decode_guard_layers),
            }
        )
    )
    if manifest_summary:
        print(
            "sparse_mlp_manifest="
            + json.dumps(
                {
                    "num_layers": len(manifest_summary.get("layers", [])),
                    "block_size": manifest_summary.get("sca_config", {}).get("block_size"),
                    "export_dtype": manifest_summary.get("export_dtype"),
                    "output_dir": str(Path(args.sparse_mlp_manifest).parent),
                }
            )
        )
        print(f"sparse_mlp_bank_load_info={manifest_load_info}")

    for idx, prompt in enumerate(prompts):
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=bool(args.do_sample),
                temperature=float(args.temperature),
                top_k=int(args.top_k),
                top_p=float(args.top_p),
                use_cache=bool(allow_cache_effective),
                task_id=0,
            )
        elapsed = time.perf_counter() - t0
        total_new = max(int(generated.shape[-1] - input_ids.shape[-1]), 1)
        tps = float(total_new / max(elapsed, 1e-9))
        diagnostics = model.get_sparse_mlp_diagnostics()
        sparse_attention_diag = model.get_sparse_attention_diagnostics()
        decoder_mirror_diag = model.get_decoder_mirror_diagnostics()
        text = tokenizer.decode(generated[0], skip_special_tokens=True)

        print(f"[prompt {idx}] latency_s={elapsed:.3f} tok_s={tps:.3f}")
        print(f"[prompt {idx}] mean_mix={sum(artifact.get('mix_values_by_layer', {}).values()) / max(len(artifact.get('mix_values_by_layer', {})), 1):.6f}")
        print(
            f"[prompt {idx}] sparse_mlp_fraction={diagnostics['mean_touched_weight_fraction']:.6f} "
            f"bytes_per_token={diagnostics['estimated_bytes_fetched_per_token']:.1f} "
            f"fallback_rate={diagnostics.get('mean_dense_fallback_rate', 0.0):.6f}"
        )
        if bool(args.enable_sparse_mlp) and bool(args.require_honest_sparse):
            fallback_rate = float(diagnostics.get("mean_dense_fallback_rate", 0.0))
            if fallback_rate > 0.0:
                raise RuntimeError(
                    f"Honest sparse check failed: dense fallback_rate={fallback_rate:.6f} > 0. "
                    "Set --no-require-honest-sparse only for debugging."
                )
        print(
            f"[prompt {idx}] sparse_attention_pages_per_step={sparse_attention_diag.get('mean_selected_pages_per_step', 0.0):.3f} "
            f"cpu_to_gpu_bytes_step={sparse_attention_diag.get('mean_bytes_cpu_to_gpu_per_step', 0.0):.1f}"
        )
        if decoder_mirror_diag.get("enabled"):
            print(
                f"[prompt {idx}] decoder_mirror_blocks={decoder_mirror_diag.get('mean_active_blocks', 0.0):.3f} "
                f"delta_ratio={decoder_mirror_diag.get('delta_norm_ratio', 0.0):.6f}"
            )
        print(f"[prompt {idx}] text={text}")

    if args.dump_attention_diagnostics_json:
        out = {
            "sparse_attention": model.get_sparse_attention_diagnostics(),
            "sparse_mlp": model.get_sparse_mlp_diagnostics(),
            "decoder_mirror": model.get_decoder_mirror_diagnostics(),
        }
        out_path = Path(args.dump_attention_diagnostics_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote diagnostics: {out_path}")


if __name__ == "__main__":
    main()
