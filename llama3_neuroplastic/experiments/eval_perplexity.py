"""eval_perplexity.py — Measure token-level perplexity of the streaming runtime.

Usage (after fitting a basis):

    python -m llama3_neuroplastic.experiments.eval_perplexity \
        --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" \
        --sparse-basis-path results/mlp_basis.pt \
        --dataset wikitext \
        --dataset-config wikitext-2-raw-v1 \
        --dataset-split test \
        --stride 512 \
        --max-tokens 4096 \
        --output-json results/ppl_eval.json

The script produces a JSON file with:
  - perplexity (exp of average cross-entropy)
  - bits_per_byte
  - num_tokens evaluated
  - per-chunk cross-entropy list (for debugging)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

try:
    from .streaming_llama_runtime import StreamingLlamaRuntime
except ImportError:
    from streaming_llama_runtime import StreamingLlamaRuntime


def _build_token_corpus(
    tokenizer: Any,
    dataset_name: str,
    dataset_config: str,
    dataset_split: str,
    text_column: str,
    max_tokens: int,
) -> torch.LongTensor:
    """Load a dataset and tokenize until we have at least max_tokens tokens."""
    if load_dataset is None:
        raise RuntimeError("datasets package is required: pip install datasets")
    dataset = load_dataset(dataset_name, dataset_config, split=dataset_split, streaming=True)
    all_ids: list[int] = []
    for row in dataset:
        text = str(row.get(text_column, "") or "").strip()
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
        if len(all_ids) >= max_tokens:
            break
    return torch.tensor(all_ids[:max_tokens], dtype=torch.long)


def evaluate_perplexity(
    runtime: StreamingLlamaRuntime,
    token_ids: torch.LongTensor,
    *,
    stride: int = 512,
    context_len: int = 2048,
) -> dict[str, Any]:
    """Sliding-window perplexity evaluation.

    Splits ``token_ids`` into overlapping windows of ``context_len`` tokens
    with a step of ``stride``. Only the last ``stride`` tokens of each window
    contribute to the NLL sum (the prefix is context only), mimicking the
    standard HuggingFace perplexity benchmark.

    Returns a dict with: perplexity, bits_per_byte, num_tokens, nll_chunks.
    """
    total_tokens = int(token_ids.shape[0])
    if total_tokens < 2:
        raise ValueError("Need at least 2 tokens to evaluate perplexity")

    nlls: list[float] = []
    evaluated_tokens = 0

    prev_end = 0
    for begin in range(0, total_tokens - 1, stride):
        end = min(begin + context_len, total_tokens)
        target_begin = max(begin, prev_end)
        if target_begin >= end - 1:
            prev_end = end
            continue

        input_ids = token_ids[begin:end].unsqueeze(0)
        target_ids = token_ids[begin + 1 : end + 1]




        with torch.no_grad():
            logits_all = _run_forward_only(runtime, input_ids)

        if logits_all is None:
            print(f"[ppl] WARNING: forward pass returned None for window [{begin}:{end}], skipping")
            prev_end = end
            continue


        score_start = int(target_begin - begin)
        score_end = int(end - 1 - begin)
        if score_start >= score_end:
            prev_end = end
            continue

        logits_slice = logits_all[0, score_start:score_end, :].float()
        target_slice = target_ids[score_start:score_end].to(logits_slice.device)

        nll = torch.nn.functional.cross_entropy(logits_slice, target_slice, reduction="sum").item()
        n = int(target_slice.shape[0])
        nlls.append(nll / max(n, 1))
        evaluated_tokens += n

        avg_so_far = sum(nlls) / len(nlls) if nlls else float("nan")
        ppl_so_far = math.exp(avg_so_far) if avg_so_far < 20 else float("inf")
        print(
            f"[ppl] window [{begin}:{end}] scored {n} tokens | "
            f"chunk NLL={nll/n:.4f} | running PPL={ppl_so_far:.2f}",
            flush=True,
        )
        prev_end = end
        if end >= total_tokens:
            break

    if not nlls:
        raise RuntimeError("No tokens were scored — check stride/context_len vs. corpus size")

    mean_nll = sum(nlls) / len(nlls)
    ppl = math.exp(mean_nll)
    bpb = mean_nll / math.log(2)
    return {
        "perplexity": ppl,
        "bits_per_byte": bpb,
        "num_tokens": evaluated_tokens,
        "nll_chunks": nlls,
        "mean_nll": mean_nll,
    }


def _run_forward_only(
    runtime: StreamingLlamaRuntime,
    input_ids: torch.LongTensor,
) -> torch.Tensor | None:
    """Run a single prefill pass and return logits for all positions.

    This bypasses the generate() sampling loop and uses the runtime prefill
    façade before replaying token logits for perplexity scoring.
    """
    try:
        with torch.no_grad():
            return runtime.score_window_logits(input_ids).detach().cpu()
    except Exception as exc:
        print(f"[ppl] forward pass error: {type(exc).__name__}: {exc}", flush=True)
        return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate streaming runtime perplexity")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--sparse-basis-path", default=None)
    parser.add_argument("--attn-head-importance-path", default=None)
    parser.add_argument("--kv-basis-path", default=None)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--context-len", type=int, default=2048)
    parser.add_argument("--vram-hot-cache-gb", type=float, default=4.0)
    parser.add_argument("--hard-cuda-cache-flush", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args(argv)

    if AutoTokenizer is None:
        print("ERROR: transformers not installed", file=sys.stderr)
        sys.exit(1)

    print(f"[ppl] Loading tokenizer from {args.model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"[ppl] Tokenizing {args.dataset}/{args.dataset_config} ({args.dataset_split})", flush=True)
    token_ids = _build_token_corpus(
        tokenizer,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        text_column=args.text_column,
        max_tokens=args.max_tokens,
    )
    print(f"[ppl] Corpus: {token_ids.shape[0]} tokens", flush=True)

    print("[ppl] Initializing StreamingLlamaRuntime ...", flush=True)
    runtime = StreamingLlamaRuntime(
        model_name_or_path=args.model_name,
        sparse_basis_path=args.sparse_basis_path,
        attn_head_importance_path=args.attn_head_importance_path,
        kv_basis_path=args.kv_basis_path,
        vram_hot_cache_gb=args.vram_hot_cache_gb,
        hard_cuda_cache_flush=bool(args.hard_cuda_cache_flush),
    )

    print(f"[ppl] Evaluating perplexity (stride={args.stride}, context={args.context_len}) ...", flush=True)
    results = evaluate_perplexity(
        runtime,
        token_ids,
        stride=args.stride,
        context_len=args.context_len,
    )

    print(
        f"\n[ppl] === RESULTS ===\n"
        f"  Perplexity:   {results['perplexity']:.3f}\n"
        f"  Bits/byte:    {results['bits_per_byte']:.4f}\n"
        f"  Tokens eval:  {results['num_tokens']}\n"
        f"  Mean NLL:     {results['mean_nll']:.4f}",
        flush=True,
    )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[ppl] Results written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
