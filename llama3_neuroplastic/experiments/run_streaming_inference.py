#!/usr/bin/env python3
from __future__ import annotations

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


SMOKE_PROMPTS = [
    "Write one sentence about Norway.",
    "Explain why recurrent attention can reduce cache pressure.",
]


def _parse_layer_selection(spec: str | None) -> Optional[List[int]]:
    if spec is None or str(spec).strip() == "":
        return None
    out: set[int] = set()
    for part in str(spec).split(","):
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
    p.add_argument("--taylor-layers", type=str, default="", help="Layer selection, for example '0-31'")
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
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    prompts = list(args.prompt) if args.prompt else list(SMOKE_PROMPTS)
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
    )

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_name), use_fast=True, local_files_only=bool(args.local_files_only))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows: List[Dict[str, Any]] = []
    for idx, prompt in enumerate(prompts):
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
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
            )
        elapsed = time.perf_counter() - t0
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        new_tokens = max(int(generated.shape[-1] - input_ids.shape[-1]), 1)
        row = {
            "prompt_idx": int(idx),
            "latency_s": float(elapsed),
            "tok_s": float(new_tokens / max(elapsed, 1e-9)),
            "text": text,
        }
        rows.append(row)
        print(f"[prompt {idx}] latency_s={row['latency_s']:.3f} tok_s={row['tok_s']:.3f}")
        print(f"[prompt {idx}] text={text}")

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
