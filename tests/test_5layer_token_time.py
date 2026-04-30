"""
5-layer per-token timing microbenchmark.
Monkey-patches rt.num_layers=5 so forward_token() only runs layers 0-4.
Requires a fully initialised StreamingLlamaRuntime (rt) with hot cache
pre-warmed for layers 0-4 and KV cache seeded to 256 tokens.
No tokenizer, no sampling, no LM head, no full prefill.

Usage:
  from <your_init_script> import build_runtime
  rt = build_runtime()          # loads model, pre-warms layers 0-4 only
  import tests.test_5layer_token_time as t
  t.run(rt)
"""
import time
import torch

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

N_LAYERS = 5
N_WARMUP = 3
N_MEASURE = 10
TARGET_MS = N_LAYERS * 2.4   # 12 ms for 5 layers


def run(rt):
    errors = []

    # Seed KV cache to 256 tokens so SDPA sees a realistic context length.
    # Use a dummy token id; position_index drives KV appends.
    dummy_token = torch.zeros(1, 1, dtype=torch.long, device=rt.device)

    orig_num_layers = rt.num_layers
    rt.num_layers = N_LAYERS   # limit loop at streaming_llama_runtime.py:7479

    try:
        # Warmup
        for pos in range(256, 256 + N_WARMUP):
            with torch.no_grad():
                rt.forward_token(dummy_token, position_index=pos, use_attention_cache=True)
        torch.cuda.synchronize(rt.device)

        # Measure
        t0 = time.perf_counter()
        for pos in range(256 + N_WARMUP, 256 + N_WARMUP + N_MEASURE):
            with torch.no_grad():
                rt.forward_token(dummy_token, position_index=pos, use_attention_cache=True)
        torch.cuda.synchronize(rt.device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        mean_token_ms = elapsed_ms / N_MEASURE
        mean_layer_ms = mean_token_ms / N_LAYERS
        projected_full = mean_layer_ms * 126

        print(f"  Mean {N_LAYERS}-layer token time : {mean_token_ms:.1f} ms")
        print(f"  Mean per-layer time            : {mean_layer_ms:.2f} ms  (target 2.4 ms)")
        print(f"  Projected 126-layer token time : {projected_full:.0f} ms  (target 303 ms)")

        def check(name, cond, detail=""):
            sym = PASS if cond else FAIL
            print(f"  {sym}  {name}" + (f": {detail}" if detail else ""))
            if not cond:
                errors.append(name)

        check(
            f"{N_LAYERS}-layer time < {TARGET_MS:.0f} ms",
            mean_token_ms < TARGET_MS,
            f"{mean_token_ms:.1f} ms",
        )
        check(
            "per-layer time < 2.4 ms",
            mean_layer_ms < 2.4,
            f"{mean_layer_ms:.2f} ms",
        )
        check(
            "projected 126-layer < 310 ms",
            projected_full < 310,
            f"{projected_full:.0f} ms",
        )

    finally:
        rt.num_layers = orig_num_layers

    if errors:
        print(f"FAILED: {errors}")
        raise SystemExit(1)
    print("All checks passed.")
