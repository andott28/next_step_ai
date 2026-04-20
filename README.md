# next_step_ai

Streaming inference runtime for Llama 3.1 405B on constrained VRAM. The maintained path visits all 126 transformer layers during real generation, but streams only the active layer to GPU and reduces per-layer transfer with learned-basis sparse MLP routing, sparse attention, sparse K/V loading, and VRAM hot-block caching.

## Maintained Surface

```text
next_step_ai/
├── llama3_neuroplastic/
│   ├── basis_fitting.py
│   ├── gqa_taylor_ssd.py
│   ├── layer_selection.py
│   ├── performance_utils.py
│   ├── token_posting_archive.py
│   ├── triton_sparse_mlp.py
│   └── experiments/
│       ├── streaming_llama_runtime.py
│       ├── run_streaming_inference.py
│       ├── init_learned_basis_from_dense_mlp.py
│       ├── init_kv_basis.py
│       ├── init_attn_share.py
│       ├── init_attn_token_posting_basis.py
│       ├── eval_perplexity.py
│       ├── benchmark.py
│       ├── verify_sparse_mlp_checkpoint.py
│       ├── verify_sparse_mlp_generation_pair.py
│       └── verify_sparse_mlp_runtime_summary.py
├── legacy/
│   └── older experimental full-model/SCA code and one-off diagnostics
├── tests/
│   └── test_streaming_llama_runtime.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

Anything under `legacy/` is not part of the supported runtime surface.

## Core Idea

Standard 405B inference does not fit in 8 GB VRAM. This runtime reduces resident VRAM and per-token transfer by combining:

1. Layer streaming: every generated token still traverses all 126 transformer layers, but only the active decoder layer is materialized on GPU.
2. Learned-basis MLP routing: covered layers route to selected FFN intermediate blocks and execute the original `SiLU(gate) * up -> down` math for those blocks.
3. Sparse attention and sparse K/V loading: attention artifacts reduce q/o and k/v transfer inside each layer.
4. NF4 hot-block caching: frequently used packed NF4 MLP blocks stay in VRAM to reduce repeated PCIe transfer.
5. Decode-time cache adaptation: cold MLP blocks and down-proj columns that are actually used during decode are promoted into the VRAM hot cache when budget remains, so calibration misses do not stay permanent misses.

Sparse execution here means less work inside each layer, not skipping transformer layers. Whole-layer skipping is not part of the maintained generation path because skipped layers would miss residual updates and K/V cache state.

The preferred sparse MLP artifact target is `intermediate_block_scores`; legacy `output_reconstruction` artifacts are approximate and should be treated as compatibility mode only. `exact_intermediate_sparse` requires an intermediate artifact with `score_weight` and `score_bias`; it will fail on legacy output artifacts with `decoder_blocks`.

## Install

```bash
pip install -e .
pip install -e ".[triton]"
pip install -e ".[eval]"
```

Requirements: Python 3.10+, PyTorch with CUDA, `bitsandbytes`, `transformers`, and enough host RAM for the selected cache settings.

## Fit Sparse MLP Basis

```bash
python -m llama3_neuroplastic.experiments.init_learned_basis_from_dense_mlp \
    --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" \
    --output-path results/mlp_basis_intermediate_full126.pt \
    --use-streaming-harness \
    --layers 0-125 \
    --max-rows-per-layer 4096 \
    --artifact-target intermediate_block_scores \
    --basis-rank 64 \
    --basis-top-k 64 \
    --sca-block-size 32 \
    --resume-save-every-batches 1 \
    --write-partial-output-every-batches 1
```

The non-streaming legacy fitting path is intentionally unsupported. Use `--use-streaming-harness`.

## Run Inference

```bash
python -m llama3_neuroplastic.experiments.run_streaming_inference \
    --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" \
    --local-files-only \
    --sparse-basis-path results/mlp_basis_intermediate_full126.pt \
    --sparse-mlp-execution auto \
    --sparse-top-k 51 \
    --attn-head-importance-path results/attn_head_importance_405b.pt \
    --attn-active-heads 9 \
    --attn-min-active-heads 9 \
    --attn-max-active-heads 9 \
    --kv-basis-path results/kv_basis_r32.pt \
    --sparse-kv-prefill-mode sparse \
    --sparse-attn-prefill-mode sparse \
    --vram-hot-cache-gb 2.0 \
    --pre-warm \
    --calibrate-hot-cache \
    --hot-cache-calibration-tokens 64 \
    --prompt-format chat \
    --prompt "What is the capital of France?"
```

Use `--pre-warm` when you want the VRAM hot cache loaded before the first prompt. Add `--calibrate-hot-cache` for throughput probes: it first loads the static hot cache, runs a short live sparse-routing pass from the actual prompt positions, replaces the hot-block map with the blocks that are actually selected by `_route_sparse_mlp(...)`, then rebuilds the VRAM cache. This intentionally makes startup longer so decode traffic is lower.

The runtime also adapts during decode. If a calibrated MLP block is missing from the VRAM hot cache, the first cold load still completes the token, then the loaded gate/up block or down-proj column is promoted into the hot cache if the budget allows. Sparse attention Q/O hot-cache reads also support partial hits: cached heads are used immediately and only missing heads take the cold path.

## 3.3 tok/s Probe

The 3.3 tok/s decode path assumes all 126 layers are still visited. The target comes from lowering per-layer traffic, not from reducing layer count. At about 23 MiB/layer, one token transfers about 2.8 GiB across all layers; on a roughly 9.3 GiB/s PCIe path, that is about 0.30 s/token, or 3.3 tok/s.

This path needs the intermediate sparse MLP artifact, sparse attention, sparse K/V prefill, live-route hot-cache calibration, decode-time hot-cache promotion, GPU LM head, and the Triton sparse MLP kernels. Do not use `results/mlp_basis_full126.pt` for this path; that file is a legacy `output_reconstruction` artifact with 512 output blocks and cannot run `exact_intermediate_sparse`.

Validated intermediate artifact shape:

```text
checkpoint = results/mlp_basis_intermediate_full126.pt
artifact_target = intermediate_block_scores
block_domain = intermediate
recommended_execution = exact_intermediate_sparse
layers = 126
num_blocks = 1664
```

Verify the artifact before probing throughput:

```bash
python -m llama3_neuroplastic.experiments.verify_sparse_mlp_checkpoint \
    --checkpoint results/mlp_basis_intermediate_full126.pt \
    --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" \
    --local-files-only \
    --expect-target intermediate_block_scores \
    --expect-block-domain intermediate \
    --expect-layers 126 \
    --expect-num-blocks 1664
```

Run the throughput probe:

```powershell
if (Test-Path .\verification_env\Scripts\Activate.ps1) { . .\verification_env\Scripts\Activate.ps1 }; $env:STREAMING_GPU_LM_HEAD="1"; $env:STREAMING_BACKGROUND_PREFETCH="1"; $env:STREAMING_WINDOWS_BATCH_PRELOAD="1"; $env:STREAMING_SHOW_PROGRESS="1"; $env:STREAMING_VRAM_HOT_CACHE_GB="5.25"; python -m llama3_neuroplastic.experiments.run_streaming_inference --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" --local-files-only --taylor-layers none --sparse-basis-path results/mlp_basis_intermediate_full126.pt --sparse-mlp-execution exact_intermediate_sparse --sparse-top-k 51 --sparse-basis-top-k 64 --sparse-mlp-prefill-mode hot_cache --vram-hot-cache-gb 5.25 --pre-warm --calibrate-hot-cache --hot-cache-calibration-tokens 64 --attn-head-importance-path results/attn_head_importance_405b.pt --attn-active-heads 5 --attn-min-active-heads 5 --attn-max-active-heads 5 --sparse-attn-prefill-mode sparse --kv-basis-path results/kv_basis_r32.pt --sparse-kv-prefill-mode sparse --prompt-format chat --max-new-tokens 64 --no-stream-output --dump-json results/throughput_3_3_probe.json --prompt "What is the capital of France?"
```

Pass conditions:

```text
decode_tok_s >= 3.30
mean decode latency <= 303 ms/token
decode traffic near 23 MiB/layer for the 3.3 tok/s target
LM head resident on GPU
VRAM hot cache remains enabled
Triton sparse MLP path remains enabled
```

Expected startup log sequence:

```text
[pre-warm] complete ...
[hot-cache-calibration] collecting live MLP routes from N token(s)
[hot-cache-calibration] updated 126/126 layers ...
[hot-cache-calibration] rebuilding VRAM hot-cache from live routes
[pre-warm] complete ...
```

If the run is slow, inspect `results/throughput_3_3_probe.json` and the console traffic line. High decode traffic means attention/K/V transfer or cold MLP block transfer is still too large; low traffic with low tok/s points to execution overhead or Triton fallback.

The main remaining performance risk is whether the calibrated hot-cache budget is large enough for the live routed blocks. If `--calibrate-hot-cache` updates far fewer than 126 layers, the calibration pass did not exercise sparse MLP routing for the full model. If it updates all layers but decode traffic is still high, raise `--vram-hot-cache-gb` until the active blocks fit or lower `--sparse-top-k` for the probe.

On Windows WDDM, the hot-cache capacity check uses both driver free VRAM and PyTorch allocator-reserved VRAM. This avoids false auto-clamping immediately after a large hot-cache rebuild when WDDM has not yet reflected deallocation in `mem_get_info()`.

## Optional Artifact Builders

```bash
python -m llama3_neuroplastic.experiments.init_kv_basis \
    --model-path "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" \
    --output-path results/kv_basis_r32.pt \
    --basis-rank 32 \
    --block-size 32 \
    --top-k 51 \
    --layers all

python -m llama3_neuroplastic.experiments.init_attn_share \
    --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" \
    --output-path results/attn_share_qokv.pt \
    --layers all

python -m llama3_neuroplastic.experiments.init_attn_token_posting_basis \
    --model-path "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" \
    --output-path results/attn_token_posting_basis.pt
```

There is currently no maintained `init_learned_attn_head_importance.py` in this tree. If you pass `--attn-head-importance-path`, the artifact must already exist.

## Benchmark

```bash
python -m llama3_neuroplastic.experiments.benchmark \
    --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" \
    --sparse-basis-path results/mlp_basis_intermediate_full126.pt \
    --attn-head-importance-path results/attn_head_importance_405b.pt \
    --taylor-layers none \
    --vram-hot-cache-gb 5.25 \
    --max-new-tokens 5 \
    --warmup-tokens 2 \
    --output-json results/benchmark.json
```

The benchmark warms the runtime, calibrates the VRAM hot cache from the actual timed prompt, rebuilds the hot cache, and records a traffic report in the output JSON.

## Evaluate Perplexity

```bash
python -m llama3_neuroplastic.experiments.eval_perplexity \
    --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" \
    --sparse-basis-path results/mlp_basis_intermediate_full126.pt \
    --dataset wikitext \
    --dataset-config wikitext-2-raw-v1 \
    --dataset-split test \
    --max-tokens 4096 \
    --output-json results/ppl_eval.json
```

## Verify Sparse MLP Artifacts

```bash
python -m llama3_neuroplastic.experiments.verify_sparse_mlp_checkpoint \
    --checkpoint results/mlp_basis_intermediate_full126.pt \
    --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" \
    --expect-target intermediate_block_scores \
    --expect-block-domain intermediate \
    --expect-layers 126 \
    --expect-num-blocks 1664

python -m llama3_neuroplastic.experiments.verify_sparse_mlp_runtime_summary \
    --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" \
    --sparse-basis-path results/mlp_basis_intermediate_full126.pt \
    --sparse-mlp-execution auto \
    --expect-execution exact_intermediate_sparse
```

## Run Tests

```bash
pytest tests/ -v
```

## Key Environment Variables

| Variable | Effect |
| --- | --- |
| `STREAMING_SPARSE_BASIS_EXECUTION` | `auto`, `output_basis_surrogate`, `routed_output_blocks`, `exact_intermediate_sparse`, or `exact_intermediate_sparse_oracle` |
| `STREAMING_SPARSE_MLP_PREFILL_MODE` | `dense`, `sparse`, or `hot_cache` |
| `STREAMING_VRAM_HOT_CACHE_GB` | Default VRAM hot-cache budget |
| `STREAMING_RAM_CACHE_MAX_GB` | Host RAM cache cap |
| `STREAMING_ALLOW_TAYLOR_WITH_SPARSE_ATTN` | Allows Taylor-SSD with sparse attention despite known divergence risk |
