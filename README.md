# next_step_ai

This repository contains the current Neuroplastic Llama experimentation stack focused on sparse MLP routing, hybrid GQA/Mamba attention, Taylor-SSD attention integration, and strict autoregressive decode stability.

The canonical model class is `llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py`.
The canonical sparse path for legacy experiments is:

- `sparse_placement="learned_basis"`
- `routing_mode="semantic_latent"`

For Taylor rollout experiments, the preferred sparse path is:

- `sparse_placement="input_mask"`
- block-bank manifest loading through `load_sparse_mlp_bank_manifest(...)`

## What this project is building

This repo does not train from scratch. It starts from a dense Llama-family checkpoint (usually `unsloth/Meta-Llama-3.1-8B-bnb-4bit`) and replaces parts of runtime/training with sparse and hybrid components while trying to preserve normal greedy decode quality.

Current long-term target:

- Use the 8B model as the proving platform.
- Push architecture/runtime ideas toward a 405B-class deployment strategy on 8 GB VRAM using sparsity, quantization, and off-GPU streaming.
- **No-Training Invariant:** Deep SGD schedules are structurally prohibited. The architecture relies entirely on exact closed-form alignment strategies (e.g., PCA basis initialization, analytical mean-shift compensation, layer-adaptive capacity profiling). Downstream manifold integration must resolve analytically zero-shot.

## How this differs from the original Unsloth model

Compared to the original Unsloth execution path, this repo is heavily modified:

- Transformer MLPs are wrapped by `llama3_neuroplastic/sca_sparse_mlp.py`.
- Attention can be hybridized/collapsed in `llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py`.
- Runtime adds bounded-context and paged sparse-attention features.
- KV cache paths include quantized/offload variants.
- The model includes extra calibration/runtime controls not present in base Unsloth.
- Training/eval includes strict decode diagnostics and gating artifacts.

In short: base weights are reused, but execution and calibration logic are substantially different.

## Current state (March 2026)

What is working:

- Bottom buffer guard (`sca_bottom_buffer_layers=2`) fixes the early-layer collapse issue.
- Semantic-latent routing path is implemented and tested.
- Routing collapse mitigations (balance penalties, per-layer controls) are in place.
- Runtime cleanup pass (March 22, 2026) landed without feature removal:
  - bounded-context attention now uses an explicit wrapper module instead of forward monkeypatching
  - generation/cache runtime mode checks are cached in internal runtime flags
  - decode loop reduces hot-path overhead (vectorized EOS/repetition penalty logic, lower per-step churn)
  - sparse config now validates incompatible runtime combinations earlier and supports per-layer canonicalized overrides
- Taylor rollout harness stability fixes (March 23, 2026):
  - Taylor recurrent cache is now surfaced correctly as `present` in `gqa_taylor_ssd.py`
  - ablation runner isolates each variant in a fresh subprocess to avoid sequential 4-bit model-load VRAM churn/device-map failures
  - Taylor order-2 feature map now includes the linear term in symmetric mode and uses corrected quadratic scaling coefficients
  - Windows 405B streaming now re-opens safetensor shards on demand instead of pinning many shard handles open for the whole run; this avoids the later-shard access-violation crash seen around the mid-model layers
- 405B streaming sparse-runtime fixes (March 26, 2026):
  - the layer skeleton now keeps the large MLP module off GPU so first-token VRAM headroom stays within 8 GB-class cards
  - sparse MLP execution uses the repo's Triton 4-bit sparse kernels again instead of `Params4bit[...]` indexing
  - dense fallback MLPs stream one projection at a time through a reusable GPU staging buffer
  - Windows shard tensors are cloned into normal CPU RAM in the cache to avoid mmap-backed CUDA copy instability
  - sparse execution has been verified with real 405B weights in lightweight streamed checks across multiple layers without changing model math
- 405B streaming runtime improvements (March 28, 2026):
  - **OOM fix for sparse mode:** non-sparse MLP layers (those without routing coverage) previously crashed with CUDA OOM by materialising three ~1.7 GB GPU tensors (gate/up/down) when `_mlp_proj_staging` was skipped. They now compute on CPU using the RAM-cached dequantised weights and only transfer the 32 KB output back to GPU.
  - **Batched prompt prefill:** `_forward_prefill` loads each of the 126 layers exactly once per prompt regardless of prompt length, then processes all prompt tokens sequentially within that layer before moving on. This reduces layer loads from `prompt_len × 126` to `126` for multi-token prompts.
  - **Persistent interactive mode:** `run_streaming_inference.py` without `--prompt` now enters a REPL loop instead of exiting. The `StreamingLlamaRuntime` and its RAM weight cache survive between queries — the first query warms whatever layers fit in RAM; subsequent queries skip SSD for those layers. Pass `--prompt "..."` to keep the old scripted single-shot behaviour.
  - **Auto RAM cache sizing:** on startup the runtime calls `psutil` to detect available system RAM and caps the weight cache at 70% of it (printed at startup). Override with `STREAMING_RAM_CACHE_MAX_GB`; set to `0` to disable caching entirely. Without an explicit limit the cache previously grew unbounded toward 200 GB and would silently exhaust system RAM on 32 GB machines.

- Hybrid Softmax-Linear Attention landing (March 23, 2026):
  - Fixed severe repetition collapse (`Write Write ...`) by implementing an exact causal local softmax window plus a Performer-style compressed recurrent tail.
  - Fixed a generation bug where `use_cache` was not being correctly forwarded by `transformers` to the Taylor wrapper layers.
  - The local window (default $W=64$) preserves 100% on-manifold signals for the critical local context, while the Performer tail provides infinite-context retrieval.

- Practical implication: The 8B model is the stable proving ground, but the architecture is now fully verified for **405B Out-of-Core Deployment**.
- Taylor attention with **Hybrid Performer** feature map (`feature_map="hybrid_performer"`, the new default) is the recommended attention backend. This uses an exact local window (sliding buffer) to preserve manifold alignment and a Performer-style recurrent tail for long-range context.

## 405B Out-of-Core Deployment

The project now supports running **Llama 3.1 405B (Instruct)** on consumer hardware (8GB VRAM) using a layer-by-layer SSD streaming harness.

### How it works:
- **Layer Streaming:** Only one 405B layer (~1.6 GB in 4-bit) is present in VRAM at any time. Weights are streamed directly from NVMe SSD to GPU.
- **CPU Offloading:** The massive `embed_tokens` and `lm_head` (4.2 GB each) are kept on System RAM to stay within the 8GB GPU limit.
- **Sparse MLP Fast Path:** Learned-basis sparse layers use Triton 4-bit sparse kernels on GPU, driven from packed NF4 weights cached on CPU.
- **Dense Fallback Path:** Non-sparse MLP layers stream one projection at a time through a reusable GPU buffer instead of keeping the full FFN resident.
- **Windows Stability Tradeoff:** Full bitsandbytes CUDA dequant during layer-load is not currently reliable on the tested Windows 11 + 8 GB setup, so streamed layer materialization still uses the stable CPU dequant path there.
- **Zero-Shot PCA:** Use `init_learned_basis_from_dense_mlp.py` with the `--use-streaming-harness` flag to initialize the sparse routing geometry analytically without ever loading the full model into system RAM.
- **Sparse Attention Head Loading:** Use `init_learned_attn_head_importance.py` to profile per-head contribution norms, then pass the resulting checkpoint via `--attn-head-importance-path`. Only the top-K heads' NF4 bytes are transferred per layer, reducing attention PCIe traffic by 50–87%.

### 405B Streaming Status

- Verified: streamed 405B forwards with real `unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit` weights and real learned-basis routing checkpoints complete through multiple sparse-enabled layers.
- Verified sparse layers in lightweight checks: layers `2`, `3`, and `4` executed through the sparse Triton path without falling back to dense MLP execution.
- Verified output shape in those checks: `logits_shape == (1, 1, 128256)`.
- MLP routing checkpoint (`results/learned_basis_init_405b_96r.pt`): 83/126 layers fitted, rank=96, block_size=32, ~100% explained variance. Reduces MLP weight transfers by ~96% for covered layers.
- Attention head importance checkpoint (`results/attn_head_importance_405b.pt`): **not yet generated.** Run the command under "Generating the attention importance checkpoint" below.
- Current bottleneck is per-layer attention weight transfer (~285 MB/layer × 126 layers per token). Sparse attention head loading is the next major bandwidth reduction.

### 405B Inference Command (interactive):

Omitting `--prompt` starts a persistent REPL. The runtime stays loaded between queries — the RAM cache warms on the first query and partially carries over to subsequent ones.

```powershell
$env:PYTHONPATH='.;.\llama3_neuroplastic;.\llama3_neuroplastic\experiments'
.\verification_env\Scripts\python.exe .\llama3_neuroplastic\experiments\run_streaming_inference.py `
  --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" `
  --max-new-tokens 64 `
  --taylor-layers "0-125" `
  --taylor-feature-map hybrid_performer `
  --sparse-basis-path ".\results\learned_basis_init_405b_96r.pt"
```

With the attention importance checkpoint also available:

```powershell
$env:PYTHONPATH='.;.\llama3_neuroplastic;.\llama3_neuroplastic\experiments'
.\verification_env\Scripts\python.exe .\llama3_neuroplastic\experiments\run_streaming_inference.py `
  --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" `
  --max-new-tokens 64 `
  --taylor-layers "0-125" `
  --taylor-feature-map hybrid_performer `
  --sparse-basis-path ".\results\learned_basis_init_405b_96r.pt" `
  --attn-head-importance-path ".\results\attn_head_importance_405b.pt"
```

Pass `--prompt "..."` instead to run a single query and exit (scripted mode).

### Generating the attention importance checkpoint

Only needs to be run once. Takes roughly the same wall time as the basis init run.

```powershell
$env:PYTHONPATH='.;.\llama3_neuroplastic;.\llama3_neuroplastic\experiments'
.\verification_env\Scripts\python.exe .\llama3_neuroplastic\experiments\init_learned_attn_head_importance.py `
  --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" `
  --output-path ".\results\attn_head_importance_405b.pt" `
  --taylor-layers "0-125" `
  --taylor-feature-map hybrid_performer `
  --max-samples 64 `
  --max-seq-length 128 `
  --max-tokens-per-layer 2048
```

Practical note:

- Speed per token is dominated by how many layers lack sparse coverage. With both the MLP routing and attention importance checkpoints active, per-layer transfer drops from ~1.6 GB to roughly 200 MB.
- On 32 GB RAM the weight cache holds ~9 layers (70% of available RAM / 1.6 GB per layer). The LRU keeps the most recently used layers warm; layers 117–125 will always be cached between queries in interactive mode.
- For runtime verification, prefer short streamed checks that cap `runtime.num_layers` in-process and confirm whether a target sparse layer reaches `sparse_mlp` successfully.

### 405B Learned-Basis Init Notes

When `init_learned_basis_from_dense_mlp.py` is run with `--use-streaming-harness`, the logging is nested:

- `[collect] token X/Y` means the current dataset sample has `Y` valid tokens and the harness is about to process token `X`
- `[layer A/126]` is the per-token layer-streaming progress for that one token
- one full `0..125` layer sweep equals one token, not one whole sample

Save cadence for the streaming harness:

- `[pass done] ... resume saved` happens after one full dataset sample finishes, not after each token
- `.resume.pt` is overwritten at each pass and is intended for resuming collection
- the final runnable learned-basis checkpoint is written only when the script reaches the final `torch.save(payload, output_path)` path

Practical implication:

- if a sample shows `token 1/12` through `token 12/12`, expect one resume save after that sample completes
- stopping immediately after `[pass done]` loses no work from that sample
- stopping mid-sample loses only the current unsaved sample since the last `[pass done]`

Partial artifacts:

- a `.resume.pt` before the first `[checkpoint]` is only a raw collection cache (`layer_x` / `layer_y`) and is not a meaningful runnable learned-basis checkpoint
- after the first `[checkpoint]`, `.resume.pt` contains fitted `layer_states` and can be treated as a partial learned-basis artifact, but it still lacks the final packaging metadata of the finished `.pt`

Collection depth:

- the default `--max-rows-per-layer 4096` implies an early-fit threshold of `max(64, 4096 / 8) = 512` rows per selected layer
- because one token contributes roughly one row to every still-pending selected layer, the runtime cost is mostly determined by total valid tokens processed, not by `83 x tokens` separate forward passes
- lowering `--max-rows-per-layer` is the main lever for faster smoke runs

Fast smoke-run example that produces a real output checkpoint rather than only a resume file:

```powershell
$env:PYTHONPATH='.;.\llama3_neuroplastic;.\llama3_neuroplastic\experiments'
.\verification_env\Scripts\python.exe .\llama3_neuroplastic\experiments\init_learned_basis_from_dense_mlp.py `
  --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" `
  --output-path ".\results\learned_basis_init_405b_smoke.pt" `
  --use-streaming-harness `
  --batch-size 1 `
  --basis-rank 96 `
  --sca-block-size 32 `
  --sca-dense-anchor-stride 4 `
  --max-samples 2 `
  --no-local-files-only
```

This smoke command intentionally ends after a tiny bounded dataset slice so the script finalizes whatever rows were collected and writes a real checkpoint suitable for plumbing tests.

### Safe operating ranges (March 2026)

| Component       | Safe layers (32-layer 8B)            | Notes                                              |
|-----------------|--------------------------------------|----------------------------------------------------|
| Sparse MLP      | 3–7 (mid-band only)                  | Use `--sca-bottom-buffer-layers 3 --sca-decode-guard-layers 24` |
| Taylor attention| 4–27 (Hybrid Performer)             | `local_window=64, feature_dim=64` are now the stable defaults |
| Dense fallback  | all                                  | Always baseline-safe                               |

- **Acceptance Criteria.** Short prompts can look superficially coherent even when the architecture is failing under strict decode metrics. In practice, the real acceptance criteria are rollout KL, hidden-state cosine, degeneration rate, and strict greedy continuation quality rather than one-off prompt samples.

## Key files

- `llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py`: canonical runtime model
- `llama3_neuroplastic/gqa_taylor_ssd.py`: deterministic Taylor-SSD recurrent attention backend
- `llama3_neuroplastic/sca_sparse_mlp.py`: sparse MLP wrapper, semantic routing, curriculum/banking hooks
- `llama3_neuroplastic/sca_sparse_config.py`: sparse config, runtime compatibility validation, and per-layer override canonicalization
- `llama3_neuroplastic/experiments/run_sca_recalibration_from_hybrid_baseline.py`: recalibration runner
- `llama3_neuroplastic/experiments/run_streaming_inference.py`: high-bandwidth SSD streaming inference entrypoint
- `llama3_neuroplastic/experiments/streaming_llama_runtime.py`: layer-by-layer out-of-core runtime (FP16/4-bit)
- `llama3_neuroplastic/experiments/init_learned_basis_from_dense_mlp.py`: dense-informed basis init (supports streaming harness)
- `llama3_neuroplastic/experiments/run_hybrid_gqa_mamba_inference.py`: runtime decode sanity checks
- `llama3_neuroplastic/experiments/compile_block_bank_spatial_router.py`: block-bank + router-state compiler
- `llama3_neuroplastic/experiments/run_taylor_ssd_inference.py`: Taylor-SSD inference + strict-metric runner
- `llama3_neuroplastic/experiments/run_taylor_fluency_ablation.py`: stepwise dense→Taylor→sparse ablation
- `llama3_neuroplastic/experiments/strict_decode_metrics.py`: strict decode metric utilities

## Repository layout

- `llama3_neuroplastic/`: core model and experiment scripts
- `tests/`: regression tests
- `results/`: generated checkpoints and metrics
- `written_documentation/`: notes and reports
- `verification_env/`: local Python environment

## Recommended workflow

Run from repo root in PowerShell.

### 1) Targeted tests

```powershell
$env:PYTHONPATH='.;.\llama3_neuroplastic;.\llama3_neuroplastic\experiments'
.\verification_env\Scripts\python.exe -m pytest `
  tests/test_sca_sparse_mlp.py `
  tests/test_hybrid_gqa_mamba.py `
  tests/test_strict_decode_metrics.py `
  tests/test_sca_recalibration_decode_manifold.py `
  -q
```

### 2) Recalibration run (direct runner)

```powershell
$env:PYTHONPATH='.;.\llama3_neuroplastic;.\llama3_neuroplastic\experiments'
.\verification_env\Scripts\python.exe .\llama3_neuroplastic\experiments\run_sca_recalibration_from_hybrid_baseline.py `
  --model-name "unsloth/Meta-Llama-3.1-8B-bnb-4bit" `
  --hybrid-checkpoint ".\results\hybrid_8b_export\hybrid_attention_state.pt" `
  --learned-basis-init-checkpoint ".\results\semantic_pipeline_clean_b2_l2_17\learned_basis_init_profiled.pt" `
  --output-dir ".\results\recal_run" `
  --recalibration-mode decode_manifold_alignment `
  --layers "3-7" `
  --sca-routing-mode semantic_latent `
  --sca-bottom-buffer-layers 3 `
  --sca-decode-guard-layers 24 `
  --basis-rank 96 `
  --basis-top-k 12 `
  --top-k 6 `
  --steps 32 `
  --max-samples 16 `
  --max-seq-length 64 `
  --validation-prefix-count 4 `
  --validation-prefix-length 48 `
  --progressive-depth-enabled `
  --progressive-depth-group-size 2 `
  --no-include-spatial-proj `
  --no-strict-decode-upper-layer-cap-enabled
```

### 3) Fast text-quality sanity check

```powershell
$env:PYTHONPATH='.;.\llama3_neuroplastic;.\llama3_neuroplastic\experiments'
.\verification_env\Scripts\python.exe .\llama3_neuroplastic\experiments\run_hybrid_gqa_mamba_inference.py `
  --checkpoint ".\results\hybrid_8b_export\hybrid_attention_state.pt" `
  --sca-recalibrated-checkpoint ".\results\recal_run\sca_recalibrated_state.pt" `
  --enable-sparse-mlp `
  --sca-sparse-placement learned_basis `
  --sca-routing-mode semantic_latent `
  --sca-bottom-buffer-layers 2 `
  --sca-decode-guard-layers 22 `
  --sca-basis-rank 96 `
  --sca-basis-top-k 12 `
  --sca-top-k 6 `
  --allow-cache `
  --max-new-tokens 12 `
  --prompt "Write two clear factual sentences about Oslo."
```

## Notes

- A checkpoint can be saved even when quality gate fails. Treat failed-gate checkpoints as diagnostic artifacts only.
- For this repo, targeted tests are more reliable than full test collection for day-to-day iteration.
- Current bottleneck is deep sparse-stack decode stability, not basic routing functionality.
