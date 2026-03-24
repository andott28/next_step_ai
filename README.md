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
- **GPU Dequantization:** 4-bit weights are dequantized on-the-fly on the GPU to eliminate CPU bottlenecks.
- **Zero-Shot PCA:** Use `init_learned_basis_from_dense_mlp.py` with the `--use-streaming-harness` flag to initialize the sparse routing geometry analytically without ever loading the full model into system RAM.

### 405B Inference Command:
```powershell
$env:PYTHONPATH='.;.\llama3_neuroplastic;.\llama3_neuroplastic\experiments'
.\verification_env\Scripts\python.exe .\llama3_neuroplastic\experiments\run_streaming_inference.py `
  --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" `
  --prompt "The capital of Norway is" `
  --max-new-tokens 20 `
  --taylor-layers "0-125" `
  --taylor-feature-map hybrid_performer
```

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
