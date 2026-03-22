# next_step_ai

This repository contains the current Neuroplastic Llama experimentation stack focused on sparse MLP routing, hybrid GQA/Mamba attention, and strict autoregressive decode stability.

The canonical model class is `llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py`.
The canonical sparse path is:

- `sparse_placement="learned_basis"`
- `routing_mode="semantic_latent"`

## What this project is building

This repo does not train from scratch. It starts from a dense Llama-family checkpoint (usually `unsloth/Meta-Llama-3.1-8B-bnb-4bit`) and replaces parts of runtime/training with sparse and hybrid components while trying to preserve normal greedy decode quality.

Current long-term target:

- Use the 8B model as the proving platform.
- Push architecture/runtime ideas toward a 405B-class deployment strategy on 8 GB VRAM using sparsity, quantization, and off-GPU streaming.

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

What is still failing:

- Deep continuous sparse stacking still degrades beyond a depth boundary.
- Typical failure boundary is around guards equivalent to sparse layers reaching into the 2-11 range.
- Strict quality gate is still failing for deep bands, even when short prompts may look coherent.

Practical implication:

- Mid-band sparse operation is usable.
- Full deep sparse stacking is not yet production-stable under strict decode metrics.

## Current architectural problems

- The current attention-collapse backend is heuristic, not a direct mathematical compile of Llama attention. The existing GQA/Mamba path is based on grouped SVD collapse plus a small recurrent block, so it is lighter than dense attention but not a faithful attention-to-SSM translation.
- The current `learned_basis` sparse MLP is a different operator class than the dense SwiGLU MLP it is trying to replace. Even with deterministic basis initialization, it still relies on latent support selection, block selection, and sparse reconstruction rather than preserving the original dense block weights.
- The current semantic-latent router is still only an approximation to dense block usage. It routes from latent magnitudes and decoder statistics rather than from a compiled per-sample dense block-support target, which is one reason the project has needed heavy recalibration.
- Deep stacked sparse MLP execution accumulates error across layers. A layer can look acceptable in isolation but still push the residual stream off-manifold when many sparse layers are active at once, which is the main source of repetition collapse and junk decode in current experiments.
- The runtime stack is still strongly coupled to standard KV-cache attention. Cache packing, KV quantization, CPU offload, bounded context, and paged sparse-attention paths all assume attention layers expose conventional key/value cache structure, which makes alternative recurrent attention backends a larger integration task.
- Short prompts can look superficially coherent even when the architecture is failing under strict decode metrics. In practice, the real acceptance criteria are rollout KL, hidden-state cosine, degeneration rate, and strict greedy continuation quality rather than one-off prompt samples.

## Key files

- `llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py`: canonical runtime model
- `llama3_neuroplastic/sca_sparse_mlp.py`: sparse MLP wrapper, semantic routing, curriculum/banking hooks
- `llama3_neuroplastic/sca_sparse_config.py`: sparse config, runtime compatibility validation, and per-layer override canonicalization
- `llama3_neuroplastic/experiments/run_sca_recalibration_from_hybrid_baseline.py`: recalibration runner
- `llama3_neuroplastic/experiments/init_learned_basis_from_dense_mlp.py`: dense-informed basis init
- `llama3_neuroplastic/experiments/run_hybrid_gqa_mamba_inference.py`: runtime decode sanity checks
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
  --layers "2-9" `
  --sca-routing-mode semantic_latent `
  --sca-bottom-buffer-layers 2 `
  --sca-decode-guard-layers 22 `
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
