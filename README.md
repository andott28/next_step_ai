# next_step_ai

This repository contains the current Neuroplastic Llama experimentation stack, with the active work centered on sparse MLP routing, hybrid GQA/Mamba attention, and strict autoregressive decode stability.

The canonical production model for current work is [`llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py`](c:/Users/andre/Desktop/Overføre/next_step_ai/llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py). The current sparse MLP path of interest is `sparse_placement="learned_basis"` with `routing_mode="semantic_latent"`.

## The model being built

This repo is not training a language model from scratch. It is building a modified Llama-family causal decoder that starts from a dense base model and then swaps in sparse or hybrid components while trying to keep normal greedy next-token decoding intact.

In the current exercised setup, the base model is usually `unsloth/Meta-Llama-3.1-8B-bnb-4bit`, loaded through Hugging Face as a 4-bit quantized `LlamaForCausalLM`. The dense teacher for all sparse work is the same model with neuroplastic routing disabled, not a separate checkpoint.

The immediate proving ground is the 8B model, but the actual project target is much more aggressive: push the architecture and runtime toward something 405B-class in capability density while remaining runnable on an 8 GB VRAM GPU through a combination of extreme sparsity, selective dense fallback elimination, hybrid attention/state-space replacements, quantization, and CPU/off-GPU memory streaming. The 8B model is therefore the experimental platform, not the final ambition.

Architecturally, the target model has four main pieces:

- A frozen Llama decoder backbone that still provides the token embeddings, residual stream, layer norms, and LM head.
- Optional hybrid attention layers where selected self-attention modules are replaced with a GQA/Mamba-style collapsed state-space block mixed with the original attention path.
- Sparse MLP wrappers around the transformer MLPs, with the current canonical path using a learned semantic latent basis rather than masking raw hidden units directly.
- Strict runtime controls for long-context and sparse decode, including bounded context, paged sparse attention, and decode-time guards intended to preserve stable autoregressive behavior.

For the current sparse MLP design, each wrapped MLP is conceptually:

- encoder from residual stream to latent coordinates
- SiLU activation in latent space
- top-k latent support selection in that semantic basis
- block scoring from latent magnitude times decoder block norms
- sparse reconstruction of the MLP update back into residual-stream coordinates

The critical point is that sparsity is supposed to live in the learned latent basis, not in the raw hidden basis. The goal is to make the sparse MLP produce updates that still lie close to the dense model's decode manifold, so downstream layers and the LM head continue to decode coherent text without decode-time hacks.

By default, the model now keeps the earliest `sca_bottom_buffer_layers=2` MLP layers dense and also keeps a top decode guard band dense. That means the intended sparse regime is the middle band of layers, where the representation is semantic enough for learned-basis routing to be viable but not so late that small errors immediately corrupt token selection.

## How this differs from the original Unsloth model

This codebase is heavily modified relative to the original `unsloth/Meta-Llama-3.1-8B-bnb-4bit` runtime. It should not be thought of as "just Unsloth plus a small plugin." The base weights still come from that model family, but large parts of the execution path, diagnostics, and calibration machinery have been replaced or wrapped.

The most important differences are:

- Transformer MLPs are wrapped by [`llama3_neuroplastic/sca_sparse_mlp.py`](c:/Users/andre/Desktop/Overføre/next_step_ai/llama3_neuroplastic/sca_sparse_mlp.py), which can replace the dense MLP update with sparse compute paths, including the current semantic-latent learned-basis route.
- Selected attention layers can be replaced with hybrid GQA/Mamba modules or fully collapsed Mamba-style attention surrogates inside [`llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py`](c:/Users/andre/Desktop/Overføre/next_step_ai/llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py).
- The runtime adds bounded-context and paged sparse-attention logic for long-context operation and strict sparse decode, rather than relying on the original dense attention/cache behavior.
- KV cache handling is modified to support int4 quantization and CPU offload paths, which are part of the memory budget strategy for low-VRAM hardware.
- The model includes new trainable control components such as `task_embedding`, `sca_layer_output_scale`, sparse routing state, and optional decoder-mirror modules that do not exist in the original base model.
- The forward path collects extensive sparse-routing and alignment telemetry, including latent support usage, block load, fallback rate, and hidden/logit alignment data, because this repo optimizes for decode stability under sparse execution rather than only ordinary inference throughput.
- The repo adds explicit recalibration and checkpoint formats for hybrid attention state, learned-basis initialization, sparse recalibration state, and strict diagnostic artifacts.

At a practical level, this means the current model should be understood as:

- original Llama weights and tokenizer as the base
- heavily modified attention runtime
- heavily modified MLP runtime
- additional calibration and sparse-control modules
- additional offload, cache, and strict-decode logic

That is why results from the original Unsloth checkpoint do not directly answer whether this modified model works: the inference graph has changed substantially.

## Current focus

The main engineering target in this repo is:

- semantic-latent sparse MLP routing
- dense-informed learned-basis initialization
- decode-manifold recalibration
- strict decode diagnostics and gating

Current guardrail defaults for the new path:

- `sca_sparse_placement=learned_basis`
- `sca_routing_mode=semantic_latent`
- `strict_decode_enable_repetition_penalty=False`
- `strict_decode_upper_layer_cap_enabled=False`
- `sca_bottom_buffer_layers=2`

The bottom-buffer guard exists because the earliest layers operate close to embedding space and can destabilize sparse routing if they are forced sparse too early.

## Design target

The model being built here is a strict-decode-compatible sparse Llama variant:

- dense enough at the bottom and decode-critical top to avoid catastrophic drift
- sparse through the middle MLP stack using semantic-latent routing
- optionally hybridized in attention through GQA/Mamba replacements
- evaluated by actual greedy autoregressive continuation quality, not only reconstruction loss

The immediate success condition is not "maximum sparsity." It is coherent text generation under shipped runtime settings with zero dense fallback and without repetition-penalty tricks compensating for model drift.

Longer-term, the design target is a model/runtime stack that scales the same ideas far beyond 8B and makes a 405B-class deployment story plausible on commodity 8 GB VRAM by aggressively reducing active compute and active memory per token. That target is not solved in this repository yet; the current work is the engineering path toward it.

## Repository layout

- [`llama3_neuroplastic/`](c:/Users/andre/Desktop/Overføre/next_step_ai/llama3_neuroplastic) - core model code, sparse MLP wrappers, diagnostics, and runners
- [`tests/`](c:/Users/andre/Desktop/Overføre/next_step_ai/tests) - targeted regression tests for sparse routing, recalibration, and decode metrics
- [`run_semantic_sparse_pipeline.ps1`](c:/Users/andre/Desktop/Overføre/next_step_ai/run_semantic_sparse_pipeline.ps1) - PowerShell wrapper that runs profile -> learned-basis init -> recalibration -> strict diagnostic
- [`written_documentation/`](c:/Users/andre/Desktop/Overføre/next_step_ai/written_documentation) - research notes, execution reports, and root-cause writeups
- [`verification_env/`](c:/Users/andre/Desktop/Overføre/next_step_ai/verification_env) - local Python environment used for the current workflow in this repo

## Canonical scripts

- [`llama3_neuroplastic/run_sca_layer_decode_profile.py`](c:/Users/andre/Desktop/Overføre/next_step_ai/llama3_neuroplastic/run_sca_layer_decode_profile.py)
  Profiles one sparse layer at a time under strict decode and writes per-layer impact scores plus recommended budget overrides.
- [`llama3_neuroplastic/init_learned_basis_from_dense_mlp.py`](c:/Users/andre/Desktop/Overføre/next_step_ai/llama3_neuroplastic/init_learned_basis_from_dense_mlp.py)
  Builds dense-informed learned-basis initialization from teacher-forced and dense greedy-rollout prefixes.
- [`llama3_neuroplastic/run_sca_recalibration_from_hybrid_baseline.py`](c:/Users/andre/Desktop/Overføre/next_step_ai/llama3_neuroplastic/run_sca_recalibration_from_hybrid_baseline.py)
  Trains sparse MLP recalibration, including `recalibration_mode="decode_manifold_alignment"`.
- [`llama3_neuroplastic/run_sca_diagnostic_wipe.py`](c:/Users/andre/Desktop/Overføre/next_step_ai/llama3_neuroplastic/run_sca_diagnostic_wipe.py)
  Runs strict decode evaluation and emits a consolidated JSON artifact.
- [`llama3_neuroplastic/run_hybrid_gqa_mamba_inference.py`](c:/Users/andre/Desktop/Overføre/next_step_ai/llama3_neuroplastic/run_hybrid_gqa_mamba_inference.py)
  Fast runtime sanity-check script for dense vs sparse text quality and runtime flags.

## Recommended workflow

Run everything from the repo root in PowerShell.

### 1. Targeted tests

```powershell
.\verification_env\Scripts\python.exe -m pytest `
  tests/test_sca_sparse_mlp.py `
  tests/test_hybrid_gqa_mamba.py `
  tests/test_strict_decode_metrics.py `
  tests/test_sca_recalibration_decode_manifold.py `
  -q
```

### 2. End-to-end semantic sparse pipeline

This is the recommended entrypoint when you want the full profile -> init -> recalibration -> diagnostic flow without running unrelated tasks:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_semantic_sparse_pipeline.ps1 `
  -ModelName "unsloth/Meta-Llama-3.1-8B-bnb-4bit" `
  -HybridCheckpoint ".\results\hybrid_8b_export\hybrid_attention_state.pt" `
  -OutputRoot ".\results\semantic_pipeline_run" `
  -LayerSelection "2-17" `
  -LimitBasisInitToSelectedLayers `
  -ScaBottomBufferLayers 2 `
  -ValidationPrefixCount 32 `
  -MaxNewTokens 6 `
  -MaxSamples 64 `
  -MaxSeqLength 96 `
  -Steps 24 `
  -SkipTests
```

### 3. Very fast text-quality sanity check

Use this after a recalibration run to see whether strict sparse decode still collapses into gibberish:

```powershell
.\verification_env\Scripts\python.exe .\llama3_neuroplastic\run_hybrid_gqa_mamba_inference.py `
  --checkpoint ".\results\hybrid_8b_export\hybrid_attention_state.pt" `
  --sca-recalibrated-checkpoint ".\results\semantic_pipeline_run\recal_decode_manifold\sca_recalibrated_state.pt" `
  --enable-sparse-mlp `
  --sca-sparse-placement learned_basis `
  --sca-routing-mode semantic_latent `
  --sca-bottom-buffer-layers 2 `
  --sca-basis-rank 96 `
  --sca-basis-top-k 12 `
  --sca-top-k 3 `
  --allow-cache `
  --max-new-tokens 12 `
  --prompt "Write two clear factual sentences about Oslo."
```

## Important code paths

- [`llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py`](c:/Users/andre/Desktop/Overføre/next_step_ai/llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py)
  Canonical runtime model.
- [`llama3_neuroplastic/sca_sparse_mlp.py`](c:/Users/andre/Desktop/Overføre/next_step_ai/llama3_neuroplastic/sca_sparse_mlp.py)
  Sparse MLP wrapper and semantic-latent routing logic.
- [`llama3_neuroplastic/sca_sparse_config.py`](c:/Users/andre/Desktop/Overføre/next_step_ai/llama3_neuroplastic/sca_sparse_config.py)
  Sparse routing config, per-layer override maps, and helper methods.
- [`llama3_neuroplastic/strict_decode_metrics.py`](c:/Users/andre/Desktop/Overføre/next_step_ai/llama3_neuroplastic/strict_decode_metrics.py)
  Reusable decode metrics and balance summaries.

## Legacy paths

These remain in the tree but are not the primary implementation target for the semantic-latent sparse work:

- [`llama3_neuroplastic/neuroplastic_llama.py`](c:/Users/andre/Desktop/Overføre/next_step_ai/llama3_neuroplastic/neuroplastic_llama.py)
- [`llama3_neuroplastic/train_llama_sca_objective.py`](c:/Users/andre/Desktop/Overføre/next_step_ai/llama3_neuroplastic/train_llama_sca_objective.py)
- decoder-mirror paths, except for explicit ablation use

## Environment notes

- The working flow in this repo uses repo-local scripts directly, not a polished package install path.
- [`setup.py`](c:/Users/andre/Desktop/Overføre/next_step_ai/setup.py) is partly legacy and should not be treated as the primary way to run the project.
- The repo expects local code layout rather than downloading `neuroplastic_lib` from elsewhere. If `neuroplastic_lib` exists in or near the repo, the PowerShell pipeline attempts to add it to `PYTHONPATH`.
- Windows + PowerShell is the currently exercised path.

## Known caveats

- A recalibration checkpoint that fails the strict quality gate can still be saved for inspection. Do not treat such a checkpoint as decode-stable.
- Full `pytest` collection may still hit legacy tests that depend on modules outside the currently validated path. Use the targeted test set above unless you are intentionally fixing those legacy paths.
- Sparse text quality should be judged under strict runtime settings, not only teacher-forced reconstruction metrics.
