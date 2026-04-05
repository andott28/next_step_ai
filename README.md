# next_step_ai

This repository contains the current Neuroplastic Llama experimentation stack focused on training-faithful learned-basis MLP execution, layer-streamed 405B inference, sparse attention/KV transfer, Taylor-SSD integration, and strict autoregressive decode regression testing.

The canonical model class is `llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py`.

For the 405B streamed runtime, the source of truth is split intentionally:

- CLI and wiring: `llama3_neuroplastic/experiments/run_streaming_inference.py`
- Runtime implementation: `llama3_neuroplastic/experiments/streaming_llama_runtime.py`
- Learned-basis artifact contract: `llama3_neuroplastic/basis_fitting.py`
- Learned-basis semantic execution: `llama3_neuroplastic/sca_sparse_mlp.py`
- Sparse config defaults and per-layer basis-top-k semantics: `llama3_neuroplastic/sca_sparse_config.py`

## What this project is building

This repo does not train from scratch. It starts from dense Llama-family checkpoints and replaces parts of runtime and calibration with sparse and hybrid components while trying to preserve normal greedy decode quality.

Current target:

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

## Current status

- The 405B streamed runtime is correctness-first and stable on the current sparse MLP path.
- The learned-basis MLP execution now matches the training-side semantics in the critical latent and bias handling paths.
- Sparse-attention decode is stable again, but the runtime is still far below the `2.8 tok/s` throughput target because attention remains compute-dense and uncovered MLP layers still use exact dense guards.

## 405B Runtime Architecture

The project supports running **Llama 3.1 405B Instruct** on an 8 GB card using a layer-streamed out-of-core runtime plus sparse sidecar artifacts.

### Active artifacts

- `results/mlp_basis.pt`
  - learned-basis MLP checkpoint
  - `110/126` covered layers
  - layers `2..111`
  - output-space blocks over the `16384`-dim MLP output
  - `block_size=32`, `num_blocks=512`, `basis_rank=64`
- `results/attn_head_importance_405b.pt`
  - full `126/126` layer coverage
  - head-importance metadata for sparse q/o transfer
- `results/kv_basis_r32.pt`
  - sparse K/V column-block routing artifact
  - `block_size=32`, `basis_rank=32`, `top_k=51`

### How the runtime currently works

- **Layer streaming:** only one decoder layer skeleton is materialized on GPU at a time. Large base weights are RAM-cached and streamed into that skeleton.
- **Embed/lm_head placement:** `embed_tokens` and `lm_head` stay off the 8 GB hot path. The runtime favors GPU headroom for layer execution over keeping those giant tensors resident.
- **Covered MLP layers:** layers present in `results/mlp_basis.pt` run through the training-faithful learned-basis path in `streaming_llama_runtime.py`.
  - Shared latent path: `encoder linear -> SiLU -> latent top-k mask`
  - Latent support top-k is separate from block-routing top-k and follows checkpoint config or the training default from `sca_sparse_config.py`
  - Default executor is `full_output_latent`, meaning the runtime reconstructs the full MLP output from only the active latent coordinates
  - The old routed-block executor is still available only as a debug switch via `STREAMING_SPARSE_BASIS_EXECUTION=routed_blocks`
  - Decoder bias is not broadcast to inactive blocks
- **Uncovered MLP layers:** layers `0`, `1`, and `112..125` use an exact chunked 4-bit dense guard path on GPU.
  - This is correctness-first
  - It is not a zero-pass-through approximation
  - It is also the single most expensive remaining MLP path
- **Decode attention:** sparse attention currently reduces transfer, not dense attention math.
  - q/o weights are loaded only for selected heads using `results/attn_head_importance_405b.pt`
  - those partial weights are scattered into dense q/o skeleton tensors
  - the normal dense attention module then runs on that skeleton
  - on the tested 8 GB RTX 2080 path, `--attn-active-heads 64 --attn-max-active-heads 64` typically produces a 64-head candidate pool but only `16/128` live decode heads
- **Decode KV:** K/V uses routed column-block transfer from `results/kv_basis_r32.pt`
  - active blocks are predicted from the post-layernorm hidden states
  - active K/V columns are loaded and scattered into dense K/V skeleton tensors
  - the downstream attention compute remains dense
- **Prefill policy:** prefill defaults to dense q/o and dense K/V even when sparse artifacts are present.
  - this is deliberate
  - sparse prefill was the source of the old `AppBundle` corruption
  - the current runtime keeps prefill correctness above first-token speed
- **Taylor policy:** Taylor is auto-disabled when streamed sparse attention is active unless `STREAMING_ALLOW_TAYLOR_WITH_SPARSE_ATTN=1` is set.

## 405B Streaming Status

- The correctness regression that produced `AppBundle` is fixed.
- The runtime now logs `non-selected layers use exact streamed dense guard execution`, which reflects the real behavior.
- The sparse MLP side is now much closer to the intended architecture than the attention side.
- The runtime is still **well below** the historical `2.8 tok/s` target on the test machine.

### Why throughput is still low

- Sparse MLP execution is now computationally sparse for the `110` covered layers, but attention is still mostly transfer-sparse and compute-dense.
- The `16` uncovered MLP layers still pay the exact chunked dense guard cost every decode token.
- Prefill intentionally stays dense for q/o and K/V, so first-token latency remains expensive.
- The VRAM hot-cache can auto-clamp off when free VRAM falls below the configured safety margin, which pushes more traffic back onto the RAM->GPU path.

Bounded current measurements on the test machine:

- `2` runtime layers and `1` generated token: `18.0s`, `overall=1762.06 MB/layer`
- `2` runtime layers and `2` generated tokens: `46.5s`, `decode=1510.06 MB/layer`, `overall=1636.06 MB/layer`

Those bounded probes hit the early dense-guard layers, so they are not representative of the sparse-covered middle band. They are still useful because they show where the current byte budget is being burned.

## 405B Commands

The sections above describe the architecture and current performance envelope. The commands below are the current repo-supported entrypoints for that runtime.

### 405B Inference Command (interactive prompt REPL)

Omitting `--prompt` starts a persistent plain-prompt REPL. The runtime stays alive between queries, decoded tokens stream to stdout, and the prompt cache is reused by longest-common-prefix matching. Use `/reset` to clear the cached prefix state.

```powershell
.\verification_env\Scripts\python.exe -u -m llama3_neuroplastic.experiments.run_streaming_inference `
  --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" `
  --local-files-only `
  --sparse-basis-path results/mlp_basis.pt `
  --sparse-top-k 51 `
  --attn-head-importance-path results/attn_head_importance_405b.pt `
  --attn-active-heads 64 `
  --attn-max-active-heads 64 `
  --kv-basis-path results/kv_basis_r32.pt `
  --kv-sparse-top-k 51 `
  --vram-hot-cache-gb 2.0 `
  --max-new-tokens 64
```

Pass `--prompt "..."` to run a single query and exit.

### 405B Inference Command (scripted single-shot)

```powershell
.\verification_env\Scripts\python.exe -u -m llama3_neuroplastic.experiments.run_streaming_inference `
  --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" `
  --local-files-only `
  --sparse-basis-path results/mlp_basis.pt `
  --sparse-top-k 51 `
  --attn-head-importance-path results/attn_head_importance_405b.pt `
  --attn-active-heads 64 `
  --attn-max-active-heads 64 `
  --kv-basis-path results/kv_basis_r32.pt `
  --kv-sparse-top-k 51 `
  --vram-hot-cache-gb 2.0 `
  --max-new-tokens 32 `
  --prompt "The capital of France is"
```

Practical notes:

- `run_streaming_inference.py` passes `--prompt` straight to the tokenizer. It does not apply a chat template for instruct checkpoints.
- `--attn-active-heads 64 --attn-max-active-heads 64` means a 64-head candidate pool, not guaranteed 64 live decode heads.
- On the tested 8 GB path, the healthy sparse-attention decode log shape is usually `active_heads=16/128 ... min=16 max=64`.

### 405B Throughput / Correctness Micro-Probe

Use the exact production command surface plus `--max-runtime-layers` for a cheap probe before paying for a full `126`-layer run.

```powershell
.\verification_env\Scripts\python.exe -u -m llama3_neuroplastic.experiments.run_streaming_inference `
  --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" `
  --local-files-only `
  --sparse-basis-path results/mlp_basis.pt `
  --sparse-top-k 51 `
  --attn-head-importance-path results/attn_head_importance_405b.pt `
  --attn-active-heads 64 `
  --attn-max-active-heads 64 `
  --kv-basis-path results/kv_basis_r32.pt `
  --kv-sparse-top-k 51 `
  --vram-hot-cache-gb 2.0 `
  --max-runtime-layers 4 `
  --max-new-tokens 2 `
  --no-stream-output `
  --prompt "The capital of France is"
```

Healthy signals for this micro-probe:

- no `AppBundle`
- no `zero-pass-through` wording in the sparse MLP log
- no Taylor non-finite decode crash
- a traffic report is emitted at the end

Do not expect `2.8 tok/s` from the current runtime architecture. The micro-probe is for regression detection and hot-path attribution, not for proving the final throughput target.

### Rebuilding the attention importance checkpoint

The attention checkpoint already exists in `results/attn_head_importance_405b.pt`. Re-run this only if you want to regenerate it.

```powershell
.\verification_env\Scripts\python.exe -u -m llama3_neuroplastic.experiments.init_learned_attn_head_importance `
  --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" `
  --output-path ".\results\attn_head_importance_405b.pt" `
  --taylor-layers "0-125" `
  --taylor-feature-map hybrid_performer `
  --max-samples 64 `
  --max-seq-length 128 `
  --max-tokens-per-layer 2048
```

Practical note:

- Speed per token is now dominated by per-layer byte traffic and dense attention compute, not by whether the sparse MLP path itself is active.
- On 32 GB RAM the weight cache budget is auto-sized from currently available RAM at startup.
- For runtime verification, prefer short streamed checks that use `--max-runtime-layers` and inspect the emitted traffic report before paying for a full benchmark.

### Current limits

1. The sparse learned-basis MLP path is now the corrected reference implementation, but the exact dense guard path for uncovered layers is still expensive.
2. Sparse attention and sparse KV currently save transfer, not dense attention compute.
3. If more latency work is needed, the next target is the attention/runtime hot path, not the learned-basis MLP semantics.

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
- because one token contributes roughly one row to every still-pending selected layer, the runtime cost is mostly determined by total valid tokens processed, not by `selected_layers x tokens` separate forward passes
- lowering `--max-rows-per-layer` is the main lever for faster smoke runs

Fast smoke-run example that produces a real output checkpoint rather than only a resume file:

```powershell
.\verification_env\Scripts\python.exe -u -m llama3_neuroplastic.experiments.init_learned_basis_from_dense_mlp `
  --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" `
  --output-path ".\results\mlp_basis_smoke.pt" `
  --use-streaming-harness `
  --batch-size 1 `
  --basis-rank 64 `
  --sca-block-size 32 `
  --sca-dense-anchor-stride 4 `
  --max-samples 2 `
  --no-local-files-only
```

This smoke command intentionally ends after a tiny bounded dataset slice so the script finalizes whatever rows were collected and writes a real checkpoint suitable for plumbing tests.

**Acceptance Criteria.** Short prompts can look superficially coherent even when the architecture is failing under strict decode metrics. In practice, the real acceptance criteria are rollout KL, hidden-state cosine, degeneration rate, and strict greedy continuation quality rather than one-off prompt samples.

## Key files

- `llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py`: canonical modified model class used by basis-initialization and broader project logic
- `llama3_neuroplastic/gqa_taylor_ssd.py`: deterministic Taylor-SSD recurrent attention backend
- `llama3_neuroplastic/sca_sparse_mlp.py`: sparse MLP wrapper, semantic routing, curriculum/banking hooks, and learned-basis execution logic
- `llama3_neuroplastic/sca_sparse_config.py`: sparse config, runtime compatibility validation, and per-layer override canonicalization
- `llama3_neuroplastic/triton_sparse_mlp.py`: Triton sparse-weight kernels used by legacy and debug sparse-weight paths; not the default executor for `results/mlp_basis.pt`
- `llama3_neuroplastic/basis_fitting.py`: closed-form learned-basis fitting utilities
- `llama3_neuroplastic/experiments/run_streaming_inference.py`: high-bandwidth SSD streaming inference entrypoint
- `llama3_neuroplastic/experiments/streaming_llama_runtime.py`: layer-by-layer out-of-core runtime (FP16/4-bit)
- `llama3_neuroplastic/experiments/init_learned_basis_from_dense_mlp.py`: dense-informed basis init and resume workflow for learned-basis checkpoints
- `tests/test_streaming_llama_runtime.py`: focused regression coverage for the streaming runtime

## Repository layout

- `llama3_neuroplastic/`: core model code and the remaining experiment entrypoints
- `tests/`: focused runtime regression tests
- `results/`: generated checkpoints needed for sparse 405B inference plus any kept diagnostics
- `written_documentation/`: notes, reports, and repo mapping documents
- `verification_env/`: local Python environment

## Recommended workflow

Run from repo root in PowerShell.

### 1) Runtime regression test

```powershell
.\verification_env\Scripts\python.exe -m pytest `
  tests/test_streaming_llama_runtime.py `
  -q
```

### 2) Learned-basis init / resume

```powershell
.\verification_env\Scripts\python.exe -u -m llama3_neuroplastic.experiments.init_learned_basis_from_dense_mlp `
  --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" `
  --output-path ".\results\mlp_basis.pt" `
  --use-streaming-harness `
  --basis-rank 64 `
  --sca-block-size 32 `
  --resume-save-every-batches 1 `
  --write-partial-output-every-batches 1
```

### 3) Streaming inference run

```powershell
.\verification_env\Scripts\python.exe -u -m llama3_neuroplastic.experiments.run_streaming_inference `
  --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" `
  --local-files-only `
  --prompt "The capital of France is" `
  --max-new-tokens 32 `
  --sparse-basis-path ".\results\mlp_basis.pt" `
  --sparse-top-k 51 `
  --attn-head-importance-path ".\results\attn_head_importance_405b.pt" `
  --attn-active-heads 64 `
  --attn-max-active-heads 64 `
  --kv-basis-path ".\results\kv_basis_r32.pt" `
  --kv-sparse-top-k 51 `
  --vram-hot-cache-gb 2.0
```

## Notes

- This repo is now inference-first. The critical reproducible path is: build or resume the learned-basis checkpoint, then run the streaming inference entrypoint with the 405B sparse artifacts.
- The thin wrapper modules under `llama3_neuroplastic/` are intentionally not required. Use the direct `experiments/` paths shown above.
- Results pruning is acceptable as long as the runtime-critical checkpoints remain present, especially `results/mlp_basis.pt`, `results/attn_head_importance_405b.pt`, and `results/kv_basis_r32.pt`.
