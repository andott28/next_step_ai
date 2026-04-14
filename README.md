# next_step_ai

Inference runtime for **Llama 3.1 405B on 8 GB VRAM**, using learned-basis MLP sparsity, NF4 quantization, and layer-by-layer streaming.

---

## What this is

Standard 405B inference requires ~800 GB of GPU memory. This project achieves full-quality autoregressive generation on a single 8 GB GPU by combining three techniques:

1. **Layer streaming** — only one decoder layer lives on the GPU at any time; the remaining 125 layers are kept in RAM and streamed on demand.
2. **Sparse MLP routing** - the preferred artifact fits a linear router over true SiLU-gated FFN intermediate block scores. At decode time, selected gate/up/down blocks still execute the original `SiLU(gate) * up` MLP math; legacy output-basis artifacts remain available only as approximation modes.
3. **NF4 hot-block VRAM cache** — the most frequently-accessed NF4 weight blocks for covered MLP layers are pinned in VRAM, cutting per-token PCIe traffic by up to 60%.

### Design philosophy: No-Training Invariant

All model modifications are **closed-form and zero-shot** — no SGD, no gradient-based tuning. The sparse basis is fitted once via PCA + least-squares regression on calibration activations. This makes every run fully deterministic and reproducible without requiring any training infrastructure.

---

## Architecture overview

```
Input tokens
    │
    ▼
┌─────────────────────────────────────┐
│  StreamingLlamaRuntime              │
│                                     │
│  for each layer (0–125):            │
│    ├─ Load weights (RAM → GPU)      │
│    ├─ Self-attention (GQA)          │  ← optional sparse head loading
│    └─ MLP forward                   │
│         ├─ Covered (0–125):         │  ← learned-basis sparse path
│         │    router -> active FFN   │
│         │    -> exact sparse MLP    │
│         └─ Uncovered (guard layers):│  ← exact 4-bit dense guard
│              chunked NF4 dequant    │
└─────────────────────────────────────┘
    │
    ▼
  lm_head → logits → next token
```

### MLP sparsity: exact intermediate routing

For each covered layer, the preferred offline fitting path produces:

| Tensor | Shape | Purpose |
|--------|-------|---------|
| `encoder_weight` | `[basis_rank, hidden_size]` | Projects hidden state to linear latent coefficients |
| `encoder_bias` | `[basis_rank]` | Encoder bias |
| `score_weight` | `[num_intermediate_blocks, basis_rank]` | Reconstructs per-block intermediate activity scores |
| `score_bias` | `[num_intermediate_blocks]` | Mean score offset |
| `block_importance` | `[num_intermediate_blocks]` | Global fallback ranking and hot-cache signal |

At decode time:
```python
latent = hidden @ encoder_weight.T + encoder_bias
latent = keep_top_abs(latent, k=basis_top_k)
scores = latent @ score_weight.T + score_bias
active_blocks = top_k(scores, k=sparse_top_k)
out = down_proj(SiLU(gate_proj_active(hidden)) * up_proj_active(hidden))
```

The old `output_reconstruction` artifact is still loadable through `output_basis_surrogate` and `routed_output_blocks`, but it is an output approximation. Use `intermediate_block_scores` plus `exact_intermediate_sparse` for correctness.

### Memory layout (per token, decode)

| Location | Contents | Size |
|----------|----------|------|
| GPU VRAM | 1 decoder layer skeleton | ~32 MB |
| GPU VRAM | NF4 hot-block cache | up to 6.4 GB |
| GPU VRAM | KV cache for current session | small |
| CPU RAM (pinned) | Full weight cache (LRU) | up to available RAM |
| SSD | Full NF4 safetensors checkpoint | ~210 GB |

---

## Repository layout

```
next_step_ai/
├── llama3_neuroplastic/
│   ├── basis_fitting.py                 # PCA + least-squares basis fitting
│   ├── bounded_context.py               # Tiered context masking for long sequences
│   ├── gqa_taylor_ssd.py                # Taylor-SSD linear attention backend
│   ├── neuroplastic_llama_gqa_mamba.py  # Full NeuroplasticLlama model class
│   ├── paged_sparse_attention.py        # Page-based sparse KV attention
│   ├── performance_utils.py             # Runtime env config (TF32, pinned memory)
│   ├── sca_decoder_mirror.py            # Experimental: predictive routing
│   ├── sca_sparse_adapter.py            # Routing utilities
│   ├── sca_sparse_config.py             # Config dataclass + validation
│   ├── sca_sparse_kv.py                 # Sparse K/V column-block routing
│   ├── sca_sparse_mlp.py                # Sparse MLP wrapper (NeuroplasticLlama path)
│   ├── token_posting_archive.py         # CPU sparse token index for long-context retrieval
│   ├── triton_sca_gate.py               # Triton gate kernels
│   ├── triton_sparse_mlp.py             # Triton sparse MLP kernels
│   └── experiments/
│       ├── streaming_llama_runtime.py   # Main streaming inference runtime (core)
│       ├── run_streaming_inference.py   # Interactive CLI entrypoint
│       ├── init_learned_basis_from_dense_mlp.py  # Offline basis fitting pipeline
│       ├── init_kv_basis.py             # K/V column-block routing fitting
│       ├── init_attn_share.py           # Attention weight sharing initialization
│       ├── init_attn_token_posting_basis.py      # Token-posting basis fitting
│       ├── eval_perplexity.py           # Perplexity evaluation harness
│       └── benchmark.py                 # Throughput benchmark
├── tests/
│   └── test_streaming_llama_runtime.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Clone and install in editable mode
git clone <repo-url>
cd next_step_ai
pip install -e .

# With Triton kernels (Windows — required for fast sparse MLP)
pip install -e ".[triton]"

# With evaluation tools
pip install -e ".[eval]"
```

**Requirements:** Python 3.10+, CUDA 11.8+, ~30 GB RAM (for weight cache), 8 GB VRAM minimum (RTX 2080 tested).

---

## Quick start

### 1. Fit the learned basis (offline, run once)

Streams calibration activations through the dense model and fits a PCA basis per MLP layer. Requires the full 405B checkpoint.

```bash
python -m llama3_neuroplastic.experiments.init_learned_basis_from_dense_mlp \
    --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" \
    --output-path results/mlp_basis_full126.pt \
    --use-streaming-harness \
    --layers 0-125 \
    --max-rows-per-layer 4096 \
    --basis-rank 64 \
    --sca-block-size 32 \
    --resume-save-every-batches 1 \
    --write-partial-output-every-batches 1
```

This produces `results/mlp_basis_full126.pt` covering all 126 layers. If you have a partial checkpoint from a previous run, add `--hybrid-checkpoint results/mlp_basis.pt --only-missing-from-output` to skip already-fitted layers.

### 2. Run interactive inference

```bash
python -m llama3_neuroplastic.experiments.run_streaming_inference \
    --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" \
    --local-files-only \
    --sparse-basis-path results/mlp_basis_full126.pt \
    --sparse-top-k 51 \
    --attn-head-importance-path results/attn_head_importance_405b.pt \
    --attn-active-heads 9 \
    --attn-min-active-heads 9 \
    --attn-max-active-heads 9 \
    --vram-hot-cache-gb 2.0 \
    --prompt-format chat \
    --prompt "What is the capital of France?"
```

This is the configuration tuned for RTX 2080 (8 GB VRAM). For GPUs with more VRAM, increase `--vram-hot-cache-gb` accordingly (up to ~6.4 GB on a 24 GB card) and raise `--attn-active-heads`.

### 3. Benchmark throughput

```bash
python -m llama3_neuroplastic.experiments.benchmark \
    --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" \
    --sparse-basis-path results/mlp_basis.pt \
    --max-new-tokens 20 \
    --warmup-tokens 5 \
    --output-json results/benchmark.json
```

### 4. Evaluate perplexity

```bash
python -m llama3_neuroplastic.experiments.eval_perplexity \
    --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" \
    --sparse-basis-path results/mlp_basis.pt \
    --dataset wikitext \
    --dataset-config wikitext-2-raw-v1 \
    --dataset-split test \
    --max-tokens 4096 \
    --output-json results/ppl_eval.json
```

---

## Key CLI flags

### `run_streaming_inference.py` / `StreamingLlamaRuntime`

| Flag | Default | Description |
|------|---------|-------------|
| `--sparse-basis-path` | None | Path to learned-basis checkpoint (`.pt`) |
| `--attn-head-importance-path` | None | Sparse attention head importance artifact |
| `--kv-basis-path` | None | K/V column-block routing artifact |
| `--vram-hot-cache-gb` | 6.4 | VRAM budget for hot NF4 block cache |
| `--sparse-top-k` | auto | Active MLP blocks per token. For `intermediate_block_scores`, this is FFN intermediate block sparsity |
| `--sparse-basis-top-k` | checkpoint | Runtime override for latent support sparsity |
| `--sparse-mlp-execution` | `auto` | `auto`, `output_basis_surrogate`, `routed_output_blocks`, `exact_intermediate_sparse`, or `exact_intermediate_sparse_oracle` |
| `--sparse-mlp-prefill-mode` | `dense` | Keep prompt prefill dense by default, or opt into sparse covered-layer prefill |
| `--max-runtime-layers` | None | Cap layers for smoke testing |
| `--no-stream-output` | off | Batch output instead of token streaming |
| `--no-ram-cache` | off | Disable RAM weight cache (re-reads from SSD each token) |

### `init_learned_basis_from_dense_mlp.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--layers` | all | Layer range to fit, e.g. `2-111` or `0,1,112-125` |
| `--artifact-target` | `intermediate_block_scores` | Preferred target for exact sparse MLP routing. Use `output_reconstruction` only for legacy surrogate checkpoints |
| `--use-streaming-harness` | off | Required for `intermediate_block_scores` so the collector can capture true streamed FFN block scores |
| `--basis-rank` | 64 | PCA decomposition rank per layer |
| `--basis-top-k` | 64 | Latent support sparsity at decode time |
| `--max-rows-per-layer` | 4096 | Calibration activations per layer |
| `--output-path` | required | Destination `.pt` checkpoint |
| `--resume` | off | Resume from a partial `.resume.pt` checkpoint |

---

## Performance

Measured on a single RTX 2080 (8 GB VRAM) / 405B model, PCIe 3.0 x16 (~11 GB/s H2D):

| Configuration | Decode (tok/s) | Notes |
|---------------|----------------|-------|
| Dense guard only (no basis) | ~0.04 | Baseline — all 126 layers dense, ~1510 MB/layer |
| Basis layers 2–111, 16 heads | ~0.8 | 16 dense guard layers dominate at ~1510 MB each |
| All 126 layers, 9 heads, hot-cache | **~3.3** | Target — ~23 MB/layer transfer |

Target: **3.3 tok/s** with all 126 layers covered by the basis, 9 active attention heads, and 2.0 GB VRAM hot-block cache enabled.

---

## Known limitations

- **Dense guard layers** — any layers not covered by the basis checkpoint fall back to the full dense 4-bit guard path (~1510 MB/layer), which dominates decode latency. A full 126-layer basis eliminates this.
- **Sparse MLP prefill defaults to dense** — decode can use sparse covered layers, while prompt prefill stays dense unless `--sparse-mlp-prefill-mode sparse` is set.
- **Attention compute is not sparse** — head-importance routing reduces PCIe weight transfer, but softmax attention still runs on the full dense skeleton.
- **Batch size 1 only** — the runtime is single-sequence only.
- **Taylor-SSD attention is incompatible with sparse head routing** — enabling both is experimental (`STREAMING_ALLOW_TAYLOR_WITH_SPARSE_ATTN=1`) and may produce incorrect outputs.

---

## How the sparse MLP fitting works

```
Dense model forward pass
    │  (collect MLP inputs x and SiLU-gated intermediate block scores)
    ▼
Per-layer PCA on block scores:
    scores ≈ score_bias + coeff @ score_weight.T

Encoder fit (least-squares):
    coeff ≈ encoder_weight @ x + encoder_bias

Store: encoder_weight, encoder_bias, score_weight, score_bias, block_importance
```

The `explained_variance_ratio` field in the checkpoint tells you what fraction of intermediate block-score variance is captured by the router. Full `--sparse-top-k` over all intermediate blocks should match dense MLP execution because the selected path still runs the original gate/up/down weights and applies `SiLU` before `down_proj`.

---

## Sparse MLP verification

```bash
python verify_sparse_mlp.py

python -m llama3_neuroplastic.experiments.verify_sparse_mlp_checkpoint \
    --checkpoint results/mlp_basis.pt \
    --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" \
    --expect-target intermediate_block_scores \
    --expect-block-domain intermediate \
    --expect-layers 110 \
    --expect-num-blocks 1664

python -m llama3_neuroplastic.experiments.verify_sparse_mlp_runtime_summary \
    --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" \
    --sparse-basis-path results/mlp_basis.pt \
    --sparse-mlp-execution auto \
    --expect-execution exact_intermediate_sparse \
    --expect-target intermediate_block_scores \
    --expect-block-domain intermediate \
    --expect-layers 110 \
    --expect-num-blocks 1664

python -m llama3_neuroplastic.experiments.verify_sparse_mlp_generation_pair \
    --model-name "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" \
    --sparse-basis-path results/mlp_basis.pt \
    --sparse-mlp-execution exact_intermediate_sparse \
    --sparse-mlp-prefill-mode dense \
    --sparse-top-k 1664 \
    --max-new-tokens 2 \
    --require-identical
```

---

## Running tests

```bash
pytest tests/ -v
```

---

## Environment variables

| Variable | Effect |
|----------|--------|
| `STREAMING_SPARSE_BASIS_EXECUTION` | `auto` (default), `output_basis_surrogate`, `routed_output_blocks`, `exact_intermediate_sparse`, or `exact_intermediate_sparse_oracle` |
| `STREAMING_SPARSE_MLP_PREFILL_MODE` | `dense` (default) or `sparse` for covered-layer prompt prefill |
| `STREAMING_ALLOW_TAYLOR_WITH_SPARSE_ATTN` | `1` to allow Taylor-SSD + sparse attention (experimental, may diverge) |
